"""Isolated SQLite storage and catalog for external engineering knowledge."""

import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from src.knowledge_models import KnowledgeDocument, ReliabilityTier, SourceType
from src.models import DatasetOwnershipError
from src.storage import EvidenceStore


class KnowledgeStoreError(RuntimeError):
    """Raised when an engineering knowledge store operation cannot be safely completed."""


class KnowledgeStore:
    """Manage the persistent library of external engineering manuals, standards, and research."""

    SCHEMA_VERSION = 1
    REQUIRED_TABLES = {"knowledge_documents"}

    def __init__(
        self,
        store: EvidenceStore,
        db_path: Optional[Path] = None,
    ) -> None:
        if not isinstance(store, EvidenceStore):
            raise TypeError("KnowledgeStore requires an EvidenceStore instance.")
        self.store = store
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.db_path = project_root / "data" / "knowledge" / "knowledge_library.db"
        else:
            self.db_path = Path(db_path).resolve()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection: Optional[sqlite3.Connection] = None
        try:
            connection = sqlite3.connect(str(self.db_path), timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            return connection
        except sqlite3.Error:
            if connection is not None:
                connection.close()
            raise KnowledgeStoreError("The knowledge database is unavailable.") from None

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except sqlite3.IntegrityError:
            connection.rollback()
            raise KnowledgeStoreError(
                "The knowledge store rejected a duplicate or inconsistent document record."
            ) from None
        except sqlite3.Error:
            connection.rollback()
            raise KnowledgeStoreError(
                "The knowledge store could not complete the database operation."
            ) from None
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                tables = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                if version > self.SCHEMA_VERSION:
                    raise KnowledgeStoreError(
                        f"Knowledge database schema version {version} is newer than supported."
                    )
                if version == self.SCHEMA_VERSION:
                    if not self.REQUIRED_TABLES.issubset(tables):
                        raise KnowledgeStoreError(
                            "Knowledge database schema is marked current but is incomplete."
                        )
                    return
                if tables:
                    raise KnowledgeStoreError(
                        "Knowledge database contains unversioned tables; refusing initialization."
                    )

                connection.execute("BEGIN IMMEDIATE")
                self._create_schema(connection)
                connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
                connection.commit()
        except KnowledgeStoreError:
            raise
        except OSError:
            raise KnowledgeStoreError(
                "The knowledge database directory is unavailable."
            ) from None
        except sqlite3.Error:
            raise KnowledgeStoreError(
                "The knowledge database schema could not be initialized."
            ) from None

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE knowledge_documents (
                document_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                title TEXT NOT NULL,
                publisher_author TEXT NOT NULL,
                publication_year INTEGER,
                edition_revision TEXT,
                source_type TEXT NOT NULL CHECK (
                    source_type IN (
                        'OEM_MANUAL',
                        'ENGINEERING_STANDARD',
                        'PEER_REVIEWED_RESEARCH',
                        'INDUSTRY_BULLETIN',
                        'GENERAL_REFERENCE'
                    )
                ),
                reliability_tier TEXT NOT NULL CHECK (
                    reliability_tier IN (
                        'TIER_1_STANDARD',
                        'TIER_2_OEM',
                        'TIER_3_CURATED_RESEARCH',
                        'TIER_4_GENERAL'
                    )
                ),
                equipment_category TEXT,
                equipment_subtype TEXT,
                manufacturer TEXT,
                sha256_hash TEXT,
                copyright_license TEXT,
                rights_attribution TEXT,
                is_proprietary_restricted INTEGER NOT NULL DEFAULT 0 CHECK (
                    is_proprietary_restricted IN (0, 1)
                ),
                ingested_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_knowledge_docs_scope
            ON knowledge_documents(dataset_id, source_type, reliability_tier)
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_knowledge_docs_equipment
            ON knowledge_documents(equipment_category, equipment_subtype, manufacturer)
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX uq_knowledge_sha256_scope
            ON knowledge_documents(coalesce(dataset_id, '__GLOBAL__'), sha256_hash)
            WHERE sha256_hash IS NOT NULL
            """
        )

    def _require_dataset_scope(self, dataset_id: Optional[str]) -> Optional[str]:
        """Validate dataset if provided; None indicates global technical knowledge."""
        if dataset_id is None or dataset_id == "":
            return None
        cleaned = str(dataset_id).strip()
        if not cleaned:
            return None
        self.store.require_dataset_id(cleaned, "Knowledge store", cleaned)
        return cleaned

    @staticmethod
    def _required_text(value: Any, field_name: str, max_length: int = 300) -> str:
        if not isinstance(value, str):
            raise KnowledgeStoreError(f"{field_name} is required.")
        cleaned = value.strip()
        if not cleaned or len(cleaned) > max_length or "\x00" in cleaned:
            raise KnowledgeStoreError(
                f"{field_name} must contain 1 to {max_length} valid characters."
            )
        return cleaned

    @staticmethod
    def _optional_text(value: Any, field_name: str, max_length: int = 300) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise KnowledgeStoreError(f"{field_name} must be text when provided.")
        cleaned = value.strip()
        if not cleaned:
            return None
        if len(cleaned) > max_length or "\x00" in cleaned:
            raise KnowledgeStoreError(
                f"{field_name} must be at most {max_length} valid characters."
            )
        return cleaned

    @staticmethod
    def _validate_year(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            year = int(value)
        except (ValueError, TypeError):
            raise KnowledgeStoreError("Publication year must be a valid integer.") from None
        current_year = datetime.now(timezone.utc).year + 1
        if year < 1800 or year > current_year:
            raise KnowledgeStoreError(f"Publication year must be between 1800 and {current_year}.")
        return year

    @staticmethod
    def _validate_sha256(value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        if not isinstance(value, str):
            raise KnowledgeStoreError("SHA-256 hash must be a 64-character hexadecimal string.")
        cleaned = value.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", cleaned):
            raise KnowledgeStoreError("SHA-256 hash must be a 64-character hexadecimal string.")
        return cleaned

    @classmethod
    def _validate_document_inputs(
        cls,
        fields: Mapping[str, Any],
    ) -> Dict[str, Any]:
        title = cls._required_text(fields.get("title"), "Document title", 300)
        publisher_author = cls._required_text(
            fields.get("publisher_author"), "Publisher/Author", 200
        )

        source_type_val = fields.get("source_type")
        if isinstance(source_type_val, SourceType):
            source_type = source_type_val.value
        elif isinstance(source_type_val, str) and source_type_val in SourceType.__members__:
            source_type = SourceType[source_type_val].value
        else:
            raise KnowledgeStoreError(
                f"Invalid source_type. Supported values: {list(SourceType.__members__)}"
            )

        reliability_val = fields.get("reliability_tier")
        if isinstance(reliability_val, ReliabilityTier):
            reliability_tier = reliability_val.value
        elif isinstance(reliability_val, str) and reliability_val in ReliabilityTier.__members__:
            reliability_tier = ReliabilityTier[reliability_val].value
        else:
            raise KnowledgeStoreError(
                f"Invalid reliability_tier. Supported values: {list(ReliabilityTier.__members__)}"
            )

        return {
            "title": title,
            "publisher_author": publisher_author,
            "source_type": source_type,
            "reliability_tier": reliability_tier,
            "publication_year": cls._validate_year(fields.get("publication_year")),
            "edition_revision": cls._optional_text(
                fields.get("edition_revision"), "Edition/Revision", 100
            ),
            "equipment_category": cls._optional_text(
                fields.get("equipment_category"), "Equipment category", 80
            ),
            "equipment_subtype": cls._optional_text(
                fields.get("equipment_subtype"), "Equipment subtype", 80
            ),
            "manufacturer": cls._optional_text(
                fields.get("manufacturer"), "Manufacturer", 120
            ),
            "sha256_hash": cls._validate_sha256(fields.get("sha256_hash")),
            "copyright_license": cls._optional_text(
                fields.get("copyright_license"), "Copyright/License", 300
            ),
            "rights_attribution": cls._optional_text(
                fields.get("rights_attribution"), "Rights attribution", 300
            ),
            "is_proprietary_restricted": 1 if fields.get("is_proprietary_restricted") else 0,
        }

    def add_document(
        self,
        title: str,
        publisher_author: str,
        source_type: SourceType,
        reliability_tier: ReliabilityTier,
        dataset_id: Optional[str] = None,
        publication_year: Optional[int] = None,
        edition_revision: Optional[str] = None,
        equipment_category: Optional[str] = None,
        equipment_subtype: Optional[str] = None,
        manufacturer: Optional[str] = None,
        sha256_hash: Optional[str] = None,
        copyright_license: Optional[str] = None,
        rights_attribution: Optional[str] = None,
        is_proprietary_restricted: bool = False,
        document_id: Optional[str] = None,
    ) -> KnowledgeDocument:
        """Add an engineering technical document to the knowledge store."""
        valid_dataset_id = self._require_dataset_scope(dataset_id)
        validated = self._validate_document_inputs(
            {
                "title": title,
                "publisher_author": publisher_author,
                "source_type": source_type,
                "reliability_tier": reliability_tier,
                "publication_year": publication_year,
                "edition_revision": edition_revision,
                "equipment_category": equipment_category,
                "equipment_subtype": equipment_subtype,
                "manufacturer": manufacturer,
                "sha256_hash": sha256_hash,
                "copyright_license": copyright_license,
                "rights_attribution": rights_attribution,
                "is_proprietary_restricted": is_proprietary_restricted,
            }
        )

        doc_id = (
            self._required_text(document_id, "Document ID", 64)
            if document_id
            else f"KNOW-{uuid.uuid4().hex[:12].upper()}"
        )
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO knowledge_documents (
                    document_id, dataset_id, title, publisher_author, publication_year,
                    edition_revision, source_type, reliability_tier, equipment_category,
                    equipment_subtype, manufacturer, sha256_hash, copyright_license,
                    rights_attribution, is_proprietary_restricted, ingested_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    valid_dataset_id,
                    validated["title"],
                    validated["publisher_author"],
                    validated["publication_year"],
                    validated["edition_revision"],
                    validated["source_type"],
                    validated["reliability_tier"],
                    validated["equipment_category"],
                    validated["equipment_subtype"],
                    validated["manufacturer"],
                    validated["sha256_hash"],
                    validated["copyright_license"],
                    validated["rights_attribution"],
                    validated["is_proprietary_restricted"],
                    now,
                    now,
                    now,
                ),
            )

        doc = self.get_document(doc_id)
        if doc is None:
            raise KnowledgeStoreError("Failed to retrieve newly inserted knowledge document.")
        return doc

    def get_document(self, document_id: str) -> Optional[KnowledgeDocument]:
        """Fetch a single knowledge document by ID (read-only)."""
        doc_id = self._required_text(document_id, "Document ID", 64)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_documents WHERE document_id = ?",
                (doc_id,),
            ).fetchone()
        return self._row_to_document(row) if row is not None else None

    def list_documents(
        self,
        dataset_id: Optional[str] = None,
        include_global: bool = True,
        source_type: Optional[SourceType] = None,
        reliability_tier: Optional[ReliabilityTier] = None,
        equipment_category: Optional[str] = None,
    ) -> List[KnowledgeDocument]:
        """Query knowledge documents matching scope and filters (read-only).

        If dataset_id is None, returns only global documents.
        If dataset_id is given, returns documents for that dataset and, if
        include_global is True, global documents as well. Never returns documents
        belonging to other datasets.
        """
        valid_dataset_id = self._require_dataset_scope(dataset_id)
        conditions: List[str] = []
        params: List[Any] = []

        if valid_dataset_id is None:
            conditions.append("dataset_id IS NULL")
        else:
            if include_global:
                conditions.append("(dataset_id = ? OR dataset_id IS NULL)")
                params.append(valid_dataset_id)
            else:
                conditions.append("dataset_id = ?")
                params.append(valid_dataset_id)

        if source_type is not None:
            st_val = source_type.value if isinstance(source_type, SourceType) else source_type
            conditions.append("source_type = ?")
            params.append(st_val)

        if reliability_tier is not None:
            rt_val = (
                reliability_tier.value
                if isinstance(reliability_tier, ReliabilityTier)
                else reliability_tier
            )
            conditions.append("reliability_tier = ?")
            params.append(rt_val)

        if equipment_category is not None:
            conditions.append("equipment_category = ?")
            params.append(equipment_category)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"""
            SELECT * FROM knowledge_documents
            {where_clause}
            ORDER BY lower(title), document_id
        """

        with closing(self._connect()) as connection:
            rows = connection.execute(query, params).fetchall()

        return [self._row_to_document(row) for row in rows]

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> KnowledgeDocument:
        return KnowledgeDocument(
            document_id=row["document_id"],
            dataset_id=row["dataset_id"],
            title=row["title"],
            publisher_author=row["publisher_author"],
            publication_year=row["publication_year"],
            edition_revision=row["edition_revision"],
            source_type=SourceType(row["source_type"]),
            reliability_tier=ReliabilityTier(row["reliability_tier"]),
            equipment_category=row["equipment_category"],
            equipment_subtype=row["equipment_subtype"],
            manufacturer=row["manufacturer"],
            sha256_hash=row["sha256_hash"],
            copyright_license=row["copyright_license"],
            rights_attribution=row["rights_attribution"],
            is_proprietary_restricted=bool(row["is_proprietary_restricted"]),
            ingested_at=row["ingested_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def integrity_check(self) -> str:
        """Return the SQLite integrity check result."""
        with closing(self._connect()) as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        return str(row[0])

    def foreign_key_check(self) -> List[Dict[str, Any]]:
        """Return SQLite foreign-key diagnostics."""
        with closing(self._connect()) as connection:
            rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        return [dict(row) for row in rows]

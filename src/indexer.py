"""Persistent local full-text search index for GhostFault evidence chunks.

Backed by SQLite FTS5 for deterministic, explainable keyword retrieval with BM25 ranking.
Provides atomic index updates, document change detection, and full rebuild capabilities.
"""

import sqlite3
import json
import hashlib
import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import asdict
from datetime import datetime, timezone

from src.models import (
    EvidenceChunk,
    DocumentMetadata,
    DocumentStatus,
    DatasetAssociationIssue,
    DatasetOwnershipError,
    DEFAULT_DATASET_ID,
    current_iso_timestamp,
)
from src.chunker import TextChunker, compute_content_hash
from src.storage import EvidenceStore


class SearchIndex:
    """
    Manages a persistent SQLite FTS5 index stored in `data/evidence/index/evidence_index.db`.
    """

    LATEST_SCHEMA_VERSION = 3
    APPROVED_OWNERSHIP_REPAIR = {
        "DOC-5A82BA6A3A4C",
        "DOC-C6C0555F1F0D",
        "DOC-9406F467ECCE",
        "DOC-71B10A3BA0C8",
        "DOC-381BC31C33A9",
        "DOC-16FE64012FD1",
    }
    FAILED_OWNERSHIP_REPAIR_DOCUMENT = "DOC-1C401DB99DE6"

    def __init__(self, db_path: Optional[Path] = None, chunker: Optional[TextChunker] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.db_path = project_root / "data" / "evidence" / "index" / "evidence_index.db"
        else:
            self.db_path = Path(db_path).resolve()

        self.chunker = chunker if chunker is not None else TextChunker()
        self._ensure_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Establish SQLite connection with foreign keys and row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_db(self) -> None:
        """Initialize the index and apply backed-up, versioned schema migrations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        database_exists = self.db_path.exists() and self.db_path.stat().st_size > 0
        conn = self._get_connection()
        try:
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            tables_before = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            version_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
            ).fetchone()
            migration_version = 0
            if version_row:
                migration_row = conn.execute(
                    "SELECT MAX(version) FROM schema_migrations"
                ).fetchone()
                migration_version = int(migration_row[0] or 0)
            if migration_version and user_version and migration_version != user_version:
                raise RuntimeError(
                    "Evidence index schema version metadata disagrees: "
                    f"schema_migrations={migration_version}, user_version={user_version}."
                )
            current_version = migration_version or user_version
            if current_version > self.LATEST_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Evidence index schema version {current_version} is newer than "
                    f"supported version {self.LATEST_SCHEMA_VERSION}."
                )
            if (
                current_version == self.LATEST_SCHEMA_VERSION
                and migration_version == self.LATEST_SCHEMA_VERSION
                and user_version == self.LATEST_SCHEMA_VERSION
            ):
                migration_versions = [
                    row[0]
                    for row in conn.execute(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    ).fetchall()
                ]
                if migration_versions != [1, 2, 3]:
                    raise RuntimeError(
                        "Evidence index migration history is incomplete or inconsistent."
                    )
                if not self._latest_schema_is_valid(conn):
                    raise RuntimeError(
                        "Evidence index schema version is current but its structure is invalid."
                    )
                if conn.execute("PRAGMA foreign_key_check").fetchall():
                    raise RuntimeError(
                        "Evidence index schema has foreign-key integrity issues."
                    )
                return
            if database_exists and tables_before and current_version < self.LATEST_SCHEMA_VERSION:
                self._backup_database_before_migration(conn, current_version + 1)
            conn.execute("PRAGMA foreign_keys = OFF")

            conn.execute("BEGIN IMMEDIATE")
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    )
                    """
                )
                if version_row and not user_version:
                    conn.execute(
                        "PRAGMA user_version = " + str(migration_version)
                    )
                if current_version == 0:
                    self._create_base_schema(conn)
                    self._record_schema_version(conn, 1)
                    current_version = 1
                elif not migration_version:
                    for adopted_version in range(1, current_version + 1):
                        self._record_schema_version(conn, adopted_version)

                if current_version < 2:
                    self._ensure_dataset_columns(conn)
                    self._record_schema_version(conn, 2)
                    current_version = 2
                elif current_version == 2:
                    self._ensure_dataset_columns(conn)

                if current_version < 3:
                    self._remove_dataset_column_defaults(conn)
                    self._record_schema_version(conn, 3)
                    current_version = 3

                conn.execute(f"PRAGMA user_version = {current_version}")
                foreign_key_issues = conn.execute("PRAGMA foreign_key_check").fetchall()
                if foreign_key_issues:
                    raise RuntimeError(
                        f"Evidence index migration found {len(foreign_key_issues)} "
                        "foreign-key integrity issue(s)."
                    )
            conn.execute("PRAGMA foreign_keys = ON")
        finally:
            conn.close()

    @staticmethod
    def _create_base_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_documents (
                document_id TEXT PRIMARY KEY,
                source_filename TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                page_number INTEGER,
                start_offset INTEGER,
                end_offset INTEGER,
                created_at TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES indexed_documents(document_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                source_filename,
                text,
                tokenize = 'porter unicode61'
            )
            """
        )

    @staticmethod
    def _record_schema_version(conn: sqlite3.Connection, version: int) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, current_iso_timestamp()),
        )

    @staticmethod
    def _ensure_dataset_columns(conn: sqlite3.Connection) -> None:
        for table_name in ("indexed_documents", "evidence_chunks"):
            columns = {
                row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})")
            }
            if "dataset_id" not in columns:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN dataset_id TEXT")

    @staticmethod
    def _latest_schema_is_valid(conn: sqlite3.Connection) -> bool:
        for table_name in ("indexed_documents", "evidence_chunks"):
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            dataset_column = next(
                (row for row in columns if row["name"] == "dataset_id"),
                None,
            )
            if dataset_column is None or dataset_column["dflt_value"] is not None:
                return False
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
        ).fetchone() is not None

    @staticmethod
    def _remove_dataset_column_defaults(conn: sqlite3.Connection) -> None:
        dataset_defaults = {}
        for table_name in ("indexed_documents", "evidence_chunks"):
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            dataset_column = next(row for row in columns if row["name"] == "dataset_id")
            dataset_defaults[table_name] = dataset_column["dflt_value"]
        if not any(dataset_defaults.values()):
            return

        conn.execute(
            """
            CREATE TABLE indexed_documents_v3 (
                document_id TEXT PRIMARY KEY,
                source_filename TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL,
                dataset_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE evidence_chunks_v3 (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                source_filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                page_number INTEGER,
                start_offset INTEGER,
                end_offset INTEGER,
                created_at TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                dataset_id TEXT,
                FOREIGN KEY (document_id)
                    REFERENCES indexed_documents(document_id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            INSERT INTO indexed_documents_v3 (
                document_id, source_filename, content_hash, chunk_count,
                indexed_at, dataset_id
            )
            SELECT document_id, source_filename, content_hash, chunk_count,
                   indexed_at, dataset_id
            FROM indexed_documents
            """
        )
        conn.execute(
            """
            INSERT INTO evidence_chunks_v3 (
                chunk_id, document_id, chunk_index, text, source_filename,
                file_type, page_number, start_offset, end_offset, created_at,
                content_hash, dataset_id
            )
            SELECT chunk_id, document_id, chunk_index, text, source_filename,
                   file_type, page_number, start_offset, end_offset, created_at,
                   content_hash, dataset_id
            FROM evidence_chunks
            """
        )
        conn.execute("DROP TABLE evidence_chunks")
        conn.execute("DROP TABLE indexed_documents")
        conn.execute(
            "ALTER TABLE indexed_documents_v3 RENAME TO indexed_documents"
        )
        conn.execute("ALTER TABLE evidence_chunks_v3 RENAME TO evidence_chunks")

    def _backup_database_before_migration(
        self,
        connection: sqlite3.Connection,
        target_version: int,
    ) -> Path:
        """Create and verify a SQLite-consistent backup before a schema migration."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = self.db_path.with_name(
            f"{self.db_path.name}.schema-v{target_version}-{timestamp}.bak"
        )
        backup_connection = sqlite3.connect(str(backup_path))
        try:
            connection.backup(backup_connection)
            result = backup_connection.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                raise RuntimeError(
                    f"SQLite schema backup failed integrity verification: {backup_path}"
                )
        finally:
            backup_connection.close()
        return backup_path

    @staticmethod
    def _repair_journal_path(store: EvidenceStore, repair_id: str) -> Path:
        try:
            normalized_id = str(uuid.UUID(repair_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise DatasetOwnershipError("Ownership repair journal ID is invalid.") from exc
        return store.metadata_dir / "ownership-repair" / f"{normalized_id}.json"

    @staticmethod
    def _write_repair_journal(path: Path, journal: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        payload = json.dumps(journal, indent=2, sort_keys=True, ensure_ascii=False)
        with open(temporary_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)

    @staticmethod
    def _read_repair_journal(path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                journal = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetOwnershipError(
                f"Ownership repair journal cannot be read safely: {path}"
            ) from exc
        if not isinstance(journal, dict) or journal.get("version") != 1:
            raise DatasetOwnershipError("Ownership repair journal has an unsupported format.")
        return journal

    def preview_document_ownership_repair(
        self,
        store: EvidenceStore,
        assignments: Dict[str, str],
        required_status: Optional[DocumentStatus] = DocumentStatus.PROCESSED,
    ) -> Dict[str, Any]:
        """Preview all catalog, SQLite and FTS ownership layers without writing."""
        errors: List[str] = []
        if required_status is None:
            required_status = DocumentStatus.PROCESSED
        if set(assignments) != self.APPROVED_OWNERSHIP_REPAIR:
            rejected = sorted(set(assignments) - self.APPROVED_OWNERSHIP_REPAIR)
            missing = sorted(self.APPROVED_OWNERSHIP_REPAIR - set(assignments))
            if rejected:
                errors.append(f"Unapproved document IDs: {', '.join(rejected)}.")
            if missing:
                errors.append(f"Approved document IDs omitted: {', '.join(missing)}.")
        if required_status != DocumentStatus.PROCESSED:
            errors.append("Ownership repair only accepts processed documents.")

        for document_id, target_dataset_id in assignments.items():
            if target_dataset_id != DEFAULT_DATASET_ID:
                errors.append(
                    f"Document '{document_id}' repair target must be "
                    f"'{DEFAULT_DATASET_ID}'."
                )
                continue
            try:
                store.require_dataset_id(target_dataset_id, "Repair target", document_id)
            except DatasetOwnershipError as exc:
                errors.append(str(exc))

        catalog = store.read_catalog()
        catalog_by_id: Dict[str, List[DocumentMetadata]] = {}
        for metadata in catalog:
            catalog_by_id.setdefault(metadata.document_id, []).append(metadata)
        duplicate_catalog_ids = sorted(
            document_id for document_id, records in catalog_by_id.items() if len(records) != 1
        )
        if duplicate_catalog_ids:
            errors.append(
                "Duplicate catalog document IDs: " + ", ".join(duplicate_catalog_ids) + "."
            )

        approved_ids = sorted(self.APPROVED_OWNERSHIP_REPAIR)
        with self._get_connection() as conn:
            indexed_rows = conn.execute(
                "SELECT document_id, dataset_id, source_filename, content_hash, "
                "chunk_count FROM indexed_documents"
            ).fetchall()
            indexed_by_id: Dict[str, List[sqlite3.Row]] = {}
            for row in indexed_rows:
                indexed_by_id.setdefault(row["document_id"], []).append(row)

            chunk_rows = conn.execute(
                "SELECT chunk_id, document_id, dataset_id, source_filename, file_type, "
                "text, content_hash FROM evidence_chunks"
            ).fetchall()
            chunks_by_document: Dict[str, List[sqlite3.Row]] = {}
            chunks_by_id: Dict[str, List[sqlite3.Row]] = {}
            for row in chunk_rows:
                chunks_by_document.setdefault(row["document_id"], []).append(row)
                chunks_by_id.setdefault(row["chunk_id"], []).append(row)
            fts_rows = conn.execute(
                "SELECT chunk_id, document_id, source_filename, text FROM chunks_fts"
            ).fetchall()
            fts_by_chunk: Dict[str, List[sqlite3.Row]] = {}
            for row in fts_rows:
                fts_by_chunk.setdefault(row["chunk_id"], []).append(row)

            items: List[Dict[str, Any]] = []
            for document_id in approved_ids:
                records = catalog_by_id.get(document_id, [])
                metadata = records[0] if len(records) == 1 else None
                indexed = indexed_by_id.get(document_id, [])
                chunks = chunks_by_document.get(document_id, [])
                fts_relevant_ids = {row["chunk_id"] for row in chunks}
                fts_relevant_ids.update(
                    row["chunk_id"]
                    for row in fts_rows
                    if row["document_id"] == document_id
                )
                fts_findings: List[str] = []
                for chunk_id in sorted(fts_relevant_ids):
                    relational_matches = chunks_by_id.get(chunk_id, [])
                    fts_matches = fts_by_chunk.get(chunk_id, [])
                    if len(relational_matches) != 1:
                        fts_findings.append(f"{chunk_id}: relational row count {len(relational_matches)}")
                    if len(fts_matches) != 1:
                        fts_findings.append(f"{chunk_id}: FTS row count {len(fts_matches)}")
                    if len(relational_matches) == 1 and len(fts_matches) == 1:
                        relation = relational_matches[0]
                        fts = fts_matches[0]
                        if (
                            relation["document_id"] != fts["document_id"]
                            or relation["source_filename"] != fts["source_filename"]
                            or relation["text"] != fts["text"]
                        ):
                            fts_findings.append(f"{chunk_id}: FTS content/provenance mismatch")

                catalog_owner = metadata.dataset_id if metadata else None
                index_owners = [row["dataset_id"] for row in indexed]
                chunk_owners = [row["dataset_id"] for row in chunks]
                record_errors: List[str] = []
                if metadata is None:
                    record_errors.append(
                        "catalog document missing or duplicated"
                    )
                elif metadata.ingestion_status != DocumentStatus.PROCESSED:
                    record_errors.append(
                        f"catalog status is {metadata.ingestion_status.value}, not processed"
                    )
                if len(indexed) != 1:
                    record_errors.append(f"indexed-document row count is {len(indexed)}")
                if not chunks:
                    record_errors.append("no relational chunks are present")
                if any(owner not in (None, "", DEFAULT_DATASET_ID) for owner in [catalog_owner, *index_owners, *chunk_owners]):
                    record_errors.append("unexpected non-default ownership value is present")
                if metadata and indexed and indexed[0]["source_filename"] != metadata.original_filename:
                    record_errors.append("indexed filename differs from catalog provenance")
                if any(
                    row["document_id"] not in indexed_by_id
                    for row in chunks
                ):
                    record_errors.append("orphan relational chunks are present")
                if fts_findings:
                    record_errors.append("FTS/relational consistency failed")
                if record_errors:
                    errors.append(f"Document '{document_id}': {'; '.join(record_errors)}.")

                items.append({
                    "document_id": document_id,
                    "catalog_owner": catalog_owner,
                    "indexed_document_owners": index_owners,
                    "chunk_owners": [
                        {"chunk_id": row["chunk_id"], "dataset_id": row["dataset_id"]}
                        for row in chunks
                    ],
                    "fts": {
                        "consistent": not fts_findings,
                        "findings": fts_findings,
                    },
                    "proposed_owner": assignments.get(document_id),
                    "catalog_status": metadata.ingestion_status.value if metadata else None,
                    "record_errors": record_errors,
                })

            failed_records = catalog_by_id.get(self.FAILED_OWNERSHIP_REPAIR_DOCUMENT, [])
            failed_indexed = indexed_by_id.get(self.FAILED_OWNERSHIP_REPAIR_DOCUMENT, [])
            failed_chunks = chunks_by_document.get(self.FAILED_OWNERSHIP_REPAIR_DOCUMENT, [])
            failed_fts = [
                row for row in fts_rows
                if row["document_id"] == self.FAILED_OWNERSHIP_REPAIR_DOCUMENT
            ]
            failed_ok = (
                len(failed_records) == 1
                and failed_records[0].ingestion_status == DocumentStatus.FAILED
                and failed_records[0].dataset_id is None
                and not failed_indexed
                and not failed_chunks
                and not failed_fts
            )
            if not failed_ok:
                errors.append(
                    f"Failed document '{self.FAILED_OWNERSHIP_REPAIR_DOCUMENT}' must remain "
                    "failed, unassigned, and absent from index/chunks/FTS."
                )

            registry = store.list_datasets()
            registry_ids = {dataset.dataset_id for dataset in registry}
            if len(registry_ids) != len(registry):
                errors.append("Dataset registry contains duplicate IDs.")
            if DEFAULT_DATASET_ID not in registry_ids:
                errors.append(f"Target dataset '{DEFAULT_DATASET_ID}' is not registered.")

            snapshot = {
                "catalog": [metadata.to_dict() for metadata in catalog],
                "registry": [dataset.to_dict() for dataset in registry],
                "indexed": [
                    {key: row[key] for key in row.keys()}
                    for row in indexed_rows
                ],
                "chunks": [
                    {key: row[key] for key in row.keys()}
                    for row in chunk_rows
                ],
                "fts": [
                    {key: row[key] for key in row.keys()}
                    for row in fts_rows
                ],
            }
        source_fingerprint = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        changes_required = sum(
            1
            for item in items
            if item["catalog_owner"] != DEFAULT_DATASET_ID
            or any(owner != DEFAULT_DATASET_ID for owner in item["indexed_document_owners"])
            or any(chunk["dataset_id"] != DEFAULT_DATASET_ID for chunk in item["chunk_owners"])
        )
        return {
            "ready": not errors,
            "changes_required": changes_required,
            "source_fingerprint": source_fingerprint,
            "items": items,
            "errors": errors,
            "failed_document_preserved": failed_ok,
        }

    def _repair_assignments_from_plan(self, plan: Dict[str, Any]) -> Dict[str, str]:
        assignments = {
            item.get("document_id"): item.get("proposed_owner")
            for item in plan.get("items", [])
            if isinstance(item, dict)
        }
        if set(assignments) != self.APPROVED_OWNERSHIP_REPAIR:
            raise DatasetOwnershipError("Ownership repair plan has an invalid document scope.")
        if any(owner != DEFAULT_DATASET_ID for owner in assignments.values()):
            raise DatasetOwnershipError("Ownership repair plan contains an invalid target dataset.")
        return assignments

    def apply_document_ownership_repair(
        self,
        store: EvidenceStore,
        plan: Dict[str, Any],
        required_status: Optional[DocumentStatus] = DocumentStatus.PROCESSED,
    ) -> Dict[str, Any]:
        """Apply a journaled cross-layer repair and verify it before reporting success."""
        if not isinstance(plan, dict) or not plan.get("ready"):
            raise DatasetOwnershipError("Ownership repair plan is invalid or not ready.")
        assignments = self._repair_assignments_from_plan(plan)
        fresh_plan = self.preview_document_ownership_repair(
            store, assignments, required_status=required_status
        )
        if not fresh_plan["ready"]:
            raise DatasetOwnershipError(
                "Ownership repair preconditions failed: " + " ".join(fresh_plan["errors"])
            )
        if fresh_plan["changes_required"] == 0:
            return {
                "changed_document_ids": [],
                "already_applied_document_ids": sorted(assignments),
                "changes_applied": 0,
                "verified": True,
            }
        if fresh_plan["source_fingerprint"] != plan.get("source_fingerprint"):
            raise DatasetOwnershipError(
                "Ownership repair preview is stale; create a fresh preview before applying."
            )

        repair_id = str(uuid.uuid4())
        journal_path = self._repair_journal_path(store, repair_id)
        catalog = store.read_catalog()
        before_catalog = {
            metadata.document_id: metadata.to_dict()
            for metadata in catalog
            if metadata.document_id in assignments
        }
        with self._get_connection() as conn:
            indexed_before = [
                dict(row)
                for row in conn.execute(
                    "SELECT document_id, dataset_id, content_hash, chunk_count "
                    "FROM indexed_documents WHERE document_id IN (?, ?, ?, ?, ?, ?)",
                    tuple(sorted(assignments)),
                ).fetchall()
            ]
            chunks_before = [
                dict(row)
                for row in conn.execute(
                    "SELECT chunk_id, document_id, dataset_id, content_hash "
                    "FROM evidence_chunks WHERE document_id IN (?, ?, ?, ?, ?, ?) "
                    "ORDER BY chunk_id",
                    tuple(sorted(assignments)),
                ).fetchall()
            ]
        journal = {
            "version": 1,
            "repair_id": repair_id,
            "status": "prepared",
            "assignments": assignments,
            "before_catalog": before_catalog,
            "before_indexed_documents": indexed_before,
            "before_chunks": chunks_before,
            "source_fingerprint": plan["source_fingerprint"],
        }
        self._write_repair_journal(journal_path, journal)
        return self._continue_ownership_repair(store, journal_path, journal)

    def _continue_ownership_repair(
        self,
        store: EvidenceStore,
        journal_path: Path,
        journal: Dict[str, Any],
    ) -> Dict[str, Any]:
        assignments = self._validate_repair_journal_scope(journal)
        if journal.get("status") in {"prepared", "catalog_applied"}:
            self._apply_repair_catalog(store, journal, assignments)
            journal["status"] = "catalog_applied"
            self._write_repair_journal(journal_path, journal)
        if journal.get("status") in {"catalog_applied", "index_applied"}:
            self._apply_repair_index(journal, assignments)
            journal["status"] = "index_applied"
            self._write_repair_journal(journal_path, journal)

        verification = self.preview_document_ownership_repair(store, assignments)
        if not verification["ready"] or verification["changes_required"] != 0:
            raise DatasetOwnershipError(
                "Ownership repair did not pass post-apply verification: "
                + " ".join(verification["errors"])
            )
        journal["status"] = "verified"
        self._write_repair_journal(journal_path, journal)
        return {
            "repair_id": journal["repair_id"],
            "changed_document_ids": sorted(assignments),
            "already_applied_document_ids": [],
            "changes_applied": len(assignments),
            "verified": True,
        }

    def _validate_repair_journal_scope(self, journal: Dict[str, Any]) -> Dict[str, str]:
        assignments = journal.get("assignments")
        if not isinstance(assignments, dict) or set(assignments) != self.APPROVED_OWNERSHIP_REPAIR:
            raise DatasetOwnershipError("Ownership repair journal has an invalid document scope.")
        if any(owner != DEFAULT_DATASET_ID for owner in assignments.values()):
            raise DatasetOwnershipError("Ownership repair journal has an invalid target.")
        return assignments

    def _apply_repair_catalog(
        self,
        store: EvidenceStore,
        journal: Dict[str, Any],
        assignments: Dict[str, str],
    ) -> None:
        documents = store.read_catalog()
        by_id: Dict[str, List[DocumentMetadata]] = {}
        for metadata in documents:
            by_id.setdefault(metadata.document_id, []).append(metadata)
        for document_id, target in assignments.items():
            records = by_id.get(document_id, [])
            before = journal["before_catalog"].get(document_id)
            if len(records) != 1 or before is None:
                raise DatasetOwnershipError(
                    f"Catalog record '{document_id}' is missing or duplicated during repair."
                )
            current = records[0]
            expected_before = DocumentMetadata.from_dict(before)
            current_without_owner = current.to_dict()
            current_without_owner["dataset_id"] = expected_before.dataset_id
            if current_without_owner != expected_before.to_dict():
                raise DatasetOwnershipError(
                    f"Catalog metadata for '{document_id}' changed during repair."
                )
            if current.dataset_id not in (expected_before.dataset_id, target):
                raise DatasetOwnershipError(
                    f"Catalog ownership for '{document_id}' changed unexpectedly."
                )
            if current.ingestion_status != DocumentStatus.PROCESSED:
                raise DatasetOwnershipError(
                    f"Document '{document_id}' is no longer processed."
                )
            current.dataset_id = target
        store._write_catalog(documents)

    def _apply_repair_index(
        self,
        journal: Dict[str, Any],
        assignments: Dict[str, str],
    ) -> None:
        expected_indexed = {
            row["document_id"]: row
            for row in journal["before_indexed_documents"]
        }
        expected_chunks = {
            row["chunk_id"]: row
            for row in journal["before_chunks"]
        }
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for document_id, target in assignments.items():
                current = conn.execute(
                    "SELECT document_id, dataset_id, content_hash, chunk_count "
                    "FROM indexed_documents WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
                before = expected_indexed.get(document_id)
                if len(current) != 1 or before is None:
                    raise DatasetOwnershipError(
                        f"Indexed document '{document_id}' is missing or duplicated during repair."
                    )
                row = dict(current[0])
                if row["dataset_id"] not in (before["dataset_id"], target):
                    raise DatasetOwnershipError(
                        f"Indexed ownership for '{document_id}' changed unexpectedly."
                    )
                if any(row[key] != before[key] for key in ("content_hash", "chunk_count")):
                    raise DatasetOwnershipError(
                        f"Indexed document '{document_id}' content changed during repair."
                    )
                conn.execute(
                    "UPDATE indexed_documents SET dataset_id = ? WHERE document_id = ?",
                    (target, document_id),
                )

            current_chunks = {
                row["chunk_id"]: dict(row)
                for row in conn.execute(
                    "SELECT chunk_id, document_id, dataset_id, content_hash "
                    "FROM evidence_chunks WHERE document_id IN (?, ?, ?, ?, ?, ?)",
                    tuple(sorted(assignments)),
                ).fetchall()
            }
            if set(current_chunks) != set(expected_chunks):
                raise DatasetOwnershipError(
                    "Relational chunk set changed during ownership repair."
                )
            for chunk_id, before in expected_chunks.items():
                row = current_chunks[chunk_id]
                if (
                    row["document_id"] != before["document_id"]
                    or row["content_hash"] != before["content_hash"]
                    or row["dataset_id"] not in (before["dataset_id"], DEFAULT_DATASET_ID)
                ):
                    raise DatasetOwnershipError(
                        f"Relational chunk '{chunk_id}' changed unexpectedly during repair."
                    )
                conn.execute(
                    "UPDATE evidence_chunks SET dataset_id = ? WHERE chunk_id = ?",
                    (DEFAULT_DATASET_ID, chunk_id),
                )

    def recover_document_ownership_repair(
        self,
        store: EvidenceStore,
        repair_id: str,
        action: str = "resume",
    ) -> Dict[str, Any]:
        """Resume or roll back an interrupted repair from its durable journal."""
        if action not in {"resume", "rollback"}:
            raise ValueError("Repair recovery action must be 'resume' or 'rollback'.")
        journal_path = self._repair_journal_path(store, repair_id)
        journal = self._read_repair_journal(journal_path)
        if journal.get("repair_id") != str(uuid.UUID(repair_id)):
            raise DatasetOwnershipError("Ownership repair journal ID does not match its filename.")
        if journal.get("status") == "verified" and action == "resume":
            return {"repair_id": repair_id, "status": "verified", "verified": True}
        if journal.get("status") == "rolled_back":
            if action == "rollback":
                return {"repair_id": repair_id, "status": "rolled_back", "verified": True}
            raise DatasetOwnershipError(
                "A rolled-back repair cannot be resumed; create a fresh preview and apply it."
            )
        assignments = self._validate_repair_journal_scope(journal)
        if action == "resume":
            result = self._continue_ownership_repair(store, journal_path, journal)
            result["recovered"] = True
            return result
        return self._rollback_ownership_repair(store, journal_path, journal, assignments)

    def _rollback_ownership_repair(
        self,
        store: EvidenceStore,
        journal_path: Path,
        journal: Dict[str, Any],
        assignments: Dict[str, str],
    ) -> Dict[str, Any]:
        before_catalog = journal["before_catalog"]
        documents = store.read_catalog()
        by_id: Dict[str, List[DocumentMetadata]] = {}
        for metadata in documents:
            by_id.setdefault(metadata.document_id, []).append(metadata)
        for document_id, target in assignments.items():
            records = by_id.get(document_id, [])
            before = before_catalog.get(document_id)
            if len(records) != 1 or before is None:
                raise DatasetOwnershipError(
                    f"Cannot roll back: catalog record '{document_id}' is missing or duplicated."
                )
            owner_before = before.get("dataset_id")
            if records[0].dataset_id not in (owner_before, target):
                raise DatasetOwnershipError(
                    f"Cannot roll back unexpected catalog owner for '{document_id}'."
                )
            current_without_owner = records[0].to_dict()
            current_without_owner["dataset_id"] = owner_before
            if current_without_owner != before:
                raise DatasetOwnershipError(
                    f"Cannot roll back catalog metadata changed for '{document_id}'."
                )
            records[0].dataset_id = owner_before
        store._write_catalog(documents)
        journal["status"] = "rollback_catalog_applied"
        self._write_repair_journal(journal_path, journal)

        indexed_before = {
            row["document_id"]: row for row in journal["before_indexed_documents"]
        }
        chunks_before = {row["chunk_id"]: row for row in journal["before_chunks"]}
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for document_id, target in assignments.items():
                before = indexed_before.get(document_id)
                current = conn.execute(
                    "SELECT dataset_id, content_hash, chunk_count FROM indexed_documents "
                    "WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
                if before is None or len(current) != 1:
                    raise DatasetOwnershipError(
                        f"Cannot roll back missing indexed row for '{document_id}'."
                    )
                row = dict(current[0])
                if row["dataset_id"] not in (before["dataset_id"], target):
                    raise DatasetOwnershipError(
                        f"Cannot roll back unexpected indexed owner for '{document_id}'."
                    )
                if row["content_hash"] != before["content_hash"] or row["chunk_count"] != before["chunk_count"]:
                    raise DatasetOwnershipError(
                        f"Cannot roll back changed indexed content for '{document_id}'."
                    )
                conn.execute(
                    "UPDATE indexed_documents SET dataset_id = ? WHERE document_id = ?",
                    (before["dataset_id"], document_id),
                )
            current_chunks = {
                row["chunk_id"]: dict(row)
                for row in conn.execute(
                    "SELECT chunk_id, document_id, dataset_id, content_hash "
                    "FROM evidence_chunks WHERE document_id IN (?, ?, ?, ?, ?, ?)",
                    tuple(sorted(assignments)),
                ).fetchall()
            }
            if set(current_chunks) != set(chunks_before):
                raise DatasetOwnershipError("Cannot roll back changed relational chunk set.")
            for chunk_id, before in chunks_before.items():
                row = current_chunks[chunk_id]
                if (
                    row["document_id"] != before["document_id"]
                    or row["content_hash"] != before["content_hash"]
                    or row["dataset_id"] not in (before["dataset_id"], DEFAULT_DATASET_ID)
                ):
                    raise DatasetOwnershipError(
                        f"Cannot roll back changed relational chunk '{chunk_id}'."
                    )
                conn.execute(
                    "UPDATE evidence_chunks SET dataset_id = ? WHERE chunk_id = ?",
                    (before["dataset_id"], chunk_id),
                )
        journal["status"] = "rolled_back"
        self._write_repair_journal(journal_path, journal)
        return {
            "repair_id": journal["repair_id"],
            "status": "rolled_back",
            "verified": True,
        }

    def get_index_stats(self) -> Dict[str, Any]:
        """Return counts of indexed documents, chunks, and last indexing timestamp."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), MAX(indexed_at) FROM indexed_documents")
            doc_row = cursor.fetchone()
            doc_count = doc_row[0] if doc_row else 0
            last_indexed = doc_row[1] if doc_row else None

            cursor.execute("SELECT COUNT(*) FROM evidence_chunks")
            chunk_row = cursor.fetchone()
            chunk_count = chunk_row[0] if chunk_row else 0

            return {
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "last_indexed_at": last_indexed,
            }

    def is_document_indexed(
        self,
        document_id: str,
        current_content_hash: Optional[str] = None,
        current_dataset_id: Optional[str] = None,
    ) -> bool:
        """Check if a document is already indexed, and whether its hash and dataset_id match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content_hash, dataset_id FROM indexed_documents WHERE document_id = ?",
                (document_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False
            if current_content_hash is not None and row["content_hash"] != current_content_hash:
                return False
            if current_dataset_id is not None and row["dataset_id"] != current_dataset_id:
                return False
            return True

    def remove_document(self, document_id: str) -> None:
        """Remove all chunks and FTS records for a document within a transaction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM evidence_chunks WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM indexed_documents WHERE document_id = ?", (document_id,))
            conn.commit()

    def index_document(
        self,
        metadata: DocumentMetadata,
        extracted_text: str,
        page_number: Optional[int] = None,
        store: Optional[EvidenceStore] = None,
    ) -> int:
        """
        Index or re-index a single document:
        - If unchanged, skips re-indexing.
        - If updated or new, replaces existing chunks atomically.
        Returns number of chunks indexed.
        """
        if metadata.dataset_id is None or not str(metadata.dataset_id).strip():
            raise DatasetOwnershipError(
                f"Document '{metadata.document_id}' has no dataset ID and cannot be indexed."
            )

        if store is None:
            raise DatasetOwnershipError(
                "Indexing requires an EvidenceStore to validate canonical catalog ownership."
            )
        catalog_metadata = store.get_metadata(metadata.document_id)
        if catalog_metadata is None:
            raise DatasetOwnershipError(
                f"Document '{metadata.document_id}' is not present in the evidence catalog."
            )
        if (
            metadata.dataset_id != catalog_metadata.dataset_id
            or metadata.original_filename != catalog_metadata.original_filename
            or metadata.file_type != catalog_metadata.file_type
        ):
            raise DatasetOwnershipError(
                f"Document '{metadata.document_id}' metadata does not match the canonical "
                "catalog ownership and provenance."
            )
        if catalog_metadata.ingestion_status != DocumentStatus.PROCESSED:
            raise DatasetOwnershipError(
                f"Document '{metadata.document_id}' is not processed and cannot be indexed."
            )
        target_dataset_id = store.validate_document_ownership(
            catalog_metadata,
            require_catalog_match=True,
        ).dataset_id
        metadata = catalog_metadata

        if not extracted_text or not extracted_text.strip():
            # If document has no text, remove from index if previously indexed
            self.remove_document(metadata.document_id)
            return 0

        doc_hash = compute_content_hash(extracted_text)

        # Check if already indexed with identical content and dataset_id
        if self.is_document_indexed(metadata.document_id, doc_hash, target_dataset_id):
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT chunk_count FROM indexed_documents WHERE document_id = ?",
                    (metadata.document_id,),
                )
                row = cursor.fetchone()
                return row["chunk_count"] if row else 0

        # Generate fresh chunks
        chunks = self.chunker.chunk_document(metadata, extracted_text, page_number)
        if not chunks:
            self.remove_document(metadata.document_id)
            return 0

        # Ensure all chunks have the document's dataset_id
        for chunk in chunks:
            chunk.dataset_id = target_dataset_id

        timestamp = current_iso_timestamp()

        # Atomic transaction: remove old entries and insert new chunks
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Clean any previous chunks for this document
            cursor.execute("DELETE FROM chunks_fts WHERE document_id = ?", (metadata.document_id,))
            cursor.execute("DELETE FROM evidence_chunks WHERE document_id = ?", (metadata.document_id,))
            cursor.execute("DELETE FROM indexed_documents WHERE document_id = ?", (metadata.document_id,))

            # Insert tracking record
            cursor.execute(
                """
                INSERT INTO indexed_documents (document_id, source_filename, content_hash, chunk_count, indexed_at, dataset_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (metadata.document_id, metadata.original_filename, doc_hash, len(chunks), timestamp, target_dataset_id),
            )

            # Insert chunks into relational table and FTS5 table
            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT INTO evidence_chunks (
                        chunk_id, document_id, chunk_index, text, source_filename,
                        file_type, page_number, start_offset, end_offset, created_at, content_hash, dataset_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.source_filename,
                        chunk.file_type,
                        chunk.page_number,
                        chunk.start_offset,
                        chunk.end_offset,
                        chunk.created_at,
                        chunk.content_hash,
                        target_dataset_id,
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, document_id, source_filename, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chunk.chunk_id, chunk.document_id, chunk.source_filename, chunk.text),
                )

            conn.commit()

        return len(chunks)

    def sync_with_store(self, store: EvidenceStore) -> Dict[str, Any]:
        """
        Apply a fresh, validated synchronization preview.
        """
        plan = self.preview_sync_with_store(store)
        if not plan["ready"]:
            return {
                "indexed_documents": 0,
                "total_chunks": 0,
                "removed_stale": 0,
                "errors": plan["errors"],
                "informational": plan["informational"],
                "preview": plan,
            }
        return self.apply_sync_with_store(store, plan)

    def _maintenance_integrity_status(
        self,
        store: EvidenceStore,
    ) -> Tuple[List[DatasetAssociationIssue], List[Dict[str, str]]]:
        """Separate only confirmed failed, unassigned, unindexed catalog records."""
        issues = self.validate_integrity(store)
        catalog = store.read_catalog()
        catalog_records: Dict[str, List[DocumentMetadata]] = {}
        for metadata in catalog:
            catalog_records.setdefault(metadata.document_id, []).append(metadata)

        with self._get_connection() as conn:
            indexed_document_ids = {
                row["document_id"]
                for row in conn.execute(
                    "SELECT document_id FROM indexed_documents"
                ).fetchall()
            }
            chunk_document_ids = {
                row["document_id"]
                for row in conn.execute(
                    "SELECT DISTINCT document_id FROM evidence_chunks"
                ).fetchall()
            }
            fts_document_ids = {
                row["document_id"]
                for row in conn.execute(
                    "SELECT DISTINCT document_id FROM chunks_fts"
                ).fetchall()
            }

        expected_failed_ids = {
            document_id
            for document_id, records in catalog_records.items()
            if len(records) == 1
            and records[0].ingestion_status == DocumentStatus.FAILED
            and records[0].dataset_id is None
            and document_id not in indexed_document_ids
            and document_id not in chunk_document_ids
            and document_id not in fts_document_ids
        }
        informational = [
            {
                "code": "expected_failed_unindexed_document",
                "document_id": document_id,
                "message": (
                    f"Document '{document_id}' is failed, unassigned, and absent from "
                    "the index; maintenance will leave it unchanged."
                ),
            }
            for document_id in sorted(expected_failed_ids)
        ]
        blocking_issues = [
            issue
            for issue in issues
            if not (
                issue.code == "missing_dataset_id"
                and issue.entity_type == "document"
                and issue.entity_id in expected_failed_ids
            )
        ]
        return blocking_issues, informational

    def preview_sync_with_store(self, store: EvidenceStore) -> Dict[str, Any]:
        """Preview an idempotent catalog-to-index sync without mutating either store."""
        catalog = store.read_catalog()
        catalog_by_id = {doc.document_id: doc for doc in catalog}
        issues, informational = self._maintenance_integrity_status(store)
        repairable_codes = {"index_missing_indexed_document"}
        blocking_issues = [issue for issue in issues if issue.code not in repairable_codes]
        errors = [issue.message for issue in blocking_issues]
        indexed_ids = {
            row["document_id"]
            for row in self._get_connection().execute(
                "SELECT document_id FROM indexed_documents"
            ).fetchall()
        }
        processed_docs = [
            doc for doc in catalog
            if doc.ingestion_status == DocumentStatus.PROCESSED
        ]
        to_index: List[str] = []
        if not errors:
            for metadata in processed_docs:
                extracted_text = store.get_extracted_text(metadata)
                if extracted_text is None:
                    errors.append(
                        f"Cannot synchronize '{metadata.document_id}': extracted text "
                        "is missing from local storage."
                    )
                    continue
                if not self.is_document_indexed(
                    metadata.document_id,
                    compute_content_hash(extracted_text),
                    metadata.dataset_id,
                ):
                    to_index.append(metadata.document_id)

        catalog_snapshot = [
            {
                "document_id": doc.document_id,
                "dataset_id": doc.dataset_id,
                "status": doc.ingestion_status.value,
                "updated_at": doc.updated_at,
            }
            for doc in catalog
        ]
        return {
            "ready": not errors,
            "to_index_document_ids": to_index,
            "unresolved_index_document_ids": sorted(indexed_ids - set(catalog_by_id)),
            "catalog_snapshot": catalog_snapshot,
            "errors": errors,
            "informational": informational,
        }

    def apply_sync_with_store(
        self,
        store: EvidenceStore,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply an unchanged sync preview and verify its result; retries are idempotent."""
        if not isinstance(plan, dict) or not plan.get("ready"):
            raise DatasetOwnershipError("Index synchronization preview is invalid or not ready.")
        current_plan = self.preview_sync_with_store(store)
        if not current_plan["ready"]:
            raise DatasetOwnershipError(
                "Index synchronization was not applied: "
                + " ".join(current_plan["errors"])
            )
        if (
            current_plan["catalog_snapshot"] != plan.get("catalog_snapshot")
            or current_plan["to_index_document_ids"] != plan.get("to_index_document_ids")
        ):
            raise DatasetOwnershipError(
                "Index synchronization preview is stale; create a new preview before applying."
            )

        indexed_count = 0
        total_chunks = 0
        errors: List[str] = []
        for document_id in plan["to_index_document_ids"]:
            metadata = store.get_metadata(document_id)
            if metadata is None:
                errors.append(
                    f"Cannot synchronize '{document_id}': catalog document disappeared."
                )
                continue
            extracted_text = store.get_extracted_text(metadata)
            if extracted_text is None:
                errors.append(
                    f"Cannot synchronize '{document_id}': extracted text is missing."
                )
                continue
            try:
                chunks_added = self.index_document(
                    metadata,
                    extracted_text,
                    store=store,
                )
            except (DatasetOwnershipError, OSError, sqlite3.Error) as exc:
                errors.append(f"Failed to index {document_id}: {exc}")
                continue
            if chunks_added > 0:
                indexed_count += 1
                total_chunks += chunks_added

        verification_issues, informational = self._maintenance_integrity_status(store)
        errors.extend(issue.message for issue in verification_issues)
        return {
            "indexed_documents": indexed_count,
            "total_chunks": total_chunks,
            "removed_stale": 0,
            "errors": errors,
            "informational": informational,
            "verification_issues": verification_issues,
            "verified": not verification_issues,
        }

    def validate_integrity(
        self,
        store: EvidenceStore,
    ) -> List[DatasetAssociationIssue]:
        """Report catalog, registry, index, chunk, and FTS ownership inconsistencies."""
        catalog = store.read_catalog()
        catalog_by_id: Dict[str, DocumentMetadata] = {}
        issues = store.validate_catalog_ownership()
        registered_ids = {dataset.dataset_id for dataset in store.list_datasets()}

        for metadata in catalog:
            if metadata.document_id in catalog_by_id:
                issues.append(
                    DatasetAssociationIssue(
                        code="duplicate_document_id",
                        entity_type="document",
                        entity_id=metadata.document_id,
                        dataset_id=metadata.dataset_id,
                        message=f"Document ID '{metadata.document_id}' appears more than once in the catalog.",
                    )
                )
            else:
                catalog_by_id[metadata.document_id] = metadata

        with self._get_connection() as conn:
            indexed_rows = conn.execute(
                "SELECT document_id, dataset_id, source_filename, chunk_count FROM indexed_documents"
            ).fetchall()
            chunk_rows = conn.execute(
                """
                SELECT c.chunk_id, c.document_id, c.dataset_id, c.source_filename, c.file_type,
                       c.text, d.dataset_id AS document_dataset_id,
                       d.source_filename AS indexed_source_filename
                FROM evidence_chunks c
                LEFT JOIN indexed_documents d ON d.document_id = c.document_id
                """
            ).fetchall()
            fts_rows = conn.execute(
                "SELECT chunk_id, document_id, source_filename, text FROM chunks_fts"
            ).fetchall()
            chunk_counts = {
                row["document_id"]: row["chunk_count"]
                for row in conn.execute(
                    "SELECT document_id, COUNT(*) AS chunk_count FROM evidence_chunks GROUP BY document_id"
                ).fetchall()
            }
            relational_chunk_ids = {
                row["chunk_id"] for row in conn.execute(
                    "SELECT chunk_id FROM evidence_chunks"
                ).fetchall()
            }
            fts_document_by_chunk = {
                row["chunk_id"]: row["document_id"] for row in fts_rows
            }
            fts_by_chunk = {row["chunk_id"]: row for row in fts_rows}
            indexed_chunk_ids = set(fts_document_by_chunk)

        for row in indexed_rows:
            document_id = row["document_id"]
            indexed_dataset_id = row["dataset_id"]
            metadata = catalog_by_id.get(document_id)
            if metadata is None:
                issues.append(
                    DatasetAssociationIssue(
                        code="index_missing_document",
                        entity_type="indexed_document",
                        entity_id=document_id,
                        dataset_id=indexed_dataset_id,
                        message=f"Indexed document '{document_id}' has no catalog document.",
                    )
                )
            elif metadata.dataset_id != indexed_dataset_id:
                issues.append(
                    DatasetAssociationIssue(
                        code="index_document_dataset_mismatch",
                        entity_type="indexed_document",
                        entity_id=document_id,
                        dataset_id=indexed_dataset_id,
                        message=(
                            f"Indexed document '{document_id}' belongs to dataset "
                            f"'{indexed_dataset_id}', but its catalog record belongs to "
                            f"'{metadata.dataset_id}'."
                        ),
                    )
                )
            if metadata is not None and metadata.original_filename != row["source_filename"]:
                issues.append(
                    DatasetAssociationIssue(
                        code="index_document_filename_mismatch",
                        entity_type="indexed_document",
                        entity_id=document_id,
                        dataset_id=indexed_dataset_id,
                        message=(
                            f"Indexed document '{document_id}' filename does not match "
                            "its catalog record."
                        ),
                    )
                )

            if not indexed_dataset_id or not str(indexed_dataset_id).strip():
                issues.append(
                    DatasetAssociationIssue(
                        code="index_missing_dataset_id",
                        entity_type="indexed_document",
                        entity_id=document_id,
                        dataset_id=None,
                        message=f"Indexed document '{document_id}' has no dataset ID.",
                    )
                )
            elif indexed_dataset_id not in registered_ids:
                issues.append(
                    DatasetAssociationIssue(
                        code="index_unregistered_dataset_id",
                        entity_type="indexed_document",
                        entity_id=document_id,
                        dataset_id=indexed_dataset_id,
                        message=(
                            f"Indexed document '{document_id}' references unregistered "
                            f"dataset '{indexed_dataset_id}'."
                        ),
                    )
                )
            if chunk_counts.get(document_id, 0) != row["chunk_count"]:
                issues.append(
                    DatasetAssociationIssue(
                        code="index_chunk_count_mismatch",
                        entity_type="indexed_document",
                        entity_id=document_id,
                        dataset_id=indexed_dataset_id,
                        message=(
                            f"Indexed document '{document_id}' records {row['chunk_count']} "
                            f"chunks, but {chunk_counts.get(document_id, 0)} chunk records exist."
                        ),
                    )
                )

        for row in chunk_rows:
            if row["document_dataset_id"] is None:
                issues.append(
                    DatasetAssociationIssue(
                        code="index_chunk_missing_document",
                        entity_type="indexed_chunk",
                        entity_id=row["chunk_id"],
                        dataset_id=row["dataset_id"],
                        message=(
                            f"Indexed chunk '{row['chunk_id']}' references missing "
                            f"indexed document '{row['document_id']}'."
                        ),
                    )
                )
            metadata = catalog_by_id.get(row["document_id"])
            if metadata is not None and (
                row["source_filename"] != metadata.original_filename
                or row["file_type"] != metadata.file_type
            ):
                issues.append(
                    DatasetAssociationIssue(
                        code="index_chunk_provenance_mismatch",
                        entity_type="indexed_chunk",
                        entity_id=row["chunk_id"],
                        dataset_id=row["dataset_id"],
                        message=(
                            f"Indexed chunk '{row['chunk_id']}' filename or file type "
                            f"does not match catalog document '{row['document_id']}'."
                        ),
                    )
                )
            if (
                row["indexed_source_filename"] is not None
                and row["source_filename"] != row["indexed_source_filename"]
            ):
                issues.append(
                    DatasetAssociationIssue(
                        code="index_chunk_filename_mismatch",
                        entity_type="indexed_chunk",
                        entity_id=row["chunk_id"],
                        dataset_id=row["dataset_id"],
                        message=(
                            f"Indexed chunk '{row['chunk_id']}' filename does not match "
                            f"indexed document '{row['document_id']}'."
                        ),
                    )
                )
            if (
                row["document_dataset_id"] is not None
                and row["dataset_id"] != row["document_dataset_id"]
            ):
                issues.append(
                    DatasetAssociationIssue(
                        code="index_chunk_dataset_mismatch",
                        entity_type="indexed_chunk",
                        entity_id=row["chunk_id"],
                        dataset_id=row["dataset_id"],
                        message=(
                            f"Indexed chunk '{row['chunk_id']}' dataset ID does not match "
                            f"indexed document '{row['document_id']}'."
                        ),
                    )
                )

        for chunk_id in relational_chunk_ids - indexed_chunk_ids:
            issues.append(
                DatasetAssociationIssue(
                    code="index_chunk_missing_fts_record",
                    entity_type="indexed_chunk",
                    entity_id=chunk_id,
                    dataset_id=None,
                    message=f"Indexed chunk '{chunk_id}' is absent from the FTS table.",
                )
            )
        for chunk_id in indexed_chunk_ids - relational_chunk_ids:
            issues.append(
                DatasetAssociationIssue(
                    code="fts_record_missing_index_chunk",
                    entity_type="fts_record",
                    entity_id=chunk_id,
                    dataset_id=None,
                    message=f"FTS record '{chunk_id}' has no relational evidence chunk.",
                )
            )
        for row in chunk_rows:
            fts_document_id = fts_document_by_chunk.get(row["chunk_id"])
            if fts_document_id is not None and fts_document_id != row["document_id"]:
                issues.append(
                    DatasetAssociationIssue(
                        code="fts_document_mismatch",
                        entity_type="fts_record",
                        entity_id=row["chunk_id"],
                        dataset_id=row["dataset_id"],
                        message=(
                            f"FTS record '{row['chunk_id']}' references document "
                            f"'{fts_document_id}', not '{row['document_id']}'."
                        ),
                    )
                )
            fts_row = fts_by_chunk.get(row["chunk_id"])
            if fts_row is not None and (
                fts_row["source_filename"] != row["source_filename"]
                or fts_row["text"] != row["text"]
            ):
                issues.append(
                    DatasetAssociationIssue(
                        code="fts_chunk_content_mismatch",
                        entity_type="fts_record",
                        entity_id=row["chunk_id"],
                        dataset_id=row["dataset_id"],
                        message=(
                            f"FTS record '{row['chunk_id']}' content or filename does not "
                            "match its relational evidence chunk."
                        ),
                    )
                )

        return issues

    def summarize_integrity(
        self,
        store: EvidenceStore,
    ) -> Dict[str, Any]:
        """Summarize global ownership integrity by overall scope and unique affected documents."""
        issues = self.validate_integrity(store)
        registered_ids = {dataset.dataset_id for dataset in store.list_datasets()}
        findings_by_scope: Dict[str, Dict[str, Any]] = {}

        def ensure_scope(scope: str) -> Dict[str, Any]:
            if scope not in findings_by_scope:
                findings_by_scope[scope] = {
                    "total_findings": 0,
                    "affected_document_count": 0,
                    "issue_codes": [],
                }
            return findings_by_scope[scope]

        unique_documents: set[str] = set()
        chunk_document_map: Dict[str, str] = {}
        with self._get_connection() as conn:
            for row in conn.execute(
                "SELECT chunk_id, document_id FROM evidence_chunks"
            ).fetchall():
                chunk_document_map[row["chunk_id"]] = row["document_id"]

        for issue in issues:
            if issue.dataset_id in registered_ids:
                scope = issue.dataset_id
            elif issue.dataset_id is None:
                scope = "UNASSIGNED"
            elif issue.dataset_id:
                scope = "UNREGISTERED"
            else:
                scope = "UNASSIGNED"
            scope_summary = ensure_scope(scope)
            scope_summary["total_findings"] += 1
            if issue.code not in scope_summary["issue_codes"]:
                scope_summary["issue_codes"].append(issue.code)

            entity_id = issue.entity_id
            if issue.entity_type in {"indexed_chunk", "fts_record"} and entity_id in chunk_document_map:
                entity_id = chunk_document_map[entity_id]
            if issue.entity_type in {"document", "indexed_document", "indexed_chunk", "fts_record"}:
                unique_documents.add(entity_id)

            if issue.entity_type in {"document", "indexed_document"}:
                scope_summary["affected_document_count"] = len({
                    item for item in unique_documents if item == entity_id
                })

        for scope in findings_by_scope:
            scope_summary = findings_by_scope[scope]
            if scope_summary["affected_document_count"] == 0:
                scope_summary["affected_document_count"] = len({
                    issue.entity_id
                    for issue in issues
                    if (
                        (issue.dataset_id in registered_ids and issue.dataset_id == scope)
                        or (issue.dataset_id is None and scope == "UNASSIGNED")
                        or (issue.dataset_id and issue.dataset_id not in registered_ids and scope == "UNREGISTERED")
                    )
                })

        for dataset_id in sorted(registered_ids):
            ensure_scope(dataset_id)
        ensure_scope("UNASSIGNED")
        ensure_scope("UNREGISTERED")

        findings_by_scope = dict(sorted(findings_by_scope.items()))
        return {
            "total_findings": len(issues),
            "unique_affected_document_count": len(unique_documents),
            "findings_by_scope": findings_by_scope,
            "project_wide_diagnostics": findings_by_scope,
            "active_dataset_diagnostics": findings_by_scope,
        }

    def validate_dataset_integrity(
        self,
        store: EvidenceStore,
        dataset_id: str,
    ) -> List[DatasetAssociationIssue]:
        """Return integrity issues affecting this dataset without blocking clean datasets."""
        self._validate_dataset_scope(store, dataset_id)
        catalog = store.read_catalog()
        target_document_ids = {
            metadata.document_id
            for metadata in catalog
            if metadata.dataset_id == dataset_id
        }

        with self._get_connection() as conn:
            indexed_document_ids = {
                row["document_id"]
                for row in conn.execute(
                    "SELECT document_id FROM indexed_documents WHERE dataset_id = ?",
                    (dataset_id,),
                ).fetchall()
            }
            chunk_rows = conn.execute(
                """
                SELECT chunk_id, document_id
                FROM evidence_chunks
                WHERE dataset_id = ?
                """,
                (dataset_id,),
            ).fetchall()
            fts_target_chunk_ids = {
                row["chunk_id"]
                for row in conn.execute(
                    "SELECT chunk_id, document_id FROM chunks_fts"
                ).fetchall()
                if row["document_id"] in target_document_ids | indexed_document_ids
            }

        target_document_ids |= indexed_document_ids
        target_chunk_ids = {
            row["chunk_id"] for row in chunk_rows
        } | fts_target_chunk_ids
        target_document_ids |= {
            row["document_id"] for row in chunk_rows
        }

        return [
            issue
            for issue in self.validate_integrity(store)
            if issue.dataset_id == dataset_id
            or issue.entity_id in target_document_ids
            or issue.entity_id in target_chunk_ids
        ]

    @staticmethod
    def _validate_dataset_scope(store: EvidenceStore, dataset_id: Optional[str]) -> None:
        store.require_dataset_id(dataset_id, "Evidence search", "active dataset")

    def rebuild_index(self, store: EvidenceStore) -> Dict[str, Any]:
        """
        Completely wipe and rebuild the FTS5 index from the catalog and extracted text.
        Atomic and safe: does not touch original uploaded evidence files.
        """
        integrity_issues, informational = self._maintenance_integrity_status(store)
        if integrity_issues:
            details = " ".join(issue.message for issue in integrity_issues[:5])
            raise DatasetOwnershipError(
                "Cannot rebuild the evidence index while dataset ownership or index "
                f"integrity issues remain ({len(integrity_issues)} issue(s)). {details}"
            )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks_fts")
            cursor.execute("DELETE FROM evidence_chunks")
            cursor.execute("DELETE FROM indexed_documents")
            conn.commit()

        result = self.sync_with_store(store)
        result["informational"] = result.get("informational", informational)
        return result

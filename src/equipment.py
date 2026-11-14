"""Dataset-scoped equipment registry stored separately from evidence and history."""

import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import pandas as pd

from src.equipment_taxonomy import (
    CLASSIFICATION_CANDIDATE,
    CLASSIFICATION_UNCLASSIFIED,
    EQUIPMENT_TAXONOMY,
    TAXONOMY_VERSION,
)
from src.models import DatasetOwnershipError
from src.storage import EvidenceStore, EvidenceStoreError


class EquipmentRegistryError(RuntimeError):
    """Raised when an equipment registry operation cannot be safely completed."""


class EquipmentRegistry:
    """Persist equipment, components, and explicit incident links in a separate SQLite store."""

    SCHEMA_VERSION = 1
    REQUIRED_TABLES = {
        "equipment_assets",
        "equipment_components",
        "incident_equipment_associations",
    }
    CRITICALITY_VALUES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    ASSOCIATION_ROLES = {"PRIMARY", "RELATED"}
    ASSOCIATION_STATUSES = {"PENDING_REVIEW", "REVIEWED", "REJECTED"}

    def __init__(
        self,
        store: EvidenceStore,
        db_path: Optional[Path] = None,
    ) -> None:
        if not isinstance(store, EvidenceStore):
            raise TypeError("EquipmentRegistry requires an EvidenceStore.")
        self.store = store
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.db_path = project_root / "data" / "equipment" / "equipment_registry.db"
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
            raise EquipmentRegistryError(
                "The equipment registry database is unavailable."
            ) from None

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except sqlite3.IntegrityError:
            connection.rollback()
            raise EquipmentRegistryError(
                "The equipment registry rejected a duplicate or inconsistent record."
            ) from None
        except sqlite3.Error:
            connection.rollback()
            raise EquipmentRegistryError(
                "The equipment registry could not complete the database operation."
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
                    raise EquipmentRegistryError(
                        f"Equipment database schema version {version} is newer than supported."
                    )
                if version == self.SCHEMA_VERSION:
                    if not self.REQUIRED_TABLES.issubset(tables):
                        raise EquipmentRegistryError(
                            "Equipment database schema is marked current but is incomplete."
                        )
                    return
                if tables:
                    raise EquipmentRegistryError(
                        "Equipment database contains unversioned tables; refusing initialization."
                    )

                connection.execute("BEGIN IMMEDIATE")
                self._create_schema(connection)
                connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
                connection.commit()
        except EquipmentRegistryError:
            raise
        except OSError:
            raise EquipmentRegistryError(
                "The equipment database directory is unavailable."
            ) from None
        except sqlite3.Error:
            raise EquipmentRegistryError(
                "The equipment database schema could not be initialized."
            ) from None

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE equipment_assets (
                dataset_id TEXT NOT NULL,
                equipment_id TEXT NOT NULL,
                asset_name TEXT NOT NULL,
                category_code TEXT NOT NULL,
                subtype_code TEXT,
                manufacturer TEXT,
                model TEXT,
                serial_number TEXT,
                asset_tag TEXT,
                legacy_machine_id TEXT,
                installation_date TEXT,
                location TEXT,
                criticality TEXT CHECK (
                    criticality IS NULL OR criticality IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
                ),
                taxonomy_version TEXT NOT NULL,
                classification_status TEXT NOT NULL CHECK (
                    classification_status IN ('CANDIDATE', 'UNCLASSIFIED')
                ),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (dataset_id, equipment_id)
            )
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX uq_equipment_asset_tag
            ON equipment_assets(dataset_id, lower(asset_tag))
            WHERE asset_tag IS NOT NULL
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_equipment_assets_dataset_category
            ON equipment_assets(dataset_id, category_code, asset_name)
            """
        )
        connection.execute(
            """
            CREATE TABLE equipment_components (
                dataset_id TEXT NOT NULL,
                component_id TEXT NOT NULL,
                equipment_id TEXT NOT NULL,
                parent_component_id TEXT,
                component_name TEXT NOT NULL,
                component_code TEXT,
                manufacturer TEXT,
                model TEXT,
                serial_number TEXT,
                taxonomy_version TEXT NOT NULL,
                classification_status TEXT NOT NULL CHECK (
                    classification_status IN ('CANDIDATE', 'UNCLASSIFIED')
                ),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (dataset_id, component_id),
                UNIQUE (dataset_id, equipment_id, component_id),
                FOREIGN KEY (dataset_id, equipment_id)
                    REFERENCES equipment_assets(dataset_id, equipment_id)
                    ON DELETE RESTRICT,
                FOREIGN KEY (dataset_id, equipment_id, parent_component_id)
                    REFERENCES equipment_components(dataset_id, equipment_id, component_id)
                    ON DELETE RESTRICT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_equipment_components_parent
            ON equipment_components(dataset_id, equipment_id, parent_component_id)
            """
        )
        connection.execute(
            """
            CREATE TABLE incident_equipment_associations (
                association_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                incident_id TEXT NOT NULL,
                equipment_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('PRIMARY', 'RELATED')),
                link_source TEXT NOT NULL CHECK (link_source = 'USER_SELECTED'),
                review_status TEXT NOT NULL CHECK (
                    review_status IN ('PENDING_REVIEW', 'REVIEWED', 'REJECTED')
                ),
                review_notes TEXT,
                reviewed_by TEXT,
                reviewed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (dataset_id, incident_id, equipment_id),
                FOREIGN KEY (dataset_id, equipment_id)
                    REFERENCES equipment_assets(dataset_id, equipment_id)
                    ON DELETE RESTRICT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_incident_equipment_dataset_incident
            ON incident_equipment_associations(dataset_id, incident_id, review_status)
            """
        )

    def _require_dataset(self, dataset_id: str) -> None:
        self.store.require_dataset_id(dataset_id, "Equipment registry", dataset_id)

    @staticmethod
    def _normalize_uuid(value: str, field_name: str) -> str:
        try:
            return str(uuid.UUID(str(value)))
        except (ValueError, TypeError, AttributeError):
            raise EquipmentRegistryError(f"{field_name} must be a valid UUID.") from None

    @staticmethod
    def _required_text(value: Any, field_name: str, max_length: int = 200) -> str:
        if not isinstance(value, str):
            raise EquipmentRegistryError(f"{field_name} is required.")
        cleaned = value.strip()
        if not cleaned or len(cleaned) > max_length or "\x00" in cleaned:
            raise EquipmentRegistryError(
                f"{field_name} must contain 1 to {max_length} valid characters."
            )
        return cleaned

    @classmethod
    def _optional_text(
        cls,
        value: Any,
        field_name: str,
        max_length: int = 200,
    ) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise EquipmentRegistryError(f"{field_name} must be text when provided.")
        cleaned = value.strip()
        if not cleaned:
            return None
        if len(cleaned) > max_length or "\x00" in cleaned:
            raise EquipmentRegistryError(
                f"{field_name} must be at most {max_length} valid characters."
            )
        return cleaned

    @staticmethod
    def _validate_date(value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            raise EquipmentRegistryError("Installation date must use YYYY-MM-DD.")
        if isinstance(value, date):
            return value.isoformat()
        if not isinstance(value, str):
            raise EquipmentRegistryError("Installation date must use YYYY-MM-DD.")
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise EquipmentRegistryError("Installation date must use YYYY-MM-DD.") from None
        if parsed.isoformat() != value:
            raise EquipmentRegistryError("Installation date must use YYYY-MM-DD.")
        return parsed.isoformat()

    @classmethod
    def _validate_asset_fields(
        cls,
        fields: Mapping[str, Any],
    ) -> Dict[str, Any]:
        asset_name = cls._required_text(fields.get("asset_name"), "Equipment name")
        category = cls._required_text(fields.get("category_code"), "Equipment category", 80)
        definition = EQUIPMENT_TAXONOMY.get(category)
        if definition is None:
            raise EquipmentRegistryError(
                "Unknown equipment category. Choose a candidate category or unclassified."
            )

        subtype = cls._optional_text(fields.get("subtype_code"), "Equipment subtype", 80)
        if category != "unclassified" and subtype is not None:
            if subtype not in definition["subtypes"]:
                raise EquipmentRegistryError(
                    "Equipment subtype is not supported by the selected candidate category."
                )

        criticality = cls._optional_text(fields.get("criticality"), "Criticality", 20)
        if criticality is not None:
            criticality = criticality.upper()
            if criticality not in cls.CRITICALITY_VALUES:
                raise EquipmentRegistryError(
                    "Criticality must be LOW, MEDIUM, HIGH, or CRITICAL."
                )

        return {
            "asset_name": asset_name,
            "category_code": category,
            "subtype_code": subtype,
            "manufacturer": cls._optional_text(fields.get("manufacturer"), "Manufacturer", 120),
            "model": cls._optional_text(fields.get("model"), "Model", 120),
            "serial_number": cls._optional_text(fields.get("serial_number"), "Serial number", 120),
            "asset_tag": cls._optional_text(fields.get("asset_tag"), "Asset tag", 120),
            "legacy_machine_id": cls._optional_text(
                fields.get("legacy_machine_id"), "Legacy machine identifier", 120
            ),
            "installation_date": cls._validate_date(fields.get("installation_date")),
            "location": cls._optional_text(fields.get("location"), "Location", 200),
            "criticality": criticality,
            "taxonomy_version": TAXONOMY_VERSION,
            "classification_status": (
                CLASSIFICATION_UNCLASSIFIED
                if category == "unclassified"
                else CLASSIFICATION_CANDIDATE
            ),
        }

    @staticmethod
    def _check_duplicate_identity(
        connection: sqlite3.Connection,
        dataset_id: str,
        fields: Mapping[str, Any],
        exclude_equipment_id: Optional[str] = None,
    ) -> None:
        if fields.get("asset_tag"):
            query = """
                SELECT 1 FROM equipment_assets
                WHERE dataset_id = ? AND lower(asset_tag) = lower(?)
            """
            params: List[Any] = [dataset_id, fields["asset_tag"]]
            if exclude_equipment_id:
                query += " AND equipment_id != ?"
                params.append(exclude_equipment_id)
            if connection.execute(query, params).fetchone():
                raise EquipmentRegistryError(
                    "An equipment record with this asset tag already exists in the dataset."
                )

        if fields.get("serial_number"):
            query = """
                SELECT 1 FROM equipment_assets
                WHERE dataset_id = ?
                  AND lower(serial_number) = lower(?)
                  AND lower(coalesce(manufacturer, '')) = lower(coalesce(?, ''))
            """
            params = [dataset_id, fields["serial_number"], fields.get("manufacturer")]
            if exclude_equipment_id:
                query += " AND equipment_id != ?"
                params.append(exclude_equipment_id)
            if connection.execute(query, params).fetchone():
                raise EquipmentRegistryError(
                    "A matching manufacturer and serial number already exists in this dataset."
                )

    def create_equipment(
        self,
        dataset_id: str,
        asset_name: str,
        category_code: str,
        subtype_code: Optional[str] = None,
        manufacturer: Optional[str] = None,
        model: Optional[str] = None,
        serial_number: Optional[str] = None,
        asset_tag: Optional[str] = None,
        legacy_machine_id: Optional[str] = None,
        installation_date: Optional[Union[str, date]] = None,
        location: Optional[str] = None,
        criticality: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._require_dataset(dataset_id)
        fields = self._validate_asset_fields(
            {
                "asset_name": asset_name,
                "category_code": category_code,
                "subtype_code": subtype_code,
                "manufacturer": manufacturer,
                "model": model,
                "serial_number": serial_number,
                "asset_tag": asset_tag,
                "legacy_machine_id": legacy_machine_id,
                "installation_date": installation_date,
                "location": location,
                "criticality": criticality,
            }
        )
        equipment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction() as connection:
            self._check_duplicate_identity(connection, dataset_id, fields)
            connection.execute(
                """
                INSERT INTO equipment_assets (
                    dataset_id, equipment_id, asset_name, category_code, subtype_code,
                    manufacturer, model, serial_number, asset_tag, legacy_machine_id,
                    installation_date, location, criticality, taxonomy_version,
                    classification_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    equipment_id,
                    fields["asset_name"],
                    fields["category_code"],
                    fields["subtype_code"],
                    fields["manufacturer"],
                    fields["model"],
                    fields["serial_number"],
                    fields["asset_tag"],
                    fields["legacy_machine_id"],
                    fields["installation_date"],
                    fields["location"],
                    fields["criticality"],
                    fields["taxonomy_version"],
                    fields["classification_status"],
                    now,
                    now,
                ),
            )
        return self.get_equipment(dataset_id, equipment_id)

    def list_equipment(self, dataset_id: str) -> List[Dict[str, Any]]:
        self._require_dataset(dataset_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM equipment_assets
                WHERE dataset_id = ?
                ORDER BY lower(asset_name), equipment_id
                """,
                (dataset_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_equipment(self, dataset_id: str, equipment_id: str) -> Optional[Dict[str, Any]]:
        self._require_dataset(dataset_id)
        normalized_id = self._normalize_uuid(equipment_id, "Equipment ID")
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM equipment_assets
                WHERE dataset_id = ? AND equipment_id = ?
                """,
                (dataset_id, normalized_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def update_equipment(
        self,
        dataset_id: str,
        equipment_id: str,
        updates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        self._require_dataset(dataset_id)
        normalized_id = self._normalize_uuid(equipment_id, "Equipment ID")
        allowed = {
            "asset_name",
            "category_code",
            "subtype_code",
            "manufacturer",
            "model",
            "serial_number",
            "asset_tag",
            "legacy_machine_id",
            "installation_date",
            "location",
            "criticality",
        }
        if not isinstance(updates, Mapping) or not updates or set(updates) - allowed:
            raise EquipmentRegistryError("Equipment update contains no fields or unsupported fields.")

        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM equipment_assets WHERE dataset_id = ? AND equipment_id = ?",
                (dataset_id, normalized_id),
            ).fetchone()
            if existing is None:
                raise EquipmentRegistryError("Equipment record was not found in the active dataset.")
            merged = dict(existing)
            merged.update(updates)
            fields = self._validate_asset_fields(merged)
            self._check_duplicate_identity(connection, dataset_id, fields, normalized_id)

            if (
                fields["category_code"] != existing["category_code"]
                or fields["subtype_code"] != existing["subtype_code"]
            ):
                existing_components = connection.execute(
                    """
                    SELECT component_code FROM equipment_components
                    WHERE dataset_id = ? AND equipment_id = ? AND component_code IS NOT NULL
                    """,
                    (dataset_id, normalized_id),
                ).fetchall()
                supported = {
                    code for code, _ in self.component_options(fields["category_code"])
                }
                if any(row["component_code"] not in supported for row in existing_components):
                    raise EquipmentRegistryError(
                        "Equipment category cannot change while it has incompatible classified components."
                    )

            fields["updated_at"] = datetime.now(timezone.utc).isoformat()
            assignments = ", ".join(f"{name} = ?" for name in fields)
            connection.execute(
                f"UPDATE equipment_assets SET {assignments} "
                "WHERE dataset_id = ? AND equipment_id = ?",
                (*fields.values(), dataset_id, normalized_id),
            )
        return self.get_equipment(dataset_id, normalized_id)

    @staticmethod
    def category_options():
        from src.equipment_taxonomy import category_options

        return category_options()

    @staticmethod
    def subtype_options(category_code: str):
        from src.equipment_taxonomy import subtype_options

        return subtype_options(category_code)

    @staticmethod
    def component_options(category_code: str):
        from src.equipment_taxonomy import component_options

        return component_options(category_code)

    def delete_equipment(self, dataset_id: str, equipment_id: str) -> None:
        self._require_dataset(dataset_id)
        normalized_id = self._normalize_uuid(equipment_id, "Equipment ID")
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT equipment_id FROM equipment_assets
                WHERE dataset_id = ? AND equipment_id = ?
                """,
                (dataset_id, normalized_id),
            ).fetchone()
            if row is None:
                raise EquipmentRegistryError("Equipment record was not found in the active dataset.")
            dependent_components = connection.execute(
                """
                SELECT 1 FROM equipment_components
                WHERE dataset_id = ? AND equipment_id = ? LIMIT 1
                """,
                (dataset_id, normalized_id),
            ).fetchone()
            linked_incident = connection.execute(
                """
                SELECT 1 FROM incident_equipment_associations
                WHERE dataset_id = ? AND equipment_id = ? LIMIT 1
                """,
                (dataset_id, normalized_id),
            ).fetchone()
            if dependent_components or linked_incident:
                raise EquipmentRegistryError(
                    "Equipment cannot be deleted while components or incident associations exist."
                )
            connection.execute(
                "DELETE FROM equipment_assets WHERE dataset_id = ? AND equipment_id = ?",
                (dataset_id, normalized_id),
            )

    @classmethod
    def _validate_component_code(
        cls,
        category_code: str,
        component_code: Optional[str],
    ) -> tuple[Optional[str], str]:
        if component_code is None or not str(component_code).strip():
            return None, CLASSIFICATION_UNCLASSIFIED
        code = cls._required_text(component_code, "Component type", 80)
        supported = {item_code for item_code, _ in cls.component_options(category_code)}
        if code not in supported:
            raise EquipmentRegistryError(
                "Component type is not supported by the equipment's candidate taxonomy."
            )
        return code, CLASSIFICATION_CANDIDATE

    def _require_equipment(
        self,
        connection: sqlite3.Connection,
        dataset_id: str,
        equipment_id: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT equipment_id, category_code
            FROM equipment_assets WHERE dataset_id = ? AND equipment_id = ?
            """,
            (dataset_id, equipment_id),
        ).fetchone()
        if row is None:
            raise EquipmentRegistryError("Equipment record was not found in the active dataset.")
        return row

    @staticmethod
    def _validate_parent(
        connection: sqlite3.Connection,
        dataset_id: str,
        equipment_id: str,
        parent_component_id: Optional[str],
        component_id: Optional[str] = None,
    ) -> Optional[str]:
        if parent_component_id is None or parent_component_id == "":
            return None
        parent_id = EquipmentRegistry._normalize_uuid(parent_component_id, "Parent component ID")
        if component_id is not None and parent_id == component_id:
            raise EquipmentRegistryError("A component cannot be its own parent.")
        parent = connection.execute(
            """
            SELECT component_id FROM equipment_components
            WHERE dataset_id = ? AND equipment_id = ? AND component_id = ?
            """,
            (dataset_id, equipment_id, parent_id),
        ).fetchone()
        if parent is None:
            raise EquipmentRegistryError(
                "Parent component must belong to this equipment and dataset."
            )
        if component_id is not None:
            cycle = connection.execute(
                """
                WITH RECURSIVE subtree(component_id) AS (
                    SELECT component_id FROM equipment_components
                    WHERE dataset_id = ? AND equipment_id = ? AND component_id = ?
                    UNION ALL
                    SELECT child.component_id
                    FROM equipment_components AS child
                    JOIN subtree ON child.parent_component_id = subtree.component_id
                    WHERE child.dataset_id = ? AND child.equipment_id = ?
                )
                SELECT 1 FROM subtree WHERE component_id = ? LIMIT 1
                """,
                (
                    dataset_id,
                    equipment_id,
                    component_id,
                    dataset_id,
                    equipment_id,
                    parent_id,
                ),
            ).fetchone()
            if cycle:
                raise EquipmentRegistryError("Component hierarchy cannot contain cycles.")
        return parent_id

    def create_component(
        self,
        dataset_id: str,
        equipment_id: str,
        component_name: str,
        component_code: Optional[str] = None,
        parent_component_id: Optional[str] = None,
        manufacturer: Optional[str] = None,
        model: Optional[str] = None,
        serial_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._require_dataset(dataset_id)
        normalized_equipment_id = self._normalize_uuid(equipment_id, "Equipment ID")
        name = self._required_text(component_name, "Component name")
        component_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction() as connection:
            equipment = self._require_equipment(connection, dataset_id, normalized_equipment_id)
            normalized_code, status = self._validate_component_code(
                equipment["category_code"], component_code
            )
            parent_id = self._validate_parent(
                connection, dataset_id, normalized_equipment_id, parent_component_id
            )
            connection.execute(
                """
                INSERT INTO equipment_components (
                    dataset_id, component_id, equipment_id, parent_component_id,
                    component_name, component_code, manufacturer, model, serial_number,
                    taxonomy_version, classification_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    component_id,
                    normalized_equipment_id,
                    parent_id,
                    name,
                    normalized_code,
                    self._optional_text(manufacturer, "Component manufacturer", 120),
                    self._optional_text(model, "Component model", 120),
                    self._optional_text(serial_number, "Component serial number", 120),
                    TAXONOMY_VERSION,
                    status,
                    now,
                    now,
                ),
            )
        return self.get_component(dataset_id, normalized_equipment_id, component_id)

    def list_components(
        self,
        dataset_id: str,
        equipment_id: str,
    ) -> List[Dict[str, Any]]:
        self._require_dataset(dataset_id)
        normalized_id = self._normalize_uuid(equipment_id, "Equipment ID")
        with closing(self._connect()) as connection:
            self._require_equipment(connection, dataset_id, normalized_id)
            rows = connection.execute(
                """
                SELECT * FROM equipment_components
                WHERE dataset_id = ? AND equipment_id = ?
                ORDER BY coalesce(parent_component_id, ''), lower(component_name)
                """,
                (dataset_id, normalized_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_component(
        self,
        dataset_id: str,
        equipment_id: str,
        component_id: str,
    ) -> Optional[Dict[str, Any]]:
        self._require_dataset(dataset_id)
        normalized_equipment_id = self._normalize_uuid(equipment_id, "Equipment ID")
        normalized_component_id = self._normalize_uuid(component_id, "Component ID")
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM equipment_components
                WHERE dataset_id = ? AND equipment_id = ? AND component_id = ?
                """,
                (dataset_id, normalized_equipment_id, normalized_component_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def update_component(
        self,
        dataset_id: str,
        equipment_id: str,
        component_id: str,
        updates: Mapping[str, Any],
    ) -> Dict[str, Any]:
        self._require_dataset(dataset_id)
        normalized_equipment_id = self._normalize_uuid(equipment_id, "Equipment ID")
        normalized_component_id = self._normalize_uuid(component_id, "Component ID")
        allowed = {
            "component_name",
            "component_code",
            "parent_component_id",
            "manufacturer",
            "model",
            "serial_number",
        }
        if not isinstance(updates, Mapping) or not updates or set(updates) - allowed:
            raise EquipmentRegistryError("Component update contains no fields or unsupported fields.")
        with self._transaction() as connection:
            equipment = self._require_equipment(connection, dataset_id, normalized_equipment_id)
            current = connection.execute(
                """
                SELECT * FROM equipment_components
                WHERE dataset_id = ? AND equipment_id = ? AND component_id = ?
                """,
                (dataset_id, normalized_equipment_id, normalized_component_id),
            ).fetchone()
            if current is None:
                raise EquipmentRegistryError("Component was not found in the active dataset.")
            values = dict(current)
            values.update(updates)
            component_name = self._required_text(values["component_name"], "Component name")
            code, status = self._validate_component_code(
                equipment["category_code"], values.get("component_code")
            )
            parent_id = self._validate_parent(
                connection,
                dataset_id,
                normalized_equipment_id,
                values.get("parent_component_id"),
                normalized_component_id,
            )
            fields = {
                "component_name": component_name,
                "component_code": code,
                "parent_component_id": parent_id,
                "manufacturer": self._optional_text(
                    values.get("manufacturer"), "Component manufacturer", 120
                ),
                "model": self._optional_text(values.get("model"), "Component model", 120),
                "serial_number": self._optional_text(
                    values.get("serial_number"), "Component serial number", 120
                ),
                "taxonomy_version": TAXONOMY_VERSION,
                "classification_status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            assignments = ", ".join(f"{name} = ?" for name in fields)
            connection.execute(
                f"UPDATE equipment_components SET {assignments} "
                "WHERE dataset_id = ? AND equipment_id = ? AND component_id = ?",
                (
                    *fields.values(),
                    dataset_id,
                    normalized_equipment_id,
                    normalized_component_id,
                ),
            )
        return self.get_component(dataset_id, normalized_equipment_id, normalized_component_id)

    def delete_component(
        self,
        dataset_id: str,
        equipment_id: str,
        component_id: str,
    ) -> None:
        self._require_dataset(dataset_id)
        normalized_equipment_id = self._normalize_uuid(equipment_id, "Equipment ID")
        normalized_component_id = self._normalize_uuid(component_id, "Component ID")
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT component_id FROM equipment_components
                WHERE dataset_id = ? AND equipment_id = ? AND component_id = ?
                """,
                (dataset_id, normalized_equipment_id, normalized_component_id),
            ).fetchone()
            if row is None:
                raise EquipmentRegistryError("Component was not found in the active dataset.")
            child = connection.execute(
                """
                SELECT 1 FROM equipment_components
                WHERE dataset_id = ? AND equipment_id = ? AND parent_component_id = ?
                LIMIT 1
                """,
                (dataset_id, normalized_equipment_id, normalized_component_id),
            ).fetchone()
            if child:
                raise EquipmentRegistryError(
                    "A component cannot be deleted while child components exist."
                )
            connection.execute(
                """
                DELETE FROM equipment_components
                WHERE dataset_id = ? AND equipment_id = ? AND component_id = ?
                """,
                (dataset_id, normalized_equipment_id, normalized_component_id),
            )

    @staticmethod
    def _validate_incident_reference(
        dataset_id: str,
        incident_id: str,
        incidents: pd.DataFrame,
    ) -> str:
        normalized_incident_id = EquipmentRegistry._required_text(
            incident_id, "Incident ID", 256
        )
        if not isinstance(incidents, pd.DataFrame) or "incident_id" not in incidents.columns:
            raise EquipmentRegistryError("Incident records are unavailable for association validation.")

        if "dataset_id" in incidents.columns:
            scoped = incidents[
                incidents["dataset_id"].astype(str).str.strip() == dataset_id
            ]
        elif dataset_id == "DATASET-DEFAULT":
            scoped = incidents
        else:
            raise DatasetOwnershipError(
                "Legacy incidents without dataset ownership are limited to DATASET-DEFAULT."
            )
        scoped = scoped[scoped["incident_id"].notna()]
        matches = scoped[
            scoped["incident_id"].astype(str).str.strip() == normalized_incident_id
        ]
        if len(matches) != 1:
            raise EquipmentRegistryError(
                "Incident must exist exactly once in the active dataset."
            )
        return normalized_incident_id

    def create_incident_association(
        self,
        dataset_id: str,
        incident_id: str,
        equipment_id: str,
        incidents: pd.DataFrame,
        role: str = "RELATED",
        review_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._require_dataset(dataset_id)
        normalized_incident_id = self._validate_incident_reference(
            dataset_id, incident_id, incidents
        )
        normalized_equipment_id = self._normalize_uuid(equipment_id, "Equipment ID")
        role_value = self._required_text(role, "Association role", 20).upper()
        if role_value not in self.ASSOCIATION_ROLES:
            raise EquipmentRegistryError("Association role must be PRIMARY or RELATED.")
        notes = self._optional_text(review_notes, "Review notes", 1000)
        association_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction() as connection:
            self._require_equipment(connection, dataset_id, normalized_equipment_id)
            connection.execute(
                """
                INSERT INTO incident_equipment_associations (
                    association_id, dataset_id, incident_id, equipment_id,
                    role, link_source, review_status, review_notes,
                    reviewed_by, reviewed_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'USER_SELECTED', 'PENDING_REVIEW', ?, NULL, NULL, ?, ?)
                """,
                (
                    association_id,
                    dataset_id,
                    normalized_incident_id,
                    normalized_equipment_id,
                    role_value,
                    notes,
                    now,
                    now,
                ),
            )
        return self.get_incident_association(dataset_id, association_id)

    def list_incident_associations(
        self,
        dataset_id: str,
        incident_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self._require_dataset(dataset_id)
        query = """
            SELECT a.*, e.asset_name, e.category_code, e.subtype_code,
                   e.classification_status AS equipment_classification_status
            FROM incident_equipment_associations AS a
            JOIN equipment_assets AS e
              ON e.dataset_id = a.dataset_id AND e.equipment_id = a.equipment_id
            WHERE a.dataset_id = ?
        """
        parameters: List[Any] = [dataset_id]
        if incident_id is not None:
            normalized_incident_id = self._required_text(incident_id, "Incident ID", 256)
            query += " AND a.incident_id = ?"
            parameters.append(normalized_incident_id)
        query += " ORDER BY a.incident_id, a.created_at, a.association_id"
        with closing(self._connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def get_incident_association(
        self,
        dataset_id: str,
        association_id: str,
    ) -> Optional[Dict[str, Any]]:
        self._require_dataset(dataset_id)
        normalized_id = self._normalize_uuid(association_id, "Association ID")
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT a.*, e.asset_name, e.category_code, e.subtype_code,
                       e.classification_status AS equipment_classification_status
                FROM incident_equipment_associations AS a
                JOIN equipment_assets AS e
                  ON e.dataset_id = a.dataset_id AND e.equipment_id = a.equipment_id
                WHERE a.dataset_id = ? AND a.association_id = ?
                """,
                (dataset_id, normalized_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def review_incident_association(
        self,
        dataset_id: str,
        association_id: str,
        decision: str,
        reviewer: str,
        review_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._require_dataset(dataset_id)
        normalized_id = self._normalize_uuid(association_id, "Association ID")
        decision_value = self._required_text(decision, "Review decision", 30).upper()
        if decision_value not in {"REVIEWED", "REJECTED"}:
            raise EquipmentRegistryError("Decision must be REVIEWED or REJECTED.")
        reviewer_name = self._required_text(reviewer, "Reviewer", 120)
        notes = self._optional_text(review_notes, "Review notes", 1000)
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT review_status FROM incident_equipment_associations
                WHERE dataset_id = ? AND association_id = ?
                """,
                (dataset_id, normalized_id),
            ).fetchone()
            if row is None:
                raise EquipmentRegistryError(
                    "Incident-equipment association was not found in the active dataset."
                )
            if row["review_status"] != "PENDING_REVIEW":
                raise EquipmentRegistryError(
                    "Only pending incident-equipment associations can be reviewed."
                )
            connection.execute(
                """
                UPDATE incident_equipment_associations
                SET review_status = ?, review_notes = ?, reviewed_by = ?,
                    reviewed_at = ?, updated_at = ?
                WHERE dataset_id = ? AND association_id = ?
                """,
                (
                    decision_value,
                    notes,
                    reviewer_name,
                    now,
                    now,
                    dataset_id,
                    normalized_id,
                ),
            )
        return self.get_incident_association(dataset_id, normalized_id)

    def diagnose_missing_incidents(
        self,
        dataset_id: str,
        incidents: pd.DataFrame,
    ) -> List[Dict[str, str]]:
        """Report active-dataset associations whose legacy incident no longer exists."""
        self._require_dataset(dataset_id)
        if "incident_id" not in incidents.columns:
            known_incidents = set()
        elif "dataset_id" in incidents.columns:
            known_incidents = set(
                incidents.loc[
                    (incidents["dataset_id"].astype(str).str.strip() == dataset_id)
                    & incidents["incident_id"].notna(),
                    "incident_id",
                ].astype(str).str.strip()
            )
        elif dataset_id == "DATASET-DEFAULT":
            known_incidents = set(
                incidents.loc[incidents["incident_id"].notna(), "incident_id"]
                .astype(str)
                .str.strip()
            )
        else:
            known_incidents = set()

        return [
            {
                "code": "missing_incident",
                "association_id": row["association_id"],
                "incident_id": row["incident_id"],
                "message": "Association references an incident not present in the active dataset.",
            }
            for row in self.list_incident_associations(dataset_id)
            if row["incident_id"] not in known_incidents
        ]

    def diagnose_unregistered_dataset_references(self) -> List[Dict[str, Any]]:
        """Return aggregate diagnostics only for equipment rows referencing removed/unknown datasets."""
        try:
            registered_ids = {dataset.dataset_id for dataset in self.store.list_datasets()}
        except EvidenceStoreError:
            raise EquipmentRegistryError(
                "The registered dataset list is unavailable for equipment diagnostics."
            ) from None
        findings: List[Dict[str, Any]] = []
        with closing(self._connect()) as connection:
            for table, entity_type in (
                ("equipment_assets", "equipment"),
                ("equipment_components", "component"),
                ("incident_equipment_associations", "incident_association"),
            ):
                rows = connection.execute(
                    f"SELECT dataset_id, COUNT(*) AS record_count FROM {table} "
                    "GROUP BY dataset_id"
                ).fetchall()
                for row in rows:
                    if row["dataset_id"] not in registered_ids:
                        findings.append(
                            {
                                "code": "unregistered_dataset",
                                "dataset_id": row["dataset_id"],
                                "entity_type": entity_type,
                                "record_count": int(row["record_count"]),
                            }
                        )
        return findings

    def foreign_key_check(self) -> List[Dict[str, Any]]:
        """Return SQLite's foreign-key diagnostics for isolated validation and maintenance."""
        with closing(self._connect()) as connection:
            rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        return [dict(row) for row in rows]

    def integrity_check(self) -> str:
        """Return the SQLite integrity-check result."""
        with closing(self._connect()) as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        return str(row[0])


"""Persistent, dataset-scoped investigation history and activity storage."""

import json
import math
import sqlite3
import uuid
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from src.models import (
    CaseHypothesisOutcome,
    CaseInspection,
    CasePriority,
    CaseStatus,
    DatasetOwnershipError,
    FaultAnalysisResult,
    HypothesisDetermination,
    InspectionType,
    InvestigationCase,
    OutcomeConfidence,
)
from src.storage import EvidenceStore


class HistoryStoreError(RuntimeError):
    """Raised when investigation history cannot be safely read or written."""


class CaseAlreadyExistsError(HistoryStoreError):
    """Raised when attempting to create a duplicate case for the same run ID."""


class CaseTransitionError(HistoryStoreError):
    """Raised when attempting an illegal case lifecycle transition."""


class InvestigationHistoryStore:
    """Store immutable analysis snapshots and follow-up case tracking separately from the evidence search index."""

    SCHEMA_VERSION = 2
    REQUIRED_V1_TABLES = {"investigation_runs", "activity_events"}
    REQUIRED_V2_TABLES = {
        "investigation_runs",
        "activity_events",
        "investigation_cases",
        "case_inspections",
        "case_hypothesis_outcomes",
        "case_status_history",
    }
    ALLOWED_CASE_TRANSITIONS = {
        CaseStatus.OPEN: {CaseStatus.IN_REVIEW},
        CaseStatus.IN_REVIEW: {CaseStatus.ACTION_TAKEN, CaseStatus.RESOLVED},
        CaseStatus.ACTION_TAKEN: {CaseStatus.RESOLVED},
        CaseStatus.RESOLVED: {CaseStatus.CLOSED, CaseStatus.OPEN},
        CaseStatus.CLOSED: {CaseStatus.OPEN},
    }

    def __init__(self, db_path: Optional[Path] = None, allow_migration: bool = False):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.db_path = (
                project_root / "data" / "history" / "investigation_history.db"
            )
        else:
            self.db_path = Path(db_path).resolve()
        self.allow_migration = allow_migration
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def backup_database(self) -> Path:
        """Create and verify a timestamped SQLite-consistent backup."""
        if not self.db_path.exists():
            raise HistoryStoreError(f"Cannot back up non-existent database: {self.db_path}")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        backup_path = None
        for suffix in range(1000):
            candidate_suffix = "" if suffix == 0 else f".{suffix}"
            candidate = self.db_path.with_name(
                f"{self.db_path.name}.bak.{timestamp}{candidate_suffix}"
            )
            try:
                with candidate.open("xb"):
                    pass
            except FileExistsError:
                continue
            backup_path = candidate
            break
        if backup_path is None:
            raise HistoryStoreError(
                f"Could not reserve a unique backup path beside {self.db_path}."
            )

        try:
            source_uri = self.db_path.as_uri() + "?mode=ro"
            with closing(sqlite3.connect(source_uri, uri=True)) as source:
                with closing(sqlite3.connect(str(backup_path))) as destination:
                    source.backup(destination)
                    destination.commit()

            with closing(sqlite3.connect(backup_path.as_uri() + "?mode=ro", uri=True)) as verification:
                integrity = verification.execute("PRAGMA integrity_check").fetchone()[0]
                foreign_key_issues = verification.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()
            if integrity != "ok" or foreign_key_issues:
                raise HistoryStoreError(
                    "SQLite backup verification failed: "
                    f"integrity_check={integrity!r}, "
                    f"foreign_key_violations={len(foreign_key_issues)}."
                )
        except Exception:
            if backup_path.exists():
                backup_path.unlink()
            raise

        return backup_path

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if version > self.SCHEMA_VERSION:
                raise HistoryStoreError(
                    f"History schema version {version} is newer than supported "
                    f"version {self.SCHEMA_VERSION}."
                )
            if version == 0 and tables:
                raise HistoryStoreError(
                    "History database contains unversioned tables; refusing to "
                    "initialize over an unknown schema."
                )
            if version == self.SCHEMA_VERSION:
                if not self.REQUIRED_V2_TABLES.issubset(tables):
                    raise HistoryStoreError(
                        "History database schema is marked current but is incomplete."
                    )
                return

            if version == 1:
                if not self.allow_migration:
                    # In safe / read-only mode, do not auto-migrate existing v1 db
                    return
                # Migrate from v1 to v2
                self._migrate_v1_to_v2(connection)
                return

            # Fresh database initialization (version == 0)
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._create_v1_tables(connection)
                self._create_v2_tables(connection)
                connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _create_v1_tables(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE investigation_runs (
                run_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                incident_id TEXT NOT NULL,
                analyzed_at TEXT NOT NULL,
                status TEXT NOT NULL,
                incident_snapshot_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                serialization_version INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX investigation_runs_dataset_time
            ON investigation_runs(dataset_id, analyzed_at DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE activity_events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT,
                dataset_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES investigation_runs(run_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX activity_events_dataset_time
            ON activity_events(dataset_id, occurred_at DESC)
            """
        )

    def _create_v2_tables(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE investigation_cases (
                case_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                incident_id TEXT NOT NULL,
                run_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                title TEXT NOT NULL,
                priority TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                closed_at TEXT,
                closure_summary TEXT,
                FOREIGN KEY(run_id) REFERENCES investigation_runs(run_id)
                    ON DELETE RESTRICT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_cases_dataset_status
            ON investigation_cases(dataset_id, status)
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_cases_dataset_incident
            ON investigation_cases(dataset_id, incident_id)
            """
        )

        connection.execute(
            """
            CREATE TABLE case_inspections (
                inspection_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                inspected_at TEXT NOT NULL,
                inspector_name TEXT NOT NULL,
                inspection_type TEXT NOT NULL,
                observations TEXT NOT NULL,
                physical_action_taken TEXT NOT NULL,
                parts_replaced_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES investigation_cases(case_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_inspections_case_time
            ON case_inspections(case_id, inspected_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_inspections_dataset
            ON case_inspections(dataset_id)
            """
        )

        connection.execute(
            """
            CREATE TABLE case_hypothesis_outcomes (
                outcome_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                failure_mode TEXT NOT NULL,
                determination TEXT NOT NULL,
                confidence TEXT NOT NULL,
                notes TEXT NOT NULL,
                determined_at TEXT NOT NULL,
                determined_by TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES investigation_cases(case_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_outcomes_case
            ON case_hypothesis_outcomes(case_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_outcomes_dataset
            ON case_hypothesis_outcomes(dataset_id)
            """
        )

        connection.execute(
            """
            CREATE TABLE case_status_history (
                transition_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                from_status TEXT NOT NULL,
                to_status TEXT NOT NULL,
                reason TEXT NOT NULL,
                changed_by TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES investigation_cases(case_id)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_case_status_history_case
            ON case_status_history(case_id, changed_at DESC)
            """
        )

    def _migrate_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        """Migrate a version 1 database to version 2 non-destructively."""
        connection.execute("BEGIN IMMEDIATE")
        try:
            # Rebuild activity_events to ensure run_id does not have a UNIQUE constraint
            connection.execute(
                """
                CREATE TABLE activity_events_v2_migration (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    dataset_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES investigation_runs(run_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                INSERT INTO activity_events_v2_migration (
                    event_id, run_id, dataset_id, event_type, occurred_at,
                    entity_type, entity_id, summary, details_json
                )
                SELECT
                    event_id, run_id, dataset_id, event_type, occurred_at,
                    entity_type, entity_id, summary, details_json
                FROM activity_events
                """
            )
            connection.execute("DROP TABLE activity_events")
            connection.execute(
                "ALTER TABLE activity_events_v2_migration RENAME TO activity_events"
            )
            connection.execute(
                """
                CREATE INDEX activity_events_dataset_time
                ON activity_events(dataset_id, occurred_at DESC)
                """
            )

            self._create_v2_tables(connection)
            connection.execute("PRAGMA user_version = 2")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {
                str(key): InvestigationHistoryStore._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [InvestigationHistoryStore._json_safe(item) for item in value]
        if type(value).__name__ == "NAType":
            return None
        if hasattr(value, "item"):
            item = value.item() if hasattr(value, "item") else value
            if item is value:
                raise HistoryStoreError(
                    f"Unsupported scalar value in investigation snapshot: "
                    f"{type(value).__name__}."
                )
            return InvestigationHistoryStore._json_safe(item)
        if hasattr(value, "value") and isinstance(value.value, (str, int, float)):
            return InvestigationHistoryStore._json_safe(value.value)
        raise HistoryStoreError(
            f"Unsupported value in investigation snapshot: {type(value).__name__}."
        )

    @staticmethod
    def _require_scope(store: EvidenceStore, dataset_id: str) -> None:
        store.require_dataset_id(dataset_id, "Investigation history", dataset_id)

    def save_run(
        self,
        store: EvidenceStore,
        run_id: str,
        dataset_id: str,
        incident_snapshot: Mapping[str, Any],
        result: Union[FaultAnalysisResult, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Insert one immutable analysis snapshot and its activity event idempotently."""
        self._require_scope(store, dataset_id)
        try:
            normalized_run_id = str(uuid.UUID(run_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise HistoryStoreError("Investigation run ID must be a UUID.") from exc

        snapshot = self._json_safe(incident_snapshot)
        result_payload = result.to_dict() if isinstance(result, FaultAnalysisResult) else result
        result_data = self._json_safe(result_payload)
        if not isinstance(snapshot, dict) or not isinstance(result_data, dict):
            raise HistoryStoreError("Incident snapshot and analysis result must be objects.")

        incident_id = str(snapshot.get("incident_id") or "")
        if not incident_id or incident_id != str(result_data.get("incident_id") or ""):
            raise HistoryStoreError(
                "Incident snapshot and analysis result must identify the same incident."
            )
        if snapshot.get("dataset_id") != dataset_id:
            raise DatasetOwnershipError(
                f"Incident '{incident_id}' does not belong to history dataset "
                f"'{dataset_id}'."
            )
        incident_facts = result_data.get("incident_facts")
        if not isinstance(incident_facts, dict) or incident_facts.get("dataset_id") != dataset_id:
            raise DatasetOwnershipError(
                f"Analysis result for incident '{incident_id}' does not carry "
                f"dataset ownership '{dataset_id}'."
            )
        if incident_facts != snapshot:
            raise HistoryStoreError(
                "Saved incident snapshot does not match the facts analyzed."
            )
        self._validate_result_evidence_scope(result_data, dataset_id, incident_id)

        analyzed_at = result_data.get("analyzed_at")
        status = result_data.get("status")
        if not isinstance(analyzed_at, str) or not analyzed_at:
            raise HistoryStoreError("Analysis result is missing its timestamp.")
        if not isinstance(status, str) or not status:
            raise HistoryStoreError("Analysis result is missing its status.")

        snapshot_json = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
        result_json = json.dumps(result_data, sort_keys=True, ensure_ascii=False)
        event_id = f"investigation:{normalized_run_id}"
        summary = f"Investigation {status.replace('_', ' ')} for incident {incident_id}."
        event_details = json.dumps(
            {"run_id": normalized_run_id, "status": status},
            sort_keys=True,
        )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT dataset_id, incident_id, analyzed_at, status, "
                "incident_snapshot_json, result_json, serialization_version "
                "FROM investigation_runs WHERE run_id = ?",
                (normalized_run_id,),
            ).fetchone()
            if existing is not None:
                expected = (
                    dataset_id,
                    incident_id,
                    analyzed_at,
                    status,
                    snapshot_json,
                    result_json,
                    1,
                )
                actual = tuple(existing)
                if actual != expected:
                    raise HistoryStoreError(
                        "Investigation run ID already exists with different content."
                    )
                event = connection.execute(
                    "SELECT event_id, dataset_id, event_type, occurred_at, entity_type, "
                    "entity_id, summary, details_json FROM activity_events WHERE run_id = ?",
                    (normalized_run_id,),
                ).fetchone()
                expected_event = (
                    event_id,
                    dataset_id,
                    "investigation_run_saved",
                    analyzed_at,
                    "investigation_run",
                    normalized_run_id,
                    summary,
                    event_details,
                )
                if event is not None and tuple(event) != expected_event:
                    raise HistoryStoreError(
                        "Investigation activity event conflicts with its saved run."
                    )
                if event is None:
                    self._insert_activity_event(
                        connection,
                        event_id,
                        normalized_run_id,
                        dataset_id,
                        analyzed_at,
                        incident_id,
                        summary,
                        event_details,
                    )
                return {"run_id": normalized_run_id, "created": False}

            connection.execute(
                """
                INSERT INTO investigation_runs (
                    run_id, dataset_id, incident_id, analyzed_at, status,
                    incident_snapshot_json, result_json, serialization_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_run_id,
                    dataset_id,
                    incident_id,
                    analyzed_at,
                    status,
                    snapshot_json,
                    result_json,
                    1,
                ),
            )
            self._insert_activity_event(
                connection,
                event_id,
                normalized_run_id,
                dataset_id,
                analyzed_at,
                incident_id,
                summary,
                event_details,
            )
            return {"run_id": normalized_run_id, "created": True}

    @staticmethod
    def _validate_result_evidence_scope(
        result: Dict[str, Any],
        dataset_id: str,
        incident_id: str,
    ) -> None:
        def require_reference_owner(reference: Any) -> None:
            if not isinstance(reference, dict) or reference.get("dataset_id") != dataset_id:
                raise DatasetOwnershipError(
                    f"Investigation evidence for incident '{incident_id}' does not "
                    f"belong to dataset '{dataset_id}'."
                )

        retrieved = result.get("retrieved_evidence", [])
        if not isinstance(retrieved, list):
            raise HistoryStoreError("Retrieved evidence in the analysis result is invalid.")
        for item in retrieved:
            chunk = item.get("chunk") if isinstance(item, dict) else None
            require_reference_owner(chunk)

        hypotheses = result.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            raise HistoryStoreError("Hypotheses in the analysis result are invalid.")
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                raise HistoryStoreError("Hypothesis in the analysis result is invalid.")
            for key in ("supporting_evidence", "contradictory_evidence"):
                references = hypothesis.get(key, [])
                if not isinstance(references, list):
                    raise HistoryStoreError(
                        f"{key} in the analysis result is invalid."
                    )
                for reference in references:
                    require_reference_owner(reference)

        excluded = result.get("excluded_evidence", [])
        if not isinstance(excluded, list):
            raise HistoryStoreError("Excluded evidence in the analysis result is invalid.")
        for reference in excluded:
            require_reference_owner(reference)

    @staticmethod
    def _insert_activity_event(
        connection: sqlite3.Connection,
        event_id: str,
        run_id: str,
        dataset_id: str,
        occurred_at: str,
        incident_id: str,
        summary: str,
        details_json: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO activity_events (
                event_id, run_id, dataset_id, event_type, occurred_at,
                entity_type, entity_id, summary, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                run_id,
                dataset_id,
                "investigation_run_saved",
                occurred_at,
                "investigation_run",
                run_id,
                summary,
                details_json,
            ),
        )

    def list_runs(
        self,
        store: EvidenceStore,
        dataset_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List run summaries for exactly one currently registered dataset."""
        self._require_scope(store, dataset_id)
        if limit < 1:
            raise ValueError("History result limit must be positive.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, dataset_id, incident_id, analyzed_at, status
                FROM investigation_runs
                WHERE dataset_id = ?
                ORDER BY analyzed_at DESC, run_id DESC
                LIMIT ?
                """,
                (dataset_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(
        self,
        store: EvidenceStore,
        dataset_id: str,
        run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load a saved snapshot only when its dataset matches the requested scope."""
        self._require_scope(store, dataset_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, dataset_id, incident_id, analyzed_at, status,
                       incident_snapshot_json, result_json, serialization_version
                FROM investigation_runs
                WHERE run_id = ? AND dataset_id = ?
                """,
                (run_id, dataset_id),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["incident_snapshot"] = json.loads(record.pop("incident_snapshot_json"))
        record["result"] = json.loads(record.pop("result_json"))
        snapshot = record["incident_snapshot"]
        result = record["result"]
        if record["serialization_version"] != 1:
            raise HistoryStoreError(
                "Investigation history record uses an unsupported serialization version."
            )
        if (
            not isinstance(snapshot, dict)
            or not isinstance(result, dict)
            or snapshot.get("dataset_id") != dataset_id
            or not isinstance(result.get("incident_facts"), dict)
            or result["incident_facts"].get("dataset_id") != dataset_id
        ):
            raise DatasetOwnershipError(
                "Saved investigation snapshot does not match the requested dataset."
            )
        if (
            snapshot.get("incident_id") != record["incident_id"]
            or result.get("incident_id") != record["incident_id"]
            or result.get("incident_facts") != snapshot
            or result.get("status") != record["status"]
            or result.get("analyzed_at") != record["analyzed_at"]
        ):
            raise HistoryStoreError(
                "Saved investigation snapshot is inconsistent with its history record."
            )
        self._validate_result_evidence_scope(result, dataset_id, record["incident_id"])
        return record

    def list_activity(
        self,
        store: EvidenceStore,
        dataset_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List activity records for exactly one currently registered dataset."""
        self._require_scope(store, dataset_id)
        if limit < 1:
            raise ValueError("Activity result limit must be positive.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, dataset_id, event_type, occurred_at,
                       entity_type, entity_id, summary, details_json
                FROM activity_events
                WHERE dataset_id = ?
                ORDER BY occurred_at DESC, event_id DESC
                LIMIT ?
                """,
                (dataset_id, limit),
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["details"] = json.loads(event.pop("details_json"))
            events.append(event)
        return events

    # =========================================================================
    # Milestone 8: Case Management, Follow-up Inspections & Outcome Tracking
    # =========================================================================

    def create_case(
        self,
        store: EvidenceStore,
        dataset_id: str,
        run_id: str,
        title: str,
        priority: Union[CasePriority, str] = CasePriority.MEDIUM,
        created_by: str = "Reliability Engineer",
    ) -> Dict[str, Any]:
        """Initialize an investigation follow-up case from a saved run."""
        self._require_scope(store, dataset_id)
        if not title or not str(title).strip():
            raise HistoryStoreError("Case title cannot be empty.")
        clean_title = str(title).strip()

        # Validate that the run exists and belongs to the active dataset
        run_record = self.get_run(store, dataset_id, run_id)
        if run_record is None:
            raise HistoryStoreError(
                f"Cannot create case: investigation run '{run_id}' not found in dataset '{dataset_id}'."
            )

        incident_id = run_record["incident_id"]
        now_utc = datetime.now(timezone.utc).isoformat()
        case_id = str(uuid.uuid4())
        priority_val = priority.value if isinstance(priority, CasePriority) else str(priority)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # Duplicate prevention check
            existing = connection.execute(
                "SELECT case_id, dataset_id FROM investigation_cases WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                raise CaseAlreadyExistsError(
                    f"A follow-up case '{existing['case_id']}' already exists for run '{run_id}'."
                )

            connection.execute(
                """
                INSERT INTO investigation_cases (
                    case_id, dataset_id, incident_id, run_id, status, title,
                    priority, created_at, updated_at, closed_at, closure_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    case_id,
                    dataset_id,
                    incident_id,
                    run_id,
                    CaseStatus.OPEN.value,
                    clean_title,
                    priority_val,
                    now_utc,
                    now_utc,
                ),
            )

            # Record initial state transition in case_status_history
            transition_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO case_status_history (
                    transition_id, case_id, dataset_id, from_status, to_status,
                    reason, changed_by, changed_at
                ) VALUES (?, ?, ?, '', ?, ?, ?, ?)
                """,
                (
                    transition_id,
                    case_id,
                    dataset_id,
                    CaseStatus.OPEN.value,
                    "Case initialized from diagnostic investigation run",
                    str(created_by).strip() or "Reliability Engineer",
                    now_utc,
                ),
            )

            # Append to project/dataset activity timeline
            event_id = f"case_created:{case_id}"
            connection.execute(
                """
                INSERT INTO activity_events (
                    event_id, run_id, dataset_id, event_type, occurred_at,
                    entity_type, entity_id, summary, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    run_id,
                    dataset_id,
                    "case_created",
                    now_utc,
                    "investigation_case",
                    case_id,
                    f"Follow-up case opened for incident {incident_id}: '{clean_title}'.",
                    json.dumps(
                        {"case_id": case_id, "run_id": run_id, "incident_id": incident_id, "status": CaseStatus.OPEN.value},
                        sort_keys=True,
                    ),
                ),
            )
            connection.commit()

        return self.get_case(store, dataset_id, case_id)

    def get_case(
        self,
        store: EvidenceStore,
        dataset_id: str,
        case_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load a case along with its inspections, outcomes, and audit history."""
        self._require_scope(store, dataset_id)
        with self._connect() as connection:
            case_row = connection.execute(
                """
                SELECT case_id, dataset_id, incident_id, run_id, status,
                       title, priority, created_at, updated_at, closed_at, closure_summary
                FROM investigation_cases
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()
            if case_row is None:
                return None

            case_data = dict(case_row)
            if case_data["dataset_id"] != dataset_id:
                raise DatasetOwnershipError(
                    f"Case '{case_id}' does not belong to active dataset '{dataset_id}'."
                )

            # Cross-verify that the grounding run also belongs to the requested dataset
            run_row = connection.execute(
                "SELECT dataset_id FROM investigation_runs WHERE run_id = ?",
                (case_data["run_id"],),
            ).fetchone()
            if run_row is None or run_row["dataset_id"] != dataset_id:
                raise DatasetOwnershipError(
                    f"Grounding investigation run for case '{case_id}' does not belong to dataset '{dataset_id}'."
                )

            # Load inspections
            insp_rows = connection.execute(
                """
                SELECT inspection_id, case_id, dataset_id, inspected_at,
                       inspector_name, inspection_type, observations,
                       physical_action_taken, parts_replaced_json, created_at
                FROM case_inspections
                WHERE case_id = ? AND dataset_id = ?
                ORDER BY inspected_at ASC, created_at ASC
                """,
                (case_id, dataset_id),
            ).fetchall()
            inspections = []
            for r in insp_rows:
                d = dict(r)
                d["parts_replaced"] = json.loads(d.pop("parts_replaced_json") or "[]")
                inspections.append(d)

            # Load hypothesis outcomes
            out_rows = connection.execute(
                """
                SELECT outcome_id, case_id, dataset_id, hypothesis_id,
                       failure_mode, determination, confidence, notes,
                       determined_at, determined_by
                FROM case_hypothesis_outcomes
                WHERE case_id = ? AND dataset_id = ?
                ORDER BY determined_at ASC
                """,
                (case_id, dataset_id),
            ).fetchall()
            outcomes = [dict(r) for r in out_rows]

            # Load status history
            hist_rows = connection.execute(
                """
                SELECT transition_id, case_id, dataset_id, from_status,
                       to_status, reason, changed_by, changed_at
                FROM case_status_history
                WHERE case_id = ? AND dataset_id = ?
                ORDER BY changed_at ASC
                """,
                (case_id, dataset_id),
            ).fetchall()
            status_history = [dict(r) for r in hist_rows]

        case_data["inspections"] = inspections
        case_data["hypothesis_outcomes"] = outcomes
        case_data["status_history"] = status_history
        return case_data

    def get_case_by_run_id(
        self,
        store: EvidenceStore,
        dataset_id: str,
        run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve case linked to a specific run ID within the active dataset."""
        self._require_scope(store, dataset_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT case_id FROM investigation_cases WHERE run_id = ? AND dataset_id = ?",
                (run_id, dataset_id),
            ).fetchone()
            if row is None:
                return None
            case_id = row["case_id"]
        return self.get_case(store, dataset_id, case_id)

    def list_cases(
        self,
        store: EvidenceStore,
        dataset_id: str,
        status: Optional[Union[CaseStatus, str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List cases scoped to the active dataset."""
        self._require_scope(store, dataset_id)
        if limit < 1:
            raise ValueError("Limit must be positive.")

        query = """
            SELECT case_id, dataset_id, incident_id, run_id, status,
                   title, priority, created_at, updated_at, closed_at, closure_summary
            FROM investigation_cases
            WHERE dataset_id = ?
        """
        params: List[Any] = [dataset_id]

        if status:
            status_val = status.value if isinstance(status, CaseStatus) else str(status)
            query += " AND status = ?"
            params.append(status_val)

        query += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        return [dict(r) for r in rows]

    def log_inspection(
        self,
        store: EvidenceStore,
        dataset_id: str,
        case_id: str,
        inspector_name: str,
        inspection_type: Union[InspectionType, str],
        observations: str,
        physical_action_taken: str,
        parts_replaced: Optional[List[str]] = None,
        inspected_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a physical shop-floor maintenance inspection to a case."""
        self._require_scope(store, dataset_id)
        if not inspector_name or not str(inspector_name).strip():
            raise HistoryStoreError("Inspector name cannot be empty.")
        if not observations or not str(observations).strip():
            raise HistoryStoreError("Observations cannot be empty.")

        insp_type_val = (
            inspection_type.value
            if isinstance(inspection_type, InspectionType)
            else str(inspection_type)
        )
        now_utc = datetime.now(timezone.utc).isoformat()
        insp_time = inspected_at or now_utc
        inspection_id = str(uuid.uuid4())
        parts_list = list(parts_replaced or [])
        parts_json = json.dumps(parts_list, ensure_ascii=False)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            case_row = connection.execute(
                "SELECT case_id, dataset_id, incident_id, run_id, status FROM investigation_cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if case_row is None:
                raise HistoryStoreError(f"Case '{case_id}' does not exist.")
            if case_row["dataset_id"] != dataset_id:
                raise DatasetOwnershipError(
                    f"Case '{case_id}' does not belong to active dataset '{dataset_id}'."
                )

            connection.execute(
                """
                INSERT INTO case_inspections (
                    inspection_id, case_id, dataset_id, inspected_at,
                    inspector_name, inspection_type, observations,
                    physical_action_taken, parts_replaced_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inspection_id,
                    case_id,
                    dataset_id,
                    insp_time,
                    str(inspector_name).strip(),
                    insp_type_val,
                    str(observations).strip(),
                    str(physical_action_taken or "").strip(),
                    parts_json,
                    now_utc,
                ),
            )

            # Update case timestamp
            connection.execute(
                "UPDATE investigation_cases SET updated_at = ? WHERE case_id = ?",
                (now_utc, case_id),
            )

            # Log to activity events
            event_id = f"inspection:{inspection_id}"
            connection.execute(
                """
                INSERT INTO activity_events (
                    event_id, run_id, dataset_id, event_type, occurred_at,
                    entity_type, entity_id, summary, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    case_row["run_id"],
                    dataset_id,
                    "case_inspection_logged",
                    now_utc,
                    "case_inspection",
                    inspection_id,
                    f"Inspection logged for case {case_id} ({insp_type_val}) by {inspector_name.strip()}.",
                    json.dumps(
                        {"case_id": case_id, "inspection_id": inspection_id, "type": insp_type_val},
                        sort_keys=True,
                    ),
                ),
            )
            connection.commit()

        return {
            "inspection_id": inspection_id,
            "case_id": case_id,
            "dataset_id": dataset_id,
            "inspected_at": insp_time,
            "inspector_name": inspector_name.strip(),
            "inspection_type": insp_type_val,
            "observations": observations.strip(),
            "physical_action_taken": physical_action_taken.strip(),
            "parts_replaced": parts_list,
            "created_at": now_utc,
        }

    def record_hypothesis_outcome(
        self,
        store: EvidenceStore,
        dataset_id: str,
        case_id: str,
        hypothesis_id: str,
        failure_mode: str,
        determination: Union[HypothesisDetermination, str],
        confidence: Union[OutcomeConfidence, str] = OutcomeConfidence.MEDIUM,
        notes: str = "",
        determined_by: str = "Reliability Engineer",
    ) -> Dict[str, Any]:
        """Record or update physical ground-truth evaluation for an investigation hypothesis."""
        self._require_scope(store, dataset_id)
        if not hypothesis_id or not str(hypothesis_id).strip():
            raise HistoryStoreError("Hypothesis ID cannot be empty.")

        det_val = (
            determination.value
            if isinstance(determination, HypothesisDetermination)
            else str(determination)
        )
        conf_val = (
            confidence.value
            if isinstance(confidence, OutcomeConfidence)
            else str(confidence)
        )
        now_utc = datetime.now(timezone.utc).isoformat()
        outcome_id = str(uuid.uuid4())

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            case_row = connection.execute(
                "SELECT case_id, dataset_id, run_id, status FROM investigation_cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if case_row is None:
                raise HistoryStoreError(f"Case '{case_id}' does not exist.")
            if case_row["dataset_id"] != dataset_id:
                raise DatasetOwnershipError(
                    f"Case '{case_id}' does not belong to active dataset '{dataset_id}'."
                )

            # Insert new outcome record
            connection.execute(
                """
                INSERT INTO case_hypothesis_outcomes (
                    outcome_id, case_id, dataset_id, hypothesis_id,
                    failure_mode, determination, confidence, notes,
                    determined_at, determined_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome_id,
                    case_id,
                    dataset_id,
                    str(hypothesis_id).strip(),
                    str(failure_mode or "").strip(),
                    det_val,
                    conf_val,
                    str(notes or "").strip(),
                    now_utc,
                    str(determined_by).strip() or "Reliability Engineer",
                ),
            )

            # Update case timestamp
            connection.execute(
                "UPDATE investigation_cases SET updated_at = ? WHERE case_id = ?",
                (now_utc, case_id),
            )

            # Append to activity events
            event_id = f"outcome:{outcome_id}"
            connection.execute(
                """
                INSERT INTO activity_events (
                    event_id, run_id, dataset_id, event_type, occurred_at,
                    entity_type, entity_id, summary, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    case_row["run_id"],
                    dataset_id,
                    "case_outcome_recorded",
                    now_utc,
                    "case_hypothesis_outcome",
                    outcome_id,
                    f"Hypothesis outcome recorded for {case_id}: {det_val.upper()} ({str(failure_mode or '').strip()}).",
                    json.dumps(
                        {"case_id": case_id, "hypothesis_id": hypothesis_id, "determination": det_val},
                        sort_keys=True,
                    ),
                ),
            )
            connection.commit()

        return {
            "outcome_id": outcome_id,
            "case_id": case_id,
            "dataset_id": dataset_id,
            "hypothesis_id": str(hypothesis_id).strip(),
            "failure_mode": str(failure_mode or "").strip(),
            "determination": det_val,
            "confidence": conf_val,
            "notes": str(notes or "").strip(),
            "determined_at": now_utc,
            "determined_by": str(determined_by).strip(),
        }

    def transition_case_status(
        self,
        store: EvidenceStore,
        dataset_id: str,
        case_id: str,
        new_status: Union[CaseStatus, str],
        reason: str,
        changed_by: str = "Reliability Engineer",
        closure_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate and apply a case status transition with full audit ledger tracking."""
        try:
            target_status = (
                new_status if isinstance(new_status, CaseStatus) else CaseStatus(new_status)
            )
        except (TypeError, ValueError) as exc:
            valid_statuses = ", ".join(status.value for status in CaseStatus)
            raise CaseTransitionError(
                f"Invalid case status {new_status!r}. Expected one of: {valid_statuses}."
            ) from exc

        self._require_scope(store, dataset_id)
        if not reason or not str(reason).strip():
            raise HistoryStoreError("Transition reason cannot be empty.")

        clean_reason = str(reason).strip()
        clean_changed_by = str(changed_by).strip() or "Reliability Engineer"
        now_utc = datetime.now(timezone.utc).isoformat()
        transition_id = str(uuid.uuid4())

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            case_row = connection.execute(
                """
                SELECT case_id, dataset_id, run_id, incident_id, status,
                       created_at, updated_at, closed_at, closure_summary
                FROM investigation_cases
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()
            if case_row is None:
                raise HistoryStoreError(f"Case '{case_id}' does not exist.")
            if case_row["dataset_id"] != dataset_id:
                raise DatasetOwnershipError(
                    f"Case '{case_id}' does not belong to active dataset '{dataset_id}'."
                )

            current_status = case_row["status"]
            if current_status == target_status.value:
                connection.rollback()
                return self.get_case(store, dataset_id, case_id)

            try:
                current_case_status = CaseStatus(current_status)
            except ValueError as exc:
                raise CaseTransitionError(
                    f"Cannot transition case with unrecognized current status {current_status!r}."
                ) from exc
            if target_status not in self.ALLOWED_CASE_TRANSITIONS[current_case_status]:
                raise CaseTransitionError(
                    f"Transition from '{current_status}' to '{target_status.value}' is not allowed."
                )

            if target_status == CaseStatus.ACTION_TAKEN:
                # Must have at least 1 inspection logged
                insp_count = connection.execute(
                    "SELECT COUNT(*) FROM case_inspections WHERE case_id = ? AND dataset_id = ?",
                    (case_id, dataset_id),
                ).fetchone()[0]
                if insp_count == 0:
                    raise CaseTransitionError(
                        "Cannot transition to 'action_taken': at least one physical inspection or maintenance action must be logged."
                    )

            elif target_status == CaseStatus.RESOLVED:
                # Must have at least 1 hypothesis outcome recorded
                outcome_count = connection.execute(
                    "SELECT COUNT(*) FROM case_hypothesis_outcomes WHERE case_id = ? AND dataset_id = ?",
                    (case_id, dataset_id),
                ).fetchone()[0]
                if outcome_count == 0:
                    raise CaseTransitionError(
                        "Cannot transition to 'resolved': at least one hypothesis outcome determination must be recorded."
                    )

            elif target_status == CaseStatus.CLOSED:
                if not closure_summary or not str(closure_summary).strip():
                    raise CaseTransitionError(
                        "Cannot close case without a documented closure summary."
                    )

            elif target_status == CaseStatus.OPEN:
                # Reopening from RESOLVED or CLOSED
                if current_status not in (CaseStatus.RESOLVED.value, CaseStatus.CLOSED.value):
                    raise CaseTransitionError(
                        f"Cannot reopen case from status '{current_status}'. Only RESOLVED or CLOSED cases can be reopened."
                    )
                if len(clean_reason) < 5:
                    raise CaseTransitionError(
                        "Reopening a case requires a detailed, documented justification (at least 5 characters)."
                    )

            # Apply status update
            closed_at_val = (
                now_utc
                if target_status in (CaseStatus.RESOLVED, CaseStatus.CLOSED)
                else None
            )
            closure_sum_val = (
                str(closure_summary).strip()
                if target_status == CaseStatus.CLOSED
                else case_row["closure_summary"]
            )

            connection.execute(
                """
                UPDATE investigation_cases
                SET status = ?, updated_at = ?, closed_at = ?, closure_summary = ?
                WHERE case_id = ?
                """,
                (target_status.value, now_utc, closed_at_val, closure_sum_val, case_id),
            )

            # Append to case_status_history
            connection.execute(
                """
                INSERT INTO case_status_history (
                    transition_id, case_id, dataset_id, from_status,
                    to_status, reason, changed_by, changed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transition_id,
                    case_id,
                    dataset_id,
                    current_status,
                    target_status.value,
                    clean_reason,
                    clean_changed_by,
                    now_utc,
                ),
            )

            # Append to activity_events
            event_id = f"case_transition:{transition_id}"
            connection.execute(
                """
                INSERT INTO activity_events (
                    event_id, run_id, dataset_id, event_type, occurred_at,
                    entity_type, entity_id, summary, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    case_row["run_id"],
                    dataset_id,
                    "case_status_changed",
                    now_utc,
                    "case_status_transition",
                    transition_id,
                    f"Case {case_id} transitioned from {current_status} to {target_status.value}: {clean_reason}",
                    json.dumps(
                        {"case_id": case_id, "from": current_status, "to": target_status.value, "by": clean_changed_by},
                        sort_keys=True,
                    ),
                ),
            )
            connection.commit()

        return self.get_case(store, dataset_id, case_id)

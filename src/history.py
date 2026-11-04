"""Persistent, dataset-scoped investigation history and activity storage."""

import json
import math
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from src.models import DatasetOwnershipError, FaultAnalysisResult
from src.storage import EvidenceStore


class HistoryStoreError(RuntimeError):
    """Raised when investigation history cannot be safely read or written."""


class InvestigationHistoryStore:
    """Store immutable analysis snapshots separately from the evidence search index."""

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.db_path = (
                project_root / "data" / "history" / "investigation_history.db"
            )
        else:
            self.db_path = Path(db_path).resolve()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

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
                required = {"investigation_runs", "activity_events"}
                if not required.issubset(tables):
                    raise HistoryStoreError(
                        "History database schema is marked current but is incomplete."
                    )
                return

            connection.execute("BEGIN IMMEDIATE")
            try:
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
                        run_id TEXT NOT NULL UNIQUE,
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
                connection.execute(
                    f"PRAGMA user_version = {self.SCHEMA_VERSION}"
                )
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

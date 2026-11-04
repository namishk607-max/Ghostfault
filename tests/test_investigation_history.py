import sqlite3
import uuid

import pytest
from streamlit.testing.v1 import AppTest

from src.history import HistoryStoreError, InvestigationHistoryStore
from src.hub import compute_recent_investigation_activity
from src.models import (
    AnalysisStatus,
    DatasetOwnershipError,
    DEFAULT_DATASET_ID,
    FaultAnalysisResult,
)
from src.storage import EvidenceStore


@pytest.fixture
def history_context(tmp_path):
    evidence_store = EvidenceStore(tmp_path / "evidence")
    evidence_store.ensure_default_dataset()
    evidence_store.create_dataset("DATASET-HISTORY-B", "History Dataset B")
    history_path = tmp_path / "history" / "investigation_history.db"
    history_store = InvestigationHistoryStore(history_path)
    return evidence_store, history_store, history_path


def _snapshot(dataset_id=DEFAULT_DATASET_ID, incident_id="INC-HISTORY-1"):
    return {
        "incident_id": incident_id,
        "dataset_id": dataset_id,
        "machine_id": "M-104",
        "symptom": "Bearing vibration",
        "date": "2026-01-02",
    }


def _result(snapshot, status="completed", analyzed_at="2026-01-02T03:04:05+00:00"):
    return {
        "incident_id": snapshot["incident_id"],
        "incident_facts": dict(snapshot),
        "status": status,
        "analyzed_at": analyzed_at,
        "retrieved_evidence": [],
        "hypotheses": [],
        "missing_information": [],
        "limitations": [],
    }


def test_history_schema_is_versioned_and_separate_from_evidence_index(history_context):
    evidence_store, history_store, history_path = history_context

    assert history_store.db_path == history_path.resolve()
    assert history_store.db_path != (
        evidence_store.base_dir / "index" / "evidence_index.db"
    )
    with sqlite3.connect(history_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"investigation_runs", "activity_events"}.issubset(tables)


def test_save_run_is_idempotent_and_creates_one_activity_event(history_context):
    evidence_store, history_store, history_path = history_context
    snapshot = _snapshot()
    run_id = str(uuid.uuid4())

    first = history_store.save_run(
        evidence_store, run_id, DEFAULT_DATASET_ID, snapshot, _result(snapshot)
    )
    retry = history_store.save_run(
        evidence_store, run_id, DEFAULT_DATASET_ID, snapshot, _result(snapshot)
    )

    assert first == {"run_id": run_id, "created": True}
    assert retry == {"run_id": run_id, "created": False}
    with sqlite3.connect(history_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM investigation_runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM activity_events").fetchone()[0] == 1
    assert len(history_store.list_runs(evidence_store, DEFAULT_DATASET_ID)) == 1
    assert len(history_store.list_activity(evidence_store, DEFAULT_DATASET_ID)) == 1


def test_fault_analysis_result_model_is_persisted_as_a_snapshot(history_context):
    evidence_store, history_store, _ = history_context
    snapshot = _snapshot()
    result = FaultAnalysisResult(
        incident_id=snapshot["incident_id"],
        incident_facts=dict(snapshot),
        retrieved_evidence=[],
        hypotheses=[],
        missing_information=[],
        limitations=["No matching evidence was available."],
        status=AnalysisStatus.NO_EVIDENCE,
        analyzed_at="2026-01-02T03:04:05+00:00",
    )
    run_id = str(uuid.uuid4())

    history_store.save_run(
        evidence_store, run_id, DEFAULT_DATASET_ID, snapshot, result
    )

    saved = history_store.get_run(evidence_store, DEFAULT_DATASET_ID, run_id)
    assert saved["result"]["status"] == AnalysisStatus.NO_EVIDENCE.value
    assert saved["result"]["incident_facts"] == snapshot


def test_run_id_cannot_be_reused_for_different_snapshot(history_context):
    evidence_store, history_store, _ = history_context
    snapshot = _snapshot()
    run_id = str(uuid.uuid4())
    history_store.save_run(
        evidence_store, run_id, DEFAULT_DATASET_ID, snapshot, _result(snapshot)
    )

    with pytest.raises(HistoryStoreError, match="different content"):
        history_store.save_run(
            evidence_store,
            run_id,
            DEFAULT_DATASET_ID,
            snapshot,
            _result(snapshot, status="no_evidence"),
        )


def test_history_write_requires_registered_matching_dataset_ownership(history_context):
    evidence_store, history_store, _ = history_context
    snapshot = _snapshot()

    with pytest.raises(DatasetOwnershipError, match="does not belong"):
        history_store.save_run(
            evidence_store,
            str(uuid.uuid4()),
            "DATASET-HISTORY-B",
            snapshot,
            _result(snapshot),
        )
    with pytest.raises(DatasetOwnershipError, match="unregistered dataset"):
        history_store.list_runs(evidence_store, "DATASET-NOT-REGISTERED")


@pytest.mark.parametrize(
    "evidence_section",
    ["retrieved_evidence", "hypothesis_evidence", "excluded_evidence"],
)
def test_history_write_rejects_evidence_from_another_dataset(
    history_context,
    evidence_section,
):
    evidence_store, history_store, _ = history_context
    snapshot = _snapshot("DATASET-HISTORY-B")
    result = _result(snapshot)
    reference = {
        "chunk_id": "CHUNK-DATASET-A",
        "document_id": "DOC-DATASET-A",
        "dataset_id": DEFAULT_DATASET_ID,
    }
    if evidence_section == "retrieved_evidence":
        result["retrieved_evidence"] = [{"chunk": reference}]
    elif evidence_section == "hypothesis_evidence":
        result["hypotheses"] = [
            {"supporting_evidence": [reference], "contradictory_evidence": []}
        ]
    else:
        result["excluded_evidence"] = [reference]

    with pytest.raises(DatasetOwnershipError, match="does not belong"):
        history_store.save_run(
            evidence_store,
            str(uuid.uuid4()),
            "DATASET-HISTORY-B",
            snapshot,
            result,
        )


def test_history_snapshots_normalize_non_finite_incident_values(history_context):
    evidence_store, history_store, _ = history_context
    snapshot = _snapshot()
    snapshot["optional_measurement"] = float("nan")
    run_id = str(uuid.uuid4())
    history_store.save_run(
        evidence_store,
        run_id,
        DEFAULT_DATASET_ID,
        snapshot,
        _result(snapshot),
    )

    saved = history_store.get_run(evidence_store, DEFAULT_DATASET_ID, run_id)

    assert saved["incident_snapshot"]["optional_measurement"] is None
    assert saved["result"]["incident_facts"]["optional_measurement"] is None


def test_list_and_detail_are_strictly_dataset_scoped(history_context):
    evidence_store, history_store, _ = history_context
    snapshot_a = _snapshot(DEFAULT_DATASET_ID, "INC-HISTORY-A")
    snapshot_b = _snapshot("DATASET-HISTORY-B", "INC-HISTORY-B")
    run_a = str(uuid.uuid4())
    run_b = str(uuid.uuid4())
    history_store.save_run(
        evidence_store,
        run_a,
        DEFAULT_DATASET_ID,
        snapshot_a,
        _result(snapshot_a),
    )
    history_store.save_run(
        evidence_store,
        run_b,
        "DATASET-HISTORY-B",
        snapshot_b,
        _result(snapshot_b),
    )

    assert [row["run_id"] for row in history_store.list_runs(
        evidence_store, DEFAULT_DATASET_ID
    )] == [run_a]
    assert [row["run_id"] for row in history_store.list_runs(
        evidence_store, "DATASET-HISTORY-B"
    )] == [run_b]
    assert history_store.get_run(evidence_store, "DATASET-HISTORY-B", run_a) is None
    assert history_store.get_run(evidence_store, DEFAULT_DATASET_ID, run_b) is None


def test_detail_read_rejects_a_record_with_tampered_dataset_ownership(history_context):
    evidence_store, history_store, history_path = history_context
    snapshot = _snapshot(DEFAULT_DATASET_ID, "INC-HISTORY-A")
    run_id = str(uuid.uuid4())
    history_store.save_run(
        evidence_store,
        run_id,
        DEFAULT_DATASET_ID,
        snapshot,
        _result(snapshot),
    )
    with sqlite3.connect(history_path) as connection:
        connection.execute(
            "UPDATE investigation_runs SET dataset_id = ? WHERE run_id = ?",
            ("DATASET-HISTORY-B", run_id),
        )

    with pytest.raises(
        DatasetOwnershipError,
        match="does not match the requested dataset",
    ):
        history_store.get_run(evidence_store, "DATASET-HISTORY-B", run_id)


def test_get_run_returns_saved_snapshot_without_reanalysis(history_context):
    evidence_store, history_store, _ = history_context
    snapshot = _snapshot()
    result = _result(snapshot)
    run_id = str(uuid.uuid4())
    history_store.save_run(
        evidence_store, run_id, DEFAULT_DATASET_ID, snapshot, result
    )

    saved = history_store.get_run(evidence_store, DEFAULT_DATASET_ID, run_id)

    assert saved["incident_snapshot"] == snapshot
    assert saved["result"] == result


def test_history_page_renders_a_saved_run_from_temporary_storage(history_context):
    evidence_store, history_store, history_path = history_context
    snapshot = _snapshot()
    run_id = str(uuid.uuid4())
    history_store.save_run(
        evidence_store, run_id, DEFAULT_DATASET_ID, snapshot, _result(snapshot)
    )
    script = f"""
from pathlib import Path
from app import render_investigation_history
from src.history import InvestigationHistoryStore
from src.storage import EvidenceStore

evidence_store = EvidenceStore(Path({str(evidence_store.base_dir)!r}))
history_store = InvestigationHistoryStore(Path({str(history_path)!r}))
render_investigation_history(
    history_store,
    evidence_store,
    evidence_store.get_dataset({DEFAULT_DATASET_ID!r}),
)
"""

    app = AppTest.from_string(script).run()

    assert not app.exception


def test_run_and_activity_event_are_inserted_transactionally(history_context):
    evidence_store, history_store, history_path = history_context
    snapshot = _snapshot()
    run_id = str(uuid.uuid4())
    with sqlite3.connect(history_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_activity
            BEFORE INSERT ON activity_events
            BEGIN
                SELECT RAISE(ABORT, 'simulated activity failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="simulated activity failure"):
        history_store.save_run(
            evidence_store, run_id, DEFAULT_DATASET_ID, snapshot, _result(snapshot)
        )

    with sqlite3.connect(history_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM investigation_runs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM activity_events").fetchone()[0] == 0


def test_project_activity_aggregation_reads_each_registered_scope(history_context):
    evidence_store, history_store, _ = history_context
    snapshot_a = _snapshot(DEFAULT_DATASET_ID, "INC-HISTORY-A")
    snapshot_b = _snapshot("DATASET-HISTORY-B", "INC-HISTORY-B")
    history_store.save_run(
        evidence_store,
        str(uuid.uuid4()),
        DEFAULT_DATASET_ID,
        snapshot_a,
        _result(snapshot_a),
    )
    history_store.save_run(
        evidence_store,
        str(uuid.uuid4()),
        "DATASET-HISTORY-B",
        snapshot_b,
        _result(snapshot_b),
    )

    activity = compute_recent_investigation_activity(
        history_store,
        evidence_store,
        limit=10,
    )

    assert {event["dataset_id"] for event in activity} == {
        DEFAULT_DATASET_ID,
        "DATASET-HISTORY-B",
    }
    assert len(activity) == 2

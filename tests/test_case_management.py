"""Automated tests for GhostFault Milestone 8: Case Follow-up & Outcome Tracking."""

import sqlite3
import uuid
from pathlib import Path
import pytest

from src.history import (
    InvestigationHistoryStore,
    HistoryStoreError,
    CaseAlreadyExistsError,
    CaseTransitionError,
)
from src.models import (
    DEFAULT_DATASET_ID,
    DatasetOwnershipError,
    CaseStatus,
    CasePriority,
    InspectionType,
    HypothesisDetermination,
    OutcomeConfidence,
)
from src.storage import EvidenceStore


@pytest.fixture
def m8_context(tmp_path):
    evidence_store = EvidenceStore(tmp_path / "evidence")
    evidence_store.ensure_default_dataset()
    evidence_store.create_dataset("DATASET-M8-B", "M8 Dataset B")
    history_path = tmp_path / "history" / "investigation_history.db"
    history_store = InvestigationHistoryStore(history_path, allow_migration=True)
    return evidence_store, history_store, history_path


def _create_sample_run(evidence_store, history_store, dataset_id=DEFAULT_DATASET_ID, incident_id="INC-M8-1"):
    snapshot = {
        "incident_id": incident_id,
        "dataset_id": dataset_id,
        "machine_id": "M-104",
        "symptom": "Bearing vibration and high temperature",
        "date": "2026-03-01",
    }
    result = {
        "incident_id": incident_id,
        "incident_facts": dict(snapshot),
        "status": "completed",
        "analyzed_at": "2026-03-01T10:00:00+00:00",
        "retrieved_evidence": [],
        "hypotheses": [
            {
                "hypothesis_id": "HYP-1",
                "failure_mode": "Bearing inner race fatigue",
                "mechanism": "Fatigue spalling causes elevated vibration and temperature.",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            {
                "hypothesis_id": "HYP-2",
                "failure_mode": "Lubrication degradation",
                "mechanism": "Viscosity breakdown leading to metal contact.",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
        ],
        "missing_information": [],
        "limitations": [],
    }
    run_id = str(uuid.uuid4())
    history_store.save_run(evidence_store, run_id, dataset_id, snapshot, result)
    return run_id, snapshot, result


def test_m8_database_schema_initialization(m8_context):
    """Verify v2 schema tables and indexes exist on fresh database."""
    _, history_store, history_path = m8_context
    assert history_store.SCHEMA_VERSION == 2

    with sqlite3.connect(history_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert InvestigationHistoryStore.REQUIRED_V2_TABLES.issubset(tables)


def test_v1_to_v2_migration_on_copy(tmp_path):
    """Verify safe, non-destructive migration of a v1 database to v2."""
    v1_db_path = tmp_path / "v1_test.db"
    
    # Initialize v1 tables manually
    with sqlite3.connect(v1_db_path) as conn:
        conn.execute(
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
        conn.execute(
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
                details_json TEXT NOT NULL
            )
            """
        )
        # Insert a v1 row
        conn.execute(
            """
            INSERT INTO investigation_runs VALUES (
                'run-v1-test', 'DATASET-DEFAULT', 'INC-1', '2026-01-01T00:00:00Z',
                'completed', '{}', '{}', 1
            )
            """
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()

    # Now open with allow_migration=False -> version remains 1
    store_no_mig = InvestigationHistoryStore(v1_db_path, allow_migration=False)
    with sqlite3.connect(v1_db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1

    # Now open with allow_migration=True -> migrates safely to 2
    store_mig = InvestigationHistoryStore(v1_db_path, allow_migration=True)
    with sqlite3.connect(v1_db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert InvestigationHistoryStore.REQUIRED_V2_TABLES.issubset(tables)
        # Verify existing data preserved
        row = conn.execute("SELECT incident_id FROM investigation_runs WHERE run_id='run-v1-test'").fetchone()
        assert row[0] == "INC-1"


def test_create_case_and_duplicate_prevention(m8_context):
    """Verify case creation and strict one-case-per-run duplicate prevention."""
    evidence_store, history_store, _ = m8_context
    run_id, _, _ = _create_sample_run(evidence_store, history_store)

    case = history_store.create_case(
        evidence_store,
        DEFAULT_DATASET_ID,
        run_id,
        title="Bearing Investigation Follow-up",
        priority=CasePriority.HIGH,
        created_by="Lead Engineer",
    )

    assert case["case_id"] is not None
    assert case["run_id"] == run_id
    assert case["status"] == CaseStatus.OPEN.value
    assert case["priority"] == CasePriority.HIGH.value
    assert case["incident_id"] == "INC-M8-1"
    assert len(case["status_history"]) == 1
    assert case["status_history"][0]["to_status"] == CaseStatus.OPEN.value

    # Verify duplicate prevention
    with pytest.raises(CaseAlreadyExistsError, match="already exists"):
        history_store.create_case(
            evidence_store,
            DEFAULT_DATASET_ID,
            run_id,
            title="Duplicate Attempt",
        )


def test_log_inspection_observations_separated(m8_context):
    """Verify technician shop-floor observations and replaced parts logging."""
    evidence_store, history_store, _ = m8_context
    run_id, _, _ = _create_sample_run(evidence_store, history_store)
    case = history_store.create_case(evidence_store, DEFAULT_DATASET_ID, run_id, "Inspection Case")

    inspection = history_store.log_inspection(
        evidence_store,
        DEFAULT_DATASET_ID,
        case["case_id"],
        inspector_name="Tech Dave",
        inspection_type=InspectionType.PHYSICAL_DISASSEMBLY,
        observations="Significant spalling observed on inner ring of drive-end bearing.",
        physical_action_taken="Removed defective bearing and pressed on replacement.",
        parts_replaced=["BRG-6205-C3", "SEAL-30X52X7"],
    )

    assert inspection["inspection_id"] is not None
    assert inspection["inspector_name"] == "Tech Dave"
    assert inspection["parts_replaced"] == ["BRG-6205-C3", "SEAL-30X52X7"]

    # Verify loaded in case
    loaded_case = history_store.get_case(evidence_store, DEFAULT_DATASET_ID, case["case_id"])
    assert len(loaded_case["inspections"]) == 1
    assert loaded_case["inspections"][0]["observations"] == "Significant spalling observed on inner ring of drive-end bearing."


def test_independent_hypothesis_outcome_tracking(m8_context):
    """Verify independent outcome evaluations for different hypotheses."""
    evidence_store, history_store, _ = m8_context
    run_id, _, _ = _create_sample_run(evidence_store, history_store)
    case = history_store.create_case(evidence_store, DEFAULT_DATASET_ID, run_id, "Outcome Case")

    # Evaluate HYP-1 as verified correct
    out1 = history_store.record_hypothesis_outcome(
        evidence_store,
        DEFAULT_DATASET_ID,
        case["case_id"],
        hypothesis_id="HYP-1",
        failure_mode="Bearing inner race fatigue",
        determination=HypothesisDetermination.VERIFIED_CORRECT,
        confidence=OutcomeConfidence.HIGH,
        notes="Teardown confirmed deep micro-flaking and spalling on inner raceway.",
        determined_by="Senior Inspector",
    )
    assert out1["determination"] == HypothesisDetermination.VERIFIED_CORRECT.value

    # Evaluate HYP-2 as incorrect
    out2 = history_store.record_hypothesis_outcome(
        evidence_store,
        DEFAULT_DATASET_ID,
        case["case_id"],
        hypothesis_id="HYP-2",
        failure_mode="Lubrication degradation",
        determination=HypothesisDetermination.INCORRECT,
        confidence=OutcomeConfidence.HIGH,
        notes="Grease was clean, unoxidized, and present in proper quantity.",
        determined_by="Senior Inspector",
    )
    assert out2["determination"] == HypothesisDetermination.INCORRECT.value

    loaded_case = history_store.get_case(evidence_store, DEFAULT_DATASET_ID, case["case_id"])
    assert len(loaded_case["hypothesis_outcomes"]) == 2


def test_case_status_transition_rules_and_reopening(m8_context):
    """Verify lifecycle state rules, transition barriers, and documented reopening."""
    evidence_store, history_store, _ = m8_context
    run_id, _, _ = _create_sample_run(evidence_store, history_store)
    case = history_store.create_case(evidence_store, DEFAULT_DATASET_ID, run_id, "Lifecycle Case")
    case_id = case["case_id"]

    # 1. Transition to IN_REVIEW
    c1 = history_store.transition_case_status(
        evidence_store, DEFAULT_DATASET_ID, case_id, CaseStatus.IN_REVIEW, reason="Assigned to tech"
    )
    assert c1["status"] == CaseStatus.IN_REVIEW.value

    # 2. Cannot transition to ACTION_TAKEN without an inspection
    with pytest.raises(CaseTransitionError, match="at least one physical inspection"):
        history_store.transition_case_status(
            evidence_store, DEFAULT_DATASET_ID, case_id, CaseStatus.ACTION_TAKEN, reason="Try action"
        )

    # Log inspection
    history_store.log_inspection(
        evidence_store,
        DEFAULT_DATASET_ID,
        case_id,
        inspector_name="Tech Dave",
        inspection_type=InspectionType.OPERATIONAL_TEST,
        observations="Runout excessive",
        physical_action_taken="Realigned coupling",
    )

    # Now ACTION_TAKEN succeeds
    c2 = history_store.transition_case_status(
        evidence_store, DEFAULT_DATASET_ID, case_id, CaseStatus.ACTION_TAKEN, reason="Realignment done"
    )
    assert c2["status"] == CaseStatus.ACTION_TAKEN.value

    # 3. Cannot transition to RESOLVED without a hypothesis outcome determination
    with pytest.raises(CaseTransitionError, match="at least one hypothesis outcome"):
        history_store.transition_case_status(
            evidence_store, DEFAULT_DATASET_ID, case_id, CaseStatus.RESOLVED, reason="Try resolve"
        )

    # Record outcome
    history_store.record_hypothesis_outcome(
        evidence_store,
        DEFAULT_DATASET_ID,
        case_id,
        hypothesis_id="HYP-1",
        failure_mode="Bearing fatigue",
        determination=HypothesisDetermination.VERIFIED_CORRECT,
    )

    # Now RESOLVED succeeds
    c3 = history_store.transition_case_status(
        evidence_store, DEFAULT_DATASET_ID, case_id, CaseStatus.RESOLVED, reason="Verified and running"
    )
    assert c3["status"] == CaseStatus.RESOLVED.value

    # 4. Cannot CLOSE without closure_summary
    with pytest.raises(CaseTransitionError, match="closure summary"):
        history_store.transition_case_status(
            evidence_store, DEFAULT_DATASET_ID, case_id, CaseStatus.CLOSED, reason="Close case"
        )

    c4 = history_store.transition_case_status(
        evidence_store,
        DEFAULT_DATASET_ID,
        case_id,
        CaseStatus.CLOSED,
        reason="Administrative sign-off",
        closure_summary="Machine returned to production with normal vibration readings.",
    )
    assert c4["status"] == CaseStatus.CLOSED.value
    assert c4["closure_summary"] is not None

    # 5. Reopening: requires documented justification
    with pytest.raises(CaseTransitionError, match="justification"):
        history_store.transition_case_status(
            evidence_store, DEFAULT_DATASET_ID, case_id, CaseStatus.OPEN, reason="bad"
        )

    c_reopened = history_store.transition_case_status(
        evidence_store,
        DEFAULT_DATASET_ID,
        case_id,
        CaseStatus.OPEN,
        reason="Vibration symptoms recurred 48 hours post-teardown during thermal expansion.",
    )
    assert c_reopened["status"] == CaseStatus.OPEN.value
    assert len(c_reopened["status_history"]) >= 5


def test_dataset_isolation_and_cross_dataset_tamper_rejection(m8_context):
    """Verify strict dataset scoping and tamper detection on all M8 operations."""
    evidence_store, history_store, history_path = m8_context
    run_a, _, _ = _create_sample_run(evidence_store, history_store, DEFAULT_DATASET_ID, "INC-A")
    run_b, _, _ = _create_sample_run(evidence_store, history_store, "DATASET-M8-B", "INC-B")

    case_a = history_store.create_case(evidence_store, DEFAULT_DATASET_ID, run_a, "Case A")
    case_b = history_store.create_case(evidence_store, "DATASET-M8-B", run_b, "Case B")

    # List scoping
    assert [c["case_id"] for c in history_store.list_cases(evidence_store, DEFAULT_DATASET_ID)] == [case_a["case_id"]]
    assert [c["case_id"] for c in history_store.list_cases(evidence_store, "DATASET-M8-B")] == [case_b["case_id"]]

    # Cross-dataset read rejection with DatasetOwnershipError
    with pytest.raises(DatasetOwnershipError, match="does not belong to active dataset"):
        history_store.get_case(evidence_store, "DATASET-M8-B", case_a["case_id"])
    with pytest.raises(DatasetOwnershipError, match="does not belong to active dataset"):
        history_store.get_case(evidence_store, DEFAULT_DATASET_ID, case_b["case_id"])

    # Non-existent case returns None
    assert history_store.get_case(evidence_store, DEFAULT_DATASET_ID, str(uuid.uuid4())) is None

    # Cross-dataset action rejection
    with pytest.raises(DatasetOwnershipError):
        history_store.log_inspection(
            evidence_store,
            "DATASET-M8-B",
            case_a["case_id"],
            inspector_name="Attacker",
            inspection_type=InspectionType.OPERATIONAL_TEST,
            observations="Cross dataset attempt",
            physical_action_taken="",
        )

    # Tampered database row rejection: Grounding run mismatch
    with sqlite3.connect(history_path) as conn:
        conn.execute(
            "UPDATE investigation_cases SET dataset_id = 'DATASET-M8-B' WHERE case_id = ?",
            (case_a["case_id"],),
        )

    with pytest.raises(DatasetOwnershipError, match="Grounding investigation run for case"):
        history_store.get_case(evidence_store, "DATASET-M8-B", case_a["case_id"])


def test_migration_removes_unique_constraint_on_activity_events_run_id(tmp_path):
    """Verify v1->v2 migration rebuilds activity_events without UNIQUE constraint on run_id."""
    v1_db_path = tmp_path / "v1_history.db"
    
    # 1. Create a legacy v1 database where activity_events had run_id TEXT NOT NULL UNIQUE
    with sqlite3.connect(v1_db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
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
        conn.execute(
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
        conn.execute(
            """
            CREATE INDEX activity_events_dataset_time
            ON activity_events(dataset_id, occurred_at DESC)
            """
        )
        conn.execute("PRAGMA user_version = 1")

        # Insert a run and initial activity event
        run_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO investigation_runs VALUES (
                ?, 'DATASET-DEFAULT', 'INC-V1', '2026-03-01T00:00:00+00:00',
                'completed', '{"incident_id": "INC-V1", "dataset_id": "DATASET-DEFAULT"}',
                '{"incident_id": "INC-V1", "incident_facts": {"incident_id": "INC-V1", "dataset_id": "DATASET-DEFAULT"}, "status": "completed", "analyzed_at": "2026-03-01T00:00:00+00:00"}',
                1
            )
            """,
            (run_id,),
        )
        conn.execute(
            """
            INSERT INTO activity_events VALUES (
                ?, ?, 'DATASET-DEFAULT', 'investigation_run_saved',
                '2026-03-01T00:00:00+00:00', 'investigation_run', ?,
                'Run saved', '{}'
            )
            """,
            (f"investigation:{run_id}", run_id, run_id),
        )
        conn.commit()

        # Verify that inserting a second activity event with the same run_id fails on v1
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed: activity_events.run_id"):
            conn.execute(
                """
                INSERT INTO activity_events VALUES (
                    ?, ?, 'DATASET-DEFAULT', 'case_created',
                    '2026-03-01T00:01:00+00:00', 'investigation_case', 'case-1',
                    'Case created', '{}'
                )
                """,
                (f"case_created:case-1", run_id),
            )

    # 2. Run migration via InvestigationHistoryStore(allow_migration=True)
    evidence_store = EvidenceStore(tmp_path / "evidence")
    evidence_store.ensure_default_dataset()
    store = InvestigationHistoryStore(v1_db_path, allow_migration=True)

    # 3. Confirm UNIQUE constraint is absent after migration by checking PRAGMA index_list
    with sqlite3.connect(v1_db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        indexes = conn.execute("PRAGMA index_list(activity_events)").fetchall()
        # Ensure no unique index exists for run_id
        for idx in indexes:
            is_unique = idx[2]
            idx_name = idx[1]
            cols = [c[2] for c in conn.execute(f"PRAGMA index_info('{idx_name}')").fetchall()]
            if cols == ["run_id"]:
                assert not is_unique, f"run_id still has a unique index: {idx_name}"

    # 4. Verify regression requirements:
    # a) Creating a case when an investigation already has a saved activity event succeeds
    case = store.create_case(evidence_store, DEFAULT_DATASET_ID, run_id, "Post-migration Case")
    assert case["case_id"] is not None
    assert case["run_id"] == run_id

    # b) Preventing duplicate cases for the same run
    with pytest.raises(CaseAlreadyExistsError):
        store.create_case(evidence_store, DEFAULT_DATASET_ID, run_id, "Duplicate Case")

    # c) Allowing multiple activity events for one run
    with sqlite3.connect(v1_db_path) as conn:
        events = conn.execute("SELECT event_id, run_id, event_type FROM activity_events WHERE run_id = ?", (run_id,)).fetchall()
        assert len(events) >= 2  # The original run event + case_created event

    # d) Allowing multiple status-history entries for one case
    store.transition_case_status(
        evidence_store, DEFAULT_DATASET_ID, case["case_id"],
        CaseStatus.IN_REVIEW, reason="First transition"
    )
    store.record_hypothesis_outcome(
        evidence_store, DEFAULT_DATASET_ID, case["case_id"],
        hypothesis_id="HYP-1", failure_mode="Bearing Wear",
        determination=HypothesisDetermination.VERIFIED_CORRECT,
        confidence=OutcomeConfidence.HIGH, notes="Verified during teardown"
    )
    store.transition_case_status(
        evidence_store, DEFAULT_DATASET_ID, case["case_id"],
        CaseStatus.RESOLVED, reason="Second transition"
    )
    loaded_case = store.get_case(evidence_store, DEFAULT_DATASET_ID, case["case_id"])
    assert len(loaded_case["status_history"]) == 3  # Initial OPEN + IN_REVIEW + RESOLVED


def _case_write_snapshot(db_path, case_id):
    with sqlite3.connect(db_path) as connection:
        case = connection.execute(
            """
            SELECT status, updated_at, closed_at, closure_summary
            FROM investigation_cases WHERE case_id = ?
            """,
            (case_id,),
        ).fetchone()
        status_history = connection.execute(
            """
            SELECT * FROM case_status_history
            WHERE case_id = ? ORDER BY transition_id
            """,
            (case_id,),
        ).fetchall()
        activity_events = connection.execute(
            """
            SELECT * FROM activity_events
            WHERE entity_id = ? OR run_id = (
                SELECT run_id FROM investigation_cases WHERE case_id = ?
            )
            ORDER BY event_id
            """,
            (case_id, case_id),
        ).fetchall()
    return case, status_history, activity_events


@pytest.mark.parametrize("invalid_status", ["unknown", "", None])
def test_invalid_case_status_is_rejected_without_writes(m8_context, invalid_status):
    evidence_store, history_store, history_path = m8_context
    run_id, _, _ = _create_sample_run(evidence_store, history_store)
    case = history_store.create_case(
        evidence_store, DEFAULT_DATASET_ID, run_id, "Invalid status test"
    )
    before = _case_write_snapshot(history_path, case["case_id"])

    with pytest.raises(CaseTransitionError, match="Invalid case status"):
        history_store.transition_case_status(
            evidence_store,
            DEFAULT_DATASET_ID,
            case["case_id"],
            invalid_status,
            reason="Must reject an invalid status",
        )

    assert _case_write_snapshot(history_path, case["case_id"]) == before


@pytest.mark.parametrize(
    "disallowed_status",
    [CaseStatus.ACTION_TAKEN, CaseStatus.RESOLVED, CaseStatus.CLOSED],
)
def test_disallowed_open_case_transitions_leave_database_unchanged(
    m8_context, disallowed_status
):
    evidence_store, history_store, history_path = m8_context
    run_id, _, _ = _create_sample_run(evidence_store, history_store)
    case = history_store.create_case(
        evidence_store, DEFAULT_DATASET_ID, run_id, "Disallowed transition test"
    )
    before = _case_write_snapshot(history_path, case["case_id"])

    with pytest.raises(CaseTransitionError, match="is not allowed"):
        history_store.transition_case_status(
            evidence_store,
            DEFAULT_DATASET_ID,
            case["case_id"],
            disallowed_status,
            reason="This direct transition is disallowed",
            closure_summary="Provided to isolate transition validation.",
        )

    assert _case_write_snapshot(history_path, case["case_id"]) == before


def test_resolved_case_can_be_reopened_with_documented_reason(m8_context):
    evidence_store, history_store, _ = m8_context
    run_id, _, _ = _create_sample_run(evidence_store, history_store)
    case = history_store.create_case(
        evidence_store, DEFAULT_DATASET_ID, run_id, "Resolved reopening test"
    )
    history_store.transition_case_status(
        evidence_store,
        DEFAULT_DATASET_ID,
        case["case_id"],
        CaseStatus.IN_REVIEW,
        reason="Assigned for follow-up",
    )
    history_store.log_inspection(
        evidence_store,
        DEFAULT_DATASET_ID,
        case["case_id"],
        inspector_name="Test inspector",
        inspection_type=InspectionType.OPERATIONAL_TEST,
        observations="Inspection completed.",
        physical_action_taken="No action required.",
    )
    history_store.transition_case_status(
        evidence_store,
        DEFAULT_DATASET_ID,
        case["case_id"],
        CaseStatus.ACTION_TAKEN,
        reason="Inspection action recorded",
    )
    history_store.record_hypothesis_outcome(
        evidence_store,
        DEFAULT_DATASET_ID,
        case["case_id"],
        hypothesis_id="HYP-1",
        failure_mode="Bearing fatigue",
        determination=HypothesisDetermination.VERIFIED_CORRECT,
    )
    history_store.transition_case_status(
        evidence_store,
        DEFAULT_DATASET_ID,
        case["case_id"],
        CaseStatus.RESOLVED,
        reason="Outcome verified",
    )

    reopened = history_store.transition_case_status(
        evidence_store,
        DEFAULT_DATASET_ID,
        case["case_id"],
        CaseStatus.OPEN,
        reason="Additional symptoms require renewed investigation",
    )

    assert reopened["status"] == CaseStatus.OPEN.value


def test_backup_database_uses_sqlite_snapshot_and_preserves_rows(m8_context):
    evidence_store, history_store, history_path = m8_context
    run_id, _, _ = _create_sample_run(evidence_store, history_store)
    history_store.create_case(
        evidence_store, DEFAULT_DATASET_ID, run_id, "Backup preservation test"
    )

    def database_rows(path):
        with sqlite3.connect(path) as connection:
            tables = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                )
            ]
            return {
                table: connection.execute(
                    f'SELECT * FROM "{table}" ORDER BY rowid'
                ).fetchall()
                for table in tables
            }

    source_rows = database_rows(history_path)
    backup_path = history_store.backup_database()
    assert backup_path != history_path
    assert backup_path.exists()

    with sqlite3.connect(backup_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert database_rows(backup_path) == source_rows
    assert database_rows(history_path) == source_rows

    first_backup_bytes = backup_path.read_bytes()
    second_backup_path = history_store.backup_database()
    assert second_backup_path != backup_path
    assert second_backup_path.exists()
    assert backup_path.read_bytes() == first_backup_bytes
    assert database_rows(second_backup_path) == source_rows

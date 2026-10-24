"""Automated unit tests for Cross-Incident Comparison Engine (Milestone 5.1)."""

import pytest
import pandas as pd
from pathlib import Path

from src.models import (
    RecurrenceScope,
    CrossIncidentComparisonResult,
    SymptomGroup,
    IncidentTimelineEntry,
)
from src.cross_incident import (
    normalize_symptom_text,
    parse_incident_date,
    compute_recurrence_scope,
    CrossIncidentComparisonEngine,
    generate_incident_timeline,
)


def test_two_incidents_same_symptom_grouped():
    """Verify that incidents with identical or equivalent symptoms are grouped together."""
    engine = CrossIncidentComparisonEngine()
    incidents = [
        {
            "incident_id": "INC-001",
            "machine_id": "M-104",
            "date": "2024-01-10",
            "symptom": "High bearing temperature and abnormal acoustic pitch",
            "maintenance_action": "Lubricated bearing",
            "outcome": "Resolved",
        },
        {
            "incident_id": "INC-002",
            "machine_id": "M-104",
            "date": "2024-02-15",
            "symptom": "Elevated bearing temperature and abnormal acoustic pitch",
            "maintenance_action": "Replaced bearing",
            "outcome": "Resolved",
        },
    ]

    result = engine.compare_incidents(incidents)
    assert result.total_incidents_analyzed == 2
    assert len(result.symptom_groups) == 1

    group = result.symptom_groups[0]
    assert group.occurrences_count == 2
    assert "INC-001" in group.incident_ids
    assert "INC-002" in group.incident_ids
    assert group.recurrence_scope == RecurrenceScope.SAME_MACHINE
    assert len(group.original_symptoms) == 2


def test_different_symptoms_not_grouped_broad_keyword_isolation():
    """
    Ensure symptoms are NOT grouped merely because they share a broad word (e.g., 'bearing').
    'High bearing temperature' and 'Abnormal bearing vibration noise' must be distinct groups.
    """
    engine = CrossIncidentComparisonEngine()
    incidents = [
        {
            "incident_id": "INC-001",
            "machine_id": "M-104",
            "date": "2024-01-10",
            "symptom": "High bearing temperature",
            "maintenance_action": "Lubricated bearing",
        },
        {
            "incident_id": "INC-002",
            "machine_id": "M-104",
            "date": "2024-02-15",
            "symptom": "Bearing vibration noise",
            "maintenance_action": "Inspected housing",
        },
    ]

    result = engine.compare_incidents(incidents)
    assert result.total_incidents_analyzed == 2
    assert len(result.symptom_groups) == 2
    norm_symptoms = [g.normalized_symptom for g in result.symptom_groups]
    assert any("temperature" in s for s in norm_symptoms)
    assert any("noise" in s for s in norm_symptoms)


def test_repeated_symptoms_same_machine():
    """Verify recurrence scope is SAME_MACHINE when recurring on identical machine."""
    engine = CrossIncidentComparisonEngine()
    incidents = [
        {"incident_id": "INC-1", "machine_id": "M-205", "date": "2024-01-01", "symptom": "Spindle positioning drift"},
        {"incident_id": "INC-2", "machine_id": "M-205", "date": "2024-02-01", "symptom": "Spindle positioning drift"},
    ]
    result = engine.compare_incidents(incidents)
    assert len(result.symptom_groups) == 1
    assert result.symptom_groups[0].recurrence_scope == RecurrenceScope.SAME_MACHINE


def test_similar_symptoms_different_machines():
    """Verify recurrence scope is CROSS_MACHINE when recurring across distinct machines."""
    engine = CrossIncidentComparisonEngine()
    incidents = [
        {"incident_id": "INC-1", "machine_id": "M-104", "date": "2024-01-01", "symptom": "Hydraulic line pressure drop"},
        {"incident_id": "INC-2", "machine_id": "M-205", "date": "2024-02-01", "symptom": "Hydraulic line pressure drop"},
    ]
    result = engine.compare_incidents(incidents)
    assert len(result.symptom_groups) == 1
    assert result.symptom_groups[0].recurrence_scope == RecurrenceScope.CROSS_MACHINE
    assert set(result.symptom_groups[0].machine_ids) == {"M-104", "M-205"}


def test_missing_machine_ids_not_treated_as_matching():
    """
    Ensure missing machine IDs are not treated as equal to each other or to known machines.
    Recurrence scope must reflect UNDETERMINED or MIXED, and limitation recorded.
    """
    engine = CrossIncidentComparisonEngine()
    incidents = [
        {"incident_id": "INC-1", "machine_id": None, "date": "2024-01-01", "symptom": "Pneumatic actuator stroke delay"},
        {"incident_id": "INC-2", "machine_id": None, "date": "2024-02-01", "symptom": "Pneumatic actuator stroke delay"},
    ]
    result = engine.compare_incidents(incidents)
    assert len(result.symptom_groups) == 1
    # Two missing machine IDs must NOT be labeled SAME_MACHINE
    assert result.symptom_groups[0].recurrence_scope == RecurrenceScope.UNDETERMINED
    assert result.missing_fields_summary["missing_machine_id"] == 2
    assert any(lim.code == "UNASSIGNED_MACHINE_RECORDS" for lim in result.limitations)


def test_missing_or_invalid_dates_handled_safely():
    """Verify missing, blank, or invalid dates are handled safely without crashing or fabrication."""
    engine = CrossIncidentComparisonEngine()
    incidents = [
        {"incident_id": "INC-1", "machine_id": "M-104", "date": "invalid-date", "symptom": "Bearing overheating"},
        {"incident_id": "INC-2", "machine_id": "M-104", "date": None, "symptom": "Bearing overheating"},
        {"incident_id": "INC-3", "machine_id": "M-104", "date": "2024-03-15", "symptom": "Bearing overheating"},
    ]
    result = engine.compare_incidents(incidents)
    assert result.missing_fields_summary["missing_or_invalid_date"] == 2

    # Timeline entries must preserve raw values
    dates_map = {t.incident_id: (t.raw_date, t.parsed_date, t.is_date_valid) for t in result.timeline_entries}
    assert dates_map["INC-1"] == ("invalid-date", None, False)
    assert dates_map["INC-2"] == (None, None, False)
    assert dates_map["INC-3"] == ("2024-03-15", "2024-03-15", True)


def test_missing_maintenance_actions():
    """Verify missing maintenance actions are tracked in summary and limitations."""
    engine = CrossIncidentComparisonEngine()
    incidents = [
        {"incident_id": "INC-1", "machine_id": "M-104", "date": "2024-01-01", "symptom": "Motor vibration", "maintenance_action": None},
        {"incident_id": "INC-2", "machine_id": "M-104", "date": "2024-01-02", "symptom": "Motor vibration", "maintenance_action": "Tightened mount"},
    ]
    result = engine.compare_incidents(incidents)
    assert result.missing_fields_summary["missing_maintenance_action"] == 1
    assert any(lim.code == "INCOMPLETE_MAINTENANCE_HISTORY" for lim in result.limitations)


def test_empty_incident_data():
    """Ensure engine handles empty list or DataFrame cleanly without error."""
    engine = CrossIncidentComparisonEngine()

    result_empty_list = engine.compare_incidents([])
    assert result_empty_list.total_incidents_analyzed == 0
    assert len(result_empty_list.symptom_groups) == 0
    assert len(result_empty_list.timeline_entries) == 0

    result_empty_df = engine.compare_incidents(pd.DataFrame())
    assert result_empty_df.total_incidents_analyzed == 0


def test_invalid_or_incomplete_input():
    """Verify malformed records with missing fields are normalized safely."""
    engine = CrossIncidentComparisonEngine()
    malformed = [
        {},
        {"extra_field": "test"},
    ]
    result = engine.compare_incidents(malformed)
    assert result.total_incidents_analyzed == 2
    assert result.missing_fields_summary["missing_machine_id"] == 2
    assert result.missing_fields_summary["missing_symptom"] == 2


def test_preservation_of_incident_ids_and_original_symptom_text():
    """Ensure all original incident IDs, raw symptoms, and actions are retained verbatim."""
    engine = CrossIncidentComparisonEngine()
    original_text = "   High bearing temperature & abnormal pitch...   "
    incidents = [
        {
            "incident_id": "INC-SPECIAL-99",
            "machine_id": "M-310",
            "date": "2024-05-20",
            "symptom": original_text,
            "maintenance_action": "Inspected bearing housing",
            "outcome": "Resolved",
        }
    ]
    result = engine.compare_incidents(incidents)
    assert result.compared_incident_ids == ["INC-SPECIAL-99"]
    group = result.symptom_groups[0]
    assert group.incident_ids == ["INC-SPECIAL-99"]
    assert group.original_symptoms[0]["symptom"] == original_text.strip()
    assert result.timeline_entries[0].symptom == original_text.strip()
    assert result.timeline_entries[0].original_record["symptom"] == original_text


def test_no_unsupported_root_cause_claims():
    """
    Ensure the cross-incident comparison result does NOT output root cause assertions
    and includes explicit limitation warnings.
    """
    engine = CrossIncidentComparisonEngine()
    incidents = [
        {"incident_id": "INC-1", "machine_id": "M-104", "date": "2024-01-01", "symptom": "High bearing temperature"},
        {"incident_id": "INC-2", "machine_id": "M-104", "date": "2024-02-01", "symptom": "High bearing temperature"},
    ]
    result = engine.compare_incidents(incidents)
    # Result must explicitly state limitation on root causes
    lim_codes = [lim.code for lim in result.limitations]
    assert "NO_ROOT_CAUSE_INFERENCE" in lim_codes
    assert "CONSERVATIVE_GROUPING" in lim_codes

    # Ensure no hypothesis/root-cause fields are added to SymptomGroup
    group_dict = result.symptom_groups[0].to_dict()
    assert "root_cause" not in group_dict
    assert "verified_cause" not in group_dict


def test_real_dataset_comparison():
    """Verify comparing the actual sample dataset works deterministically."""
    from src.data_loader import load_incidents_data
    df = load_incidents_data()
    engine = CrossIncidentComparisonEngine()
    result = engine.compare_incidents(df)

    assert result.total_incidents_analyzed == len(df)
    assert len(result.compared_incident_ids) == len(df)
    assert len(result.timeline_entries) == len(df)
    assert len(result.symptom_groups) >= 1


# ==============================================================================
# Milestone 5.2 Tests: Incident Timeline Ordering & Segregation
# ==============================================================================

def test_timeline_correct_chronological_ordering():
    """Verify incidents with valid dates are sorted strictly in chronological ascending order."""
    incidents = [
        {"incident_id": "INC-003", "machine_id": "M-104", "date": "2024-03-18", "symptom": "Pressure drop"},
        {"incident_id": "INC-001", "machine_id": "M-104", "date": "2024-01-12", "symptom": "Bearing noise"},
        {"incident_id": "INC-002", "machine_id": "M-205", "date": "2024-02-04", "symptom": "Vibration"},
    ]
    dated, undated, missing = generate_incident_timeline(incidents)

    assert len(dated) == 3
    assert len(undated) == 0
    assert [d.incident_id for d in dated] == ["INC-001", "INC-002", "INC-003"]
    assert [d.parsed_date for d in dated] == ["2024-01-12", "2024-02-04", "2024-03-18"]


def test_timeline_multiple_incidents_on_same_date_deterministic():
    """Ensure ties on identical dates are broken deterministically by incident ID ascending."""
    incidents = [
        {"incident_id": "INC-B", "machine_id": "M-104", "date": "2024-02-01", "symptom": "Symptom B"},
        {"incident_id": "INC-A", "machine_id": "M-205", "date": "2024-02-01", "symptom": "Symptom A"},
        {"incident_id": "INC-C", "machine_id": "M-310", "date": "2024-02-01", "symptom": "Symptom C"},
    ]
    dated, undated, _ = generate_incident_timeline(incidents)

    assert len(dated) == 3
    assert [d.incident_id for d in dated] == ["INC-A", "INC-B", "INC-C"]


def test_timeline_missing_and_invalid_dates_segregated():
    """Verify records with missing or invalid dates are placed in undated group, preserving raw values."""
    incidents = [
        {"incident_id": "INC-VALID-1", "machine_id": "M-104", "date": "2024-01-15", "symptom": "Symptom 1"},
        {"incident_id": "INC-INVALID", "machine_id": "M-104", "date": "not-a-date", "symptom": "Symptom 2"},
        {"incident_id": "INC-MISSING", "machine_id": "M-205", "date": None, "symptom": "Symptom 3"},
        {"incident_id": "INC-EMPTY", "machine_id": "M-310", "date": "   ", "symptom": "Symptom 4"},
        {"incident_id": "INC-VALID-2", "machine_id": "M-104", "date": "2024-02-20", "symptom": "Symptom 5"},
    ]
    dated, undated, missing = generate_incident_timeline(incidents)

    assert len(dated) == 2
    assert [d.incident_id for d in dated] == ["INC-VALID-1", "INC-VALID-2"]

    assert len(undated) == 3
    undated_ids = [u.incident_id for u in undated]
    assert "INC-INVALID" in undated_ids
    assert "INC-MISSING" in undated_ids
    assert "INC-EMPTY" in undated_ids

    # Raw dates preserved
    inv_entry = next(u for u in undated if u.incident_id == "INC-INVALID")
    assert inv_entry.raw_date == "not-a-date"
    assert inv_entry.parsed_date is None
    assert inv_entry.is_date_valid is False

    missing_entry = next(u for u in undated if u.incident_id == "INC-MISSING")
    assert missing_entry.raw_date is None
    assert missing_entry.is_date_valid is False


def test_timeline_preserves_incident_attributes():
    """Verify machine ID, symptom text, maintenance actions, and outcomes are completely preserved."""
    incidents = [
        {
            "incident_id": "INC-FULL-01",
            "machine_id": "M-999",
            "date": "2024-04-10",
            "symptom": "Specific hydraulic actuator seal bypass",
            "maintenance_action": "Replaced high-pressure nitrile O-ring seal",
            "outcome": "Resolved - 210 bar holding pressure verified",
        }
    ]
    dated, undated, _ = generate_incident_timeline(incidents)

    assert len(dated) == 1
    item = dated[0]
    assert item.incident_id == "INC-FULL-01"
    assert item.machine_id == "M-999"
    assert item.parsed_date == "2024-04-10"
    assert item.symptom == "Specific hydraulic actuator seal bypass"
    assert item.maintenance_action == "Replaced high-pressure nitrile O-ring seal"
    assert item.outcome == "Resolved - 210 bar holding pressure verified"
    assert item.original_record["incident_id"] == "INC-FULL-01"


def test_timeline_empty_incident_input():
    """Verify empty list or empty DataFrame produces empty timeline safely."""
    dated_list, undated_list, _ = generate_incident_timeline([])
    assert len(dated_list) == 0
    assert len(undated_list) == 0

    dated_df, undated_df, _ = generate_incident_timeline(pd.DataFrame())
    assert len(dated_df) == 0
    assert len(undated_df) == 0


def test_timeline_missing_optional_fields():
    """Verify records with missing optional machine_id, action, or outcome serialize gracefully."""
    incidents = [
        {
            "incident_id": "INC-MINIMAL",
            "date": "2024-05-01",
            "symptom": "General vibration warning",
        }
    ]
    dated, undated, missing = generate_incident_timeline(incidents)
    assert len(dated) == 1
    entry = dated[0]
    assert entry.incident_id == "INC-MINIMAL"
    assert entry.machine_id is None
    assert entry.maintenance_action is None
    assert entry.outcome is None
    assert missing["missing_machine_id"] == 1
    assert missing["missing_maintenance_action"] == 1


def test_timeline_on_real_sample_dataset():
    """Verify timeline execution against the genuine 6-row sample dataset."""
    from src.data_loader import load_incidents_data
    df = load_incidents_data()
    dated, undated, missing = generate_incident_timeline(df)

    assert len(dated) == 6
    assert len(undated) == 0
    # Expected chronological order by date
    expected_order = [
        "INC-2024-001",  # 2024-01-12
        "INC-2024-004",  # 2024-01-28
        "INC-2024-002",  # 2024-02-04
        "INC-2024-006",  # 2024-02-19
        "INC-2024-005",  # 2024-03-02
        "INC-2024-003",  # 2024-03-18
    ]
    assert [d.incident_id for d in dated] == expected_order


# ==============================================================================
# Milestone 5.4 Tests: Dashboard Integration & Filtered Grounding
# ==============================================================================

def test_m54_integration_single_machine_filter_comparison_and_linking():
    """Verify cross-incident comparison and evidence linking operate cleanly on a filtered machine dataset."""
    from src.data_loader import load_incidents_data, filter_by_machine
    from src.indexer import SearchIndex
    from src.retriever import EvidenceRetriever
    from src.evidence_linker import CrossIncidentEvidenceLinker

    df = load_incidents_data()
    m104_df = filter_by_machine(df, "M-104")
    assert len(m104_df) == 3

    comp_engine = CrossIncidentComparisonEngine()
    comp_result = comp_engine.compare_incidents(m104_df)

    assert comp_result.total_incidents_analyzed == 3
    assert len(comp_result.compared_incident_ids) == 3
    assert all(entry.machine_id == "M-104" for entry in comp_result.timeline_entries)

    retriever = EvidenceRetriever(SearchIndex())
    linker = CrossIncidentEvidenceLinker(retriever)
    link_result = linker.link_comparison_evidence(comp_result)

    assert link_result.total_groups_evaluated == len(comp_result.symptom_groups)
    assert len(link_result.group_results) == 3
    # Check that safety limitations are present
    assert any(lim.code == "NO_SHARED_ROOT_CAUSE" for lim in link_result.global_limitations)


def test_m54_integration_empty_filtered_incidents():
    """Verify cross-incident comparison and linking handle empty filtered dataframe without exceptions."""
    from src.indexer import SearchIndex
    from src.retriever import EvidenceRetriever
    from src.evidence_linker import CrossIncidentEvidenceLinker

    empty_df = pd.DataFrame(columns=["incident_id", "machine_id", "date", "symptom", "maintenance_action", "outcome"])
    comp_engine = CrossIncidentComparisonEngine()
    comp_result = comp_engine.compare_incidents(empty_df)

    assert comp_result.total_incidents_analyzed == 0
    assert len(comp_result.symptom_groups) == 0

    retriever = EvidenceRetriever(SearchIndex())
    linker = CrossIncidentEvidenceLinker(retriever)
    link_result = linker.link_comparison_evidence(comp_result)

    assert link_result.total_groups_evaluated == 0
    assert len(link_result.group_results) == 0



"""Automated unit tests for Cross-Incident Evidence Linking (Milestone 5.3)."""

from pathlib import Path
import pytest

from src.models import (
    DocumentMetadata,
    DocumentStatus,
    EvidenceLinkStatus,
    PassageQuality,
    EvidenceContextType,
    MachineScope,
    RecurrenceScope,
    SymptomGroup,
    CrossIncidentComparisonResult,
    current_iso_timestamp,
)
from src.indexer import SearchIndex
from src.retriever import EvidenceRetriever
from src.cross_incident import CrossIncidentComparisonEngine
from src.evidence_linker import CrossIncidentEvidenceLinker


def make_doc_meta(doc_id: str, filename: str) -> DocumentMetadata:
    """Helper to generate DocumentMetadata for test search index."""
    return DocumentMetadata(
        document_id=doc_id,
        original_filename=filename,
        file_type="txt",
        source_type="test_upload",
        storage_reference=f"originals/{doc_id}_{filename}",
        extraction_status="success",
        ingestion_status=DocumentStatus.PROCESSED,
        created_at=current_iso_timestamp(),
        updated_at=current_iso_timestamp(),
        extracted_text_reference=f"extracted/{doc_id}.txt",
        extracted_character_count=100,
    )


@pytest.fixture
def evidence_linking_setup(tmp_path: Path):
    """Fixture providing a populated test SearchIndex, EvidenceRetriever, and Linker."""
    db_path = tmp_path / "evidence_linker_test.db"
    index = SearchIndex(db_path=db_path)

    # Document 1: Bearing Maintenance Reference
    doc1 = make_doc_meta("DOC-BEAR-01", "bearing_maintenance_guide.txt")
    text1 = (
        "Drive-End Roller Bearing Technical Guide.\n"
        "Operating limits: Normal bearing temperature should remain under 65C. "
        "High bearing temperature accompanied by abnormal acoustic pitch indicates roller surface fatigue "
        "or inadequate lubrication. Recommended maintenance action: lubricate drive-end roller bearing."
    )
    index.index_document(doc1, text1)

    # Document 2: M-104 Specific Inspection Record
    doc2 = make_doc_meta("DOC-M104-01", "m104_field_report.txt")
    text2 = (
        "Equipment Inspection Report: Machine M-104.\n"
        "During peak load operation, unit M-104 exhibited high bearing temperature and acoustic pitch distress. "
        "Technician retorqued housing bolts and inspected lubricant film."
    )
    index.index_document(doc2, text2)

    # Document 3: Hydraulic Maintenance Reference
    doc3 = make_doc_meta("DOC-HYDR-01", "hydraulic_systems_sop.txt")
    text3 = (
        "Hydraulic Circuit Standard Operating Procedure.\n"
        "A sudden pressure drop across the primary hydraulic circuit typically indicates a clogged return filter. "
        "Action: Replace 10-micron filter cartridge and bleed air from circuit."
    )
    index.index_document(doc3, text3)

    # Document 4: Search-term-only metadata chunk
    doc4 = make_doc_meta("DOC-TERMS-01", "indexing_keywords.txt")
    text4 = "Search terms: bearing temperature, acoustic pitch, roller bearing, lubrication, vibration."
    index.index_document(doc4, text4)

    retriever = EvidenceRetriever(index)
    linker = CrossIncidentEvidenceLinker(retriever)
    return index, retriever, linker


def test_relevant_passage_links_to_symptom_group(evidence_linking_setup):
    """Verify relevant substantive passages link to the correct symptom group."""
    _, _, linker = evidence_linking_setup

    group = SymptomGroup(
        group_id="SYMGRP-BEARING-01",
        normalized_symptom="acoustic pitch unusual + bearing high temperature",
        canonical_label="High bearing temperature and abnormal acoustic pitch",
        incident_ids=["INC-2024-001"],
        machine_ids=["M-104"],
        recurrence_scope=RecurrenceScope.SINGLETON,
        occurrences_count=1,
        grouping_basis="Single incident observation",
        original_symptoms=[{"incident_id": "INC-2024-001", "symptom": "High bearing temperature and abnormal acoustic pitch"}],
        maintenance_actions=[{"incident_id": "INC-2024-001", "maintenance_action": "Lubricated drive-end roller bearing"}],
    )

    result = linker.link_symptom_group(group)
    assert result.status == EvidenceLinkStatus.SUPPORTED
    assert len(result.supporting_evidence) >= 1
    assert any(ev.document_id == "DOC-BEAR-01" for ev in result.supporting_evidence)

    # Check link properties
    link = result.supporting_evidence[0]
    assert link.group_id == "SYMGRP-BEARING-01"
    assert "INC-2024-001" in link.related_incident_ids
    assert link.quality == PassageQuality.SUBSTANTIVE_TECHNICAL
    assert link.snippet != ""
    assert "bearing" in link.snippet.lower()


def test_irrelevant_passages_excluded_and_no_support_when_no_match(evidence_linking_setup):
    """Verify irrelevant symptom queries return explicit no-support result."""
    _, _, linker = evidence_linking_setup

    group = SymptomGroup(
        group_id="SYMGRP-UNMATCHED-01",
        normalized_symptom="pneumatic valve air leak completely unknown component",
        canonical_label="Pneumatic valve air leak completely unknown component",
        incident_ids=["INC-999"],
        machine_ids=["M-999"],
        recurrence_scope=RecurrenceScope.SINGLETON,
        occurrences_count=1,
        grouping_basis="Test",
        original_symptoms=[{"incident_id": "INC-999", "symptom": "Pneumatic valve air leak completely unknown component"}],
        maintenance_actions=[],
    )

    result = linker.link_symptom_group(group)
    assert result.status == EvidenceLinkStatus.NO_SUPPORTING_EVIDENCE
    assert len(result.supporting_evidence) == 0
    assert any("No matching evidence passages" in lim or "Evidence links" in lim for lim in result.limitations)


def test_search_terms_and_metadata_passages_excluded_from_substantive_support(evidence_linking_setup):
    """Verify search-term lists and metadata passages are excluded from substantive supporting evidence."""
    _, _, linker = evidence_linking_setup

    group = SymptomGroup(
        group_id="SYMGRP-BEARING-01",
        normalized_symptom="bearing acoustic pitch",
        canonical_label="bearing acoustic pitch",
        incident_ids=["INC-2024-001"],
        machine_ids=["M-104"],
        recurrence_scope=RecurrenceScope.SINGLETON,
        occurrences_count=1,
        grouping_basis="Test",
        original_symptoms=[{"incident_id": "INC-2024-001", "symptom": "bearing acoustic pitch"}],
        maintenance_actions=[],
    )

    result = linker.link_symptom_group(group)
    # The search-term-only chunk from DOC-TERMS-01 must be in excluded_evidence, NOT supporting_evidence
    supporting_doc_ids = [ev.document_id for ev in result.supporting_evidence]
    assert "DOC-TERMS-01" not in supporting_doc_ids

    excluded_doc_ids = [ev.document_id for ev in result.excluded_evidence]
    assert "DOC-TERMS-01" in excluded_doc_ids
    exc_item = next(ev for ev in result.excluded_evidence if ev.document_id == "DOC-TERMS-01")
    assert exc_item.quality == PassageQuality.SEARCH_TERMS_METADATA


def test_preservation_of_document_and_passage_provenance(evidence_linking_setup):
    """Verify document ID, source filename, chunk ID, chunk index, and offsets are accurately tracked."""
    _, _, linker = evidence_linking_setup

    group = SymptomGroup(
        group_id="SYMGRP-HYDR-01",
        normalized_symptom="hydraulic pressure drop",
        canonical_label="Intermittent pressure drop across primary hydraulic circuit",
        incident_ids=["INC-2024-003"],
        machine_ids=["M-104"],
        recurrence_scope=RecurrenceScope.SINGLETON,
        occurrences_count=1,
        grouping_basis="Test",
        original_symptoms=[{"incident_id": "INC-2024-003", "symptom": "Intermittent pressure drop across primary hydraulic circuit"}],
        maintenance_actions=[],
    )

    result = linker.link_symptom_group(group)
    assert result.status == EvidenceLinkStatus.SUPPORTED
    link = result.supporting_evidence[0]
    assert link.document_id == "DOC-HYDR-01"
    assert link.source_filename == "hydraulic_systems_sop.txt"
    assert "CHK" in link.chunk_id
    assert isinstance(link.chunk_index, int)


def test_duplicate_passage_handling_preserves_provenance(tmp_path: Path):
    """Verify duplicate copies of identical passages are grouped under primary link without losing provenance."""
    db_path = tmp_path / "dup_test.db"
    index = SearchIndex(db_path=db_path)

    # Document 1 (TXT)
    doc1 = make_doc_meta("DOC-DUP-01", "spindle_guide.txt")
    text = "Spindle drive thermal overload. Inspect closed-loop glycol chiller circuit and clean radiator fins."
    index.index_document(doc1, text)

    # Document 2 (PDF duplicate of identical text)
    doc2 = make_doc_meta("DOC-DUP-02", "spindle_guide.pdf")
    index.index_document(doc2, text)

    retriever = EvidenceRetriever(index)
    linker = CrossIncidentEvidenceLinker(retriever)

    group = SymptomGroup(
        group_id="SYMGRP-SPINDLE-01",
        normalized_symptom="spindle chiller glycol thermal trip",
        canonical_label="Spindle drive thermal trip during high-torque milling cycles",
        incident_ids=["INC-2024-005"],
        machine_ids=["M-205"],
        recurrence_scope=RecurrenceScope.SINGLETON,
        occurrences_count=1,
        grouping_basis="Test",
        original_symptoms=[{"incident_id": "INC-2024-005", "symptom": "Spindle drive thermal trip"}],
        maintenance_actions=[],
    )

    result = linker.link_symptom_group(group)
    assert result.status == EvidenceLinkStatus.SUPPORTED
    # Should be deduplicated into 1 primary link
    assert len(result.supporting_evidence) == 1
    primary_link = result.supporting_evidence[0]
    # Provenance of the duplicate copy must be preserved
    assert primary_link.duplicate_source_copies is not None
    assert len(primary_link.duplicate_source_copies) == 1
    dup_copy = primary_link.duplicate_source_copies[0]
    assert dup_copy["document_id"] in ["DOC-DUP-01", "DOC-DUP-02"]


def test_general_guidance_distinguished_from_incident_specific_evidence(evidence_linking_setup):
    """Verify passages mentioning the exact machine are tagged EXACT_MACHINE, while manuals are GENERAL_REFERENCE."""
    _, _, linker = evidence_linking_setup

    group = SymptomGroup(
        group_id="SYMGRP-BEARING-01",
        normalized_symptom="bearing temperature acoustic pitch",
        canonical_label="High bearing temperature and abnormal acoustic pitch",
        incident_ids=["INC-2024-001"],
        machine_ids=["M-104"],
        recurrence_scope=RecurrenceScope.SINGLETON,
        occurrences_count=1,
        grouping_basis="Test",
        original_symptoms=[{"incident_id": "INC-2024-001", "symptom": "High bearing temperature"}],
        maintenance_actions=[],
    )

    result = linker.link_symptom_group(group)
    assert result.status == EvidenceLinkStatus.SUPPORTED

    scopes = {ev.document_id: ev.machine_scope for ev in result.supporting_evidence}
    # DOC-M104-01 specifically mentions M-104
    assert scopes.get("DOC-M104-01") == MachineScope.EXACT_MACHINE
    # DOC-BEAR-01 does not mention any machine ID
    assert scopes.get("DOC-BEAR-01") == MachineScope.GENERAL_REFERENCE


def test_illustrative_measurements_caution_flagged(tmp_path: Path):
    """Verify passages containing illustrative/synthetic test values are classified and flagged with caution."""
    db_path = tmp_path / "illustrative_test.db"
    index = SearchIndex(db_path=db_path)

    doc = make_doc_meta("DOC-ILLUST-01", "test_bench_notes.txt")
    text = (
        "Vibration test bench log.\n"
        "Illustrative test data for harmonic vibration under simulated peak load: recorded 8.4 mm/s RMS."
    )
    index.index_document(doc, text)

    retriever = EvidenceRetriever(index)
    linker = CrossIncidentEvidenceLinker(retriever)

    group = SymptomGroup(
        group_id="SYMGRP-VIB-01",
        normalized_symptom="harmonic load peak recurring under vibration",
        canonical_label="Recurrent harmonic vibration under peak load",
        incident_ids=["INC-2024-002"],
        machine_ids=["M-104"],
        recurrence_scope=RecurrenceScope.SINGLETON,
        occurrences_count=1,
        grouping_basis="Test",
        original_symptoms=[{"incident_id": "INC-2024-002", "symptom": "Recurrent harmonic vibration under peak load"}],
        maintenance_actions=[],
    )

    result = linker.link_symptom_group(group)
    assert result.status == EvidenceLinkStatus.SUPPORTED
    link = result.supporting_evidence[0]
    assert link.context_type == EvidenceContextType.ILLUSTRATIVE_TEST_VALUE
    assert "Caution: Contains illustrative test readings" in link.relevance_reason


def test_unknown_machine_scope_handling(tmp_path: Path):
    """Verify symptom groups with no assigned machine ID classify general evidence as GENERAL_REFERENCE."""
    db_path = tmp_path / "unknown_scope.db"
    index = SearchIndex(db_path=db_path)

    doc = make_doc_meta("DOC-GEN-01", "pneumatic_sop.txt")
    text = (
        "Pneumatic Cylinder Technical Guideline.\n\n"
        "Delayed stroke extension is frequently caused by a worn rod wiper seal or internal proportional solenoid defect. "
        "Perform pneumatic pressure decay test across the actuator cylinder barrel."
    )
    index.index_document(doc, text)

    retriever = EvidenceRetriever(index)
    linker = CrossIncidentEvidenceLinker(retriever)

    group = SymptomGroup(
        group_id="SYMGRP-PNEUM-01",
        normalized_symptom="actuator delayed extension pneumatic stroke",
        canonical_label="Pneumatic actuator delayed stroke extension",
        incident_ids=["INC-2024-006"],
        machine_ids=[], # No machine ID
        recurrence_scope=RecurrenceScope.UNDETERMINED,
        occurrences_count=1,
        grouping_basis="Test",
        original_symptoms=[{"incident_id": "INC-2024-006", "symptom": "Pneumatic actuator delayed stroke extension"}],
        maintenance_actions=[],
    )

    result = linker.link_symptom_group(group)
    assert result.status == EvidenceLinkStatus.SUPPORTED
    link = result.supporting_evidence[0]
    assert link.machine_scope == MachineScope.GENERAL_REFERENCE
    assert link.referenced_machine_id is None


def test_multiple_incidents_in_one_symptom_group(evidence_linking_setup):
    """Verify evidence links correctly associate with all incident IDs in a multi-incident group."""
    _, _, linker = evidence_linking_setup

    group = SymptomGroup(
        group_id="SYMGRP-MULTI-01",
        normalized_symptom="acoustic pitch unusual + bearing high temperature",
        canonical_label="High bearing temperature and abnormal acoustic pitch",
        incident_ids=["INC-2024-001", "INC-2024-010"],
        machine_ids=["M-104", "M-205"],
        recurrence_scope=RecurrenceScope.CROSS_MACHINE,
        occurrences_count=2,
        grouping_basis="Cross-machine recurrence",
        original_symptoms=[
            {"incident_id": "INC-2024-001", "symptom": "High bearing temperature and abnormal acoustic pitch"},
            {"incident_id": "INC-2024-010", "symptom": "Elevated bearing temperature & acoustic pitch"},
        ],
        maintenance_actions=[],
    )

    result = linker.link_symptom_group(group)
    assert result.status == EvidenceLinkStatus.SUPPORTED
    for ev in result.supporting_evidence:
        assert set(ev.related_incident_ids) == {"INC-2024-001", "INC-2024-010"}


def test_evidence_links_do_not_imply_common_root_cause(evidence_linking_setup):
    """Verify evidence links explicitly disclaim root-cause proof."""
    _, _, linker = evidence_linking_setup

    group = SymptomGroup(
        group_id="SYMGRP-BEARING-01",
        normalized_symptom="bearing temperature acoustic pitch",
        canonical_label="High bearing temperature and abnormal acoustic pitch",
        incident_ids=["INC-2024-001"],
        machine_ids=["M-104"],
        recurrence_scope=RecurrenceScope.SINGLETON,
        occurrences_count=1,
        grouping_basis="Test",
        original_symptoms=[{"incident_id": "INC-2024-001", "symptom": "High bearing temperature"}],
        maintenance_actions=[],
    )

    result = linker.link_symptom_group(group)
    assert any("does not prove common root cause" in ev.link_explanation for ev in result.supporting_evidence)
    assert any("do not establish shared root causes" in lim for lim in result.limitations)


def test_empty_comparison_results_and_empty_indexes(tmp_path: Path):
    """Verify empty comparison results and empty search indexes are handled safely without exceptions."""
    empty_index = SearchIndex(db_path=tmp_path / "empty.db")
    empty_retriever = EvidenceRetriever(empty_index)
    linker = CrossIncidentEvidenceLinker(empty_retriever)

    # Empty comparison result
    empty_comp = CrossIncidentComparisonResult(
        comparison_id="CMP-EMPTY",
        compared_incident_ids=[],
        total_incidents_analyzed=0,
        symptom_groups=[],
        timeline_entries=[],
        missing_fields_summary={},
        limitations=[],
        compared_at=current_iso_timestamp(),
    )
    result = linker.link_comparison_evidence(empty_comp)
    assert result.total_groups_evaluated == 0
    assert len(result.group_results) == 0
    assert result.supported_groups_count == 0

    # Non-empty group with empty index
    group = SymptomGroup(
        group_id="SYMGRP-TEST",
        normalized_symptom="test symptom",
        canonical_label="Test symptom",
        incident_ids=["INC-01"],
        machine_ids=["M-104"],
        recurrence_scope=RecurrenceScope.SINGLETON,
        occurrences_count=1,
        grouping_basis="Test",
        original_symptoms=[{"incident_id": "INC-01", "symptom": "Test symptom"}],
        maintenance_actions=[],
    )
    group_res = linker.link_symptom_group(group)
    assert group_res.status == EvidenceLinkStatus.NO_SUPPORTING_EVIDENCE
    assert len(group_res.supporting_evidence) == 0


def test_link_comparison_evidence_against_real_sample_dataset():
    """Verify full comparison evidence linking workflow on genuine sample dataset and real search index."""
    from src.data_loader import load_incidents_data
    df = load_incidents_data()
    comp_engine = CrossIncidentComparisonEngine()
    comp_result = comp_engine.compare_incidents(df)

    real_index = SearchIndex()
    real_retriever = EvidenceRetriever(real_index)
    linker = CrossIncidentEvidenceLinker(real_retriever)

    link_result = linker.link_comparison_evidence(comp_result)
    assert link_result.total_groups_evaluated == len(comp_result.symptom_groups)
    assert len(link_result.group_results) == len(comp_result.symptom_groups)
    assert len(link_result.global_limitations) >= 2
    assert any(lim.code == "NO_SHARED_ROOT_CAUSE" for lim in link_result.global_limitations)

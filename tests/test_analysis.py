"""Automated test suite for GhostFault Milestone 4, 4.1, 4.1.1, and 4.1.2: Substantive Evidence Relevance Validation."""

from pathlib import Path
import pytest

from src.models import (
    DocumentMetadata,
    DocumentStatus,
    AnalysisStatus,
    InvestigativeHypothesis,
    EvidenceReference,
    FaultAnalysisResult,
    EvidenceContextType,
    MachineScope,
    PassageQuality,
    EvidenceCounts,
)
from src.indexer import SearchIndex
from src.retriever import EvidenceRetriever
from src.analyzer import (
    FaultAnalysisEngine,
    extract_sentence_bounded_snippet,
    classify_evidence_context,
    classify_passage_quality,
    normalize_text_for_similarity,
    generate_support_reason,
    FAILURE_PATTERNS,
)


def make_doc_meta(doc_id: str, filename: str) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id,
        original_filename=filename,
        file_type="txt",
        source_type="test",
        storage_reference=f"originals/{doc_id}_{filename}",
        extraction_status="success",
        ingestion_status=DocumentStatus.PROCESSED,
        created_at="2026-10-18T00:00:00+00:00",
        updated_at="2026-10-18T00:00:00+00:00",
        extracted_text_reference=f"extracted/{doc_id}.txt",
        extracted_character_count=100,
    )


@pytest.fixture
def test_analyzer_setup(tmp_path: Path):
    """Sets up an isolated SearchIndex, EvidenceRetriever, and FaultAnalysisEngine."""
    db_file = tmp_path / "test_analysis_index.db"
    index = SearchIndex(db_path=db_file)

    # Ingest document 1: Bearing and acoustic diagnostics for M-104
    doc1 = make_doc_meta("DOC-BEAR01", "bearing_maintenance_guide.txt")
    text1 = (
        "Drive-End Bearing Diagnostic Guide for M-104: High bearing temperature and abnormal acoustic pitch "
        "indicate roller element distress or lubricant breakdown. Check housing bolt torque."
    )
    index.index_document(doc1, text1)

    # Ingest document 2: Hydraulic maintenance bulletin (no machine tag)
    doc2 = make_doc_meta("DOC-HYD01", "hydraulic_bulletin.txt")
    text2 = (
        "Hydraulic Bulletin: Clogged 10-micron return filter cartridge leads to intermittent pressure drop "
        "and cavitation across the primary circuit."
    )
    index.index_document(doc2, text2)

    # Ingest document 3: Vibration report explicitly for a DIFFERENT machine M-999
    doc3 = make_doc_meta("DOC-CROSS01", "cross_unit_vibration.txt")
    text3 = (
        "Special Investigation for M-999: Recurrent harmonic vibration and elastomer coupling insert wear. "
        "Shaft alignment was out of specification."
    )
    index.index_document(doc3, text3)

    retriever = EvidenceRetriever(index)
    engine = FaultAnalysisEngine(retriever)
    return engine, retriever, index


# ==============================================================================
# Milestone 4 Baseline Tests
# ==============================================================================

def test_analysis_valid_incident_bearing(test_analyzer_setup):
    """Verify analysis of a valid bearing incident produces grounded hypotheses."""
    engine, _, _ = test_analyzer_setup
    incident = {
        "incident_id": "INC-2024-001",
        "machine_id": "M-104",
        "date": "2024-01-12",
        "symptom": "High bearing temperature and abnormal acoustic pitch",
        "maintenance_action": "Lubricated drive-end roller bearing and retorqued housing bolts",
        "outcome": "Resolved - Bearing temperature stabilized within nominal 65C operating limit",
    }

    result = engine.analyze_incident(incident)

    assert result.status == AnalysisStatus.COMPLETED
    assert result.incident_id == "INC-2024-001"
    assert len(result.retrieved_evidence) > 0
    assert len(result.hypotheses) > 0

    hyp = result.hypotheses[0]
    assert "Bearing" in hyp.title
    assert len(hyp.supporting_evidence) > 0

    ref = hyp.supporting_evidence[0]
    assert ref.document_id == "DOC-BEAR01"
    assert ref.source_filename == "bearing_maintenance_guide.txt"
    assert "acoustic pitch" in ref.snippet.lower() or "bearing" in ref.snippet.lower()


def test_evidence_references_resolve_to_actual_results(test_analyzer_setup):
    """Ensure every supporting evidence reference in hypotheses matches an actual retrieved chunk."""
    engine, _, _ = test_analyzer_setup
    incident = {
        "incident_id": "INC-2024-003",
        "machine_id": "M-104",
        "date": "2024-03-18",
        "symptom": "Intermittent pressure drop across primary hydraulic circuit",
        "maintenance_action": "Replaced clogged 10-micron hydraulic return filter cartridge",
        "outcome": "Resolved - System line pressure restored to stable 210 bar",
    }

    result = engine.analyze_incident(incident)
    retrieved_chunk_ids = {r.chunk.chunk_id for r in result.retrieved_evidence}

    for hyp in result.hypotheses:
        for ref in hyp.supporting_evidence:
            assert ref.chunk_id in retrieved_chunk_ids, (
                f"Evidence reference {ref.chunk_id} did not originate in retrieved results!"
            )


def test_different_machine_identifiers_handled_safely(test_analyzer_setup):
    """
    Ensure passages mentioning a different machine (e.g., M-999) are NOT silently treated
    as confirmed evidence for the target machine (M-104), but flagged as contradictory/cross-machine.
    """
    engine, _, _ = test_analyzer_setup
    incident = {
        "incident_id": "INC-2024-002",
        "machine_id": "M-104",
        "date": "2024-02-04",
        "symptom": "Recurrent harmonic vibration under peak load",
        "maintenance_action": "Replaced degraded elastomer coupling insert and re-aligned shaft",
        "outcome": "Resolved - Vibration velocity decreased",
    }

    result = engine.analyze_incident(incident)

    assert len(result.hypotheses) > 0
    hyp = result.hypotheses[0]
    assert len(hyp.contradictory_evidence) > 0
    cross_ref = hyp.contradictory_evidence[0]
    assert cross_ref.document_id == "DOC-CROSS01"
    assert "M-999" in cross_ref.relevance_reason
    assert cross_ref.machine_scope == MachineScope.CROSS_MACHINE_CONFLICT


def test_no_evidence_produces_insufficient_evidence_outcome(tmp_path: Path):
    """Ensure an empty search index produces NO_EVIDENCE outcome without fabricated claims."""
    empty_index = SearchIndex(db_path=tmp_path / "empty.db")
    retriever = EvidenceRetriever(empty_index)
    engine = FaultAnalysisEngine(retriever)

    incident = {
        "incident_id": "INC-TEST-999",
        "machine_id": "M-104",
        "symptom": "Unseen anomaly in mysterious subsystem",
        "maintenance_action": "Inspected mystery unit",
        "outcome": "Unknown",
    }

    result = engine.analyze_incident(incident)
    assert result.status == AnalysisStatus.NO_EVIDENCE
    assert len(result.retrieved_evidence) == 0
    assert len(result.hypotheses) == 0
    assert len(result.missing_information) > 0
    assert any("absence of indexed evidence" in lim.lower() for lim in result.limitations)


def test_irrelevant_matches_do_not_become_hypotheses(test_analyzer_setup):
    """Ensure that retrieval of unrelated text does not fabricate a diagnostic hypothesis."""
    engine, _, _ = test_analyzer_setup
    incident = {
        "incident_id": "INC-2024-006",
        "machine_id": "M-310",
        "date": "2024-02-19",
        "symptom": "Pneumatic actuator delayed stroke extension",
        "maintenance_action": "Replaced worn rod wiper seal",
        "outcome": "Resolved",
    }

    result = engine.analyze_incident(incident)
    assert result.status in (AnalysisStatus.INSUFFICIENT_EVIDENCE, AnalysisStatus.NO_EVIDENCE)
    assert len(result.hypotheses) == 0


def test_missing_incident_fields_handled_gracefully(test_analyzer_setup):
    """Verify that an incident missing symptom or machine_id fails cleanly without crashing."""
    engine, _, _ = test_analyzer_setup
    broken_incident = {
        "incident_id": "INC-BROKEN",
    }

    result = engine.analyze_incident(broken_incident)
    assert result.status == AnalysisStatus.FAILED
    assert result.error_message is not None
    assert "missing" in result.error_message.lower()


def test_no_fabricated_provenance(test_analyzer_setup):
    """Verify facts originate directly from incident record and chunks from index."""
    engine, _, _ = test_analyzer_setup
    incident = {
        "incident_id": "INC-2024-001",
        "machine_id": "M-104",
        "symptom": "High bearing temperature and abnormal acoustic pitch",
        "maintenance_action": "Lubricated bearing",
        "outcome": "Resolved",
    }

    result = engine.analyze_incident(incident)
    assert result.incident_facts["incident_id"] == "INC-2024-001"
    assert result.incident_facts["machine_id"] == "M-104"

    for r in result.retrieved_evidence:
        assert r.chunk.document_id.startswith("DOC-")
        assert len(r.chunk.text) > 0


# ==============================================================================
# Milestone 4.1 & 4.1.1 Tests
# ==============================================================================

def test_duplicate_passages_grouped_without_loss_of_provenance():
    """Verify duplicate passage copies from different documents are grouped under primary reference."""
    ref1 = EvidenceReference(
        document_id="DOC-ORIG",
        source_filename="guide.txt",
        chunk_id="DOC-ORIG_CHK0001",
        chunk_index=1,
        snippet="Elevated bearing temperature and acoustic pitch indicate roller element wear.",
        relevance_reason="Symptom correlation",
    )
    ref2 = EvidenceReference(
        document_id="DOC-COPY",
        source_filename="guide_copy.pdf",
        chunk_id="DOC-COPY_CHK0001",
        chunk_index=1,
        snippet="Elevated bearing temperature and acoustic pitch indicate roller element wear.",
        relevance_reason="Symptom correlation",
    )

    deduped = FaultAnalysisEngine.deduplicate_evidence_references([ref1, ref2])
    assert len(deduped) == 1
    primary = deduped[0]
    assert primary.document_id == "DOC-ORIG"
    assert primary.duplicate_source_copies is not None
    assert len(primary.duplicate_source_copies) == 1
    assert primary.duplicate_source_copies[0]["document_id"] == "DOC-COPY"
    assert primary.duplicate_source_copies[0]["source_filename"] == "guide_copy.pdf"


def test_near_duplicate_passages_grouped():
    """Verify near-duplicate passages with minor whitespace or line-break differences are grouped."""
    ref1 = EvidenceReference(
        document_id="DOC-A",
        source_filename="doc_a.txt",
        chunk_id="DOC-A_CHK01",
        chunk_index=0,
        snippet="During inspection the drive end roller bearing showed an elevated operating temperature of 92 C.",
        relevance_reason="Symptom match",
    )
    ref2 = EvidenceReference(
        document_id="DOC-B",
        source_filename="doc_b.pdf",
        chunk_id="DOC-B_CHK01",
        chunk_index=0,
        snippet="During inspection, the drive-end roller bearing showed an elevated operating temperature of 92 C.",
        relevance_reason="Symptom match",
    )

    deduped = FaultAnalysisEngine.deduplicate_evidence_references([ref1, ref2])
    assert len(deduped) == 1
    assert deduped[0].duplicate_source_copies is not None
    assert len(deduped[0].duplicate_source_copies) == 1


def test_sentence_bounded_snippet_extraction():
    """Verify snippets preserve full sentence boundaries without chopping mid-word or fabricating text."""
    long_text = (
        "First complete sentence describing normal operation. "
        "Second sentence detailing hydraulic pressure drop across the 10-micron filter element. "
        "Third sentence noting cavitation sound."
    )
    snippet, is_excerpt = extract_sentence_bounded_snippet(long_text, max_length=150)
    assert snippet.endswith(".")
    assert "First complete sentence describing normal operation." in snippet
    assert "Second sentence detailing hydraulic pressure drop across the 10-micron filter element." in snippet
    assert "Third sentence" not in snippet
    assert not is_excerpt


def test_mid_sentence_chunk_labeled_as_excerpt():
    """Verify that chunks beginning mid-sentence are detected and labeled as excerpts."""
    fragment_text = "cant condition, bearing fit, alignment, and fastener torque before assigning a root cause."
    snippet, is_excerpt = extract_sentence_bounded_snippet(fragment_text)
    assert is_excerpt is True
    assert "cant condition" in snippet


def test_context_classification_illustrative_vs_oem():
    """Verify classification of illustrative test readings vs verified OEM limits."""
    text_illustrative = "Sensor readings: Drive-end bearing temperature: 92 C. Readings are illustrative test values."
    text_oem = "The manufacturer specification sets nominal operating limit at 65 C for drive-end roller bearing."
    text_general = "Inspect bearing grease for thermal oxidation and retorque housing bolts."

    assert classify_evidence_context(text_illustrative) == EvidenceContextType.ILLUSTRATIVE_TEST_VALUE
    assert classify_evidence_context(text_oem) == EvidenceContextType.VERIFIED_OEM_LIMIT
    assert classify_evidence_context(text_general) == EvidenceContextType.GENERAL_REFERENCE


def test_machine_scope_distinction():
    """Verify exact machine scope vs general reference vs cross-machine conflict."""
    text_exact = "Maintenance report for unit M-104 drive assembly."
    text_general = "General procedure for centrifugal pump alignment and bearing lubrication."
    text_conflict = "RCA report for unit M-310 pneumatic cylinder actuator."

    scope_exact, machine_exact = FaultAnalysisEngine.inspect_machine_context("M-104", text_exact)
    assert scope_exact == MachineScope.EXACT_MACHINE
    assert machine_exact == "M-104"

    scope_gen, machine_gen = FaultAnalysisEngine.inspect_machine_context("M-104", text_general)
    assert scope_gen == MachineScope.GENERAL_REFERENCE
    assert machine_gen is None

    scope_conf, machine_conf = FaultAnalysisEngine.inspect_machine_context("M-104", text_conflict)
    assert scope_conf == MachineScope.CROSS_MACHINE_CONFLICT
    assert machine_conf == "M-310"


def test_missing_machine_id_not_treated_as_match(test_analyzer_setup):
    """Verify a passage without a machine ID is marked as GENERAL_REFERENCE, not an exact match."""
    engine, _, _ = test_analyzer_setup
    incident = {
        "incident_id": "INC-2024-003",
        "machine_id": "M-104",
        "date": "2024-03-18",
        "symptom": "Intermittent pressure drop across primary hydraulic circuit",
        "maintenance_action": "Replaced clogged 10-micron hydraulic return filter cartridge",
        "outcome": "Resolved",
    }

    result = engine.analyze_incident(incident)
    assert len(result.hypotheses) > 0
    hyp = result.hypotheses[0]
    for ref in hyp.supporting_evidence:
        if ref.document_id == "DOC-HYD01":
            assert ref.machine_scope == MachineScope.GENERAL_REFERENCE
            assert ref.referenced_machine_id is None
            assert "General reference document" in ref.relevance_reason


def test_support_reason_explains_actual_relationship_not_generic_keywords():
    """Verify generate_support_reason produces grounded explanations, not raw keyword lists."""
    pattern = FAILURE_PATTERNS[0]
    text_mech = "The elevated bearing temperature and abnormal acoustic pitch may indicate increased friction or a lubrication problem."
    reason_mech = generate_support_reason(MachineScope.GENERAL_REFERENCE, EvidenceContextType.GENERAL_REFERENCE, text_mech, pattern, "M-104")

    assert "underlying mechanisms" in reason_mech or "operational distress" in reason_mech
    assert not reason_mech.startswith("Correlates symptom with")

    text_hist = "The previous maintenance entry recorded drive-end roller-bearing lubrication and fastener retorquing."
    reason_hist = generate_support_reason(MachineScope.GENERAL_REFERENCE, EvidenceContextType.GENERAL_REFERENCE, text_hist, pattern, "M-104")
    assert "historical maintenance" in reason_hist or "retorquing" in reason_hist


def test_hypothesis_tentative_qualification_and_no_unsupported_causes(test_analyzer_setup):
    """Verify hypotheses are explicitly qualified as tentative and do not claim verified root causes."""
    engine, _, _ = test_analyzer_setup
    incident = {
        "incident_id": "INC-2024-001",
        "machine_id": "M-104",
        "date": "2024-01-12",
        "symptom": "High bearing temperature and abnormal acoustic pitch",
        "maintenance_action": "Lubricated drive-end roller bearing",
        "outcome": "Resolved",
    }

    result = engine.analyze_incident(incident)
    for hyp in result.hypotheses:
        assert "symptom" in hyp.qualification.lower()
        assert "not confirm" in hyp.qualification.lower() or "requires" in hyp.qualification.lower()


# ==============================================================================
# Milestone 4.1.2 Tests: Substantive Evidence Relevance Validation
# ==============================================================================

def test_search_term_only_passage_classified_and_excluded():
    """Verify that search-term lists are identified and excluded from substantive grounding."""
    text_terms = "Search terms: bearing temperature, elevated temperature, drive-end bearing, roller bearing, abnormal acoustic pitch."
    quality, reason = classify_passage_quality(text_terms)
    assert quality == PassageQuality.SEARCH_TERMS_METADATA
    assert reason is not None
    assert "search terms" in reason.lower() or "keywords" in reason.lower()


def test_metadata_only_passage_classified_and_excluded():
    """Verify administrative document header metadata passages are excluded."""
    text_meta = "GhostFault Maintenance Evidence Record\nDocument ID: GF-EVID-BEARING-001\nRecord type: Test synthetic manual"
    quality, reason = classify_passage_quality(text_meta)
    assert quality == PassageQuality.HEADER_FOOTER_METADATA
    assert "metadata" in reason.lower() or "header" in reason.lower()


def test_substantive_maintenance_passage_accepted():
    """Verify genuine technical maintenance instructions and symptom reports are classified as substantive."""
    text_substantive = (
        "During inspection, the drive-end roller bearing showed an elevated operating temperature of 92 C. "
        "The operator reported an abnormal high-pitched acoustic tone near the drive end."
    )
    quality, reason = classify_passage_quality(text_substantive)
    assert quality == PassageQuality.SUBSTANTIVE_TECHNICAL
    assert reason is None


def test_no_hypothesis_when_only_search_terms_retrieved(tmp_path: Path):
    """Ensure that if retrieval matches ONLY search-term metadata chunks, NO hypothesis is formed."""
    index = SearchIndex(db_path=tmp_path / "terms_only_index.db")
    doc = make_doc_meta("DOC-TERMS01", "terms_index.txt")
    # File containing only search term lists
    text = "Search terms: bearing temperature, elevated temperature, roller bearing, abnormal acoustic pitch, lubrication."
    index.index_document(doc, text)

    retriever = EvidenceRetriever(index)
    engine = FaultAnalysisEngine(retriever)

    incident = {
        "incident_id": "INC-2024-001",
        "machine_id": "M-104",
        "symptom": "High bearing temperature and abnormal acoustic pitch",
        "maintenance_action": "Lubricated drive-end roller bearing",
        "outcome": "Resolved",
    }

    result = engine.analyze_incident(incident)

    # Must be INSUFFICIENT_EVIDENCE with 0 hypotheses
    assert result.status == AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert len(result.hypotheses) == 0
    # Must be recorded under excluded_evidence
    assert len(result.excluded_evidence) == 1
    assert result.excluded_evidence[0].document_id == "DOC-TERMS01"
    assert result.excluded_evidence[0].quality == PassageQuality.SEARCH_TERMS_METADATA


def test_accurate_pipeline_counts_derivation(tmp_path: Path):
    """Verify counts accurately derive raw, unique, substantive, excluded, and supporting quantities."""
    index = SearchIndex(db_path=tmp_path / "counts_index.db")

    # 1. Substantive chunk
    doc1 = make_doc_meta("DOC-S1", "manual1.txt")
    text1 = "High bearing temperature indicates drive-end roller bearing fatigue or severe lubricant oxidation."
    index.index_document(doc1, text1)

    # 2. Search terms chunk
    doc2 = make_doc_meta("DOC-T1", "manual2.txt")
    text2 = "Search terms: bearing temperature, elevated temperature, acoustic pitch."
    index.index_document(doc2, text2)

    retriever = EvidenceRetriever(index)
    engine = FaultAnalysisEngine(retriever)

    incident = {
        "incident_id": "INC-2024-001",
        "machine_id": "M-104",
        "symptom": "High bearing temperature and abnormal acoustic pitch",
        "maintenance_action": "Lubricated drive-end roller bearing",
        "outcome": "Resolved",
    }

    result = engine.analyze_incident(incident)
    counts = result.evidence_counts
    assert counts is not None
    assert counts.raw_retrieved_count >= 2
    assert counts.unique_deduplicated_count >= 2
    assert counts.substantive_retained_count == 1
    assert counts.excluded_low_value_count == 1
    assert counts.supporting_hypotheses_count == 1

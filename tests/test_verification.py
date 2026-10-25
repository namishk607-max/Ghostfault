"""Focused automated test suite for Milestone 6.1: Deterministic Evidence Verification.

Covers:
1. Relevant supporting evidence -> SUPPORTED
2. Explicitly contradictory evidence -> CONTRADICTED
3. Unrelated or insufficient evidence -> INCONCLUSIVE
4. No retrieved evidence -> INCONCLUSIVE
5. Missing or incomplete provenance -> INCONCLUSIVE
6. Keyword overlap without actual evidential support -> INCONCLUSIVE
"""

import pytest
from src.models import (
    InvestigativeHypothesis,
    EvidenceReference,
    VerificationStatus,
    EvidenceStrength,
    PassageQuality,
    EvidenceContextType,
    MachineScope,
)
from src.analyzer import (
    verify_hypothesis_evidence,
    evaluate_evidence_strength,
    derive_missing_evidence_guidance,
    FAILURE_PATTERNS,
)


def test_verification_relevant_supporting_evidence_exact_machine():
    """Test 1a: Relevant supporting evidence with exact machine match returns SUPPORTED."""
    pattern = FAILURE_PATTERNS[0]  # BEARING_DEGRADATION
    ref = EvidenceReference(
        document_id="DOC-001",
        source_filename="m104_bearing_log.txt",
        chunk_id="chk_001",
        chunk_index=0,
        snippet="Drive-End Bearing Diagnostic Guide for M-104: High bearing temperature and abnormal acoustic pitch indicate roller element distress.",
        relevance_reason="Direct incident record for M-104.",
        machine_scope=MachineScope.EXACT_MACHINE,
        context_type=EvidenceContextType.GENERAL_REFERENCE,
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
        referenced_machine_id="M-104",
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Preliminary hypothesis",
        supporting_evidence=[ref],
        contradictory_evidence=[],
        suggested_checks=["Check vibration spectrum"],
    )

    status, explanation = verify_hypothesis_evidence(hyp, incident_machine_id="M-104", pattern=pattern)

    assert status == VerificationStatus.SUPPORTED
    assert "Supported:" in explanation
    assert "M-104" in explanation
    assert "m104_bearing_log.txt" in explanation


def test_verification_relevant_supporting_evidence_corroborating_mechanism():
    """Test 1b: Relevant supporting evidence with corroborated technical mechanism returns SUPPORTED."""
    pattern = FAILURE_PATTERNS[0]  # BEARING_DEGRADATION ("mechanism_terms": ["friction", "lubrication", "fatigue", "wear", ...])
    ref = EvidenceReference(
        document_id="DOC-GEN-01",
        source_filename="general_bearing_manual.pdf",
        chunk_id="chk_002",
        chunk_index=2,
        snippet="Rolling element surface fatigue and lubrication breakdown lead to elevated temperature and acoustic pitch.",
        relevance_reason="General technical reference.",
        machine_scope=MachineScope.GENERAL_REFERENCE,
        context_type=EvidenceContextType.GENERAL_REFERENCE,
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Preliminary hypothesis",
        supporting_evidence=[ref],
        contradictory_evidence=[],
        suggested_checks=["Sample lubricant"],
    )

    status, explanation = verify_hypothesis_evidence(hyp, incident_machine_id="M-104", pattern=pattern)

    assert status == VerificationStatus.SUPPORTED
    assert "Supported:" in explanation
    assert "general_bearing_manual.pdf" in explanation
    assert "Chunk #2" in explanation


def test_verification_explicitly_contradictory_evidence_cross_machine():
    """Test 2a: Substantive contradictory evidence with cross-machine conflict returns CONTRADICTED."""
    pattern = FAILURE_PATTERNS[1]  # COUPLING_MISALIGNMENT
    ref_contra = EvidenceReference(
        document_id="DOC-999",
        source_filename="cross_unit_vibration.txt",
        chunk_id="chk_cross",
        chunk_index=1,
        snippet="Harmonic vibration and coupling wear investigated specifically on unit M-999.",
        relevance_reason="Passage explicitly references a different machine (M-999).",
        machine_scope=MachineScope.CROSS_MACHINE_CONFLICT,
        context_type=EvidenceContextType.GENERAL_REFERENCE,
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
        referenced_machine_id="M-999",
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Weak cross-machine hypothesis",
        supporting_evidence=[],
        contradictory_evidence=[ref_contra],
        suggested_checks=["Check alignment"],
    )

    status, explanation = verify_hypothesis_evidence(hyp, incident_machine_id="M-104", pattern=pattern)

    assert status == VerificationStatus.CONTRADICTED
    assert "Contradicted:" in explanation
    assert "M-999" in explanation
    assert "cross_unit_vibration.txt" in explanation


def test_verification_explicitly_contradictory_evidence_nominal_operation():
    """Test 2b: Evidence indicating nominal operation or no defect found returns CONTRADICTED."""
    pattern = FAILURE_PATTERNS[2]  # HYDRAULIC_RESTRICTION
    ref_nominal = EvidenceReference(
        document_id="DOC-HYD-NORM",
        source_filename="inspection_report.txt",
        chunk_id="chk_norm",
        chunk_index=0,
        snippet="Hydraulic circuit pressure is within nominal limit. Filter inspection passed inspection; no defect found.",
        relevance_reason="Conflicting normal operational reading.",
        machine_scope=MachineScope.GENERAL_REFERENCE,
        context_type=EvidenceContextType.GENERAL_REFERENCE,
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Contradicted hypothesis",
        supporting_evidence=[],
        contradictory_evidence=[ref_nominal],
        suggested_checks=["Re-verify pressure"],
    )

    status, explanation = verify_hypothesis_evidence(hyp, incident_machine_id="M-104", pattern=pattern)

    assert status == VerificationStatus.CONTRADICTED
    assert "Contradicted:" in explanation
    assert "nominal" in explanation or "no defect found" in explanation


def test_verification_unrelated_or_insufficient_evidence():
    """Test 3: Passages that are metadata or low substance return INCONCLUSIVE."""
    pattern = FAILURE_PATTERNS[3]
    ref_meta = EvidenceReference(
        document_id="DOC-META",
        source_filename="terms.txt",
        chunk_id="chk_meta",
        chunk_index=0,
        snippet="thermal chiller cooling coolant spindle glycol",
        relevance_reason="Excluded from hypothesis grounding",
        machine_scope=MachineScope.GENERAL_REFERENCE,
        context_type=EvidenceContextType.GENERAL_REFERENCE,
        quality=PassageQuality.SEARCH_TERMS_METADATA,
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Hypothesis with non-substantive evidence",
        supporting_evidence=[ref_meta],
        contradictory_evidence=[],
        suggested_checks=["Check chiller"],
    )

    status, explanation = verify_hypothesis_evidence(hyp, incident_machine_id="M-104", pattern=pattern)

    assert status == VerificationStatus.INCONCLUSIVE
    assert "Inconclusive:" in explanation
    assert "metadata" in explanation or "low-substance" in explanation


def test_verification_no_retrieved_evidence():
    """Test 4: Hypothesis with zero retrieved evidence returns INCONCLUSIVE."""
    pattern = FAILURE_PATTERNS[0]
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Ungrounded hypothesis",
        supporting_evidence=[],
        contradictory_evidence=[],
        suggested_checks=["Perform inspection"],
    )

    status, explanation = verify_hypothesis_evidence(hyp, incident_machine_id="M-104", pattern=pattern)

    assert status == VerificationStatus.INCONCLUSIVE
    assert "Inconclusive:" in explanation
    assert "No retrieved evidence" in explanation


def test_verification_missing_or_incomplete_provenance():
    """Test 5: Evidence with missing document_id or chunk_id returns INCONCLUSIVE."""
    pattern = FAILURE_PATTERNS[0]
    ref_no_prov = EvidenceReference(
        document_id="",  # Missing document_id
        source_filename="",
        chunk_id="",     # Missing chunk_id
        chunk_index=0,
        snippet="Some bearing wear mechanism described without source tracking.",
        relevance_reason="Unknown source",
        machine_scope=MachineScope.GENERAL_REFERENCE,
        context_type=EvidenceContextType.GENERAL_REFERENCE,
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Unprovenanced evidence",
        supporting_evidence=[ref_no_prov],
        contradictory_evidence=[],
        suggested_checks=["Check bearing"],
    )

    status, explanation = verify_hypothesis_evidence(hyp, incident_machine_id="M-104", pattern=pattern)

    assert status == VerificationStatus.INCONCLUSIVE
    assert "Inconclusive:" in explanation
    assert "provenance" in explanation


def test_verification_keyword_overlap_without_evidential_support():
    """Test 6: Incidental keyword overlap without technical failure mechanism or machine match returns INCONCLUSIVE."""
    pattern = FAILURE_PATTERNS[0]  # BEARING_DEGRADATION
    # Text mentions keywords 'bearing' and 'acoustic' in an administrative or passing context without failure mechanism
    ref_overlap = EvidenceReference(
        document_id="DOC-ADMIN-01",
        source_filename="meeting_minutes.txt",
        chunk_id="chk_overlap",
        chunk_index=1,
        snippet="During the quarterly review, the committee discussed the procurement of bearing spares and acoustic sensors for factory storage.",
        relevance_reason="General mention of keywords.",
        machine_scope=MachineScope.GENERAL_REFERENCE,
        context_type=EvidenceContextType.GENERAL_REFERENCE,
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Preliminary hypothesis",
        supporting_evidence=[ref_overlap],
        contradictory_evidence=[],
        suggested_checks=["Check bearing"],
    )

    status, explanation = verify_hypothesis_evidence(hyp, incident_machine_id="M-104", pattern=pattern)

    assert status == VerificationStatus.INCONCLUSIVE
    assert "Inconclusive:" in explanation
    assert "keywords" in explanation or "lacks specific failure mechanism" in explanation


# ==============================================================================
# Milestone 6.2 Tests: Qualitative Evidence Strength & Missing Evidence Guidance
# ==============================================================================

def test_evidence_strength_multiple_independent_passages():
    """Test 1: Multiple independent supporting passages produce STRONG evidence strength."""
    pattern = FAILURE_PATTERNS[0]
    ref1 = EvidenceReference(
        document_id="DOC-INDEP-01",
        source_filename="bearing_oem_manual.pdf",
        chunk_id="chk_01",
        chunk_index=0,
        snippet="Drive-end bearing wear causes friction and abnormal acoustic pitch.",
        relevance_reason="Direct mechanism guidance.",
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
    )
    ref2 = EvidenceReference(
        document_id="DOC-INDEP-02",
        source_filename="plant_maintenance_log_2023.txt",
        chunk_id="chk_02",
        chunk_index=3,
        snippet="M-104 bearing lubrication breakdown noted during scheduled overhaul.",
        relevance_reason="Historical incident record.",
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
        machine_scope=MachineScope.EXACT_MACHINE,
        referenced_machine_id="M-104",
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Qualified hypothesis",
        supporting_evidence=[ref1, ref2],
        contradictory_evidence=[],
        suggested_checks=["Check vibration"],
    )

    strength, explanation = evaluate_evidence_strength(
        hyp,
        verification_status=VerificationStatus.SUPPORTED,
        incident_machine_id="M-104",
    )

    assert strength == EvidenceStrength.STRONG
    assert "Multiple independent" in explanation
    assert "2 distinct sources" in explanation


def test_evidence_strength_repeated_or_duplicate_chunks_not_falsely_counted():
    """Test 2: Repeated or duplicate chunks from the same document are NOT counted as independent STRONG corroboration."""
    pattern = FAILURE_PATTERNS[0]
    # Two chunks from the same document DOC-SINGLE-01
    ref1 = EvidenceReference(
        document_id="DOC-SINGLE-01",
        source_filename="same_manual.pdf",
        chunk_id="chk_10",
        chunk_index=1,
        snippet="Bearing acoustic pitch indicates fatigue.",
        relevance_reason="Manual section 1.",
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
    )
    ref2 = EvidenceReference(
        document_id="DOC-SINGLE-01",
        source_filename="same_manual.pdf",
        chunk_id="chk_11",
        chunk_index=2,
        snippet="Lubrication breakdown causes friction in roller bearings.",
        relevance_reason="Manual section 2.",
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Qualified hypothesis",
        supporting_evidence=[ref1, ref2],
        contradictory_evidence=[],
        suggested_checks=["Inspect roller"],
    )

    strength, explanation = evaluate_evidence_strength(
        hyp,
        verification_status=VerificationStatus.SUPPORTED,
        incident_machine_id="M-104",
    )

    assert strength == EvidenceStrength.LIMITED
    assert "single source document" in explanation or "lacks independent multi-source corroboration" in explanation


def test_evidence_strength_with_contradictory_evidence():
    """Test 3: Contradictory evidence results in LIMITED strength and explicitly notes limitations."""
    pattern = FAILURE_PATTERNS[1]
    ref_contra = EvidenceReference(
        document_id="DOC-999",
        source_filename="cross_machine.txt",
        chunk_id="chk_cross",
        chunk_index=0,
        snippet="Vibration analyzed for machine M-999.",
        relevance_reason="Cross-machine conflict.",
        machine_scope=MachineScope.CROSS_MACHINE_CONFLICT,
        referenced_machine_id="M-999",
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Contradicted hypothesis",
        supporting_evidence=[],
        contradictory_evidence=[ref_contra],
        suggested_checks=["Check alignment"],
    )

    strength, explanation = evaluate_evidence_strength(
        hyp,
        verification_status=VerificationStatus.CONTRADICTED,
        incident_machine_id="M-104",
    )

    assert strength == EvidenceStrength.LIMITED
    assert "Contradictory evidence detected" in explanation


def test_evidence_strength_insufficient_or_missing_evidence():
    """Test 4: When evidence is inconclusive or missing, evidence strength is INSUFFICIENT."""
    pattern = FAILURE_PATTERNS[0]
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Ungrounded",
        supporting_evidence=[],
        contradictory_evidence=[],
        suggested_checks=[],
    )

    strength, explanation = evaluate_evidence_strength(
        hyp,
        verification_status=VerificationStatus.INCONCLUSIVE,
        incident_machine_id="M-104",
    )

    assert strength == EvidenceStrength.INSUFFICIENT
    assert "insufficient" in explanation.lower()


def test_missing_evidence_guidance_grounded_in_available_data():
    """Test 5: Missing evidence guidance suggests specific gap derived from hypothesis when not in records."""
    pattern = FAILURE_PATTERNS[0]  # BEARING_DEGRADATION
    incident_record = {
        "incident_id": "INC-001",
        "machine_id": "M-104",
        "symptom": "High bearing temperature and abnormal acoustic pitch",
        "maintenance_action": "Lubricated bearing",
    }
    ref = EvidenceReference(
        document_id="DOC-01",
        source_filename="guide.txt",
        chunk_id="chk_01",
        chunk_index=0,
        snippet="Bearing wear causes acoustic pitch.",
        relevance_reason="Relevant",
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Preliminary",
        supporting_evidence=[ref],
        contradictory_evidence=[],
        suggested_checks=[],
    )

    guidance = derive_missing_evidence_guidance(hyp, incident_record, pattern)

    assert "not found in the available records" in guidance
    assert "bearing temperature measurement log" in guidance or "time-series" in guidance


def test_missing_evidence_guidance_no_unsupported_claim():
    """Test 6: When gap is already covered in existing snippets, does not make unsupported claim."""
    pattern = FAILURE_PATTERNS[0]
    incident_record = {
        "incident_id": "INC-001",
        "machine_id": "M-104",
        "symptom": "High bearing temperature",
        "maintenance_action": "Inspected bearing",
    }
    # Snippet already includes temperature trend log over time and lubricant grease lab report
    ref = EvidenceReference(
        document_id="DOC-01",
        source_filename="comprehensive_report.txt",
        chunk_id="chk_01",
        chunk_index=0,
        snippet="Continuous temperature log trend over time verified. Laboratory lubricant grease analysis confirmed particle contamination.",
        relevance_reason="Comprehensive documentation",
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Preliminary",
        supporting_evidence=[ref],
        contradictory_evidence=[],
        suggested_checks=[],
    )

    guidance = derive_missing_evidence_guidance(hyp, incident_record, pattern)

    # When all tracked gaps are present in the record, fall back cautiously
    assert guidance == "No specific evidence gap identified."


def test_m6_1_verification_statuses_preserved():
    """Test 7: M6.1 verification statuses (SUPPORTED, CONTRADICTED, INCONCLUSIVE) and provenance are fully preserved."""
    pattern = FAILURE_PATTERNS[0]
    ref = EvidenceReference(
        document_id="DOC-PROV-100",
        source_filename="provenance_source.txt",
        chunk_id="chk_prov_100",
        chunk_index=4,
        snippet="Bearing roller element surface fatigue causes wear.",
        relevance_reason="Direct mechanism evidence.",
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
    )
    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Preliminary",
        supporting_evidence=[ref],
        contradictory_evidence=[],
        suggested_checks=["Check bearing"],
    )

    status, explanation = verify_hypothesis_evidence(hyp, incident_machine_id="M-104", pattern=pattern)
    strength, str_explanation = evaluate_evidence_strength(hyp, verification_status=status, incident_machine_id="M-104")

    # M6.1 status remains SUPPORTED
    assert status == VerificationStatus.SUPPORTED
    assert "DOC-PROV-100" in ref.document_id
    assert "chk_prov_100" in ref.chunk_id
    assert ref.chunk_index == 4
    # M6.2 strength is evaluated alongside without overriding
    assert strength == EvidenceStrength.LIMITED
    assert "single documented source" in str_explanation

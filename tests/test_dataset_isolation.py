"""Milestone 7.4 Comprehensive Cross-Dataset Protection & Isolation Test Suite.

Proves:
1. Two datasets can be created and persist after reload.
2. Blank names are rejected.
3. Duplicate dataset IDs cannot be introduced.
4. The Default Dataset and its existing records remain intact.
5. Documents uploaded to Dataset A receive Dataset A's ID.
6. Documents uploaded to Dataset B receive Dataset B's ID.
7. Evidence belonging only to Dataset A cannot appear in Dataset B's search results.
8. Evidence from Dataset A cannot support a fault hypothesis in Dataset B.
9. Evidence verification respects dataset scope.
10. Cross-incident comparisons do not mix incidents from different datasets.
11. Cross-incident evidence linking respects dataset scope.
12. Empty datasets work without crashing.
13. Invalid active dataset IDs are handled safely.
14. Existing M1–M7.3 functionality remains intact.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.models import (
    Dataset,
    DatasetOwnershipError,
    DocumentMetadata,
    DocumentStatus,
    InvestigativeHypothesis,
    EvidenceReference,
    VerificationStatus,
    EvidenceStrength,
    PassageQuality,
    EvidenceContextType,
    MachineScope,
    AnalysisStatus,
    RecurrenceScope,
    EvidenceLinkStatus,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_DESCRIPTION,
)
from src.storage import EvidenceStore
from src.ingestion import IngestionService
from src.indexer import SearchIndex
from src.retriever import EvidenceRetriever
from src.analyzer import (
    FaultAnalysisEngine,
    evaluate_evidence_strength,
    derive_missing_evidence_guidance,
    verify_hypothesis_evidence,
    FAILURE_PATTERNS,
)
from src.cross_incident import CrossIncidentComparisonEngine
from src.evidence_linker import CrossIncidentEvidenceLinker
from src.hub import (
    create_dataset_entry,
    resolve_active_dataset,
    set_active_dataset_id,
    compute_workspace_summary,
    compute_dataset_registry,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def isolated_env(tmp_path: Path):
    """Set up isolated storage, index, and services under tmp_path."""
    evidence_dir = tmp_path / "evidence"
    index_db_path = tmp_path / "index" / "evidence_index.db"

    store = EvidenceStore(base_dir=evidence_dir)
    store.ensure_default_dataset()

    index = SearchIndex(db_path=index_db_path)
    ingestion_service = IngestionService(store=store)
    retriever = EvidenceRetriever(index=index, store=store)
    analyzer = FaultAnalysisEngine(retriever=retriever)
    comparison_engine = CrossIncidentComparisonEngine(store=store)
    evidence_linker = CrossIncidentEvidenceLinker(retriever=retriever)

    return {
        "store": store,
        "index": index,
        "ingestion": ingestion_service,
        "retriever": retriever,
        "analyzer": analyzer,
        "comparison": comparison_engine,
        "linker": evidence_linker,
        "evidence_dir": evidence_dir,
    }


# ==============================================================================
# Tests 1-4: Dataset Creation, Persistence, and Validation
# ==============================================================================


def test_1_two_datasets_created_and_persist_after_reload(isolated_env):
    """Test 1: Two datasets can be created and persist across store reload."""
    store: EvidenceStore = isolated_env["store"]

    # Create Dataset A
    success_a, ds_a, err_a = create_dataset_entry(
        store,
        name="Milling Line 4 Maintenance",
        description="Records for Milling Line 4 spindle overhauls.",
    )
    assert success_a is True
    assert ds_a is not None
    assert err_a is None
    assert ds_a.dataset_id.startswith("DATASET-")

    # Create Dataset B
    success_b, ds_b, err_b = create_dataset_entry(
        store,
        name="Packaging Facility Heat Exchangers",
        description="Cooling loop inspection and descaling logs.",
    )
    assert success_b is True
    assert ds_b is not None
    assert err_b is None
    assert ds_b.dataset_id != ds_a.dataset_id

    # Simulate fresh session reload from disk
    reloaded_store = EvidenceStore(base_dir=isolated_env["evidence_dir"])
    datasets = reloaded_store.list_datasets()
    ds_map = {d.dataset_id: d for d in datasets}

    assert ds_a.dataset_id in ds_map
    assert ds_b.dataset_id in ds_map
    assert ds_map[ds_a.dataset_id].name == "Milling Line 4 Maintenance"
    assert ds_map[ds_b.dataset_id].name == "Packaging Facility Heat Exchangers"
    assert ds_map[ds_a.dataset_id].description == "Records for Milling Line 4 spindle overhauls."


def test_2_blank_names_are_rejected(isolated_env):
    """Test 2: Blank or invalid dataset names are rejected with clear errors."""
    store: EvidenceStore = isolated_env["store"]

    # Test empty string
    success, ds, err = create_dataset_entry(store, name="")
    assert success is False
    assert ds is None
    assert "blank" in err.lower()

    # Test whitespace-only string
    success, ds, err = create_dataset_entry(store, name="    \t  \n  ")
    assert success is False
    assert ds is None
    assert "blank" in err.lower()

    # Test single character (below 2 chars limit)
    success, ds, err = create_dataset_entry(store, name="X")
    assert success is False
    assert ds is None
    assert "between 2 and 100 characters" in err

    # Test excessively long name (> 100 chars)
    long_name = "A" * 105
    success, ds, err = create_dataset_entry(store, name=long_name)
    assert success is False
    assert ds is None
    assert "between 2 and 100 characters" in err


def test_3_duplicate_dataset_ids_cannot_be_introduced(isolated_env):
    """Test 3: Duplicate dataset IDs cannot be created or introduced."""
    store: EvidenceStore = isolated_env["store"]

    # Create first dataset with explicit ID
    success_1, ds_1, _ = create_dataset_entry(
        store,
        name="First Dataset",
        dataset_id="DATASET-UNIQUE-01",
    )
    assert success_1 is True

    # Attempt to create second dataset with same ID
    success_2, ds_2, err_2 = create_dataset_entry(
        store,
        name="Second Dataset Different Name",
        dataset_id="DATASET-UNIQUE-01",
    )
    assert success_2 is False
    assert ds_2 is None
    assert "already exists" in err_2.lower()

    # Ensure registry contains only one instance
    matches = [d for d in store.list_datasets() if d.dataset_id == "DATASET-UNIQUE-01"]
    assert len(matches) == 1
    assert matches[0].name == "First Dataset"


def test_4_default_dataset_and_existing_records_remain_intact(isolated_env):
    """Test 4: Creating new datasets does not overwrite or destroy Default Dataset."""
    store: EvidenceStore = isolated_env["store"]

    # Verify default dataset exists
    default_ds = store.get_dataset(DEFAULT_DATASET_ID)
    assert default_ds is not None
    assert default_ds.name == DEFAULT_DATASET_NAME

    # Add custom datasets
    create_dataset_entry(store, name="Project Alpha")
    create_dataset_entry(store, name="Project Beta")

    # Verify default dataset remains intact
    reloaded_default = store.get_dataset(DEFAULT_DATASET_ID)
    assert reloaded_default is not None
    assert reloaded_default.dataset_id == DEFAULT_DATASET_ID
    assert reloaded_default.name == DEFAULT_DATASET_NAME
    assert reloaded_default.description == DEFAULT_DATASET_DESCRIPTION

    # Ensure list_datasets contains default + 2 new
    all_ds = store.list_datasets()
    assert len(all_ds) == 3
    assert any(d.dataset_id == DEFAULT_DATASET_ID for d in all_ds)


# ==============================================================================
# Tests 5-6: Ingestion Dataset Association
# ==============================================================================


def test_5_and_6_documents_uploaded_to_dataset_receive_correct_id(isolated_env):
    """Test 5 & 6: Documents uploaded to Dataset A/B receive the respective dataset's ID."""
    store: EvidenceStore = isolated_env["store"]
    ingest: IngestionService = isolated_env["ingestion"]

    # Create Dataset A and Dataset B
    _, ds_a, _ = create_dataset_entry(store, name="Turbine Facility Alpha")
    _, ds_b, _ = create_dataset_entry(store, name="Hydraulics Plant Beta")

    # Ingest document into Dataset A
    content_a = b"Drive-end bearing lubrication nominal limits and acoustic pitch inspection."
    doc_a = ingest.ingest_document(
        filename="manual_alpha.txt",
        content=content_a,
        dataset_id=ds_a.dataset_id,
    )
    assert doc_a.ingestion_status == DocumentStatus.PROCESSED
    assert doc_a.dataset_id == ds_a.dataset_id

    # Ingest document into Dataset B
    content_b = b"Hydraulic circuit pressure differential and return line filter replacement."
    doc_b = ingest.ingest_document(
        filename="manual_beta.txt",
        content=content_b,
        dataset_id=ds_b.dataset_id,
    )
    assert doc_b.ingestion_status == DocumentStatus.PROCESSED
    assert doc_b.dataset_id == ds_b.dataset_id

    # Verify persistence in catalog
    catalog = store.read_catalog()
    cat_map = {d.document_id: d for d in catalog}

    assert cat_map[doc_a.document_id].dataset_id == ds_a.dataset_id
    assert cat_map[doc_b.document_id].dataset_id == ds_b.dataset_id


# ==============================================================================
# Tests 7: FTS5 Dataset-Scoped Search Isolation
# ==============================================================================


def test_7_evidence_from_dataset_a_cannot_appear_in_dataset_b_search(isolated_env):
    """Test 7: Evidence from Dataset A cannot appear in Dataset B's search results."""
    store: EvidenceStore = isolated_env["store"]
    ingest: IngestionService = isolated_env["ingestion"]
    index: SearchIndex = isolated_env["index"]
    retriever: EvidenceRetriever = isolated_env["retriever"]

    _, ds_a, _ = create_dataset_entry(store, name="Dataset A Facility")
    _, ds_b, _ = create_dataset_entry(store, name="Dataset B Facility")

    # Ingest doc in Dataset A
    doc_a = ingest.ingest_document(
        filename="bearing_spec.txt",
        content=b"Spindle roller bearing high temperature and acoustic pitch wear indicators.",
        dataset_id=ds_a.dataset_id,
    )
    text_a = store.get_extracted_text(doc_a)
    index.index_document(doc_a, text_a, store=store)

    # Ingest doc in Dataset B
    doc_b = ingest.ingest_document(
        filename="hydraulic_spec.txt",
        content=b"Hydraulic circuit pressure drop and return line filter cartridge replacement.",
        dataset_id=ds_b.dataset_id,
    )
    text_b = store.get_extracted_text(doc_b)
    index.index_document(doc_b, text_b, store=store)

    valid_ids = [d.dataset_id for d in store.list_datasets()]

    # Query 'bearing' under Dataset A -> should return result from Dataset A
    results_a = retriever.search(
        query="bearing",
        dataset_id=ds_a.dataset_id,
        valid_dataset_ids=valid_ids,
    )
    assert len(results_a) > 0
    assert all(r.chunk.dataset_id == ds_a.dataset_id for r in results_a)
    assert all(r.chunk.document_id == doc_a.document_id for r in results_a)

    # Query 'bearing' under Dataset B -> MUST return ZERO results (strict isolation!)
    results_b = retriever.search(
        query="bearing",
        dataset_id=ds_b.dataset_id,
        valid_dataset_ids=valid_ids,
    )
    assert len(results_b) == 0

    # Query 'hydraulic' under Dataset A -> MUST return ZERO results
    results_hyd_a = retriever.search(
        query="hydraulic",
        dataset_id=ds_a.dataset_id,
        valid_dataset_ids=valid_ids,
    )
    assert len(results_hyd_a) == 0

    # Query 'hydraulic' under Dataset B -> should return result from Dataset B
    results_hyd_b = retriever.search(
        query="hydraulic",
        dataset_id=ds_b.dataset_id,
        valid_dataset_ids=valid_ids,
    )
    assert len(results_hyd_b) > 0
    assert all(r.chunk.dataset_id == ds_b.dataset_id for r in results_hyd_b)


# ==============================================================================
# Tests 8-9: Fault Analysis and Verification Scope Isolation
# ==============================================================================


def test_8_evidence_from_dataset_a_cannot_support_hypothesis_in_dataset_b(isolated_env):
    """Test 8: Evidence from Dataset A cannot support a fault hypothesis in Dataset B."""
    store: EvidenceStore = isolated_env["store"]
    ingest: IngestionService = isolated_env["ingestion"]
    index: SearchIndex = isolated_env["index"]
    analyzer: FaultAnalysisEngine = isolated_env["analyzer"]
    retriever: EvidenceRetriever = isolated_env["retriever"]

    _, ds_a, _ = create_dataset_entry(store, name="Bearing Dataset A")
    _, ds_b, _ = create_dataset_entry(store, name="Hydraulic Dataset B")

    # Ingest bearing documentation ONLY into Dataset A
    doc_a = ingest.ingest_document(
        filename="bearing_oem.txt",
        content=(
            b"Drive-End Bearing Diagnostic Guide for M-104: High bearing temperature "
            b"accompanied by abnormal acoustic pitch indicates roller element surface distress "
            b"and lubrication breakdown."
        ),
        dataset_id=ds_a.dataset_id,
    )
    index.index_document(doc_a, store.get_extracted_text(doc_a), store=store)

    # Incident with bearing symptoms, but belonging to Dataset B!
    incident_in_b = {
        "incident_id": "INC-B-001",
        "machine_id": "M-104",
        "symptom": "Elevated bearing temperature and abnormal acoustic pitch",
        "maintenance_action": "Inspected bearing housing",
        "outcome": "Pending diagnosis",
        "dataset_id": ds_b.dataset_id,
    }

    evidence_from_a = retriever.search(
        "bearing temperature",
        dataset_id=ds_a.dataset_id,
    )
    with pytest.raises(DatasetOwnershipError, match="untagged, foreign"):
        analyzer.analyze_incident(
            incident_in_b,
            evidence_results=evidence_from_a,
            dataset_id=ds_b.dataset_id,
        )

    # Run fault analysis scoped to Dataset B
    analysis_res = analyzer.analyze_incident(incident_in_b, dataset_id=ds_b.dataset_id)

    # Result in Dataset B MUST NOT use Dataset A's evidence
    assert analysis_res.status in {AnalysisStatus.NO_EVIDENCE, AnalysisStatus.INSUFFICIENT_EVIDENCE}
    assert len(analysis_res.retrieved_evidence) == 0
    for hyp in analysis_res.hypotheses:
        assert len(hyp.supporting_evidence) == 0


def test_9_evidence_verification_respects_dataset_scope(isolated_env):
    """Test 9: Evidence verification rejects evidence outside the active dataset."""
    _, ds_a, _ = create_dataset_entry(isolated_env["store"], name="Dataset A")
    _, ds_b, _ = create_dataset_entry(isolated_env["store"], name="Dataset B")

    pattern = FAILURE_PATTERNS[0]  # Bearing Degradation
    ref_from_a = EvidenceReference(
        document_id="DOC-A-01",
        source_filename="bearing_log.txt",
        chunk_id="chk_001",
        chunk_index=0,
        snippet="High bearing temperature and abnormal acoustic pitch on drive unit.",
        relevance_reason="Direct bearing maintenance record.",
        machine_scope=MachineScope.EXACT_MACHINE,
        context_type=EvidenceContextType.GENERAL_REFERENCE,
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
        referenced_machine_id="M-104",
        dataset_id=ds_a.dataset_id,
    )

    hyp = InvestigativeHypothesis(
        title=pattern["title"],
        description=pattern["description"],
        qualification="Hypothesis for incident in Dataset B",
        supporting_evidence=[ref_from_a],
        contradictory_evidence=[],
        suggested_checks=[],
    )

    # Verify hypothesis scoped to Dataset B
    with pytest.raises(DatasetOwnershipError, match="foreign evidence"):
        verify_hypothesis_evidence(hyp, dataset_id=ds_b.dataset_id)
    ref_from_a.dataset_id = None
    with pytest.raises(DatasetOwnershipError, match="untagged or foreign evidence"):
        verify_hypothesis_evidence(hyp, dataset_id=ds_b.dataset_id)


def test_scoped_verification_rejects_forged_canonical_provenance(isolated_env):
    """Matching dataset labels alone cannot make fabricated evidence trustworthy."""
    store: EvidenceStore = isolated_env["store"]
    retriever: EvidenceRetriever = isolated_env["retriever"]
    _, dataset, _ = create_dataset_entry(store, name="Canonical Verification Dataset")
    reference = EvidenceReference(
        document_id="DOC-NOT-IN-CATALOG",
        source_filename="fabricated.txt",
        chunk_id="CHUNK-NOT-IN-INDEX",
        chunk_index=0,
        snippet="Nominal bearing temperature with no defects observed.",
        relevance_reason="Fabricated test reference",
        dataset_id=dataset.dataset_id,
    )
    hypothesis = InvestigativeHypothesis(
        title="Bearing degradation",
        description="Test",
        qualification="Preliminary",
        supporting_evidence=[reference],
        contradictory_evidence=[],
        suggested_checks=[],
    )

    with pytest.raises(DatasetOwnershipError, match="canonical stored provenance"):
        verify_hypothesis_evidence(
            hypothesis,
            dataset_id=dataset.dataset_id,
            evidence_retriever=retriever,
        )


def test_untagged_injected_evidence_and_foreign_incident_are_rejected(isolated_env):
    store: EvidenceStore = isolated_env["store"]
    ingest: IngestionService = isolated_env["ingestion"]
    index: SearchIndex = isolated_env["index"]
    retriever: EvidenceRetriever = isolated_env["retriever"]
    analyzer: FaultAnalysisEngine = isolated_env["analyzer"]
    _, dataset_a, _ = create_dataset_entry(store, name="Injection Dataset A")
    _, dataset_b, _ = create_dataset_entry(store, name="Injection Dataset B")

    doc = ingest.ingest_document(
        filename="bearing_source.txt",
        content=b"Bearing temperature and acoustic pitch indicate roller element wear.",
        dataset_id=dataset_a.dataset_id,
    )
    index.index_document(doc, store.get_extracted_text(doc), store=store)
    evidence = retriever.search("bearing temperature", dataset_id=dataset_a.dataset_id)
    evidence[0].chunk.dataset_id = None

    incident_b = {
        "incident_id": "INC-INJECT-B",
        "machine_id": "M-104",
        "symptom": "Bearing temperature and acoustic pitch",
        "maintenance_action": "Inspect roller bearing",
        "dataset_id": dataset_b.dataset_id,
    }
    with pytest.raises(DatasetOwnershipError, match="untagged, foreign"):
        analyzer.analyze_incident(
            incident_b,
            evidence_results=evidence,
            dataset_id=dataset_b.dataset_id,
        )

    incident_a = dict(incident_b, dataset_id=dataset_a.dataset_id)
    with pytest.raises(DatasetOwnershipError, match="not active dataset"):
        analyzer.analyze_incident(incident_a, dataset_id=dataset_b.dataset_id)


def test_scoped_strength_and_missing_guidance_reject_foreign_evidence(isolated_env):
    _, dataset_a, _ = create_dataset_entry(isolated_env["store"], name="Guidance Dataset A")
    _, dataset_b, _ = create_dataset_entry(isolated_env["store"], name="Guidance Dataset B")
    ref = EvidenceReference(
        document_id="DOC-GUIDANCE-A",
        source_filename="manual.txt",
        chunk_id="CHK-GUIDANCE-A",
        chunk_index=0,
        snippet="Continuous bearing temperature trend and lubricant analysis.",
        relevance_reason="Test reference",
        dataset_id=dataset_a.dataset_id,
    )
    hypothesis = InvestigativeHypothesis(
        title="Bearing degradation",
        description="Test",
        qualification="Preliminary",
        supporting_evidence=[ref],
        contradictory_evidence=[],
        suggested_checks=[],
    )
    with pytest.raises(DatasetOwnershipError, match="Evidence strength rejected"):
        evaluate_evidence_strength(
            hypothesis,
            VerificationStatus.SUPPORTED,
            dataset_id=dataset_b.dataset_id,
        )
    with pytest.raises(DatasetOwnershipError, match="Missing-evidence guidance rejected"):
        derive_missing_evidence_guidance(
            hypothesis,
            {"symptom": "bearing temperature", "maintenance_action": ""},
            dataset_id=dataset_b.dataset_id,
        )


# ==============================================================================
# Tests 10-11: Cross-Incident Analysis and Linking Scope Isolation
# ==============================================================================


def test_10_cross_incident_comparison_does_not_mix_datasets(isolated_env):
    """Test 10: Cross-incident comparisons do not mix incidents from different datasets."""
    comparison_engine: CrossIncidentComparisonEngine = isolated_env["comparison"]
    _, ds_a, _ = create_dataset_entry(isolated_env["store"], name="Dataset A")
    _, ds_b, _ = create_dataset_entry(isolated_env["store"], name="Dataset B")

    incidents_mixed = [
        # Incidents in Dataset A
        {
            "incident_id": "INC-A-01",
            "machine_id": "M-101",
            "date": "2026-01-10",
            "symptom": "Elevated bearing temperature",
            "maintenance_action": "Greased bearing",
            "outcome": "Resolved",
            "dataset_id": ds_a.dataset_id,
        },
        {
            "incident_id": "INC-A-02",
            "machine_id": "M-102",
            "date": "2026-01-15",
            "symptom": "Elevated bearing temperature",
            "maintenance_action": "Replaced roller",
            "outcome": "Resolved",
            "dataset_id": ds_a.dataset_id,
        },
        # Incidents in Dataset B (identical symptom, different dataset)
        {
            "incident_id": "INC-B-01",
            "machine_id": "M-201",
            "date": "2026-02-01",
            "symptom": "Elevated bearing temperature",
            "maintenance_action": "Cleaned housing",
            "outcome": "Resolved",
            "dataset_id": ds_b.dataset_id,
        },
    ]

    with pytest.raises(DatasetOwnershipError, match="not comparison dataset"):
        comparison_engine.compare_incidents(incidents_mixed, dataset_id=ds_a.dataset_id)

    res_a = comparison_engine.compare_incidents(incidents_mixed[:2], dataset_id=ds_a.dataset_id)
    assert res_a.total_incidents_analyzed == 2
    assert res_a.dataset_id == ds_a.dataset_id
    res_b = comparison_engine.compare_incidents(incidents_mixed[2:], dataset_id=ds_b.dataset_id)
    assert res_b.total_incidents_analyzed == 1
    assert res_b.dataset_id == ds_b.dataset_id


def test_11_cross_incident_evidence_linking_respects_dataset_scope(isolated_env):
    """Test 11: Cross-incident evidence linking respects dataset scope."""
    store: EvidenceStore = isolated_env["store"]
    ingest: IngestionService = isolated_env["ingestion"]
    index: SearchIndex = isolated_env["index"]
    comparison_engine: CrossIncidentComparisonEngine = isolated_env["comparison"]
    evidence_linker: CrossIncidentEvidenceLinker = isolated_env["linker"]

    _, ds_a, _ = create_dataset_entry(store, name="Dataset A")
    _, ds_b, _ = create_dataset_entry(store, name="Dataset B")

    # Ingest evidence into Dataset B only
    doc_b = ingest.ingest_document(
        filename="hydraulic_manual.txt",
        content=b"Hydraulic circuit pressure drops occur when return filter elements clog.",
        dataset_id=ds_b.dataset_id,
    )
    index.index_document(doc_b, store.get_extracted_text(doc_b), store=store)

    # Incidents in Dataset A with hydraulic symptoms
    incidents_a = [
        {
            "incident_id": "INC-A-01",
            "machine_id": "M-301",
            "date": "2026-01-10",
            "symptom": "Hydraulic circuit pressure drops",
            "maintenance_action": "Checked filter",
            "outcome": "Pending",
            "dataset_id": ds_a.dataset_id,
        }
    ]

    comp_res_a = comparison_engine.compare_incidents(incidents_a, dataset_id=ds_a.dataset_id)

    with pytest.raises(DatasetOwnershipError, match="different dataset"):
        evidence_linker.link_comparison_evidence(
            comp_res_a,
            dataset_id=ds_b.dataset_id,
        )

    # Link evidence scoped to Dataset A -> MUST NOT link Dataset B's evidence!
    linking_res_a = evidence_linker.link_comparison_evidence(comp_res_a, dataset_id=ds_a.dataset_id)
    assert linking_res_a.supported_groups_count == 0
    for grp_res in linking_res_a.group_results:
        assert grp_res.status == EvidenceLinkStatus.NO_SUPPORTING_EVIDENCE
        assert len(grp_res.supporting_evidence) == 0


# ==============================================================================
# Tests 12-14: Empty Datasets, Invalid IDs, and Regression Safety
# ==============================================================================


def test_12_empty_datasets_work_without_crashing(isolated_env):
    """Test 12: Empty datasets function safely across all components without crashing."""
    store: EvidenceStore = isolated_env["store"]
    retriever: EvidenceRetriever = isolated_env["retriever"]
    analyzer: FaultAnalysisEngine = isolated_env["analyzer"]
    comparison: CrossIncidentComparisonEngine = isolated_env["comparison"]

    _, empty_ds, _ = create_dataset_entry(store, name="Empty Factory Unit")

    # Workspace summary for empty dataset
    summary = compute_workspace_summary(empty_ds.dataset_id, pd.DataFrame(), [])
    assert summary["total_incidents"] == 0
    assert summary["total_documents"] == 0
    assert summary["incidents"] == []
    assert summary["documents"] == []

    # Evidence search in empty dataset
    results = retriever.search("bearing vibration", dataset_id=empty_ds.dataset_id)
    assert results == []

    # Cross-incident comparison on empty dataset
    comp_res = comparison.compare_incidents([], dataset_id=empty_ds.dataset_id)
    assert comp_res.total_incidents_analyzed == 0
    assert comp_res.symptom_groups == []

    # Fault analysis on incident in empty dataset with no matching evidence
    inc_rec = {
        "incident_id": "INC-EMPTY-01",
        "machine_id": "M-999",
        "symptom": "Bearing vibration",
        "maintenance_action": "Inspected",
        "outcome": "Pending",
        "dataset_id": empty_ds.dataset_id,
    }
    analysis_res = analyzer.analyze_incident(inc_rec, dataset_id=empty_ds.dataset_id)
    assert analysis_res.status in {AnalysisStatus.NO_EVIDENCE, AnalysisStatus.INSUFFICIENT_EVIDENCE}


def test_13_invalid_active_dataset_ids_handled_safely(isolated_env):
    """Test 13: Invalid or non-existent active dataset IDs are safely handled."""
    store: EvidenceStore = isolated_env["store"]
    retriever: EvidenceRetriever = isolated_env["retriever"]
    all_datasets = store.list_datasets()
    valid_ids = [d.dataset_id for d in all_datasets]

    # Resolve non-existent active dataset ID
    resolved_ds, resolved_id, warning = resolve_active_dataset(
        "NON-EXISTENT-DATASET-999",
        all_datasets,
        default_id=DEFAULT_DATASET_ID,
    )
    assert resolved_id == DEFAULT_DATASET_ID
    assert warning is not None
    assert "invalid or was not found" in warning

    # Search with invalid dataset ID against valid_dataset_ids
    with pytest.raises(DatasetOwnershipError, match="unregistered dataset"):
        retriever.search(
            "spindle bearing",
            dataset_id="NON-EXISTENT-DATASET-999",
            valid_dataset_ids=valid_ids,
        )

    # Attempt to set active dataset ID with unregistered ID
    state = {}
    success, err = set_active_dataset_id(state, "NON-EXISTENT-DATASET-999", valid_dataset_ids=valid_ids)
    assert success is False
    assert "not present in registered datasets" in err


def test_switching_active_dataset_invalidates_cached_analysis_results():
    state = {
        "active_dataset_id": "DATASET-A",
        "analysis_result_INC-1_DATASET-A": object(),
        "analysis_result_INC-2_DATASET-B": object(),
        "unrelated_state": "preserved",
    }

    success, error = set_active_dataset_id(
        state,
        "DATASET-B",
        valid_dataset_ids=["DATASET-A", "DATASET-B"],
    )

    assert success is True
    assert error is None
    assert state["active_dataset_id"] == "DATASET-B"
    assert not any(key.startswith("analysis_result_") for key in state)
    assert state["unrelated_state"] == "preserved"


def test_14_existing_m1_to_m73_behavior_intact(isolated_env):
    """Test 14: Existing default dataset and legacy behavior remains functional."""
    store: EvidenceStore = isolated_env["store"]
    retriever: EvidenceRetriever = isolated_env["retriever"]

    # Default dataset is registered
    default_ds = store.get_dataset(DEFAULT_DATASET_ID)
    assert default_ds is not None

    # Retrieval without a dataset ID must fail closed instead of searching project-wide.
    with pytest.raises(DatasetOwnershipError, match="no dataset ID"):
        retriever.search("any query", dataset_id=None)

    # Registry computation includes default dataset
    registry = compute_dataset_registry([default_ds], None, [])
    assert len(registry) == 1
    assert registry[0]["dataset_id"] == DEFAULT_DATASET_ID


# ---------------------------------------------------------------------------
# Milestone 7.4.1 Tests: Navigation State, UI Routing, and Scoped Catalog
# ---------------------------------------------------------------------------


def test_15_navigation_page_sync():
    """Navigation state is canonical and radio synchronization is a separate step."""
    from src.hub import (
        get_navigation_page,
        set_navigation_page,
        sync_navigation_widget_state,
        sync_navigation_from_widget,
        NAVIGATION_PAGES,
        NAV_HUB,
        NAV_WORKSPACE,
        NAV_FAULT_ANALYSIS,
    )

    state = {"nav_radio": NAV_HUB}
    # Default without session state should be NAV_HUB
    assert get_navigation_page(state, default_page=NAV_HUB) == NAV_HUB

    # Programmatic navigation updates the canonical key but never the widget key.
    success = set_navigation_page(state, NAV_WORKSPACE, valid_pages=NAVIGATION_PAGES)
    assert success is True
    assert state.get("ghostfault_nav") == NAV_WORKSPACE
    assert state.get("nav_radio") == NAV_HUB
    assert get_navigation_page(state) == NAV_WORKSPACE
    assert sync_navigation_widget_state(state, NAVIGATION_PAGES) == NAV_WORKSPACE
    assert state.get("nav_radio") == NAV_WORKSPACE

    # A user-selected radio value is copied back by its callback.
    state["nav_radio"] = NAV_FAULT_ANALYSIS
    assert sync_navigation_from_widget(state, NAVIGATION_PAGES) is True
    assert get_navigation_page(state) == NAV_FAULT_ANALYSIS

    # Rejecting invalid navigation target
    success = set_navigation_page(state, "NON_EXISTENT_PAGE", valid_pages=NAVIGATION_PAGES)
    assert success is False
    # State remains on last valid page
    assert state.get("ghostfault_nav") == NAV_FAULT_ANALYSIS
    assert state.get("nav_radio") == NAV_FAULT_ANALYSIS


def test_16_catalog_scoping_by_active_dataset(isolated_env):
    """Milestone 7.4.1: Test document catalog scoping respects active dataset without dropping records."""
    store: EvidenceStore = isolated_env["store"]
    ingestion: IngestionService = isolated_env["ingestion"]

    _, ds_a, _ = create_dataset_entry(store, name="Dataset Alpha")
    _, ds_b, _ = create_dataset_entry(store, name="Dataset Beta")

    meta_a = ingestion.ingest_document(
        filename="doc_a.txt",
        content=b"Spindle bearing abnormal temperature in dataset A.",
        dataset_id=ds_a.dataset_id,
    )
    meta_b = ingestion.ingest_document(
        filename="doc_b.txt",
        content=b"Gearbox gear tooth chatter in dataset B.",
        dataset_id=ds_b.dataset_id,
    )

    all_docs = store.read_catalog()
    assert len(all_docs) >= 2

    # Scoped to Dataset A
    scoped_a = [d for d in all_docs if d.dataset_id == ds_a.dataset_id]
    assert len(scoped_a) == 1
    assert scoped_a[0].dataset_id == ds_a.dataset_id
    assert scoped_a[0].document_id == meta_a.document_id

    # Scoped to Dataset B
    scoped_b = [d for d in all_docs if d.dataset_id == ds_b.dataset_id]
    assert len(scoped_b) == 1
    assert scoped_b[0].dataset_id == ds_b.dataset_id
    assert scoped_b[0].document_id == meta_b.document_id

    # Scoped to a brand new empty dataset
    _, empty_ds, _ = create_dataset_entry(store, name="Empty Test Dataset")
    scoped_empty = [d for d in all_docs if d.dataset_id == empty_ds.dataset_id]
    assert len(scoped_empty) == 0


def test_17_fault_analysis_empty_dataset_handling():
    """Milestone 7.4.1: Test incident filtering produces empty DataFrame for empty custom dataset."""
    raw_df = pd.DataFrame([
        {"incident_id": "INC-001", "machine_id": "CNC-01", "dataset_id": DEFAULT_DATASET_ID, "symptom": "Vibration"},
        {"incident_id": "INC-002", "machine_id": "CNC-02", "dataset_id": DEFAULT_DATASET_ID, "symptom": "Overheating"},
    ])

    custom_dataset_id = "DATASET-CUSTOM-123"

    # Active incidents scoped to custom dataset with 0 records
    active_incidents = raw_df[raw_df["dataset_id"] == custom_dataset_id].copy()
    assert active_incidents.empty
    assert len(active_incidents) == 0

    # Ensure Default Dataset records are never borrowed
    assert not any(active_incidents["incident_id"] == "INC-001")

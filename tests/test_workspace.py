"""Focused tests for Milestone 7.3: Dataset Workspaces and Active Dataset Selection.

Tests cover:
1. Default active-dataset initialization.
2. Selection of an existing custom dataset.
3. Switching between datasets.
4. Invalid active dataset IDs.
5. Deleted or missing active dataset handling.
6. Persistence of active selection across ordinary reruns (session state helpers).
7. Dataset workspace summary calculations.
8. Correct incident counts by dataset.
9. Correct evidence-document counts by dataset.
10. Empty workspace states.
11. Missing and invalid dataset associations.
12. Opening a dataset workspace from the registry uses the exact dataset ID.
13. Opening the workspace does not create duplicate datasets.
14. Legacy incidents remain visible under the default dataset in-memory.
15. Existing CSV files and evidence metadata are not modified merely by switching datasets.
16. Validation helper tests (validate_active_dataset_id, store.validate_dataset_id).
17. Navigation constants updated with NAV_WORKSPACE.
"""

import tempfile
from pathlib import Path
from typing import List, Optional
import pandas as pd
import pytest

from src.models import (
    Dataset,
    DocumentMetadata,
    DocumentStatus,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_DESCRIPTION,
)
from src.storage import EvidenceStore
from src.data_loader import load_incidents_data
from src.hub import (
    NAV_HUB,
    NAV_WORKSPACE,
    NAV_INCIDENTS,
    NAV_EVIDENCE_INGESTION,
    NAV_EVIDENCE_SEARCH,
    NAV_FAULT_ANALYSIS,
    NAV_CROSS_INCIDENT,
    NAV_INVESTIGATION_HISTORY,
    NAVIGATION_PAGES,
    SESSION_KEY_ACTIVE_DATASET,
    SESSION_KEY_NAV,
    validate_active_dataset_id,
    resolve_active_dataset,
    get_active_dataset_id,
    set_active_dataset_id,
    compute_workspace_summary,
    find_invalid_document_associations,
    find_missing_document_associations,
    find_invalid_incident_associations,
)


# ==============================================================================
# Test Fixtures & Helpers
# ==============================================================================


def _make_dataset(
    dataset_id: str,
    name: str,
    description: Optional[str] = "Test description",
    created_at: str = "2024-06-01T12:00:00Z",
) -> Dataset:
    return Dataset(
        dataset_id=dataset_id,
        name=name,
        description=description,
        created_at=created_at,
    )


def _make_doc_meta(
    doc_id: str,
    filename: str,
    status: DocumentStatus = DocumentStatus.PROCESSED,
    dataset_id: Optional[str] = DEFAULT_DATASET_ID,
    chars: int = 500,
    created_at: str = "2024-06-01T12:00:00Z",
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id,
        original_filename=filename,
        file_type=Path(filename).suffix.lstrip("."),
        source_type="test",
        storage_reference=f"originals/{doc_id}_{filename}",
        extraction_status="completed" if status == DocumentStatus.PROCESSED else "failed",
        ingestion_status=status,
        created_at=created_at,
        updated_at=created_at,
        extracted_character_count=chars if status == DocumentStatus.PROCESSED else 0,
        dataset_id=dataset_id,
    )


def _make_incidents_df(include_dataset_col: bool = True) -> pd.DataFrame:
    data = {
        "incident_id": ["INC-001", "INC-002", "INC-003", "INC-004"],
        "machine_id": ["M-104", "M-104", "M-205", "M-300"],
        "date": ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"],
        "symptom": ["High Vibration", "High Vibration", "Pressure Drop", "Thermal Runaway"],
        "maintenance_action": ["Bearing Replaced", "Shaft Realignment", "Valve Cleaned", "Coolant Flushed"],
        "outcome": ["Resolved", "Verified", "Pending", "Resolved"],
    }
    if include_dataset_col:
        data["dataset_id"] = [
            DEFAULT_DATASET_ID,
            DEFAULT_DATASET_ID,
            "DATASET-HYDRAULICS",
            "DATASET-HYDRAULICS",
        ]
    return pd.DataFrame(data)


# ==============================================================================
# 1. Default Active-Dataset Initialization
# ==============================================================================


class TestActiveDatasetInitialization:
    def test_default_active_dataset_when_none(self):
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        resolved_ds, resolved_id, warning = resolve_active_dataset(None, datasets)

        assert resolved_ds is not None
        assert resolved_ds.dataset_id == DEFAULT_DATASET_ID
        assert resolved_id == DEFAULT_DATASET_ID
        assert warning is None

    def test_default_active_dataset_when_empty_string(self):
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        resolved_ds, resolved_id, warning = resolve_active_dataset("   ", datasets)

        assert resolved_ds is not None
        assert resolved_ds.dataset_id == DEFAULT_DATASET_ID
        assert resolved_id == DEFAULT_DATASET_ID
        assert warning is None

    def test_get_active_dataset_id_defaults_correctly(self):
        session_state = {}
        active_id = get_active_dataset_id(session_state)
        assert active_id == DEFAULT_DATASET_ID


# ==============================================================================
# 2. Selection of an Existing Custom Dataset
# ==============================================================================


class TestSelectExistingDataset:
    def test_select_custom_dataset(self):
        ds_default = _make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)
        ds_custom = _make_dataset("DATASET-PLANT-B", "Plant B Maintenance")
        datasets = [ds_default, ds_custom]

        resolved_ds, resolved_id, warning = resolve_active_dataset("DATASET-PLANT-B", datasets)

        assert resolved_ds is not None
        assert resolved_ds.dataset_id == "DATASET-PLANT-B"
        assert resolved_ds.name == "Plant B Maintenance"
        assert resolved_id == "DATASET-PLANT-B"
        assert warning is None


# ==============================================================================
# 3. Switching Between Datasets
# ==============================================================================


class TestSwitchingDatasets:
    def test_switching_updates_session_state(self):
        session_state = {SESSION_KEY_ACTIVE_DATASET: DEFAULT_DATASET_ID}
        valid_ids = [DEFAULT_DATASET_ID, "DATASET-TURBINES"]

        success, err = set_active_dataset_id(session_state, "DATASET-TURBINES", valid_ids)
        assert success is True
        assert err is None
        assert session_state[SESSION_KEY_ACTIVE_DATASET] == "DATASET-TURBINES"

        # Switch back
        success, err = set_active_dataset_id(session_state, DEFAULT_DATASET_ID, valid_ids)
        assert success is True
        assert err is None
        assert session_state[SESSION_KEY_ACTIVE_DATASET] == DEFAULT_DATASET_ID


# ==============================================================================
# 4. Invalid Active Dataset IDs
# ==============================================================================


class TestInvalidActiveDatasetIDs:
    def test_invalid_active_id_falls_back_to_default_with_warning(self):
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        resolved_ds, resolved_id, warning = resolve_active_dataset("INVALID-NONEXISTENT-ID", datasets)

        assert resolved_ds is not None
        assert resolved_ds.dataset_id == DEFAULT_DATASET_ID
        assert resolved_id == DEFAULT_DATASET_ID
        assert warning is not None
        assert "INVALID-NONEXISTENT-ID" in warning
        assert "Reset" in warning or "reset" in warning.lower()

    def test_validate_active_dataset_id_helper(self):
        valid = [DEFAULT_DATASET_ID, "DATASET-PUMPS"]
        assert validate_active_dataset_id(DEFAULT_DATASET_ID, valid) is True
        assert validate_active_dataset_id("DATASET-PUMPS", valid) is True
        assert validate_active_dataset_id("UNKNOWN-DS", valid) is False
        assert validate_active_dataset_id("", valid) is False
        assert validate_active_dataset_id(None, valid) is False


# ==============================================================================
# 5. Deleted or Missing Active Dataset Handling
# ==============================================================================


class TestDeletedOrMissingDataset:
    def test_previously_selected_deleted_dataset_resets_safely(self):
        # Simulating that DATASET-DELETED was previously selected, but is no longer in registry
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]

        resolved_ds, resolved_id, warning = resolve_active_dataset("DATASET-DELETED", datasets)

        assert resolved_ds is not None
        assert resolved_ds.dataset_id == DEFAULT_DATASET_ID
        assert resolved_id == DEFAULT_DATASET_ID
        assert "DATASET-DELETED" in warning

    def test_empty_registry_handling(self):
        resolved_ds, resolved_id, warning = resolve_active_dataset(DEFAULT_DATASET_ID, [])
        assert resolved_ds is None
        assert resolved_id == DEFAULT_DATASET_ID
        assert warning is not None


# ==============================================================================
# 6. Persistence of Active Selection Across Ordinary Reruns
# ==============================================================================


class TestSessionStatePersistence:
    def test_active_selection_persists_across_calls(self):
        session_state = {}
        # Initial call defaults
        assert get_active_dataset_id(session_state) == DEFAULT_DATASET_ID

        # Set active to custom
        set_active_dataset_id(session_state, "DATASET-GEARS")

        # Emulate subsequent rerun reads
        for _ in range(5):
            assert get_active_dataset_id(session_state) == "DATASET-GEARS"

    def test_set_active_dataset_id_rejects_empty(self):
        session_state = {}
        success, err = set_active_dataset_id(session_state, "")
        assert success is False
        assert "empty" in err.lower()

    def test_set_active_dataset_id_rejects_unregistered_when_valid_ids_passed(self):
        session_state = {}
        valid_ids = ["DS-1", "DS-2"]
        success, err = set_active_dataset_id(session_state, "DS-UNREGISTERED", valid_ids)
        assert success is False
        assert "not present" in err.lower()


# ==============================================================================
# 7, 8, 9, 10. Workspace Summary Calculations & Counts
# ==============================================================================


class TestWorkspaceSummaryCalculations:
    def test_workspace_summary_for_default_dataset(self):
        df = _make_incidents_df()
        docs = [
            _make_doc_meta("DOC-1", "doc1.pdf", dataset_id=DEFAULT_DATASET_ID, chars=300),
            _make_doc_meta("DOC-2", "doc2.txt", dataset_id=DEFAULT_DATASET_ID, chars=400),
            _make_doc_meta("DOC-3", "doc3.txt", dataset_id="DATASET-HYDRAULICS", chars=600),
        ]

        summary = compute_workspace_summary(DEFAULT_DATASET_ID, df, docs)

        assert summary["dataset_id"] == DEFAULT_DATASET_ID
        assert summary["total_incidents"] == 2
        assert len(summary["incidents"]) == 2
        assert summary["unique_machines"] == ["M-104"]
        assert summary["total_documents"] == 2
        assert summary["processed_documents"] == 2
        assert summary["failed_documents"] == 0
        assert summary["total_extracted_characters"] == 700

    def test_workspace_summary_for_custom_dataset(self):
        df = _make_incidents_df()
        docs = [
            _make_doc_meta("DOC-1", "doc1.pdf", dataset_id=DEFAULT_DATASET_ID),
            _make_doc_meta("DOC-3", "doc3.txt", dataset_id="DATASET-HYDRAULICS", chars=800),
        ]

        summary = compute_workspace_summary("DATASET-HYDRAULICS", df, docs)

        assert summary["dataset_id"] == "DATASET-HYDRAULICS"
        assert summary["total_incidents"] == 2
        assert len(summary["incidents"]) == 2
        assert sorted(summary["unique_machines"]) == ["M-205", "M-300"]
        assert summary["total_documents"] == 1
        assert summary["processed_documents"] == 1
        assert summary["total_extracted_characters"] == 800

    def test_empty_workspace_states(self):
        summary = compute_workspace_summary("DATASET-EMPTY", None, [])

        assert summary["dataset_id"] == "DATASET-EMPTY"
        assert summary["total_incidents"] == 0
        assert summary["incidents"] == []
        assert summary["unique_machines"] == []
        assert summary["unique_symptoms"] == []
        assert summary["total_documents"] == 0
        assert summary["processed_documents"] == 0
        assert summary["failed_documents"] == 0
        assert summary["documents"] == []
        assert summary["total_extracted_characters"] == 0

    def test_empty_dataframe_and_no_docs(self):
        summary = compute_workspace_summary("DATASET-EMPTY", pd.DataFrame(), [])
        assert summary["total_incidents"] == 0
        assert summary["total_documents"] == 0

    def test_workspace_with_failed_documents(self):
        docs = [
            _make_doc_meta("DOC-1", "a.pdf", DocumentStatus.PROCESSED, dataset_id="DS-FAIL", chars=250),
            _make_doc_meta("DOC-2", "bad.pdf", DocumentStatus.FAILED, dataset_id="DS-FAIL", chars=0),
        ]
        summary = compute_workspace_summary("DS-FAIL", None, docs)
        assert summary["total_documents"] == 2
        assert summary["processed_documents"] == 1
        assert summary["failed_documents"] == 1
        assert summary["total_extracted_characters"] == 250


# ==============================================================================
# 11. Missing and Invalid Dataset Associations
# ==============================================================================


class TestMissingAndInvalidAssociations:
    def test_find_invalid_document_associations(self):
        docs = [
            _make_doc_meta("DOC-1", "a.pdf", dataset_id=DEFAULT_DATASET_ID),
            _make_doc_meta("DOC-2", "b.pdf", dataset_id="VALID-CUSTOM"),
            _make_doc_meta("DOC-3", "c.pdf", dataset_id="ORPHANED-DATASET"),
            _make_doc_meta("DOC-4", "d.pdf", dataset_id=None),
            _make_doc_meta("DOC-5", "e.pdf", dataset_id="   "),
        ]
        valid_ids = [DEFAULT_DATASET_ID, "VALID-CUSTOM"]

        invalid = find_invalid_document_associations(docs, valid_ids)
        assert len(invalid) == 1
        invalid_ids = {d.document_id for d in invalid}
        assert invalid_ids == {"DOC-3"}

        missing = find_missing_document_associations(docs)
        assert {d.document_id for d in missing} == {"DOC-4", "DOC-5"}

    def test_find_invalid_incident_associations(self):
        df = _make_incidents_df()
        valid_ids = [DEFAULT_DATASET_ID]  # "DATASET-HYDRAULICS" is not in valid_ids

        invalid = find_invalid_incident_associations(df, valid_ids)
        assert len(invalid) == 2  # INC-003 and INC-004 have DATASET-HYDRAULICS
        assert invalid[0]["incident_id"] == "INC-003"


# ==============================================================================
# 12 & 13. Exact Dataset ID Usage & No Duplicate Creation
# ==============================================================================


class TestWorkspaceRegistryInteraction:
    def test_workspace_action_identifies_exact_dataset_id(self):
        # Two datasets with similar names but different IDs
        ds1 = _make_dataset("DS-001", "Turbine Group", created_at="2024-01-01T00:00:00Z")
        ds2 = _make_dataset("DS-002", "Turbine Group", created_at="2024-02-01T00:00:00Z")
        datasets = [ds1, ds2]

        resolved_ds, resolved_id, _ = resolve_active_dataset("DS-002", datasets)
        assert resolved_id == "DS-002"
        assert resolved_ds.created_at == "2024-02-01T00:00:00Z"

    def test_opening_workspace_does_not_create_duplicate_datasets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = EvidenceStore(Path(tmp_dir))
            store.ensure_default_dataset()
            store.create_dataset("DS-TEST", "Test Dataset")

            initial_count = len(store.list_datasets())
            assert initial_count == 2

            # Resolving or reading workspace multiple times does not alter dataset file
            for _ in range(5):
                datasets = store.list_datasets()
                resolved_ds, _, _ = resolve_active_dataset("DS-TEST", datasets)
                assert resolved_ds.dataset_id == "DS-TEST"
                _ = compute_workspace_summary("DS-TEST", None, store.read_catalog())

            assert len(store.list_datasets()) == initial_count


# ==============================================================================
# 14. Legacy Incidents Visible Under Default Dataset
# ==============================================================================


class TestLegacyIncidentsCompatibility:
    def test_legacy_incidents_without_dataset_col_match_default_dataset(self):
        # DataFrame without dataset_id column
        legacy_df = _make_incidents_df(include_dataset_col=False)

        summary_default = compute_workspace_summary(DEFAULT_DATASET_ID, legacy_df, [])
        assert summary_default["total_incidents"] == 4  # All match default
        assert len(summary_default["incidents"]) == 4

        # But a custom dataset summary does NOT match legacy incidents
        summary_custom = compute_workspace_summary("DATASET-CUSTOM", legacy_df, [])
        assert summary_custom["total_incidents"] == 0
        assert len(summary_custom["incidents"]) == 0

    def test_legacy_incidents_via_data_loader_in_memory_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "incidents.csv"
            # Write CSV WITHOUT dataset_id column
            legacy_df = _make_incidents_df(include_dataset_col=False)
            legacy_df.to_csv(csv_path, index=False)

            # Load via data_loader
            loaded_df = load_incidents_data(csv_path)
            assert "dataset_id" in loaded_df.columns

            # In memory, all rows have DEFAULT_DATASET_ID
            summary = compute_workspace_summary(DEFAULT_DATASET_ID, loaded_df, [])
            assert summary["total_incidents"] == 4

            # Verify the CSV on disk was NOT mutated
            raw_disk_df = pd.read_csv(csv_path)
            assert "dataset_id" not in raw_disk_df.columns


# ==============================================================================
# 15. Switching Datasets Does Not Modify Data On Disk
# ==============================================================================


class TestNoDataMutationOnSwitching:
    def test_switching_does_not_modify_csv_or_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = EvidenceStore(Path(tmp_dir))
            store.ensure_default_dataset()
            store.create_dataset("DS-CUSTOM-A", "Custom A")
            store.create_dataset("DS-CUSTOM-B", "Custom B")

            # Add a document
            doc_meta = _make_doc_meta("DOC-001", "manual.pdf", dataset_id="DS-CUSTOM-A")
            store.add_or_update_metadata(doc_meta)

            csv_path = Path(tmp_dir) / "incidents.csv"
            orig_df = _make_incidents_df(include_dataset_col=True)
            orig_df.to_csv(csv_path, index=False)
            csv_content_before = csv_path.read_text(encoding="utf-8")
            catalog_content_before = store.catalog_file.read_text(encoding="utf-8")
            datasets_content_before = store.datasets_file.read_text(encoding="utf-8")

            # Simulate session state dataset switching
            session_state = {SESSION_KEY_ACTIVE_DATASET: DEFAULT_DATASET_ID}
            datasets = store.list_datasets()
            valid_ids = [d.dataset_id for d in datasets]

            # Switch to A
            set_active_dataset_id(session_state, "DS-CUSTOM-A", valid_ids)
            _ = compute_workspace_summary("DS-CUSTOM-A", orig_df, store.read_catalog())

            # Switch to B
            set_active_dataset_id(session_state, "DS-CUSTOM-B", valid_ids)
            _ = compute_workspace_summary("DS-CUSTOM-B", orig_df, store.read_catalog())

            # Switch back to Default
            set_active_dataset_id(session_state, DEFAULT_DATASET_ID, valid_ids)
            _ = compute_workspace_summary(DEFAULT_DATASET_ID, orig_df, store.read_catalog())

            # Assert files on disk are byte-identical
            assert csv_path.read_text(encoding="utf-8") == csv_content_before
            assert store.catalog_file.read_text(encoding="utf-8") == catalog_content_before
            assert store.datasets_file.read_text(encoding="utf-8") == datasets_content_before


# ==============================================================================
# 16. EvidenceStore validate_dataset_id helper
# ==============================================================================


class TestEvidenceStoreValidationHelper:
    def test_store_validate_dataset_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = EvidenceStore(Path(tmp_dir))
            store.ensure_default_dataset()
            store.create_dataset("DS-VALID", "Valid Dataset")

            assert store.validate_dataset_id(DEFAULT_DATASET_ID) is True
            assert store.validate_dataset_id("DS-VALID") is True
            assert store.validate_dataset_id("DS-NONEXISTENT") is False
            assert store.validate_dataset_id("") is False
            assert store.validate_dataset_id(None) is False


# ==============================================================================
# 17. Navigation Constants updated with NAV_WORKSPACE
# ==============================================================================


class TestWorkspaceNavigationConstants:
    def test_nav_workspace_in_navigation_pages(self):
        assert NAV_WORKSPACE in NAVIGATION_PAGES
        assert NAVIGATION_PAGES[0] == NAV_HUB
        assert NAVIGATION_PAGES[1] == NAV_WORKSPACE
        assert NAV_INCIDENTS in NAVIGATION_PAGES
        assert NAV_EVIDENCE_INGESTION in NAVIGATION_PAGES
        assert NAV_EVIDENCE_SEARCH in NAVIGATION_PAGES
        assert NAV_FAULT_ANALYSIS in NAVIGATION_PAGES
        assert NAV_CROSS_INCIDENT in NAVIGATION_PAGES
        assert NAV_INVESTIGATION_HISTORY in NAVIGATION_PAGES
        assert len(NAVIGATION_PAGES) == 8
        assert len(NAVIGATION_PAGES) == len(set(NAVIGATION_PAGES))

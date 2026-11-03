"""Focused tests for Milestone 7.2: Central Intelligence Hub logic.

Tests cover project summary calculations, dataset registry, incident overview,
evidence summary, recent activity, and navigation constants.
All tests use temporary directories and isolated fixtures.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.models import (
    DocumentMetadata,
    DocumentStatus,
    Dataset,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_DESCRIPTION,
    DatasetValidationError,
    current_iso_timestamp,
)
from src.storage import EvidenceStore
from src.hub import (
    compute_project_summary,
    compute_dataset_registry,
    compute_unassociated_counts,
    compute_incident_overview,
    compute_evidence_summary,
    compute_recent_documents,
    NAVIGATION_PAGES,
    NAV_HUB,
    NAV_WORKSPACE,
    NAV_INCIDENTS,
    NAV_EVIDENCE_INGESTION,
    NAV_EVIDENCE_SEARCH,
    NAV_FAULT_ANALYSIS,
    NAV_CROSS_INCIDENT,
)
from src.data_loader import load_incidents_data, DatasetLoadError


# ==============================================================================
# Fixtures
# ==============================================================================


def _make_dataset(dataset_id: str, name: str, description: str = "") -> Dataset:
    return Dataset(
        dataset_id=dataset_id,
        name=name,
        description=description,
        created_at=current_iso_timestamp(),
    )


def _make_doc_meta(
    doc_id: str,
    filename: str,
    status: DocumentStatus = DocumentStatus.PROCESSED,
    dataset_id: str = DEFAULT_DATASET_ID,
    created_at: str = "",
) -> DocumentMetadata:
    ts = created_at if created_at else current_iso_timestamp()
    return DocumentMetadata(
        document_id=doc_id,
        original_filename=filename,
        file_type=filename.rsplit(".", 1)[-1] if "." in filename else "txt",
        source_type="test",
        storage_reference=f"originals/{doc_id}_{filename}",
        extraction_status="success" if status == DocumentStatus.PROCESSED else "failed",
        ingestion_status=status,
        created_at=ts,
        updated_at=ts,
        extracted_character_count=1000 if status == DocumentStatus.PROCESSED else 0,
        dataset_id=dataset_id,
    )


def _make_incidents_df(rows=None) -> pd.DataFrame:
    if rows is None:
        rows = [
            {"incident_id": "INC-001", "machine_id": "M-104", "date": "2024-06-01",
             "symptom": "Vibration spike", "maintenance_action": "Replaced bearing",
             "outcome": "Resolved", "dataset_id": DEFAULT_DATASET_ID},
            {"incident_id": "INC-002", "machine_id": "M-205", "date": "2024-06-15",
             "symptom": "Hydraulic leak", "maintenance_action": "Sealed valve",
             "outcome": "Monitoring", "dataset_id": DEFAULT_DATASET_ID},
            {"incident_id": "INC-003", "machine_id": "M-104", "date": "2024-07-01",
             "symptom": "Temperature alarm", "maintenance_action": "Coolant flush",
             "outcome": "Resolved", "dataset_id": "DS-CUSTOM"},
        ]
    return pd.DataFrame(rows)


# ==============================================================================
# Test: Project Summary
# ==============================================================================


class TestProjectSummary:
    def test_basic_project_summary(self):
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        df = _make_incidents_df()
        docs = [
            _make_doc_meta("DOC-001", "manual.pdf"),
            _make_doc_meta("DOC-002", "log.txt"),
            _make_doc_meta("DOC-003", "bad.pdf", DocumentStatus.FAILED),
        ]

        summary = compute_project_summary(datasets, df, docs)

        assert summary["total_datasets"] == 1
        assert summary["total_incidents"] == 3
        assert summary["total_documents"] == 3
        assert summary["processed_documents"] == 2
        assert summary["failed_documents"] == 1
        assert summary["unique_machines"] == 2  # M-104, M-205
        assert summary["unique_symptoms"] == 3

    def test_empty_project_summary(self):
        summary = compute_project_summary([], None, [])

        assert summary["total_datasets"] == 0
        assert summary["total_incidents"] == 0
        assert summary["total_documents"] == 0
        assert summary["processed_documents"] == 0
        assert summary["failed_documents"] == 0
        assert summary["unique_machines"] == 0
        assert summary["unique_symptoms"] == 0

    def test_summary_with_empty_dataframe(self):
        summary = compute_project_summary([], pd.DataFrame(), [])
        assert summary["total_incidents"] == 0
        assert summary["unique_machines"] == 0

    def test_summary_counts_not_double_counted(self):
        """Verify that incidents and documents are counted exactly once each."""
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        df = _make_incidents_df()
        docs = [_make_doc_meta("DOC-001", "a.pdf"), _make_doc_meta("DOC-002", "b.txt")]

        summary = compute_project_summary(datasets, df, docs)

        assert summary["total_incidents"] == 3
        assert summary["total_documents"] == 2


# ==============================================================================
# Test: Dataset Registry
# ==============================================================================


class TestDatasetRegistry:
    def test_single_default_dataset_with_counts(self):
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        df = _make_incidents_df([
            {"incident_id": "INC-001", "machine_id": "M-104", "date": "2024-06-01",
             "symptom": "Vibration", "maintenance_action": "Fixed", "outcome": "OK",
             "dataset_id": DEFAULT_DATASET_ID},
        ])
        docs = [_make_doc_meta("DOC-001", "a.pdf", dataset_id=DEFAULT_DATASET_ID)]

        registry = compute_dataset_registry(datasets, df, docs)

        assert len(registry) == 1
        assert registry[0]["dataset_id"] == DEFAULT_DATASET_ID
        assert registry[0]["incident_count"] == 1
        assert registry[0]["document_count"] == 1

    def test_multiple_datasets_correct_counts(self):
        datasets = [
            _make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME),
            _make_dataset("DS-CUSTOM", "Custom Project"),
        ]
        df = _make_incidents_df()
        docs = [
            _make_doc_meta("DOC-001", "a.pdf", dataset_id=DEFAULT_DATASET_ID),
            _make_doc_meta("DOC-002", "b.txt", dataset_id="DS-CUSTOM"),
            _make_doc_meta("DOC-003", "c.pdf", dataset_id="DS-CUSTOM"),
        ]

        registry = compute_dataset_registry(datasets, df, docs)

        default_row = next(r for r in registry if r["dataset_id"] == DEFAULT_DATASET_ID)
        custom_row = next(r for r in registry if r["dataset_id"] == "DS-CUSTOM")

        assert default_row["incident_count"] == 2  # INC-001, INC-002
        assert default_row["document_count"] == 1
        assert custom_row["incident_count"] == 1  # INC-003
        assert custom_row["document_count"] == 2

    def test_no_duplicate_datasets_during_computation(self):
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        df = _make_incidents_df()
        docs = []

        registry = compute_dataset_registry(datasets, df, docs)

        dataset_ids = [r["dataset_id"] for r in registry]
        assert len(dataset_ids) == len(set(dataset_ids)), "Duplicate dataset IDs found in registry"

    def test_empty_datasets_registry(self):
        registry = compute_dataset_registry([], None, [])
        assert registry == []

    def test_registry_with_no_incidents(self):
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        registry = compute_dataset_registry(datasets, pd.DataFrame(), [])
        assert len(registry) == 1
        assert registry[0]["incident_count"] == 0
        assert registry[0]["document_count"] == 0


# ==============================================================================
# Test: Unassociated Counts
# ==============================================================================


class TestUnassociatedCounts:
    def test_no_unassociated_when_all_match(self):
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        df = _make_incidents_df([
            {"incident_id": "INC-001", "machine_id": "M-104", "date": "2024-01-01",
             "symptom": "Test", "maintenance_action": "Fix", "outcome": "OK",
             "dataset_id": DEFAULT_DATASET_ID},
        ])
        docs = [_make_doc_meta("DOC-001", "a.pdf", dataset_id=DEFAULT_DATASET_ID)]

        result = compute_unassociated_counts(datasets, df, docs)
        assert result["unassociated_incidents"] == 0
        assert result["unassociated_documents"] == 0

    def test_detects_unassociated_records(self):
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        df = _make_incidents_df()  # INC-003 has dataset_id "DS-CUSTOM" which is not registered
        docs = [
            _make_doc_meta("DOC-001", "a.pdf", dataset_id=DEFAULT_DATASET_ID),
            _make_doc_meta("DOC-002", "b.txt", dataset_id="NONEXISTENT"),
        ]

        result = compute_unassociated_counts(datasets, df, docs)
        assert result["unassociated_incidents"] == 1  # INC-003 -> DS-CUSTOM
        assert result["unassociated_documents"] == 1  # DOC-002 -> NONEXISTENT

    def test_none_dataset_id_counted_as_unassociated(self):
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        docs = [_make_doc_meta("DOC-001", "a.pdf", dataset_id=None)]

        result = compute_unassociated_counts(datasets, None, docs)
        assert result["unassociated_documents"] == 1


# ==============================================================================
# Test: Incident Overview
# ==============================================================================


class TestIncidentOverview:
    def test_overview_table_fields(self):
        df = _make_incidents_df()
        rows = compute_incident_overview(df)

        assert len(rows) == 3
        assert rows[0]["Incident ID"] == "INC-001"
        assert rows[0]["Machine ID"] == "M-104"
        assert rows[0]["Date"] == "2024-06-01"
        assert "Vibration" in rows[0]["Symptom"]
        assert rows[0]["Dataset"] == DEFAULT_DATASET_ID

    def test_empty_overview(self):
        rows = compute_incident_overview(None)
        assert rows == []

        rows = compute_incident_overview(pd.DataFrame())
        assert rows == []

    def test_overview_does_not_modify_source(self):
        df = _make_incidents_df()
        original_len = len(df)
        _ = compute_incident_overview(df)
        assert len(df) == original_len  # Not modified


# ==============================================================================
# Test: Evidence Summary
# ==============================================================================


class TestEvidenceSummary:
    def test_evidence_summary_counts(self):
        docs = [
            _make_doc_meta("DOC-001", "manual.pdf"),
            _make_doc_meta("DOC-002", "log.txt"),
            _make_doc_meta("DOC-003", "bad.pdf", DocumentStatus.FAILED),
        ]

        summary = compute_evidence_summary(docs)

        assert summary["total_documents"] == 3
        assert summary["processed_documents"] == 2
        assert summary["failed_documents"] == 1
        assert summary["type_counts"]["PDF"] == 2
        assert summary["type_counts"]["TXT"] == 1
        assert summary["total_extracted_characters"] == 2000  # 2 processed * 1000

    def test_empty_evidence_summary(self):
        summary = compute_evidence_summary([])
        assert summary["total_documents"] == 0
        assert summary["type_counts"] == {}
        assert summary["total_extracted_characters"] == 0


# ==============================================================================
# Test: Recent Documents
# ==============================================================================


class TestRecentDocuments:
    def test_recent_documents_ordering(self):
        docs = [
            _make_doc_meta("DOC-001", "old.pdf", created_at="2024-01-01T00:00:00+00:00"),
            _make_doc_meta("DOC-002", "mid.txt", created_at="2024-06-01T00:00:00+00:00"),
            _make_doc_meta("DOC-003", "new.pdf", created_at="2024-12-01T00:00:00+00:00"),
        ]

        recent = compute_recent_documents(docs, limit=2)

        assert len(recent) == 2
        assert recent[0]["document_id"] == "DOC-003"
        assert recent[1]["document_id"] == "DOC-002"

    def test_recent_documents_empty(self):
        assert compute_recent_documents([], limit=5) == []

    def test_recent_documents_limit(self):
        docs = [_make_doc_meta(f"DOC-{i:03d}", f"doc{i}.txt") for i in range(10)]
        recent = compute_recent_documents(docs, limit=3)
        assert len(recent) == 3


# ==============================================================================
# Test: Navigation Constants
# ==============================================================================


class TestNavigationConstants:
    def test_all_navigation_pages_present(self):
        assert NAV_HUB in NAVIGATION_PAGES
        assert NAV_WORKSPACE in NAVIGATION_PAGES
        assert NAV_INCIDENTS in NAVIGATION_PAGES
        assert NAV_EVIDENCE_INGESTION in NAVIGATION_PAGES
        assert NAV_EVIDENCE_SEARCH in NAVIGATION_PAGES
        assert NAV_FAULT_ANALYSIS in NAVIGATION_PAGES
        assert NAV_CROSS_INCIDENT in NAVIGATION_PAGES

    def test_hub_is_first_page(self):
        assert NAVIGATION_PAGES[0] == NAV_HUB

    def test_no_duplicate_navigation_entries(self):
        assert len(NAVIGATION_PAGES) == len(set(NAVIGATION_PAGES))


# ==============================================================================
# Test: Integration with EvidenceStore (isolated temp directory)
# ==============================================================================


class TestHubWithEvidenceStore:
    def test_hub_reads_datasets_without_creating_duplicates(self):
        """Calling ensure_default_dataset + list_datasets repeatedly must not duplicate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvidenceStore(base_dir=Path(tmpdir))

            # First call
            store.ensure_default_dataset()
            datasets_1 = store.list_datasets()

            # Second call (simulates re-rendering hub)
            store.ensure_default_dataset()
            datasets_2 = store.list_datasets()

            assert len(datasets_1) == 1
            assert len(datasets_2) == 1
            assert datasets_1[0].dataset_id == DEFAULT_DATASET_ID

    def test_hub_does_not_modify_csv_on_load(self):
        """Loading incidents for hub must not mutate the CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "incidents.csv"
            csv_path.write_text(
                "incident_id,machine_id,date,symptom,maintenance_action,outcome\n"
                "INC-001,M-104,2024-01-01,Vibration,Fixed,Resolved\n",
                encoding="utf-8",
            )

            original_content = csv_path.read_text(encoding="utf-8")
            df = load_incidents_data(csv_path=csv_path)
            after_content = csv_path.read_text(encoding="utf-8")

            # File must not be modified
            assert original_content == after_content
            # In-memory default dataset_id must be populated
            assert "dataset_id" in df.columns
            assert df["dataset_id"].iloc[0] == DEFAULT_DATASET_ID

    def test_hub_catalog_read_does_not_mutate_catalog(self):
        """Reading catalog for hub counts must not modify catalog.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvidenceStore(base_dir=Path(tmpdir))

            doc = _make_doc_meta("DOC-001", "test.pdf")
            store.add_or_update_metadata(doc)

            original_content = store.catalog_file.read_text(encoding="utf-8")
            docs = store.read_catalog()
            after_content = store.catalog_file.read_text(encoding="utf-8")

            assert original_content == after_content
            assert len(docs) == 1

    def test_registry_with_legacy_csv_no_dataset_column(self):
        """Ensure hub handles legacy CSV without dataset_id column gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "incidents.csv"
            csv_path.write_text(
                "incident_id,machine_id,date,symptom,maintenance_action,outcome\n"
                "INC-001,M-104,2024-01-01,Vibration,Fixed,Resolved\n"
                "INC-002,M-205,2024-02-01,Leak,Sealed,Monitoring\n",
                encoding="utf-8",
            )

            df = load_incidents_data(csv_path=csv_path)
            datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
            docs = []

            registry = compute_dataset_registry(datasets, df, docs)

            assert len(registry) == 1
            assert registry[0]["incident_count"] == 2  # Both assigned to default in memory

    def test_malformed_optional_metadata_handled(self):
        """Documents with None dataset_id are handled without errors."""
        datasets = [_make_dataset(DEFAULT_DATASET_ID, DEFAULT_DATASET_NAME)]
        docs = [_make_doc_meta("DOC-001", "a.pdf", dataset_id=None)]

        # Should not raise
        registry = compute_dataset_registry(datasets, None, docs)
        assert registry[0]["document_count"] == 0  # None != DEFAULT_DATASET_ID

        summary = compute_evidence_summary(docs)
        assert summary["total_documents"] == 1

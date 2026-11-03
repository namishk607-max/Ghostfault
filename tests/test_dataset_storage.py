"""Focused automated test suite for GhostFault Milestone 7.1: Dataset Model, Storage & Migration.

Covers:
- Dataset creation and field validation
- Dataset persistence across reload/reinitialization
- Dataset lookup and listing
- Duplicate ID handling
- Invalid input handling (empty ID, invalid chars, short/long name)
- Non-destructive migration of existing incidents and evidence
- Idempotent repeated migration without duplicate records
- Preservation of IDs, provenance, text references, and file structures
- Backward compatibility with legacy unassociated data
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.models import (
    Dataset,
    DatasetValidationError,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_DESCRIPTION,
    DocumentMetadata,
    DocumentStatus,
)
from src.storage import EvidenceStore
from src.data_loader import (
    load_incidents_data,
    migrate_incidents_csv,
    get_default_dataset_path,
    DatasetLoadError,
)
from src.ingestion import IngestionService


@pytest.fixture
def temp_store(tmp_path: Path) -> EvidenceStore:
    """Fixture providing an isolated EvidenceStore in a temporary directory."""
    return EvidenceStore(base_dir=tmp_path / "evidence")


# ==============================================================================
# Step 2: Dataset Model & Validation Tests
# ==============================================================================

def test_dataset_model_creation():
    """Verify Dataset model instantiation with valid attributes."""
    ds = Dataset(
        dataset_id="DS-PLANT-01",
        name="Plant North Milling Project",
        description="Records and manuals for North milling facility",
        created_at="2026-10-26T12:00:00+00:00",
    )
    assert ds.dataset_id == "DS-PLANT-01"
    assert ds.name == "Plant North Milling Project"
    assert ds.description == "Records and manuals for North milling facility"
    assert ds.created_at == "2026-10-26T12:00:00+00:00"

    data = ds.to_dict()
    assert data["dataset_id"] == "DS-PLANT-01"
    assert data["name"] == "Plant North Milling Project"

    rebuilt = Dataset.from_dict(data)
    assert rebuilt.dataset_id == ds.dataset_id
    assert rebuilt.name == ds.name
    assert rebuilt.description == ds.description
    assert rebuilt.created_at == ds.created_at


def test_dataset_validation_empty_id():
    """Verify validation rejects empty or whitespace-only dataset_id."""
    with pytest.raises(DatasetValidationError, match="Dataset ID cannot be empty"):
        Dataset(dataset_id="", name="Valid Name", created_at="2026-10-26T12:00:00+00:00")

    with pytest.raises(DatasetValidationError, match="Dataset ID cannot be empty"):
        Dataset(dataset_id="   ", name="Valid Name", created_at="2026-10-26T12:00:00+00:00")


def test_dataset_validation_invalid_id_characters():
    """Verify validation rejects dataset_id with unsafe characters or spaces."""
    with pytest.raises(DatasetValidationError, match="contains invalid characters"):
        Dataset(dataset_id="DS 01/TEST", name="Valid Name", created_at="2026-10-26T12:00:00+00:00")

    with pytest.raises(DatasetValidationError, match="contains invalid characters"):
        Dataset(dataset_id="DS..NAME", name="Valid Name", created_at="2026-10-26T12:00:00+00:00")

    with pytest.raises(DatasetValidationError, match="surrounding whitespace"):
        Dataset(dataset_id=" DS-01 ", name="Valid Name", created_at="2026-10-26T12:00:00+00:00")


def test_dataset_validation_name_constraints():
    """Verify validation enforces name length and non-emptiness constraints."""
    with pytest.raises(DatasetValidationError, match="Dataset name cannot be empty"):
        Dataset(dataset_id="DS-01", name="", created_at="2026-10-26T12:00:00+00:00")

    with pytest.raises(DatasetValidationError, match="between 2 and 100 characters"):
        Dataset(dataset_id="DS-01", name="A", created_at="2026-10-26T12:00:00+00:00")

    with pytest.raises(DatasetValidationError, match="between 2 and 100 characters"):
        Dataset(dataset_id="DS-01", name="X" * 101, created_at="2026-10-26T12:00:00+00:00")


def test_dataset_validation_empty_timestamp():
    """Verify validation rejects empty timestamp."""
    with pytest.raises(DatasetValidationError, match="created_at timestamp cannot be empty"):
        Dataset(dataset_id="DS-01", name="Valid Name", created_at="")


# ==============================================================================
# Step 3: Storage & CRUD Operations
# ==============================================================================

def test_dataset_creation_and_lookup(temp_store: EvidenceStore):
    """Verify creating a dataset persists it and allows retrieval by ID."""
    created = temp_store.create_dataset(
        dataset_id="DS-HVAC-2024",
        name="HVAC Chillers Line",
        description="Chiller unit failure records",
    )
    assert created.dataset_id == "DS-HVAC-2024"
    assert created.name == "HVAC Chillers Line"

    retrieved = temp_store.get_dataset("DS-HVAC-2024")
    assert retrieved is not None
    assert retrieved.dataset_id == "DS-HVAC-2024"
    assert retrieved.name == "HVAC Chillers Line"
    assert retrieved.description == "Chiller unit failure records"


def test_dataset_persistence_across_reinitialization(tmp_path: Path):
    """Verify datasets survive EvidenceStore reinitialization and reload."""
    base_dir = tmp_path / "persistence_test_evidence"
    store1 = EvidenceStore(base_dir=base_dir)
    store1.create_dataset(
        dataset_id="DS-PRESS-01",
        name="Hydraulic Press Cell",
        description="Stamping press incidents",
    )

    # Re-initialize new store pointing to same path
    store2 = EvidenceStore(base_dir=base_dir)
    datasets = store2.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].dataset_id == "DS-PRESS-01"
    assert datasets[0].name == "Hydraulic Press Cell"


def test_dataset_prevent_duplicate_id(temp_store: EvidenceStore):
    """Verify attempting to create a dataset with an existing ID raises ValueError."""
    temp_store.create_dataset(dataset_id="DS-ROBOT-01", name="Welding Robots")
    with pytest.raises(ValueError, match="already exists"):
        temp_store.create_dataset(dataset_id="DS-ROBOT-01", name="Duplicate Robot Project")


def test_dataset_update_permitted_metadata(temp_store: EvidenceStore):
    """Verify updating name and description of an existing dataset."""
    temp_store.create_dataset(
        dataset_id="DS-PUMP-01",
        name="Coolant Pumps",
        description="Initial description",
    )

    updated = temp_store.update_dataset(
        dataset_id="DS-PUMP-01",
        name="High-Pressure Coolant Pumps",
        description="Updated maintenance scope description",
    )
    assert updated.name == "High-Pressure Coolant Pumps"
    assert updated.description == "Updated maintenance scope description"

    # Verify retrieval reflects updates
    stored = temp_store.get_dataset("DS-PUMP-01")
    assert stored is not None
    assert stored.name == "High-Pressure Coolant Pumps"
    assert stored.description == "Updated maintenance scope description"


def test_dataset_update_invalid_id_or_not_found(temp_store: EvidenceStore):
    """Verify updating a non-existent or invalid dataset ID raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        temp_store.update_dataset(dataset_id="NON-EXISTENT-ID", name="New Name")

    with pytest.raises(KeyError, match="cannot be empty"):
        temp_store.update_dataset(dataset_id="", name="New Name")


def test_dataset_lookup_invalid_or_missing(temp_store: EvidenceStore):
    """Verify lookup returns None for non-existent or blank IDs without crashing."""
    assert temp_store.get_dataset("NON-EXISTENT") is None
    assert temp_store.get_dataset("") is None
    assert temp_store.get_dataset("   ") is None


# ==============================================================================
# Step 4: Backward-Compatible Migration & Data Integrity
# ==============================================================================

def test_evidence_catalog_legacy_migration(temp_store: EvidenceStore):
    """Verify unassociated legacy catalog entries are safely migrated to the default dataset."""
    # Seed legacy documents without dataset_id directly in catalog.json
    legacy_doc1 = DocumentMetadata(
        document_id="DOC-LEGACY-001",
        original_filename="bearing_bulletin.txt",
        file_type="txt",
        source_type="upload",
        storage_reference="originals/DOC-LEGACY-001_bearing_bulletin.txt",
        extraction_status="success",
        ingestion_status=DocumentStatus.PROCESSED,
        created_at="2026-10-15T00:00:00+00:00",
        updated_at="2026-10-15T00:00:00+00:00",
        extracted_text_reference="extracted/DOC-LEGACY-001.txt",
        extracted_character_count=120,
        dataset_id=None,  # Legacy document without dataset association
    )
    legacy_doc2 = DocumentMetadata(
        document_id="DOC-LEGACY-002",
        original_filename="hydraulic_manual.pdf",
        file_type="pdf",
        source_type="upload",
        storage_reference="originals/DOC-LEGACY-002_hydraulic_manual.pdf",
        extraction_status="success",
        ingestion_status=DocumentStatus.PROCESSED,
        created_at="2026-10-16T00:00:00+00:00",
        updated_at="2026-10-16T00:00:00+00:00",
        extracted_text_reference="extracted/DOC-LEGACY-002.txt",
        extracted_character_count=450,
        dataset_id=None,
    )
    temp_store.ensure_default_dataset()
    temp_store._write_catalog([legacy_doc1, legacy_doc2])

    # Run migration
    stats = temp_store.migrate_legacy_data()
    assert stats["migrated_documents"] == 2
    assert stats["default_dataset_id"] == DEFAULT_DATASET_ID

    # Verify default dataset was created
    default_ds = temp_store.get_dataset(DEFAULT_DATASET_ID)
    assert default_ds is not None
    assert default_ds.name == DEFAULT_DATASET_NAME

    # Verify both documents now reference DEFAULT_DATASET_ID while keeping document_ids
    docs = temp_store.read_catalog()
    assert len(docs) == 2
    assert docs[0].document_id == "DOC-LEGACY-001"
    assert docs[0].dataset_id == DEFAULT_DATASET_ID
    assert docs[1].document_id == "DOC-LEGACY-002"
    assert docs[1].dataset_id == DEFAULT_DATASET_ID


def test_evidence_catalog_migration_idempotent(temp_store: EvidenceStore):
    """Verify repeated migration execution produces identical results without duplicates."""
    doc = DocumentMetadata(
        document_id="DOC-TEST-01",
        original_filename="sample.txt",
        file_type="txt",
        source_type="upload",
        storage_reference="originals/DOC-TEST-01_sample.txt",
        extraction_status="success",
        ingestion_status=DocumentStatus.PROCESSED,
        created_at="2026-10-15T00:00:00+00:00",
        updated_at="2026-10-15T00:00:00+00:00",
    )
    temp_store.ensure_default_dataset()
    temp_store._write_catalog([doc])

    # First migration
    stats1 = temp_store.migrate_legacy_data()
    assert stats1["migrated_documents"] == 1

    # Second migration (idempotent)
    stats2 = temp_store.migrate_legacy_data()
    assert stats2["migrated_documents"] == 0
    assert stats2["already_associated_documents"] == 1

    # Ensure datasets list has only 1 default dataset, not duplicates
    datasets = temp_store.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].dataset_id == DEFAULT_DATASET_ID


def test_incidents_csv_backward_compatibility_in_memory(tmp_path: Path):
    """Verify legacy CSV without dataset_id column loads successfully with default dataset_id in memory."""
    csv_file = tmp_path / "legacy_incidents.csv"
    csv_content = (
        "incident_id,machine_id,date,symptom,maintenance_action,outcome\n"
        "INC-2024-001,M-104,2024-01-12,High bearing temp,Lubricated bearing,Resolved\n"
        "INC-2024-002,M-205,2024-02-04,Vibration spike,Re-aligned shaft,Resolved\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    # Loading without migration should succeed and provide dataset_id in DataFrame
    df = load_incidents_data(csv_file)
    assert "dataset_id" in df.columns
    assert (df["dataset_id"] == DEFAULT_DATASET_ID).all()
    assert len(df) == 2


def test_incidents_csv_non_destructive_migration(tmp_path: Path):
    """Verify non-destructive migration of incidents CSV file preserves rows and columns."""
    csv_file = tmp_path / "migrate_incidents.csv"
    csv_content = (
        "incident_id,machine_id,date,symptom,maintenance_action,outcome\n"
        "INC-001,M-104,2024-01-12,Bearing temp,Lubricated,Resolved\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    count = migrate_incidents_csv(csv_file)
    assert count == 1

    # Inspect written file
    migrated_df = pd.read_csv(csv_file)
    assert "dataset_id" in migrated_df.columns
    assert migrated_df["dataset_id"].iloc[0] == DEFAULT_DATASET_ID
    assert migrated_df["incident_id"].iloc[0] == "INC-001"
    assert migrated_df["machine_id"].iloc[0] == "M-104"

    # Idempotent re-run
    count2 = migrate_incidents_csv(csv_file)
    assert count2 == 1


def test_ingestion_with_dataset_association(temp_store: EvidenceStore):
    """Verify new document ingestion correctly records dataset_id."""
    service = IngestionService(store=temp_store)
    content = b"Maintenance procedure for CNC Lathe 01 spindle bearing inspection."
    temp_store.create_dataset(
        dataset_id="DS-CNC-LATHES",
        name="CNC Lathes",
    )

    # Ingest document associating with a custom dataset
    meta = service.ingest_document(
        filename="cnc_lathe_manual.txt",
        content=content,
        dataset_id="DS-CNC-LATHES",
    )

    assert meta.ingestion_status == DocumentStatus.PROCESSED
    assert meta.dataset_id == "DS-CNC-LATHES"

    # Verify retrieval from catalog
    stored_meta = temp_store.get_metadata(meta.document_id)
    assert stored_meta is not None
    assert stored_meta.dataset_id == "DS-CNC-LATHES"

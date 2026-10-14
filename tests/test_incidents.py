"""Automated unit tests for GhostFault incident data handling and filtering."""

from pathlib import Path
import pytest
import pandas as pd

from src.data_loader import (
    REQUIRED_COLUMNS,
    DatasetLoadError,
    load_incidents_data,
    filter_by_machine,
    get_incident_by_id,
    get_default_dataset_path,
)


def test_default_dataset_exists_and_loads():
    """Verify that sample dataset file exists at expected path and loads successfully."""
    path = get_default_dataset_path()
    assert path.exists(), f"Default sample incidents file missing at {path}"

    df = load_incidents_data(path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 6, "Milestone 1 requires at least 6 synthetic incidents"


def test_required_columns_present():
    """Ensure all required columns exist in the loaded dataset."""
    df = load_incidents_data()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Expected column '{col}' not found in dataset"


def test_unique_incident_ids():
    """Ensure each incident has a non-empty, unique incident_id."""
    df = load_incidents_data()
    incident_ids = df["incident_id"].dropna().tolist()
    assert len(incident_ids) == len(df), "Null values found in incident_id column"
    assert len(incident_ids) == len(set(incident_ids)), "Duplicate incident IDs detected"


def test_multiple_machines_present():
    """Ensure dataset includes multiple distinct machines."""
    df = load_incidents_data()
    distinct_machines = df["machine_id"].unique()
    assert len(distinct_machines) >= 2, "Expected records from multiple machines"


def test_filter_machine_m104():
    """Ensure machine M-104 filter returns at least 3 incidents and only M-104 records."""
    df = load_incidents_data()
    m104_records = filter_by_machine(df, "M-104")
    assert len(m104_records) >= 3, "Expected at least 3 incidents for machine M-104"
    assert (m104_records["machine_id"] == "M-104").all()


def test_filter_machine_m205():
    """Ensure machine M-205 filter returns at least 2 incidents and only M-205 records."""
    df = load_incidents_data()
    m205_records = filter_by_machine(df, "M-205")
    assert len(m205_records) >= 2, "Expected at least 2 incidents for machine M-205"
    assert (m205_records["machine_id"] == "M-205").all()


def test_filter_all_machines():
    """Ensure filtering by 'All Machines' or None returns the full dataset."""
    df = load_incidents_data()
    assert len(filter_by_machine(df, "All Machines")) == len(df)
    assert len(filter_by_machine(df, None)) == len(df)


def test_get_incident_by_id():
    """Verify single record retrieval by incident ID."""
    df = load_incidents_data()
    first_id = df.iloc[0]["incident_id"]
    record = get_incident_by_id(df, first_id)
    assert record is not None
    assert record["incident_id"] == first_id

    # Non-existent ID returns None
    assert get_incident_by_id(df, "NON-EXISTENT-999") is None


def test_missing_csv_raises_error(tmp_path: Path):
    """Verify DatasetLoadError is raised when file does not exist."""
    fake_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(DatasetLoadError) as exc_info:
        load_incidents_data(fake_path)
    assert "not found" in str(exc_info.value).lower()


def test_malformed_csv_missing_columns(tmp_path: Path):
    """Verify DatasetLoadError is raised when columns are missing."""
    malformed_file = tmp_path / "malformed.csv"
    malformed_file.write_text("incident_id,machine_id\nINC-1,M-1\n", encoding="utf-8")
    with pytest.raises(DatasetLoadError) as exc_info:
        load_incidents_data(malformed_file)
    assert "missing required columns" in str(exc_info.value).lower()

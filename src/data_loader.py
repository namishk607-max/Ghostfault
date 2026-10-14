"""Data access and filtering routines for GhostFault incident records."""

from pathlib import Path
from typing import List, Optional
import pandas as pd

REQUIRED_COLUMNS: List[str] = [
    "incident_id",
    "machine_id",
    "date",
    "symptom",
    "maintenance_action",
    "outcome",
]


class DatasetLoadError(Exception):
    """Raised when the incident dataset cannot be read or validated."""


def get_default_dataset_path() -> Path:
    """Return absolute path to sample incidents dataset relative to project root."""
    # src/data_loader.py -> parent is src/ -> parent is project root
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "sample" / "incidents.csv"


def load_incidents_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and validate the maintenance incidents dataset.

    Args:
        csv_path: Path to the CSV file. If None, uses default relative sample path.

    Returns:
        pd.DataFrame containing validated incident records.

    Raises:
        DatasetLoadError: If file is missing, empty, or missing required columns.
    """
    path = csv_path if csv_path is not None else get_default_dataset_path()

    if not path.exists():
        raise DatasetLoadError(f"Incident dataset not found at expected path: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DatasetLoadError(f"Failed to read CSV at {path}: {exc}") from exc

    if df.empty:
        raise DatasetLoadError(f"The incident dataset at {path} is empty.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DatasetLoadError(
            f"Dataset is missing required columns: {missing_cols}. Found columns: {list(df.columns)}"
        )

    return df


def filter_by_machine(df: pd.DataFrame, machine_id: Optional[str]) -> pd.DataFrame:
    """
    Filter incidents dataframe by machine_id.

    Args:
        df: Input DataFrame.
        machine_id: Target machine ID, or None/"All Machines" to return all records.

    Returns:
        Filtered pd.DataFrame.
    """
    if machine_id is None or machine_id.strip() == "" or machine_id == "All Machines":
        return df.copy()

    return df[df["machine_id"] == machine_id].copy()


def get_incident_by_id(df: pd.DataFrame, incident_id: str) -> Optional[pd.Series]:
    """Return single incident record series by ID, or None if not found."""
    matches = df[df["incident_id"] == incident_id]
    if matches.empty:
        return None
    return matches.iloc[0]

"""Data access and filtering routines for GhostFault incident records."""

from pathlib import Path
from typing import Iterable, List, Optional
import pandas as pd

from src.models import (
    DEFAULT_DATASET_ID,
    DatasetAssociationIssue,
    DatasetOwnershipError,
)

REQUIRED_COLUMNS: List[str] = [
    "incident_id",
    "machine_id",
    "date",
    "symptom",
    "maintenance_action",
    "outcome",
]

OPTIONAL_COLUMNS: List[str] = [
    "dataset_id",
]


class DatasetLoadError(Exception):
    """Raised when the incident dataset cannot be read or validated."""


def validate_incident_dataset_ownership(
    incidents_df: pd.DataFrame,
    valid_dataset_ids: Iterable[str],
) -> List[DatasetAssociationIssue]:
    """Report incident rows with missing or unregistered dataset IDs."""
    valid_ids = set(valid_dataset_ids)
    if "dataset_id" not in incidents_df.columns:
        return [
            DatasetAssociationIssue(
                code="missing_dataset_id",
                entity_type="incident",
                entity_id=str(row.get("incident_id", row_index)),
                dataset_id=None,
                message="Incident has no dataset ID.",
            )
            for row_index, row in incidents_df.iterrows()
        ]

    issues: List[DatasetAssociationIssue] = []
    for row_index, row in incidents_df.iterrows():
        incident_id = str(row.get("incident_id", row_index))
        value = row.get("dataset_id")
        if pd.isna(value) or not str(value).strip():
            issues.append(
                DatasetAssociationIssue(
                    code="missing_dataset_id",
                    entity_type="incident",
                    entity_id=incident_id,
                    dataset_id=None,
                    message=f"Incident '{incident_id}' has no dataset ID.",
                )
            )
            continue

        dataset_id = str(value)
        if dataset_id not in valid_ids:
            issues.append(
                DatasetAssociationIssue(
                    code="unregistered_dataset_id",
                    entity_type="incident",
                    entity_id=incident_id,
                    dataset_id=dataset_id,
                    message=(
                        f"Incident '{incident_id}' references unregistered dataset "
                        f"'{dataset_id}'."
                    ),
                )
            )
    return issues


def require_valid_incident_dataset_ownership(
    incidents_df: pd.DataFrame,
    valid_dataset_ids: Iterable[str],
) -> None:
    """Raise a meaningful ownership error when incident dataset references are invalid."""
    issues = validate_incident_dataset_ownership(incidents_df, valid_dataset_ids)
    if issues:
        raise DatasetOwnershipError(" ".join(issue.message for issue in issues))


def get_default_dataset_path() -> Path:
    """Return absolute path to sample incidents dataset relative to project root."""
    # src/data_loader.py -> parent is src/ -> parent is project root
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "sample" / "incidents.csv"


def load_incidents_data(
    csv_path: Optional[Path] = None,
    dataset_id: Optional[str] = None,
    valid_dataset_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load and validate the maintenance incidents dataset.
    Ensures backward compatibility: if dataset_id column is absent, it is populated
    with DEFAULT_DATASET_ID in memory without mutating the source CSV destructively.

    Args:
        csv_path: Path to the CSV file. If None, uses default relative sample path.
        dataset_id: Optional filter for a specific dataset ID.
        valid_dataset_ids: Optional dataset registry used to validate ownership.

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

    # Backward compatibility: populate default dataset_id if not present or contains nulls
    if "dataset_id" not in df.columns:
        df["dataset_id"] = DEFAULT_DATASET_ID
    else:
        df["dataset_id"] = df["dataset_id"].fillna(DEFAULT_DATASET_ID)

    if valid_dataset_ids is not None:
        issues = validate_incident_dataset_ownership(df, valid_dataset_ids)
        if issues:
            raise DatasetLoadError(
                "Invalid incident dataset associations: "
                + " ".join(issue.message for issue in issues)
            )

    if dataset_id is not None and dataset_id.strip():
        df = df[df["dataset_id"] == dataset_id.strip()].copy()

    return df


def migrate_incidents_csv(
    csv_path: Optional[Path] = None,
    default_dataset_id: str = DEFAULT_DATASET_ID,
) -> int:
    """
    Non-destructive, idempotent migration for incidents CSV file.
    If the CSV does not have a dataset_id column, adds it and writes back atomically.
    Returns the number of records associated with default_dataset_id.
    """
    path = csv_path if csv_path is not None else get_default_dataset_path()
    if not path.exists():
        return 0

    df = pd.read_csv(path)
    if df.empty:
        return 0

    if "dataset_id" in df.columns and df["dataset_id"].notna().all():
        return int((df["dataset_id"] == default_dataset_id).sum())

    if "dataset_id" not in df.columns:
        df["dataset_id"] = default_dataset_id
    else:
        df["dataset_id"] = df["dataset_id"].fillna(default_dataset_id)

    # Write atomically
    temp_path = path.with_suffix(".csv.tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)

    return len(df)


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

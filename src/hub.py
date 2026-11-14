"""Central Intelligence Hub logic for GhostFault (Milestone 7.2).

Provides testable helper functions for project-wide and dataset-specific
summary calculations. All functions read from existing data sources without
mutating CSV files, evidence catalogs, or dataset registries.
"""

from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
import pandas as pd

from src.models import (
    DocumentMetadata,
    DocumentStatus,
    Dataset,
    DatasetValidationError,
    DEFAULT_DATASET_ID,
)
from src.storage import EvidenceStore
from src.data_loader import load_incidents_data, DatasetLoadError


# ==============================================================================
# Project-Wide Summary Calculations
# ==============================================================================


def compute_project_summary(
    datasets: List[Dataset],
    incidents_df: Optional[pd.DataFrame],
    documents: List[DocumentMetadata],
) -> Dict[str, Any]:
    """
    Compute project-wide summary metrics from existing persisted data.

    Args:
        datasets: All registered datasets from the dataset catalog.
        incidents_df: Loaded incidents DataFrame (may include in-memory dataset_id).
        documents: All document metadata records from the evidence catalog.

    Returns:
        Dictionary of project-wide summary counts and metadata.
    """
    total_datasets = len(datasets)
    total_incidents = len(incidents_df) if incidents_df is not None else 0
    total_documents = len(documents)

    processed_documents = sum(
        1 for d in documents if d.ingestion_status == DocumentStatus.PROCESSED
    )
    failed_documents = total_documents - processed_documents

    unique_machines = 0
    unique_symptoms = 0
    if incidents_df is not None and not incidents_df.empty:
        if "machine_id" in incidents_df.columns:
            unique_machines = incidents_df["machine_id"].nunique()
        if "symptom" in incidents_df.columns:
            unique_symptoms = incidents_df["symptom"].nunique()

    return {
        "total_datasets": total_datasets,
        "total_incidents": total_incidents,
        "total_documents": total_documents,
        "processed_documents": processed_documents,
        "failed_documents": failed_documents,
        "unique_machines": unique_machines,
        "unique_symptoms": unique_symptoms,
    }


# ==============================================================================
# Dataset Registry Calculations
# ==============================================================================


def compute_dataset_registry(
    datasets: List[Dataset],
    incidents_df: Optional[pd.DataFrame],
    documents: List[DocumentMetadata],
) -> List[Dict[str, Any]]:
    """
    Compute per-dataset summary rows for the dataset registry view.

    Uses actual dataset_id associations from incidents and documents.
    Does not create, modify, or duplicate any datasets.

    Args:
        datasets: All registered datasets.
        incidents_df: Loaded incidents DataFrame with dataset_id column.
        documents: All document metadata records.

    Returns:
        List of dictionaries, one per dataset, with counts and metadata.
    """
    registry_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        # Count incidents associated with this dataset
        incident_count = 0
        if incidents_df is not None and not incidents_df.empty:
            if "dataset_id" in incidents_df.columns:
                incident_count = int(
                    (incidents_df["dataset_id"] == ds.dataset_id).sum()
                )

        # Count documents associated with this dataset
        doc_count = sum(
            1 for doc in documents if doc.dataset_id == ds.dataset_id
        )
        processed_doc_count = sum(
            1 for doc in documents
            if doc.dataset_id == ds.dataset_id
            and doc.ingestion_status == DocumentStatus.PROCESSED
        )

        registry_rows.append({
            "dataset_id": ds.dataset_id,
            "name": ds.name,
            "description": ds.description or "",
            "created_at": ds.created_at,
            "incident_count": incident_count,
            "document_count": doc_count,
            "processed_document_count": processed_doc_count,
        })

    return registry_rows


def compute_unassociated_counts(
    datasets: List[Dataset],
    incidents_df: Optional[pd.DataFrame],
    documents: List[DocumentMetadata],
) -> Dict[str, int]:
    """
    Count incidents and documents not associated with any registered dataset.

    These represent orphaned records or legacy data not yet migrated.

    Returns:
        Dictionary with unassociated_incidents and unassociated_documents counts.
    """
    dataset_ids = {ds.dataset_id for ds in datasets}

    unassociated_incidents = 0
    if incidents_df is not None and not incidents_df.empty:
        if "dataset_id" in incidents_df.columns:
            unassociated_incidents = int(
                (~incidents_df["dataset_id"].isin(dataset_ids)).sum()
            )

    unassociated_documents = sum(
        1 for doc in documents
        if doc.dataset_id is None or doc.dataset_id not in dataset_ids
    )

    return {
        "unassociated_incidents": unassociated_incidents,
        "unassociated_documents": unassociated_documents,
    }


def find_missing_document_associations(
    documents: List[DocumentMetadata],
) -> List[DocumentMetadata]:
    """Find documents with no dataset assignment."""
    return [doc for doc in documents if not doc.dataset_id or not doc.dataset_id.strip()]


# ==============================================================================
# Incident Overview Calculations
# ==============================================================================


def compute_incident_overview(
    incidents_df: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:
    """
    Build a compact incident overview table from the loaded incidents DataFrame.

    Uses only columns present in the existing schema. Does not modify
    the source DataFrame or CSV file.

    Returns:
        List of dictionaries for rendering as a table.
    """
    if incidents_df is None or incidents_df.empty:
        return []

    rows: List[Dict[str, Any]] = []
    for _, row in incidents_df.iterrows():
        entry: Dict[str, Any] = {
            "Incident ID": str(row.get("incident_id", "")),
            "Machine ID": str(row.get("machine_id", "Unassigned")),
            "Date": str(row.get("date", "Not recorded")),
            "Symptom": str(row.get("symptom", "")),
            "Dataset": str(row.get("dataset_id", DEFAULT_DATASET_ID)),
        }
        rows.append(entry)

    return rows


# ==============================================================================
# Evidence Document Summary
# ==============================================================================


def compute_evidence_summary(
    documents: List[DocumentMetadata],
) -> Dict[str, Any]:
    """
    Compute summary statistics for all cataloged evidence documents.

    Returns:
        Dictionary with document counts by status and type.
    """
    total = len(documents)
    processed = sum(
        1 for d in documents if d.ingestion_status == DocumentStatus.PROCESSED
    )
    failed = total - processed

    # Count by file type
    type_counts: Dict[str, int] = {}
    for doc in documents:
        ft = doc.file_type.upper() if doc.file_type else "UNKNOWN"
        type_counts[ft] = type_counts.get(ft, 0) + 1

    # Total extracted characters
    total_chars = sum(d.extracted_character_count for d in documents)

    return {
        "total_documents": total,
        "processed_documents": processed,
        "failed_documents": failed,
        "type_counts": type_counts,
        "total_extracted_characters": total_chars,
    }


# ==============================================================================
# Recent Activity (M7.2 scope: only from persisted timestamps)
# ==============================================================================


def compute_recent_documents(
    documents: List[DocumentMetadata],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return the most recently ingested documents based on persisted created_at timestamps.

    Does not fabricate timestamps or create activity history.

    Args:
        documents: All document metadata records.
        limit: Maximum number of recent documents to return.

    Returns:
        List of dictionaries with recent document metadata.
    """
    if not documents:
        return []

    # Sort by created_at descending (ISO timestamps sort lexicographically)
    sorted_docs = sorted(documents, key=lambda d: d.created_at, reverse=True)

    recent: List[Dict[str, Any]] = []
    for doc in sorted_docs[:limit]:
        recent.append({
            "document_id": doc.document_id,
            "original_filename": doc.original_filename,
            "file_type": doc.file_type.upper() if doc.file_type else "UNKNOWN",
            "status": doc.ingestion_status.value.upper(),
            "created_at": doc.created_at[:19].replace("T", " ") if doc.created_at else "Unknown",
            "dataset_id": doc.dataset_id or "Unassociated",
        })

    return recent


def compute_recent_investigation_activity(
    history_store: Any,
    evidence_store: EvidenceStore,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Aggregate recent activity by querying each registered dataset separately."""
    events: List[Dict[str, Any]] = []
    for dataset in evidence_store.list_datasets():
        events.extend(
            history_store.list_activity(
                evidence_store,
                dataset.dataset_id,
                limit=limit,
            )
        )
    events.sort(
        key=lambda event: (event["occurred_at"], event["event_id"]),
        reverse=True,
    )
    return events[:limit]


# ==============================================================================
# Navigation Targets (Milestone 7.2 & 7.3)
# ==============================================================================

# Navigation page identifiers used by the hub and sidebar
NAV_HUB = "Central Intelligence Hub"
NAV_WORKSPACE = "Dataset Workspace"
NAV_INCIDENTS = "Maintenance Incidents"
NAV_EVIDENCE_INGESTION = "Evidence Ingestion"
NAV_EVIDENCE_SEARCH = "Evidence Search"
NAV_FAULT_ANALYSIS = "Fault Analysis"
NAV_CROSS_INCIDENT = "Cross-Incident Analysis"
NAV_INVESTIGATION_HISTORY = "Investigation History & Activity"
NAV_EQUIPMENT = "Equipment Intelligence"

NAVIGATION_PAGES = [
    NAV_HUB,
    NAV_WORKSPACE,
    NAV_INCIDENTS,
    NAV_EVIDENCE_INGESTION,
    NAV_EVIDENCE_SEARCH,
    NAV_FAULT_ANALYSIS,
    NAV_CROSS_INCIDENT,
    NAV_INVESTIGATION_HISTORY,
    NAV_EQUIPMENT,
]


# ==============================================================================
# Milestone 7.3: Active Dataset State & Resolution
# ==============================================================================

SESSION_KEY_ACTIVE_DATASET = "active_dataset_id"
SESSION_KEY_NAV = "ghostfault_nav"
SESSION_KEY_NAV_WIDGET = "nav_radio"


def validate_active_dataset_id(
    dataset_id: Optional[str],
    valid_dataset_ids: Iterable[str],
) -> bool:
    """
    Validate whether a dataset_id is a non-empty string that exists in valid_dataset_ids.

    Args:
        dataset_id: Candidate dataset identifier.
        valid_dataset_ids: Iterable of registered dataset identifiers.

    Returns:
        True if valid and present, False otherwise.
    """
    if not dataset_id or not isinstance(dataset_id, str):
        return False
    return dataset_id.strip() in set(valid_dataset_ids)


def resolve_active_dataset(
    current_active_id: Optional[str],
    registered_datasets: List[Dataset],
    default_id: str = DEFAULT_DATASET_ID,
) -> Tuple[Optional[Dataset], str, Optional[str]]:
    """
    Resolve the active dataset object and identifier against registered datasets.

    Rules:
    - If current_active_id matches a registered dataset:
        returns (matching_dataset, current_active_id, None)
    - If registered_datasets is empty:
        returns (None, default_id, "No registered datasets available in the registry.")
    - If current_active_id is None or empty:
        resolves to default_id (or first registered dataset) with no warning.
    - If current_active_id is not in registered_datasets (missing, deleted, or invalid):
        resolves to default_id (or first registered dataset) with a clear warning:
        "Active dataset '{current_active_id}' is invalid or was not found in the registry. Reset to '{fallback.name}' ({fallback_id})."

    Args:
        current_active_id: Dataset ID currently stored in session state.
        registered_datasets: All datasets from persistent storage.
        default_id: Fallback dataset ID (defaults to DEFAULT_DATASET_ID).

    Returns:
        Tuple of (resolved_dataset, resolved_dataset_id, warning_message).
    """
    if not registered_datasets:
        return None, default_id, "No registered datasets available in the registry."

    dataset_map = {ds.dataset_id: ds for ds in registered_datasets}

    # If valid and present in registry
    if current_active_id and current_active_id in dataset_map:
        return dataset_map[current_active_id], current_active_id, None

    # Fallback dataset: prefer default_id, otherwise first registered dataset
    fallback_ds = dataset_map.get(default_id, registered_datasets[0])
    fallback_id = fallback_ds.dataset_id

    # If current_active_id was None or empty, standard initialization without warning
    if not current_active_id or not str(current_active_id).strip():
        return fallback_ds, fallback_id, None

    # Invalid or deleted active dataset ID
    warning = (
        f"Active dataset '{current_active_id}' is invalid or was not found in the registry. "
        f"Reset to '{fallback_ds.name}' ({fallback_id})."
    )
    return fallback_ds, fallback_id, warning


def get_active_dataset_id(
    session_state: Dict[str, Any],
    default_id: str = DEFAULT_DATASET_ID,
) -> str:
    """
    Retrieve active dataset ID from session state dictionary.

    Falls back to default_id if not set or invalid.
    """
    val = session_state.get(SESSION_KEY_ACTIVE_DATASET)
    if val and isinstance(val, str) and val.strip():
        return val.strip()
    return default_id


def set_active_dataset_id(
    session_state: Dict[str, Any],
    dataset_id: str,
    valid_dataset_ids: Optional[Iterable[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Update active dataset ID in session state dictionary with validation.

    Args:
        session_state: Streamlit session state or standard dictionary.
        dataset_id: Candidate dataset identifier to activate.
        valid_dataset_ids: Optional iterable of registered dataset IDs to validate against.

    Returns:
        Tuple of (success: bool, error_message: Optional[str]).
    """
    if not dataset_id or not isinstance(dataset_id, str) or not dataset_id.strip():
        return False, "Dataset ID cannot be empty."

    clean_id = dataset_id.strip()
    if valid_dataset_ids is not None:
        valid_set = set(valid_dataset_ids)
        if clean_id not in valid_set:
            return False, f"Dataset ID '{clean_id}' is not present in registered datasets."

    previous_id = session_state.get(SESSION_KEY_ACTIVE_DATASET)
    if previous_id and previous_id != clean_id:
        for key in list(session_state.keys()):
            if isinstance(key, str) and (
                key.startswith("analysis_result_")
                or key.startswith("investigation_run_id_")
            ):
                del session_state[key]
        session_state.pop("selected_investigation_run_id", None)
    session_state[SESSION_KEY_ACTIVE_DATASET] = clean_id
    return True, None


def get_navigation_page(
    session_state: Dict[str, Any],
    default_page: str = NAV_HUB,
) -> str:
    """Return the authoritative navigation page, or its default."""
    val = session_state.get(SESSION_KEY_NAV)
    if isinstance(val, str) and val in NAVIGATION_PAGES:
        return val
    return default_page


def set_navigation_page(
    session_state: Dict[str, Any],
    page: str,
    valid_pages: Optional[Iterable[str]] = None,
) -> bool:
    """Update canonical navigation state without changing the radio widget key."""
    valid = set(valid_pages) if valid_pages is not None else set(NAVIGATION_PAGES)
    if not isinstance(page, str) or page not in valid:
        return False
    session_state[SESSION_KEY_NAV] = page
    return True


def sync_navigation_widget_state(
    session_state: Dict[str, Any],
    valid_pages: Optional[Iterable[str]] = None,
) -> str:
    """Copy canonical navigation state to the radio key before creating its widget."""
    valid = list(valid_pages) if valid_pages is not None else NAVIGATION_PAGES
    default_page = valid[0] if valid else NAV_HUB
    page = get_navigation_page(session_state, default_page)
    if page not in valid:
        page = default_page
    if session_state.get(SESSION_KEY_NAV) != page:
        session_state[SESSION_KEY_NAV] = page
    session_state[SESSION_KEY_NAV_WIDGET] = page
    return page


def sync_navigation_from_widget(
    session_state: Dict[str, Any],
    valid_pages: Optional[Iterable[str]] = None,
) -> bool:
    """Copy a user-selected radio value into canonical navigation state."""
    return set_navigation_page(
        session_state,
        session_state.get(SESSION_KEY_NAV_WIDGET),
        valid_pages,
    )


# ==============================================================================
# Milestone 7.3: Workspace Summary Calculations
# ==============================================================================


def compute_workspace_summary(
    dataset_id: str,
    incidents_df: Optional[pd.DataFrame],
    documents: List[DocumentMetadata],
) -> Dict[str, Any]:
    """
    Compute dataset-specific summary metrics and record lists for a given dataset ID.

    Matches records strictly using their actual dataset_id.
    Does not mutate source records, rewrite CSVs, or reassign datasets.

    Args:
        dataset_id: The ID of the dataset to summarize.
        incidents_df: Loaded incidents DataFrame.
        documents: All document metadata records from the evidence catalog.

    Returns:
        Dictionary containing:
        - dataset_id: str
        - total_incidents: int
        - incidents: List[Dict[str, Any]]
        - unique_machines: List[str]
        - unique_symptoms: List[str]
        - total_documents: int
        - processed_documents: int
        - failed_documents: int
        - documents: List[DocumentMetadata]
        - total_extracted_characters: int
    """
    if not dataset_id or not isinstance(dataset_id, str):
        target_id = ""
    else:
        target_id = dataset_id.strip()

    # 1. Incidents matching dataset_id
    matched_incidents: List[Dict[str, Any]] = []
    unique_machines: List[str] = []
    unique_symptoms: List[str] = []

    if incidents_df is not None and not incidents_df.empty and target_id:
        if "dataset_id" in incidents_df.columns:
            subset = incidents_df[incidents_df["dataset_id"] == target_id]
        else:
            # Legacy CSV: only matches if target_id == DEFAULT_DATASET_ID
            if target_id == DEFAULT_DATASET_ID:
                subset = incidents_df
            else:
                subset = pd.DataFrame()

        if not subset.empty:
            if "machine_id" in subset.columns:
                unique_machines = sorted(
                    [str(m) for m in subset["machine_id"].dropna().unique().tolist()]
                )
            if "symptom" in subset.columns:
                unique_symptoms = sorted(
                    [str(s) for s in subset["symptom"].dropna().unique().tolist()]
                )

            for _, row in subset.iterrows():
                matched_incidents.append({
                    "incident_id": str(row.get("incident_id", "")),
                    "machine_id": str(row.get("machine_id", "Unassigned")),
                    "date": str(row.get("date", "Not recorded")),
                    "symptom": str(row.get("symptom", "")),
                    "maintenance_action": str(row.get("maintenance_action", "")),
                    "outcome": str(row.get("outcome", "")),
                    "dataset_id": str(row.get("dataset_id", target_id)),
                })

    # 2. Documents matching dataset_id
    matched_docs = [
        doc for doc in documents
        if doc.dataset_id is not None and doc.dataset_id == target_id
    ]
    processed_count = sum(
        1 for d in matched_docs if d.ingestion_status == DocumentStatus.PROCESSED
    )
    failed_count = sum(
        1 for d in matched_docs if d.ingestion_status != DocumentStatus.PROCESSED
    )
    total_extracted_chars = sum(d.extracted_character_count for d in matched_docs)

    return {
        "dataset_id": target_id,
        "total_incidents": len(matched_incidents),
        "incidents": matched_incidents,
        "unique_machines": unique_machines,
        "unique_symptoms": unique_symptoms,
        "total_documents": len(matched_docs),
        "processed_documents": processed_count,
        "failed_documents": failed_count,
        "documents": matched_docs,
        "total_extracted_characters": total_extracted_chars,
    }


# ==============================================================================
# Milestone 7.3: Integrity & Validation Inspection Helpers
# ==============================================================================


def find_invalid_document_associations(
    documents: List[DocumentMetadata],
    valid_dataset_ids: Iterable[str],
) -> List[DocumentMetadata]:
    """
    Find documents with a non-empty dataset_id absent from the registry.

    Args:
        documents: All document metadata records from the evidence catalog.
        valid_dataset_ids: Iterable of registered dataset identifiers.

    Returns:
        Documents with dangling dataset references. Missing associations are
        returned by find_missing_document_associations instead.
    """
    valid_set = set(valid_dataset_ids)
    return [
        doc for doc in documents
        if doc.dataset_id
        and doc.dataset_id.strip()
        and doc.dataset_id.strip() not in valid_set
    ]


def find_invalid_incident_associations(
    incidents_df: Optional[pd.DataFrame],
    valid_dataset_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    """
    Find incidents with a non-empty dataset_id absent from the registry.

    Missing IDs are returned by find_missing_incident_associations instead.

    Args:
        incidents_df: Incidents DataFrame.
        valid_dataset_ids: Iterable of registered dataset identifiers.

    Returns:
        List of incident record dictionaries with invalid or unregistered dataset associations.
    """
    if incidents_df is None or incidents_df.empty or "dataset_id" not in incidents_df.columns:
        return []

    valid_set = set(valid_dataset_ids)
    has_dataset_id = incidents_df["dataset_id"].notna() & incidents_df["dataset_id"].astype(str).str.strip().ne("")
    invalid_mask = has_dataset_id & ~incidents_df["dataset_id"].isin(valid_set)
    invalid_rows = incidents_df[invalid_mask]
    return invalid_rows.to_dict(orient="records")


def find_missing_incident_associations(
    incidents_df: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:
    """Find incidents with a blank dataset ID when the field is present."""
    if incidents_df is None or incidents_df.empty or "dataset_id" not in incidents_df.columns:
        return []
    missing_mask = incidents_df["dataset_id"].isna() | incidents_df["dataset_id"].astype(str).str.strip().eq("")
    return incidents_df[missing_mask].to_dict(orient="records")


# ==============================================================================
# Milestone 7.4: Dataset Creation Operations
# ==============================================================================


def create_dataset_entry(
    store: EvidenceStore,
    name: str,
    description: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> Tuple[bool, Optional[Dataset], Optional[str]]:
    """
    Validate parameters, auto-generate collision-free ID if needed, and persist a new dataset.

    Enforces validation:
    - Rejects blank or empty names.
    - Enforces length between 2 and 100 characters.
    - Prevents duplicate dataset IDs.
    - Persists using EvidenceStore.create_dataset.

    Args:
        store: EvidenceStore instance.
        name: Name of the dataset.
        description: Optional description.
        dataset_id: Optional dataset ID. If None, auto-generated automatically.

    Returns:
        Tuple of (success: bool, dataset: Optional[Dataset], error_message: Optional[str]).
    """
    if name is None or not str(name).strip():
        return False, None, "Dataset name cannot be blank."

    clean_name = str(name).strip()
    if len(clean_name) < 2 or len(clean_name) > 100:
        return (
            False,
            None,
            f"Dataset name must be between 2 and 100 characters. Given: {len(clean_name)}",
        )

    clean_desc = str(description).strip() if description and str(description).strip() else None

    # Handle dataset ID: generate automatically if not specified
    if dataset_id and str(dataset_id).strip():
        clean_id = str(dataset_id).strip()
        if store.get_dataset(clean_id) is not None:
            return False, None, f"Dataset with ID '{clean_id}' already exists."
    else:
        # Auto-generate unique ID guaranteed not to collide
        cand_id = store.generate_dataset_id()
        while store.get_dataset(cand_id) is not None:
            cand_id = store.generate_dataset_id()
        clean_id = cand_id

    try:
        new_ds = store.create_dataset(
            dataset_id=clean_id,
            name=clean_name,
            description=clean_desc,
        )
        return True, new_ds, None
    except (DatasetValidationError, ValueError) as exc:
        return False, None, str(exc)
    except Exception as exc:
        return False, None, f"Failed to persist dataset: {exc}"

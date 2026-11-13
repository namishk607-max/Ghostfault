"""GhostFault - Streamlit Application.

Evidence-first industrial failure intelligence dashboard foundation.
Includes:
- Milestone 1: Incident dashboard, filters, inspector.
- Milestone 2: Evidence document ingestion & management.
- Milestone 3: Chunking, indexing, and deterministic retrieval.
- Milestone 4 / 4.1 / 4.1.2: Evidence-grounded fault analysis & substantive relevance validation.
- Milestone 7.2: Central Intelligence Hub, dataset registry, and navigation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import sqlite3
import uuid
from typing import Optional, Dict, Any, List

from src.data_loader import (
    load_incidents_data,
    filter_by_machine,
    get_incident_by_id,
    DatasetLoadError,
)
from src.ingestion import IngestionService
from src.storage import EvidenceStore
from src.models import (
    DocumentStatus,
    AnalysisStatus,
    EvidenceContextType,
    MachineScope,
    PassageQuality,
    EvidenceCounts,
    VerificationStatus,
    EvidenceStrength,
    RecurrenceScope,
    EvidenceLinkStatus,
    IncidentTimelineEntry,
    SymptomGroup,
    GroupEvidenceLinkResult,
    CrossIncidentEvidenceLinkingResult,
    DEFAULT_DATASET_ID,
    Dataset,
    DatasetOwnershipError,
)
from src.indexer import SearchIndex
from src.retriever import EvidenceRetriever
from src.analyzer import FaultAnalysisEngine
from src.ai_gateway import (
    GatewayError,
    load_gateway_config,
    request_commentary,
)
from src.cross_incident import CrossIncidentComparisonEngine, generate_incident_timeline
from src.evidence_linker import CrossIncidentEvidenceLinker
from src.history import HistoryStoreError, InvestigationHistoryStore
from src.hub import (
    compute_project_summary,
    compute_dataset_registry,
    compute_unassociated_counts,
    compute_incident_overview,
    compute_evidence_summary,
    compute_recent_documents,
    compute_recent_investigation_activity,
    compute_workspace_summary,
    resolve_active_dataset,
    set_active_dataset_id,
    sync_navigation_widget_state,
    sync_navigation_from_widget,
    find_invalid_document_associations,
    find_missing_document_associations,
    find_invalid_incident_associations,
    find_missing_incident_associations,
    create_dataset_entry,
    get_navigation_page,
    set_navigation_page,
    NAVIGATION_PAGES,
    NAV_HUB,
    NAV_WORKSPACE,
    NAV_INCIDENTS,
    NAV_EVIDENCE_INGESTION,
    NAV_EVIDENCE_SEARCH,
    NAV_FAULT_ANALYSIS,
    NAV_CROSS_INCIDENT,
    NAV_INVESTIGATION_HISTORY,
)


def navigate_to(page: str) -> None:
    """Set canonical navigation state from a widget callback."""
    if not set_navigation_page(st.session_state, page, NAVIGATION_PAGES):
        st.session_state["navigation_error"] = f"Invalid navigation target: {page}"


def sync_sidebar_navigation() -> None:
    """Persist a sidebar radio selection before the app reruns."""
    if not sync_navigation_from_widget(st.session_state, NAVIGATION_PAGES):
        st.session_state["navigation_error"] = "The selected navigation page is invalid."


def select_active_dataset(widget_key: str, valid_dataset_ids: List[str]) -> None:
    """Persist a dataset widget selection in the canonical active-dataset state."""
    success, error = set_active_dataset_id(
        st.session_state,
        st.session_state.get(widget_key),
        valid_dataset_ids,
    )
    if not success:
        st.session_state["active_dataset_error"] = error


def open_dataset_workspace(dataset_id: str, valid_dataset_ids: List[str]) -> None:
    """Select a registered dataset and route to its workspace from a callback."""
    success, error = set_active_dataset_id(
        st.session_state,
        dataset_id,
        valid_dataset_ids,
    )
    if not success:
        st.session_state["active_dataset_error"] = error
        return
    navigate_to(NAV_WORKSPACE)


def open_created_dataset_workspace() -> None:
    """Open the dataset created in the hub after its registry reload."""
    dataset_id = st.session_state.get("dataset_create_last_id")
    valid_dataset_ids = st.session_state.get("registered_dataset_ids", [])
    if not dataset_id:
        st.session_state["active_dataset_error"] = "No recently created dataset is available."
        return
    success, error = set_active_dataset_id(
        st.session_state,
        dataset_id,
        valid_dataset_ids,
    )
    if not success:
        st.session_state["active_dataset_error"] = error
        return
    st.session_state.pop("dataset_create_success_msg", None)
    navigate_to(NAV_WORKSPACE)


def migrate_legacy_document_associations(store: EvidenceStore) -> None:
    """Run the legacy association migration only after an explicit hub action."""
    st.session_state["legacy_migration_result"] = store.migrate_legacy_data()

# Page configuration
st.set_page_config(
    page_title="GhostFault - Industrial Failure Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-adaptive styling
st.markdown(
    """
    <style>
    .ghost-title {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
    }
    .ghost-subtitle {
        font-size: 1.05rem;
        opacity: 0.8;
        margin-top: 2px;
        margin-bottom: 1.5rem;
    }
    .planned-badge {
        display: inline-block;
        background-color: rgba(148, 163, 184, 0.15);
        color: inherit;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        margin-left: 8px;
        vertical-align: middle;
    }
    .status-badge-success {
        display: inline-block;
        background-color: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .status-badge-failed {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .status-badge-warning {
        display: inline-block;
        background-color: rgba(234, 179, 8, 0.15);
        color: #ca8a04;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(234, 179, 8, 0.4);
    }
    .badge-scope-exact {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.15);
        color: #2563eb;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .badge-scope-general {
        display: inline-block;
        background-color: rgba(100, 116, 139, 0.15);
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }
    .badge-context-illustrative {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        color: #d97706;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-excerpt {
        display: inline-block;
        background-color: rgba(168, 85, 247, 0.15);
        color: #9333ea;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .badge-excluded {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .badge-verification-supported {
        display: inline-block;
        background-color: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .badge-verification-contradicted {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-verification-inconclusive {
        display: inline-block;
        background-color: rgba(234, 179, 8, 0.15);
        color: #ca8a04;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(234, 179, 8, 0.4);
    }
    .badge-strength-strong {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #059669;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-strength-limited {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        color: #d97706;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-strength-insufficient {
        display: inline-block;
        background-color: rgba(148, 163, 184, 0.15);
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_ingestion_service() -> IngestionService:
    """Initialize and cache the evidence ingestion service instance."""
    return IngestionService()


@st.cache_resource
def get_search_components() -> tuple[SearchIndex, EvidenceRetriever, FaultAnalysisEngine, CrossIncidentComparisonEngine, CrossIncidentEvidenceLinker]:
    """Initialize and cache the SearchIndex, EvidenceRetriever, FaultAnalysisEngine, and CrossIncident services."""
    index = SearchIndex()
    retriever = EvidenceRetriever(index, store=EvidenceStore())
    analyzer = FaultAnalysisEngine(retriever)
    comparison_engine = CrossIncidentComparisonEngine(store=EvidenceStore())
    evidence_linker = CrossIncidentEvidenceLinker(retriever)
    return index, retriever, analyzer, comparison_engine, evidence_linker


@st.cache_resource
def get_investigation_history_store() -> InvestigationHistoryStore:
    """Initialize and cache the separate persistent investigation history store."""
    import os
    env_history_db = os.environ.get("GHOSTFAULT_HISTORY_DB")
    if env_history_db:
        allow_mig = os.environ.get("GHOSTFAULT_ALLOW_MIGRATION", "").lower() in ("1", "true", "yes")
        return InvestigationHistoryStore(Path(env_history_db), allow_migration=allow_mig)
    return InvestigationHistoryStore()


def render_evidence_ingestion(
    service: IngestionService,
    search_index: SearchIndex,
    active_dataset: Optional[Dataset] = None,
) -> None:
    """Render Milestone 2 Evidence Ingestion interface with auto-indexing."""
    st.subheader("📥 Evidence Ingestion & Document Management")
    st.markdown(
        "Upload technical maintenance manuals, operator logs, or failure sheets (`.pdf`, `.txt`). "
        "Files are validated, stored securely, cataloged, and indexed for deterministic evidence retrieval."
    )

    if active_dataset is None:
        st.error("Evidence ingestion requires a registered active dataset.")
        return

    st.info(
        f"**Active dataset:** {active_dataset.name} (`{active_dataset.dataset_id}`). "
        "New uploads will be associated with this workspace."
    )

    upload_col, info_col = st.columns([3, 2])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Choose a maintenance document to ingest",
            type=["pdf", "txt"],
            help="Select a text-based PDF or TXT file. Scanned PDFs without OCR text layers will be flagged.",
        )

        if uploaded_file is not None:
            if st.button("Process & Ingest Document", key="btn_ingest_doc"):
                with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                    content_bytes = uploaded_file.getvalue()
                    target_ds_id = active_dataset.dataset_id
                    meta = service.ingest_document(
                        filename=uploaded_file.name,
                        content=content_bytes,
                        source_type="dashboard_upload",
                        dataset_id=target_ds_id,
                    )

                if meta.ingestion_status == DocumentStatus.PROCESSED:
                    extracted_text = service.store.get_extracted_text(meta)
                    if extracted_text:
                        chunks_indexed = search_index.index_document(
                            meta,
                            extracted_text,
                            store=service.store,
                        )
                        st.success(
                            f"✅ Document successfully ingested and indexed! Assigned ID: `{meta.document_id}` "
                            f"({meta.extracted_character_count:,} characters extracted into {chunks_indexed} chunks)."
                        )
                    else:
                        st.success(
                            f"✅ Document ingested! Assigned ID: `{meta.document_id}`, but no extracted text was available to index."
                        )
                else:
                    st.error(
                        f"❌ Ingestion Failed (`{meta.error_code}`): {meta.error_message}"
                    )

    with info_col:
        with st.container(border=True):
            st.markdown("**Ingestion & Search Pipeline Specifications:**")
            st.markdown(
                """
                - **Supported Formats:** Text-based `.pdf`, `.txt`
                - **Collision Strategy:** Collision-resistant UUID `DOC-XXXXXXXXXXXX`
                - **Chunking Strategy:** Sentence-aligned window (600 chars, 100 overlap)
                - **Search Index:** Local persistent SQLite FTS5 with BM25 ranking
                - **Storage:** Local isolated filesystem under `data/evidence/`
                """
            )

    all_docs = service.list_documents()
    target_ds_id = active_dataset.dataset_id
    scoped_docs = [d for d in all_docs if d.dataset_id == target_ds_id]

    st.markdown(f"#### 📄 Evidence Documents: **{active_dataset.name}**")
    st.caption(f"Showing documents cataloged specifically under workspace `{target_ds_id}`.")

    if scoped_docs:
        tot_docs = len(scoped_docs)
        succ_docs = sum(1 for d in scoped_docs if d.ingestion_status == DocumentStatus.PROCESSED)
        fail_docs = tot_docs - succ_docs
        
        met_col1, met_col2, met_col3 = st.columns(3)
        with met_col1:
            st.metric("Workspace Documents", tot_docs)
        with met_col2:
            st.metric("Successfully Ingested", succ_docs)
        with met_col3:
            st.metric("Failed Ingestions", fail_docs)

        catalog_table_data = []
        for d in scoped_docs:
            catalog_table_data.append({
                "Document ID": d.document_id,
                "Original Filename": d.original_filename,
                "Type": d.file_type.upper(),
                "Status": d.ingestion_status.value.upper(),
                "Extracted Chars": d.extracted_character_count,
                "Ingested At (UTC)": d.created_at[:19].replace("T", " "),
            })

        st.dataframe(
            pd.DataFrame(catalog_table_data),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Document Inspector & Text Preview (Workspace Scope)")
        doc_options = {d.document_id: f"{d.document_id} — {d.original_filename} ({d.ingestion_status.value})" for d in scoped_docs}
        selected_doc_id = st.selectbox(
            "Select cataloged document to inspect:",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            key=f"doc_inspector_select_{target_ds_id}",
        )

        selected_meta = service.store.get_metadata(selected_doc_id)
        if selected_meta is not None:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.caption("Document ID")
                    st.markdown(f"**`{selected_meta.document_id}`**")
                with c2:
                    st.caption("Status")
                    badge_class = "status-badge-success" if selected_meta.ingestion_status == DocumentStatus.PROCESSED else "status-badge-failed"
                    st.markdown(f'<span class="{badge_class}">{selected_meta.ingestion_status.value.upper()}</span>', unsafe_allow_html=True)
                with c3:
                    st.caption("Original File")
                    st.markdown(f"**{selected_meta.original_filename}**")
                with c4:
                    st.caption("Character Count")
                    st.markdown(f"**{selected_meta.extracted_character_count:,} chars**")

                if selected_meta.error_message:
                    st.warning(f"**Ingestion Error [{selected_meta.error_code}]:** {selected_meta.error_message}")

                if selected_meta.extracted_text_reference:
                    text_content = service.store.get_extracted_text(selected_meta)
                    if text_content:
                        st.caption("Extracted Text Preview:")
                        preview_length = 3000
                        if len(text_content) > preview_length:
                            st.text_area(
                                "Previewing first 3,000 characters",
                                value=text_content[:preview_length] + "\n\n... [Content truncated for preview]",
                                height=200,
                                disabled=True,
                            )
                        else:
                            st.text_area("Full Extracted Text", value=text_content, height=200, disabled=True)
                    else:
                        st.info("No extracted text file found on disk.")
    else:
        st.info(
            f"No evidence documents have been uploaded to **{active_dataset.name if active_dataset else 'this workspace'}** (`{target_ds_id}`) yet. "
            "Use the upload form above to add documents to this workspace."
        )

    # Optional Project-Wide Catalog View
    if all_docs:
        with st.expander("🌐 Project-Wide Document Catalog (All Datasets)", expanded=False):
            st.caption(f"Showing all {len(all_docs)} cataloged documents across all datasets:")
            pw_table = []
            for d in all_docs:
                pw_table.append({
                    "Document ID": d.document_id,
                    "Original Filename": d.original_filename,
                    "Type": d.file_type.upper(),
                    "Status": d.ingestion_status.value.upper(),
                    "Dataset ID": d.dataset_id or "Unassociated",
                    "Extracted Chars": d.extracted_character_count,
                    "Ingested At (UTC)": d.created_at[:19].replace("T", " "),
                })
            st.dataframe(
                pd.DataFrame(pw_table),
                use_container_width=True,
                hide_index=True,
            )


def render_evidence_search(
    retriever: EvidenceRetriever,
    search_index: SearchIndex,
    store: EvidenceStore,
    active_dataset: Optional[Dataset] = None,
) -> None:
    """Render Milestone 3 Evidence Search & Index Management interface."""
    st.subheader("🔎 Evidence Search & Passage Retrieval")
    st.markdown(
        "Query ingested maintenance records, manuals, and technical notes using deterministic keyword search. "
        "Every retrieved passage is ranked using SQLite FTS5 BM25 and includes full provenance tracking back to its source document."
    )
    sync_result = st.session_state.pop("index_sync_result", None)
    if sync_result is not None:
        if sync_result["errors"]:
            st.error(
                f"Index sync completed with {len(sync_result['errors'])} error(s)."
            )
            for error in sync_result["errors"]:
                st.warning(error)
        else:
            st.success(
                f"Synced {sync_result['indexed_documents']} docs "
                f"({sync_result['total_chunks']} chunks)."
            )
        for item in sync_result.get("informational", []):
            st.info(item["message"])

    rebuild_result = st.session_state.pop("index_rebuild_result", None)
    if rebuild_result is not None:
        if rebuild_result["errors"]:
            st.error(
                f"Index rebuild completed with {len(rebuild_result['errors'])} error(s)."
            )
            for error in rebuild_result["errors"]:
                st.warning(error)
        else:
            st.success(f"Rebuilt index with {rebuild_result['total_chunks']} chunks.")
        for item in rebuild_result.get("informational", []):
            st.info(item["message"])

    if active_dataset is None:
        st.error("Evidence Search requires a registered active dataset. Select a dataset before searching.")
        return

    st.info(
        f"**Active dataset:** {active_dataset.name} (`{active_dataset.dataset_id}`). "
        "Search results are restricted to catalog-owned documents and chunks in this dataset."
    )
    st.caption(
        "Index statistics and Sync/Rebuild actions below cover the project-wide index; search results remain scoped to the active dataset."
    )

    stats = search_index.get_index_stats()
    with st.container(border=True):
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns([2, 2, 3, 3])
        with stat_c1:
            st.metric("Indexed Documents", stats["document_count"])
        with stat_c2:
            st.metric("Searchable Chunks", stats["chunk_count"])
        with stat_c3:
            last_ts = stats["last_indexed_at"][:19].replace("T", " ") if stats["last_indexed_at"] else "Never"
            st.metric("Last Index Sync", last_ts)
        with stat_c4:
            st.caption("Index Maintenance Actions")
            action_btn_c1, action_btn_c2 = st.columns(2)
            with action_btn_c1:
                if st.button("🔄 Sync Index", help="Synchronize index with all processed catalog documents"):
                    sync_info = search_index.sync_with_store(store)
                    st.session_state["index_sync_result"] = sync_info
                    st.rerun()
            with action_btn_c2:
                if st.button("⚡ Rebuild Index", help="Wipe and rebuild full search index from catalog"):
                    try:
                        rebuild_info = search_index.rebuild_index(store)
                    except DatasetOwnershipError as exc:
                        st.error(f"Index rebuild was not performed: {exc}")
                    else:
                        st.session_state["index_rebuild_result"] = rebuild_info
                        st.rerun()

    target_ds_id = active_dataset.dataset_id
    integrity_issues = search_index.validate_dataset_integrity(store, target_ds_id)
    if integrity_issues:
        st.error(
            f"Evidence Search is disabled for this dataset: {len(integrity_issues)} "
            "catalog/index integrity issue(s) prevent trusting its indexed evidence. "
            "No ownership was changed. Review the diagnostics and resolve the catalog/index "
            "mismatch explicitly before searching."
        )
        with st.expander("Dataset retrieval integrity diagnostics", expanded=True):
            for issue in integrity_issues:
                st.warning(issue.message)

    catalog = store.read_catalog()
    scoped_docs = [
        d for d in catalog
        if d.dataset_id == target_ds_id and d.ingestion_status == DocumentStatus.PROCESSED
    ]
    doc_filter_options = ["All Documents"] + [d.document_id for d in scoped_docs]

    search_col1, search_col2, search_col3 = st.columns([5, 2, 1])
    with search_col1:
        query = st.text_input(
            "Search evidence corpus",
            placeholder="e.g., hydraulic pressure filter, bearing acoustic pitch, vibration",
            key="search_query_input",
        )
    with search_col2:
        selected_doc_filter = st.selectbox(
            "Filter by Source Document",
            options=doc_filter_options,
            index=0,
            key="search_doc_filter",
        )
    with search_col3:
        max_results = st.number_input("Max Results", min_value=1, max_value=50, value=5, step=1)

    st.caption(
        "ℹ️ **Note on Search Scores:** Scores represent SQLite FTS5 BM25 keyword relevance, "
        "not diagnostic confidence or probability of machine failure."
    )

    if query.strip() and not integrity_issues:
        with st.spinner("Searching evidence chunks..."):
            all_datasets = store.list_datasets()
            valid_ids = [d.dataset_id for d in all_datasets]
            try:
                results = retriever.search(
                    query=query,
                    limit=int(max_results),
                    document_id_filter=selected_doc_filter,
                    dataset_id=target_ds_id,
                    valid_dataset_ids=valid_ids,
                )
            except DatasetOwnershipError as exc:
                st.error(f"Evidence search could not safely return results: {exc}")
                return

        if results:
            st.markdown(f"#### Retrieved Evidence Passages ({len(results)} match{'es' if len(results) > 1 else ''})")
            for i, res in enumerate(results, start=1):
                chunk = res.chunk
                with st.container(border=True):
                    head_c1, head_c2, head_c3 = st.columns([3, 2, 1])
                    with head_c1:
                        st.markdown(f"**#{i} Source:** `{chunk.source_filename}`")
                    with head_c2:
                        st.caption(f"Doc ID: `{chunk.document_id}` | Chunk #{chunk.chunk_index}")
                    with head_c3:
                        st.markdown(f"**Score:** `{res.score:.2f}`")

                    st.markdown(f"**Matched Passage:**\n\n> {res.matched_snippet}")

                    meta_tags = []
                    if res.matched_terms:
                        meta_tags.append(f"**Matched Terms:** `{'`, `'.join(res.matched_terms)}`")
                    if chunk.start_offset is not None and chunk.end_offset is not None:
                        meta_tags.append(f"**Char Offsets:** `{chunk.start_offset} - {chunk.end_offset}`")
                    if chunk.page_number is not None:
                        meta_tags.append(f"**Page:** `{chunk.page_number}`")
                    else:
                        meta_tags.append("**Page:** *Unavailable (flattened extracted text)*")

                    st.markdown(" • ".join(meta_tags))
        else:
            st.info(f"No matching evidence passages found for query: `{query}`. Try different keywords or sync the index.")
    elif query.strip():
        st.info("Search was not run because this dataset has unresolved catalog/index integrity issues.")


def render_fault_analysis(
    analyzer: FaultAnalysisEngine,
    selected_record: Optional[pd.Series],
    active_dataset: Optional[Dataset] = None,
    history_store: Optional[InvestigationHistoryStore] = None,
    evidence_store: Optional[EvidenceStore] = None,
) -> None:
    """Render Milestone 4.1.2 Evidence-Grounded Fault Analysis interface."""
    st.subheader("🧠 Evidence-Grounded Fault Analysis")
    st.markdown(
        "Generate explainable, evidence-backed investigative hypotheses for the selected incident. "
        "The engine validates passage substance, excludes low-value keyword lists, deduplicates identical source copies, "
        "and distinguishes machine scopes with complete provenance traceability."
    )

    if active_dataset is None:
        st.error("Fault analysis requires a registered active dataset.")
        return

    st.info(
        f"**Active dataset:** {active_dataset.name} (`{active_dataset.dataset_id}`). "
        "Incident ownership and evidence provenance are validated against this dataset."
    )

    if selected_record is None:
        st.info("Please select an incident above to perform fault analysis.")
        return

    incident_dict = selected_record.to_dict()
    incident_id = incident_dict.get("incident_id", "UNKNOWN")
    target_ds_id = active_dataset.dataset_id

    analysis_key = f"analysis_result_{incident_id}_{target_ds_id}"
    run_id_key = f"investigation_run_id_{incident_id}_{target_ds_id}"

    btn_col, status_col = st.columns([2, 5])
    with btn_col:
        run_analysis = st.button("🚀 Run Fault Analysis", key=f"btn_run_analysis_{incident_id}")

    if run_analysis or analysis_key in st.session_state:
        if run_analysis:
            st.session_state[run_id_key] = str(uuid.uuid4())
            with st.spinner("Analyzing incident symptoms against retrieved evidence corpus..."):
                try:
                    analysis_result = analyzer.analyze_incident(
                        incident_dict,
                        dataset_id=target_ds_id,
                    )
                except DatasetOwnershipError as exc:
                    st.error(f"Fault analysis was not performed: {exc}")
                    st.session_state.pop(analysis_key, None)
                    st.session_state.pop(run_id_key, None)
                    return
                st.session_state[analysis_key] = analysis_result
        else:
            analysis_result = st.session_state[analysis_key]

        if history_store is None or evidence_store is None:
            st.error("Investigation history storage is unavailable; the analysis was not saved.")
        else:
            run_id = st.session_state.get(run_id_key)
            if not run_id:
                run_id = str(uuid.uuid4())
                st.session_state[run_id_key] = run_id
            persisted_key = f"investigation_history_saved_{run_id}"
            if not st.session_state.get(persisted_key):
                try:
                    saved = history_store.save_run(
                        evidence_store,
                        run_id,
                        target_ds_id,
                        incident_dict,
                        analysis_result,
                    )
                except (HistoryStoreError, DatasetOwnershipError, sqlite3.Error) as exc:
                    st.error(f"Investigation result could not be saved to history: {exc}")
                else:
                    st.session_state[persisted_key] = True
                    if saved["created"]:
                        st.success("Investigation snapshot saved to history.")

        with status_col:
            if analysis_result.status == AnalysisStatus.COMPLETED:
                st.markdown(
                    f'Analysis Status: <span class="status-badge-success">COMPLETED ({len(analysis_result.hypotheses)} HYPOTHESIS)</span>',
                    unsafe_allow_html=True,
                )
            elif analysis_result.status == AnalysisStatus.INSUFFICIENT_EVIDENCE:
                st.markdown(
                    'Analysis Status: <span class="status-badge-warning">INSUFFICIENT EVIDENCE</span>',
                    unsafe_allow_html=True,
                )
            elif analysis_result.status == AnalysisStatus.NO_EVIDENCE:
                st.markdown(
                    'Analysis Status: <span class="status-badge-failed">NO MATCHING EVIDENCE</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'Analysis Status: <span class="status-badge-failed">FAILED: {analysis_result.error_message}</span>',
                    unsafe_allow_html=True,
                )

        # Milestone 4.1.2: Evidence Count Pipeline Metrics Bar
        counts: Optional[EvidenceCounts] = getattr(analysis_result, "evidence_counts", None)
        if counts:
            with st.container(border=True):
                st.caption("**Evidence Pipeline Derivation Counts:**")
                cnt_c1, cnt_c2, cnt_c3, cnt_c4, cnt_c5 = st.columns(5)
                with cnt_c1:
                    st.metric("Raw Retrieved", counts.raw_retrieved_count)
                with cnt_c2:
                    st.metric("Unique Passages", counts.unique_deduplicated_count)
                with cnt_c3:
                    st.metric("Substantive Retained", counts.substantive_retained_count)
                with cnt_c4:
                    st.metric("Excluded Low-Value", counts.excluded_low_value_count)
                with cnt_c5:
                    st.metric("Hypothesis-Supporting", counts.supporting_hypotheses_count)

        st.markdown("---")

        col_hyp, col_ev = st.columns([5, 4])

        with col_hyp:
            st.markdown("#### 🔬 Investigative Hypotheses")
            if analysis_result.hypotheses:
                for idx, hyp in enumerate(analysis_result.hypotheses, start=1):
                    with st.container(border=True):
                        st.markdown(f"##### Hypothesis #{idx}: {hyp.title}")
                        st.markdown(f"*{hyp.description}*")
                        st.caption(f"**Qualification:** {hyp.qualification}")

                        # Milestone 6.1: Deterministic Evidence Verification Status
                        v_status = getattr(hyp, "verification_status", VerificationStatus.INCONCLUSIVE)
                        v_expl = getattr(hyp, "verification_explanation", "")
                        if v_status == VerificationStatus.SUPPORTED:
                            v_badge = '<span class="badge-verification-supported">SUPPORTED</span>'
                        elif v_status == VerificationStatus.CONTRADICTED:
                            v_badge = '<span class="badge-verification-contradicted">CONTRADICTED</span>'
                        else:
                            v_badge = '<span class="badge-verification-inconclusive">INCONCLUSIVE</span>'

                        # Milestone 6.2: Qualitative Evidence Strength
                        e_strength = getattr(hyp, "evidence_strength", EvidenceStrength.INSUFFICIENT)
                        e_strength_expl = getattr(hyp, "evidence_strength_explanation", "")
                        if e_strength == EvidenceStrength.STRONG:
                            s_badge = '<span class="badge-strength-strong">STRONG</span>'
                        elif e_strength == EvidenceStrength.LIMITED:
                            s_badge = '<span class="badge-strength-limited">LIMITED</span>'
                        else:
                            s_badge = '<span class="badge-strength-insufficient">INSUFFICIENT</span>'

                        st.markdown(
                            f"**Evidence Verification:** {v_badge} &nbsp;|&nbsp; **Evidence Strength:** {s_badge}",
                            unsafe_allow_html=True,
                        )
                        if v_expl:
                            st.caption(f"ℹ️ **Verification:** {v_expl}")
                        if e_strength_expl:
                            st.caption(f"📊 **Strength Assessment:** {e_strength_expl}")

                        # Milestone 6.2: Grounded Missing Evidence Guidance
                        missing_guidance = getattr(hyp, "missing_evidence_guidance", "No specific evidence gap identified.")
                        if missing_guidance and missing_guidance != "No specific evidence gap identified.":
                            st.info(f"🔍 **Missing Evidence Guidance:** {missing_guidance}")
                        else:
                            st.caption("🔍 **Missing Evidence Guidance:** No specific evidence gap identified.")

                        # Retrieved Supporting References (accurate wording separating incident-specific from general)
                        if hyp.supporting_evidence:
                            st.markdown(f"**Retrieved Supporting References ({len(hyp.supporting_evidence)}):**")
                            for s_ref in hyp.supporting_evidence:
                                scope = getattr(s_ref, "machine_scope", MachineScope.GENERAL_REFERENCE)
                                ref_machine = getattr(s_ref, "referenced_machine_id", None)
                                ctx_type = getattr(s_ref, "context_type", EvidenceContextType.GENERAL_REFERENCE)
                                is_excerpt = getattr(s_ref, "is_excerpt", False)
                                dup_copies = getattr(s_ref, "duplicate_source_copies", None)

                                scope_badge = (
                                    f'<span class="badge-scope-exact">Machine {ref_machine} Record</span> '
                                    if scope == MachineScope.EXACT_MACHINE
                                    else '<span class="badge-scope-general">General Technical Reference</span> '
                                )
                                context_badge = (
                                    '<span class="badge-context-illustrative">⚠️ Illustrative Test Values</span> '
                                    if ctx_type == EvidenceContextType.ILLUSTRATIVE_TEST_VALUE
                                    else ""
                                )
                                excerpt_badge = (
                                    '<span class="badge-excerpt">Excerpt</span> '
                                    if is_excerpt
                                    else ""
                                )

                                st.markdown(
                                    f"- **Source:** `{s_ref.source_filename}` (`{s_ref.document_id}` / Chunk #{s_ref.chunk_index}) "
                                    f"{scope_badge}{context_badge}{excerpt_badge}\n"
                                    f"  > {s_ref.snippet}\n"
                                    f"  *Support Basis:* {s_ref.relevance_reason}",
                                    unsafe_allow_html=True,
                                )

                                if dup_copies:
                                    dup_list = ", ".join(
                                        [f"`{c['source_filename']}` (`{c['document_id']}`)" for c in dup_copies]
                                    )
                                    st.caption(f"  ℹ️ *Identical/near-duplicate passage copies detected in:* {dup_list} *(grouped to prevent false independent corroboration)*")

                        # Contradictory or Cross-Machine Evidence
                        if hyp.contradictory_evidence:
                            st.markdown("⚠️ **Contradictory / Cross-Machine Flags:**")
                            for c_ref in hyp.contradictory_evidence:
                                st.warning(
                                    f"**Cross-Unit Alert in `{c_ref.source_filename}` (`{c_ref.document_id}`):**\n\n"
                                    f"{c_ref.relevance_reason}\n\n"
                                    f"> {c_ref.snippet}"
                                )

                        # Recommended Technical Checks
                        st.markdown("**Suggested Technical Checks (Requires OEM Procedures & Human Judgment):**")
                        for check in hyp.suggested_checks:
                            st.markdown(f"- 🔧 {check}")
            else:
                st.info(
                    "No diagnostic hypotheses could be substantiated. "
                    "The deterministic rule engine requires concrete, substantive corroborating evidence in the index "
                    "before proposing an investigative hypothesis."
                )

        with col_ev:
            st.markdown("#### 📑 Grounding Evidence Passages")
            substantive_count_label = counts.substantive_retained_count if counts else len(analysis_result.retrieved_evidence)
            st.caption(f"Retained {substantive_count_label} substantive passages matching incident keywords.")
            if analysis_result.retrieved_evidence:
                for ev in analysis_result.retrieved_evidence:
                    with st.container(border=True):
                        st.markdown(f"**Source:** `{ev.chunk.source_filename}` (`{ev.chunk.document_id}`)")
                        st.markdown(f"> {ev.matched_snippet}")
                        st.caption(f"Chunk #{ev.chunk.chunk_index} • BM25 Score: `{ev.score:.2f}`")
            else:
                st.warning("No evidence passages retrieved for this incident.")

            # Milestone 4.1.2: Excluded or Low-Value Retrieval Results Section
            excluded_list = getattr(analysis_result, "excluded_evidence", None)
            if excluded_list:
                st.markdown("#### 🚫 Excluded or Low-Value Retrieval Results")
                st.caption("Passages containing only indexing search terms or header metadata are excluded from grounding hypotheses:")
                for exc in excluded_list:
                    with st.container(border=True):
                        st.markdown(f"**Source:** `{exc.source_filename}` (`{exc.document_id}`) <span class='badge-excluded'>Excluded</span>", unsafe_allow_html=True)
                        st.markdown(f"> {exc.snippet}")
                        st.caption(f"⚠️ *Reason for exclusion:* {exc.exclusion_reason}")

            st.markdown("#### ⚠️ Missing Information & Scope Limitations")
            with st.container(border=True):
                st.markdown("**Missing Diagnostic Data:**")
                for missing in analysis_result.missing_information:
                    st.markdown(f"- ❓ {missing}")

                st.markdown("---")
                st.markdown("**Safety & Technical Limitations:**")
                for lim in analysis_result.limitations:
                    st.markdown(f"- ⚠️ {lim}")

        st.markdown("---")
        st.markdown("#### Optional AI Commentary")
        st.caption(
                "AI-generated, unverified commentary is separate from deterministic analysis and is never "
                "saved to investigation history. If enabled, only the selected incident's symptom, "
                "maintenance action, and up to three deterministic hypothesis summaries are sent to the "
                "configured gateway; evidence passages and dataset identifiers are not sent."
        )
        st.warning(
                "The configured AI provider may process operational information. Enable this only if "
                "sharing the selected incident summary with that provider is permitted."
        )
        ai_enabled = st.checkbox(
            "Enable optional AI commentary for this session",
            key=f"ai_commentary_enabled_{target_ds_id}",
            value=False,
        )
        if ai_enabled:
            try:
                try:
                    streamlit_secrets = st.secrets
                except FileNotFoundError:
                    streamlit_secrets = {}
                gateway_config = load_gateway_config(
                    True,
                    secrets=streamlit_secrets,
                )
            except GatewayError as exc:
                st.info(str(exc))
            else:
                run_id = st.session_state.get(run_id_key)
                if not run_id:
                    run_id = str(uuid.uuid4())
                    st.session_state[run_id_key] = run_id
                commentary_key = f"ai_commentary_{target_ds_id}_{run_id}"
                if st.button(
                    "Request AI-generated, unverified commentary",
                    key=f"btn_ai_commentary_{target_ds_id}_{run_id}",
                ):
                    with st.spinner("Requesting separate AI commentary..."):
                        try:
                            commentary = request_commentary(
                                gateway_config,
                                incident_dict,
                                analysis_result,
                                target_ds_id,
                            )
                        except GatewayError as exc:
                            st.error(str(exc))
                            st.session_state.pop(commentary_key, None)
                        else:
                            st.session_state[commentary_key] = commentary
                saved_commentary = st.session_state.get(commentary_key)
                if saved_commentary:
                    st.warning("AI-generated, unverified commentary — not evidence or a verified finding.")
                    st.text(saved_commentary)


def render_incident_timeline(
    df: pd.DataFrame,
    dataset_id: Optional[str] = None,
    store: Optional[EvidenceStore] = None,
) -> None:
    """Render Milestone 5.2 Incident Timeline section."""
    st.subheader("⏱️ Incident Timeline")
    st.markdown(
        "Chronological progression of maintenance incidents based on verified incident dates. "
        "*Note: Chronological proximity or ordering does not establish causation or verified root cause.*"
    )
    if dataset_id:
        st.caption(f"Dataset scope: `{dataset_id}`")

    if df.empty:
        st.info("No incident records available to construct a timeline.")
        return

    dated_entries, undated_entries, missing_counts = generate_incident_timeline(
        df,
        dataset_id=dataset_id,
        store=store,
    )

    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.metric("Dated Incidents", len(dated_entries))
    with t_col2:
        st.metric("Undated / Unverified Dates", len(undated_entries))
    with t_col3:
        if dated_entries:
            span_text = f"{dated_entries[0].parsed_date} → {dated_entries[-1].parsed_date}"
        else:
            span_text = "N/A"
        st.metric("Temporal Span", span_text)

    # 1. Chronological Timeline Table for Dated Incidents
    if dated_entries:
        st.markdown("##### 📅 Chronological Sequence")
        timeline_rows = []
        for entry in dated_entries:
            timeline_rows.append({
                "Date": entry.parsed_date,
                "Incident ID": entry.incident_id,
                "Machine ID": entry.machine_id or "Unassigned",
                "Reported Symptom": entry.symptom,
                "Maintenance Action Taken": entry.maintenance_action or "Not recorded",
                "Outcome / Verification": entry.outcome or "Not recorded",
            })
        timeline_df = pd.DataFrame(timeline_rows)
        st.dataframe(
            timeline_df,
            use_container_width=True,
            column_config={
                "Date": st.column_config.TextColumn("Date (ISO)", width="small"),
                "Incident ID": st.column_config.TextColumn("Incident ID", width="small"),
                "Machine ID": st.column_config.TextColumn("Machine ID", width="small"),
                "Reported Symptom": st.column_config.TextColumn("Reported Symptom", width="large"),
                "Maintenance Action Taken": st.column_config.TextColumn("Maintenance Action Taken", width="large"),
                "Outcome / Verification": st.column_config.TextColumn("Outcome / Verification", width="large"),
            },
            hide_index=True,
        )

    # 2. Undated / Unverified Date Incidents
    if undated_entries:
        st.markdown("##### ⚠️ Incidents with Missing or Unverified Dates")
        st.caption("The following records lack a verified date format and cannot be placed on the chronological timeline without guessing:")
        undated_rows = []
        for entry in undated_entries:
            undated_rows.append({
                "Incident ID": entry.incident_id,
                "Machine ID": entry.machine_id or "Unassigned",
                "Raw Date Value": entry.raw_date or "None / Blank",
                "Reported Symptom": entry.symptom,
                "Maintenance Action Taken": entry.maintenance_action or "Not recorded",
            })
        undated_df = pd.DataFrame(undated_rows)
        st.dataframe(
            undated_df,
            use_container_width=True,
            column_config={
                "Incident ID": st.column_config.TextColumn("Incident ID", width="small"),
                "Machine ID": st.column_config.TextColumn("Machine ID", width="small"),
                "Raw Date Value": st.column_config.TextColumn("Raw Date Field", width="medium"),
                "Reported Symptom": st.column_config.TextColumn("Reported Symptom", width="large"),
                "Maintenance Action Taken": st.column_config.TextColumn("Maintenance Action Taken", width="large"),
            },
            hide_index=True,
        )


def render_cross_incident_analysis(
    df: pd.DataFrame,
    comparison_engine: CrossIncidentComparisonEngine,
    evidence_linker: CrossIncidentEvidenceLinker,
    active_dataset: Optional[Dataset] = None,
) -> None:
    """Render Milestone 5.4 Cross-Incident Comparison & Evidence Linking interface."""
    st.subheader("🔗 Cross-Incident Comparison & Evidence Grounding")
    st.markdown(
        "Deterministic cross-incident pattern comparison and substantive evidence linking. "
        "Groups matching normalized symptom profiles across maintenance history and connects them to verified "
        "evidence documentation."
    )

    if active_dataset is None:
        st.error("Cross-incident analysis requires a registered active dataset.")
        return

    st.info(
        f"**Active dataset:** {active_dataset.name} (`{active_dataset.dataset_id}`). "
        "Incident comparison, timelines, and evidence links are restricted to this dataset."
    )

    # Safety Notice
    st.info(
        "⚠️ **Safety Notice:** Evidence links provide context for investigation; they do not prove a common root cause. "
        "Shared technical guidance or recurring symptoms across different machines must not be assumed to be identical physical failures."
    )

    if df.empty:
        st.info(
            f"No incidents are registered in **{active_dataset.name}** "
            f"(`{active_dataset.dataset_id}`); cross-incident comparison has no records to process."
        )
        return

    target_ds_id = active_dataset.dataset_id

    # Deterministic comparison using M5.1 engine (scoped to dataset)
    try:
        comparison_result = comparison_engine.compare_incidents(df, dataset_id=target_ds_id)
    except DatasetOwnershipError as exc:
        st.error(f"Cross-incident comparison was not performed: {exc}")
        return

    # Top-level metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Incidents Compared", comparison_result.total_incidents_analyzed)
    with m_col2:
        st.metric("Symptom Groups", len(comparison_result.symptom_groups))
    with m_col3:
        recurrent_count = sum(1 for g in comparison_result.symptom_groups if g.recurrence_scope in {RecurrenceScope.SAME_MACHINE, RecurrenceScope.CROSS_MACHINE, RecurrenceScope.MIXED})
        st.metric("Recurrent Groups", recurrent_count)
    with m_col4:
        singleton_count = sum(1 for g in comparison_result.symptom_groups if g.recurrence_scope == RecurrenceScope.SINGLETON)
        st.metric("Unique / Singleton", singleton_count)

    # Execute M5.3 Evidence Linking (scoped to dataset)
    try:
        link_result = evidence_linker.link_comparison_evidence(
            comparison_result,
            dataset_id=target_ds_id,
        )
    except DatasetOwnershipError as exc:
        st.error(f"Cross-incident evidence linking was not performed: {exc}")
        return

    # Render each symptom group and its evidence links
    st.markdown("##### 🔍 Symptom Groups & Evidence Links")
    for group_res in link_result.group_results:
        # Badge for recurrence scope
        if group_res.recurrence_scope == RecurrenceScope.SAME_MACHINE:
            scope_badge = "<span class='badge-scope-exact'>Same Machine Recurrence</span>"
        elif group_res.recurrence_scope == RecurrenceScope.CROSS_MACHINE:
            scope_badge = "<span class='badge-scope-cross'>Cross-Machine Recurrence</span>"
        elif group_res.recurrence_scope == RecurrenceScope.MIXED:
            scope_badge = "<span class='badge-scope-cross'>Mixed Machine Scope</span>"
        else:
            scope_badge = "<span class='badge-scope-general'>Singleton / Isolated</span>"

        # Badge for link status
        if group_res.status == EvidenceLinkStatus.SUPPORTED:
            status_badge = "<span class='badge-context-oem'>Supported by Substantive Evidence</span>"
        elif group_res.status == EvidenceLinkStatus.INSUFFICIENT_SUBSTANTIVE_EVIDENCE:
            status_badge = "<span class='badge-excluded'>Insufficient Substantive Evidence</span>"
        else:
            status_badge = "<span class='badge-scope-general'>No Supporting Evidence Found</span>"

        with st.container(border=True):
            header_col1, header_col2 = st.columns([3, 2])
            with header_col1:
                st.markdown(f"**Symptom Group:** {group_res.canonical_label}")
                st.caption(f"Group ID: `{group_res.group_id}` | Normalized Key: `{group_res.normalized_symptom}`")
            with header_col2:
                st.markdown(f"{scope_badge} &nbsp; {status_badge}", unsafe_allow_html=True)
                st.caption(f"Related Incidents ({len(group_res.incident_ids)}): `{'`, `'.join(group_res.incident_ids)}`")

            # Evidence linking details
            if group_res.supporting_evidence:
                st.markdown(f"**Verified Evidence Passages ({len(group_res.supporting_evidence)} substantive):**")
                for ev in group_res.supporting_evidence:
                    with st.container(border=True):
                        ev_c1, ev_c2 = st.columns([3, 1])
                        with ev_c1:
                            st.markdown(f"**Source Document:** `{ev.source_filename}` (`{ev.document_id}`)")
                            st.caption(f"Chunk ID: `{ev.chunk_id}` | Chunk #{ev.chunk_index}")
                        with ev_c2:
                            if ev.retrieval_score is not None:
                                st.markdown(f"**Score:** `{ev.retrieval_score:.2f}`")

                        # Machine scope badge
                        if ev.machine_scope == MachineScope.EXACT_MACHINE:
                            mach_badge = f"<span class='badge-scope-exact'>Exact Machine ({ev.referenced_machine_id})</span>"
                        elif ev.machine_scope == MachineScope.CROSS_MACHINE_CONFLICT:
                            mach_badge = f"<span class='badge-scope-cross'>Cross-Machine Conflict ({ev.referenced_machine_id})</span>"
                        else:
                            mach_badge = "<span class='badge-scope-general'>General Technical Reference</span>"

                        # Context badge
                        if ev.context_type == EvidenceContextType.ILLUSTRATIVE_TEST_VALUE:
                            ctx_badge = "<span class='badge-context-illustrative'>Illustrative Test Value</span>"
                        elif ev.context_type == EvidenceContextType.VERIFIED_OEM_LIMIT:
                            ctx_badge = "<span class='badge-context-oem'>Verified OEM Limit</span>"
                        else:
                            ctx_badge = "<span class='badge-scope-general'>General Reference</span>"

                        st.markdown(f"{mach_badge} &nbsp; {ctx_badge}", unsafe_allow_html=True)
                        st.markdown(f"> {ev.snippet}")
                        st.caption(f"ℹ️ **Relevance Reason:** {ev.relevance_reason}")
                        st.caption(f"🔗 **Link Explanation:** {ev.link_explanation}")

                        if ev.matched_query_terms:
                            st.caption(f"**Matched Query Terms:** `{'`, `'.join(ev.matched_query_terms)}`")

                        # Duplicate source copies audit
                        if ev.duplicate_source_copies:
                            with st.expander(f"Audited Duplicate Source Copies ({len(ev.duplicate_source_copies)})"):
                                for dup in ev.duplicate_source_copies:
                                    st.markdown(
                                        f"- Document ID: `{dup.get('document_id')}` | File: `{dup.get('source_filename')}` | "
                                        f"Chunk #{dup.get('chunk_index')} (`{dup.get('chunk_id')}`)"
                                    )

                        # Provenance limitations
                        if ev.provenance_limitations:
                            for plim in ev.provenance_limitations:
                                st.caption(f"⚠️ *Provenance Note:* {plim}")

            elif group_res.status == EvidenceLinkStatus.INSUFFICIENT_SUBSTANTIVE_EVIDENCE:
                st.warning(
                    "All retrieved passages for this symptom group consisted exclusively of search-term lists or "
                    "administrative document headers and were excluded from substantive link support."
                )
            else:
                st.info(
                    f"No substantive maintenance evidence was found in the local index matching query terms: `{group_res.query_used}`. "
                    "This indicates an evidence gap in the local documentation, not proof that a fault does not exist."
                )

            # Excluded evidence drawer if any
            if group_res.excluded_evidence:
                with st.expander(f"Excluded Low-Value or Metadata Passages ({len(group_res.excluded_evidence)})"):
                    for exc in group_res.excluded_evidence:
                        st.markdown(f"**Excluded Document:** `{exc.source_filename}` (`{exc.document_id}`)")
                        st.markdown(f"> {exc.snippet}")
                        st.caption(f"⚠️ Reason: Non-substantive passage ({exc.quality.value})")

    # Global Limitations Drawer
    if link_result.global_limitations or comparison_result.limitations:
        with st.expander("Analytical Boundaries & Data Quality Limitations"):
            st.markdown("**Comparison & Temporal Limitations:**")
            for lim in comparison_result.limitations:
                st.markdown(f"- ⚠️ **{lim.code}:** {lim.description}")
            st.markdown("---")
            st.markdown("**Evidence Linking Boundaries:**")
            for glim in link_result.global_limitations:
                st.markdown(f"- ⚠️ **{glim.code}:** {glim.description}")


def render_dataset_workspace(
    store: EvidenceStore,
    incidents_df: Optional[pd.DataFrame],
    active_dataset: Dataset,
    active_warning: Optional[str] = None,
) -> None:
    """Render the Milestone 7.3 Dataset Workspace for the currently selected active dataset."""
    top_c1, top_c2 = st.columns([4, 1])
    with top_c1:
        st.markdown(f"### 📁 Dataset Workspace: **{active_dataset.name}**")
        st.caption(f"Dataset ID: `{active_dataset.dataset_id}` — Isolated Workspace")
    with top_c2:
        st.write("")
        st.button(
            "⬅️ Hub Overview",
            key="ws_top_btn_hub",
            help="Return to Central Intelligence Hub",
            use_container_width=True,
            on_click=navigate_to,
            args=(NAV_HUB,),
        )

    st.markdown(
        "Dedicated workspace for inspecting incidents, evidence documents, and metadata "
        "associated with the currently active dataset."
    )

    if active_warning:
        st.warning(f"⚠️ **Selection Notice:** {active_warning}")

    # Header Card: Active Workspace Identity
    with st.container(border=True):
        hdr_c1, hdr_c2 = st.columns([3, 1])
        with hdr_c1:
            st.markdown(f"## {active_dataset.name}")
            st.markdown(
                f"**Dataset ID:** `{active_dataset.dataset_id}` &nbsp;|&nbsp; "
                f"<span class='badge-verification-supported'>🟢 ACTIVE WORKSPACE</span>",
                unsafe_allow_html=True,
            )
            if active_dataset.description:
                st.markdown(f"*{active_dataset.description}*")
            else:
                st.caption("No description provided.")

            created_display = (
                active_dataset.created_at[:19].replace("T", " ")
                if active_dataset.created_at
                else "Not recorded"
            )
            st.caption(f"📅 **Created (UTC):** {created_display}")

        with hdr_c2:
            all_datasets = store.list_datasets()
            ds_ids = [d.dataset_id for d in all_datasets]
            if len(ds_ids) > 1 and active_dataset.dataset_id in ds_ids:
                cur_idx = ds_ids.index(active_dataset.dataset_id)
                st.session_state["workspace_switch_dataset"] = active_dataset.dataset_id
                st.selectbox(
                    "Switch Workspace:",
                    options=ds_ids,
                    index=cur_idx,
                    format_func=lambda did: f"{next((d.name for d in all_datasets if d.dataset_id == did), did)}",
                    key="workspace_switch_dataset",
                    help="Switch to a different dataset workspace.",
                    on_change=select_active_dataset,
                    args=("workspace_switch_dataset", ds_ids),
                )

    # Workspace Summary Calculations
    documents = store.read_catalog()
    ws_summary = compute_workspace_summary(
        active_dataset.dataset_id,
        incidents_df,
        documents,
    )

    st.markdown("#### 📊 Workspace Overview")
    w_m1, w_m2, w_m3, w_m4 = st.columns(4)
    with w_m1:
        st.metric("Associated Incidents", ws_summary["total_incidents"])
    with w_m2:
        st.metric("Evidence Documents", ws_summary["total_documents"])
    with w_m3:
        st.metric("Distinct Machines", len(ws_summary["unique_machines"]))
    with w_m4:
        chars = ws_summary["total_extracted_characters"]
        chars_lbl = f"{chars:,}" if chars < 1_000_000 else f"{chars / 1_000_000:.1f}M"
        st.metric("Extracted Characters", chars_lbl)

    st.markdown("---")

    # Associated Incidents Section
    st.markdown("#### 📋 Associated Incidents")
    if ws_summary["total_incidents"] == 0:
        st.info(
            f"No maintenance incidents are currently associated with **{active_dataset.name}** "
            f"(`{active_dataset.dataset_id}`)."
        )
    else:
        st.caption(
            f"Showing {ws_summary['total_incidents']} incident(s) associated with this dataset:"
        )
        inc_df = pd.DataFrame(ws_summary["incidents"])
        st.dataframe(
            inc_df[["incident_id", "machine_id", "date", "symptom", "maintenance_action", "outcome"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "incident_id": st.column_config.TextColumn("Incident ID", width="small"),
                "machine_id": st.column_config.TextColumn("Machine ID", width="small"),
                "date": st.column_config.TextColumn("Date", width="small"),
                "symptom": st.column_config.TextColumn("Reported Symptom", width="large"),
                "maintenance_action": st.column_config.TextColumn("Maintenance Action", width="large"),
                "outcome": st.column_config.TextColumn("Outcome", width="medium"),
            },
        )

        with st.expander("🔍 Inspect Incident Details (Workspace)", expanded=False):
            available_inc_ids = [inc["incident_id"] for inc in ws_summary["incidents"]]
            sel_inc_id = st.selectbox(
                "Select incident to inspect:",
                options=available_inc_ids,
                key="ws_incident_inspect_select",
            )
            selected_rec = next(
                (inc for inc in ws_summary["incidents"] if inc["incident_id"] == sel_inc_id),
                None,
            )
            if selected_rec:
                with st.container(border=True):
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.caption("Incident ID")
                        st.markdown(f"**`{selected_rec['incident_id']}`**")
                    with sc2:
                        st.caption("Machine ID")
                        st.markdown(f"**`{selected_rec['machine_id']}`**")
                    with sc3:
                        st.caption("Date")
                        st.markdown(f"**{selected_rec['date']}**")

                    st.markdown(f"**Symptom:** {selected_rec['symptom']}")
                    st.markdown(f"**Maintenance Action:** {selected_rec['maintenance_action']}")
                    st.markdown(f"**Outcome:** {selected_rec['outcome']}")

                    st.button(
                        "Open in Incident Management 📋",
                        key="ws_link_inc_mgmt",
                        on_click=navigate_to,
                        args=(NAV_INCIDENTS,),
                    )

    st.markdown("---")

    # Associated Evidence Documents Section
    st.markdown("#### 📄 Associated Evidence Documents")
    if ws_summary["total_documents"] == 0:
        st.info(
            f"No evidence documents are currently associated with **{active_dataset.name}** "
            f"(`{active_dataset.dataset_id}`). "
            "Navigate to **Evidence Ingestion** to upload maintenance manuals and logs for this dataset."
        )
    else:
        st.caption(
            f"Showing {ws_summary['total_documents']} document(s) ({ws_summary['processed_documents']} processed, "
            f"{ws_summary['failed_documents']} failed):"
        )
        doc_rows = []
        for d in ws_summary["documents"]:
            doc_rows.append({
                "Document ID": d.document_id,
                "Original Filename": d.original_filename,
                "Type": d.file_type.upper() if d.file_type else "UNKNOWN",
                "Status": d.ingestion_status.value.upper(),
                "Extracted Chars": d.extracted_character_count,
                "Ingested At (UTC)": d.created_at[:19].replace("T", " ") if d.created_at else "Unknown",
            })

        st.dataframe(
            pd.DataFrame(doc_rows),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("🔍 Inspect Document Text (Workspace)", expanded=False):
            doc_ids = [d.document_id for d in ws_summary["documents"]]
            sel_doc_id = st.selectbox(
                "Select evidence document to preview:",
                options=doc_ids,
                format_func=lambda x: f"{x} — {next((d.original_filename for d in ws_summary['documents'] if d.document_id == x), x)}",
                key="ws_doc_preview_select",
            )
            selected_meta = store.get_metadata(sel_doc_id)
            if selected_meta:
                with st.container(border=True):
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        st.caption("Document ID")
                        st.markdown(f"**`{selected_meta.document_id}`**")
                    with dc2:
                        st.caption("Status")
                        st.markdown(f"**`{selected_meta.ingestion_status.value.upper()}`**")
                    with dc3:
                        st.caption("Characters Extracted")
                        st.markdown(f"**{selected_meta.extracted_character_count:,}**")

                    if selected_meta.error_message:
                        st.warning(f"Ingestion Error: {selected_meta.error_message}")

                    try:
                        text_content = store.get_extracted_text(selected_meta)
                    except Exception:
                        text_content = None

                    if text_content:
                        st.caption("Extracted Text Preview:")
                        preview_len = 2000
                        val = text_content[:preview_len] + ("\n\n... [Content truncated for preview]" if len(text_content) > preview_len else "")
                        st.text_area("Content Preview", value=val, height=180, disabled=True, key=f"ws_preview_{sel_doc_id}")
                    else:
                        st.info("No extracted text available for this document.")

                    st.button(
                        "Open in Document Catalog 📥",
                        key="ws_link_catalog",
                        on_click=navigate_to,
                        args=(NAV_EVIDENCE_INGESTION,),
                    )

    # Data Integrity Warnings
    all_datasets = store.list_datasets()
    valid_ids = [d.dataset_id for d in all_datasets]
    missing_docs = find_missing_document_associations(documents)
    invalid_docs = find_invalid_document_associations(documents, valid_ids)
    if missing_docs or invalid_docs:
        with st.container(border=True):
            if missing_docs:
                st.warning(
                    f"**Unassigned evidence:** {len(missing_docs)} document(s) have no dataset ID. "
                    "They are not associated with this workspace or with any newly created dataset."
                )
            if invalid_docs:
                st.warning(
                    f"**Unregistered dataset references:** {len(invalid_docs)} document(s) have a dataset ID "
                    "that is not present in the registry. These records are left unchanged."
                )

    st.markdown("---")

    # Scope & Navigation Boundary
    with st.container(border=True):
        st.markdown("**ℹ️ Scope & Navigation Boundary**")
        st.caption(
            f"You are viewing the workspace for **{active_dataset.name}** (`{active_dataset.dataset_id}`). "
            "All incident browsing, evidence retrieval, fault analysis, and cross-incident comparisons are strictly isolated to this active dataset."
        )
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            st.button(
                "🏠 Central Intelligence Hub",
                key="ws_nav_btn_hub",
                use_container_width=True,
                on_click=navigate_to,
                args=(NAV_HUB,),
            )
        with n2:
            st.button(
                "📋 Incident Management",
                key="ws_nav_btn_inc",
                use_container_width=True,
                on_click=navigate_to,
                args=(NAV_INCIDENTS,),
            )
        with n3:
            st.button(
                "📥 Evidence Ingestion",
                key="ws_nav_btn_ingest",
                use_container_width=True,
                on_click=navigate_to,
                args=(NAV_EVIDENCE_INGESTION,),
            )
        with n4:
            st.button(
                "🔎 Evidence Search",
                key="ws_nav_btn_search",
                use_container_width=True,
                on_click=navigate_to,
                args=(NAV_EVIDENCE_SEARCH,),
            )


def render_investigation_history(
    history_store: InvestigationHistoryStore,
    evidence_store: EvidenceStore,
    active_dataset: Optional[Dataset],
) -> None:
    """Display saved investigation snapshots and activity for the active dataset."""
    st.subheader("Investigation History & Activity")
    if active_dataset is None:
        st.error("Investigation history requires a registered active dataset.")
        return

    dataset_id = active_dataset.dataset_id
    st.info(
        f"History is restricted to **{active_dataset.name}** (`{dataset_id}`). "
        "Opening a record displays its saved snapshot and never reruns analysis."
    )
    runs = history_store.list_runs(evidence_store, dataset_id, limit=500)
    if not runs:
        st.info("No investigations have been saved for this dataset.")
    else:
        statuses = ["All statuses"] + sorted({run["status"] for run in runs})
        filter_col, date_col = st.columns(2)
        with filter_col:
            selected_status = st.selectbox(
                "Filter by status",
                options=statuses,
                key=f"history_status_{dataset_id}",
            )
        run_dates = [
            datetime.fromisoformat(run["analyzed_at"].replace("Z", "+00:00")).date()
            for run in runs
        ]
        with date_col:
            selected_dates = st.date_input(
                "Filter by date",
                value=(min(run_dates), max(run_dates)),
                key=f"history_dates_{dataset_id}",
            )
        if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
            start_date = selected_dates[0] or min(run_dates)
            end_date = selected_dates[1] or max(run_dates)
        else:
            start_date = end_date = selected_dates
        filtered_runs = [
            run
            for run in runs
            if (selected_status == "All statuses" or run["status"] == selected_status)
            and start_date
            <= datetime.fromisoformat(
                run["analyzed_at"].replace("Z", "+00:00")
            ).date()
            <= end_date
        ]
        if filtered_runs:
            st.dataframe(
                [
                    {
                        "Run time (UTC)": run["analyzed_at"][:19].replace("T", " "),
                        "Incident": run["incident_id"],
                        "Status": run["status"],
                        "Run ID": run["run_id"],
                    }
                    for run in filtered_runs
                ],
                hide_index=True,
            )
            selected_run_id = st.selectbox(
                "Open saved investigation",
                options=[run["run_id"] for run in filtered_runs],
                format_func=lambda run_id: next(
                    (
                        f"{run['incident_id']} — {run['status']} — "
                        f"{run['analyzed_at'][:19].replace('T', ' ')}"
                        for run in filtered_runs
                        if run["run_id"] == run_id
                    ),
                    run_id,
                ),
                key=f"history_run_{dataset_id}",
            )
            saved_run = history_store.get_run(
                evidence_store,
                dataset_id,
                selected_run_id,
            )
            if saved_run is None:
                st.warning("The selected investigation is no longer available in this dataset.")
            else:
                st.markdown("#### Saved incident snapshot")
                st.json(saved_run["incident_snapshot"])
                st.markdown("#### Saved analysis result")
                st.json(saved_run["result"])

                # Milestone 8: Follow-up Case Tracking Section
                st.markdown("---")
                st.markdown("### 📋 Investigation Case & Outcome Tracking (M8)")
                existing_case = history_store.get_case_by_run_id(
                    evidence_store, dataset_id, selected_run_id
                )
                if existing_case is None:
                    st.info("No follow-up case has been opened for this investigation run yet.")
                    with st.expander("➕ Open Follow-Up Case for this Investigation", expanded=False):
                        with st.form(key=f"create_case_form_{selected_run_id}"):
                            default_title = f"Case for {saved_run['incident_id']} - {saved_run['incident_snapshot'].get('symptom', 'Inspection')}"
                            c_title = st.text_input("Case Title", value=default_title)
                            c_priority = st.selectbox(
                                "Priority",
                                options=["medium", "high", "critical", "low"],
                                index=0,
                            )
                            c_by = st.text_input("Opened By", value="Reliability Engineer")
                            submit_case = st.form_submit_button("Create Case")
                            if submit_case:
                                try:
                                    created_case = history_store.create_case(
                                        evidence_store,
                                        dataset_id,
                                        selected_run_id,
                                        title=c_title,
                                        priority=c_priority,
                                        created_by=c_by,
                                    )
                                    st.success(f"✅ Case `{created_case['case_id']}` created successfully!")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Failed to create case: {exc}")
                else:
                    render_case_detail_panel(
                        history_store,
                        evidence_store,
                        dataset_id,
                        existing_case,
                        saved_run,
                    )
        else:
            st.info("No saved investigations match the selected filters.")

    st.markdown("---")
    st.markdown("#### Dataset activity")
    events = history_store.list_activity(evidence_store, dataset_id, limit=100)
    if events:
        st.dataframe(
            [
                {
                    "Time (UTC)": event["occurred_at"][:19].replace("T", " "),
                    "Activity": event["summary"],
                    "Run ID": event["entity_id"],
                }
                for event in events
            ],
            hide_index=True,
        )
    else:
        st.info("No investigation activity has been recorded for this dataset.")


def render_case_detail_panel(
    history_store: InvestigationHistoryStore,
    evidence_store: EvidenceStore,
    dataset_id: str,
    case_data: Dict[str, Any],
    saved_run: Dict[str, Any],
) -> None:
    """Render the full interactive detail panel for an M8 investigation case."""
    case_id = case_data["case_id"]
    status_str = case_data["status"]
    status_badge = {
        "open": "🔵 OPEN",
        "in_review": "🟡 IN REVIEW",
        "action_taken": "🟠 ACTION TAKEN",
        "resolved": "🟢 RESOLVED",
        "closed": "⚪ CLOSED",
    }.get(status_str, status_str.upper())

    with st.container(border=True):
        c_top1, c_top2, c_top3 = st.columns(3)
        with c_top1:
            st.caption("Case ID")
            st.markdown(f"**`{case_id}`**")
        with c_top2:
            st.caption("Status")
            st.markdown(f"**{status_badge}**")
        with c_top3:
            st.caption("Priority")
            st.markdown(f"**{case_data['priority'].upper()}**")

        st.markdown(f"**Title:** {case_data['title']}")
        st.caption(f"Created: {case_data['created_at'][:19].replace('T', ' ')} UTC | Last Updated: {case_data['updated_at'][:19].replace('T', ' ')} UTC")
        if case_data.get("closure_summary"):
            st.info(f"**Closure Summary:** {case_data['closure_summary']}")

    # 1. Independent Hypothesis Outcome Determinations
    st.markdown("#### 🔍 Diagnostic Hypotheses & Physical Outcome Verification")
    hypotheses = saved_run["result"].get("hypotheses", [])
    if not hypotheses:
        st.info("No diagnostic hypotheses were generated in this investigation run.")
    else:
        existing_outcomes = {o["hypothesis_id"]: o for o in case_data.get("hypothesis_outcomes", [])}
        for idx, hyp in enumerate(hypotheses, 1):
            h_id = hyp.get("hypothesis_id", f"HYP-{idx}")
            f_mode = hyp.get("failure_mode", f"Failure Mode {idx}")
            with st.container(border=True):
                st.markdown(f"**Hypothesis {idx}: {f_mode}** (`{h_id}`)")
                st.markdown(f"*Mechanism / Rationale:* {hyp.get('mechanism', 'N/A')}")
                outcome_rec = existing_outcomes.get(h_id)
                if outcome_rec:
                    st.success(
                        f"**Outcome Determination:** `{outcome_rec['determination'].upper()}` | "
                        f"Confidence: `{outcome_rec['confidence'].upper()}` | "
                        f"Evaluated By: {outcome_rec['determined_by']} ({outcome_rec['determined_at'][:19].replace('T', ' ')} UTC)\n\n"
                        f"*Teardown Notes:* {outcome_rec['notes']}"
                    )
                else:
                    st.warning("⚠️ Outcome not yet determined by physical inspection.")

                with st.expander(f"Record / Update Outcome for {f_mode}", expanded=False):
                    with st.form(key=f"outcome_form_{case_id}_{h_id}"):
                        det = st.selectbox(
                            "Determination",
                            options=["verified_correct", "partially_correct", "incorrect", "inconclusive"],
                            index=0,
                        )
                        conf = st.selectbox("Confidence", options=["high", "medium", "low"], index=0)
                        notes = st.text_area("Teardown & Verification Notes", placeholder="Describe physical findings...")
                        eval_by = st.text_input("Evaluator", value="Reliability Engineer")
                        save_out = st.form_submit_button("Record Outcome")
                        if save_out:
                            try:
                                history_store.record_hypothesis_outcome(
                                    evidence_store,
                                    dataset_id,
                                    case_id,
                                    hypothesis_id=h_id,
                                    failure_mode=f_mode,
                                    determination=det,
                                    confidence=conf,
                                    notes=notes,
                                    determined_by=eval_by,
                                )
                                st.success("Outcome recorded successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error recording outcome: {e}")

    # 2. Append-Only Physical Maintenance & Inspection Log
    st.markdown("#### 🛠️ Physical Maintenance & Inspection Logs (Shop-Floor Observations)")
    st.caption("Technician observations are strictly separated from verified document evidence.")
    inspections = case_data.get("inspections", [])
    if inspections:
        for insp in inspections:
            with st.container(border=True):
                st.markdown(
                    f"**Inspection:** `{insp['inspection_type'].upper()}` on {insp['inspected_at'][:10]} "
                    f"by **{insp['inspector_name']}**"
                )
                st.markdown(f"**Observations:** {insp['observations']}")
                if insp['physical_action_taken']:
                    st.markdown(f"**Physical Action Taken:** {insp['physical_action_taken']}")
                if insp.get("parts_replaced"):
                    st.markdown(f"**Parts Replaced:** {', '.join(insp['parts_replaced'])}")
    else:
        st.info("No physical maintenance inspections logged for this case yet.")

    with st.expander("➕ Log New Maintenance Inspection", expanded=False):
        with st.form(key=f"log_inspection_form_{case_id}"):
            i_inspector = st.text_input("Inspector Name", value="Field Technician")
            i_type = st.selectbox(
                "Inspection Type",
                options=[
                    "physical_disassembly",
                    "vibration_check",
                    "thermal_scan",
                    "lubricant_analysis",
                    "operational_test",
                ],
            )
            i_obs = st.text_area("Shop-Floor Observations (Unverified User Input)", placeholder="Observed wear, debris, clearance...")
            i_action = st.text_input("Physical Action Taken", placeholder="e.g. Replaced inner bearing race...")
            i_parts = st.text_input("Parts Replaced (comma-separated)", placeholder="e.g. BRG-6205, SEAL-30X52X7")
            submit_insp = st.form_submit_button("Log Inspection")
            if submit_insp:
                parts_arr = [p.strip() for p in i_parts.split(",") if p.strip()] if i_parts else []
                try:
                    history_store.log_inspection(
                        evidence_store,
                        dataset_id,
                        case_id,
                        inspector_name=i_inspector,
                        inspection_type=i_type,
                        observations=i_obs,
                        physical_action_taken=i_action,
                        parts_replaced=parts_arr,
                    )
                    st.success("Inspection logged successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to log inspection: {e}")

    # 3. Case Status Transition Rules & Lifecycle Ledger
    st.markdown("#### 🔄 Case Lifecycle & Status Transitions")
    with st.expander("Update Case Status / Reopen Case", expanded=False):
        with st.form(key=f"status_transition_form_{case_id}"):
            allowed_transitions = ["in_review", "action_taken", "resolved", "closed", "open"]
            next_status = st.selectbox("Target Status", options=allowed_transitions, index=0)
            trans_reason = st.text_area("Transition Justification / Reopening Reason", placeholder="Explain why status is changing...")
            closure_sum = st.text_area("Closure Summary (Mandatory if transitioning to CLOSED)", placeholder="Summary of resolution...")
            changed_by = st.text_input("Updated By", value="Reliability Engineer")
            submit_trans = st.form_submit_button("Apply Status Transition")
            if submit_trans:
                try:
                    history_store.transition_case_status(
                        evidence_store,
                        dataset_id,
                        case_id,
                        new_status=next_status,
                        reason=trans_reason,
                        changed_by=changed_by,
                        closure_summary=closure_sum if next_status == "closed" else None,
                    )
                    st.success(f"Case transitioned to `{next_status.upper()}` successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Transition error: {e}")

    # 4. Status History Ledger
    st.markdown("#### 📜 Status Change Audit Ledger")
    status_hist = case_data.get("status_history", [])
    if status_hist:
        st.dataframe(
            [
                {
                    "Time (UTC)": h["changed_at"][:19].replace("T", " "),
                    "From": (h["from_status"] or "INIT").upper(),
                    "To": h["to_status"].upper(),
                    "Reason": h["reason"],
                    "Changed By": h["changed_by"],
                }
                for h in status_hist
            ],
            hide_index=True,
        )


def render_central_hub(
    store: EvidenceStore,
    incidents_df: Optional[pd.DataFrame],
    active_dataset: Optional[Dataset] = None,
    history_store: Optional[InvestigationHistoryStore] = None,
) -> None:
    """Render the M7.2 Central Intelligence Hub: project overview, dataset registry, and navigation."""

    st.markdown("### 🏠 Central Intelligence Hub")
    st.markdown(
        "Project-wide overview of maintenance datasets, incidents, and evidence documents. "
        "Navigate to detailed investigation features using the sidebar or the quick-access cards below."
    )

    # --- Gather data from existing sources (read-only) ---
    store.ensure_default_dataset()
    datasets = store.list_datasets()
    documents = store.read_catalog()

    if active_dataset:
        st.info(
            f"ℹ️ **Global Hub View:** You are viewing the project-wide Central Intelligence Hub. "
            f"Active workspace: **{active_dataset.name}** (`{active_dataset.dataset_id}`). "
            f"To view and investigate records specific only to **{active_dataset.name}**, open its dedicated workspace below."
        )

    # --- SECTION 1: Project Overview Metrics ---
    summary = compute_project_summary(datasets, incidents_df, documents)
    evidence_summary = compute_evidence_summary(documents)

    st.markdown("#### 📊 Project-Wide Statistics")
    st.caption("Aggregate metrics calculated across all registered maintenance datasets and document catalogs.")
    m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
    with m_c1:
        st.metric("Datasets", summary["total_datasets"])
    with m_c2:
        st.metric("Total Incidents", summary["total_incidents"])
    with m_c3:
        st.metric("Evidence Documents", summary["total_documents"])
    with m_c4:
        st.metric("Unique Machines", summary["unique_machines"])
    with m_c5:
        st.metric("Unique Symptoms", summary["unique_symptoms"])

    with st.container(border=True):
        ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
        with ev_c1:
            st.metric("Processed Documents", evidence_summary["processed_documents"])
        with ev_c2:
            st.metric("Failed Ingestions", evidence_summary["failed_documents"])
        with ev_c3:
            chars = evidence_summary["total_extracted_characters"]
            chars_label = f"{chars:,}" if chars < 1_000_000 else f"{chars / 1_000_000:.1f}M"
            st.metric("Extracted Characters", chars_label)
        with ev_c4:
            type_str = ", ".join(
                f"{k}: {v}" for k, v in sorted(evidence_summary["type_counts"].items())
            ) if evidence_summary["type_counts"] else "None"
            st.metric("Document Types", type_str)

    st.markdown("---")

    # --- SECTION 2: Dataset Registry ---
    st.markdown("#### 📁 Dataset Registry")
    st.caption(
        "All registered datasets and their associated record counts. "
        "Select a dataset below to open its dedicated workspace."
    )

    registry_rows = compute_dataset_registry(datasets, incidents_df, documents)

    if registry_rows:
        registry_table = []
        for row in registry_rows:
            created_display = row["created_at"][:19].replace("T", " ") if row["created_at"] else "Unknown"
            registry_table.append({
                "Dataset Name": row["name"],
                "Dataset ID": row["dataset_id"],
                "Description": row["description"] or "—",
                "Created": created_display,
                "Incidents": row["incident_count"],
                "Documents": row["document_count"],
                "Processed Docs": row["processed_document_count"],
            })

        st.dataframe(
            pd.DataFrame(registry_table),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Dataset Name": st.column_config.TextColumn("Dataset Name", width="medium"),
                "Dataset ID": st.column_config.TextColumn("Dataset ID", width="medium"),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "Created": st.column_config.TextColumn("Created (UTC)", width="small"),
                "Incidents": st.column_config.NumberColumn("Incidents", width="small"),
                "Documents": st.column_config.NumberColumn("Documents", width="small"),
                "Processed Docs": st.column_config.NumberColumn("Processed", width="small"),
            },
        )

        # Milestone 7.3: Open Dataset Workspace action from Registry
        st.markdown("##### 📁 Open Dataset Workspace")
        reg_c1, reg_c2 = st.columns([3, 1])
        with reg_c1:
            target_open_id = st.selectbox(
                "Select a dataset workspace to open:",
                options=[r["dataset_id"] for r in registry_rows],
                format_func=lambda did: f"{next((r['name'] for r in registry_rows if r['dataset_id'] == did), did)} ({did})",
                key="hub_open_ws_select",
                help="Switch active dataset to this selection and open its workspace.",
            )
        with reg_c2:
            st.write("")
            st.button(
                "Open Workspace ➡️",
                key="hub_btn_open_ws",
                use_container_width=True,
                on_click=open_dataset_workspace,
                args=(target_open_id, [d.dataset_id for d in datasets]),
            )

        # Milestone 7.4: Create Dataset Form
        st.markdown("##### ➕ Create New Dataset")
        with st.expander("Create a new maintenance dataset", expanded=False):
            candidate_id = store.generate_dataset_id()
            while store.get_dataset(candidate_id) is not None:
                candidate_id = store.generate_dataset_id()

            st.caption(f"Unique Dataset ID (Auto-Generated): `{candidate_id}`")

            with st.form("create_dataset_form", clear_on_submit=True):
                new_name = st.text_input(
                    "Dataset Name *",
                    placeholder="e.g. Facility 2 CNC Spindle Logs, Bearing Replacement 2026",
                    help="Descriptive name for the maintenance dataset (2 to 100 characters).",
                    key="input_create_dataset_name",
                )
                new_desc = st.text_area(
                    "Description (Optional)",
                    placeholder="Optional notes regarding the operational scope, machine series, or facility unit...",
                    help="Optional summary description.",
                    key="input_create_dataset_desc",
                )
                create_btn = st.form_submit_button("Create Dataset ➕", use_container_width=True)

                if create_btn:
                    success, new_ds, err_msg = create_dataset_entry(
                        store,
                        name=new_name,
                        description=new_desc,
                        dataset_id=candidate_id,
                    )
                    if success and new_ds:
                        st.session_state["dataset_create_success_msg"] = (
                            f"Dataset '{new_ds.name}' ({new_ds.dataset_id}) created successfully!"
                        )
                        st.session_state["dataset_create_last_id"] = new_ds.dataset_id
                        st.session_state["dataset_create_last_name"] = new_ds.name
                        st.rerun()
                    else:
                        st.error(f"Validation Error: {err_msg}")

        # Display success banner & offer immediate navigation if newly created
        if "dataset_create_success_msg" in st.session_state:
            st.success(f"✅ {st.session_state['dataset_create_success_msg']}")
            new_id = st.session_state.get("dataset_create_last_id")
            new_name = st.session_state.get("dataset_create_last_name", new_id)
            sc1, sc2 = st.columns([3, 1])
            with sc1:
                st.caption(f"Ready to investigate **{new_name}** (`{new_id}`).")
            with sc2:
                st.button(
                    f"Open '{new_name}' Workspace ➡️",
                    key="btn_open_just_created_ws",
                    use_container_width=True,
                    on_click=open_created_dataset_workspace,
                )

    else:
        st.info("No datasets registered yet. The default dataset will be created automatically when data is loaded.")

    # Keep missing assignments distinct from dangling references; neither is
    # implicitly attached to the active or newly created dataset.
    unassociated = compute_unassociated_counts(datasets, incidents_df, documents)
    missing_docs = find_missing_document_associations(documents)
    invalid_docs = find_invalid_document_associations(
        documents,
        [dataset.dataset_id for dataset in datasets],
    )
    invalid_incidents = find_invalid_incident_associations(
        incidents_df,
        [dataset.dataset_id for dataset in datasets],
    )
    missing_incidents = find_missing_incident_associations(incidents_df)
    if (
        unassociated["unassociated_incidents"] > 0
        or missing_incidents
        or missing_docs
        or invalid_docs
        or invalid_incidents
    ):
        with st.container(border=True):
            st.markdown("#### Data association integrity")
            if missing_incidents:
                st.warning(
                    f"{len(missing_incidents)} incident(s) have no dataset ID."
                )
            if missing_docs:
                st.warning(
                    f"{len(missing_docs)} evidence document(s) have no dataset ID. "
                    "These are unassigned legacy records, not documents referencing a nonexistent dataset."
                )
                st.caption(
                    "The legacy migration associates only records with a missing dataset ID to Default Dataset. "
                    "It does not assign them to the active or newly created dataset, and does not change records "
                    "that already contain a dataset ID."
                )
                confirm_migration = st.checkbox(
                    "I explicitly approve associating these unassigned legacy documents with Default Dataset.",
                    key="confirm_legacy_document_migration",
                )
                st.button(
                    f"Associate {len(missing_docs)} unassigned document(s) with Default Dataset",
                    key="migrate_legacy_documents",
                    disabled=not confirm_migration,
                    on_click=migrate_legacy_document_associations,
                    args=(store,),
                )
            if invalid_docs:
                st.warning(
                    f"{len(invalid_docs)} document(s) contain a non-empty dataset ID that is not registered. "
                    "Those references are left unchanged and are not included in the legacy migration."
                )
            if invalid_incidents:
                st.warning(
                    f"{len(invalid_incidents)} incident(s) contain non-empty dataset IDs that are not registered. "
                    "Incident source data has not been changed."
                )
    migration_result = st.session_state.pop("legacy_migration_result", None)
    if migration_result:
        st.success(
            f"Legacy migration completed: {migration_result['migrated_documents']} document(s) "
            f"associated with Default Dataset; {migration_result['already_associated_documents']} "
            "already had an association."
        )

    st.markdown("---")

    # --- SECTION 3: Incident Overview ---
    st.markdown("#### 📋 Project-Wide Incidents Overview")
    overview_rows = compute_incident_overview(incidents_df)

    if overview_rows:
        st.caption(f"Showing all {len(overview_rows)} incidents across all registered datasets. (To view incidents for a single dataset, open its Dataset Workspace.)")
        st.dataframe(
            pd.DataFrame(overview_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Incident ID": st.column_config.TextColumn("Incident ID", width="small"),
                "Machine ID": st.column_config.TextColumn("Machine ID", width="small"),
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Symptom": st.column_config.TextColumn("Reported Symptom", width="large"),
                "Dataset": st.column_config.TextColumn("Dataset", width="medium"),
            },
        )

        st.caption(
            "💡 **Tip:** To inspect incident details or run fault analysis, navigate to "
            "**Incident Management** or **Fault Analysis** using the sidebar."
        )
    else:
        st.info("No incident records available. Load incident data to populate this overview.")

    st.markdown("---")

    # --- SECTION 4: Recent Activity ---
    st.markdown("#### 🕐 Project-Wide Recent Activity")

    recent_docs = compute_recent_documents(documents, limit=len(documents))
    document_timestamps = {
        document.document_id: document.created_at
        for document in documents
    }
    recent_events = (
        compute_recent_investigation_activity(history_store, store, limit=10)
        if history_store is not None
        else []
    )
    activity_rows = [
        {
            "time": document_timestamps.get(
                document["document_id"],
                document["created_at"],
            ),
            "activity": f"Evidence ingestion ({document['status']})",
            "entity": document["original_filename"],
            "dataset": document["dataset_id"],
        }
        for document in recent_docs
    ]
    activity_rows.extend(
        {
            "time": event["occurred_at"],
            "activity": event["summary"],
            "entity": event["entity_id"],
            "dataset": event["dataset_id"],
        }
        for event in recent_events
    )

    def activity_timestamp(row: Dict[str, Any]) -> float:
        try:
            return datetime.fromisoformat(row["time"].replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError, OSError):
            return 0.0

    activity_rows.sort(key=activity_timestamp, reverse=True)
    if activity_rows:
        st.caption(
            "Recent evidence ingestion timestamps and persisted investigation activity "
            "across registered datasets."
        )
        recent_table = [
            {
                "Time": row["time"][:19].replace("T", " "),
                "Activity": row["activity"],
                "Record": row["entity"],
                "Dataset": row["dataset"],
            }
            for row in activity_rows[:10]
        ]
        st.dataframe(
            pd.DataFrame(recent_table),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No evidence ingestion or investigation activity has been recorded yet.")

    st.caption(
        "Project-wide activity is summarized here. Open Investigation History & Activity "
        "to review saved analysis snapshots within the active dataset."
    )

    st.markdown("---")

    # --- SECTION 5: Quick Navigation Cards ---
    st.markdown("#### 🧭 Quick Navigation")
    st.caption("Jump to specific investigation features:")

    nav_c1, nav_c2, nav_c3 = st.columns(3)
    with nav_c1:
        with st.container(border=True):
            st.markdown("**📁 Dataset Workspace**")
            ws_label = active_dataset.name if active_dataset else "Selected Dataset"
            st.caption(f"Inspect records for active workspace: **{ws_label}**.")
            st.button(
                "Open Active Workspace",
                key="hub_nav_ws",
                on_click=navigate_to,
                args=(NAV_WORKSPACE,),
            )

        with st.container(border=True):
            st.markdown("**📋 Incident Management**")
            st.caption("Browse, filter, and inspect maintenance incidents.")
            st.button(
                "Go to Incidents",
                key="hub_nav_incidents",
                on_click=navigate_to,
                args=(NAV_INCIDENTS,),
            )

    with nav_c2:
        with st.container(border=True):
            st.markdown("**📥 Evidence Ingestion**")
            st.caption("Upload and process maintenance documents (PDF, TXT).")
            st.button(
                "Go to Ingestion",
                key="hub_nav_ingestion",
                on_click=navigate_to,
                args=(NAV_EVIDENCE_INGESTION,),
            )

        with st.container(border=True):
            st.markdown("**🔎 Evidence Search**")
            st.caption("Query the indexed evidence corpus with keyword search.")
            st.button(
                "Go to Search",
                key="hub_nav_search",
                on_click=navigate_to,
                args=(NAV_EVIDENCE_SEARCH,),
            )

    with nav_c3:
        with st.container(border=True):
            st.markdown("**🧠 Fault Analysis**")
            st.caption("Run evidence-grounded fault analysis on selected incidents.")
            st.button(
                "Go to Fault Analysis",
                key="hub_nav_analysis",
                on_click=navigate_to,
                args=(NAV_FAULT_ANALYSIS,),
            )

        with st.container(border=True):
            st.markdown("**🗂️ Investigation History & Activity**")
            st.caption("Review saved analysis snapshots within the active dataset.")
            st.button(
                "Open Investigation History",
                key="hub_nav_history",
                on_click=navigate_to,
                args=(NAV_INVESTIGATION_HISTORY,),
            )

        with st.container(border=True):
            st.markdown("**🔗 Cross-Incident Analysis**")
            st.caption("Compare symptom patterns and evidence links across incidents.")
            st.button(
                "Go to Cross-Incident",
                key="hub_nav_cross",
                on_click=navigate_to,
                args=(NAV_CROSS_INCIDENT,),
            )

    # Scope disclaimer
    st.caption(
        "ℹ️ **Project-Wide Scope Notice:** The Central Intelligence Hub shows project-wide aggregate statistics, all registered datasets, and overall activity. "
        "Individual Dataset Workspaces provide strict dataset-isolated records, evidence retrieval, and fault analysis."
    )


def main() -> None:
    st.markdown('<div class="ghost-title">GhostFault</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ghost-subtitle">Evidence-first industrial failure intelligence</div>',
        unsafe_allow_html=True,
    )
    navigation_error = st.session_state.pop("navigation_error", None)
    if navigation_error:
        st.error(navigation_error)
    active_dataset_error = st.session_state.pop("active_dataset_error", None)
    if active_dataset_error:
        st.error(active_dataset_error)

    # --- Initialize Components and Storage ---
    ingestion_service = get_ingestion_service()
    search_index, retriever, analyzer, comparison_engine, evidence_linker = get_search_components()
    history_store = get_investigation_history_store()

    # Ensure default dataset exists and load registry
    ingestion_service.store.ensure_default_dataset()
    registered_datasets = ingestion_service.store.list_datasets()

    try:
        raw_df = load_incidents_data(
            valid_dataset_ids=[dataset.dataset_id for dataset in registered_datasets]
        )
    except DatasetLoadError as err:
        st.error(
            f"**Failed to load maintenance records:** {err}\n\n"
            "Please ensure the data file exists at `data/sample/incidents.csv`, contains all required columns, "
            "and only references registered datasets."
        )
        st.stop()

    # Milestone 7.3: Active dataset state resolution
    current_active_id = st.session_state.get("active_dataset_id", DEFAULT_DATASET_ID)
    active_dataset, resolved_active_id, active_warning = resolve_active_dataset(
        current_active_id,
        registered_datasets,
        default_id=DEFAULT_DATASET_ID,
    )
    st.session_state["registered_dataset_ids"] = [
        dataset.dataset_id for dataset in registered_datasets
    ]
    if current_active_id != resolved_active_id:
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and (
                key.startswith("analysis_result_")
                or key.startswith("investigation_run_id_")
            ):
                del st.session_state[key]
        st.session_state.pop("selected_investigation_run_id", None)
    st.session_state["active_dataset_id"] = resolved_active_id
    st.session_state["active_dataset_selector_widget"] = resolved_active_id

    # --- Sidebar: Navigation + Controls ---
    with st.sidebar:
        st.header("Navigation")

        sync_navigation_widget_state(st.session_state, NAVIGATION_PAGES)

        selected_page = st.radio(
            "Go to:",
            options=NAVIGATION_PAGES,
            key="nav_radio",
            on_change=sync_sidebar_navigation,
        )

        st.divider()

        # Milestone 7.3: Active Dataset Selector
        st.header("Active Dataset")
        ds_options = [d.dataset_id for d in registered_datasets]
        ds_index = ds_options.index(resolved_active_id) if resolved_active_id in ds_options else 0

        st.selectbox(
            "Select Active Dataset:",
            options=ds_options,
            index=ds_index,
            format_func=lambda did: f"{next((d.name for d in registered_datasets if d.dataset_id == did), did)} ({did})",
            key="active_dataset_selector_widget",
            help="Select the active dataset workspace.",
            on_change=select_active_dataset,
            args=("active_dataset_selector_widget", ds_options),
        )

        # Prominent display of current active workspace
        with st.container(border=True):
            st.caption("Active Workspace:")
            st.markdown(f"**{active_dataset.name}**")
            st.caption(f"`{active_dataset.dataset_id}`")
            if selected_page != NAV_WORKSPACE:
                st.button(
                    "📁 Go to Workspace",
                    key="sidebar_btn_go_ws",
                    use_container_width=True,
                    on_click=navigate_to,
                    args=(NAV_WORKSPACE,),
                )

        st.divider()

        # Dataset scoping: incidents scoped to active workspace
        if "dataset_id" in raw_df.columns:
            active_incidents_df = raw_df[raw_df["dataset_id"] == active_dataset.dataset_id].copy()
        else:
            if active_dataset.dataset_id == DEFAULT_DATASET_ID:
                active_incidents_df = raw_df.copy()
            else:
                active_incidents_df = raw_df.iloc[0:0].copy()

        # Machine filter (shown for pages that use incident data)
        pages_with_machine_filter = {NAV_INCIDENTS, NAV_FAULT_ANALYSIS, NAV_CROSS_INCIDENT}
        if selected_page in pages_with_machine_filter:
            st.header("Controls & Filters")
            avail_machines = sorted([str(m) for m in active_incidents_df["machine_id"].dropna().unique().tolist()]) if not active_incidents_df.empty else []
            machine_options = ["All Machines"] + avail_machines
            selected_machine = st.selectbox(
                "Select Machine ID",
                options=machine_options,
                index=0,
                help="Filter maintenance incidents by specific machine unit.",
            )
        else:
            selected_machine = "All Machines"

        st.divider()
        st.markdown(
            """
            **Dataset Info:**
            - Environment: *Local Foundation*
            - Source: *Synthetic Maintenance Records*
            - Format: *CSV (`data/sample/incidents.csv`)*
            """
        )

    filtered_df = filter_by_machine(active_incidents_df, selected_machine)

    # --- Page Routing ---
    if selected_page == NAV_HUB:
        render_central_hub(
            ingestion_service.store,
            raw_df,
            active_dataset=active_dataset,
            history_store=history_store,
        )

    elif selected_page == NAV_WORKSPACE:
        render_dataset_workspace(
            ingestion_service.store,
            raw_df,
            active_dataset=active_dataset,
            active_warning=active_warning,
        )

    elif selected_page == NAV_INCIDENTS:
        # Existing M1 incident management (metrics, table, detail inspector, timeline)
        st.caption(
            f"ℹ️ Scope: Active Dataset — **{active_dataset.name}** (`{active_dataset.dataset_id}`). "
            f"Showing {len(filtered_df)} incident(s)."
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Incidents", value=len(filtered_df))
        with col2:
            distinct_machines = filtered_df["machine_id"].nunique()
            st.metric(label="Distinct Machines", value=distinct_machines)
        with col3:
            distinct_symptoms = filtered_df["symptom"].nunique()
            st.metric(label="Distinct Symptoms", value=distinct_symptoms)
        with col4:
            active_filter_label = "All Units" if selected_machine == "All Machines" else selected_machine
            st.metric(label="Filter Applied", value=active_filter_label)

        st.markdown("---")

        st.subheader("Maintenance Incidents Table")
        if filtered_df.empty:
            st.info("No records match the current filter selection.")
        else:
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    "incident_id": st.column_config.TextColumn("Incident ID", width="small"),
                    "machine_id": st.column_config.TextColumn("Machine ID", width="small"),
                    "date": st.column_config.DateColumn("Date", width="small"),
                    "symptom": st.column_config.TextColumn("Reported Symptom", width="large"),
                    "maintenance_action": st.column_config.TextColumn("Maintenance Action Taken", width="large"),
                    "outcome": st.column_config.TextColumn("Outcome / Verification", width="large"),
                },
                hide_index=True,
            )

        st.subheader("Incident Detail Inspector")
        if not filtered_df.empty:
            available_ids = filtered_df["incident_id"].tolist()
            selected_id = st.selectbox(
                "Select an incident to inspect details:",
                options=available_ids,
                index=0,
                key=f"inspector_select_{selected_machine}",
            )

            selected_record = get_incident_by_id(filtered_df, selected_id)
            if selected_record is not None:
                with st.container(border=True):
                    sub_col1, sub_col2, sub_col3 = st.columns(3)
                    with sub_col1:
                        st.caption("Incident ID")
                        st.markdown(f"**`{selected_record['incident_id']}`**")
                    with sub_col2:
                        st.caption("Machine ID")
                        st.markdown(f"**`{selected_record['machine_id']}`**")
                    with sub_col3:
                        st.caption("Incident Date")
                        st.markdown(f"**{selected_record['date']}**")

                    st.markdown("---")
                    st.markdown(f"**Reported Symptom:**\n\n{selected_record['symptom']}")
                    st.markdown(f"**Maintenance Action Taken:**\n\n{selected_record['maintenance_action']}")
                    st.markdown(f"**Outcome & Verification:**\n\n{selected_record['outcome']}")
        else:
            st.info("No incident available for the selected filter.")

        st.markdown("---")

        # Milestone 5.2: Incident Timeline
        render_incident_timeline(
            filtered_df,
            dataset_id=active_dataset.dataset_id,
            store=ingestion_service.store,
        )

    elif selected_page == NAV_EVIDENCE_INGESTION:
        render_evidence_ingestion(
            ingestion_service,
            search_index,
            active_dataset=active_dataset,
        )

    elif selected_page == NAV_EVIDENCE_SEARCH:
        render_evidence_search(
            retriever,
            search_index,
            ingestion_service.store,
            active_dataset=active_dataset,
        )

    elif selected_page == NAV_FAULT_ANALYSIS:
        # Need incident selection for fault analysis
        selected_record: Optional[pd.Series] = None
        if active_incidents_df.empty:
            st.subheader("🧠 Evidence-Grounded Fault Analysis")
            ds_name = active_dataset.name if active_dataset else "Selected Dataset"
            ds_id = active_dataset.dataset_id if active_dataset else DEFAULT_DATASET_ID
            st.info(
                f"ℹ️ **No Incidents in Active Dataset:** The active workspace **{ds_name}** (`{ds_id}`) "
                f"contains 0 maintenance incidents. To analyze faults, either switch to a dataset with registered incidents "
                f"(such as **Default Dataset**) or register incidents in this dataset. Cross-dataset incident borrowing is strictly disabled to preserve isolation."
            )
            render_fault_analysis(
                analyzer,
                None,
                active_dataset=active_dataset,
                history_store=history_store,
                evidence_store=ingestion_service.store,
            )
        else:
            if not filtered_df.empty:
                available_ids = filtered_df["incident_id"].tolist()
                selected_id = st.selectbox(
                    "Select an incident for fault analysis:",
                    options=available_ids,
                    index=0,
                    key=f"analysis_incident_select_{selected_machine}",
                )
                selected_record = get_incident_by_id(filtered_df, selected_id)

                if selected_record is not None:
                    with st.container(border=True):
                        sel_c1, sel_c2, sel_c3 = st.columns(3)
                        with sel_c1:
                            st.caption("Selected Incident")
                            st.markdown(f"**`{selected_record['incident_id']}`**")
                        with sel_c2:
                            st.caption("Machine")
                            st.markdown(f"**`{selected_record['machine_id']}`**")
                        with sel_c3:
                            st.caption("Symptom")
                            st.markdown(f"*{selected_record['symptom']}*")

            render_fault_analysis(
                analyzer,
                selected_record,
                active_dataset=active_dataset,
                history_store=history_store,
                evidence_store=ingestion_service.store,
            )

    elif selected_page == NAV_CROSS_INCIDENT:
        render_cross_incident_analysis(
            filtered_df,
            comparison_engine,
            evidence_linker,
            active_dataset=active_dataset,
        )

    elif selected_page == NAV_INVESTIGATION_HISTORY:
        render_investigation_history(
            history_store,
            ingestion_service.store,
            active_dataset,
        )


if __name__ == "__main__":
    main()

"""GhostFault - Streamlit Application.

Evidence-first industrial failure intelligence dashboard foundation.
Includes Milestone 1 incident inspection, Milestone 2 document ingestion,
and Milestone 3 evidence chunking, indexing, and deterministic retrieval.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional

from src.data_loader import (
    load_incidents_data,
    filter_by_machine,
    get_incident_by_id,
    DatasetLoadError,
)
from src.ingestion import IngestionService
from src.storage import EvidenceStore
from src.models import DocumentStatus
from src.indexer import SearchIndex
from src.retriever import EvidenceRetriever

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
    .result-card {
        border-left: 4px solid #3b82f6;
        margin-bottom: 12px;
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
def get_search_components() -> tuple[SearchIndex, EvidenceRetriever]:
    """Initialize and cache the SearchIndex and EvidenceRetriever instances."""
    index = SearchIndex()
    retriever = EvidenceRetriever(index)
    return index, retriever


def render_evidence_ingestion(service: IngestionService, search_index: SearchIndex) -> None:
    """Render Milestone 2 Evidence Ingestion interface with auto-indexing."""
    st.subheader("📥 Evidence Ingestion & Document Management")
    st.markdown(
        "Upload technical maintenance manuals, operator logs, or failure sheets (`.pdf`, `.txt`). "
        "Files are validated, stored securely, cataloged, and indexed for deterministic evidence retrieval."
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
                    meta = service.ingest_document(
                        filename=uploaded_file.name,
                        content=content_bytes,
                        source_type="dashboard_upload",
                    )

                if meta.ingestion_status == DocumentStatus.PROCESSED:
                    # Automatically index document chunks
                    extracted_text = service.store.get_extracted_text(meta)
                    if extracted_text:
                        chunks_indexed = search_index.index_document(meta, extracted_text)
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
                - **Chunking Strategy:** Configurable window (600 chars, 100 overlap) with natural boundaries
                - **Search Index:** Local persistent SQLite FTS5 with BM25 ranking
                - **Storage:** Local isolated filesystem under `data/evidence/`
                """
            )

    # Ingested Document Catalog & Preview
    docs = service.list_documents()
    if docs:
        st.markdown("#### Cataloged Evidence Documents")
        
        tot_docs = len(docs)
        succ_docs = sum(1 for d in docs if d.ingestion_status == DocumentStatus.PROCESSED)
        fail_docs = tot_docs - succ_docs
        
        met_col1, met_col2, met_col3 = st.columns(3)
        with met_col1:
            st.metric("Total Cataloged Documents", tot_docs)
        with met_col2:
            st.metric("Successfully Ingested", succ_docs)
        with met_col3:
            st.metric("Failed Ingestions", fail_docs)

        # Tabular overview
        catalog_table_data = []
        for d in docs:
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

        # Document Inspector / Extracted Text Preview
        st.markdown("#### Document Inspector & Text Preview")
        doc_options = {d.document_id: f"{d.document_id} — {d.original_filename} ({d.ingestion_status.value})" for d in docs}
        selected_doc_id = st.selectbox(
            "Select cataloged document to inspect:",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            key="doc_inspector_select",
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
        st.info("No documents have been ingested yet. Upload a `.pdf` or `.txt` file above to begin.")


def render_evidence_search(retriever: EvidenceRetriever, search_index: SearchIndex, store: EvidenceStore) -> None:
    """Render Milestone 3 Evidence Search & Index Management interface."""
    st.subheader("🔎 Evidence Search & Passage Retrieval")
    st.markdown(
        "Query ingested maintenance records, manuals, and technical notes using deterministic keyword search. "
        "Every retrieved passage is ranked using SQLite FTS5 BM25 and includes full provenance tracking back to its source document."
    )

    # Index Status & Sync Controls
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
                    st.success(f"Synced {sync_info['indexed_documents']} docs ({sync_info['total_chunks']} chunks).")
                    st.rerun()
            with action_btn_c2:
                if st.button("⚡ Rebuild Index", help="Wipe and rebuild full search index from catalog"):
                    rebuild_info = search_index.rebuild_index(store)
                    st.success(f"Rebuilt index with {rebuild_info['total_chunks']} chunks.")
                    st.rerun()

    # Search query input bar
    catalog = store.read_catalog()
    doc_filter_options = ["All Documents"] + [d.document_id for d in catalog if d.ingestion_status == DocumentStatus.PROCESSED]

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

    # Disclaimer on retrieval scores
    st.caption(
        "ℹ️ **Note on Search Scores:** Scores represent SQLite FTS5 BM25 keyword relevance, "
        "not diagnostic confidence or probability of machine failure."
    )

    if query.strip():
        with st.spinner("Searching evidence chunks..."):
            results = retriever.search(
                query=query,
                limit=int(max_results),
                document_id_filter=selected_doc_filter,
            )

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

                    # Highlighted snippet
                    st.markdown(f"**Matched Passage:**\n\n> {res.matched_snippet}")

                    # Term provenance and offsets
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


def main() -> None:
    # Header
    st.markdown('<div class="ghost-title">GhostFault</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ghost-subtitle">Evidence-first industrial failure intelligence</div>',
        unsafe_allow_html=True,
    )

    # Attempt to load dataset
    try:
        raw_df = load_incidents_data()
    except DatasetLoadError as err:
        st.error(
            f"**Failed to load maintenance records:** {err}\n\n"
            "Please ensure the data file exists at `data/sample/incidents.csv` and contains all required columns."
        )
        st.stop()

    # Sidebar: Filtering Controls
    with st.sidebar:
        st.header("Controls & Filters")
        machine_options = ["All Machines"] + sorted(raw_df["machine_id"].unique().tolist())
        selected_machine = st.selectbox(
            "Select Machine ID",
            options=machine_options,
            index=0,
            help="Filter maintenance incidents by specific machine unit.",
        )

        st.divider()
        st.markdown(
            """
            **Dataset Info:**
            - Environment: *Local Foundation*
            - Source: *Synthetic Maintenance Records*
            - Format: *CSV (`data/sample/incidents.csv`)*
            """
        )

    # Apply Machine Filter
    filtered_df = filter_by_machine(raw_df, selected_machine)

    # Key Metrics Bar
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

    # Incident Table
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

    # Incident Inspector / Details Selector
    st.subheader("Incident Detail Inspector")
    if not filtered_df.empty:
        available_ids = filtered_df["incident_id"].tolist()
        selected_id = st.selectbox(
            "Select an incident to inspect details:",
            options=available_ids,
            index=0,
            key=f"inspector_select_{selected_machine}",
        )

        record = get_incident_by_id(filtered_df, selected_id)
        if record is not None:
            with st.container(border=True):
                sub_col1, sub_col2, sub_col3 = st.columns(3)
                with sub_col1:
                    st.caption("Incident ID")
                    st.markdown(f"**`{record['incident_id']}`**")
                with sub_col2:
                    st.caption("Machine ID")
                    st.markdown(f"**`{record['machine_id']}`**")
                with sub_col3:
                    st.caption("Incident Date")
                    st.markdown(f"**{record['date']}**")

                st.markdown("---")
                st.markdown(f"**Reported Symptom:**\n\n{record['symptom']}")
                st.markdown(f"**Maintenance Action Taken:**\n\n{record['maintenance_action']}")
                st.markdown(f"**Outcome & Verification:**\n\n{record['outcome']}")
    else:
        st.info("No incident available for the selected filter.")

    st.markdown("---")

    # Shared singletons for ingestion and search
    ingestion_service = get_ingestion_service()
    search_index, retriever = get_search_components()

    # Milestone 2: Evidence Ingestion & Document Management
    render_evidence_ingestion(ingestion_service, search_index)

    st.markdown("---")

    # Milestone 3: Evidence Search & Passage Retrieval
    render_evidence_search(retriever, search_index, ingestion_service.store)

    st.markdown("---")

    # Future Modules Section
    st.markdown("### Next Phase Modules")

    st.markdown(
        '#### Fault Analysis <span class="planned-badge">Planned Feature</span>',
        unsafe_allow_html=True,
    )
    st.info(
        "**Automated Diagnostic Hypothesis is a planned feature.**\n\n"
        "Future milestones will synthesize multi-source telemetry, maintenance records, and retrieved evidence passages "
        "to generate grounded failure hypotheses with confidence scores and citations."
    )
    st.selectbox(
        "Hypothesis Target Mode (UI Preview)",
        options=["Deterministic Rule Engine", "Evidence-grounded LLM synthesis"],
        disabled=True,
        key="preview_fault_analysis_select",
    )
    st.button("Run Diagnostic Inference", disabled=True, key="preview_fault_analysis_btn")


if __name__ == "__main__":
    main()

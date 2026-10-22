"""GhostFault - Streamlit Application.

Evidence-first industrial failure intelligence dashboard foundation.
Includes:
- Milestone 1: Incident dashboard, filters, inspector.
- Milestone 2: Evidence document ingestion & management.
- Milestone 3: Chunking, indexing, and deterministic retrieval.
- Milestone 4 / 4.1 / 4.1.2: Evidence-grounded fault analysis & substantive relevance validation.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

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
)
from src.indexer import SearchIndex
from src.retriever import EvidenceRetriever
from src.analyzer import FaultAnalysisEngine

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
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_ingestion_service() -> IngestionService:
    """Initialize and cache the evidence ingestion service instance."""
    return IngestionService()


@st.cache_resource
def get_search_components() -> tuple[SearchIndex, EvidenceRetriever, FaultAnalysisEngine]:
    """Initialize and cache the SearchIndex, EvidenceRetriever, and FaultAnalysisEngine instances."""
    index = SearchIndex()
    retriever = EvidenceRetriever(index)
    analyzer = FaultAnalysisEngine(retriever)
    return index, retriever, analyzer


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
                - **Chunking Strategy:** Sentence-aligned window (600 chars, 100 overlap)
                - **Search Index:** Local persistent SQLite FTS5 with BM25 ranking
                - **Storage:** Local isolated filesystem under `data/evidence/`
                """
            )

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


def render_fault_analysis(analyzer: FaultAnalysisEngine, selected_record: Optional[pd.Series]) -> None:
    """Render Milestone 4.1.2 Evidence-Grounded Fault Analysis interface."""
    st.subheader("🧠 Evidence-Grounded Fault Analysis")
    st.markdown(
        "Generate explainable, evidence-backed investigative hypotheses for the selected incident. "
        "The engine validates passage substance, excludes low-value keyword lists, deduplicates identical source copies, "
        "and distinguishes machine scopes with complete provenance traceability."
    )

    if selected_record is None:
        st.info("Please select an incident above to perform fault analysis.")
        return

    incident_dict = selected_record.to_dict()
    incident_id = incident_dict.get("incident_id", "UNKNOWN")

    analysis_key = f"analysis_result_{incident_id}"

    btn_col, status_col = st.columns([2, 5])
    with btn_col:
        run_analysis = st.button("🚀 Run Fault Analysis", key=f"btn_run_analysis_{incident_id}")

    if run_analysis or analysis_key in st.session_state:
        if run_analysis:
            with st.spinner("Analyzing incident symptoms against retrieved evidence corpus..."):
                analysis_result = analyzer.analyze_incident(incident_dict)
                st.session_state[analysis_key] = analysis_result
        else:
            analysis_result = st.session_state[analysis_key]

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


def main() -> None:
    st.markdown('<div class="ghost-title">GhostFault</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ghost-subtitle">Evidence-first industrial failure intelligence</div>',
        unsafe_allow_html=True,
    )

    try:
        raw_df = load_incidents_data()
    except DatasetLoadError as err:
        st.error(
            f"**Failed to load maintenance records:** {err}\n\n"
            "Please ensure the data file exists at `data/sample/incidents.csv` and contains all required columns."
        )
        st.stop()

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

    filtered_df = filter_by_machine(raw_df, selected_machine)

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
    selected_record: Optional[pd.Series] = None
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

    ingestion_service = get_ingestion_service()
    search_index, retriever, analyzer = get_search_components()

    # Milestone 4.1.2: Evidence-Grounded Fault Analysis
    render_fault_analysis(analyzer, selected_record)

    st.markdown("---")

    # Milestone 3: Evidence Search & Passage Retrieval
    render_evidence_search(retriever, search_index, ingestion_service.store)

    st.markdown("---")

    # Milestone 2: Evidence Ingestion & Document Management
    render_evidence_ingestion(ingestion_service, search_index)


if __name__ == "__main__":
    main()

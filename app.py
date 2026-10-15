"""GhostFault - Streamlit Application.

Evidence-first industrial failure intelligence dashboard foundation.
Includes Milestone 1 incident inspection and Milestone 2 document ingestion.
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
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_ingestion_service() -> IngestionService:
    """Initialize and cache the evidence ingestion service instance."""
    return IngestionService()


def render_evidence_ingestion(service: IngestionService) -> None:
    """Render Milestone 2 Evidence Ingestion interface."""
    st.subheader("📥 Evidence Ingestion & Document Management")
    st.markdown(
        "Upload technical maintenance manuals, operator logs, or failure sheets (`.pdf`, `.txt`). "
        "Files are validated, stored securely, and cataloged with extracted text references."
    )

    upload_col, info_col = st.columns([3, 2])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Choose a maintenance document to ingest",
            type=["pdf", "txt"],
            help="Select a text-based PDF or TXT file. Scanned PDFs without OCR text layers will be flagged.",
        )

        if uploaded_file is not None:
            # Prevent duplicate processing of the exact same upload state
            file_key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
            if st.button("Process & Ingest Document", key="btn_ingest_doc"):
                with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                    content_bytes = uploaded_file.getvalue()
                    meta = service.ingest_document(
                        filename=uploaded_file.name,
                        content=content_bytes,
                        source_type="dashboard_upload",
                    )
                    st.session_state[file_key] = meta.document_id

                if meta.ingestion_status == DocumentStatus.PROCESSED:
                    st.success(
                        f"✅ Document successfully ingested! Assigned ID: `{meta.document_id}` "
                        f"({meta.extracted_character_count:,} characters extracted)."
                    )
                else:
                    st.error(
                        f"❌ Ingestion Failed (`{meta.error_code}`): {meta.error_message}"
                    )

    with info_col:
        with st.container(border=True):
            st.markdown("**Ingestion Pipeline Specifications:**")
            st.markdown(
                """
                - **Supported Formats:** Text-based `.pdf`, `.txt`
                - **Collision Strategy:** Collision-resistant UUID `DOC-XXXXXXXXXXXX`
                - **Path Sanitization:** RFC/filesystem traversal safe
                - **OCR Policy:** Plain selectable text only (Milestone 2 scope)
                - **Storage:** Local isolated filesystem under `data/evidence/`
                """
            )

    # Ingested Document Catalog & Preview
    docs = service.list_documents()
    if docs:
        st.markdown("#### Cataloged Evidence Documents")
        
        # Summary metrics for ingested corpus
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

    # Milestone 2: Evidence Ingestion & Document Management
    ingestion_service = get_ingestion_service()
    render_evidence_ingestion(ingestion_service)

    st.markdown("---")

    # Future Modules Section
    st.markdown("### Next Phase Modules")

    col_search, col_analysis = st.columns(2)

    with col_search:
        st.markdown(
            '#### Evidence Search <span class="planned-badge">Planned Feature</span>',
            unsafe_allow_html=True,
        )
        st.info(
            "**Semantic Evidence Retrieval is a planned feature.**\n\n"
            "Future versions will index ingested documents and historical shift logs "
            "using vector embeddings for cross-document evidence verification."
        )
        st.text_input(
            "Search evidence records (UI Preview)",
            placeholder="e.g., high harmonic vibration coupling",
            disabled=True,
            key="preview_evidence_search_input",
        )
        st.button("Search Evidence Corpus", disabled=True, key="preview_evidence_search_btn")

    with col_analysis:
        st.markdown(
            '#### Fault Analysis <span class="planned-badge">Planned Feature</span>',
            unsafe_allow_html=True,
        )
        st.info(
            "**Automated Diagnostic Hypothesis is a planned feature.**\n\n"
            "Future milestones will synthesize multi-source telemetry and evidence records "
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

"""GhostFault - Milestone 1 Streamlit Application.

Evidence-first industrial failure intelligence dashboard foundation.
"""

import streamlit as st
import pandas as pd
from src.data_loader import (
    load_incidents_data,
    filter_by_machine,
    get_incident_by_id,
    DatasetLoadError,
)

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
    </style>
    """,
    unsafe_allow_html=True,
)


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
            - Environment: *Milestone 1 Local Foundation*
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
            "Future versions will index technical machine manuals, historical shift logs, and RCA reports "
            "using vector embeddings for cross-document evidence verification."
        )
        st.text_input(
            "Search evidence records (UI Preview)",
            placeholder="e.g., high harmonic vibration coupling",
            disabled=True,
        )
        st.button("Search Evidence Corpus", disabled=True)

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
        )
        st.button("Run Diagnostic Inference", disabled=True)


if __name__ == "__main__":
    main()

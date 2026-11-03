"""Regression coverage for M7.3 navigation and dataset association state."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.data_loader import get_default_dataset_path, load_incidents_data
from src.indexer import SearchIndex
from src.hub import (
    NAV_HUB,
    NAV_INCIDENTS,
    NAV_WORKSPACE,
    NAVIGATION_PAGES,
    compute_dataset_registry,
    compute_unassociated_counts,
    compute_workspace_summary,
    find_invalid_document_associations,
    find_missing_document_associations,
)
from src.models import (
    DEFAULT_DATASET_ID,
    Dataset,
    DocumentMetadata,
    DocumentStatus,
)
from src.storage import EvidenceStore


_NAVIGATION_APP = """
import streamlit as st
from src.hub import (
    DEFAULT_DATASET_ID,
    NAV_HUB,
    NAV_INCIDENTS,
    NAV_WORKSPACE,
    NAVIGATION_PAGES,
    get_navigation_page,
    set_active_dataset_id,
    set_navigation_page,
    sync_navigation_from_widget,
    sync_navigation_widget_state,
)

DATASET_IDS = [DEFAULT_DATASET_ID, "DATASET-CUSTOM"]

def on_sidebar_navigation():
    sync_navigation_from_widget(st.session_state, NAVIGATION_PAGES)

def on_dataset_change(widget_key):
    set_active_dataset_id(st.session_state, st.session_state[widget_key], DATASET_IDS)

def open_selected_workspace():
    set_active_dataset_id(
        st.session_state,
        st.session_state["hub_open_ws_select"],
        DATASET_IDS,
    )
    set_navigation_page(st.session_state, NAV_WORKSPACE, NAVIGATION_PAGES)

def open_incidents():
    set_navigation_page(st.session_state, NAV_INCIDENTS, NAVIGATION_PAGES)

st.session_state.setdefault("ghostfault_nav", NAV_HUB)
st.session_state.setdefault("active_dataset_id", DEFAULT_DATASET_ID)
sync_navigation_widget_state(st.session_state, NAVIGATION_PAGES)
st.session_state["sidebar_dataset"] = st.session_state["active_dataset_id"]

page = st.radio(
    "Go to",
    NAVIGATION_PAGES,
    key="nav_radio",
    on_change=on_sidebar_navigation,
)
st.session_state["workspace_name"] = st.session_state["active_dataset_id"]
st.selectbox(
    "Active dataset",
    DATASET_IDS,
    key="sidebar_dataset",
    on_change=on_dataset_change,
    args=("sidebar_dataset",),
)

if page == NAV_HUB:
    st.selectbox("Workspace dataset", DATASET_IDS, key="hub_open_ws_select")
    st.button("Open workspace", key="hub_open_ws", on_click=open_selected_workspace)
elif page == NAV_WORKSPACE:
    st.session_state["workspace_switch_dataset"] = st.session_state["active_dataset_id"]
    st.selectbox(
        "Switch workspace",
        DATASET_IDS,
        key="workspace_switch_dataset",
        on_change=on_dataset_change,
        args=("workspace_switch_dataset",),
    )
    st.button("Incident Management shortcut", key="workspace_to_incidents", on_click=open_incidents)

st.write("Current page:", page)
"""


def test_sidebar_navigation_switches_to_every_registered_page():
    app = AppTest.from_string(_NAVIGATION_APP).run()
    assert not app.exception

    for page in NAVIGATION_PAGES:
        app.radio(key="nav_radio").set_value(page).run()
        assert not app.exception
        assert app.session_state["ghostfault_nav"] == page
        assert app.radio(key="nav_radio").value == page


def test_hub_workspace_shortcut_and_workspace_incident_shortcut():
    app = AppTest.from_string(_NAVIGATION_APP).run()
    app.selectbox(key="hub_open_ws_select").select("DATASET-CUSTOM").run()
    app.button(key="hub_open_ws").click().run()

    assert not app.exception
    assert app.session_state["ghostfault_nav"] == NAV_WORKSPACE
    assert app.session_state["active_dataset_id"] == "DATASET-CUSTOM"
    assert app.selectbox(key="sidebar_dataset").value == "DATASET-CUSTOM"
    assert app.selectbox(key="workspace_switch_dataset").value == "DATASET-CUSTOM"

    app.button(key="workspace_to_incidents").click().run()
    assert not app.exception
    assert app.session_state["ghostfault_nav"] == NAV_INCIDENTS


def test_dataset_selector_updates_workspace_without_a_second_selection():
    app = AppTest.from_string(_NAVIGATION_APP)
    app.session_state["ghostfault_nav"] = NAV_WORKSPACE
    app.run()
    assert not app.exception

    app.selectbox(key="sidebar_dataset").select("DATASET-CUSTOM").run()
    assert not app.exception
    assert app.session_state["active_dataset_id"] == "DATASET-CUSTOM"
    assert app.selectbox(key="sidebar_dataset").value == "DATASET-CUSTOM"
    assert app.selectbox(key="workspace_switch_dataset").value == "DATASET-CUSTOM"


def test_actual_app_navigation_and_active_dataset_controls_are_consistent(
    tmp_path: Path,
    monkeypatch,
):
    evidence_root = tmp_path / "evidence"
    index_path = tmp_path / "index" / "evidence_index.db"
    original_store_init = EvidenceStore.__init__
    original_index_init = SearchIndex.__init__

    def init_isolated_store(self, base_dir=None):
        original_store_init(self, evidence_root)

    def init_isolated_index(self, db_path=None, chunker=None):
        original_index_init(self, index_path, chunker)

    st.cache_resource.clear()
    monkeypatch.setattr(EvidenceStore, "__init__", init_isolated_store)
    monkeypatch.setattr(SearchIndex, "__init__", init_isolated_index)
    store = EvidenceStore()
    store.ensure_default_dataset()
    store.create_dataset("DATASET-CUSTOM", "Custom diagnostics")
    _write_raw_legacy_catalog_fixture(
        store,
        [_make_document(store, f"DOC-LEGACY-UI-{index}", None) for index in range(7)],
    )

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()
    try:
        assert not app.exception
        assert all(doc.dataset_id is None for doc in store.read_catalog())

        app.checkbox(key="confirm_legacy_document_migration").check().run()
        assert not app.exception
        app.button(key="migrate_legacy_documents").click().run()
        assert not app.exception
        assert all(
            doc.dataset_id == DEFAULT_DATASET_ID
            for doc in store.read_catalog()
        )

        for page in NAVIGATION_PAGES:
            app.radio(key="nav_radio").set_value(page).run()
            assert not app.exception
            assert app.session_state["ghostfault_nav"] == page

        app.radio(key="nav_radio").set_value(NAV_HUB).run()
        app.selectbox(key="hub_open_ws_select").select("DATASET-CUSTOM").run()
        app.button(key="hub_btn_open_ws").click().run()
        assert not app.exception
        assert app.session_state["ghostfault_nav"] == NAV_WORKSPACE
        assert app.session_state["active_dataset_id"] == "DATASET-CUSTOM"
        assert app.selectbox(key="active_dataset_selector_widget").value == "DATASET-CUSTOM"
        assert app.selectbox(key="workspace_switch_dataset").value == "DATASET-CUSTOM"

        app.button(key="ws_nav_btn_inc").click().run()
        assert not app.exception
        assert app.session_state["ghostfault_nav"] == NAV_INCIDENTS

        app.selectbox(key="active_dataset_selector_widget").select(DEFAULT_DATASET_ID).run()
        assert not app.exception
        assert app.session_state["active_dataset_id"] == DEFAULT_DATASET_ID
        app.radio(key="nav_radio").set_value(NAV_WORKSPACE).run()
        assert not app.exception
        incident_metric = next(
            metric for metric in app.metric
            if metric.label == "Associated Incidents"
        )
        assert incident_metric.value == "6"
        assert app.selectbox(key="active_dataset_selector_widget").value == DEFAULT_DATASET_ID
        assert app.selectbox(key="workspace_switch_dataset").value == DEFAULT_DATASET_ID
    finally:
        st.cache_resource.clear()


def test_legacy_sample_incidents_belong_to_default_workspace_in_memory():
    csv_path = get_default_dataset_path()
    original_bytes = csv_path.read_bytes()

    incidents = load_incidents_data(csv_path)
    datasets = [
        Dataset(
            dataset_id=DEFAULT_DATASET_ID,
            name="Default Dataset",
            description="",
            created_at="2026-01-01T00:00:00Z",
        )
    ]
    default_summary = compute_workspace_summary(DEFAULT_DATASET_ID, incidents, [])
    custom_summary = compute_workspace_summary("DATASET-NEW", incidents, [])
    registry = compute_dataset_registry(datasets, incidents, [])

    assert len(incidents) == 6
    assert set(incidents["dataset_id"]) == {DEFAULT_DATASET_ID}
    assert default_summary["total_incidents"] == 6
    assert custom_summary["total_incidents"] == 0
    assert registry[0]["incident_count"] == 6
    assert csv_path.read_bytes() == original_bytes


def _make_document(
    store: EvidenceStore,
    doc_id: str,
    dataset_id: Optional[str],
) -> DocumentMetadata:
    filename = f"{doc_id}.txt"
    original_ref = store.save_original(doc_id, filename, f"original:{doc_id}".encode())
    extracted_ref = store.save_extracted_text(doc_id, f"Extracted text for {doc_id}")
    return DocumentMetadata(
        document_id=doc_id,
        original_filename=filename,
        file_type="txt",
        source_type="test",
        storage_reference=original_ref,
        extraction_status="success",
        ingestion_status=DocumentStatus.PROCESSED,
        created_at="2026-10-01T00:00:00+00:00",
        updated_at="2026-10-01T00:00:00+00:00",
        extracted_text_reference=extracted_ref,
        extracted_character_count=len(f"Extracted text for {doc_id}"),
        dataset_id=dataset_id,
    )


def _write_raw_legacy_catalog_fixture(store: EvidenceStore, documents: list[DocumentMetadata]) -> None:
    """Write intentionally legacy or corrupt catalog JSON without using the validated catalog API.

    This helper is narrow and explicit: it exists only for tests that intentionally model a
    legacy or dangling catalog state that the application is expected to report, not repair,
    through the normal ownership validation path.
    """
    payload = [document.to_dict() for document in documents]
    temp_path = store.catalog_file.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temp_path.replace(store.catalog_file)


def test_seven_missing_document_associations_are_not_invalid_ids_or_auto_reassigned(tmp_path: Path):
    store = EvidenceStore(base_dir=tmp_path / "evidence")
    store.ensure_default_dataset()
    store.create_dataset("DATASET-NEW", "New diagnostics")
    docs = [
        _make_document(store, f"DOC-LEGACY-{index}", None)
        for index in range(7)
    ]
    docs.append(_make_document(store, "DOC-DANGLING", "DATASET-NOT-REGISTERED"))
    _write_raw_legacy_catalog_fixture(store, docs)

    valid_ids = [dataset.dataset_id for dataset in store.list_datasets()]
    missing = find_missing_document_associations(store.read_catalog())
    invalid = find_invalid_document_associations(store.read_catalog(), valid_ids)
    counts = compute_unassociated_counts(store.list_datasets(), pd.DataFrame(), store.read_catalog())

    assert len(missing) == 7
    assert {doc.document_id for doc in invalid} == {"DOC-DANGLING"}
    assert counts["unassociated_documents"] == 8
    assert compute_workspace_summary(DEFAULT_DATASET_ID, None, store.read_catalog())["total_documents"] == 0
    assert compute_workspace_summary("DATASET-NEW", None, store.read_catalog())["total_documents"] == 0

    before_documents = [doc.to_dict() for doc in store.read_catalog()]
    protected_files = {
        store.base_dir / doc.storage_reference: (store.base_dir / doc.storage_reference).read_bytes()
        for doc in docs
    }
    protected_files.update({
        store.base_dir / doc.extracted_text_reference:
            (store.base_dir / doc.extracted_text_reference).read_bytes()
        for doc in docs
        if doc.extracted_text_reference
    })

    migration = store.migrate_legacy_data()
    migrated_docs = store.read_catalog()
    after_documents = [doc.to_dict() for doc in migrated_docs]

    assert migration["migrated_documents"] == 7
    assert [doc["document_id"] for doc in after_documents] == [
        doc["document_id"] for doc in before_documents
    ]
    for before, after in zip(before_documents, after_documents):
        expected = dict(before)
        if expected["document_id"].startswith("DOC-LEGACY-"):
            expected["dataset_id"] = DEFAULT_DATASET_ID
        assert after == expected
    assert store.get_metadata("DOC-DANGLING").dataset_id == "DATASET-NOT-REGISTERED"
    assert all(path.read_bytes() == content for path, content in protected_files.items())

    second_migration = store.migrate_legacy_data()
    assert second_migration["migrated_documents"] == 0
    assert second_migration["already_associated_documents"] == 8

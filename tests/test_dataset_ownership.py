"""M7.4.1 validation for dataset ownership across local persistence layers."""

import json
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from src.data_loader import (
    DatasetLoadError,
    load_incidents_data,
    require_valid_incident_dataset_ownership,
    validate_incident_dataset_ownership,
)
from src.indexer import SearchIndex
from src.ingestion import IngestionService
from src.models import (
    DEFAULT_DATASET_ID,
    DatasetAssociationIssue,
    DatasetOwnershipError,
    DocumentMetadata,
    DocumentStatus,
)
from src.storage import EvidenceStore


@pytest.fixture
def ownership_store(tmp_path: Path) -> EvidenceStore:
    store = EvidenceStore(base_dir=tmp_path / "evidence")
    store.ensure_default_dataset()
    store.create_dataset("DATASET-TEST-B", "Test Dataset B")
    return store


@pytest.fixture
def ownership_index(tmp_path: Path) -> SearchIndex:
    return SearchIndex(db_path=tmp_path / "index" / "evidence.db")


def _document(
    store: EvidenceStore,
    document_id: str,
    dataset_id: Optional[str],
    text: str = "Bearing inspection procedure. Replace the damaged seal.",
) -> DocumentMetadata:
    original_reference = store.save_original(
        document_id,
        f"{document_id}.txt",
        text.encode("utf-8"),
    )
    extracted_reference = store.save_extracted_text(document_id, text)
    return DocumentMetadata(
        document_id=document_id,
        original_filename=f"{document_id}.txt",
        file_type="txt",
        source_type="test",
        storage_reference=original_reference,
        extraction_status="success",
        ingestion_status=DocumentStatus.PROCESSED,
        created_at="2026-11-01T00:00:00+00:00",
        updated_at="2026-11-01T00:00:00+00:00",
        extracted_text_reference=extracted_reference,
        extracted_character_count=len(text),
        dataset_id=dataset_id,
    )


def _codes(issues: list[DatasetAssociationIssue]) -> set[str]:
    return {issue.code for issue in issues}


def _write_catalog_fixture(store: EvidenceStore, documents: list[DocumentMetadata]) -> None:
    payload = [document.to_dict() for document in documents]
    temp_path = store.catalog_file.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temp_path.replace(store.catalog_file)


def _failed_unassigned_document(
    store: EvidenceStore,
    document_id: str = "DOC-FAILED-UNASSIGNED",
) -> DocumentMetadata:
    metadata = _document(store, document_id, None)
    metadata.ingestion_status = DocumentStatus.FAILED
    _write_catalog_fixture(store, [metadata])
    return metadata


def test_registered_dataset_and_document_ownership_are_valid(ownership_store):
    metadata = _document(
        ownership_store,
        "DOC-VALID",
        "DATASET-TEST-B",
    )
    ownership_store.add_or_update_metadata(metadata)

    assert ownership_store.validate_dataset_id("DATASET-TEST-B")
    assert ownership_store.require_dataset_id(
        metadata.dataset_id,
        "Document",
        metadata.document_id,
    ).dataset_id == "DATASET-TEST-B"
    assert ownership_store.validate_document_ownership(
        metadata,
        require_catalog_match=True,
    ).dataset_id == "DATASET-TEST-B"
    assert ownership_store.validate_catalog_ownership() == []


def test_unknown_document_dataset_is_rejected_before_ingestion_writes(ownership_store):
    service = IngestionService(store=ownership_store)

    with pytest.raises(DatasetOwnershipError, match="unregistered dataset"):
        service.ingest_document(
            "unknown-owner.txt",
            b"Document content must not be persisted.",
            dataset_id="DATASET-NOT-REGISTERED",
        )

    assert ownership_store.read_catalog() == []
    assert list(ownership_store.originals_dir.iterdir()) == []
    assert list(ownership_store.extracted_dir.iterdir()) == []


def test_blank_document_dataset_is_rejected_instead_of_defaulted(ownership_store):
    service = IngestionService(store=ownership_store)

    with pytest.raises(DatasetOwnershipError, match="cannot be blank"):
        service.ingest_document(
            "blank-owner.txt",
            b"Document content.",
            dataset_id=" ",
        )

    assert ownership_store.read_catalog() == []


def test_missing_and_unregistered_document_associations_are_distinct(ownership_store):
    with pytest.raises(DatasetOwnershipError, match="ownership is required"):
        ownership_store.save_catalog([
            _document(ownership_store, "DOC-MISSING", None),
            _document(ownership_store, "DOC-DANGLING", "DATASET-DELETED"),
        ])

    _write_catalog_fixture(
        ownership_store,
        [
            _document(ownership_store, "DOC-MISSING", None),
            _document(ownership_store, "DOC-DANGLING", "DATASET-DELETED"),
        ],
    )

    issues = ownership_store.validate_catalog_ownership()
    assert _codes(issues) == {"missing_dataset_id", "unregistered_dataset_id"}
    assert {issue.entity_id for issue in issues if issue.code == "missing_dataset_id"} == {
        "DOC-MISSING"
    }
    assert {
        issue.entity_id
        for issue in issues
        if issue.code == "unregistered_dataset_id"
    } == {"DOC-DANGLING"}
    assert ownership_store.get_metadata("DOC-MISSING").dataset_id is None
    assert ownership_store.get_metadata("DOC-DANGLING").dataset_id == "DATASET-DELETED"


def test_incident_ownership_validation_reports_missing_and_unknown_ids():
    incidents = pd.DataFrame([
        {"incident_id": "INC-VALID", "dataset_id": DEFAULT_DATASET_ID},
        {"incident_id": "INC-MISSING", "dataset_id": None},
        {"incident_id": "INC-DANGLING", "dataset_id": "DATASET-DELETED"},
    ])

    issues = validate_incident_dataset_ownership(
        incidents,
        [DEFAULT_DATASET_ID],
    )
    assert validate_incident_dataset_ownership(
        incidents.iloc[[0]],
        [DEFAULT_DATASET_ID],
    ) == []
    assert [(issue.entity_id, issue.code) for issue in issues] == [
        ("INC-MISSING", "missing_dataset_id"),
        ("INC-DANGLING", "unregistered_dataset_id"),
    ]
    with pytest.raises(DatasetOwnershipError, match="INC-DANGLING"):
        require_valid_incident_dataset_ownership(
            incidents,
            [DEFAULT_DATASET_ID],
        )


def test_load_incidents_validates_registered_owners_without_rewriting_csv(tmp_path):
    csv_path = tmp_path / "incidents.csv"
    csv_path.write_text(
        "incident_id,machine_id,date,symptom,maintenance_action,outcome,dataset_id\n"
        "INC-1,M-100,2026-01-01,Vibration,Inspected,Pending,DATASET-UNKNOWN\n",
        encoding="utf-8",
    )
    original = csv_path.read_bytes()

    with pytest.raises(DatasetLoadError, match="unregistered dataset"):
        load_incidents_data(
            csv_path,
            valid_dataset_ids=[DEFAULT_DATASET_ID],
        )

    assert csv_path.read_bytes() == original


def test_legacy_incident_loading_assigns_default_only_in_memory(tmp_path):
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text(
        "incident_id,machine_id,date,symptom,maintenance_action,outcome\n"
        "INC-LEGACY,M-100,2026-01-01,Vibration,Inspected,Pending\n",
        encoding="utf-8",
    )
    original = csv_path.read_bytes()

    incidents = load_incidents_data(
        csv_path,
        valid_dataset_ids=[DEFAULT_DATASET_ID],
    )

    assert incidents.loc[0, "dataset_id"] == DEFAULT_DATASET_ID
    assert csv_path.read_bytes() == original


def test_index_references_catalog_document_with_matching_registered_owner(
    ownership_store,
    ownership_index,
):
    metadata = _document(ownership_store, "DOC-INDEXED", "DATASET-TEST-B")
    ownership_store.add_or_update_metadata(metadata)

    count = ownership_index.index_document(
        metadata,
        ownership_store.get_extracted_text(metadata),
        store=ownership_store,
    )

    assert count > 0
    assert ownership_index.validate_integrity(ownership_store) == []


def test_index_integrity_detects_missing_catalog_document(ownership_store, ownership_index):
    orphan = _document(
        ownership_store,
        "DOC-ORPHAN-INDEX",
        DEFAULT_DATASET_ID,
    )
    with pytest.raises(DatasetOwnershipError, match="not present in the evidence catalog"):
        ownership_index.index_document(
            orphan,
            "Orphaned indexed evidence chunk.",
            store=ownership_store,
        )

    assert ownership_index.get_index_stats()["document_count"] == 0


def test_index_sync_reports_unassigned_documents_without_assigning_or_reindexing(
    ownership_store,
    ownership_index,
):
    metadata = _document(ownership_store, "DOC-UNASSIGNED-SYNC", "DATASET-TEST-B")
    ownership_store.add_or_update_metadata(metadata)
    ownership_index.index_document(
        metadata,
        ownership_store.get_extracted_text(metadata),
        store=ownership_store,
    )
    metadata.dataset_id = None
    _write_catalog_fixture(ownership_store, [metadata])

    sync_result = ownership_index.sync_with_store(ownership_store)

    assert len(sync_result["errors"]) >= 1
    assert "has no dataset ID" in sync_result["errors"][0]
    assert ownership_store.get_metadata(metadata.document_id).dataset_id is None
    assert ownership_index.get_index_stats()["document_count"] == 1
    assert "index_document_dataset_mismatch" in _codes(
        ownership_index.validate_integrity(ownership_store)
    )


def test_maintenance_reports_expected_failed_unindexed_document_as_informational(
    ownership_store,
    ownership_index,
):
    metadata = _failed_unassigned_document(ownership_store)

    preview = ownership_index.preview_sync_with_store(ownership_store)

    assert preview["ready"] is True
    assert preview["errors"] == []
    assert preview["to_index_document_ids"] == []
    assert preview["informational"] == [
        {
            "code": "expected_failed_unindexed_document",
            "document_id": metadata.document_id,
            "message": (
                f"Document '{metadata.document_id}' is failed, unassigned, and absent "
                "from the index; maintenance will leave it unchanged."
            ),
        }
    ]
    assert "missing_dataset_id" in _codes(
        ownership_index.validate_integrity(ownership_store)
    )
    assert ownership_store.get_metadata(metadata.document_id).dataset_id is None
    assert ownership_index.get_index_stats()["document_count"] == 0


@pytest.mark.parametrize("index_artifact", ["indexed_document", "chunk", "fts"])
def test_failed_unassigned_document_with_any_index_artifact_blocks_maintenance(
    ownership_store,
    ownership_index,
    index_artifact,
):
    metadata = _failed_unassigned_document(ownership_store)
    with ownership_index._get_connection() as connection:
        if index_artifact in {"indexed_document", "chunk"}:
            connection.execute(
                """
                INSERT INTO indexed_documents (
                    document_id, source_filename, content_hash, chunk_count,
                    indexed_at, dataset_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.document_id,
                    metadata.original_filename,
                    "failed-hash",
                    1 if index_artifact == "chunk" else 0,
                    metadata.created_at,
                    None,
                ),
            )
        if index_artifact == "chunk":
            connection.execute(
                """
                INSERT INTO evidence_chunks (
                    chunk_id, document_id, chunk_index, text, source_filename,
                    file_type, page_number, start_offset, end_offset, created_at,
                    content_hash, dataset_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "CHUNK-FAILED-UNASSIGNED",
                    metadata.document_id,
                    0,
                    "Failed document must not be indexed.",
                    metadata.original_filename,
                    metadata.file_type,
                    None,
                    0,
                    39,
                    metadata.created_at,
                    "failed-hash",
                    None,
                ),
            )
        if index_artifact == "fts":
            connection.execute(
                """
                INSERT INTO chunks_fts (chunk_id, document_id, source_filename, text)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "CHUNK-FAILED-UNASSIGNED",
                    metadata.document_id,
                    metadata.original_filename,
                    "Failed document must not be indexed.",
                ),
            )

    preview = ownership_index.preview_sync_with_store(ownership_store)

    assert preview["ready"] is False
    assert preview["errors"]
    assert preview["informational"] == []
    assert ownership_store.get_metadata(metadata.document_id).dataset_id is None
    with pytest.raises(DatasetOwnershipError, match="Cannot rebuild"):
        ownership_index.rebuild_index(ownership_store)


def test_processed_unassigned_document_remains_blocking_for_maintenance(
    ownership_store,
    ownership_index,
):
    metadata = _document(ownership_store, "DOC-PROCESSED-UNASSIGNED", None)
    _write_catalog_fixture(ownership_store, [metadata])

    preview = ownership_index.preview_sync_with_store(ownership_store)

    assert preview["ready"] is False
    assert any("has no dataset ID" in error for error in preview["errors"])
    assert preview["informational"] == []
    with pytest.raises(DatasetOwnershipError, match="Cannot rebuild"):
        ownership_index.rebuild_index(ownership_store)


def test_rebuild_allows_and_preserves_expected_failed_unindexed_document(
    ownership_store,
    ownership_index,
):
    metadata = _failed_unassigned_document(ownership_store)

    result = ownership_index.rebuild_index(ownership_store)

    assert result["verified"] is True
    assert result["errors"] == []
    assert result["indexed_documents"] == 0
    assert result["total_chunks"] == 0
    assert result["informational"][0]["document_id"] == metadata.document_id
    assert ownership_store.get_metadata(metadata.document_id).dataset_id is None
    assert ownership_index.get_index_stats()["document_count"] == 0


def test_rebuild_refuses_to_destroy_index_when_ownership_is_inconsistent(
    ownership_store,
    ownership_index,
):
    metadata = _document(ownership_store, "DOC-REBUILD-PROTECT", "DATASET-TEST-B")
    ownership_store.add_or_update_metadata(metadata)
    ownership_index.index_document(
        metadata,
        "Existing indexed content must remain available.",
        store=ownership_store,
    )
    with sqlite3.connect(ownership_index.db_path) as connection:
        connection.execute(
            "UPDATE indexed_documents SET dataset_id = NULL WHERE document_id = ?",
            (metadata.document_id,),
        )
        connection.execute(
            "UPDATE evidence_chunks SET dataset_id = NULL WHERE document_id = ?",
            (metadata.document_id,),
        )
        connection.commit()
    before_stats = ownership_index.get_index_stats()
    before_results = ownership_index._get_connection().execute(
        "SELECT chunk_id, text FROM evidence_chunks"
    ).fetchall()

    with pytest.raises(DatasetOwnershipError, match="Cannot rebuild"):
        ownership_index.rebuild_index(ownership_store)

    assert ownership_index.get_index_stats() == before_stats
    after_results = ownership_index._get_connection().execute(
        "SELECT chunk_id, text FROM evidence_chunks"
    ).fetchall()
    assert [tuple(row) for row in after_results] == [tuple(row) for row in before_results]


def test_index_integrity_detects_owner_mismatch_and_chunk_mismatch(
    ownership_store,
    ownership_index,
):
    catalog_document = _document(
        ownership_store,
        "DOC-OWNER-MISMATCH",
        "DATASET-TEST-B",
    )
    ownership_store.add_or_update_metadata(catalog_document)
    ownership_index.index_document(
        catalog_document,
        "Bearing temperature rose during a loaded operation.",
        store=ownership_store,
    )
    unvalidated_copy = DocumentMetadata.from_dict(catalog_document.to_dict())
    unvalidated_copy.dataset_id = DEFAULT_DATASET_ID
    with pytest.raises(DatasetOwnershipError, match="does not match the canonical"):
        ownership_index.index_document(
            unvalidated_copy,
            "Bearing temperature rose during a loaded operation.",
            store=ownership_store,
        )

    with sqlite3.connect(ownership_index.db_path) as connection:
        connection.execute(
            "UPDATE indexed_documents SET dataset_id = ? WHERE document_id = ?",
            (DEFAULT_DATASET_ID, catalog_document.document_id),
        )
        connection.execute(
            "UPDATE evidence_chunks SET dataset_id = ? WHERE document_id = ?",
            ("DATASET-TEST-B", catalog_document.document_id),
        )
        connection.commit()

    issues = ownership_index.validate_integrity(ownership_store)
    assert "index_document_dataset_mismatch" in _codes(issues)
    assert "index_chunk_dataset_mismatch" in _codes(issues)
    preview = ownership_index.preview_sync_with_store(ownership_store)
    assert preview["ready"] is False
    with pytest.raises(DatasetOwnershipError, match="Cannot rebuild"):
        ownership_index.rebuild_index(ownership_store)


def test_indexing_with_store_rejects_missing_or_mismatched_canonical_document(
    ownership_store,
    ownership_index,
):
    metadata = _document(ownership_store, "DOC-NOT-CATALOGED", DEFAULT_DATASET_ID)
    with pytest.raises(DatasetOwnershipError, match="not present in the evidence catalog"):
        ownership_index.index_document(
            metadata,
            "Evidence text.",
            store=ownership_store,
        )
    assert ownership_index.get_index_stats()["document_count"] == 0


def test_validation_is_read_only_for_existing_catalog_files_and_index(
    ownership_store,
    ownership_index,
):
    metadata = _document(ownership_store, "DOC-PRESERVE", "DATASET-TEST-B")
    ownership_store.add_or_update_metadata(metadata)
    ownership_index.index_document(
        metadata,
        "Previously indexed text.",
        store=ownership_store,
    )
    metadata.dataset_id = None
    _write_catalog_fixture(ownership_store, [metadata])

    catalog_before = ownership_store.catalog_file.read_bytes()
    registry_before = ownership_store.datasets_file.read_bytes()
    with sqlite3.connect(ownership_index.db_path) as connection:
        index_before = connection.execute(
            "SELECT document_id, dataset_id FROM indexed_documents"
        ).fetchall()
        chunks_before = connection.execute(
            "SELECT chunk_id, document_id, dataset_id, text FROM evidence_chunks"
        ).fetchall()

    first_issues = ownership_index.validate_integrity(ownership_store)
    second_issues = ownership_index.validate_integrity(ownership_store)

    assert first_issues == second_issues
    assert ownership_store.catalog_file.read_bytes() == catalog_before
    assert ownership_store.datasets_file.read_bytes() == registry_before
    with sqlite3.connect(ownership_index.db_path) as connection:
        assert connection.execute(
            "SELECT document_id, dataset_id FROM indexed_documents"
        ).fetchall() == index_before
        assert connection.execute(
            "SELECT chunk_id, document_id, dataset_id, text FROM evidence_chunks"
        ).fetchall() == chunks_before


def test_global_integrity_summary_includes_every_registered_scope_and_unique_counts(
    ownership_store,
    ownership_index,
):
    ownership_store.create_dataset("DATASET-EMPTY", "Empty Dataset")
    metadata = _document(ownership_store, "DOC-GLOBAL-INTEGRITY", "DATASET-TEST-B")
    ownership_store.add_or_update_metadata(metadata)
    ownership_index.index_document(metadata, "Indexed evidence.", store=ownership_store)
    metadata.dataset_id = None
    _write_catalog_fixture(ownership_store, [metadata])

    summary = ownership_index.summarize_integrity(ownership_store)

    assert {
        DEFAULT_DATASET_ID,
        "DATASET-TEST-B",
        "DATASET-EMPTY",
        "UNASSIGNED",
    } <= set(summary["findings_by_scope"])
    assert summary["total_findings"] > summary["unique_affected_document_count"]
    assert summary["unique_affected_document_count"] == 1
    assert summary["findings_by_scope"]["UNASSIGNED"]["affected_document_count"] == 1


APPROVED_REPAIR_DOCUMENTS = (
    "DOC-5A82BA6A3A4C",
    "DOC-C6C0555F1F0D",
    "DOC-9406F467ECCE",
    "DOC-71B10A3BA0C8",
    "DOC-381BC31C33A9",
    "DOC-16FE64012FD1",
)


def _seed_confirmed_repair_fixture(store, index):
    documents = []
    for position, document_id in enumerate(APPROVED_REPAIR_DOCUMENTS):
        metadata = _document(
            store,
            document_id,
            DEFAULT_DATASET_ID,
            text=f"Synthetic processed repair evidence {position}.",
        )
        store.add_or_update_metadata(metadata)
        index.index_document(metadata, store.get_extracted_text(metadata), store=store)
        metadata.dataset_id = None
        documents.append(metadata)

    failed_id = "DOC-1C401DB99DE6"
    failed = DocumentMetadata(
        document_id=failed_id,
        original_filename="failed.pdf",
        file_type="pdf",
        source_type="test",
        storage_reference=store.save_original(
            failed_id,
            "failed.pdf",
            b"failed extraction",
        ),
        extraction_status="failed",
        ingestion_status=DocumentStatus.FAILED,
        created_at="2026-11-01T00:00:00+00:00",
        updated_at="2026-11-01T00:00:00+00:00",
        dataset_id=None,
    )
    _write_catalog_fixture(store, documents + [failed])
    return {
        document_id: DEFAULT_DATASET_ID
        for document_id in APPROVED_REPAIR_DOCUMENTS
    }


def _repair_fts_snapshot(index):
    with index._get_connection() as connection:
        return [
            tuple(row)
            for row in connection.execute(
                "SELECT chunk_id, document_id, source_filename, text "
                "FROM chunks_fts ORDER BY chunk_id"
            ).fetchall()
        ]


def test_confirmed_ownership_repair_is_scoped_retryable_and_preserves_failed_document(
    ownership_store,
    ownership_index,
):
    assignments = _seed_confirmed_repair_fixture(ownership_store, ownership_index)
    stats_before = ownership_index.get_index_stats()
    fts_before = _repair_fts_snapshot(ownership_index)
    catalog_before = ownership_store.catalog_file.read_bytes()
    database_before = ownership_index.db_path.read_bytes()

    plan = ownership_store.preview_document_ownership_repair(
        assignments,
        ownership_index,
        required_status=DocumentStatus.PROCESSED,
    )
    assert plan["ready"] is True
    assert plan["changes_required"] == 6
    assert all(item["catalog_owner"] is None for item in plan["items"])
    assert all(item["indexed_document_owners"] == [DEFAULT_DATASET_ID] for item in plan["items"])
    assert all(
        all(chunk["dataset_id"] == DEFAULT_DATASET_ID for chunk in item["chunk_owners"])
        for item in plan["items"]
    )
    assert all(item["fts"]["consistent"] for item in plan["items"])
    assert ownership_store.catalog_file.read_bytes() == catalog_before
    assert ownership_index.db_path.read_bytes() == database_before

    first_apply = ownership_store.apply_document_ownership_repair(
        plan,
        ownership_index,
        required_status=DocumentStatus.PROCESSED,
    )
    second_apply = ownership_store.apply_document_ownership_repair(
        plan,
        ownership_index,
        required_status=DocumentStatus.PROCESSED,
    )

    assert set(first_apply["changed_document_ids"]) == set(assignments)
    assert first_apply["verified"] is True
    assert set(second_apply["already_applied_document_ids"]) == set(assignments)
    assert ownership_store.get_metadata("DOC-1C401DB99DE6").dataset_id is None
    assert ownership_index.get_index_stats() == stats_before
    assert _repair_fts_snapshot(ownership_index) == fts_before
    after = ownership_store.preview_document_ownership_repair(assignments, ownership_index)
    assert after["ready"] is True
    assert after["changes_required"] == 0
    with sqlite3.connect(ownership_index.db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM evidence_chunks WHERE dataset_id = ?",
            (DEFAULT_DATASET_ID,),
        ).fetchone()[0] == 6
        assert connection.execute(
            "SELECT COUNT(*) FROM indexed_documents WHERE dataset_id = ?",
            (DEFAULT_DATASET_ID,),
        ).fetchone()[0] == 6


def test_ownership_repair_rejects_unapproved_ids_and_targets(ownership_store, ownership_index):
    assignments = _seed_confirmed_repair_fixture(ownership_store, ownership_index)

    unapproved = dict(assignments)
    unapproved["DOC-OTHER"] = DEFAULT_DATASET_ID
    assert not ownership_store.preview_document_ownership_repair(
        unapproved, ownership_index
    )["ready"]

    invalid_target = {document_id: "DATASET-NOT-REGISTERED" for document_id in assignments}
    assert not ownership_store.preview_document_ownership_repair(
        invalid_target, ownership_index
    )["ready"]


def test_ownership_repair_preview_rejects_fts_mismatch(ownership_store, ownership_index):
    assignments = _seed_confirmed_repair_fixture(ownership_store, ownership_index)
    with ownership_index._get_connection() as connection:
        connection.execute(
            "UPDATE chunks_fts SET text = 'corrupt fts content' "
            "WHERE document_id = ?",
            (APPROVED_REPAIR_DOCUMENTS[0],),
        )

    plan = ownership_store.preview_document_ownership_repair(assignments, ownership_index)

    assert not plan["ready"]
    assert any("FTS/relational consistency failed" in error for error in plan["errors"])


def test_ownership_repair_preview_rejects_missing_index_rows(ownership_store, ownership_index):
    assignments = _seed_confirmed_repair_fixture(ownership_store, ownership_index)
    with ownership_index._get_connection() as connection:
        connection.execute(
            "DELETE FROM indexed_documents WHERE document_id = ?",
            (APPROVED_REPAIR_DOCUMENTS[0],),
        )

    plan = ownership_store.preview_document_ownership_repair(assignments, ownership_index)

    assert not plan["ready"]
    first_item = next(
        item for item in plan["items"]
        if item["document_id"] == APPROVED_REPAIR_DOCUMENTS[0]
    )
    assert any("indexed-document row count is 0" in error for error in first_item["record_errors"])


def test_ownership_repair_preview_rejects_unexpected_existing_owner(
    ownership_store,
    ownership_index,
):
    assignments = _seed_confirmed_repair_fixture(ownership_store, ownership_index)
    with ownership_index._get_connection() as connection:
        connection.execute(
            "UPDATE indexed_documents SET dataset_id = ? WHERE document_id = ?",
            ("DATASET-TEST-B", APPROVED_REPAIR_DOCUMENTS[0]),
        )

    plan = ownership_store.preview_document_ownership_repair(assignments, ownership_index)

    assert not plan["ready"]
    assert any(
        APPROVED_REPAIR_DOCUMENTS[0] in error and "unexpected non-default ownership" in error
        for error in plan["errors"]
    )


def test_ownership_repair_can_resume_after_interruption(ownership_store, ownership_index, monkeypatch):
    assignments = _seed_confirmed_repair_fixture(ownership_store, ownership_index)
    plan = ownership_store.preview_document_ownership_repair(assignments, ownership_index)
    original_apply_index = ownership_index._apply_repair_index

    def interrupt_before_index_update(journal, requested_assignments):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(ownership_index, "_apply_repair_index", interrupt_before_index_update)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        ownership_store.apply_document_ownership_repair(plan, ownership_index)
    journal_path = next((ownership_store.metadata_dir / "ownership-repair").glob("*.json"))
    repair_id = journal_path.stem
    assert ownership_store.get_metadata(APPROVED_REPAIR_DOCUMENTS[0]).dataset_id == DEFAULT_DATASET_ID
    monkeypatch.setattr(ownership_index, "_apply_repair_index", original_apply_index)

    recovered = ownership_store.recover_document_ownership_repair(
        repair_id, ownership_index, action="resume"
    )

    assert recovered["verified"] is True
    assert recovered["recovered"] is True
    assert ownership_store.preview_document_ownership_repair(assignments, ownership_index)[
        "changes_required"
    ] == 0


def test_ownership_repair_can_roll_back_from_journal(ownership_store, ownership_index):
    assignments = _seed_confirmed_repair_fixture(ownership_store, ownership_index)
    before_fts = _repair_fts_snapshot(ownership_index)
    plan = ownership_store.preview_document_ownership_repair(assignments, ownership_index)
    result = ownership_store.apply_document_ownership_repair(plan, ownership_index)

    rollback = ownership_store.recover_document_ownership_repair(
        result["repair_id"], ownership_index, action="rollback"
    )

    assert rollback["status"] == "rolled_back"
    assert all(
        ownership_store.get_metadata(document_id).dataset_id is None
        for document_id in APPROVED_REPAIR_DOCUMENTS
    )
    with sqlite3.connect(ownership_index.db_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM indexed_documents WHERE dataset_id = ?",
            (DEFAULT_DATASET_ID,),
        ).fetchone()[0] == 6
        assert connection.execute(
            "SELECT COUNT(*) FROM evidence_chunks WHERE dataset_id = ?",
            (DEFAULT_DATASET_ID,),
        ).fetchone()[0] == 6
    assert _repair_fts_snapshot(ownership_index) == before_fts


def test_legacy_migration_is_explicit_idempotent_and_preserves_dangling_owner(
    ownership_store,
):
    missing = _document(ownership_store, "DOC-LEGACY-MISSING", None)
    dangling = _document(ownership_store, "DOC-LEGACY-DANGLING", "DATASET-REMOVED")
    _write_catalog_fixture(ownership_store, [missing, dangling])
    original_file_bytes = {
        ownership_store.base_dir / metadata.storage_reference:
            (ownership_store.base_dir / metadata.storage_reference).read_bytes()
        for metadata in (missing, dangling)
    }
    original_file_bytes.update({
        ownership_store.base_dir / metadata.extracted_text_reference:
            (ownership_store.base_dir / metadata.extracted_text_reference).read_bytes()
        for metadata in (missing, dangling)
    })

    first = ownership_store.migrate_legacy_data()
    migrated_catalog = ownership_store.read_catalog()
    second = ownership_store.migrate_legacy_data()

    assert first["migrated_documents"] == 1
    assert second["migrated_documents"] == 0
    assert ownership_store.get_metadata("DOC-LEGACY-MISSING").dataset_id == DEFAULT_DATASET_ID
    assert ownership_store.get_metadata("DOC-LEGACY-DANGLING").dataset_id == "DATASET-REMOVED"
    assert [doc.document_id for doc in migrated_catalog] == [
        "DOC-LEGACY-MISSING",
        "DOC-LEGACY-DANGLING",
    ]
    assert all(path.read_bytes() == content for path, content in original_file_bytes.items())


def test_registered_dataset_cannot_be_removed_through_registry_save(ownership_store):
    before = ownership_store.datasets_file.read_bytes()
    with pytest.raises(DatasetOwnershipError, match="Dataset deletion is not supported"):
        ownership_store.save_datasets([
            dataset
            for dataset in ownership_store.list_datasets()
            if dataset.dataset_id != "DATASET-TEST-B"
        ])

    assert ownership_store.datasets_file.read_bytes() == before
    assert ownership_store.validate_dataset_id("DATASET-TEST-B")


def test_default_dataset_remains_available_for_legacy_ingestion(
    tmp_path,
    ownership_index,
):
    store = EvidenceStore(base_dir=tmp_path / "empty-evidence")
    store.ensure_default_dataset()
    service = IngestionService(store=store)
    metadata = service.ingest_document(
        "legacy-default.txt",
        b"Default dataset ownership remains explicit.",
        dataset_id=DEFAULT_DATASET_ID,
    )

    assert metadata.dataset_id == DEFAULT_DATASET_ID
    assert store.validate_dataset_id(DEFAULT_DATASET_ID)
    assert ownership_index.index_document(
        metadata,
        store.get_extracted_text(metadata),
        store=store,
    ) > 0
    assert ownership_index.validate_integrity(store) == []

"""Dataset-scoped retrieval regression tests using isolated evidence storage."""

import pytest

from src.indexer import SearchIndex
from src.models import (
    DEFAULT_DATASET_ID,
    DatasetOwnershipError,
    DocumentMetadata,
    DocumentStatus,
)
from src.retriever import EvidenceRetriever
from src.storage import EvidenceStore


@pytest.fixture
def scoped_retriever(tmp_path):
    store = EvidenceStore(base_dir=tmp_path / "evidence")
    store.ensure_default_dataset()
    dataset_a = store.create_dataset("DATASET-A", "Dataset A")
    dataset_b = store.create_dataset("DATASET-B", "Dataset B")
    index = SearchIndex(db_path=tmp_path / "index" / "evidence_index.db")

    documents = {}
    for doc_id, filename, dataset_id, text in (
        ("DOC-A", "a_manual.txt", dataset_a.dataset_id, "Shared bearing temperature guidance."),
        ("DOC-B", "b_manual.txt", dataset_b.dataset_id, "Shared bearing temperature record."),
        ("DOC-A-RANK", "a_detailed.txt", dataset_a.dataset_id, "Shared bearing temperature guidance with detailed bearing temperature limits."),
    ):
        metadata = DocumentMetadata(
            document_id=doc_id,
            original_filename=filename,
            file_type="txt",
            source_type="test",
            storage_reference=f"originals/{doc_id}_{filename}",
            extraction_status="success",
            ingestion_status=DocumentStatus.PROCESSED,
            created_at="2026-11-01T00:00:00+00:00",
            updated_at="2026-11-01T00:00:00+00:00",
            extracted_text_reference=f"extracted/{doc_id}.txt",
            extracted_character_count=len(text),
            dataset_id=dataset_id,
        )
        store.add_or_update_metadata(metadata)
        index.index_document(metadata, text, store=store)
        documents[doc_id] = metadata

    return EvidenceRetriever(index, store), index, store, dataset_a, dataset_b, documents


def test_each_dataset_retrieves_only_its_own_evidence(scoped_retriever):
    retriever, _, _, dataset_a, dataset_b, documents = scoped_retriever

    results_a = retriever.search("bearing temperature", dataset_id=dataset_a.dataset_id)
    results_b = retriever.search("bearing temperature", dataset_id=dataset_b.dataset_id)

    assert results_a
    assert {result.chunk.document_id for result in results_a} == {"DOC-A", "DOC-A-RANK"}
    assert all(result.chunk.dataset_id == dataset_a.dataset_id for result in results_a)
    assert {result.chunk.document_id for result in results_b} == {"DOC-B"}
    assert all(result.chunk.dataset_id == dataset_b.dataset_id for result in results_b)
    assert all(
        result.chunk.source_filename == documents[result.chunk.document_id].original_filename
        for result in results_a + results_b
    )


def test_identical_terms_do_not_leak_between_datasets(scoped_retriever):
    retriever, _, _, dataset_a, dataset_b, _ = scoped_retriever

    results_a = retriever.search("shared bearing", dataset_id=dataset_a.dataset_id)
    results_b = retriever.search("shared bearing", dataset_id=dataset_b.dataset_id)

    assert results_a and results_b
    assert {result.chunk.dataset_id for result in results_a} == {dataset_a.dataset_id}
    assert {result.chunk.dataset_id for result in results_b} == {dataset_b.dataset_id}


@pytest.mark.parametrize("dataset_id", [None, "", "   "])
def test_missing_dataset_id_is_rejected(scoped_retriever, dataset_id):
    retriever, *_ = scoped_retriever

    with pytest.raises(DatasetOwnershipError, match="no dataset ID|cannot be blank"):
        retriever.search("bearing", dataset_id=dataset_id)


def test_unregistered_dataset_id_is_rejected(scoped_retriever):
    retriever, *_ = scoped_retriever

    with pytest.raises(DatasetOwnershipError, match="unregistered dataset"):
        retriever.search("bearing", dataset_id="DATASET-UNKNOWN")


def test_catalog_store_is_required_even_with_a_dataset_id(scoped_retriever):
    _, index, _, dataset_a, _, _ = scoped_retriever

    with pytest.raises(DatasetOwnershipError, match="requires an EvidenceStore"):
        EvidenceRetriever(index).search("bearing", dataset_id=dataset_a.dataset_id)


def test_catalog_index_mismatch_blocks_results_without_repair(scoped_retriever):
    retriever, index, store, dataset_a, dataset_b, _ = scoped_retriever

    with index._get_connection() as conn:
        conn.execute(
            "UPDATE indexed_documents SET dataset_id = ? WHERE document_id = ?",
            (dataset_b.dataset_id, "DOC-A"),
        )

    with pytest.raises(DatasetOwnershipError, match="integrity issue"):
        retriever.search("bearing", dataset_id=dataset_a.dataset_id)

    assert store.get_metadata("DOC-A").dataset_id == dataset_a.dataset_id
    with index._get_connection() as conn:
        owner = conn.execute(
            "SELECT dataset_id FROM indexed_documents WHERE document_id = ?",
            ("DOC-A",),
        ).fetchone()["dataset_id"]
    assert owner == dataset_b.dataset_id


def test_results_preserve_chunk_provenance_and_ranking(scoped_retriever):
    retriever, _, _, dataset_a, _, _ = scoped_retriever

    results = retriever.search("bearing temperature", dataset_id=dataset_a.dataset_id)

    assert results[0].chunk.document_id == "DOC-A-RANK"
    assert results[0].score >= results[-1].score
    assert all(result.chunk.chunk_id and result.chunk.chunk_index == 0 for result in results)
    assert all("**" in result.matched_snippet for result in results)


def test_integrity_issue_in_other_dataset_does_not_block_clean_dataset(scoped_retriever):
    retriever, index, _, dataset_a, dataset_b, _ = scoped_retriever

    with index._get_connection() as conn:
        conn.execute(
            "UPDATE indexed_documents SET dataset_id = ? WHERE document_id = ?",
            (DEFAULT_DATASET_ID, "DOC-B"),
        )

    results = retriever.search("bearing temperature", dataset_id=dataset_a.dataset_id)

    assert results
    assert all(result.chunk.dataset_id == dataset_a.dataset_id for result in results)

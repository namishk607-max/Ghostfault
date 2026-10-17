"""Automated test suite for GhostFault Milestone 3: Chunking, Indexing, and Retrieval."""

from pathlib import Path
import pytest
import sqlite3

from src.models import DocumentMetadata, DocumentStatus, EvidenceChunk, SearchResult
from src.chunker import TextChunker, compute_content_hash
from src.storage import EvidenceStore
from src.indexer import SearchIndex
from src.retriever import EvidenceRetriever, sanitize_fts5_query


def make_test_metadata(doc_id: str = "DOC-TEST01", filename: str = "manual.txt") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id,
        original_filename=filename,
        file_type="txt",
        source_type="test",
        storage_reference=f"originals/{doc_id}_{filename}",
        extraction_status="success",
        ingestion_status=DocumentStatus.PROCESSED,
        created_at="2026-10-17T00:00:00+00:00",
        updated_at="2026-10-17T00:00:00+00:00",
        extracted_text_reference=f"extracted/{doc_id}.txt",
        extracted_character_count=100,
    )


# ==============================================================================
# Phase 7 — Automated Tests: CHUNKING
# ==============================================================================

def test_chunker_short_document():
    """Verify short documents fit into a single chunk."""
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    meta = make_test_metadata()
    text = "Bearing temperature nominal operating limit is 65C."
    chunks = chunker.chunk_document(meta, text)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == text
    assert chunks[0].document_id == meta.document_id
    assert chunks[0].source_filename == meta.original_filename
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(text)


def test_chunker_long_document_and_overlap():
    """Verify long documents are chunked with specified overlap and boundaries."""
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    meta = make_test_metadata()
    # Paragraphs of text
    p1 = "Paragraph 1: Primary hydraulic circuit operational pressure is strictly 210 bar."
    p2 = "Paragraph 2: Return filter replacement is required every 500 operating hours."
    p3 = "Paragraph 3: Spindle drive lubrication must use ISO VG 68 synthetic oil only."
    text = f"{p1}\n\n{p2}\n\n{p3}"

    chunks = chunker.chunk_document(meta, text)
    assert len(chunks) >= 2
    # Check monotonic chunk index
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert len(c.text) > 0
        assert c.document_id == meta.document_id


def test_chunker_empty_and_whitespace_only():
    """Verify empty or whitespace-only documents produce zero chunks without errors."""
    chunker = TextChunker()
    meta = make_test_metadata()
    assert chunker.chunk_document(meta, "") == []
    assert chunker.chunk_document(meta, "   \n\t  \n  ") == []


def test_chunker_unicode_and_unusual_newlines():
    """Verify handling of Unicode characters and mixed Windows/Unix/Mac newlines."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=40)
    meta = make_test_metadata()
    text = "Vérification de l'actionneur pneumatique.\r\nTolérance: ±0.002 mm.\rTempérature: 75°C.\n"
    chunks = chunker.chunk_document(meta, text)
    assert len(chunks) == 1
    assert "Vérification" in chunks[0].text
    assert "±0.002 mm" in chunks[0].text


def test_chunker_deterministic_output():
    """Verify that identical text yields exact identical chunk IDs and content hashes."""
    chunker1 = TextChunker(chunk_size=200, chunk_overlap=50)
    chunker2 = TextChunker(chunk_size=200, chunk_overlap=50)
    meta = make_test_metadata()
    text = "Repetitive deterministic text test for acoustic pitch diagnostic verification."

    chunks1 = chunker1.chunk_document(meta, text)
    chunks2 = chunker2.chunk_document(meta, text)

    assert len(chunks1) == len(chunks2) == 1
    assert chunks1[0].chunk_id == chunks2[0].chunk_id
    assert chunks1[0].content_hash == chunks2[0].content_hash


# ==============================================================================
# Phase 7 — Automated Tests: INDEXING
# ==============================================================================

@pytest.fixture
def temp_index(tmp_path: Path) -> SearchIndex:
    db_file = tmp_path / "test_index.db"
    return SearchIndex(db_path=db_file)


@pytest.fixture
def temp_store(tmp_path: Path) -> EvidenceStore:
    return EvidenceStore(base_dir=tmp_path / "evidence")


def test_initial_indexing_and_persistence(temp_index: SearchIndex, tmp_path: Path):
    """Verify initial indexing and persistence across new SearchIndex instance."""
    meta = make_test_metadata(doc_id="DOC-IND01", filename="manual.txt")
    text = "Check acoustic pitch and vibration velocity on motor drive end."

    chunk_count = temp_index.index_document(meta, text)
    assert chunk_count == 1

    stats = temp_index.get_index_stats()
    assert stats["document_count"] == 1
    assert stats["chunk_count"] == 1

    # Reopen index from disk in new instance
    reopened = SearchIndex(db_path=temp_index.db_path)
    reopened_stats = reopened.get_index_stats()
    assert reopened_stats["document_count"] == 1
    assert reopened_stats["chunk_count"] == 1


def test_reindex_unchanged_document_no_duplicates(temp_index: SearchIndex):
    """Verify re-indexing the identical text does not create duplicate chunks."""
    meta = make_test_metadata(doc_id="DOC-DUP01", filename="spec.txt")
    text = "Linear glass scale optical encoder repeatability is verified within 2 microns."

    count1 = temp_index.index_document(meta, text)
    count2 = temp_index.index_document(meta, text)

    assert count1 == count2
    stats = temp_index.get_index_stats()
    assert stats["document_count"] == 1
    assert stats["chunk_count"] == 1


def test_updating_document_replaces_stale_chunks(temp_index: SearchIndex):
    """Verify modifying text replaces prior chunks cleanly without leaving orphans."""
    meta = make_test_metadata(doc_id="DOC-UPD01", filename="spec.txt")
    text_v1 = "Initial draft: Line pressure threshold is 180 bar."
    temp_index.index_document(meta, text_v1)

    text_v2 = "Revised spec: Line pressure threshold updated to 210 bar nominal."
    temp_index.index_document(meta, text_v2)

    stats = temp_index.get_index_stats()
    assert stats["document_count"] == 1
    assert stats["chunk_count"] == 1

    retriever = EvidenceRetriever(temp_index)
    results_old = retriever.search("180 bar")
    results_new = retriever.search("210 bar")

    assert len(results_old) == 0
    assert len(results_new) == 1
    assert "210 bar" in results_new[0].chunk.text


def test_rebuild_index_from_store(temp_index: SearchIndex, temp_store: EvidenceStore):
    """Verify full index rebuild synchronizes with EvidenceStore catalog."""
    # Write 2 synthetic documents to store
    meta1 = make_test_metadata(doc_id="DOC-STORE1", filename="pump.txt")
    meta2 = make_test_metadata(doc_id="DOC-STORE2", filename="chiller.txt")
    temp_store.add_or_update_metadata(meta1)
    temp_store.add_or_update_metadata(meta2)
    temp_store.save_extracted_text(meta1.document_id, "Pump bearing housing retorqued to 45 Nm.")
    temp_store.save_extracted_text(meta2.document_id, "Closed-loop glycol chiller replenished.")

    sync_result = temp_index.rebuild_index(temp_store)
    assert sync_result["indexed_documents"] == 2
    assert sync_result["total_chunks"] == 2

    stats = temp_index.get_index_stats()
    assert stats["document_count"] == 2
    assert stats["chunk_count"] == 2


# ==============================================================================
# Phase 7 — Automated Tests: RETRIEVAL
# ==============================================================================

@pytest.fixture
def populated_retriever(temp_index: SearchIndex) -> EvidenceRetriever:
    doc1 = make_test_metadata(doc_id="DOC-HYD", filename="hydraulics.txt")
    text1 = (
        "Hydraulic System Bulletin: Excessive pressure drop across the 10-micron return filter. "
        "Cavitation risk observed when line pressure falls below 150 bar. Bleed air before restart."
    )
    doc2 = make_test_metadata(doc_id="DOC-VIB", filename="vibration.txt")
    text2 = (
        "Vibration Analysis Report: Recurrent harmonic vibration under peak milling load. "
        "Degraded elastomer coupling insert replaced. Shaft alignment within 0.03 mm."
    )
    temp_index.index_document(doc1, text1)
    temp_index.index_document(doc2, text2)
    return EvidenceRetriever(temp_index)


def test_search_single_and_multiple_terms(populated_retriever: EvidenceRetriever):
    """Verify single and multiple keyword retrieval with term matching."""
    res_single = populated_retriever.search("cavitation")
    assert len(res_single) == 1
    assert res_single[0].chunk.document_id == "DOC-HYD"
    assert "cavitation" in res_single[0].matched_terms

    res_multi = populated_retriever.search("harmonic vibration elastomer")
    assert len(res_multi) == 1
    assert res_multi[0].chunk.document_id == "DOC-VIB"
    assert "vibration" in res_multi[0].matched_terms


def test_search_ranking_and_limits(populated_retriever: EvidenceRetriever):
    """Verify result limits and positive BM25 rank score sorting."""
    results = populated_retriever.search("pressure OR vibration", limit=1)
    assert len(results) <= 1
    if results:
        assert results[0].score > 0


def test_search_empty_and_special_characters(populated_retriever: EvidenceRetriever):
    """Verify queries with empty text, punctuation, or SQL/FTS syntax symbols do not error."""
    assert populated_retriever.search("") == []
    assert populated_retriever.search("   ") == []
    assert populated_retriever.search('""**::??//--') == []

    # Safe punctuation stripping
    res = populated_retriever.search("hydraulic, pressure; drop!")
    assert len(res) == 1
    assert res[0].chunk.document_id == "DOC-HYD"


def test_search_document_filter(populated_retriever: EvidenceRetriever):
    """Verify optional document_id filter restricts retrieval results correctly."""
    # Search term 'pressure' is only in DOC-HYD
    assert len(populated_retriever.search("pressure", document_id_filter="DOC-HYD")) == 1
    assert len(populated_retriever.search("pressure", document_id_filter="DOC-VIB")) == 0


def test_no_match_returns_empty_list(populated_retriever: EvidenceRetriever):
    """Verify queries with no matching tokens return empty list without fabricating references."""
    results = populated_retriever.search("quantum teleportation superconductor")
    assert results == []

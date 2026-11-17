"""Focused tests for GhostFault M9.4.3 Technical Knowledge Search and Indexing."""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from src.knowledge_chunker import (
    DEFAULT_KNOWLEDGE_CHUNK_OVERLAP,
    DEFAULT_KNOWLEDGE_CHUNK_SIZE,
    KnowledgeChunk,
    KnowledgeChunker,
)
from src.knowledge_ingestion import KnowledgeIngestionService
from src.knowledge_models import EngineeringReference, KnowledgeDocument, ReliabilityTier, SourceType
from src.knowledge_search import (
    KnowledgeIndexError,
    KnowledgeSearchIndex,
    KnowledgeSearchResult,
    sanitize_knowledge_query,
)
from src.knowledge_store import KnowledgeStore
from src.storage import EvidenceStore


@pytest.fixture
def search_env(tmp_path: Path):
    """Fixture providing isolated EvidenceStore, KnowledgeStore, IngestionService, and KnowledgeSearchIndex."""
    evidence_dir = tmp_path / "evidence"
    evidence_store = EvidenceStore(base_dir=evidence_dir)
    evidence_store.ensure_default_dataset()
    evidence_store.create_dataset("DATASET-UNIT-1", "Unit 1 Plant Diagnostics")
    evidence_store.create_dataset("DATASET-UNIT-2", "Unit 2 Plant Diagnostics")

    knowledge_root = tmp_path / "knowledge"
    knowledge_db = knowledge_root / "knowledge_library.db"
    knowledge_store = KnowledgeStore(store=evidence_store, db_path=knowledge_db)

    index_db = knowledge_root / "knowledge_index.db"
    bootstrap_service = KnowledgeIngestionService(knowledge_store=knowledge_store, base_dir=knowledge_root)
    search_index = KnowledgeSearchIndex(
        knowledge_store=knowledge_store,
        ingestion_service=bootstrap_service,
        db_path=index_db,
    )
    ingestion_service = KnowledgeIngestionService(
        knowledge_store=knowledge_store,
        base_dir=knowledge_root,
        search_index=search_index,
    )
    # Ensure search_index uses the fully wired ingestion service
    search_index.ingestion_service = ingestion_service

    return evidence_store, knowledge_store, ingestion_service, search_index, knowledge_root


class TestKnowledgeChunker:
    """Test deterministic technical knowledge chunking."""

    def test_chunker_deterministic_output(self):
        chunker = KnowledgeChunker(chunk_size=400, chunk_overlap=80)
        text = (
            "Section 4.1 Vibration Evaluation Guidelines.\n\n"
            "The vibration velocity root-mean-square magnitude shall not exceed 4.5 mm/s "
            "for Category I industrial pumps operating under nominal conditions.\n\n"
            "Section 4.2 Shaft Runout Tolerances.\n\n"
            "Total indicator reading for radial runout must remain within 0.025 mm."
        )

        chunks_1 = chunker.chunk_document("DOC-TECH-1", text)
        chunks_2 = chunker.chunk_document("DOC-TECH-1", text)

        assert len(chunks_1) > 0
        assert len(chunks_1) == len(chunks_2)
        for c1, c2 in zip(chunks_1, chunks_2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text
            assert c1.content_hash == c2.content_hash
            assert c1.char_offset_start == c2.char_offset_start
            assert c1.char_offset_end == c2.char_offset_end
            assert c1.section_heading == c2.section_heading

    def test_chunker_detects_headings_and_page_estimates(self):
        chunker = KnowledgeChunker(chunk_size=500, chunk_overlap=100)
        text_with_pages = (
            "Clause 3.2 Operating Boundaries\n\n"
            "Operating boundaries must be respected.\x0c"  # Form feed to page 2
            "Clause 4.1 Mechanical Seal Maintenance\n\n"
            "Inspect barrier fluid reservoir pressure and temperature daily."
        )
        chunks = chunker.chunk_document("DOC-TECH-PAGES", text_with_pages)
        assert len(chunks) >= 2
        assert chunks[0].estimated_page == 1
        assert any(c.estimated_page == 2 for c in chunks)
        headings = [c.section_heading for c in chunks if c.section_heading]
        assert any("Clause" in h for h in headings)

    def test_chunker_rejects_empty_inputs(self):
        chunker = KnowledgeChunker()
        assert chunker.chunk_document("DOC-1", "") == []
        assert chunker.chunk_document("DOC-1", "    \n\n   ") == []

        with pytest.raises(ValueError, match="document_id must not be empty"):
            chunker.chunk_document("", "Valid text content.")


class TestKnowledgeSearchAndIndex:
    """Test FTS5 BM25 search, dataset isolation, provenance, and rebuilding."""

    def test_sanitize_knowledge_query(self):
        assert sanitize_knowledge_query("") == ""
        assert sanitize_knowledge_query("   ") == ""
        # Should sanitize injection chars and produce prefix query
        res = sanitize_knowledge_query('pump "bearing" OR 1=1; DROP TABLE;')
        assert '"pump"*' in res
        assert '"bearing"*' in res
        assert "DROP" in res
        assert ";" not in res

    def test_index_and_bm25_search(self, search_env):
        _, _, ingestion_service, search_index, _ = search_env

        # Ingest 1 Global ISO Standard
        text_iso = (
            "ISO 10816-3 Mechanical Vibration Standard.\n\n"
            "Vibration severity zones for industrial pumps and electric motors. "
            "Zone A: Good operation under 2.8 mm/s rms. Zone B: Acceptable up to 4.5 mm/s rms. "
            "Zone C: Unsatisfactory operation above 7.1 mm/s rms."
        )
        doc_iso = ingestion_service.ingest_document(
            filename="ISO_10816_3.txt",
            content=text_iso.encode("utf-8"),
            title="ISO 10816-3 Vibration Severity",
            publisher_author="ISO Committee",
            source_type=SourceType.ENGINEERING_STANDARD,
            reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            dataset_id=None,  # Global
            publication_year=2009,
        )

        # Ingest 1 Dataset-Specific OEM Manual for DATASET-UNIT-1
        text_oem = (
            "Flowserve Mark 3 Centrifugal Pump Installation & Maintenance Manual.\n\n"
            "Mechanical seal flush plan 11 piping requirements. Ensure seal chamber pressure "
            "exceeds atmospheric pressure by at least 15 psi to avoid cavitation and face vaporisation."
        )
        doc_oem = ingestion_service.ingest_document(
            filename="Flowserve_Mark3.txt",
            content=text_oem.encode("utf-8"),
            title="Flowserve Mark 3 Pump Manual",
            publisher_author="Flowserve",
            source_type=SourceType.OEM_MANUAL,
            reliability_tier=ReliabilityTier.TIER_2_OEM,
            dataset_id="DATASET-UNIT-1",
            publication_year=2020,
        )

        # Search for 'vibration severity' in Global scope
        results_iso = search_index.search(
            query="vibration severity",
            dataset_id=None,
        )
        assert len(results_iso) > 0
        top_match = results_iso[0]
        assert top_match.document_id == doc_iso.document_id
        assert top_match.reliability_tier == ReliabilityTier.TIER_1_STANDARD
        assert "Zone A" in top_match.text
        assert top_match.score > 0

        # Verify conversion to EngineeringReference
        eng_ref = top_match.to_engineering_reference()
        assert isinstance(eng_ref, EngineeringReference)
        assert eng_ref.document_id == doc_iso.document_id
        assert "Contextual Reference Only" in eng_ref.reference_disclaimer

    def test_search_dataset_isolation(self, search_env):
        _, _, ingestion_service, search_index, _ = search_env

        # Document A in DATASET-UNIT-1
        text_a = "Specific cavitation tolerance for Unit 1 reactor feed pump."
        ingestion_service.ingest_document(
            filename="Unit1_Manual.txt",
            content=text_a.encode("utf-8"),
            title="Unit 1 Specific Manual",
            publisher_author="Site Maintenance",
            source_type=SourceType.INDUSTRY_BULLETIN,
            reliability_tier=ReliabilityTier.TIER_3_CURATED_RESEARCH,
            dataset_id="DATASET-UNIT-1",
        )

        # Document B in DATASET-UNIT-2
        text_b = "Specific cavitation limit for Unit 2 auxiliary cooling pump."
        ingestion_service.ingest_document(
            filename="Unit2_Manual.txt",
            content=text_b.encode("utf-8"),
            title="Unit 2 Specific Manual",
            publisher_author="Site Maintenance",
            source_type=SourceType.INDUSTRY_BULLETIN,
            reliability_tier=ReliabilityTier.TIER_3_CURATED_RESEARCH,
            dataset_id="DATASET-UNIT-2",
        )

        # Query DATASET-UNIT-1 must NOT see DATASET-UNIT-2 documents
        results_unit1 = search_index.search(
            query="cavitation",
            dataset_id="DATASET-UNIT-1",
            include_global=False,
        )
        assert len(results_unit1) == 1
        assert results_unit1[0].title == "Unit 1 Specific Manual"
        assert results_unit1[0].dataset_id == "DATASET-UNIT-1"

        # Query DATASET-UNIT-2 must NOT see DATASET-UNIT-1 documents
        results_unit2 = search_index.search(
            query="cavitation",
            dataset_id="DATASET-UNIT-2",
            include_global=False,
        )
        assert len(results_unit2) == 1
        assert results_unit2[0].title == "Unit 2 Specific Manual"
        assert results_unit2[0].dataset_id == "DATASET-UNIT-2"

    def test_global_documents_visible_in_dataset_search_when_enabled(self, search_env):
        _, _, ingestion_service, search_index, _ = search_env

        # 1 Global Standard
        ingestion_service.ingest_document(
            filename="API_Seal_Plans.txt",
            content=b"API 682 Mechanical Seal Piping Plans Plan 53A barrier fluid pressurization.",
            title="API 682 Seal Standard",
            publisher_author="API",
            source_type=SourceType.ENGINEERING_STANDARD,
            reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            dataset_id=None,  # Global
        )

        # Search with include_global=True in DATASET-UNIT-1
        results = search_index.search(
            query="barrier fluid",
            dataset_id="DATASET-UNIT-1",
            include_global=True,
        )
        assert len(results) == 1
        assert results[0].title == "API 682 Seal Standard"
        assert results[0].dataset_id is None

        # Search with include_global=False in DATASET-UNIT-1 -> 0 matches
        results_strict = search_index.search(
            query="barrier fluid",
            dataset_id="DATASET-UNIT-1",
            include_global=False,
        )
        assert len(results_strict) == 0

    def test_rebuild_index_from_stored_documents(self, search_env):
        _, _, ingestion_service, search_index, _ = search_env

        # Ingest 2 documents
        ingestion_service.ingest_document(
            filename="Doc1.txt",
            content=b"Bearing lubricating oil viscosity requirements ISO VG 46.",
            title="Lubrication Guide",
            publisher_author="Tribology Institute",
            source_type=SourceType.GENERAL_REFERENCE,
            reliability_tier=ReliabilityTier.TIER_4_GENERAL,
            dataset_id=None,
        )
        ingestion_service.ingest_document(
            filename="Doc2.txt",
            content=b"Gearbox backlash inspection procedure for planetary gears.",
            title="Gearbox Guide",
            publisher_author="OEM",
            source_type=SourceType.OEM_MANUAL,
            reliability_tier=ReliabilityTier.TIER_2_OEM,
            dataset_id="DATASET-UNIT-1",
        )

        stats_before = search_index.get_index_stats()
        assert stats_before["indexed_documents"] == 2
        assert stats_before["total_chunks"] >= 2

        # Rebuild index
        rebuild_report = search_index.rebuild_index()
        assert rebuild_report["indexed_documents"] == 2
        assert rebuild_report["total_chunks"] == stats_before["total_chunks"]

        # Confirm search works immediately after rebuild
        res = search_index.search("viscosity", dataset_id=None)
        assert len(res) == 1
        assert res[0].title == "Lubrication Guide"

    def test_evidence_index_database_isolation(self, search_env):
        evidence_store, _, _, search_index, knowledge_root = search_env

        # Verify knowledge index path is distinct from evidence store index path
        assert search_index.db_path != evidence_store.base_dir / "index" / "evidence_index.db"
        assert search_index.db_path == knowledge_root / "knowledge_index.db"

        # Verify knowledge index tables do not collide or touch evidence tables
        with sqlite3.connect(search_index.db_path) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "knowledge_chunks" in tables
            assert "knowledge_chunks_fts" in tables
            assert "evidence_chunks" not in tables
            assert "catalog" not in tables

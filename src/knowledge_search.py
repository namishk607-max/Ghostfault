"""Separate SQLite FTS5 search index and retriever for engineering technical knowledge.

Maintains complete isolation from the incident evidence index (`evidence_index.db`).
Provides BM25 ranking, scoped retrieval (Global vs Dataset), provenance reporting,
and automated index synchronization/rebuilding from ingested documents.
"""

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.knowledge_chunker import KnowledgeChunk, KnowledgeChunker
from src.knowledge_ingestion import KnowledgeIngestionService
from src.knowledge_models import EngineeringReference, KnowledgeDocument, ReliabilityTier, SourceType
from src.knowledge_store import KnowledgeStore, KnowledgeStoreError


class KnowledgeIndexError(RuntimeError):
    """Raised when knowledge indexing or search operations fail."""


@dataclass(frozen=True)
class KnowledgeSearchResult:
    """A ranked search match from the engineering knowledge base."""

    document_id: str
    chunk_id: str
    title: str
    publisher_author: str
    source_type: SourceType
    reliability_tier: ReliabilityTier
    text: str
    score: float
    dataset_id: Optional[str] = None
    estimated_page: Optional[int] = None
    section_heading: Optional[str] = None
    publication_year: Optional[int] = None
    equipment_category: Optional[str] = None
    equipment_subtype: Optional[str] = None
    manufacturer: Optional[str] = None
    sha256_hash: Optional[str] = None
    reference_disclaimer: str = (
        "Contextual Reference Only: External engineering knowledge, standards, "
        "and manuals provide technical baseline context and do not constitute "
        "verified factory incident evidence or root-cause determinations."
    )

    def to_engineering_reference(self) -> EngineeringReference:
        """Convert search match into a formal EngineeringReference."""
        citation_parts = [self.title]
        if self.section_heading:
            citation_parts.append(self.section_heading)
        if self.estimated_page:
            citation_parts.append(f"p. {self.estimated_page}")
        citation_str = ", ".join(citation_parts)

        page_str = str(self.estimated_page) if self.estimated_page else None
        return EngineeringReference(
            document_id=self.document_id,
            title=self.title,
            citation=citation_str,
            source_type=self.source_type,
            source_reliability=self.reliability_tier,
            retrieval_relevance=round(self.score, 4),
            page_reference=page_str,
            section_reference=self.section_heading,
            dataset_id=self.dataset_id,
            reference_disclaimer=self.reference_disclaimer,
        )


def sanitize_knowledge_query(query: str) -> str:
    """Sanitize user query for SQLite FTS5 prefix matching without injection."""
    if not query:
        return ""
    tokens = re.findall(r"[a-zA-Z0-9_\-]+", query)
    cleaned_tokens = [t.strip("-").strip("_") for t in tokens if len(t.strip("-").strip("_")) > 0]
    if not cleaned_tokens:
        return ""
    # Safe prefix matching conjunction: "term1"* AND "term2"*
    return " AND ".join(f'"{t}"*' for t in cleaned_tokens)


class KnowledgeSearchIndex:
    """Independent SQLite FTS5 search index for engineering technical documents.

    Stored separately in `data/knowledge/knowledge_index.db`.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        ingestion_service: KnowledgeIngestionService,
        db_path: Optional[Path] = None,
        chunker: Optional[KnowledgeChunker] = None,
    ) -> None:
        if not isinstance(knowledge_store, KnowledgeStore):
            raise TypeError("KnowledgeSearchIndex requires a KnowledgeStore.")
        if not isinstance(ingestion_service, KnowledgeIngestionService):
            raise TypeError("KnowledgeSearchIndex requires a KnowledgeIngestionService.")

        self.knowledge_store = knowledge_store
        self.ingestion_service = ingestion_service
        self.chunker = chunker if chunker is not None else KnowledgeChunker()

        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.db_path = project_root / "data" / "knowledge" / "knowledge_index.db"
        else:
            self.db_path = Path(db_path).resolve()

        self._ensure_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _ensure_db(self) -> None:
        """Create versioned FTS5 tables and metadata store."""
        with closing(self._get_connection()) as conn:
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if user_version == 0:
                conn.execute("BEGIN IMMEDIATE")
                # Table for chunk metadata and storage
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        dataset_id TEXT,
                        chunk_index INTEGER NOT NULL,
                        char_offset_start INTEGER NOT NULL,
                        char_offset_end INTEGER NOT NULL,
                        char_count INTEGER NOT NULL,
                        content_hash TEXT NOT NULL,
                        estimated_page INTEGER,
                        section_heading TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kchunks_doc
                    ON knowledge_chunks(document_id, dataset_id)
                    """
                )
                # FTS5 Virtual Table for full-text search
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        document_id UNINDEXED,
                        dataset_id UNINDEXED,
                        text,
                        tokenize = 'unicode61 remove_diacritics 2'
                    )
                    """
                )
                conn.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
                conn.commit()
            elif user_version > self.SCHEMA_VERSION:
                raise KnowledgeIndexError(
                    f"Knowledge index schema version {user_version} is newer than supported."
                )

    def is_document_indexed(self, document_id: str) -> bool:
        """Check if document has chunks indexed."""
        with closing(self._get_connection()) as conn:
            row = conn.execute(
                "SELECT 1 FROM knowledge_chunks WHERE document_id = ? LIMIT 1",
                (document_id,),
            ).fetchone()
            return row is not None

    def remove_document(self, document_id: str) -> None:
        """Remove all chunks and FTS entries for a document."""
        with closing(self._get_connection()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM knowledge_chunks_fts WHERE document_id = ?",
                (document_id,),
            )
            conn.execute(
                "DELETE FROM knowledge_chunks WHERE document_id = ?",
                (document_id,),
            )
            conn.commit()

    def index_document(self, document: KnowledgeDocument) -> int:
        """Extract text, chunk, and index a single KnowledgeDocument."""
        raw_text = self.ingestion_service.get_extracted_text(document.document_id)
        if not raw_text or not raw_text.strip():
            return 0

        chunks = self.chunker.chunk_document(document.document_id, raw_text)
        if not chunks:
            return 0

        with closing(self._get_connection()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            # Remove any stale chunks first
            conn.execute("DELETE FROM knowledge_chunks_fts WHERE document_id = ?", (document.document_id,))
            conn.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (document.document_id,))

            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()

            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO knowledge_chunks (
                        chunk_id, document_id, dataset_id, chunk_index,
                        char_offset_start, char_offset_end, char_count,
                        content_hash, estimated_page, section_heading,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        document.document_id,
                        document.dataset_id,
                        chunk.chunk_index,
                        chunk.char_offset_start,
                        chunk.char_offset_end,
                        chunk.char_count,
                        chunk.content_hash,
                        chunk.estimated_page,
                        chunk.section_heading,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO knowledge_chunks_fts (
                        chunk_id, document_id, dataset_id, text
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        document.document_id,
                        document.dataset_id,
                        chunk.text,
                    ),
                )
            conn.commit()

        return len(chunks)

    def rebuild_index(self) -> Dict[str, Any]:
        """Re-index all documents present in the KnowledgeStore."""
        # Clean existing tables
        with closing(self._get_connection()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM knowledge_chunks_fts")
            conn.execute("DELETE FROM knowledge_chunks")
            conn.commit()

        docs = self.knowledge_store.list_documents(dataset_id=None)  # Global docs
        # Also query all dataset-scoped documents
        with closing(self.knowledge_store._connect()) as store_conn:
            dataset_rows = store_conn.execute(
                "SELECT DISTINCT dataset_id FROM knowledge_documents WHERE dataset_id IS NOT NULL"
            ).fetchall()

        all_docs = list(docs)
        for row in dataset_rows:
            ds_id = row["dataset_id"]
            all_docs.extend(self.knowledge_store.list_documents(dataset_id=ds_id, include_global=False))

        indexed_docs = 0
        total_chunks = 0
        for doc in all_docs:
            count = self.index_document(doc)
            if count > 0:
                indexed_docs += 1
                total_chunks += count

        return {
            "indexed_documents": indexed_docs,
            "total_chunks": total_chunks,
        }

    def search(
        self,
        query: str,
        dataset_id: Optional[str] = None,
        include_global: bool = True,
        source_type: Optional[SourceType] = None,
        reliability_tier: Optional[ReliabilityTier] = None,
        limit: int = 10,
    ) -> List[KnowledgeSearchResult]:
        """Search engineering technical knowledge using BM25 ranking.

        Guarantees strict dataset scope:
        - If dataset_id is None: returns only Global documents.
        - If dataset_id is given: returns matches for dataset_id (and global if include_global=True).
        Never returns documents from other datasets.
        """
        sanitized = sanitize_knowledge_query(query)
        if not sanitized:
            return []

        # Scope restrictions
        scope_conditions = []
        scope_params: List[Any] = []

        if dataset_id is None or dataset_id == "":
            scope_conditions.append("c.dataset_id IS NULL")
        else:
            if include_global:
                scope_conditions.append("(c.dataset_id = ? OR c.dataset_id IS NULL)")
                scope_params.append(dataset_id)
            else:
                scope_conditions.append("c.dataset_id = ?")
                scope_params.append(dataset_id)

        scope_sql = " AND " + " AND ".join(scope_conditions) if scope_conditions else ""

        # We join FTS matches with knowledge_chunks and query metadata from knowledge_store
        sql = f"""
            SELECT
                fts.chunk_id,
                fts.document_id,
                fts.text,
                bm25(knowledge_chunks_fts) AS bm25_rank,
                c.dataset_id,
                c.estimated_page,
                c.section_heading,
                c.char_offset_start,
                c.char_offset_end
            FROM knowledge_chunks_fts AS fts
            JOIN knowledge_chunks AS c ON fts.chunk_id = c.chunk_id
            WHERE knowledge_chunks_fts MATCH ?
            {scope_sql}
            ORDER BY bm25_rank ASC
            LIMIT ?
        """
        params = [sanitized] + scope_params + [limit * 2]  # fetch extra for post-filtering if needed

        with closing(self._get_connection()) as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        # Fetch document metadata for matched documents
        doc_ids = list({row["document_id"] for row in rows})
        doc_meta_map: Dict[str, KnowledgeDocument] = {}
        for did in doc_ids:
            doc = self.knowledge_store.get_document(did)
            if doc:
                doc_meta_map[did] = doc

        results: List[KnowledgeSearchResult] = []
        for row in rows:
            did = row["document_id"]
            meta = doc_meta_map.get(did)
            if not meta:
                continue

            # Optional filter by source_type or reliability_tier
            if source_type is not None and meta.source_type != source_type:
                continue
            if reliability_tier is not None and meta.reliability_tier != reliability_tier:
                continue

            # Convert negative BM25 score to positive score for clarity
            raw_bm25 = float(row["bm25_rank"])
            score = abs(raw_bm25)

            results.append(
                KnowledgeSearchResult(
                    document_id=did,
                    chunk_id=row["chunk_id"],
                    title=meta.title,
                    publisher_author=meta.publisher_author,
                    source_type=meta.source_type,
                    reliability_tier=meta.reliability_tier,
                    text=row["text"],
                    score=score,
                    dataset_id=row["dataset_id"],
                    estimated_page=row["estimated_page"],
                    section_heading=row["section_heading"],
                    publication_year=meta.publication_year,
                    equipment_category=meta.equipment_category,
                    equipment_subtype=meta.equipment_subtype,
                    manufacturer=meta.manufacturer,
                    sha256_hash=meta.sha256_hash,
                )
            )
            if len(results) >= limit:
                break

        return results

    def get_index_stats(self) -> Dict[str, Any]:
        """Return counts of indexed documents and chunks."""
        with closing(self._get_connection()) as conn:
            total_chunks = conn.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()[0]
            indexed_docs = conn.execute("SELECT COUNT(DISTINCT document_id) FROM knowledge_chunks").fetchone()[0]
        return {
            "indexed_documents": indexed_docs,
            "total_chunks": total_chunks,
        }

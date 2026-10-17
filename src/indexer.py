"""Persistent local full-text search index for GhostFault evidence chunks.

Backed by SQLite FTS5 for deterministic, explainable keyword retrieval with BM25 ranking.
Provides atomic index updates, document change detection, and full rebuild capabilities.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import asdict

from src.models import EvidenceChunk, DocumentMetadata, DocumentStatus, current_iso_timestamp
from src.chunker import TextChunker, compute_content_hash
from src.storage import EvidenceStore


class SearchIndex:
    """
    Manages a persistent SQLite FTS5 index stored in `data/evidence/index/evidence_index.db`.
    """

    def __init__(self, db_path: Optional[Path] = None, chunker: Optional[TextChunker] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent
            self.db_path = project_root / "data" / "evidence" / "index" / "evidence_index.db"
        else:
            self.db_path = Path(db_path).resolve()

        self.chunker = chunker if chunker is not None else TextChunker()
        self._ensure_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Establish SQLite connection with foreign keys and row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        """Create index directory and initialize SQLite FTS5 tables and metadata tracking."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Metadata tracking table: tracks indexed documents and content hash
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_documents (
                    document_id TEXT PRIMARY KEY,
                    source_filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                )
                """
            )

            # Relational store for evidence chunks
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    page_number INTEGER,
                    start_offset INTEGER,
                    end_offset INTEGER,
                    created_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES indexed_documents(document_id) ON DELETE CASCADE
                )
                """
            )

            # SQLite FTS5 Virtual Table for full-text search
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    source_filename,
                    text,
                    tokenize = 'porter unicode61'
                )
                """
            )
            conn.commit()

    def get_index_stats(self) -> Dict[str, Any]:
        """Return counts of indexed documents, chunks, and last indexing timestamp."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), MAX(indexed_at) FROM indexed_documents")
            doc_row = cursor.fetchone()
            doc_count = doc_row[0] if doc_row else 0
            last_indexed = doc_row[1] if doc_row else None

            cursor.execute("SELECT COUNT(*) FROM evidence_chunks")
            chunk_row = cursor.fetchone()
            chunk_count = chunk_row[0] if chunk_row else 0

            return {
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "last_indexed_at": last_indexed,
            }

    def is_document_indexed(self, document_id: str, current_content_hash: Optional[str] = None) -> bool:
        """Check if a document is already indexed, and whether its hash matches."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content_hash FROM indexed_documents WHERE document_id = ?",
                (document_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False
            if current_content_hash is not None:
                return row["content_hash"] == current_content_hash
            return True

    def remove_document(self, document_id: str) -> None:
        """Remove all chunks and FTS records for a document within a transaction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM evidence_chunks WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM indexed_documents WHERE document_id = ?", (document_id,))
            conn.commit()

    def index_document(
        self,
        metadata: DocumentMetadata,
        extracted_text: str,
        page_number: Optional[int] = None,
    ) -> int:
        """
        Index or re-index a single document:
        - If unchanged, skips re-indexing.
        - If updated or new, replaces existing chunks atomically.
        Returns number of chunks indexed.
        """
        if not extracted_text or not extracted_text.strip():
            # If document has no text, remove from index if previously indexed
            self.remove_document(metadata.document_id)
            return 0

        doc_hash = compute_content_hash(extracted_text)

        # Check if already indexed with identical content
        if self.is_document_indexed(metadata.document_id, doc_hash):
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT chunk_count FROM indexed_documents WHERE document_id = ?",
                    (metadata.document_id,),
                )
                row = cursor.fetchone()
                return row["chunk_count"] if row else 0

        # Generate fresh chunks
        chunks = self.chunker.chunk_document(metadata, extracted_text, page_number)
        if not chunks:
            self.remove_document(metadata.document_id)
            return 0

        timestamp = current_iso_timestamp()

        # Atomic transaction: remove old entries and insert new chunks
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Clean any previous chunks for this document
            cursor.execute("DELETE FROM chunks_fts WHERE document_id = ?", (metadata.document_id,))
            cursor.execute("DELETE FROM evidence_chunks WHERE document_id = ?", (metadata.document_id,))
            cursor.execute("DELETE FROM indexed_documents WHERE document_id = ?", (metadata.document_id,))

            # Insert tracking record
            cursor.execute(
                """
                INSERT INTO indexed_documents (document_id, source_filename, content_hash, chunk_count, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (metadata.document_id, metadata.original_filename, doc_hash, len(chunks), timestamp),
            )

            # Insert chunks into relational table and FTS5 table
            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT INTO evidence_chunks (
                        chunk_id, document_id, chunk_index, text, source_filename,
                        file_type, page_number, start_offset, end_offset, created_at, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.source_filename,
                        chunk.file_type,
                        chunk.page_number,
                        chunk.start_offset,
                        chunk.end_offset,
                        chunk.created_at,
                        chunk.content_hash,
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, document_id, source_filename, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chunk.chunk_id, chunk.document_id, chunk.source_filename, chunk.text),
                )

            conn.commit()

        return len(chunks)

    def sync_with_store(self, store: EvidenceStore) -> Dict[str, Any]:
        """
        Synchronize the index with the catalog in EvidenceStore.
        Indexes newly processed documents, updates modified text, and removes deleted/failed docs.
        Returns synchronization metrics.
        """
        catalog = store.read_catalog()
        processed_docs = [d for d in catalog if d.ingestion_status == DocumentStatus.PROCESSED]
        catalog_doc_ids = {d.document_id for d in processed_docs}

        # Check existing indexed document IDs
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT document_id FROM indexed_documents")
            existing_indexed_ids = {row["document_id"] for row in cursor.fetchall()}

        # 1. Remove documents that are no longer PROCESSED in the catalog
        stale_ids = existing_indexed_ids - catalog_doc_ids
        for stale_id in stale_ids:
            self.remove_document(stale_id)

        # 2. Index new or updated documents
        indexed_count = 0
        total_chunks = 0
        errors = []

        for meta in processed_docs:
            extracted_text = store.get_extracted_text(meta)
            if not extracted_text:
                self.remove_document(meta.document_id)
                continue

            try:
                chunks_added = self.index_document(meta, extracted_text)
                if chunks_added > 0:
                    indexed_count += 1
                    total_chunks += chunks_added
            except Exception as exc:
                errors.append(f"Failed to index {meta.document_id} ({meta.original_filename}): {exc}")

        return {
            "indexed_documents": indexed_count,
            "total_chunks": total_chunks,
            "removed_stale": len(stale_ids),
            "errors": errors,
        }

    def rebuild_index(self, store: EvidenceStore) -> Dict[str, Any]:
        """
        Completely wipe and rebuild the FTS5 index from the catalog and extracted text.
        Atomic and safe: does not touch original uploaded evidence files.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks_fts")
            cursor.execute("DELETE FROM evidence_chunks")
            cursor.execute("DELETE FROM indexed_documents")
            conn.commit()

        return self.sync_with_store(store)

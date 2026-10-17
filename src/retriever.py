"""Deterministic evidence retrieval service for GhostFault.

Provides safe query normalization, SQLite FTS5 BM25 search ranking,
snippet extraction, and source provenance reporting without external AI dependencies.
"""

import re
from typing import List, Optional, Set
from src.models import EvidenceChunk, SearchResult
from src.indexer import SearchIndex


def sanitize_fts5_query(query: str) -> str:
    """
    Sanitize and tokenize raw user query into safe FTS5 query syntax.
    Removes syntax control characters (quotes, stars, colons, parentheses),
    extracts alphanumeric terms, and constructs a prefix-matching conjunction ('term1* AND term2*').
    """
    if not query:
        return ""

    # Extract words (alphanumeric sequences, including hyphens/underscores inside words)
    tokens = re.findall(r"[a-zA-Z0-9_\-]+", query)
    cleaned_tokens = [t.strip("-").strip("_") for t in tokens if len(t.strip("-").strip("_")) > 0]

    if not cleaned_tokens:
        return ""

    # Quote each term to prevent FTS5 operator injection and append prefix wildcard
    escaped_terms = [f'"{token}"*' for token in cleaned_tokens]
    return " AND ".join(escaped_terms)


class EvidenceRetriever:
    """
    Deterministic evidence retrieval engine using SQLite FTS5 BM25 scoring.
    """

    def __init__(self, index: SearchIndex):
        self.index = index

    def search(
        self,
        query: str,
        limit: int = 10,
        document_id_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Execute deterministic full-text search across indexed evidence chunks.

        Args:
            query: User search string.
            limit: Maximum results to return (bounded between 1 and 100).
            document_id_filter: Optional filter to restrict search to a single document.

        Returns:
            List of ranked SearchResult instances with source provenance and snippet highlights.
        """
        bounded_limit = max(1, min(limit, 100))
        cleaned_query = query.strip() if query else ""

        if not cleaned_query:
            return []

        fts_query = sanitize_fts5_query(cleaned_query)
        if not fts_query:
            return []

        # Find query terms for highlighting
        raw_terms = [t.lower() for t in re.findall(r"[a-zA-Z0-9_\-]+", cleaned_query) if t]
        unique_terms: Set[str] = set(raw_terms)

        results: List[SearchResult] = []

        with self.index._get_connection() as conn:
            cursor = conn.cursor()

            # Query FTS5 table with BM25 ranking (bm25(chunks_fts) returns negative float where smaller/more negative is better)
            # In SQLite FTS5, bm25() score: lower is better, so we negate it for positive ranking display: score = -bm25()
            sql = """
                SELECT 
                    c.chunk_id,
                    c.document_id,
                    c.chunk_index,
                    c.text,
                    c.source_filename,
                    c.file_type,
                    c.page_number,
                    c.start_offset,
                    c.end_offset,
                    c.created_at,
                    c.content_hash,
                    -bm25(chunks_fts) AS rank_score,
                    highlight(chunks_fts, 3, '[[HL]]', '[[/HL]]') AS highlighted_text
                FROM chunks_fts
                JOIN evidence_chunks c ON chunks_fts.chunk_id = c.chunk_id
                WHERE chunks_fts MATCH ?
            """
            params: list = [fts_query]

            if document_id_filter and document_id_filter != "All Documents":
                sql += " AND c.document_id = ?"
                params.append(document_id_filter)

            sql += " ORDER BY rank_score DESC LIMIT ?"
            params.append(bounded_limit)

            try:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
            except Exception:
                # If MATCH syntax fails on unusual character combinations, fallback safely to empty
                return []

            for row in rows:
                chunk = EvidenceChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    source_filename=row["source_filename"],
                    file_type=row["file_type"],
                    page_number=row["page_number"],
                    start_offset=row["start_offset"],
                    end_offset=row["end_offset"],
                    created_at=row["created_at"],
                    content_hash=row["content_hash"],
                )

                # Format highlight snippet
                highlighted_text = row["highlighted_text"] or row["text"]
                # Convert markers to readable markdown bold
                clean_snippet = highlighted_text.replace("[[HL]]", "**").replace("[[/HL]]", "**")

                # Identify which query terms actually appeared
                chunk_lower = chunk.text.lower()
                matched_in_chunk = [term for term in raw_terms if term in chunk_lower]

                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=round(float(row["rank_score"]), 3),
                        matched_snippet=clean_snippet,
                        matched_terms=list(dict.fromkeys(matched_in_chunk)),
                    )
                )

        return results

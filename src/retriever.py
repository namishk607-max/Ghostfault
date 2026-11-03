"""Deterministic evidence retrieval service for GhostFault.

Provides safe query normalization, SQLite FTS5 BM25 search ranking,
snippet extraction, and source provenance reporting without external AI dependencies.
"""

import re
from typing import List, Optional, Set, Iterable
from src.models import DatasetOwnershipError, EvidenceChunk, EvidenceReference, SearchResult
from src.indexer import SearchIndex
from src.storage import EvidenceStore


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

    def __init__(self, index: SearchIndex, store: Optional[EvidenceStore] = None):
        self.index = index
        self.store = store

    def require_dataset_scope(
        self,
        dataset_id: Optional[str],
        valid_dataset_ids: Optional[Iterable[str]] = None,
    ) -> str:
        """Validate the active dataset against the authoritative local registry."""
        if self.store is None:
            raise DatasetOwnershipError(
                "Dataset-scoped retrieval requires an EvidenceStore to validate catalog ownership."
            )
        self.store.require_dataset_id(dataset_id, "Evidence search", "active dataset")
        if valid_dataset_ids is not None and dataset_id not in set(valid_dataset_ids):
            raise DatasetOwnershipError(
                f"Evidence search dataset '{dataset_id}' is not present in the supplied "
                "registered-dataset allow-list."
            )
        return dataset_id

    def validate_evidence_results(
        self,
        results: List[SearchResult],
        dataset_id: Optional[str],
    ) -> List[SearchResult]:
        """Reject supplied evidence unless every chunk exactly matches trusted stored provenance."""
        target_dataset_id = self.require_dataset_scope(dataset_id)
        if not results:
            return []
        if self.store is None:
            raise DatasetOwnershipError("Evidence cannot be validated without the evidence catalog.")

        integrity_issues = self.index.validate_dataset_integrity(
            self.store,
            target_dataset_id,
        )
        if integrity_issues:
            details = " ".join(issue.message for issue in integrity_issues[:5])
            raise DatasetOwnershipError(
                f"Evidence for dataset '{target_dataset_id}' cannot be trusted because "
                f"{len(integrity_issues)} catalog/index integrity issue(s) remain. {details}"
            )

        with self.index._get_connection() as conn:
            for result in results:
                chunk = result.chunk
                row = conn.execute(
                    """
                    SELECT c.*, d.dataset_id AS indexed_dataset_id
                    FROM evidence_chunks c
                    JOIN indexed_documents d ON d.document_id = c.document_id
                    WHERE c.chunk_id = ?
                    """,
                    (chunk.chunk_id,),
                ).fetchone()
                metadata = self.store.get_metadata(chunk.document_id)
                if (
                    row is None
                    or metadata is None
                    or metadata.dataset_id != target_dataset_id
                    or row["indexed_dataset_id"] != target_dataset_id
                    or row["dataset_id"] != target_dataset_id
                    or chunk.dataset_id != target_dataset_id
                    or row["document_id"] != chunk.document_id
                    or row["chunk_index"] != chunk.chunk_index
                    or row["text"] != chunk.text
                    or row["source_filename"] != chunk.source_filename
                    or row["file_type"] != chunk.file_type
                    or row["page_number"] != chunk.page_number
                    or row["start_offset"] != chunk.start_offset
                    or row["end_offset"] != chunk.end_offset
                    or row["content_hash"] != chunk.content_hash
                    or row["created_at"] != chunk.created_at
                    or metadata.original_filename != chunk.source_filename
                    or metadata.file_type != chunk.file_type
                ):
                    raise DatasetOwnershipError(
                        f"Supplied evidence chunk '{chunk.chunk_id}' is untagged, foreign, "
                        "or does not match the catalog and evidence index for active "
                        f"dataset '{target_dataset_id}'."
                    )
        return results

    def validate_evidence_references(
        self,
        references: List[EvidenceReference],
        dataset_id: Optional[str],
    ) -> List[EvidenceReference]:
        """Validate analysis references against canonical chunks and catalog ownership."""
        target_dataset_id = self.require_dataset_scope(dataset_id)
        if not references:
            return []
        if self.store is None:
            raise DatasetOwnershipError("Evidence references cannot be validated without the evidence catalog.")

        integrity_issues = self.index.validate_dataset_integrity(self.store, target_dataset_id)
        if integrity_issues:
            details = " ".join(issue.message for issue in integrity_issues[:5])
            raise DatasetOwnershipError(
                f"Evidence references for dataset '{target_dataset_id}' cannot be trusted "
                f"because {len(integrity_issues)} catalog/index integrity issue(s) remain. {details}"
            )

        with self.index._get_connection() as conn:
            for reference in references:
                row = conn.execute(
                    """
                    SELECT c.dataset_id, c.document_id, c.chunk_index, c.text,
                           c.source_filename, d.dataset_id AS indexed_dataset_id
                    FROM evidence_chunks c
                    JOIN indexed_documents d ON d.document_id = c.document_id
                    WHERE c.chunk_id = ?
                    """,
                    (reference.chunk_id,),
                ).fetchone()
                metadata = self.store.get_metadata(reference.document_id)
                snippet = re.sub(r"[^a-z0-9\s]", " ", reference.snippet.lower())
                snippet = " ".join(snippet.split())
                source_text = re.sub(r"[^a-z0-9\s]", " ", row["text"].lower()) if row else ""
                source_text = " ".join(source_text.split())
                if (
                    row is None
                    or metadata is None
                    or reference.dataset_id != target_dataset_id
                    or row["dataset_id"] != target_dataset_id
                    or row["indexed_dataset_id"] != target_dataset_id
                    or metadata.dataset_id != target_dataset_id
                    or row["document_id"] != reference.document_id
                    or row["chunk_index"] != reference.chunk_index
                    or row["source_filename"] != reference.source_filename
                    or not snippet
                    or snippet not in source_text
                ):
                    raise DatasetOwnershipError(
                        f"Evidence reference '{reference.chunk_id}' is untagged, foreign, "
                        "or does not match canonical stored provenance for active dataset "
                        f"'{target_dataset_id}'."
                    )
        return references

    def search(
        self,
        query: str,
        limit: int = 10,
        document_id_filter: Optional[str] = None,
        dataset_id: Optional[str] = None,
        valid_dataset_ids: Optional[Iterable[str]] = None,
        store: Optional[EvidenceStore] = None,
    ) -> List[SearchResult]:
        """
        Execute deterministic full-text search across indexed evidence chunks.

        Args:
            query: User search string.
            limit: Maximum results to return (bounded between 1 and 100).
            document_id_filter: Optional filter to restrict search to a single document.
            dataset_id: Required registered dataset ID to restrict search.
            valid_dataset_ids: Optional additional allow-list retained for caller compatibility.
            store: Optional evidence store override for catalog-backed ownership validation.

        Returns:
            List of ranked SearchResult instances with source provenance and snippet highlights.
        """
        evidence_store = store or self.store
        if store is not None and store is not self.store:
            return EvidenceRetriever(self.index, store).search(
                query=query,
                limit=limit,
                document_id_filter=document_id_filter,
                dataset_id=dataset_id,
                valid_dataset_ids=valid_dataset_ids,
            )
        target_dataset_id = self.require_dataset_scope(dataset_id, valid_dataset_ids)
        if evidence_store is None:
            raise DatasetOwnershipError(
                "Dataset-scoped retrieval requires an EvidenceStore to validate catalog ownership."
            )

        integrity_issues = self.index.validate_dataset_integrity(
            evidence_store,
            target_dataset_id,
        )
        if integrity_issues:
            details = " ".join(issue.message for issue in integrity_issues[:5])
            raise DatasetOwnershipError(
                f"Evidence retrieval is disabled for dataset '{target_dataset_id}' because "
                f"{len(integrity_issues)} catalog/index integrity issue(s) remain. "
                f"No evidence was returned and no data was changed. Review the catalog and "
                f"index ownership before searching. {details}"
            )

        catalog = evidence_store.read_catalog()
        catalog_by_id = {metadata.document_id: metadata for metadata in catalog}

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
                    c.dataset_id,
                    d.dataset_id AS indexed_dataset_id,
                    -bm25(chunks_fts) AS rank_score,
                    highlight(chunks_fts, 3, '[[HL]]', '[[/HL]]') AS highlighted_text
                FROM chunks_fts
                JOIN evidence_chunks c ON chunks_fts.chunk_id = c.chunk_id
                JOIN indexed_documents d ON d.document_id = c.document_id
                WHERE chunks_fts MATCH ?
                  AND chunks_fts.document_id = c.document_id
                  AND d.dataset_id = ?
                  AND c.dataset_id = ?
            """
            params: list = [fts_query, target_dataset_id, target_dataset_id]

            if document_id_filter and document_id_filter != "All Documents":
                sql += " AND c.document_id = ?"
                params.append(document_id_filter)

            sql += " ORDER BY rank_score DESC LIMIT ?"
            params.append(bounded_limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            for row in rows:
                metadata = catalog_by_id.get(row["document_id"])
                if (
                    metadata is None
                    or metadata.dataset_id != target_dataset_id
                    or row["dataset_id"] != target_dataset_id
                    or row["indexed_dataset_id"] != target_dataset_id
                    or row["source_filename"] != metadata.original_filename
                    or row["file_type"] != metadata.file_type
                ):
                    raise DatasetOwnershipError(
                        f"Evidence chunk '{row['chunk_id']}' does not agree with the "
                        f"registered catalog ownership and provenance for dataset "
                        f"'{target_dataset_id}'. No search results were returned."
                    )

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
                    dataset_id=row["dataset_id"] if "dataset_id" in row.keys() else None,
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

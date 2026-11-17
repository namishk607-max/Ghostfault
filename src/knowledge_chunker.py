"""Deterministic chunking service for GhostFault engineering technical documents.

Splits technical manuals, standards, and research papers into searchable passages,
preserving document ID, character offsets, section headings, and estimated page references.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional


DEFAULT_KNOWLEDGE_CHUNK_SIZE: int = 700
DEFAULT_KNOWLEDGE_CHUNK_OVERLAP: int = 120
MIN_KNOWLEDGE_CHUNK_SIZE: int = 150
MAX_KNOWLEDGE_CHUNK_SIZE: int = 4000


@dataclass(frozen=True)
class KnowledgeChunk:
    """A bounded passage extracted from an engineering knowledge document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    char_offset_start: int
    char_offset_end: int
    char_count: int
    content_hash: str
    estimated_page: Optional[int] = None
    section_heading: Optional[str] = None


class KnowledgeChunker:
    """Deterministic chunker tailored for engineering technical documentation.

    Preserves paragraph and sentence boundaries, detects standard section headings,
    estimates page numbers from form-feed or double-newline page splits,
    and guarantees identical chunks for identical inputs.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_KNOWLEDGE_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_KNOWLEDGE_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size < MIN_KNOWLEDGE_CHUNK_SIZE or chunk_size > MAX_KNOWLEDGE_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size must be between {MIN_KNOWLEDGE_CHUNK_SIZE} and {MAX_KNOWLEDGE_CHUNK_SIZE}, got {chunk_size}"
            )
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap must be >= 0 and < chunk_size ({chunk_size}), got {chunk_overlap}"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _detect_heading(window: str) -> Optional[str]:
        """Detect standard engineering section headings (e.g. '4.2 Vibration Limits', 'Section 3')."""
        match = re.search(
            r"(?:^|\n)\s*(?:(?:Section|Chapter|Clause|Part|Annex)\s+[\dA-Z.]+|[\d]+(?:\.[\d]+)+\.?)\s+([^\n]{3,60})",
            window,
            re.IGNORECASE,
        )
        if match:
            heading = match.group(0).strip()
            return heading if len(heading) <= 80 else heading[:77] + "..."
        return None

    @staticmethod
    def _estimate_page_number(text_up_to_offset: str) -> int:
        """Estimate page number based on form-feed characters (\x0c) or character position (~2500 chars/page)."""
        form_feeds = text_up_to_offset.count("\x0c")
        if form_feeds > 0:
            return form_feeds + 1
        # Fallback heuristic for plain text: ~2500 chars per standard printed technical page
        return (len(text_up_to_offset) // 2500) + 1

    def chunk_document(
        self,
        document_id: str,
        text: str,
    ) -> List[KnowledgeChunk]:
        """Split document text into a deterministic list of KnowledgeChunks."""
        if not document_id or not document_id.strip():
            raise ValueError("document_id must not be empty.")
        if not text or not text.strip():
            return []

        clean_doc_id = document_id.strip()
        text_len = len(text)
        chunks: List[KnowledgeChunk] = []

        start_idx = 0
        chunk_index = 0

        while start_idx < text_len:
            # Skip leading whitespace
            while start_idx < text_len and text[start_idx].isspace():
                start_idx += 1

            if start_idx >= text_len:
                break

            target_end = min(start_idx + self.chunk_size, text_len)

            # Check if there is an explicit page break (\x0c) before target_end
            page_break = text.find("\x0c", start_idx + 1)
            if page_break != -1 and page_break < target_end:
                end_idx = page_break + 1
            elif target_end >= text_len:
                end_idx = text_len
            else:
                # Seek natural break in second half of the chunk window
                min_cut = start_idx + (self.chunk_size // 2)
                window = text[min_cut:target_end]

                # 1. Page / Form Feed Break
                ff_match = [m.start() for m in re.finditer(r"\x0c", window)]
                if ff_match:
                    end_idx = min_cut + ff_match[-1] + 1
                else:
                    # 2. Paragraph break (\n\s*\n)
                    para_match = [m.start() for m in re.finditer(r"\n\s*\n", window)]
                    if para_match:
                        end_idx = min_cut + para_match[-1]
                    else:
                        # 3. Sentence end (. / ! / ?) followed by space, newline, or end
                        sent_match = [m.start() for m in re.finditer(r"[.!?](?:\s|$)", window)]
                        if sent_match:
                            end_idx = min_cut + sent_match[-1] + 1
                        else:
                            # 4. Single newline
                            nl_match = [m.start() for m in re.finditer(r"\n", window)]
                            if nl_match:
                                end_idx = min_cut + nl_match[-1]
                            else:
                                # 5. Word boundary
                                space_match = [m.start() for m in re.finditer(r"\s", window)]
                                if space_match:
                                    end_idx = min_cut + space_match[-1]
                                else:
                                    end_idx = target_end

            chunk_raw = text[start_idx:end_idx]
            chunk_text = chunk_raw.strip()

            if chunk_text:
                # Calculate deterministic content hash
                content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                chunk_id = f"{clean_doc_id}-CHK-{chunk_index:04d}"

                estimated_page = self._estimate_page_number(text[:start_idx])
                heading = self._detect_heading(chunk_raw[:200])

                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        document_id=clean_doc_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        char_offset_start=start_idx,
                        char_offset_end=end_idx,
                        char_count=len(chunk_text),
                        content_hash=content_hash,
                        estimated_page=estimated_page,
                        section_heading=heading,
                    )
                )
                chunk_index += 1

            # Advance with overlap
            if end_idx >= text_len:
                break

            if page_break != -1 and page_break < target_end:
                # Page break cut was made at end_idx = page_break + 1
                next_start = end_idx
            else:
                overlap_step = self.chunk_size - self.chunk_overlap
                next_start = start_idx + overlap_step

            # Monotonic progress safety
            if next_start <= start_idx:
                next_start = start_idx + 1
            start_idx = next_start

        return chunks

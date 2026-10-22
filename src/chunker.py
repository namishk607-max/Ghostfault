"""Isolated text chunking engine for GhostFault evidence documents.

Splits extracted document text into deterministic passages preserving natural
paragraph boundaries, sentence boundaries, word boundaries, and character offsets.
"""

import hashlib
import re
from typing import List, Tuple, Optional
from src.models import EvidenceChunk, DocumentMetadata, current_iso_timestamp

# Defaults for evidence passages
DEFAULT_CHUNK_SIZE: int = 600
DEFAULT_CHUNK_OVERLAP: int = 100
MIN_CHUNK_SIZE: int = 100
MAX_CHUNK_SIZE: int = 4000


def compute_content_hash(text: str) -> str:
    """Return SHA-256 hexadecimal digest of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TextChunker:
    """
    Deterministic chunking service. Splits extracted text into bounded passages
    respecting paragraph and sentence boundaries without external AI dependencies.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if chunk_size < MIN_CHUNK_SIZE or chunk_size > MAX_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}, got {chunk_size}"
            )
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap must be >= 0 and < chunk_size ({chunk_size}), got {chunk_overlap}"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into (passage_text, start_char_offset, end_char_offset).
        Guarantees natural boundaries (paragraph, sentence, space), monotonic progress,
        and avoids starting chunks mid-sentence or mid-word when overlapping.
        """
        if not text or not text.strip():
            return []

        text_len = len(text)
        passages: List[Tuple[str, int, int]] = []

        start_idx = 0
        while start_idx < text_len:
            # Skip any leading whitespace at start of chunk window
            while start_idx < text_len and text[start_idx].isspace():
                start_idx += 1

            if start_idx >= text_len:
                break

            target_end = min(start_idx + self.chunk_size, text_len)

            if target_end >= text_len:
                end_idx = text_len
            else:
                min_cut = start_idx + (self.chunk_size // 2)
                window = text[min_cut:target_end]

                # 1. Paragraph break (\n\s*\n)
                para_match = [m.start() for m in re.finditer(r"\n\s*\n", window)]
                if para_match:
                    end_idx = min_cut + para_match[-1]
                else:
                    # 2. Sentence ending (. / ! / ?) followed by space, newline, or end
                    sent_match = [m.start() for m in re.finditer(r"[.!?](?:\s|$)", window)]
                    if sent_match:
                        end_idx = min_cut + sent_match[-1] + 1
                    else:
                        # 3. Single newline
                        nl_match = [m.start() for m in re.finditer(r"\n", window)]
                        if nl_match:
                            end_idx = min_cut + nl_match[-1]
                        else:
                            # 4. Word boundary (whitespace)
                            space_match = [m.start() for m in re.finditer(r"\s", window)]
                            if space_match:
                                end_idx = min_cut + space_match[-1]
                            else:
                                end_idx = target_end

            chunk_raw = text[start_idx:end_idx]
            chunk_text = chunk_raw.strip()
            if chunk_text:
                passages.append((chunk_text, start_idx, end_idx))

            if end_idx >= text_len:
                break

            # Calculate next start position using overlap window: [end_idx - chunk_overlap, end_idx]
            overlap_min = max(start_idx + 1, end_idx - self.chunk_overlap)
            overlap_window = text[overlap_min:end_idx]

            # Priority 1: Align to a clean paragraph start inside the overlap window
            p_starts = [overlap_min + m.end() for m in re.finditer(r"\n\s*\n", overlap_window)]
            # Priority 2: Align to a clean sentence start inside the overlap window (.!? followed by whitespace)
            s_starts = [overlap_min + m.end() for m in re.finditer(r"[.!?]\s+", overlap_window)]
            # Priority 3: Align to a clean newline start
            nl_starts = [overlap_min + m.end() for m in re.finditer(r"\n+", overlap_window)]

            if p_starts:
                next_start = p_starts[0]
            elif s_starts:
                next_start = s_starts[0]
            elif nl_starts:
                next_start = nl_starts[0]
            else:
                # If no sentence/paragraph break in overlap window, advance to end_idx to guarantee complete sentences
                next_start = end_idx

            # Ensure strictly monotonic forward progress
            if next_start <= start_idx:
                next_start = end_idx

            start_idx = next_start

        return passages

    def chunk_document(
        self,
        metadata: DocumentMetadata,
        extracted_text: str,
        page_number: Optional[int] = None,
    ) -> List[EvidenceChunk]:
        """
        Convert extracted document text into a sequence of EvidenceChunk instances.
        Guarantees deterministic chunk IDs and source linkage.
        """
        if not extracted_text or not extracted_text.strip():
            return []

        raw_splits = self.split_text_with_offsets(extracted_text)
        chunks: List[EvidenceChunk] = []

        timestamp = current_iso_timestamp()

        for idx, (chunk_text, start_offset, end_offset) in enumerate(raw_splits):
            content_hash = compute_content_hash(chunk_text)
            # Deterministic chunk ID: <doc_id>_CHK<index>_<hash_prefix>
            chunk_id = f"{metadata.document_id}_CHK{idx:04d}_{content_hash[:8]}"

            chunk = EvidenceChunk(
                chunk_id=chunk_id,
                document_id=metadata.document_id,
                chunk_index=idx,
                text=chunk_text,
                source_filename=metadata.original_filename,
                file_type=metadata.file_type,
                page_number=page_number,
                start_offset=start_offset,
                end_offset=end_offset,
                created_at=timestamp,
                content_hash=content_hash,
            )
            chunks.append(chunk)

        return chunks

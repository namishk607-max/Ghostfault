"""Domain models and schema definitions for GhostFault evidence ingestion and retrieval."""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List


class DocumentStatus(str, Enum):
    """Lifecycle statuses for document ingestion."""
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class ErrorCode(str, Enum):
    """Standardized error codes for ingestion failures."""
    UNSUPPORTED_TYPE = "ERR_UNSUPPORTED_TYPE"
    EMPTY_FILE = "ERR_EMPTY_FILE"
    PATH_TRAVERSAL = "ERR_PATH_TRAVERSAL"
    INVALID_PDF = "ERR_INVALID_PDF"
    NO_EXTRACTABLE_TEXT = "ERR_NO_EXTRACTABLE_TEXT"
    ENCODING_ERROR = "ERR_ENCODING_ERROR"
    STORAGE_ERROR = "ERR_STORAGE_ERROR"
    EXTRACTION_ERROR = "ERR_EXTRACTION_ERROR"
    READ_ERROR = "ERR_READ_ERROR"
    INDEX_ERROR = "ERR_INDEX_ERROR"


@dataclass
class DocumentMetadata:
    """Structured metadata contract for every ingested document."""
    document_id: str
    original_filename: str
    file_type: str
    source_type: str
    storage_reference: str
    extraction_status: str
    ingestion_status: DocumentStatus
    created_at: str
    updated_at: str
    extracted_text_reference: Optional[str] = None
    extracted_character_count: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        data = asdict(self)
        if isinstance(self.ingestion_status, DocumentStatus):
            data["ingestion_status"] = self.ingestion_status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Deserialize metadata from dictionary."""
        raw_status = data.get("ingestion_status", DocumentStatus.FAILED.value)
        try:
            status = DocumentStatus(raw_status)
        except ValueError:
            status = DocumentStatus.FAILED

        return cls(
            document_id=data["document_id"],
            original_filename=data["original_filename"],
            file_type=data["file_type"],
            source_type=data["source_type"],
            storage_reference=data["storage_reference"],
            extraction_status=data["extraction_status"],
            ingestion_status=status,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            extracted_text_reference=data.get("extracted_text_reference"),
            extracted_character_count=int(data.get("extracted_character_count", 0)),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
        )


@dataclass
class EvidenceChunk:
    """
    Structured, validated representation of an indexed evidence chunk.
    Maintains exact source provenance back to an ingested catalog document.
    """
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source_filename: str
    file_type: str
    page_number: Optional[int]
    start_offset: Optional[int]
    end_offset: Optional[int]
    created_at: str
    content_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceChunk":
        """Deserialize chunk from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            chunk_index=int(data["chunk_index"]),
            text=data["text"],
            source_filename=data["source_filename"],
            file_type=data["file_type"],
            page_number=data.get("page_number"),
            start_offset=data.get("start_offset"),
            end_offset=data.get("end_offset"),
            created_at=data["created_at"],
            content_hash=data["content_hash"],
        )


@dataclass
class SearchResult:
    """
    A single ranked evidence passage retrieved from the search index.
    Contains source provenance and search score.
    """
    chunk: EvidenceChunk
    score: float
    matched_snippet: str
    matched_terms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize search result to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "matched_snippet": self.matched_snippet,
            "matched_terms": self.matched_terms,
        }


def current_iso_timestamp() -> str:
    """Return UTC ISO 8601 formatted timestamp string."""
    return datetime.now(timezone.utc).isoformat()

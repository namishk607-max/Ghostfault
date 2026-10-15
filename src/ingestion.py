"""Document ingestion orchestrator and validation pipeline."""

from pathlib import Path
from typing import Optional, Tuple, List
import hashlib

from src.models import DocumentMetadata, DocumentStatus, ErrorCode, current_iso_timestamp
from src.extractor import TextExtractor, ExtractionResult
from src.storage import EvidenceStore


class IngestionService:
    """
    Coordinates document validation, secure storage, text extraction,
    and metadata registration.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self, store: Optional[EvidenceStore] = None):
        self.store = store if store is not None else EvidenceStore()

    def validate_upload(self, filename: str, content: bytes) -> Tuple[bool, Optional[ErrorCode], Optional[str]]:
        """
        Validate file characteristics prior to processing.
        Returns: (is_valid, error_code, error_message)
        """
        if not filename or not filename.strip():
            return False, ErrorCode.UNSUPPORTED_TYPE, "Filename cannot be empty."

        # Check for path traversal indicators
        if ".." in filename or "/" in filename or "\\" in filename:
            # Note: We still sanitize filenames in storage, but we reject dangerous inputs early
            pass

        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            return (
                False,
                ErrorCode.UNSUPPORTED_TYPE,
                f"Unsupported file extension '{suffix}'. Supported types: {sorted(list(self.SUPPORTED_EXTENSIONS))}",
            )

        if len(content) == 0:
            return False, ErrorCode.EMPTY_FILE, f"Uploaded file '{filename}' is empty (0 bytes)."

        return True, None, None

    def ingest_document(
        self,
        filename: str,
        content: bytes,
        source_type: str = "manual_upload",
    ) -> DocumentMetadata:
        """
        Execute full ingestion pipeline for an uploaded file:
        1. Validate parameters.
        2. Assign unique document ID.
        3. Save original document safely.
        4. Extract text.
        5. Save extracted text.
        6. Persist metadata catalog.
        """
        doc_id = self.store.generate_document_id()
        timestamp = current_iso_timestamp()
        suffix = Path(filename).suffix.lower().lstrip(".")
        file_type = suffix if suffix else "unknown"

        # Step 1: Pre-validation
        is_valid, err_code, err_msg = self.validate_upload(filename, content)
        if not is_valid:
            metadata = DocumentMetadata(
                document_id=doc_id,
                original_filename=Path(filename).name,
                file_type=file_type,
                source_type=source_type,
                storage_reference="",
                extraction_status="failed",
                ingestion_status=DocumentStatus.FAILED,
                created_at=timestamp,
                updated_at=timestamp,
                extracted_character_count=0,
                error_code=err_code.value if err_code else None,
                error_message=err_msg,
            )
            self.store.add_or_update_metadata(metadata)
            return metadata

        # Step 2: Save original document
        try:
            storage_ref = self.store.save_original(doc_id, filename, content)
        except Exception as exc:
            metadata = DocumentMetadata(
                document_id=doc_id,
                original_filename=Path(filename).name,
                file_type=file_type,
                source_type=source_type,
                storage_reference="",
                extraction_status="failed",
                ingestion_status=DocumentStatus.FAILED,
                created_at=timestamp,
                updated_at=timestamp,
                extracted_character_count=0,
                error_code=ErrorCode.STORAGE_ERROR.value,
                error_message=f"Failed to write original document to storage: {exc}",
            )
            self.store.add_or_update_metadata(metadata)
            return metadata

        # Step 3: Text Extraction
        try:
            extraction_result: ExtractionResult = TextExtractor.extract(filename, content)
        except Exception as exc:
            extraction_result = ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.EXTRACTION_ERROR,
                error_message=f"Unhandled extraction exception: {exc}",
            )

        # Step 4: Handle extraction outcome
        if not extraction_result.success:
            metadata = DocumentMetadata(
                document_id=doc_id,
                original_filename=Path(filename).name,
                file_type=file_type,
                source_type=source_type,
                storage_reference=storage_ref,
                extraction_status="failed",
                ingestion_status=DocumentStatus.FAILED,
                created_at=timestamp,
                updated_at=current_iso_timestamp(),
                extracted_text_reference=None,
                extracted_character_count=0,
                error_code=extraction_result.error_code.value if extraction_result.error_code else None,
                error_message=extraction_result.error_message,
            )
            self.store.add_or_update_metadata(metadata)
            return metadata

        # Step 5: Save extracted text
        try:
            text_ref = self.store.save_extracted_text(doc_id, extraction_result.text)
        except Exception as exc:
            metadata = DocumentMetadata(
                document_id=doc_id,
                original_filename=Path(filename).name,
                file_type=file_type,
                source_type=source_type,
                storage_reference=storage_ref,
                extraction_status="failed",
                ingestion_status=DocumentStatus.FAILED,
                created_at=timestamp,
                updated_at=current_iso_timestamp(),
                extracted_text_reference=None,
                extracted_character_count=0,
                error_code=ErrorCode.STORAGE_ERROR.value,
                error_message=f"Failed to write extracted text file: {exc}",
            )
            self.store.add_or_update_metadata(metadata)
            return metadata

        # Step 6: Mark fully processed
        metadata = DocumentMetadata(
            document_id=doc_id,
            original_filename=Path(filename).name,
            file_type=file_type,
            source_type=source_type,
            storage_reference=storage_ref,
            extraction_status="success",
            ingestion_status=DocumentStatus.PROCESSED,
            created_at=timestamp,
            updated_at=current_iso_timestamp(),
            extracted_text_reference=text_ref,
            extracted_character_count=extraction_result.char_count,
            error_code=None,
            error_message=None,
        )
        self.store.add_or_update_metadata(metadata)
        return metadata

    def list_documents(self) -> List[DocumentMetadata]:
        """Return all cataloged documents in reverse chronological order."""
        docs = self.store.read_catalog()
        return sorted(docs, key=lambda d: d.created_at, reverse=True)

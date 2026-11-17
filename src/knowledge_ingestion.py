"""Local engineering document ingestion service and storage manager for GhostFault M9.4.2."""

import hashlib
import re
from pathlib import Path
from typing import Optional, Tuple

from src.extractor import ExtractionResult, TextExtractor
from src.knowledge_models import KnowledgeDocument, ReliabilityTier, SourceType
from src.knowledge_store import KnowledgeStore, KnowledgeStoreError
from src.storage import EvidenceStore


class KnowledgeIngestionError(RuntimeError):
    """Raised when knowledge document validation, extraction, or storage fails."""


class KnowledgeIngestionService:
    """Orchestrates validation, text extraction, local disk storage, and metadata registration.

    Maintains complete separation between external technical knowledge and
    internal factory incident evidence.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        base_dir: Optional[Path] = None,
    ) -> None:
        if not isinstance(knowledge_store, KnowledgeStore):
            raise TypeError("KnowledgeIngestionService requires a KnowledgeStore instance.")
        self.knowledge_store = knowledge_store
        if base_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            self.base_dir = project_root / "data" / "knowledge"
        else:
            self.base_dir = Path(base_dir).resolve()

        self.originals_dir = self.base_dir / "originals"
        self.extracted_dir = self.base_dir / "extracted"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create originals and extracted storage directories."""
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal and invalid characters."""
        base_name = Path(filename).name
        base_name = base_name.replace("..", "").replace("/", "").replace("\\", "")
        cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base_name)
        if not cleaned or cleaned.startswith("."):
            cleaned = f"doc_{cleaned}" if cleaned else "unnamed_document"
        return cleaned

    def validate_upload(
        self,
        filename: str,
        content: bytes,
    ) -> Tuple[bool, Optional[str]]:
        """Pre-validate upload characteristics.

        Returns (is_valid, error_message).
        """
        if not filename or not filename.strip():
            return False, "Filename cannot be empty."

        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            return (
                False,
                f"Unsupported file extension '{suffix}'. Only .pdf and .txt files are supported.",
            )

        if not content or len(content) == 0:
            return False, f"Uploaded file '{filename}' is empty (0 bytes)."

        return True, None

    def save_original(self, document_id: str, original_filename: str, content: bytes) -> Path:
        """Safely write raw file bytes to originals/<doc_id>_<safe_name> using atomic replacement."""
        safe_name = self.sanitize_filename(original_filename)
        storage_filename = f"{document_id}_{safe_name}"
        target_path = self.originals_dir / storage_filename

        temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
        try:
            temp_path.write_bytes(content)
            temp_path.replace(target_path)
        except OSError as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise KnowledgeIngestionError(
                f"Failed to store original knowledge document: {exc}"
            ) from None
        return target_path

    def save_extracted_text(self, document_id: str, text: str) -> Path:
        """Safely write extracted text to extracted/<doc_id>.txt using atomic replacement."""
        storage_filename = f"{document_id}.txt"
        target_path = self.extracted_dir / storage_filename

        temp_path = target_path.with_suffix(".txt.tmp")
        try:
            temp_path.write_text(text, encoding="utf-8")
            temp_path.replace(target_path)
        except OSError as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise KnowledgeIngestionError(
                f"Failed to store extracted knowledge text: {exc}"
            ) from None
        return target_path

    def get_extracted_text(self, document_id: str) -> Optional[str]:
        """Read extracted text for a given document ID."""
        target_path = self.extracted_dir / f"{document_id}.txt"
        if target_path.exists():
            try:
                return target_path.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    def ingest_document(
        self,
        filename: str,
        content: bytes,
        title: str,
        publisher_author: str,
        source_type: SourceType,
        reliability_tier: ReliabilityTier,
        dataset_id: Optional[str] = None,
        publication_year: Optional[int] = None,
        edition_revision: Optional[str] = None,
        equipment_category: Optional[str] = None,
        equipment_subtype: Optional[str] = None,
        manufacturer: Optional[str] = None,
        copyright_license: Optional[str] = None,
        rights_attribution: Optional[str] = None,
        is_proprietary_restricted: bool = False,
    ) -> KnowledgeDocument:
        """Execute full ingestion pipeline for an engineering document.

        1. Pre-validate filename and raw content bytes.
        2. Calculate SHA-256 hash from raw content bytes.
        3. Check scope duplicate existence early.
        4. Extract selectable text (pypdf for PDF, multi-encoding for TXT).
        5. Atomically save original and extracted files.
        6. Register document metadata into KnowledgeStore.
        7. On any registration failure, clean up written files.
        """
        # Step 1: Pre-validation
        is_valid, err_msg = self.validate_upload(filename, content)
        if not is_valid:
            raise KnowledgeIngestionError(err_msg or "Invalid document file.")

        # Step 2: Compute actual SHA-256 from raw bytes
        computed_sha256 = hashlib.sha256(content).hexdigest()

        # Step 3: Text extraction
        try:
            extraction_result: ExtractionResult = TextExtractor.extract(filename, content)
        except Exception as exc:
            raise KnowledgeIngestionError(f"Extraction failed unexpectedly: {exc}") from exc

        if not extraction_result.success:
            err = extraction_result.error_message or "Text extraction failed."
            raise KnowledgeIngestionError(f"Could not extract text from document: {err}")

        # Step 4: Generate ID and atomically store files
        scope_tag = dataset_id or "GLOBAL"
        doc_id = f"KNOW-{hashlib.sha256((computed_sha256 + filename + scope_tag).encode()).hexdigest()[:12].upper()}"

        original_path = self.save_original(doc_id, filename, content)
        try:
            extracted_path = self.save_extracted_text(doc_id, extraction_result.text)
        except Exception:
            if original_path.exists():
                try:
                    original_path.unlink()
                except OSError:
                    pass
            raise

        # Step 5: Register metadata in knowledge store
        try:
            document = self.knowledge_store.add_document(
                title=title,
                publisher_author=publisher_author,
                source_type=source_type,
                reliability_tier=reliability_tier,
                dataset_id=dataset_id,
                publication_year=publication_year,
                edition_revision=edition_revision,
                equipment_category=equipment_category,
                equipment_subtype=equipment_subtype,
                manufacturer=manufacturer,
                sha256_hash=computed_sha256,
                copyright_license=copyright_license,
                rights_attribution=rights_attribution,
                is_proprietary_restricted=is_proprietary_restricted,
                document_id=doc_id,
            )
            return document
        except Exception:
            # Clean up written files on database error (e.g. duplicate or invalid scope)
            if original_path.exists():
                try:
                    original_path.unlink()
                except OSError:
                    pass
            if extracted_path.exists():
                try:
                    extracted_path.unlink()
                except OSError:
                    pass
            raise

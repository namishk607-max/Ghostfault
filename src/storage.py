"""Local evidence storage manager for GhostFault.

Handles safe file storage, atomic writes, path traversal prevention,
and metadata persistence across sessions.
"""

import json
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from src.models import DocumentMetadata, DocumentStatus, ErrorCode, current_iso_timestamp


class EvidenceStore:
    """
    Manages local document storage and catalog metadata under a configurable root directory.
    Default directory: <project_root>/data/evidence/
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # Default to <project_root>/data/evidence
            project_root = Path(__file__).resolve().parent.parent
            self.base_dir = project_root / "data" / "evidence"
        else:
            self.base_dir = Path(base_dir).resolve()

        self.originals_dir = self.base_dir / "originals"
        self.extracted_dir = self.base_dir / "extracted"
        self.metadata_dir = self.base_dir / "metadata"
        self.catalog_file = self.metadata_dir / "catalog.json"

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required storage directories if they do not exist."""
        self.originals_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize input filename to prevent path traversal and unsafe characters.
        Extracts base name and removes characters outside alphanumeric, '.', '_', '-'.
        """
        # Strip directories to prevent path traversal
        base_name = Path(filename).name
        # Remove any path traversal tokens
        base_name = base_name.replace("..", "").replace("/", "").replace("\\", "")
        # Remove characters not matching safe pattern
        cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base_name)
        if not cleaned or cleaned.startswith("."):
            cleaned = f"doc_{cleaned}" if cleaned else "unnamed_document"
        return cleaned

    @staticmethod
    def generate_document_id() -> str:
        """Generate a collision-resistant unique document identifier."""
        return f"DOC-{uuid.uuid4().hex[:12].upper()}"

    def get_document_path(self, storage_reference: str) -> Path:
        """Resolve a storage reference relative to base directory."""
        return self.base_dir / storage_reference

    def save_original(self, document_id: str, original_filename: str, content: bytes) -> str:
        """
        Save the raw original file under originals/<doc_id>_<safe_filename>.
        Returns storage reference relative to base_dir.
        """
        safe_name = self.sanitize_filename(original_filename)
        storage_filename = f"{document_id}_{safe_name}"
        target_path = self.originals_dir / storage_filename

        # Write safely
        temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
        temp_path.write_bytes(content)
        temp_path.replace(target_path)

        return f"originals/{storage_filename}"

    def save_extracted_text(self, document_id: str, text: str) -> str:
        """
        Save extracted text under extracted/<doc_id>.txt.
        Returns storage reference relative to base_dir.
        """
        storage_filename = f"{document_id}.txt"
        target_path = self.extracted_dir / storage_filename

        temp_path = target_path.with_suffix(".txt.tmp")
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(target_path)

        return f"extracted/{storage_filename}"

    def get_extracted_text(self, metadata: DocumentMetadata) -> Optional[str]:
        """Read and return extracted text content if available."""
        if not metadata.extracted_text_reference:
            return None
        path = self.base_dir / metadata.extracted_text_reference
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def read_catalog(self) -> List[DocumentMetadata]:
        """Read and return list of all document metadata from catalog.json."""
        if not self.catalog_file.exists():
            return []

        try:
            with open(self.catalog_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            return [DocumentMetadata.from_dict(item) for item in data]
        except Exception:
            # If catalog is corrupted or unreadable, return empty list
            return []

    def save_catalog(self, documents: List[DocumentMetadata]) -> None:
        """Persist document catalog to catalog.json atomically."""
        data = [doc.to_dict() for doc in documents]
        temp_file = self.catalog_file.with_suffix(".json.tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.replace(self.catalog_file)

    def add_or_update_metadata(self, metadata: DocumentMetadata) -> None:
        """Insert or update a document metadata record in catalog."""
        docs = self.read_catalog()
        updated = False
        for i, doc in enumerate(docs):
            if doc.document_id == metadata.document_id:
                docs[i] = metadata
                updated = True
                break
        if not updated:
            docs.append(metadata)
        self.save_catalog(docs)

    def get_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Find a metadata record by document_id."""
        for doc in self.read_catalog():
            if doc.document_id == document_id:
                return doc
        return None

    def find_by_content_hash_or_name(self, filename: str) -> List[DocumentMetadata]:
        """Find existing records by original filename."""
        return [doc for doc in self.read_catalog() if doc.original_filename == filename]

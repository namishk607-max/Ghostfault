"""Local evidence storage manager for GhostFault.

Handles safe file storage, atomic writes, path traversal prevention,
and metadata persistence across sessions.
"""

import json
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Mapping

from src.models import (
    DocumentMetadata,
    DocumentStatus,
    ErrorCode,
    Dataset,
    DatasetValidationError,
    DatasetAssociationIssue,
    DatasetOwnershipError,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_DESCRIPTION,
    current_iso_timestamp,
)


class EvidenceStoreError(RuntimeError):
    """Raised when persisted evidence or dataset metadata cannot be read safely."""


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
        self.datasets_file = self.metadata_dir / "datasets.json"

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

    @staticmethod
    def generate_dataset_id() -> str:
        """Generate a collision-resistant unique dataset identifier."""
        return f"DATASET-{uuid.uuid4().hex[:8].upper()}"

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
                raise EvidenceStoreError(
                    f"Evidence catalog at {self.catalog_file} must contain a JSON list."
                )
            return [DocumentMetadata.from_dict(item) for item in data]
        except EvidenceStoreError:
            raise
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise EvidenceStoreError(
                f"Failed to read evidence catalog at {self.catalog_file}: {exc}"
            ) from exc

    def save_catalog(self, documents: List[DocumentMetadata]) -> None:
        """Persist catalog metadata without allowing ordinary ownership changes."""
        existing_by_id = {doc.document_id: doc for doc in self.read_catalog()}
        document_ids = [doc.document_id for doc in documents]
        if len(document_ids) != len(set(document_ids)):
            raise EvidenceStoreError("Evidence catalog cannot contain duplicate document IDs.")
        for doc in documents:
            self.require_dataset_id(doc.dataset_id, "Document", doc.document_id)
            existing = existing_by_id.get(doc.document_id)
            if existing is not None and existing.dataset_id != doc.dataset_id:
                raise DatasetOwnershipError(
                    f"Document '{doc.document_id}' ownership cannot be changed through "
                    "ordinary catalog writes. Use an explicit validated ownership repair."
                )
        self._write_catalog(documents)

    def _write_catalog(self, documents: List[DocumentMetadata]) -> None:
        """Atomically persist metadata for explicitly validated migration/repair operations."""
        data = [doc.to_dict() for doc in documents]
        temp_file = self.catalog_file.with_suffix(".json.tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.replace(self.catalog_file)

    def add_or_update_metadata(self, metadata: DocumentMetadata) -> None:
        """Insert or update a document metadata record in catalog."""
        if metadata.dataset_id == DEFAULT_DATASET_ID and self.get_dataset(DEFAULT_DATASET_ID) is None:
            self.ensure_default_dataset()
        self.require_dataset_id(metadata.dataset_id, "Document", metadata.document_id)
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

    # ==========================================================================
    # Milestone 7.1: Dataset Persistence & Migration Operations
    # ==========================================================================

    def list_datasets(self) -> List[Dataset]:
        """Read and return list of all datasets from datasets.json."""
        if not self.datasets_file.exists():
            return []

        try:
            with open(self.datasets_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise EvidenceStoreError(
                    f"Dataset registry at {self.datasets_file} must contain a JSON list."
                )
            datasets = [Dataset.from_dict(item) for item in data]
            dataset_ids = [dataset.dataset_id for dataset in datasets]
            if len(dataset_ids) != len(set(dataset_ids)):
                raise EvidenceStoreError(
                    f"Dataset registry at {self.datasets_file} contains duplicate dataset IDs."
                )
            return datasets
        except EvidenceStoreError:
            raise
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise EvidenceStoreError(
                f"Failed to read dataset registry at {self.datasets_file}: {exc}"
            ) from exc

    def save_datasets(self, datasets: List[Dataset]) -> None:
        """Persist datasets list to datasets.json atomically."""
        dataset_ids = [dataset.dataset_id for dataset in datasets]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise DatasetValidationError("Dataset registry cannot contain duplicate dataset IDs.")

        existing_ids = {dataset.dataset_id for dataset in self.list_datasets()}
        removed_ids = existing_ids - set(dataset_ids)
        if removed_ids:
            removed_list = ", ".join(sorted(removed_ids))
            raise DatasetOwnershipError(
                f"Cannot remove registered dataset(s) through save_datasets: {removed_list}. "
                "Dataset deletion is not supported."
            )

        data = [ds.to_dict() for ds in datasets]
        temp_file = self.datasets_file.with_suffix(".json.tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.replace(self.datasets_file)

    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Retrieve a dataset by its unique dataset_id. Returns None if not found or invalid."""
        if not dataset_id or not str(dataset_id).strip():
            return None
        target_id = str(dataset_id).strip()
        for ds in self.list_datasets():
            if ds.dataset_id == target_id:
                return ds
        return None

    def validate_dataset_id(self, dataset_id: Optional[str]) -> bool:
        """Check whether dataset_id exists in the persistent dataset registry."""
        if not dataset_id or not str(dataset_id).strip():
            return False
        return self.get_dataset(str(dataset_id).strip()) is not None

    def require_dataset_id(self, dataset_id: Optional[str], entity: str, entity_id: str) -> Dataset:
        """Return the registered owner dataset or raise an explanatory error."""
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise DatasetOwnershipError(
                f"{entity} '{entity_id}' has no dataset ID; ownership is required."
            )
        if dataset_id != dataset_id.strip():
            raise DatasetOwnershipError(
                f"{entity} '{entity_id}' dataset ID '{dataset_id}' has surrounding whitespace."
            )
        dataset = self.get_dataset(dataset_id)
        if dataset is None:
            raise DatasetOwnershipError(
                f"{entity} '{entity_id}' references unregistered dataset "
                f"'{dataset_id}'. Register the dataset before assigning this record."
            )
        return dataset

    def validate_document_ownership(
        self,
        metadata: DocumentMetadata,
        require_catalog_match: bool = False,
    ) -> Dataset:
        """Validate a document's dataset ID and optionally its canonical catalog record."""
        dataset = self.require_dataset_id(
            metadata.dataset_id,
            "Document",
            metadata.document_id,
        )
        if require_catalog_match:
            catalog_metadata = self.get_metadata(metadata.document_id)
            if catalog_metadata is None:
                raise DatasetOwnershipError(
                    f"Document '{metadata.document_id}' is not present in the evidence catalog."
                )
            if catalog_metadata.dataset_id != metadata.dataset_id:
                raise DatasetOwnershipError(
                    f"Document '{metadata.document_id}' dataset ID "
                    f"'{metadata.dataset_id}' does not match catalog ownership "
                    f"'{catalog_metadata.dataset_id}'."
                )
        return dataset

    def validate_catalog_ownership(self) -> List[DatasetAssociationIssue]:
        """Report missing and dangling dataset references in the document catalog."""
        registered_ids = {dataset.dataset_id for dataset in self.list_datasets()}
        issues: List[DatasetAssociationIssue] = []
        for metadata in self.read_catalog():
            dataset_id = metadata.dataset_id
            if not isinstance(dataset_id, str) or not dataset_id.strip():
                issues.append(
                    DatasetAssociationIssue(
                        code="missing_dataset_id",
                        entity_type="document",
                        entity_id=metadata.document_id,
                        dataset_id=None,
                        message=f"Document '{metadata.document_id}' has no dataset ID.",
                    )
                )
            elif dataset_id not in registered_ids:
                issues.append(
                    DatasetAssociationIssue(
                        code="unregistered_dataset_id",
                        entity_type="document",
                        entity_id=metadata.document_id,
                        dataset_id=dataset_id,
                        message=(
                            f"Document '{metadata.document_id}' references unregistered "
                            f"dataset '{dataset_id}'."
                        ),
                    )
                )
        return issues

    def preview_document_ownership_repair(
        self,
        assignments: Mapping[str, str],
        index: Any,
        required_status: Optional[DocumentStatus] = None,
    ) -> Dict[str, Any]:
        """Delegate a read-only ownership preview to the supplied index coordinator."""
        return index.preview_document_ownership_repair(
            self,
            assignments,
            required_status=required_status,
        )

    def apply_document_ownership_repair(
        self,
        plan: Dict[str, Any],
        index: Any,
        required_status: Optional[DocumentStatus] = None,
    ) -> Dict[str, Any]:
        """Delegate cross-layer ownership application to the supplied coordinator."""
        return index.apply_document_ownership_repair(
            self,
            plan,
            required_status=required_status,
        )

    def recover_document_ownership_repair(
        self,
        repair_id: str,
        index: Any,
        action: str = "resume",
    ) -> Dict[str, Any]:
        """Resume or roll back an interrupted repair using its durable journal."""
        return index.recover_document_ownership_repair(self, repair_id, action=action)

    def create_dataset(
        self,
        dataset_id: str,
        name: str,
        description: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dataset:
        """
        Create and persist a new dataset.
        Enforces validation and prevents duplicate IDs.
        Raises:
            DatasetValidationError: if validation fails.
            ValueError: if a dataset with this ID already exists.
        """
        timestamp = created_at if created_at else current_iso_timestamp()
        new_ds = Dataset(
            dataset_id=dataset_id,
            name=name,
            description=description,
            created_at=timestamp,
        )

        datasets = self.list_datasets()
        if any(d.dataset_id == new_ds.dataset_id for d in datasets):
            raise ValueError(f"Dataset with ID '{new_ds.dataset_id}' already exists.")

        datasets.append(new_ds)
        self.save_datasets(datasets)
        return new_ds

    def update_dataset(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dataset:
        """
        Update permitted metadata (name, description) of an existing dataset.
        Raises:
            KeyError: if dataset_id does not exist.
            DatasetValidationError: if new metadata is invalid.
        """
        if not dataset_id or not str(dataset_id).strip():
            raise KeyError(f"Dataset ID cannot be empty.")

        target_id = str(dataset_id).strip()
        datasets = self.list_datasets()
        target_idx = None
        for i, ds in enumerate(datasets):
            if ds.dataset_id == target_id:
                target_idx = i
                break

        if target_idx is None:
            raise KeyError(f"Dataset with ID '{target_id}' not found.")

        current_ds = datasets[target_idx]
        new_name = name if name is not None else current_ds.name
        new_desc = description if description is not None else current_ds.description

        # Create updated dataset and validate
        updated_ds = Dataset(
            dataset_id=current_ds.dataset_id,
            name=new_name,
            description=new_desc,
            created_at=current_ds.created_at,
        )

        datasets[target_idx] = updated_ds
        self.save_datasets(datasets)
        return updated_ds

    def ensure_default_dataset(self) -> Dataset:
        """
        Ensure the default dataset exists. If not, create it.
        Idempotent: safe to run repeatedly.
        """
        existing = self.get_dataset(DEFAULT_DATASET_ID)
        if existing:
            return existing
        return self.create_dataset(
            dataset_id=DEFAULT_DATASET_ID,
            name=DEFAULT_DATASET_NAME,
            description=DEFAULT_DATASET_DESCRIPTION,
        )

    def migrate_legacy_data(self) -> Dict[str, Any]:
        """
        Non-destructive, idempotent backward-compatible migration.
        Associates existing catalog documents lacking dataset_id with the default dataset.
        Preserves all IDs, provenance, text references, and file structures.
        Returns migration statistics.
        """
        # Step 1: Ensure default dataset exists
        default_ds = self.ensure_default_dataset()

        # Step 2: Migrate evidence document catalog
        docs = self.read_catalog()
        migrated_docs_count = 0
        already_migrated_count = 0

        for doc in docs:
            if (
                doc.ingestion_status == DocumentStatus.PROCESSED
                and (not doc.dataset_id or not doc.dataset_id.strip())
            ):
                doc.dataset_id = default_ds.dataset_id
                migrated_docs_count += 1
            else:
                already_migrated_count += 1

        if migrated_docs_count > 0:
            self._write_catalog(docs)

        return {
            "default_dataset_id": default_ds.dataset_id,
            "migrated_documents": migrated_docs_count,
            "already_associated_documents": already_migrated_count,
            "total_documents": len(docs),
        }

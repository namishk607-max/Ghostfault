"""Tests for GhostFault M9.4.2 local engineering document ingestion and storage."""

import hashlib
from io import BytesIO
from pathlib import Path

import pypdf
import pytest

from src.knowledge_ingestion import KnowledgeIngestionError, KnowledgeIngestionService
from src.knowledge_models import KnowledgeDocument, ReliabilityTier, SourceType
from src.knowledge_store import KnowledgeStore, KnowledgeStoreError
from src.models import DatasetOwnershipError
from src.storage import EvidenceStore


def _create_minimal_pdf(text: str) -> bytes:
    """Helper to generate a minimal valid PDF in memory using pypdf."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    # Use annotation or basic text stream
    # A standard way with pypdf to create text is writer.pages[0] or writer.add_metadata
    # In pypdf, creating text directly without reportlab can be done by writing text to a page or using standard stream:
    stream = BytesIO()
    # Alternatively, create a valid raw PDF structure:
    # A minimal valid PDF with extractable text:
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length 55 >> stream\n"
        b"BT /F1 12 Tf 100 700 Td (" + text.encode("latin-1") + b") Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000234 00000 n \n0000000340 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n413\n%%EOF\n"
    )
    return pdf_content


@pytest.fixture
def knowledge_env(tmp_path: Path):
    """Fixture providing isolated EvidenceStore, KnowledgeStore, and KnowledgeIngestionService."""
    evidence_dir = tmp_path / "evidence"
    evidence_store = EvidenceStore(base_dir=evidence_dir)
    evidence_store.ensure_default_dataset()
    evidence_store.create_dataset("DATASET-ALPHA", "Alpha Project")
    evidence_store.create_dataset("DATASET-BETA", "Beta Project")

    knowledge_root = tmp_path / "knowledge"
    knowledge_db = knowledge_root / "knowledge_library.db"
    knowledge_store = KnowledgeStore(store=evidence_store, db_path=knowledge_db)
    service = KnowledgeIngestionService(knowledge_store=knowledge_store, base_dir=knowledge_root)
    return evidence_store, knowledge_store, service, knowledge_root


class TestKnowledgeDocumentIngestion:
    """Test PDF and TXT ingestion, text extraction, SHA-256 calculation, and storage."""

    def test_ingest_valid_txt_document_global_scope(self, knowledge_env):
        _, knowledge_store, service, root_dir = knowledge_env
        raw_text = "API Standard 610: Centrifugal Pumps for Petroleum, Petrochemical and Natural Gas Industries."
        content = raw_text.encode("utf-8")
        expected_hash = hashlib.sha256(content).hexdigest()

        doc = service.ingest_document(
            filename="API_610_Standard.txt",
            content=content,
            title="API Standard 610",
            publisher_author="American Petroleum Institute",
            source_type=SourceType.ENGINEERING_STANDARD,
            reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            dataset_id=None,  # Global
            publication_year=2021,
            equipment_category="pump",
            equipment_subtype="centrifugal",
        )

        assert doc.document_id.startswith("KNOW-")
        assert doc.title == "API Standard 610"
        assert doc.dataset_id is None
        assert doc.sha256_hash == expected_hash

        # Verify storage on disk
        original_file = root_dir / "originals" / f"{doc.document_id}_API_610_Standard.txt"
        assert original_file.exists()
        assert original_file.read_bytes() == content

        extracted_file = root_dir / "extracted" / f"{doc.document_id}.txt"
        assert extracted_file.exists()
        assert extracted_file.read_text(encoding="utf-8") == raw_text

        # Verify retrieved via service helper
        retrieved_text = service.get_extracted_text(doc.document_id)
        assert retrieved_text == raw_text

    def test_ingest_valid_pdf_document_dataset_scope(self, knowledge_env):
        _, knowledge_store, service, root_dir = knowledge_env
        sample_text = "Flowserve Pump Maintenance Instructions."
        pdf_bytes = _create_minimal_pdf(sample_text)
        expected_hash = hashlib.sha256(pdf_bytes).hexdigest()

        doc = service.ingest_document(
            filename="Flowserve_Pump_Manual.pdf",
            content=pdf_bytes,
            title="Flowserve Mark 3 Centrifugal Pump Manual",
            publisher_author="Flowserve Corporation",
            source_type=SourceType.OEM_MANUAL,
            reliability_tier=ReliabilityTier.TIER_2_OEM,
            dataset_id="DATASET-ALPHA",
            publication_year=2020,
            manufacturer="Flowserve",
            is_proprietary_restricted=True,
        )

        assert doc.document_id.startswith("KNOW-")
        assert doc.dataset_id == "DATASET-ALPHA"
        assert doc.sha256_hash == expected_hash

        # Verify disk files
        original_file = root_dir / "originals" / f"{doc.document_id}_Flowserve_Pump_Manual.pdf"
        assert original_file.exists()
        assert original_file.read_bytes() == pdf_bytes

        extracted_file = root_dir / "extracted" / f"{doc.document_id}.txt"
        assert extracted_file.exists()
        extracted_text = extracted_file.read_text(encoding="utf-8")
        assert sample_text in extracted_text

    def test_reject_unsupported_file_extension(self, knowledge_env):
        _, _, service, _ = knowledge_env
        with pytest.raises(KnowledgeIngestionError, match="Unsupported file extension"):
            service.ingest_document(
                filename="manual.docx",
                content=b"Sample docx content",
                title="Title",
                publisher_author="Author",
                source_type=SourceType.GENERAL_REFERENCE,
                reliability_tier=ReliabilityTier.TIER_4_GENERAL,
            )

    def test_reject_empty_file(self, knowledge_env):
        _, _, service, _ = knowledge_env
        with pytest.raises(KnowledgeIngestionError, match="empty \\(0 bytes\\)"):
            service.ingest_document(
                filename="empty.txt",
                content=b"",
                title="Empty Title",
                publisher_author="Author",
                source_type=SourceType.GENERAL_REFERENCE,
                reliability_tier=ReliabilityTier.TIER_4_GENERAL,
            )

    def test_reject_empty_or_whitespace_text_file(self, knowledge_env):
        _, _, service, _ = knowledge_env
        with pytest.raises(KnowledgeIngestionError, match="no readable characters"):
            service.ingest_document(
                filename="whitespace.txt",
                content=b"   \n\n\t  \r\n",
                title="Whitespace Title",
                publisher_author="Author",
                source_type=SourceType.GENERAL_REFERENCE,
                reliability_tier=ReliabilityTier.TIER_4_GENERAL,
            )

    def test_reject_scanned_unextractable_pdf(self, knowledge_env):
        _, _, service, _ = knowledge_env
        # Minimal PDF with no text stream (e.g. blank page representing scanned raster image)
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        stream = BytesIO()
        writer.write(stream)
        blank_pdf_bytes = stream.getvalue()

        with pytest.raises(KnowledgeIngestionError, match="no extractable text was found"):
            service.ingest_document(
                filename="scanned_drawing.pdf",
                content=blank_pdf_bytes,
                title="Scanned Drawing",
                publisher_author="OEM",
                source_type=SourceType.OEM_MANUAL,
                reliability_tier=ReliabilityTier.TIER_2_OEM,
            )

    def test_reject_duplicate_file_in_same_scope_and_cleanup_disk(self, knowledge_env):
        _, _, service, root_dir = knowledge_env
        content = b"Exact identical standard content for ISO-9000."

        # First ingestion succeeds
        doc1 = service.ingest_document(
            filename="iso9000.txt",
            content=content,
            title="ISO 9000 Initial",
            publisher_author="ISO",
            source_type=SourceType.ENGINEERING_STANDARD,
            reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            dataset_id=None,
        )
        assert doc1 is not None

        # Count files in originals
        orig_count = len(list((root_dir / "originals").glob("*")))
        extr_count = len(list((root_dir / "extracted").glob("*")))
        assert orig_count == 1
        assert extr_count == 1

        # Second ingestion with identical content in same scope must fail
        with pytest.raises((KnowledgeStoreError, KnowledgeIngestionError)):
            service.ingest_document(
                filename="iso9000_copy.txt",
                content=content,
                title="ISO 9000 Duplicate",
                publisher_author="ISO",
                source_type=SourceType.ENGINEERING_STANDARD,
                reliability_tier=ReliabilityTier.TIER_1_STANDARD,
                dataset_id=None,
            )

        # Confirm no orphan temporary or permanent files were left for the duplicate
        assert len(list((root_dir / "originals").glob("*"))) == 1
        assert len(list((root_dir / "extracted").glob("*"))) == 1

    def test_identical_file_allowed_across_different_dataset_scopes(self, knowledge_env):
        _, _, service, _ = knowledge_env
        content = b"Common technical standard text used in multiple projects."

        # Ingest in DATASET-ALPHA
        doc_alpha = service.ingest_document(
            filename="common_spec.txt",
            content=content,
            title="Common Spec Alpha",
            publisher_author="Engineering Corp",
            source_type=SourceType.INDUSTRY_BULLETIN,
            reliability_tier=ReliabilityTier.TIER_3_CURATED_RESEARCH,
            dataset_id="DATASET-ALPHA",
        )
        assert doc_alpha.dataset_id == "DATASET-ALPHA"

        # Ingest in DATASET-BETA (different dataset scope)
        doc_beta = service.ingest_document(
            filename="common_spec.txt",
            content=content,
            title="Common Spec Beta",
            publisher_author="Engineering Corp",
            source_type=SourceType.INDUSTRY_BULLETIN,
            reliability_tier=ReliabilityTier.TIER_3_CURATED_RESEARCH,
            dataset_id="DATASET-BETA",
        )
        assert doc_beta.dataset_id == "DATASET-BETA"
        assert doc_alpha.sha256_hash == doc_beta.sha256_hash

    def test_reject_unregistered_dataset_scope(self, knowledge_env):
        _, _, service, _ = knowledge_env
        content = b"Proprietary manual for non-existent plant dataset."

        with pytest.raises(DatasetOwnershipError, match="references unregistered dataset"):
            service.ingest_document(
                filename="manual.txt",
                content=content,
                title="Unregistered Manual",
                publisher_author="Author",
                source_type=SourceType.OEM_MANUAL,
                reliability_tier=ReliabilityTier.TIER_2_OEM,
                dataset_id="DATASET-DOES-NOT-EXIST",
            )

    def test_sanitize_filename_prevents_path_traversal(self):
        malicious = "../../../etc/passwd.txt"
        cleaned = KnowledgeIngestionService.sanitize_filename(malicious)
        assert ".." not in cleaned
        assert "/" not in cleaned
        assert "\\" not in cleaned
        assert cleaned.endswith("passwd.txt")

        windows_malicious = "..\\..\\windows\\system32\\cmd.exe.txt"
        cleaned_win = KnowledgeIngestionService.sanitize_filename(windows_malicious)
        assert ".." not in cleaned_win
        assert "\\" not in cleaned_win

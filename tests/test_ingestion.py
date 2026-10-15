"""Automated test suite for GhostFault Milestone 2 evidence ingestion."""

import os
from io import BytesIO
from pathlib import Path
import pytest
import pypdf

from src.models import DocumentStatus, ErrorCode, DocumentMetadata
from src.extractor import TextExtractor
from src.storage import EvidenceStore
from src.ingestion import IngestionService


def create_sample_pdf(text: str) -> bytes:
    """Helper to synthesize a valid text-based PDF in-memory."""
    writer = pypdf.PdfWriter()
    # pypdf allows creating blank pages or adding text annotations/pages
    # To ensure robust text extraction with pypdf, we add a blank page and add annotations or form fields,
    # or write a minimal standard PDF with a valid font and text stream.
    # Minimal standard raw PDF with font and selectable text:
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length " + str(len(text) + 40).encode("ascii") + b" >>\nstream\n"
        b"BT\n/F1 12 Tf\n72 712 Td\n(" + text.encode("latin-1", "replace") + b") Tj\nET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        b"0000000266 00000 n \n0000000360 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n440\n%%EOF\n"
    )
    return pdf_content


def create_empty_pdf() -> bytes:
    """Helper to synthesize a valid PDF with zero text (e.g. blank page representing scanned doc without OCR)."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def temp_store(tmp_path: Path) -> EvidenceStore:
    """Fixture providing an isolated EvidenceStore in a temporary directory."""
    return EvidenceStore(base_dir=tmp_path / "evidence")


@pytest.fixture
def ingestion_service(temp_store: EvidenceStore) -> IngestionService:
    """Fixture providing an IngestionService configured with temporary storage."""
    return IngestionService(store=temp_store)


def test_valid_txt_ingestion(ingestion_service: IngestionService, temp_store: EvidenceStore):
    """Test successful ingestion of a valid plain text document."""
    content = "Pump bearing lubrication procedure: inspect oil viscosity every 500 hours.".encode("utf-8")
    meta = ingestion_service.ingest_document("manual_notes.txt", content)

    assert meta.ingestion_status == DocumentStatus.PROCESSED
    assert meta.extraction_status == "success"
    assert meta.original_filename == "manual_notes.txt"
    assert meta.file_type == "txt"
    assert meta.extracted_character_count == len(content)
    assert meta.extracted_text_reference is not None

    # Verify extracted text content from storage
    extracted = temp_store.get_extracted_text(meta)
    assert extracted is not None
    assert "lubrication procedure" in extracted


def test_valid_pdf_ingestion(ingestion_service: IngestionService, temp_store: EvidenceStore):
    """Test successful ingestion and text extraction of a valid PDF document."""
    pdf_bytes = create_sample_pdf("Hydraulic line pressure threshold is 210 bar.")
    meta = ingestion_service.ingest_document("hydraulic_spec.pdf", pdf_bytes)

    assert meta.ingestion_status == DocumentStatus.PROCESSED
    assert meta.extraction_status == "success"
    assert meta.file_type == "pdf"
    assert meta.extracted_character_count > 0

    extracted = temp_store.get_extracted_text(meta)
    assert extracted is not None
    assert "Hydraulic line pressure" in extracted


def test_empty_txt_file(ingestion_service: IngestionService):
    """Test that empty TXT files are rejected with ERR_EMPTY_FILE."""
    meta = ingestion_service.ingest_document("empty.txt", b"")

    assert meta.ingestion_status == DocumentStatus.FAILED
    assert meta.error_code == ErrorCode.EMPTY_FILE.value
    assert "empty" in meta.error_message.lower()


def test_whitespace_only_txt_file(ingestion_service: IngestionService):
    """Test that whitespace-only TXT files fail with ERR_NO_EXTRACTABLE_TEXT."""
    meta = ingestion_service.ingest_document("blank.txt", b"   \n\n\t   ")

    assert meta.ingestion_status == DocumentStatus.FAILED
    assert meta.error_code == ErrorCode.NO_EXTRACTABLE_TEXT.value
    assert "no readable characters" in meta.error_message.lower()


def test_invalid_pdf_file(ingestion_service: IngestionService):
    """Test that corrupt/invalid PDF files fail with ERR_INVALID_PDF."""
    corrupt_content = b"Not a real PDF file header"
    meta = ingestion_service.ingest_document("corrupted.pdf", corrupt_content)

    assert meta.ingestion_status == DocumentStatus.FAILED
    assert meta.error_code == ErrorCode.INVALID_PDF.value
    assert "pdf" in meta.error_message.lower()


def test_pdf_without_extractable_text(ingestion_service: IngestionService):
    """Test that valid PDFs with no text (e.g. blank or image scan) fail with ERR_NO_EXTRACTABLE_TEXT."""
    blank_pdf = create_empty_pdf()
    meta = ingestion_service.ingest_document("scanned_manual.pdf", blank_pdf)

    assert meta.ingestion_status == DocumentStatus.FAILED
    assert meta.error_code == ErrorCode.NO_EXTRACTABLE_TEXT.value
    assert "no extractable text" in meta.error_message.lower()


def test_unsupported_file_types(ingestion_service: IngestionService):
    """Test that unsupported extensions are rejected."""
    unsupported = ["data.csv", "image.png", "script.py", "archive.zip"]
    for name in unsupported:
        meta = ingestion_service.ingest_document(name, b"some content")
        assert meta.ingestion_status == DocumentStatus.FAILED
        assert meta.error_code == ErrorCode.UNSUPPORTED_TYPE.value
        assert "unsupported" in meta.error_message.lower()


def test_duplicate_filenames_receive_unique_ids(ingestion_service: IngestionService):
    """Test that two uploads with identical filenames receive distinct IDs and do not overwrite."""
    content1 = b"First version of maintenance procedure."
    content2 = b"Second distinct version of procedure."

    meta1 = ingestion_service.ingest_document("procedure.txt", content1)
    meta2 = ingestion_service.ingest_document("procedure.txt", content2)

    assert meta1.document_id != meta2.document_id
    assert meta1.storage_reference != meta2.storage_reference
    assert meta1.original_filename == meta2.original_filename == "procedure.txt"

    # Both documents should exist in catalog
    docs = ingestion_service.list_documents()
    assert len(docs) == 2
    ids = {d.document_id for d in docs}
    assert meta1.document_id in ids
    assert meta2.document_id in ids


def test_path_traversal_prevention(temp_store: EvidenceStore, ingestion_service: IngestionService):
    """Test that path traversal attempts in filename are sanitized safely."""
    malicious_filename = "../../../etc/passwd.txt"
    content = b"Critical diagnostic test note."

    meta = ingestion_service.ingest_document(malicious_filename, content)
    assert meta.ingestion_status == DocumentStatus.PROCESSED

    # Storage file should reside strictly inside temp_store.originals_dir
    actual_path = temp_store.get_document_path(meta.storage_reference)
    assert actual_path.exists()
    assert temp_store.originals_dir in actual_path.parents
    assert ".." not in actual_path.name


def test_metadata_persistence_and_retrieval(temp_store: EvidenceStore, ingestion_service: IngestionService):
    """Test that catalog.json correctly persists across new store instances."""
    content = b"Spindle alignment tolerance: 0.005mm."
    meta = ingestion_service.ingest_document("alignment.txt", content)

    # Instantiate a second store reading the same directory
    reloaded_store = EvidenceStore(base_dir=temp_store.base_dir)
    retrieved = reloaded_store.get_metadata(meta.document_id)

    assert retrieved is not None
    assert retrieved.document_id == meta.document_id
    assert retrieved.original_filename == "alignment.txt"
    assert retrieved.ingestion_status == DocumentStatus.PROCESSED
    assert retrieved.extracted_character_count == len(content)


def test_txt_encoding_fallback(ingestion_service: IngestionService, temp_store: EvidenceStore):
    """Test text extractor with UTF-8 BOM and Latin-1 characters."""
    latin1_text = "Régulateur de pression hydraulique: 150 bar."
    encoded_latin1 = latin1_text.encode("latin-1")

    meta = ingestion_service.ingest_document("french_manual.txt", encoded_latin1)
    assert meta.ingestion_status == DocumentStatus.PROCESSED

    extracted = temp_store.get_extracted_text(meta)
    assert extracted is not None
    assert "Régulateur" in extracted or "pression" in extracted

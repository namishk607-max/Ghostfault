"""Isolated text extraction services for PDF and TXT files."""

from dataclasses import dataclass
from io import BytesIO
from typing import Optional
import pypdf

from src.models import ErrorCode


@dataclass
class ExtractionResult:
    """Outcome of a document text extraction attempt."""
    success: bool
    text: str
    char_count: int
    error_code: Optional[ErrorCode] = None
    error_message: Optional[str] = None


class TextExtractor:
    """Isolated text extraction service for maintenance documents."""

    @staticmethod
    def extract_from_txt(content: bytes) -> ExtractionResult:
        """
        Safely decode and validate text content from raw bytes.
        Attempts UTF-8, then UTF-8 with BOM, latin-1, and utf-16.
        """
        if not content:
            return ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.EMPTY_FILE,
                error_message="The plain text document is empty (0 bytes).",
            )

        encodings = ["utf-8-sig", "utf-8", "latin-1", "utf-16"]
        decoded_text: Optional[str] = None

        for enc in encodings:
            try:
                decoded_text = content.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if decoded_text is None:
            return ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.ENCODING_ERROR,
                error_message="Unable to decode file using supported text encodings (UTF-8, Latin-1, UTF-16).",
            )

        stripped = decoded_text.strip()
        if not stripped:
            return ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.NO_EXTRACTABLE_TEXT,
                error_message="The text file contains only whitespace and no readable characters.",
            )

        return ExtractionResult(
            success=True,
            text=decoded_text,
            char_count=len(decoded_text),
        )

    @staticmethod
    def extract_from_pdf(content: bytes) -> ExtractionResult:
        """
        Extract selectable text from a PDF document using pypdf.
        Validates PDF structure and ensures extractable text exists.
        """
        if not content:
            return ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.EMPTY_FILE,
                error_message="The PDF file is empty (0 bytes).",
            )

        # Validate PDF magic header
        if not content.startswith(b"%PDF-"):
            return ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.INVALID_PDF,
                error_message="File header does not match valid PDF specification (%PDF-).",
            )

        try:
            reader = pypdf.PdfReader(BytesIO(content))
        except Exception as exc:
            return ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.INVALID_PDF,
                error_message=f"Failed to parse PDF document structure: {exc}",
            )

        if reader.is_encrypted:
            try:
                # Attempt empty password decrypt
                decrypt_success = reader.decrypt("")
                if decrypt_success == 0:
                    return ExtractionResult(
                        success=False,
                        text="",
                        char_count=0,
                        error_code=ErrorCode.INVALID_PDF,
                        error_message="PDF is password encrypted and cannot be extracted.",
                    )
            except Exception as exc:
                return ExtractionResult(
                    success=False,
                    text="",
                    char_count=0,
                    error_code=ErrorCode.INVALID_PDF,
                    error_message=f"PDF decryption failed: {exc}",
                )

        if len(reader.pages) == 0:
            return ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.INVALID_PDF,
                error_message="PDF document contains zero pages.",
            )

        extracted_pages = []
        try:
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                extracted_pages.append(page_text)
        except Exception as exc:
            return ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.EXTRACTION_ERROR,
                error_message=f"Error extracting text from PDF page: {exc}",
            )

        full_text = "\n\n".join(extracted_pages)
        stripped = full_text.strip()

        if not stripped:
            return ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.NO_EXTRACTABLE_TEXT,
                error_message=(
                    "PDF parsed successfully, but no extractable text was found. "
                    "The document may consist entirely of scanned images or raster graphics. "
                    "(Note: OCR is not supported in Milestone 2)."
                ),
            )

        return ExtractionResult(
            success=True,
            text=full_text,
            char_count=len(full_text),
        )

    @classmethod
    def extract(cls, filename: str, content: bytes) -> ExtractionResult:
        """Route to appropriate extractor based on normalized extension."""
        lower_name = filename.lower()
        if lower_name.endswith(".txt"):
            return cls.extract_from_txt(content)
        elif lower_name.endswith(".pdf"):
            return cls.extract_from_pdf(content)
        else:
            return ExtractionResult(
                success=False,
                text="",
                char_count=0,
                error_code=ErrorCode.UNSUPPORTED_TYPE,
                error_message=f"Unsupported file extension in '{filename}'. Only .pdf and .txt are supported.",
            )

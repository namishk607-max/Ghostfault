# GhostFault

> **Evidence-first industrial failure intelligence**

GhostFault connects maintenance records, machine manuals, and incident reports to identify recurring failure patterns and generate evidence-supported diagnostic hypotheses.

---

## Architecture & Directory Layout

```text
Ghostvault/
├── .gitignore               # Excludes virtual environments, caches, secrets, and data/evidence/
├── README.md                # Project documentation and setup instructions
├── requirements.txt         # Project dependencies (streamlit, pandas, pytest, pypdf)
├── app.py                   # Streamlit dashboard interface
├── data/
│   ├── sample/
│   │   └── incidents.csv    # Synthetic industrial maintenance incidents (CSV)
│   └── evidence/            # [Runtime / Local only - Git ignored]
│       ├── originals/       # Safe storage of raw uploaded files (<doc_id>_<safe_name>)
│       ├── extracted/       # Extracted plain text UTF-8 files (<doc_id>.txt)
│       └── metadata/
│           └── catalog.json # Structured document catalog and ingestion metadata
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Incident dataset validation, relative loading, and filtering
│   ├── models.py            # Ingestion contracts, DocumentMetadata, DocumentStatus, ErrorCode
│   ├── extractor.py         # Isolated text extraction for TXT and PDF (pypdf)
│   ├── storage.py           # Atomic writes, path traversal sanitization, and catalog management
│   └── ingestion.py         # Ingestion orchestration and pre-validation pipeline
└── tests/
    ├── test_incidents.py    # Unit tests for incident dataset integrity and filtering
    └── test_ingestion.py    # Unit tests for document ingestion, validation, and extraction
```

---

## Milestone 2: Evidence Ingestion & Document Management

Milestone 2 delivers a reliable local document ingestion foundation without introducing external AI APIs, vector databases, or cloud services:

- **Supported Document Formats:** Selectable text `.pdf` and plain-text `.txt` files (UTF-8, Latin-1, UTF-16).
- **Document Identity:** Collision-resistant UUID `DOC-XXXXXXXXXXXX` generated per upload. Duplicate filenames are safe and will not overwrite previous uploads.
- **Path Traversal Protection:** Filenames are sanitized, stripped of directory separators and parent traversals (`..`), and confined strictly to the configured evidence root.
- **Atomic Persistence:** Catalog updates and file writes use temporary files replaced atomically to prevent corruption.
- **Explicit Failure Detection:** Empty files, corrupt PDFs, unsupported file types, and PDFs without selectable text are flagged with clear error codes (`ERR_EMPTY_FILE`, `ERR_INVALID_PDF`, `ERR_NO_EXTRACTABLE_TEXT`, `ERR_UNSUPPORTED_TYPE`).
- **Scanned PDF Notice:** Scanned PDFs (raster images without an OCR text layer) will be flagged as containing no extractable text. OCR is not implemented in Milestone 2.

### Ingestion Contract & Metadata Schema

Every ingested document maintains the following metadata in `catalog.json`:
- `document_id`: Unique identifier (e.g., `DOC-A1B2C3D4E5F6`).
- `original_filename`: Original filename preserved for display.
- `file_type`: Detected extension type (`pdf` or `txt`).
- `source_type`: Origin (e.g., `dashboard_upload`, `manual_upload`).
- `storage_reference`: Path to original file relative to storage root.
- `extraction_status`: Extraction outcome (`success` or `failed`).
- `ingestion_status`: Ingestion lifecycle state (`uploaded`, `validating`, `processing`, `processed`, `failed`).
- `created_at`: UTC ISO 8601 creation timestamp.
- `updated_at`: UTC ISO 8601 last update timestamp.
- `extracted_text_reference`: Path to saved `.txt` file relative to storage root.
- `extracted_character_count`: Count of extracted text characters.
- `error_code`: Standardized error code if failed (or `null`).
- `error_message`: Human-readable failure explanation (or `null`).

---

## Local Setup & Launch Instructions (Windows PowerShell)

Follow these steps in **Windows PowerShell** from the project root directory:

### 1. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first, or activate via `.\.venv\Scripts\activate.bat` in CMD).*

### 2. Install / Update Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3. Run Automated Tests
```powershell
python -m pytest -v
```

### 4. Launch the Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Synthetic Data & Scope Boundary Notice

- All incident records in `data/sample/incidents.csv` are **fictional and synthetic**.
- **No external AI APIs, vector databases, embeddings, or RAG models** are configured in Milestone 2.
- The **Evidence Search** and **Fault Analysis** sections remain marked as *Planned Features* with disabled UI previews.

---

## Roadmap

- [x] **Milestone 1:** Core foundation, synthetic incidents CSV, data loader, filtering, unit tests, and dashboard.
- [x] **Milestone 2:** Evidence ingestion, secure local document storage, TXT/PDF text extraction, catalog metadata, and inspection.
- [ ] **Milestone 3:** Chunking, indexing, and local Evidence Search retrieval.
- [ ] **Milestone 4:** Fault Analysis and evidence-supported diagnostic hypotheses.

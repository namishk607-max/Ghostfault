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
│       ├── metadata/
│       │   └── catalog.json # Structured document catalog and ingestion metadata
│       └── index/
│           └── evidence_index.db # SQLite FTS5 persistent full-text search index
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Incident dataset validation, relative loading, and filtering
│   ├── models.py            # Typed domain models (Incidents, Chunks, SearchResults, Hypotheses, Scopes)
│   ├── extractor.py         # Isolated text extraction for TXT and PDF (pypdf)
│   ├── storage.py           # Atomic writes, path traversal sanitization, and catalog management
│   ├── ingestion.py         # Ingestion orchestration and pre-validation pipeline
│   ├── chunker.py           # Deterministic text chunker with natural boundaries & offsets
│   ├── indexer.py           # Persistent SQLite FTS5 index manager with change tracking
│   ├── retriever.py         # Deterministic keyword retrieval with BM25 ranking & snippet highlights
│   └── analyzer.py          # Evidence-grounded fault analysis engine & quality hardening
└── tests/
    ├── test_incidents.py    # Unit tests for incident dataset integrity and filtering
    ├── test_ingestion.py    # Unit tests for document ingestion, validation, and extraction
    ├── test_retrieval.py    # Unit tests for chunking, FTS5 indexing, and BM25 retrieval
    └── test_analysis.py     # Unit tests for fault analysis, deduplication, and quality hardening
```

---

## Milestone 4 & 4.1 / 4.1.2: Evidence-Grounded Fault Analysis & Quality Validation

Milestone 4.1 and 4.1.2 enhance evidence quality, traceability, deterministic passage validation, and strict hypothesis qualification:

### 1. Deterministic Passage Quality Classification (Milestone 4.1.2)
- **Substantive Technical Content:** Classifies genuine operational notes, maintenance steps, physical symptom reports, and component descriptions.
- **Search Terms / Metadata Exclusion:** Identifies search-term keyword lists and administrative headers (e.g. `Search terms: bearing temperature...`), preventing them from masquerading as substantive evidence.
- **Excluded Evidence Audit Drawer:** Low-value or metadata passages are displayed transparently in an "Excluded or Low-Value Retrieval Results" drawer with clear reasons for exclusion.
- **Transparent Pipeline Metrics:** Shows exact counts for Raw Retrieved, Unique (Deduplicated), Substantive Retained, Excluded Low-Value, and Supporting Hypotheses passages.

### 2. Duplicate Evidence Handling & Deduplication
- **Multi-Copy Detection:** Detects identical and near-duplicate passages across different uploads or formats (e.g., TXT vs PDF duplicates of the same manual).
- **No False Corroboration:** Repeated copies of substantially identical passages are grouped under the primary reference rather than counted as separate independent corroboration.
- **Traceability Preservation:** All genuine source filenames, document IDs, and chunk indices for every copy are preserved and displayed in the duplicate audit drawer.

### 3. Sentence-Bounded Snippets & Readability
- **Faithful Boundaries:** Snippets cut cleanly at complete sentence boundaries (`. `, `! `, `? `) or paragraph breaks, avoiding mid-word truncations.
- **Excerpt Flagging:** In the rare case where chunk start offsets fall mid-sentence, snippets are explicitly labeled with `[Excerpt - begins mid-sentence in source chunk]`.
- **No Text Fabrication:** Snippets reflect verbatim extracted text without paraphrasing or quote modification.

### 4. Context & Relationship-Specific Support Reasons
- **Genuine Relationships:** Distinguishes symptom co-occurrence, general mechanisms, historical maintenance, and physical observations rather than emitting generic keyword lists.
- **Measurement Context:** Distinguishes between recorded incident facts, verified manufacturer OEM limits, and illustrative/synthetic test values with caution badges.
- **Machine Scope Discrimination:** General technical references are clearly separated from machine-specific evidence (e.g., M-104 vs M-999 or general manuals).

### 5. Strict Hypothesis Qualification
- **Tentative Wording:** Emphasizes that symptom correlation indicates potential failure mechanisms but **does not confirm root cause** without physical technician inspection.
- **No Root Cause Filling:** Where evidence is absent or insufficient, reports `INSUFFICIENT_EVIDENCE` or `NO_EVIDENCE` rather than fabricating diagnostic causes.
- **Low-Value Protection:** If retrieval only matches search-term lists or metadata, the system returns `INSUFFICIENT_EVIDENCE` with 0 hypotheses.

---

## Local Setup & Launch Instructions (Windows PowerShell)

Follow these steps in **Windows PowerShell** from the project root directory:

### 1. Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first, or activate via `.\.venv\Scripts\activate.bat` in CMD).*

### 2. Install / Verify Dependencies
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

## Synthetic Data & Safety Disclaimer

- Incident records in `data/sample/incidents.csv` are **fictional and synthetic**.
- **No external AI APIs, cloud vector databases, or neural embeddings** are used.
- Investigative hypotheses are **preliminary technical suggestions** grounded in available documentation and require qualified human judgment and standard OEM maintenance/lockout procedures.

---

## Roadmap

- [x] **Milestone 1:** Core foundation, synthetic incidents CSV, data loader, filtering, unit tests, and dashboard.
- [x] **Milestone 2:** Evidence ingestion, secure local document storage, TXT/PDF text extraction, catalog metadata, and inspection.
- [x] **Milestone 3:** Deterministic chunking, persistent SQLite FTS5 indexing, BM25 evidence retrieval, and provenance tracking.
- [x] **Milestone 4:** Evidence-grounded fault analysis, explainable hypothesis generation, and cross-machine validation.
- [x] **Milestone 4.1:** Evidence quality and traceability hardening (deduplication, sentence boundaries, scope/context classification).
- [ ] **Milestone 5:** Future system expansion.

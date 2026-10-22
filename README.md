# GhostFault

> **Evidence-first industrial failure intelligence**

GhostFault connects maintenance records, machine manuals, and incident reports to identify recurring failure patterns and generate evidence-supported diagnostic hypotheses. It runs entirely locally — no cloud services, paid APIs, or neural embeddings required.

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

## Features

### Incident Dashboard & Inspector
- Interactive Streamlit dashboard for browsing synthetic industrial maintenance incidents.
- Machine-based filtering (e.g. M-104, M-205, M-308) with incident tables showing date, symptom, action, and outcome.
- Incident Detail Inspector: selecting any row renders a full detail panel with incident ID, machine ID, date, reported symptom, maintenance action, and outcome.
- Dark-themed, responsive layout with graceful handling of missing or malformed records.

### Document Ingestion & Evidence Storage
- Local PDF and TXT file upload via the Streamlit interface.
- Automated text extraction (plain text passthrough, pypdf-based PDF extraction).
- Secure local storage: originals preserved with sanitized filenames, extracted text stored as UTF-8.
- Structured document catalog (`catalog.json`) with document IDs, filenames, file types, ingestion status, timestamps, and error tracking.
- Validation pipeline: rejects empty files, whitespace-only content, unsupported file types, and path-traversal attempts with explicit error messages.

### Evidence Chunking, Indexing & Retrieval
- Deterministic text chunking with configurable size, overlap, and natural boundary detection (sentence and paragraph breaks).
- Persistent SQLite FTS5 full-text search index with content hashing for change tracking and stale-chunk replacement.
- BM25-ranked keyword retrieval with snippet highlighting and configurable result limits.
- Document-scoped search filtering and empty/special-character query handling.
- Full provenance: every chunk links back to its source document ID, filename, chunk index, and character offsets.

### Evidence-Grounded Fault Analysis
- Pattern-based hypothesis generation: matches incident symptoms and maintenance actions against known industrial failure patterns (bearing wear, lubrication degradation, seal failure, hydraulic faults).
- Hypotheses are linked to concrete retrieved evidence passages — no unsupported claims.
- Tentative qualification: every hypothesis explicitly states that symptom correlation does not confirm root cause without physical inspection.
- `INSUFFICIENT_EVIDENCE` / `NO_EVIDENCE` status when retrieved evidence is absent or inadequate — the system never fabricates a diagnosis.

### Duplicate Evidence Handling
- Exact-duplicate and near-duplicate passage detection across different uploads and formats (e.g. TXT vs PDF copies of the same manual).
- Grouped deduplication: repeated copies of substantially identical text are consolidated under the primary reference rather than counted as independent corroboration.
- Full provenance preserved: all genuine source filenames, document IDs, and chunk indices for every copy are listed in a duplicate audit drawer.

### Passage Quality Classification & Filtering
- Deterministic classifier identifies four passage types:
  - **Substantive Technical**: Operational notes, maintenance steps, symptom reports, and component descriptions.
  - **Search Terms / Indexing Metadata**: Keyword lists (e.g. `Search terms: bearing temperature, elevated temperature...`) — excluded from hypothesis support.
  - **Header / Footer Metadata**: Administrative document headers (Document ID, Record Type, Equipment blocks) — excluded from hypothesis support.
  - **Empty / Low-Substance**: Fragmentary tokens too brief to convey diagnostic facts.
- Low-value passages are transparently displayed under an "Excluded or Low-Value Retrieval Results" diagnostic section with the specific reason for exclusion.
- If all retrieved passages are non-substantive, the system returns `INSUFFICIENT_EVIDENCE` with zero hypotheses.

### Transparent Pipeline Metrics
- Dashboard displays five separately derived counts (not invented — computed from actual pipeline results):
  - **Raw Retrieved** — total chunks returned by keyword retrieval.
  - **Unique (Deduplicated)** — chunks after exact and near-duplicate grouping.
  - **Substantive Retained** — passages classified as genuine technical content.
  - **Excluded Low-Value** — passages filtered out (search terms, metadata, empty).
  - **Supporting Hypotheses** — passages actively grounding investigative hypotheses.

### Sentence-Bounded Snippets & Readability
- Snippets prefer complete sentence boundaries (`. `, `! `, `? `) and paragraph breaks over mid-word truncation.
- When a source chunk begins mid-sentence, the snippet is explicitly labeled as an excerpt.
- Verbatim extracted text only — no paraphrasing, rewriting, or fabricated quotations.

### Context-Aware Support Reasons
- Each evidence passage receives a relationship-specific support reason rather than a generic keyword list:
  - **Direct incident observations** — machine-specific notes from the selected unit.
  - **General technical mechanisms** — OEM or reference material discussing potential failure modes without asserting they apply to a specific machine.
  - **Historical maintenance** — past corrective actions, inspections, and retorquing records.
  - **Illustrative test context** — synthetic or reference readings clearly tagged so they are not mistaken for physical telemetry.

### Machine Scope Discrimination
- **Exact Machine** — passages referencing the selected unit (e.g. M-104) are tagged as direct machine evidence.
- **General Reference** — passages without specific machine identifiers are treated as general technical material, not an exact match.
- **Cross-Machine Conflict** — passages referencing a different unit (e.g. M-999 vs M-104) are flagged with cross-unit alert warnings.

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

## Implemented Capabilities

- [x] Core foundation, synthetic incident dataset, data loader, filtering, and unit tests
- [x] Interactive Streamlit dashboard with machine filtering and Incident Detail Inspector
- [x] Evidence ingestion with secure local document storage and TXT/PDF text extraction
- [x] Deterministic chunking, persistent SQLite FTS5 indexing, and BM25 evidence retrieval
- [x] Evidence-grounded fault analysis with explainable hypothesis generation
- [x] Duplicate and near-duplicate evidence detection with provenance preservation
- [x] Passage quality classification and non-substantive evidence filtering
- [x] Transparent pipeline metrics and excluded-evidence audit
- [x] Sentence-bounded snippets, context-aware support reasons, and machine scope discrimination
- [x] 56 automated tests across incidents, ingestion, retrieval, and analysis

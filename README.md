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
│   ├── models.py            # Ingestion & retrieval models (DocumentMetadata, EvidenceChunk, SearchResult)
│   ├── extractor.py         # Isolated text extraction for TXT and PDF (pypdf)
│   ├── storage.py           # Atomic writes, path traversal sanitization, and catalog management
│   ├── ingestion.py         # Ingestion orchestration and pre-validation pipeline
│   ├── chunker.py           # Deterministic text chunker with natural boundaries & offsets
│   ├── indexer.py           # Persistent SQLite FTS5 index manager with change tracking
│   └── retriever.py         # Deterministic keyword retrieval with BM25 ranking & snippet highlights
└── tests/
    ├── test_incidents.py    # Unit tests for incident dataset integrity and filtering
    ├── test_ingestion.py    # Unit tests for document ingestion, validation, and extraction
    └── test_retrieval.py    # Unit tests for chunking, FTS5 indexing, and BM25 retrieval
```

---

## Milestone 3: Evidence Chunking, Indexing & Retrieval

Milestone 3 delivers a reliable, deterministic local evidence retrieval engine:

### 1. Chunking Policy & Configuration Defaults
- **Chunk Size:** 600 characters (`DEFAULT_CHUNK_SIZE`), bounded between 100 and 4,000 characters.
- **Overlap:** 100 characters (`DEFAULT_CHUNK_OVERLAP`) to avoid losing context across boundaries.
- **Natural Boundary Detection:** Prefers paragraph boundaries (`\n\n`), then sentence endings (`. `, `! `, `? `), then newlines (`\n`), and spaces before falling back to fixed cuts.
- **Provenance:** Every chunk retains `document_id`, `source_filename`, `file_type`, character `start_offset`, `end_offset`, `content_hash`, and a collision-resistant deterministic `chunk_id` (`<doc_id>_CHK<index>_<hash>`).
- **Page Numbers:** Explicitly labeled as *Unavailable* when documents have been stored as flattened extracted text, avoiding fabricated page provenance.

### 2. Search Index & Storage
- **Backend:** Embedded **SQLite FTS5** virtual table (`chunks_fts`) using Porter stemming and unicode61 tokenizer.
- **Persistence Location:** `data/evidence/index/evidence_index.db` (100% local, excluded from Git).
- **Change Detection & Transactions:** Documents are fingerprinted via SHA-256. Re-indexing identical documents is a no-op; updating document text replaces stale chunks atomically within a database transaction.
- **Lifecycle Operations:** Supports automatic indexing upon successful ingestion, manual on-demand `Sync Index`, and a full zero-loss `Rebuild Index`.

### 3. Deterministic Retrieval & Ranking
- **Query Sanitization:** Strips FTS control syntax, quotes alphanumeric terms, and applies prefix wildcards (`"term"*`) joined by boolean `AND`.
- **Ranking:** Scored via SQLite FTS5 BM25 algorithm. Scores represent keyword relevance and term density within matching passages.
- **Important Score Disclaimer:** Retrieval scores reflect search relevance, **never** diagnostic confidence or failure probabilities.
- **No Hallucinated References:** Empty matches return an empty list without fabricating references.

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

## Synthetic Data & Scope Boundary Notice

- Incident records in `data/sample/incidents.csv` are **fictional and synthetic**.
- **No external AI APIs, cloud vector databases, or neural embeddings** are used.
- The **Fault Analysis** module remains marked as a *Planned Feature* with disabled UI preview.

---

## Roadmap

- [x] **Milestone 1:** Core foundation, synthetic incidents CSV, data loader, filtering, unit tests, and dashboard.
- [x] **Milestone 2:** Evidence ingestion, secure local document storage, TXT/PDF text extraction, catalog metadata, and inspection.
- [x] **Milestone 3:** Deterministic chunking, persistent SQLite FTS5 indexing, BM25 evidence retrieval, and provenance tracking.
- [ ] **Milestone 4:** Fault Analysis and evidence-supported diagnostic hypotheses.

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
│   ├── evidence/            # [Runtime / Local only - Git ignored]
│   │   ├── originals/       # Safe storage of raw uploaded files (<doc_id>_<safe_name>)
│   │   ├── extracted/       # Extracted plain text UTF-8 files (<doc_id>.txt)
│   │   ├── metadata/
│   │   │   ├── catalog.json # Document catalog and dataset associations
│   │   │   └── datasets.json # Persistent dataset registry
│   │   └── index/
│   │       └── evidence_index.db # SQLite FTS5 persistent full-text search index
│   └── history/             # [Runtime / Local only - Git ignored]
│       └── investigation_history.db # Versioned investigation snapshots and activity
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Incident dataset validation, relative loading, and filtering
│   ├── hub.py               # Project summaries, dataset state, integrity, and navigation
│   ├── history.py           # Dataset-scoped persistent investigation history and activity
│   ├── models.py            # Typed domain models (Incidents, Chunks, SearchResults, Hypotheses, Scopes)
│   ├── extractor.py         # Isolated text extraction for TXT and PDF (pypdf)
│   ├── storage.py           # Atomic writes, path traversal sanitization, and catalog management
│   ├── ingestion.py         # Ingestion orchestration and pre-validation pipeline
│   ├── chunker.py           # Deterministic text chunker with natural boundaries & offsets
│   ├── indexer.py           # Persistent SQLite FTS5 index manager with change tracking
│   ├── retriever.py         # Deterministic keyword retrieval with BM25 ranking & snippet highlights
│   ├── analyzer.py          # Evidence-grounded fault analysis engine & quality hardening
│   ├── cross_incident.py    # Cross-incident comparison and recurrence analysis engine
│   └── evidence_linker.py   # Cross-incident evidence linking service & provenance tracking
└── tests/
    ├── test_incidents.py    # Unit tests for incident dataset integrity and filtering
    ├── test_ingestion.py    # Unit tests for document ingestion, validation, and extraction
    ├── test_retrieval.py    # Unit tests for chunking, FTS5 indexing, and BM25 retrieval
    ├── test_analysis.py     # Unit tests for fault analysis, deduplication, and quality hardening
    ├── test_cross_incident.py # Unit tests for cross-incident recurrence and comparison
    ├── test_evidence_linker.py # Unit tests for cross-incident evidence linking
    ├── test_workspace.py    # Dataset workspace, active state, and association tests
    ├── test_dataset_storage.py # Dataset persistence and legacy migration tests
    ├── test_dataset_isolation.py # Dataset-scoping regression tests
    ├── test_dataset_ownership.py # Ownership validation across registry, catalog, and index
    ├── test_dataset_retrieval.py # Strict dataset-scoped evidence retrieval
    ├── test_investigation_history.py # Persistent history, activity, and dataset isolation
    └── test_m7_3_stabilization.py # Navigation, UI state, and integrity regressions
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

### Investigation History & Activity
- Fault-analysis results are saved as immutable incident and analysis snapshots in the separate `data/history/investigation_history.db` database.
- History reads and writes require a currently registered dataset and exact incident/result dataset ownership.
- Unique run IDs make persistence idempotent across Streamlit reruns; each saved run has one transactional activity event.
- Opening an old investigation displays the saved snapshot and never reruns analysis.
- The Central Intelligence Hub combines persisted investigation activity with evidence-ingestion timestamps; detailed history remains scoped to the active dataset.
- The history database has its own schema version and does not change the schema-v3 evidence search index or incidents CSV.

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

### Cross-Incident Comparison & Recurrence Analysis
- **Deterministic Symptom Grouping:** Groups exact or equivalent normalized symptom profiles (e.g. compound phrases, equivalent adjectives like `elevated` vs `high`) across incident records.
- **Broad Keyword Isolation:** Does not conflate incidents simply because they share a broad word (e.g. `bearing temperature` vs `bearing noise` remain distinct).
- **Recurrence Scope Classification:** Explicitly distinguishes:
  - `SAME_MACHINE`: Recurrence on the exact same known asset.
  - `CROSS_MACHINE`: Recurrence spanning two or more distinct assets.
  - `MIXED` / `UNDETERMINED`: Safely isolates records where machine identifiers are missing.
- **Timeline-Ready Serialization:** Formats incident timestamps for future timeline displays while preserving raw values and flagging invalid/missing dates.
- **Strict Limitation Declarations:** Never infers root causes from recurrence alone and explicitly logs data gaps (missing machine IDs, unparseable dates, missing maintenance actions).

### Incident Timeline & Temporal Sequencing
- **Chronological Ordering:** Deterministically sequences incidents by verified ISO dates (`YYYY-MM-DD`). Ties on identical dates are broken stably by incident ID.
- **Missing / Invalid Date Segregation:** Isolates unverified or missing date records in an "Incidents with Missing or Unverified Dates" section, preserving raw values without guessing.
- **Provenance & Field Preservation:** Completely retains incident IDs, machine IDs, original symptom descriptions, maintenance actions, and recorded outcomes.
- **Safety Boundary:** Explicitly notes that chronological proximity does not establish causation or verified root cause.

### Cross-Incident Evidence Linking
- **Symptom-Group Grounding:** Connects multi-incident symptom groups to relevant local evidence passages using bounded FTS5 queries derived from canonical symptoms and actions.
- **Substantive Validation:** Filters out non-substantive passages (search-term lists, indexing headers, fragments) so they never count as substantive link support.
- **Deduplication & Provenance:** Consolidates identical or near-duplicate passages across document uploads without erasing source document IDs, filenames, or chunk indices.
- **Guidance vs Equipment Context:** Distinguishes machine-specific inspection records (`EXACT_MACHINE`) from general maintenance guides (`GENERAL_REFERENCE`) and flags illustrative test readings with caution notices.
- **Causal Safety Boundary:** Explicitly establishes that evidence links provide technical context and mechanism explanations, but do not prove that recurring symptoms across different machines share an identical root cause.

### Dataset Workspaces & Scope
- The Central Intelligence Hub is project-wide; the Dataset Workspace, incident management, evidence retrieval, fault analysis (including evidence verification), and cross-incident analysis use the selected dataset.
- Evidence Search results are dataset-scoped. Index statistics, index synchronization/rebuild actions, and the separately labeled project-wide document catalog cover the full local project.
- Legacy incident CSVs without a `dataset_id` are treated as belonging to `DATASET-DEFAULT` in memory. Loading them does not rewrite the CSV.
- Documents with no dataset ID remain unassigned and are not counted in any workspace. The Hub distinguishes these from documents referencing an unregistered dataset ID.
- Associating unassigned legacy documents with `DATASET-DEFAULT` requires an explicit confirmation and action in the Hub. This migration is idempotent and does not change documents with existing dataset IDs or assign records to a newly created dataset.
- Sidebar navigation is authoritative in session state; page shortcuts update that state through callbacks, and the sidebar radio is synchronized before its widget is created.

### M7.4.1 Dataset Ownership & Integrity
- `datasets.json` is the authoritative registry of valid dataset IDs. Dataset IDs are immutable after creation; dataset deletion is not supported.
- The incident CSV owns incident `dataset_id` values. A legacy CSV with no dataset column receives `DATASET-DEFAULT` in memory; optional registry validation rejects explicit missing or unregistered references without rewriting the file.
- `catalog.json` owns document dataset IDs. Missing IDs and non-empty unregistered IDs are reported separately and remain unchanged unless the existing explicit legacy migration is used for missing IDs.
- `indexed_documents.dataset_id` and `evidence_chunks.dataset_id` are synchronized ownership snapshots, not independent sources of ownership. Indexing requires an `EvidenceStore` and validates the canonical catalog owner; metadata-only indexing is rejected. `SearchIndex.validate_integrity()` reports missing documents, dangling IDs, owner mismatches, and chunk/FTS reference inconsistencies.
- Index rebuild refuses to clear an index while ownership or index-integrity issues remain. Index synchronization reports ownership errors and does not assign missing document owners.
- The confirmed ownership repair is limited to the six approved processed document IDs and `DATASET-DEFAULT`. Preview inspects catalog, index-document, relational-chunk, registry, and FTS state; apply requires a fresh ready preview and does not modify evidence files or FTS content.
- Catalog JSON and SQLite cannot share a single atomic transaction. Repair therefore writes a durable journal at `metadata/ownership-repair/<repair-id>.json` before changing either store, then applies catalog and SQLite changes in separate recoverable stages and verifies all affected layers before marking the journal `verified`. If interrupted, call `EvidenceStore.recover_document_ownership_repair(repair_id, index, action="resume")` to safely complete it or `action="rollback"` to restore the recorded ownership values. Recovery only accepts the six approved IDs and refuses unexpected intervening changes. Keep the journal until the operation has been reviewed.
- SQLite schema versions are recorded in both `PRAGMA user_version` and `schema_migrations`: v1 is the base index, v2 adds nullable ownership columns, and v3 removes SQL-level ownership defaults without assigning or changing any values. Existing databases are backed up before migration.

### M7.4.2 Dataset-Scoped Evidence Retrieval
- Evidence retrieval requires an explicitly supplied dataset ID registered in `datasets.json`; missing, blank, and unregistered IDs fail closed.
- FTS5 search is constrained by both the indexed-document and evidence-chunk dataset snapshots, then checked against the catalog's authoritative document owner and provenance.
- Dataset-scoped integrity validation blocks retrieval when relevant catalog, index, chunk, or FTS records disagree. It reports diagnostics without migrating, repairing, or rebuilding data.
- Evidence Search displays the active dataset and its integrity status. Index statistics and maintenance actions remain explicitly project-wide.

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

### 4. Optional AI commentary setup

GhostFault's evidence retrieval, analysis, verification, and investigation history work locally
without an AI provider. AI commentary is an optional, explicitly enabled advisor and is never
treated as evidence or saved in investigation history.

Install the optional OpenAI-compatible client only if you intend to use AI commentary:

```powershell
python -m pip install -r requirements-ai.txt
```

Configure the gateway endpoint, API key, and model outside source control. For example, in
PowerShell for the current session:

```powershell
$env:OMNIROUTE_BASE_URL = "http://localhost:20128/v1"
$env:OMNIROUTE_API_KEY = "your-local-gateway-key"
$env:OMNIROUTE_MODEL = "your-configured-model"
```

Alternatively, put these settings in the ignored `.streamlit/secrets.toml` file:

```toml
OMNIROUTE_BASE_URL = "http://localhost:20128/v1"
OMNIROUTE_API_KEY = "your-local-gateway-key"
OMNIROUTE_MODEL = "your-configured-model"
```

Environment variables take precedence over Streamlit secrets. The app does not contact the
gateway until AI commentary is enabled in the analysis view and the request button is pressed.
That request sends only the selected incident's symptom, maintenance action, and a bounded summary
of up to three deterministic hypotheses; it does not send evidence passages or dataset identifiers.
Do not enable it unless sharing that incident summary with the configured provider is permitted.
AI responses are unverified commentary and cannot verify evidence, change evidence strength or case
status, or alter saved investigation snapshots. M9.2 does not perform web search.

### 5. Launch the Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Synthetic Data & Safety Disclaimer

- Incident records in `data/sample/incidents.csv` are **fictional and synthetic**.
- Local-first deterministic analysis does not require external AI APIs, cloud vector databases, or neural embeddings. Optional AI commentary can be enabled separately as described above.
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
- [x] Cross-incident comparison and recurrence analysis engine with timeline-ready serialization
- [x] Incident Timeline with chronological ordering, missing-date segregation, and temporal metrics
- [x] Cross-incident evidence linking with substantive passage validation and causal safety boundaries
- [x] Integrated cross-incident dashboard with symptom grouping, evidence links, and safety disclaimers
- [x] Persistent dataset registry, active-dataset selection, project hub, and isolated workspace summaries
- [x] Callback-safe navigation and explicit, idempotent legacy evidence-association migration
- [x] 224 automated tests across incidents, ingestion, retrieval, analysis, dataset ownership, comparison, timelines, and evidence linking

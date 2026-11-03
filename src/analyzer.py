"""Evidence-grounded fault analysis engine for GhostFault (Milestone 4.1.2).

Substantive evidence relevance validation, passage-quality classification,
deterministic support-reason generation, and accurate pipeline count derivation.
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple, Iterable
import pandas as pd

from src.models import (
    SearchResult,
    EvidenceReference,
    InvestigativeHypothesis,
    FaultAnalysisResult,
    AnalysisStatus,
    PassageQuality,
    EvidenceContextType,
    MachineScope,
    VerificationStatus,
    EvidenceStrength,
    EvidenceCounts,
    DatasetOwnershipError,
    current_iso_timestamp,
)
from src.retriever import EvidenceRetriever

FAILURE_PATTERNS = [
    {
        "pattern_id": "BEARING_DEGRADATION",
        "keywords": ["bearing", "acoustic", "pitch", "vibration", "roller"],
        "min_keyword_matches": 2,
        "title": "Drive-End Bearing Wear or Inadequate Lubrication",
        "description": (
            "Elevated temperature accompanied by abnormal acoustic pitch or high-frequency harmonic vibration "
            "is consistent with rolling element surface fatigue, cage distress, or lubricant breakdown."
        ),
        "mechanism_terms": ["friction", "lubrication", "fatigue", "wear", "torque", "fit", "housing"],
        "suggested_checks": [
            "Perform vibration spectrum analysis (focus on bearing fault frequencies: BPFO, BPFI, BSF, FTF).",
            "Inspect bearing grease/oil for thermal oxidation, metallic particulate, or water contamination.",
            "Measure shaft runout and verify housing bolt torque according to manufacturer specifications.",
        ],
    },
    {
        "pattern_id": "COUPLING_MISALIGNMENT",
        "keywords": ["coupling", "harmonic", "vibration", "alignment", "elastomer", "shaft"],
        "min_keyword_matches": 2,
        "title": "Shaft Misalignment or Mechanical Coupling Degradation",
        "description": (
            "Harmonic vibration under operational load frequently indicates angular or parallel misalignment "
            "between drive and driven shafts, or degraded elastomeric coupling elements."
        ),
        "mechanism_terms": ["alignment", "misalignment", "elastomer", "coupling", "spider", "backlash"],
        "suggested_checks": [
            "Conduct laser alignment check across drive and driven shafts under both cold and operating thermal equilibrium.",
            "Inspect flexible coupling spider/elastomer insert for cracking, extrusion, or excessive backlash.",
            "Verify foundation bolt torque and inspect for soft-foot condition.",
        ],
    },
    {
        "pattern_id": "HYDRAULIC_RESTRICTION",
        "keywords": ["hydraulic", "pressure", "filter", "cavitation", "circuit", "pump"],
        "min_keyword_matches": 2,
        "title": "Hydraulic Circuit Restriction or Return Filter Clogging",
        "description": (
            "Intermittent or sustained pressure drops across hydraulic circuits correlate with clogged return/pressure "
            "filter cartridges, pump suction cavitation, or aerated hydraulic fluid."
        ),
        "mechanism_terms": ["clogged", "filter", "cavitation", "restriction", "aerated", "relief"],
        "suggested_checks": [
            "Inspect differential pressure indicator across primary and return line filter elements.",
            "Sample hydraulic oil to verify ISO 4406 particulate cleanliness rating and viscosity grade.",
            "Bleed air from high-point circuit vents and check reservoir suction strainer for debris.",
        ],
    },
    {
        "pattern_id": "THERMAL_CHILLER_FAULT",
        "keywords": ["thermal", "trip", "chiller", "cooling", "coolant", "spindle", "glycol"],
        "min_keyword_matches": 2,
        "title": "Cooling Loop Heat Exchanger Fouling or Flow Restriction",
        "description": (
            "Thermal trip of high-torque drive units during extended milling cycles typically arises from degraded "
            "heat exchange efficiency, scale buildup in closed-loop chillers, or depleted heat transfer fluid."
        ),
        "mechanism_terms": ["fouling", "scale", "flow", "glycol", "chiller", "heat", "exchanger"],
        "suggested_checks": [
            "Verify closed-loop coolant supply and return temperature differential (delta T).",
            "Inspect heat exchanger coils for scaling, bio-fouling, or sediment accumulation.",
            "Check glycol refractometer reading and flow rate through the spindle jacket.",
        ],
    },
    {
        "pattern_id": "FEEDBACK_POSITIONING_DRIFT",
        "keywords": ["encoder", "positioning", "scale", "drift", "optical", "datum", "repeatability"],
        "min_keyword_matches": 2,
        "title": "Optical Scale Contamination or Feedback Calibration Drift",
        "description": (
            "Positioning error or datum drift without severe mechanical grinding points to contamination on optical glass "
            "scales, scanning head misalignment, or loose reference scale mounting."
        ),
        "mechanism_terms": ["contamination", "optical", "scale", "scanning", "drift", "glass", "grime"],
        "suggested_checks": [
            "Inspect linear glass scale and scanning head for oil mist, coolant splashes, or fine debris.",
            "Clean optical surfaces using recommended laboratory-grade solvent (e.g., isopropyl alcohol).",
            "Execute laser interferometer or dial indicator repeatability check across full axis travel.",
        ],
    },
    {
        "pattern_id": "PNEUMATIC_VALVE_SEAL",
        "keywords": ["pneumatic", "actuator", "stroke", "solenoid", "wiper", "seal"],
        "min_keyword_matches": 2,
        "title": "Pneumatic Actuator Seal Wear or Valve Response Delay",
        "description": (
            "Delayed stroke extension or sluggish cylinder cycle time indicates degraded rod wiper/piston seals, "
            "insufficient system supply pressure, or sticking proportional solenoid spool."
        ),
        "mechanism_terms": ["wiper", "seal", "wear", "solenoid", "spool", "leakage", "friction"],
        "suggested_checks": [
            "Perform cylinder bypass leakage test to evaluate internal piston seal integrity.",
            "Verify dynamic supply pressure at the valve manifold during simultaneous stroke actuation.",
            "Inspect proportional valve spool for particulate varnishing or gummy oil residue.",
        ],
    },
]


def classify_passage_quality(text: str) -> Tuple[PassageQuality, Optional[str]]:
    """
    Deterministic passage-quality classifier identifying substantive technical content,
    search terms/indexing metadata, header/footer metadata, or empty text.
    Returns: (quality_enum, exclusion_reason_if_not_substantive)
    """
    if not text or not text.strip():
        return PassageQuality.EMPTY_OR_LOW_SUBSTANCE, "Passage contains no readable text."

    cleaned = text.strip()
    lower = cleaned.lower()

    # Detect search terms / indexing keyword lists
    if lower.startswith("search terms:") or lower.startswith("keywords:") or lower.startswith("indexing terms:"):
        return (
            PassageQuality.SEARCH_TERMS_METADATA,
            "Passage consists exclusively of indexing search terms/keywords, not technical or operational evidence.",
        )

    # Detect pure document header/title metadata
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) <= 4:
        lower_lines = [l.lower() for l in lines]
        has_id_or_type = any(
            l.startswith("document id:") or l.startswith("record type:") or l.startswith("equipment:")
            or l.startswith("title:") or l.startswith("author:") or l.startswith("version:")
            for l in lower_lines
        )
        has_header_indicator = any(
            "evidence record" in l or "manual" in l or "report" in l or "maintenance" in l
            for l in lower_lines
        )
        if has_id_or_type or (len(lines) <= 3 and all(":" in line for line in lines)):
            if has_header_indicator or has_id_or_type:
                return (
                    PassageQuality.HEADER_FOOTER_METADATA,
                    "Passage contains only administrative document header metadata.",
                )

    # Detect very short token lists without complete sentence structure
    if len(cleaned) < 40 and not any(punct in cleaned for punct in {".", "!", "?"}):
        return (
            PassageQuality.EMPTY_OR_LOW_SUBSTANCE,
            "Passage is too brief or fragmentary to convey substantive diagnostic facts.",
        )

    return PassageQuality.SUBSTANTIVE_TECHNICAL, None


def extract_sentence_bounded_snippet(text: str, max_length: int = 400) -> Tuple[str, bool]:
    """
    Extract a readable snippet preferring complete sentence boundaries.
    Never fabricates text or alters source characters.
    Returns: (snippet_text, is_excerpt)
    """
    if not text:
        return "", False

    cleaned_text = text.strip()

    starts_mid_sentence = False
    first_char = cleaned_text[0] if cleaned_text else ""
    if first_char.islower() or first_char in {",", ";", ")", "]", "}", "-", "/"}:
        starts_mid_sentence = True

    start_pos = 0
    if starts_mid_sentence:
        sentence_starter = re.search(r"(?:[.!?]\s+|\n\s*\n|\n)([A-Z0-9])", cleaned_text[:120])
        if sentence_starter:
            start_pos = sentence_starter.start(1)
            starts_mid_sentence = False

    work_text = cleaned_text[start_pos:].strip()

    if len(work_text) <= max_length:
        return work_text, starts_mid_sentence

    min_cut = min(100, max_length // 2)
    candidate_window = work_text[min_cut:max_length]

    para_breaks = [m.start() for m in re.finditer(r"\n\s*\n", candidate_window)]
    if para_breaks:
        cut_point = min_cut + para_breaks[-1]
        return work_text[:cut_point].strip(), starts_mid_sentence

    sentence_breaks = [m.start() for m in re.finditer(r"[.!?](?:\s|$)", candidate_window)]
    if sentence_breaks:
        cut_point = min_cut + sentence_breaks[-1] + 1
        return work_text[:cut_point].strip(), starts_mid_sentence

    newline_breaks = [m.start() for m in re.finditer(r"\n", candidate_window)]
    if newline_breaks:
        cut_point = min_cut + newline_breaks[-1]
        return work_text[:cut_point].strip(), True

    word_breaks = [m.start() for m in re.finditer(r"\s", candidate_window)]
    if word_breaks:
        cut_point = min_cut + word_breaks[-1]
        return work_text[:cut_point].strip() + "...", True

    return work_text[:max_length].strip() + "...", True


def normalize_text_for_similarity(text: str) -> str:
    """Normalize text for near-duplicate passage detection."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return " ".join(clean.split())


def classify_evidence_context(chunk_text: str) -> EvidenceContextType:
    """Classify numeric and technical context within a passage."""
    lower = chunk_text.lower()
    if "illustrative test" in lower or "example maintenance evidence" in lower or "synthetic test data" in lower:
        return EvidenceContextType.ILLUSTRATIVE_TEST_VALUE
    if "oem limit" in lower or "nominal operating limit" in lower or "manufacturer specification" in lower:
        return EvidenceContextType.VERIFIED_OEM_LIMIT
    return EvidenceContextType.GENERAL_REFERENCE


def generate_support_reason(
    scope: MachineScope,
    context_type: EvidenceContextType,
    chunk_text: str,
    pattern: Dict[str, Any],
    machine_id: str,
) -> str:
    """
    Generate a concise, deterministic explanation grounded in the actual passage,
    distinguishing direct incident observations, historical maintenance,
    general technical mechanisms, and illustrative context.
    """
    lower = chunk_text.lower()
    mechanism_terms = pattern.get("mechanism_terms", [])
    has_mechanism = any(term in lower for term in mechanism_terms)

    if scope == MachineScope.EXACT_MACHINE:
        prefix = f"Direct incident record for {machine_id}."
    else:
        prefix = "General reference document."

    if "maintenance history" in lower or "previous maintenance" in lower:
        rel = "Notes historical maintenance interventions and fastener retorquing procedures."
    elif has_mechanism:
        rel = (
            f"Discusses potential underlying mechanisms ({pattern['title']}) associated with observed operational distress."
        )
    elif "fault symptoms" in lower or "reported" in lower:
        rel = "Documents co-occurring thermal and acoustic symptoms under steady load."
    else:
        rel = "Provides technical context for operating limits and inspection criteria."

    if context_type == EvidenceContextType.ILLUSTRATIVE_TEST_VALUE:
        suffix = " (Contains illustrative test readings; not verified physical sensor data)."
    else:
        suffix = ""

    return f"{prefix} {rel}{suffix}"


def _validate_scoped_references(
    references: List[EvidenceReference],
    dataset_id: Optional[str],
    evidence_retriever: Optional[EvidenceRetriever],
    operation: str,
) -> None:
    if not dataset_id or not references:
        return
    if any(not ref.dataset_id or ref.dataset_id != dataset_id for ref in references):
        raise DatasetOwnershipError(
            f"{operation} rejected untagged or foreign evidence for active dataset "
            f"'{dataset_id}'."
        )
    if evidence_retriever is None:
        raise DatasetOwnershipError(
            f"{operation} cannot trust scoped evidence without canonical catalog validation."
        )
    evidence_retriever.validate_evidence_references(references, dataset_id)


def verify_hypothesis_evidence(
    hypothesis: InvestigativeHypothesis,
    incident_machine_id: Optional[str] = None,
    pattern: Optional[Dict[str, Any]] = None,
    dataset_id: Optional[str] = None,
    evidence_retriever: Optional[EvidenceRetriever] = None,
) -> Tuple[VerificationStatus, str]:
    """
    Deterministic evidence verification for a diagnostic hypothesis (Milestone 6.1).
    Evaluates retrieved evidence to return exactly one status:
    - CONTRADICTED: Directly relevant and sufficiently reliable evidence explicitly conflicts with the hypothesis.
    - SUPPORTED: Relevant evidence provides a clear, traceable reason to consider the hypothesis plausible.
    - INCONCLUSIVE: Evidence is missing, unrelated, ambiguous, or insufficient.
    """
    raw_supporting = hypothesis.supporting_evidence or []
    raw_contradictory = hypothesis.contradictory_evidence or []

    _validate_scoped_references(
        raw_supporting + raw_contradictory,
        dataset_id,
        evidence_retriever,
        "Verification",
    )
    supporting = raw_supporting
    contradictory = raw_contradictory

    # Rule 1: No retrieved evidence at all
    if not supporting and not contradictory:
        return (
            VerificationStatus.INCONCLUSIVE,
            "Inconclusive: No retrieved evidence passages available to substantiate or contradict this hypothesis.",
        )

    # Rule 2: Explicit contradiction check
    # Check if contradictory evidence contains substantive passages
    substantive_contra = [
        c for c in contradictory
        if getattr(c, "quality", PassageQuality.SUBSTANTIVE_TECHNICAL) == PassageQuality.SUBSTANTIVE_TECHNICAL
    ]

    has_direct_conflict = False
    contra_explanations: List[str] = []

    for c in substantive_contra:
        scope = getattr(c, "machine_scope", None)
        text_lower = (c.snippet or "").lower()

        # Check for cross-machine conflict
        if scope == MachineScope.CROSS_MACHINE_CONFLICT:
            other_m = getattr(c, "referenced_machine_id", None) or "different machine"
            has_direct_conflict = True
            contra_explanations.append(
                f"Passage explicitly documents failure on {other_m} rather than target machine, conflicting with direct applicability ({c.source_filename} Chunk #{c.chunk_index})."
            )
        # Check for explicit operational contradiction or nominal operation
        elif any(term in text_lower for term in [
            "within nominal", "operating normally", "no fault detected", "no defect found",
            "passed inspection", "normal tolerance", "nominal limit", "healthy operation"
        ]):
            has_direct_conflict = True
            contra_explanations.append(
                f"Passage indicates nominal operation or no defect found, directly conflicting with active fault hypothesis ({c.source_filename} Chunk #{c.chunk_index})."
            )

    if has_direct_conflict:
        explanation = "Contradicted: " + "; ".join(contra_explanations)
        return (VerificationStatus.CONTRADICTED, explanation)

    # Rule 3: Evaluate supporting evidence for genuine substantive corroboration
    substantive_supp = [
        s for s in supporting
        if getattr(s, "quality", PassageQuality.SUBSTANTIVE_TECHNICAL) == PassageQuality.SUBSTANTIVE_TECHNICAL
    ]

    # Incomplete provenance or zero substantive evidence
    valid_provenance_supp = [
        s for s in substantive_supp
        if s.document_id and s.chunk_id and s.chunk_index is not None and s.source_filename
    ]

    if not valid_provenance_supp:
        if supporting and not substantive_supp:
            return (
                VerificationStatus.INCONCLUSIVE,
                "Inconclusive: Retrieved passages consist exclusively of indexing metadata or low-substance excerpts.",
            )
        return (
            VerificationStatus.INCONCLUSIVE,
            "Inconclusive: Retrieved evidence lacks valid document/chunk provenance required for traceable verification.",
        )

    # Check for keyword overlap alone vs genuine evidential support
    # A passage is genuine support if:
    # 1. It is an exact machine match (scope == EXACT_MACHINE), OR
    # 2. It contains explicit mechanism/causal grounding related to the pattern, OR
    # 3. It contains direct diagnostic criteria/maintenance history
    qualifying_support: List[str] = []

    # Get mechanism terms from pattern if provided
    mechanism_terms = pattern.get("mechanism_terms", []) if pattern else []

    for s in valid_provenance_supp:
        snippet_lower = (s.snippet or "").lower()
        scope = getattr(s, "machine_scope", MachineScope.GENERAL_REFERENCE)
        ctx = getattr(s, "context_type", EvidenceContextType.GENERAL_REFERENCE)

        # Illustrative test values alone are ambiguous/insufficient for physical confirmation
        if ctx == EvidenceContextType.ILLUSTRATIVE_TEST_VALUE and len(valid_provenance_supp) == 1:
            continue

        # Check exact machine corroboration
        if scope == MachineScope.EXACT_MACHINE:
            m_label = s.referenced_machine_id or incident_machine_id or "target machine"
            qualifying_support.append(
                f"Direct incident/maintenance documentation for {m_label} ({s.source_filename} Chunk #{s.chunk_index})"
            )
        # Check mechanism terms match
        elif mechanism_terms and any(term in snippet_lower for term in mechanism_terms):
            qualifying_support.append(
                f"Corroborating technical mechanism documented in {s.source_filename} (Chunk #{s.chunk_index})"
            )
        # Check general diagnostic criteria or maintenance history
        elif any(phrase in snippet_lower for phrase in [
            "diagnostic guide", "indicates", "failure mode", "root cause", "maintenance history",
            "wear", "breakdown", "fatigue", "damage", "degraded", "clogged", "misalignment"
        ]):
            qualifying_support.append(
                f"Verified diagnostic guidance in {s.source_filename} (Chunk #{s.chunk_index})"
            )

    if qualifying_support:
        # At least one substantive passage provides traceable, genuine support
        unique_reasons = list(dict.fromkeys(qualifying_support))
        explanation = "Supported: " + "; ".join(unique_reasons) + "."
        return (VerificationStatus.SUPPORTED, explanation)

    # If only keyword overlap without mechanism or exact-machine grounding:
    return (
        VerificationStatus.INCONCLUSIVE,
        "Inconclusive: Evidence matches search keywords but lacks specific failure mechanism or machine-specific corroboration.",
    )


def evaluate_evidence_strength(
    hypothesis: InvestigativeHypothesis,
    verification_status: VerificationStatus,
    incident_machine_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    evidence_retriever: Optional[EvidenceRetriever] = None,
) -> Tuple[EvidenceStrength, str]:
    """
    Qualitative evidence strength assessment for a diagnostic hypothesis (Milestone 6.2).
    Returns (EvidenceStrength, explanation).
    - STRONG: Multiple independent, directly relevant evidence passages support the same hypothesis.
    - LIMITED: Some relevant evidence exists, but it is incomplete or weakly corroborated.
    - INSUFFICIENT: Available evidence cannot reasonably establish or contradict the hypothesis.

    Rules:
    1. Qualitative assessment, not a probability or percentage.
    2. Repeated or duplicate passages from the same source document do NOT count as independent corroboration.
    3. Preserves verification status; does not override SUPPORTED, CONTRADICTED, or INCONCLUSIVE.
    4. If evidence is contradictory or provenance is inadequate, explains that limitation.
    """
    supporting = hypothesis.supporting_evidence or []
    contradictory = hypothesis.contradictory_evidence or []
    _validate_scoped_references(
        supporting + contradictory,
        dataset_id,
        evidence_retriever,
        "Evidence strength",
    )

    # Case 1: Inadequate provenance or contradictory evidence limitations
    if verification_status == VerificationStatus.CONTRADICTED:
        return (
            EvidenceStrength.LIMITED,
            "Contradictory evidence detected; findings are limited by conflicting records or cross-machine applicability.",
        )

    if verification_status == VerificationStatus.INCONCLUSIVE:
        # Check if missing or invalid provenance was the cause
        has_invalid_provenance = any(
            not s.document_id or not s.chunk_id or s.chunk_index is None
            for s in supporting
        )
        if has_invalid_provenance:
            return (
                EvidenceStrength.INSUFFICIENT,
                "Evidence strength is insufficient due to incomplete or unverifiable source provenance.",
            )
        return (
            EvidenceStrength.INSUFFICIENT,
            "Available records provide insufficient evidence to reasonably establish or contradict this hypothesis due to absent or non-diagnostic data.",
        )

    # Case 2: Verification status is SUPPORTED
    # Filter to substantive references with valid provenance
    substantive_supp = [
        s for s in supporting
        if getattr(s, "quality", PassageQuality.SUBSTANTIVE_TECHNICAL) == PassageQuality.SUBSTANTIVE_TECHNICAL
        and not getattr(s, "is_duplicate", False)
        and s.document_id and s.chunk_id
    ]

    # Count independent sources (distinct document IDs)
    distinct_doc_ids = {s.document_id for s in substantive_supp}
    distinct_filenames = {s.source_filename for s in substantive_supp if s.source_filename}

    # Also count independent substantive chunks across distinct documents or distant sections
    if len(distinct_doc_ids) >= 2 or len(substantive_supp) >= 2 and len(distinct_filenames) >= 2:
        return (
            EvidenceStrength.STRONG,
            f"Multiple independent, directly relevant evidence passages ({len(distinct_doc_ids)} distinct sources) corroborate this hypothesis without reliance on duplicated passages.",
        )
    elif len(substantive_supp) >= 1:
        # Check if it was multiple chunks from the same single document
        if len(supporting) > 1 and len(distinct_doc_ids) == 1:
            return (
                EvidenceStrength.LIMITED,
                "Corroborated by a single source document; repeated or adjacent chunks from the same document are not treated as independent multi-source verification.",
            )
        return (
            EvidenceStrength.LIMITED,
            "Supported by relevant evidence from a single documented source, but lacks independent multi-source corroboration.",
        )

    return (
        EvidenceStrength.INSUFFICIENT,
        "Available evidence provides insufficient technical basis to substantiate this hypothesis.",
    )


def derive_missing_evidence_guidance(
    hypothesis: InvestigativeHypothesis,
    incident_record: Dict[str, Any],
    pattern: Optional[Dict[str, Any]] = None,
    dataset_id: Optional[str] = None,
    evidence_retriever: Optional[EvidenceRetriever] = None,
) -> str:
    """
    Derives cautious, grounded missing-evidence guidance for a hypothesis (Milestone 6.2).
    Identifies specific evidence gaps that would help further investigation only when justified
    by the incident symptoms and available evidence.

    Rules:
    1. Distinguishes 'not found in the available records' from 'does not exist'.
    2. Only suggests gaps justified by the actual hypothesis and incident facts.
    3. If no reliable suggestion can be derived, returns 'No specific evidence gap identified.'
    """
    supporting = hypothesis.supporting_evidence or []
    _validate_scoped_references(
        supporting,
        dataset_id,
        evidence_retriever,
        "Missing-evidence guidance",
    )

    symptom = incident_record.get("symptom", "").lower()
    maintenance = incident_record.get("maintenance_action", "").lower()
    combined = f"{symptom} {maintenance}"

    # Analyze based on the hypothesis title / pattern
    title = hypothesis.title.lower()

    # Look at existing supporting snippets to see what is already documented
    existing_snippets = " ".join([s.snippet.lower() for s in supporting])

    # 1. Bearing Degradation
    if "bearing" in title:
        # Check if temperature measurement over time is missing
        if "temperature" in combined or "bearing" in combined:
            if "trend" not in existing_snippets and "temperature log" not in existing_snippets and "over time" not in existing_snippets:
                return (
                    "A continuous bearing temperature measurement log over time was not found in the available records; "
                    "time-series thermal data would help confirm steady-state heating versus friction spikes."
                )
        # Check if physical inspection or lubricant lab analysis is documented
        if "grease" not in existing_snippets and "lubricant" not in existing_snippets:
            return (
                "Lubrication condition and physical roller element inspection results were not found in the available records; "
                "lubricant particulate analysis would assist in confirming mechanical wear."
            )

    # 2. Coupling Misalignment
    if "coupling" in title or "misalignment" in title:
        if "laser alignment" not in existing_snippets and "dial indicator" not in existing_snippets:
            return (
                "Shaft laser alignment measurements and elastomer insert physical inspection records were not found in the available records; "
                "pre- and post-alignment readings would verify angular or parallel offset."
            )

    # 3. Hydraulic Restriction
    if "hydraulic" in title or "filter" in title:
        if "differential pressure" not in existing_snippets and "particle count" not in existing_snippets:
            return (
                "Differential pressure (delta-P) indicator logs across the return filter and ISO 4406 fluid cleanliness ratings were not found in the available records; "
                "continuous circuit pressure trends would establish restriction onset."
            )

    # 4. Thermal Chiller Fault
    if "chiller" in title or "cooling" in title or "heat exchanger" in title:
        if "delta t" not in existing_snippets and "flow rate" not in existing_snippets:
            return (
                "Closed-loop coolant supply and return temperature differential (delta-T) and spindle jacket flow rate measurements were not found in the available records; "
                "coolant thermal logs would confirm heat exchanger fouling."
            )

    # 5. Optical Scale / Positioning Drift
    if "scale" in title or "optical" in title or "positioning" in title:
        if "repeatability" not in existing_snippets and "interferometer" not in existing_snippets:
            return (
                "Multi-point positioning repeatability measurement records and optical scanning head inspection logs were not found in the available records; "
                "laser calibration data would distinguish mechanical backlash from scale contamination."
            )

    # 6. Pneumatic Valve / Actuator
    if "pneumatic" in title or "actuator" in title or "valve" in title:
        if "supply pressure" not in existing_snippets and "bypass leakage" not in existing_snippets:
            return (
                "Dynamic pneumatic supply pressure traces during stroke actuation and internal cylinder leakage test data were not found in the available records; "
                "pressure response logs would isolate valve spool sticking from seal blow-by."
            )

    return "No specific evidence gap identified."


class FaultAnalysisEngine:
    """
    Deterministic fault analysis engine with substantive relevance validation.
    """

    def __init__(self, retriever: EvidenceRetriever):
        self.retriever = retriever

    @staticmethod
    def extract_search_terms(incident_record: Dict[str, Any]) -> List[str]:
        """Extract meaningful technical keywords from incident symptom and maintenance fields."""
        stopwords = {
            "and", "the", "with", "from", "for", "during", "under", "over", "into",
            "after", "before", "across", "within", "without", "resolved", "verified",
            "taken", "reported", "observed", "high", "recurrent", "primary", "secondary"
        }

        combined_text = f"{incident_record.get('symptom', '')} {incident_record.get('maintenance_action', '')}"
        tokens = re.findall(r"[a-zA-Z0-9_\-]+", combined_text.lower())
        meaningful = [t for t in tokens if len(t) >= 4 and t not in stopwords]
        return list(dict.fromkeys(meaningful))

    def retrieve_incident_evidence(
        self,
        incident_record: Dict[str, Any],
        max_results: int = 10,
        dataset_id: Optional[str] = None,
        valid_dataset_ids: Optional[Iterable[str]] = None,
    ) -> List[SearchResult]:
        """Build bounded queries and retrieve matching passages scoped to dataset, preserving raw query results."""
        target_ds_id = self.retriever.require_dataset_scope(dataset_id, valid_dataset_ids)
        incident_ds_id = incident_record.get("dataset_id")
        if incident_ds_id is None and target_ds_id == "DATASET-DEFAULT":
            incident_ds_id = target_ds_id
        if incident_ds_id != target_ds_id:
            raise DatasetOwnershipError(
                f"Incident '{incident_record.get('incident_id', 'UNKNOWN_INCIDENT')}' does not "
                f"belong to active dataset '{target_ds_id}'."
            )

        search_terms = self.extract_search_terms(incident_record)
        if not search_terms:
            return []

        query_str = " ".join(search_terms[:5])
        results = self.retriever.search(
            query=query_str,
            limit=max_results,
            dataset_id=target_ds_id,
            valid_dataset_ids=valid_dataset_ids,
        )

        if not results and len(search_terms) > 0:
            seen_chunk_ids: Set[str] = set()
            fallback_results: List[SearchResult] = []
            for term in search_terms[:4]:
                sub_results = self.retriever.search(
                    query=term,
                    limit=3,
                    dataset_id=target_ds_id,
                    valid_dataset_ids=valid_dataset_ids,
                )
                for res in sub_results:
                    if res.chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(res.chunk.chunk_id)
                        fallback_results.append(res)
            results = fallback_results

        return results[:max_results]

    @staticmethod
    def inspect_machine_context(
        incident_machine_id: str,
        chunk_text: str,
    ) -> Tuple[MachineScope, Optional[str]]:
        """
        Check if an evidence passage refers to the specific incident machine,
        a different machine unit (conflict), or has no specific machine ID (general).
        """
        machine_mentions = re.findall(r"\bM-\d{3,}\b", chunk_text)
        if not machine_mentions:
            return MachineScope.GENERAL_REFERENCE, None

        if incident_machine_id in machine_mentions:
            conflicts = [m for m in machine_mentions if m != incident_machine_id]
            if conflicts:
                return MachineScope.CROSS_MACHINE_CONFLICT, conflicts[0]
            return MachineScope.EXACT_MACHINE, incident_machine_id

        return MachineScope.CROSS_MACHINE_CONFLICT, machine_mentions[0]

    @classmethod
    def deduplicate_evidence_references(
        cls,
        references: List[EvidenceReference],
    ) -> List[EvidenceReference]:
        """Group duplicate and near-duplicate passages under primary reference."""
        if not references:
            return []

        primary_refs: List[EvidenceReference] = []
        normalized_seen: Dict[str, EvidenceReference] = {}

        for ref in references:
            norm = normalize_text_for_similarity(ref.snippet)
            matched_primary: Optional[EvidenceReference] = None
            if norm in normalized_seen:
                matched_primary = normalized_seen[norm]
            else:
                for existing_norm, existing_ref in normalized_seen.items():
                    min_len = min(len(norm), len(existing_norm))
                    if min_len > 60:
                        shared_prefix = 0
                        for c1, c2 in zip(norm, existing_norm):
                            if c1 == c2:
                                shared_prefix += 1
                            else:
                                break
                        if shared_prefix / min_len > 0.80 or norm in existing_norm or existing_norm in norm:
                            matched_primary = existing_ref
                            break

            if matched_primary is not None:
                ref.is_duplicate = True
                ref.duplicate_of_chunk_id = matched_primary.chunk_id
                copy_info = {
                    "document_id": ref.document_id,
                    "source_filename": ref.source_filename,
                    "chunk_id": ref.chunk_id,
                    "chunk_index": str(ref.chunk_index),
                }
                if matched_primary.duplicate_source_copies is None:
                    matched_primary.duplicate_source_copies = []
                matched_primary.duplicate_source_copies.append(copy_info)
            else:
                normalized_seen[norm] = ref
                primary_refs.append(ref)

        return primary_refs

    def analyze_incident(
        self,
        incident_record: Dict[str, Any],
        evidence_results: Optional[List[SearchResult]] = None,
        dataset_id: Optional[str] = None,
        valid_dataset_ids: Optional[Iterable[str]] = None,
    ) -> FaultAnalysisResult:
        """
        Perform evidence-grounded fault analysis with substantive relevance validation and dataset isolation.
        """
        incident_id = incident_record.get("incident_id", "UNKNOWN_INCIDENT")
        machine_id = incident_record.get("machine_id", "UNKNOWN_MACHINE")
        symptom = incident_record.get("symptom", "")
        maintenance_action = incident_record.get("maintenance_action", "")
        target_ds_id = dataset_id if dataset_id is not None else incident_record.get("dataset_id")

        timestamp = current_iso_timestamp()

        if not symptom or not machine_id:
            return FaultAnalysisResult(
                incident_id=incident_id,
                incident_facts=incident_record,
                retrieved_evidence=[],
                hypotheses=[],
                missing_information=["Incident record lacks symptom or machine identifier."],
                limitations=["Cannot perform analysis without documented symptom and machine ID."],
                status=AnalysisStatus.FAILED,
                analyzed_at=timestamp,
                evidence_counts=EvidenceCounts(),
                excluded_evidence=[],
                error_message="Incident record is missing required symptom or machine_id.",
            )

        target_ds_id = self.retriever.require_dataset_scope(target_ds_id, valid_dataset_ids)
        incident_ds_id = incident_record.get("dataset_id")
        if incident_ds_id is None and target_ds_id == "DATASET-DEFAULT":
            incident_ds_id = target_ds_id
        if incident_ds_id != target_ds_id:
            raise DatasetOwnershipError(
                f"Incident '{incident_id}' belongs to dataset '{incident_ds_id}', not active "
                f"dataset '{target_ds_id}'."
            )

        if evidence_results is None:
            raw_evidence = self.retrieve_incident_evidence(
                incident_record,
                dataset_id=target_ds_id,
                valid_dataset_ids=valid_dataset_ids,
            )
        else:
            raw_evidence = self.retriever.validate_evidence_results(
                evidence_results,
                target_ds_id,
            )

        raw_count = len(raw_evidence)

        # Step 1: Deduplicate raw search results by chunk_id
        unique_results: List[SearchResult] = []
        seen_chunk_ids: Set[str] = set()
        for r in raw_evidence:
            if r.chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(r.chunk.chunk_id)
                unique_results.append(r)

        unique_count = len(unique_results)

        if not unique_results:
            return FaultAnalysisResult(
                incident_id=incident_id,
                incident_facts=incident_record,
                retrieved_evidence=[],
                hypotheses=[],
                missing_information=[
                    "No technical documentation or historical records matching these symptoms exist in the index.",
                    "Missing detailed sensor telemetry (temperature trends, vibration FFT spectra, pressure logs).",
                ],
                limitations=[
                    "Analysis inconclusive due to absence of indexed evidence documents.",
                    "Deterministic rules require grounded document passages to substantiate diagnostic hypotheses.",
                ],
                status=AnalysisStatus.NO_EVIDENCE,
                analyzed_at=timestamp,
                evidence_counts=EvidenceCounts(raw_retrieved_count=0, unique_deduplicated_count=0),
                excluded_evidence=[],
            )

        # Step 2: Classify quality of all unique passages
        substantive_results: List[SearchResult] = []
        excluded_refs: List[EvidenceReference] = []

        for r in unique_results:
            quality, exc_reason = classify_passage_quality(r.chunk.text)
            bounded_snippet, is_excerpt = extract_sentence_bounded_snippet(r.chunk.text)
            scope, conflicting_machine = self.inspect_machine_context(machine_id, r.chunk.text)
            ctx_type = classify_evidence_context(r.chunk.text)

            if quality == PassageQuality.SUBSTANTIVE_TECHNICAL:
                substantive_results.append(r)
            else:
                excluded_refs.append(
                    EvidenceReference(
                        document_id=r.chunk.document_id,
                        source_filename=r.chunk.source_filename,
                        chunk_id=r.chunk.chunk_id,
                        chunk_index=r.chunk.chunk_index,
                        snippet=bounded_snippet,
                        relevance_reason=f"Excluded from hypothesis grounding: {exc_reason}",
                        machine_scope=scope,
                        context_type=ctx_type,
                        quality=quality,
                        referenced_machine_id=conflicting_machine,
                        is_excerpt=is_excerpt,
                        exclusion_reason=exc_reason,
                        dataset_id=r.chunk.dataset_id,
                    )
                )

        substantive_count = len(substantive_results)
        excluded_count = len(excluded_refs)

        hypotheses: List[InvestigativeHypothesis] = []
        combined_symptom_text = f"{symptom} {maintenance_action}".lower()

        # Step 3: Match only substantive passages against failure patterns
        for pattern in FAILURE_PATTERNS:
            pattern_keywords = pattern["keywords"]
            symptom_matches = [kw for kw in pattern_keywords if kw in combined_symptom_text]
            if len(symptom_matches) < pattern["min_keyword_matches"]:
                continue

            supporting_raw: List[EvidenceReference] = []
            contradictory_raw: List[EvidenceReference] = []

            for res in substantive_results:
                chunk = res.chunk
                chunk_lower = chunk.text.lower()
                chunk_matches = [kw for kw in pattern_keywords if kw in chunk_lower]

                shared_keywords = set(symptom_matches).intersection(set(chunk_matches))
                if shared_keywords:
                    scope, conflicting_machine = self.inspect_machine_context(machine_id, chunk.text)
                    context_type = classify_evidence_context(chunk.text)
                    bounded_snippet, is_excerpt = extract_sentence_bounded_snippet(chunk.text)

                    if scope == MachineScope.CROSS_MACHINE_CONFLICT:
                        reason = (
                            f"Cross-machine conflict: Passage discusses {', '.join(shared_keywords)} but explicitly references "
                            f"a different machine ({conflicting_machine}). Must not be treated as confirmed "
                            f"evidence for {machine_id}."
                        )
                        contradictory_raw.append(
                            EvidenceReference(
                                document_id=chunk.document_id,
                                source_filename=chunk.source_filename,
                                chunk_id=chunk.chunk_id,
                                chunk_index=chunk.chunk_index,
                                snippet=bounded_snippet,
                                relevance_reason=reason,
                                machine_scope=scope,
                                context_type=context_type,
                                quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
                                referenced_machine_id=conflicting_machine,
                                is_excerpt=is_excerpt,
                                dataset_id=chunk.dataset_id,
                            )
                        )
                    else:
                        reason = generate_support_reason(scope, context_type, chunk.text, pattern, machine_id)
                        supporting_raw.append(
                            EvidenceReference(
                                document_id=chunk.document_id,
                                source_filename=chunk.source_filename,
                                chunk_id=chunk.chunk_id,
                                chunk_index=chunk.chunk_index,
                                snippet=bounded_snippet,
                                relevance_reason=reason,
                                machine_scope=scope,
                                context_type=context_type,
                                quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
                                referenced_machine_id=machine_id if scope == MachineScope.EXACT_MACHINE else None,
                                is_excerpt=is_excerpt,
                                dataset_id=chunk.dataset_id,
                            )
                        )

            deduped_supporting = self.deduplicate_evidence_references(supporting_raw)
            deduped_contradictory = self.deduplicate_evidence_references(contradictory_raw)

            if deduped_supporting or deduped_contradictory:
                if deduped_supporting and not deduped_contradictory:
                    qual = (
                        "Preliminary investigative hypothesis grounded in historical documentation and symptom correlation. "
                        "Symptom overlap indicates potential shared failure mechanisms, but does NOT confirm root cause without physical inspection."
                    )
                elif deduped_supporting and deduped_contradictory:
                    qual = (
                        "Qualified hypothesis with mixed evidence: corroborating general symptom patterns exist, "
                        "but conflicting machine references were detected in the corpus. Physical verification is mandatory."
                    )
                else:
                    qual = (
                        "Weak cross-machine hypothesis: matching symptoms were identified exclusively in records "
                        "for a DIFFERENT machine unit. Extreme caution required before applying to this machine."
                    )

                hyp = InvestigativeHypothesis(
                    title=pattern["title"],
                    description=pattern["description"],
                    qualification=qual,
                    supporting_evidence=deduped_supporting,
                    contradictory_evidence=deduped_contradictory,
                    suggested_checks=pattern["suggested_checks"],
                )
                status, explanation = verify_hypothesis_evidence(
                    hyp,
                    incident_machine_id=machine_id,
                    pattern=pattern,
                    dataset_id=target_ds_id,
                    evidence_retriever=self.retriever,
                )
                hyp.verification_status = status
                hyp.verification_explanation = explanation

                strength, strength_explanation = evaluate_evidence_strength(
                    hyp,
                    verification_status=status,
                    incident_machine_id=machine_id,
                    dataset_id=target_ds_id,
                    evidence_retriever=self.retriever,
                )
                hyp.evidence_strength = strength
                hyp.evidence_strength_explanation = strength_explanation

                gap_guidance = derive_missing_evidence_guidance(
                    hyp,
                    incident_record=incident_record,
                    pattern=pattern,
                    dataset_id=target_ds_id,
                    evidence_retriever=self.retriever,
                )
                hyp.missing_evidence_guidance = gap_guidance

                hypotheses.append(hyp)

        supporting_hyp_count = sum(len(h.supporting_evidence) for h in hypotheses)

        missing_info = [
            "Continuous high-frequency vibration time-waveform and spectral FFT waterfall data prior to the event.",
            "Laboratory lubricant degradation analysis (viscosity, particle count, oxidation index).",
            "Ambient temperature and duty cycle load profile at the time of failure.",
        ]

        limitations = [
            "Hypotheses are preliminary investigative suggestions, NOT autonomous diagnostic conclusions.",
            "Retrieval scores (BM25) reflect keyword density and relevance, never diagnostic probability or fault severity.",
            "All physical interventions must adhere to machine lockout/tagout (LOTO) and OEM maintenance procedures.",
            "Illustrative or synthetic values in reference records must not be interpreted as physical field measurements.",
        ]

        status = AnalysisStatus.COMPLETED if hypotheses else AnalysisStatus.INSUFFICIENT_EVIDENCE
        if not hypotheses:
            if excluded_count > 0 and substantive_count == 0:
                limitations.append(
                    "Retrieved passages consisted exclusively of non-diagnostic metadata or keyword lists and were excluded from grounding."
                )
            else:
                limitations.append(
                    "Retrieved passages did not contain sufficiently specific technical criteria to justify a deterministic hypothesis."
                )

        counts = EvidenceCounts(
            raw_retrieved_count=raw_count,
            unique_deduplicated_count=unique_count,
            substantive_retained_count=substantive_count,
            excluded_low_value_count=excluded_count,
            supporting_hypotheses_count=supporting_hyp_count,
        )

        return FaultAnalysisResult(
            incident_id=incident_id,
            incident_facts=incident_record,
            retrieved_evidence=unique_results,
            hypotheses=hypotheses,
            missing_information=missing_info,
            limitations=limitations,
            status=status,
            analyzed_at=timestamp,
            evidence_counts=counts,
            excluded_evidence=excluded_refs,
        )

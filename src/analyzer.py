"""Evidence-grounded fault analysis engine for GhostFault (Milestone 4.1.2).

Substantive evidence relevance validation, passage-quality classification,
deterministic support-reason generation, and accurate pipeline count derivation.
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
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
    EvidenceCounts,
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
    ) -> List[SearchResult]:
        """Build bounded queries and retrieve matching passages, preserving raw query results."""
        search_terms = self.extract_search_terms(incident_record)
        if not search_terms:
            return []

        query_str = " ".join(search_terms[:5])
        try:
            results = self.retriever.search(query=query_str, limit=max_results)
        except Exception:
            return []

        if not results and len(search_terms) > 0:
            seen_chunk_ids: Set[str] = set()
            fallback_results: List[SearchResult] = []
            for term in search_terms[:4]:
                sub_results = self.retriever.search(query=term, limit=3)
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
    ) -> FaultAnalysisResult:
        """
        Perform evidence-grounded fault analysis with substantive relevance validation.
        """
        incident_id = incident_record.get("incident_id", "UNKNOWN_INCIDENT")
        machine_id = incident_record.get("machine_id", "UNKNOWN_MACHINE")
        symptom = incident_record.get("symptom", "")
        maintenance_action = incident_record.get("maintenance_action", "")

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

        if evidence_results is None:
            raw_evidence = self.retrieve_incident_evidence(incident_record)
        else:
            raw_evidence = evidence_results

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

                hypotheses.append(
                    InvestigativeHypothesis(
                        title=pattern["title"],
                        description=pattern["description"],
                        qualification=qual,
                        supporting_evidence=deduped_supporting,
                        contradictory_evidence=deduped_contradictory,
                        suggested_checks=pattern["suggested_checks"],
                    )
                )

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

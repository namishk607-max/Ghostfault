"""Domain models and schema definitions for GhostFault evidence ingestion, retrieval, and fault analysis."""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List


class DocumentStatus(str, Enum):
    """Lifecycle statuses for document ingestion."""
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class ErrorCode(str, Enum):
    """Standardized error codes for ingestion failures."""
    UNSUPPORTED_TYPE = "ERR_UNSUPPORTED_TYPE"
    EMPTY_FILE = "ERR_EMPTY_FILE"
    PATH_TRAVERSAL = "ERR_PATH_TRAVERSAL"
    INVALID_PDF = "ERR_INVALID_PDF"
    NO_EXTRACTABLE_TEXT = "ERR_NO_EXTRACTABLE_TEXT"
    ENCODING_ERROR = "ERR_ENCODING_ERROR"
    STORAGE_ERROR = "ERR_STORAGE_ERROR"
    EXTRACTION_ERROR = "ERR_EXTRACTION_ERROR"
    READ_ERROR = "ERR_READ_ERROR"
    INDEX_ERROR = "ERR_INDEX_ERROR"
    ANALYSIS_ERROR = "ERR_ANALYSIS_ERROR"


class AnalysisStatus(str, Enum):
    """Lifecycle status of an incident fault analysis."""
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_EVIDENCE = "no_evidence"
    FAILED = "failed"


class PassageQuality(str, Enum):
    """Classification of technical substance and retrieval quality of an evidence passage."""
    SUBSTANTIVE_TECHNICAL = "substantive_technical"
    SEARCH_TERMS_METADATA = "search_terms_metadata"
    HEADER_FOOTER_METADATA = "header_footer_metadata"
    EMPTY_OR_LOW_SUBSTANCE = "empty_or_low_substance"


class EvidenceContextType(str, Enum):
    """Classification of context for numeric values or statements in evidence."""
    INCIDENT_RECORDED = "incident_recorded"
    VERIFIED_OEM_LIMIT = "verified_oem_limit"
    ILLUSTRATIVE_TEST_VALUE = "illustrative_test_value"
    GENERAL_REFERENCE = "general_reference"
    UNKNOWN_CONTEXT = "unknown_context"


class MachineScope(str, Enum):
    """Scope of machine applicability for an evidence passage."""
    EXACT_MACHINE = "exact_machine"
    GENERAL_REFERENCE = "general_reference"
    CROSS_MACHINE_CONFLICT = "cross_machine_conflict"


@dataclass
class DocumentMetadata:
    """Structured metadata contract for every ingested document."""
    document_id: str
    original_filename: str
    file_type: str
    source_type: str
    storage_reference: str
    extraction_status: str
    ingestion_status: DocumentStatus
    created_at: str
    updated_at: str
    extracted_text_reference: Optional[str] = None
    extracted_character_count: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        data = asdict(self)
        if isinstance(self.ingestion_status, DocumentStatus):
            data["ingestion_status"] = self.ingestion_status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Deserialize metadata from dictionary."""
        raw_status = data.get("ingestion_status", DocumentStatus.FAILED.value)
        try:
            status = DocumentStatus(raw_status)
        except ValueError:
            status = DocumentStatus.FAILED

        return cls(
            document_id=data["document_id"],
            original_filename=data["original_filename"],
            file_type=data["file_type"],
            source_type=data["source_type"],
            storage_reference=data["storage_reference"],
            extraction_status=data["extraction_status"],
            ingestion_status=status,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            extracted_text_reference=data.get("extracted_text_reference"),
            extracted_character_count=int(data.get("extracted_character_count", 0)),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
        )


@dataclass
class EvidenceChunk:
    """
    Structured, validated representation of an indexed evidence chunk.
    Maintains exact source provenance back to an ingested catalog document.
    """
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source_filename: str
    file_type: str
    page_number: Optional[int]
    start_offset: Optional[int]
    end_offset: Optional[int]
    created_at: str
    content_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceChunk":
        """Deserialize chunk from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            chunk_index=int(data["chunk_index"]),
            text=data["text"],
            source_filename=data["source_filename"],
            file_type=data["file_type"],
            page_number=data.get("page_number"),
            start_offset=data.get("start_offset"),
            end_offset=data.get("end_offset"),
            created_at=data["created_at"],
            content_hash=data["content_hash"],
        )


@dataclass
class SearchResult:
    """
    A single ranked evidence passage retrieved from the search index.
    Contains source provenance and search score.
    """
    chunk: EvidenceChunk
    score: float
    matched_snippet: str
    matched_terms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize search result to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "matched_snippet": self.matched_snippet,
            "matched_terms": self.matched_terms,
        }


# ==============================================================================
# Milestone 4 / 4.1 / 4.1.2 Models: Evidence-Grounded Fault Analysis
# ==============================================================================

@dataclass
class EvidenceReference:
    """
    Direct link from a hypothesis or conclusion back to a verified retrieved chunk.
    Maintains provenance, duplication tracking, machine scope, measurement context,
    quality classification, and excerpt classification.
    """
    document_id: str
    source_filename: str
    chunk_id: str
    chunk_index: int
    snippet: str
    relevance_reason: str
    machine_scope: MachineScope = MachineScope.GENERAL_REFERENCE
    context_type: EvidenceContextType = EvidenceContextType.UNKNOWN_CONTEXT
    quality: PassageQuality = PassageQuality.SUBSTANTIVE_TECHNICAL
    referenced_machine_id: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of_chunk_id: Optional[str] = None
    duplicate_source_copies: Optional[List[Dict[str, str]]] = None
    is_excerpt: bool = False
    exclusion_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.machine_scope, MachineScope):
            data["machine_scope"] = self.machine_scope.value
        if isinstance(self.context_type, EvidenceContextType):
            data["context_type"] = self.context_type.value
        if isinstance(self.quality, PassageQuality):
            data["quality"] = self.quality.value
        return data


@dataclass
class EvidenceCounts:
    """Accurately derived evidence counts at each phase of analysis."""
    raw_retrieved_count: int = 0
    unique_deduplicated_count: int = 0
    substantive_retained_count: int = 0
    excluded_low_value_count: int = 0
    supporting_hypotheses_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class InvestigativeHypothesis:
    """
    A structured, qualified diagnostic hypothesis linked to concrete evidence references.
    """
    title: str
    description: str
    qualification: str
    supporting_evidence: List[EvidenceReference]
    contradictory_evidence: List[EvidenceReference]
    suggested_checks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "qualification": self.qualification,
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "contradictory_evidence": [e.to_dict() for e in self.contradictory_evidence],
            "suggested_checks": self.suggested_checks,
        }


@dataclass
class FaultAnalysisResult:
    """
    Complete structured fault analysis result for an incident record.
    Keeps incident facts, retrieved evidence, hypotheses, and limitations explicitly separate.
    """
    incident_id: str
    incident_facts: Dict[str, Any]
    retrieved_evidence: List[SearchResult]
    hypotheses: List[InvestigativeHypothesis]
    missing_information: List[str]
    limitations: List[str]
    status: AnalysisStatus
    analyzed_at: str
    evidence_counts: Optional[EvidenceCounts] = None
    excluded_evidence: Optional[List[EvidenceReference]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "incident_facts": self.incident_facts,
            "retrieved_evidence": [r.to_dict() for r in self.retrieved_evidence],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "missing_information": self.missing_information,
            "limitations": self.limitations,
            "status": self.status.value,
            "analyzed_at": self.analyzed_at,
            "evidence_counts": self.evidence_counts.to_dict() if self.evidence_counts else None,
            "excluded_evidence": [e.to_dict() for e in self.excluded_evidence] if self.excluded_evidence else [],
            "error_message": self.error_message,
        }


def current_iso_timestamp() -> str:
    """Return UTC ISO 8601 formatted timestamp string."""
    return datetime.now(timezone.utc).isoformat()


class RecurrenceScope(str, Enum):
    """Scope of recurrence for grouped symptoms across incidents."""
    SAME_MACHINE = "same_machine"
    CROSS_MACHINE = "cross_machine"
    MIXED = "mixed"
    SINGLETON = "singleton"
    UNDETERMINED = "undetermined"


@dataclass
class IncidentTimelineEntry:
    """Timeline-ready representation of an incident preserving exact original values."""
    incident_id: str
    machine_id: Optional[str]
    raw_date: Optional[str]
    parsed_date: Optional[str]
    is_date_valid: bool
    symptom: str
    maintenance_action: Optional[str]
    outcome: Optional[str]
    original_record: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SymptomGroup:
    """Group of incidents sharing exact or equivalent normalized symptom components."""
    group_id: str
    normalized_symptom: str
    canonical_label: str
    incident_ids: List[str]
    machine_ids: List[str]
    recurrence_scope: RecurrenceScope
    occurrences_count: int
    grouping_basis: str
    original_symptoms: List[Dict[str, str]]
    maintenance_actions: List[Dict[str, Optional[str]]]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.recurrence_scope, RecurrenceScope):
            data["recurrence_scope"] = self.recurrence_scope.value
        return data


@dataclass
class ComparisonLimitation:
    """Explicit declaration of data quality boundaries and analytical limitations."""
    code: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class CrossIncidentComparisonResult:
    """
    Deterministic cross-incident comparison result.
    Maintains full provenance, timeline-ready records, recurrence scopes, and explicit limitations.
    """
    comparison_id: str
    compared_incident_ids: List[str]
    total_incidents_analyzed: int
    symptom_groups: List[SymptomGroup]
    timeline_entries: List[IncidentTimelineEntry]
    missing_fields_summary: Dict[str, int]
    limitations: List[ComparisonLimitation]
    compared_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "compared_incident_ids": self.compared_incident_ids,
            "total_incidents_analyzed": self.total_incidents_analyzed,
            "symptom_groups": [g.to_dict() for g in self.symptom_groups],
            "timeline_entries": [t.to_dict() for t in self.timeline_entries],
            "missing_fields_summary": self.missing_fields_summary,
            "limitations": [lim.to_dict() for lim in self.limitations],
            "compared_at": self.compared_at,
        }


# ==============================================================================
# Milestone 5.3 Models: Cross-Incident Evidence Linking
# ==============================================================================

class EvidenceLinkStatus(str, Enum):
    """Status of evidence linking for a symptom group."""
    SUPPORTED = "supported"
    NO_SUPPORTING_EVIDENCE = "no_supporting_evidence"
    INSUFFICIENT_SUBSTANTIVE_EVIDENCE = "insufficient_substantive_evidence"
    ERROR = "error"


@dataclass
class SymptomGroupEvidenceLink:
    """
    Structured link between a cross-incident symptom group and a verified evidence passage.
    Preserves exact provenance, machine context, snippet, relevance reason, and duplicate tracking.
    """
    group_id: str
    related_incident_ids: List[str]
    document_id: str
    source_filename: str
    chunk_id: str
    chunk_index: int
    snippet: str
    relevance_reason: str
    link_explanation: str
    machine_scope: MachineScope = MachineScope.GENERAL_REFERENCE
    context_type: EvidenceContextType = EvidenceContextType.UNKNOWN_CONTEXT
    quality: PassageQuality = PassageQuality.SUBSTANTIVE_TECHNICAL
    referenced_machine_id: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of_chunk_id: Optional[str] = None
    duplicate_source_copies: Optional[List[Dict[str, str]]] = None
    retrieval_score: Optional[float] = None
    matched_query_terms: Optional[List[str]] = None
    provenance_limitations: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.machine_scope, MachineScope):
            data["machine_scope"] = self.machine_scope.value
        if isinstance(self.context_type, EvidenceContextType):
            data["context_type"] = self.context_type.value
        if isinstance(self.quality, PassageQuality):
            data["quality"] = self.quality.value
        return data


@dataclass
class GroupEvidenceLinkResult:
    """
    Evidence-linking evaluation for a single symptom group.
    Separates substantive supporting evidence from excluded low-value passages.
    """
    group_id: str
    canonical_label: str
    normalized_symptom: str
    incident_ids: List[str]
    recurrence_scope: RecurrenceScope
    status: EvidenceLinkStatus
    supporting_evidence: List[SymptomGroupEvidenceLink]
    excluded_evidence: List[SymptomGroupEvidenceLink]
    query_used: str
    raw_retrieved_count: int
    substantive_count: int
    excluded_count: int
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "canonical_label": self.canonical_label,
            "normalized_symptom": self.normalized_symptom,
            "incident_ids": self.incident_ids,
            "recurrence_scope": self.recurrence_scope.value if isinstance(self.recurrence_scope, RecurrenceScope) else str(self.recurrence_scope),
            "status": self.status.value if isinstance(self.status, EvidenceLinkStatus) else str(self.status),
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "excluded_evidence": [e.to_dict() for e in self.excluded_evidence],
            "query_used": self.query_used,
            "raw_retrieved_count": self.raw_retrieved_count,
            "substantive_count": self.substantive_count,
            "excluded_count": self.excluded_count,
            "limitations": self.limitations,
        }


@dataclass
class CrossIncidentEvidenceLinkingResult:
    """
    Complete evidence linking result across all symptom groups in a comparison.
    """
    linking_id: str
    comparison_id: str
    total_groups_evaluated: int
    supported_groups_count: int
    unsupported_groups_count: int
    group_results: List[GroupEvidenceLinkResult]
    global_limitations: List[ComparisonLimitation]
    linked_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "linking_id": self.linking_id,
            "comparison_id": self.comparison_id,
            "total_groups_evaluated": self.total_groups_evaluated,
            "supported_groups_count": self.supported_groups_count,
            "unsupported_groups_count": self.unsupported_groups_count,
            "group_results": [g.to_dict() for g in self.group_results],
            "global_limitations": [lim.to_dict() for lim in self.global_limitations],
            "linked_at": self.linked_at,
        }



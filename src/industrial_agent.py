"""Industrial Intelligence Agent context foundation for GhostFault (Milestone M9.5.2).

Aggregates deterministic diagnostic findings, reviewed equipment associations,
contextual engineering knowledge references, and historical case outcomes into a
bounded, typed context object for advisory consumption without modifying any underlying state.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.analyzer import FaultAnalysisEngine
from src.equipment import EquipmentRegistry
from src.history import InvestigationHistoryStore
from src.knowledge_advisor import (
    ContextualTechnicalReference,
    EquipmentMatchLevel,
    TechnicalKnowledgeAdvisor,
)
from src.knowledge_models import ReliabilityTier, SourceType
from src.models import (
    DatasetOwnershipError,
    EvidenceReference,
    EvidenceStrength,
    FaultAnalysisResult,
    InvestigativeHypothesis,
    VerificationStatus,
)
from src.storage import EvidenceStore


# ---------------------------------------------------------------------------
# Max character & list limits for bounded context safety
# ---------------------------------------------------------------------------
MAX_INCIDENT_TEXT_LEN = 500
MAX_HYPOTHESIS_COUNT = 3
MAX_HYPOTHESIS_TEXT_LEN = 300
MAX_EVIDENCE_REASON_LEN = 160
MAX_SUGGESTED_CHECKS_COUNT = 3
MAX_TECH_REFS_COUNT = 4
MAX_TECH_REF_SNIPPET_LEN = 400
MAX_OUTCOMES_COUNT = 5


def _bounded_str(value: Any, max_len: int, fallback: str = "UNKNOWN") -> str:
    """Sanitize and trim string fields, marking empty/None as fallback."""
    if value is None:
        return fallback
    s = str(value).strip()
    if not s:
        return fallback
    if len(s) > max_len:
        return s[:max_len].rstrip() + "..."
    return s


# ---------------------------------------------------------------------------
# 1. Domain Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentIncidentContext:
    """Incident facts conditioned for agent reasoning."""
    incident_id: str
    dataset_id: str
    machine_id: str
    symptom: str
    maintenance_action: str
    work_order_id: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentEquipmentContext:
    """Reviewed equipment asset context."""
    equipment_id: str
    asset_name: str
    category_code: str
    category_name: str
    subtype_code: str
    subtype_name: str
    manufacturer: str  # Marked "UNKNOWN" if missing
    model: str         # Marked "UNKNOWN" if missing
    serial_number: str # Marked "UNKNOWN" if missing
    criticality: str
    association_role: str
    review_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentEvidenceItem:
    """Bounded, abstract evidence item citing source and rationale without raw passages."""
    source_filename: str
    chunk_index: int
    relevance_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentDiagnosticFinding:
    """Deterministic fault hypothesis findings with verification status."""
    title: str
    description: str
    qualification: str
    verification_status: str
    evidence_strength: str
    supporting_reasons: List[AgentEvidenceItem]
    contradictory_reasons: List[AgentEvidenceItem]
    suggested_checks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentTechnicalReference:
    """Contextual engineering literature or standard matched to the investigation."""
    document_id: str
    chunk_id: str
    title: str
    publisher_author: str
    source_type: str
    reliability_tier: str
    match_level: str
    section_heading: Optional[str]
    estimated_page: Optional[int]
    relevance_score: float
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentHistoricalOutcome:
    """Prior confirmed case inspection or hypothesis outcome."""
    case_id: str
    run_id: str
    case_status: str
    failure_mode: Optional[str]
    determination: Optional[str]
    confidence: Optional[str]
    physical_action_taken: Optional[str]
    closed_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndustrialAgentContext:
    """Complete, unified context container for industrial advisory evaluation."""
    incident: AgentIncidentContext
    equipment: Optional[AgentEquipmentContext]
    findings: List[AgentDiagnosticFinding]
    technical_references: List[AgentTechnicalReference]
    historical_outcomes: List[AgentHistoricalOutcome]
    assembled_at: str
    active_dataset_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 2. Context Assembler
# ---------------------------------------------------------------------------

class IndustrialAgentContextAssembler:
    """Gathers deterministic diagnostic and domain knowledge for an incident.

    Enforces strict read-only access, dataset isolation, bounded limits,
    and reviewed-equipment filtering.
    """

    def __init__(
        self,
        store: EvidenceStore,
        equipment_registry: EquipmentRegistry,
        knowledge_advisor: Optional[TechnicalKnowledgeAdvisor] = None,
        history_store: Optional[InvestigationHistoryStore] = None,
    ) -> None:
        if not isinstance(store, EvidenceStore):
            raise TypeError("Assembler requires a valid EvidenceStore.")
        if not isinstance(equipment_registry, EquipmentRegistry):
            raise TypeError("Assembler requires a valid EquipmentRegistry.")

        self.store = store
        self.equipment_registry = equipment_registry
        self.knowledge_advisor = knowledge_advisor
        self.history_store = history_store

    def assemble_context(
        self,
        incident_id: str,
        dataset_id: str,
        analysis_result: FaultAnalysisResult,
        incident_facts: Optional[Mapping[str, Any]] = None,
    ) -> IndustrialAgentContext:
        """Assemble bounded context for an incident.

        Raises:
            DatasetOwnershipError: If incident or analysis result does not belong to dataset_id.
            ValueError: If incident_id or analysis_result is malformed.
        """
        if not incident_id or not incident_id.strip():
            raise ValueError("Incident ID cannot be empty.")
        if not dataset_id or not dataset_id.strip():
            raise ValueError("Dataset ID cannot be empty.")
        if not isinstance(analysis_result, FaultAnalysisResult):
            raise TypeError("analysis_result must be a FaultAnalysisResult.")

        # 1. Dataset ownership validation
        # Verify dataset registration in evidence store
        datasets = self.store.list_datasets()
        valid_ids = {d.dataset_id if hasattr(d, "dataset_id") else d["dataset_id"] for d in datasets}
        if dataset_id not in valid_ids:
            raise DatasetOwnershipError(f"Dataset '{dataset_id}' is not registered.")

        # Verify analysis result matches incident and dataset
        if analysis_result.incident_id != incident_id:
            raise DatasetOwnershipError(
                f"Analysis result incident '{analysis_result.incident_id}' does not match '{incident_id}'."
            )

        facts = incident_facts or analysis_result.incident_facts or {}
        facts_dataset = facts.get("dataset_id")
        if facts_dataset and facts_dataset != dataset_id:
            raise DatasetOwnershipError(
                f"Incident facts belong to '{facts_dataset}', not active dataset '{dataset_id}'."
            )

        # 2. Assemble Incident Context
        incident_ctx = AgentIncidentContext(
            incident_id=incident_id,
            dataset_id=dataset_id,
            machine_id=_bounded_str(facts.get("machine_id"), 64, fallback="UNKNOWN"),
            symptom=_bounded_str(facts.get("symptom"), MAX_INCIDENT_TEXT_LEN, fallback="UNKNOWN"),
            maintenance_action=_bounded_str(
                facts.get("maintenance_action"), MAX_INCIDENT_TEXT_LEN, fallback="UNKNOWN"
            ),
            work_order_id=facts.get("work_order_id"),
            timestamp=facts.get("timestamp") or facts.get("incident_date"),
        )

        # 3. Assemble Equipment Context (REVIEWED associations only)
        # Never include PENDING_REVIEW or REJECTED associations as confirmed context.
        equipment_ctx = self._assemble_equipment_context(incident_id, dataset_id)

        # 4. Assemble Diagnostic Findings (Abstracted, bounded, no raw evidence passages)
        findings = self._assemble_diagnostic_findings(analysis_result)

        # 5. Assemble Technical References (from TechnicalKnowledgeAdvisor if provided)
        tech_refs = self._assemble_technical_references(
            incident_id=incident_id,
            dataset_id=dataset_id,
            analysis_result=analysis_result,
        )

        # 6. Assemble Historical Outcomes (from InvestigationHistoryStore if provided)
        historical_outcomes = self._assemble_historical_outcomes(
            incident_id=incident_id,
            dataset_id=dataset_id,
            machine_id=incident_ctx.machine_id,
        )

        assembled_at = datetime.now(timezone.utc).isoformat()

        return IndustrialAgentContext(
            incident=incident_ctx,
            equipment=equipment_ctx,
            findings=findings,
            technical_references=tech_refs,
            historical_outcomes=historical_outcomes,
            assembled_at=assembled_at,
            active_dataset_id=dataset_id,
        )

    def _assemble_equipment_context(
        self,
        incident_id: str,
        dataset_id: str,
    ) -> Optional[AgentEquipmentContext]:
        """Fetch reviewed equipment association; ignore PENDING_REVIEW and REJECTED."""
        associations = self.equipment_registry.list_incident_associations(
            dataset_id=dataset_id,
            incident_id=incident_id,
        )

        # Filter strictly for REVIEWED status
        reviewed_assocs = [
            a for a in associations
            if a.get("review_status") == "REVIEWED"
        ]

        if not reviewed_assocs:
            return None

        # Prioritize PRIMARY role if available, else first reviewed
        primary = next(
            (a for a in reviewed_assocs if a.get("association_role") == "PRIMARY"),
            reviewed_assocs[0],
        )

        eq_id = primary["equipment_id"]
        asset = self.equipment_registry.get_equipment(dataset_id=dataset_id, equipment_id=eq_id)
        if not asset:
            return None

        return AgentEquipmentContext(
            equipment_id=asset["equipment_id"],
            asset_name=_bounded_str(asset.get("asset_name"), 120),
            category_code=_bounded_str(asset.get("category_code"), 64),
            category_name=_bounded_str(asset.get("category_name"), 100),
            subtype_code=_bounded_str(asset.get("subtype_code"), 64),
            subtype_name=_bounded_str(asset.get("subtype_name"), 100),
            manufacturer=_bounded_str(asset.get("manufacturer"), 80, fallback="UNKNOWN"),
            model=_bounded_str(asset.get("model"), 80, fallback="UNKNOWN"),
            serial_number=_bounded_str(asset.get("serial_number"), 80, fallback="UNKNOWN"),
            criticality=_bounded_str(asset.get("criticality"), 32, fallback="UNKNOWN"),
            association_role=_bounded_str(primary.get("association_role"), 32),
            review_status="REVIEWED",
        )

    def _assemble_diagnostic_findings(
        self,
        analysis_result: FaultAnalysisResult,
    ) -> List[AgentDiagnosticFinding]:
        """Extract hypotheses, abstracting evidence references into reasons without raw passages."""
        findings: List[AgentDiagnosticFinding] = []

        for hypo in (analysis_result.hypotheses or [])[:MAX_HYPOTHESIS_COUNT]:
            # Convert supporting evidence (extracting metadata & reason, NOT raw passage)
            supporting = [
                AgentEvidenceItem(
                    source_filename=_bounded_str(ref.source_filename, 100),
                    chunk_index=ref.chunk_index,
                    relevance_reason=_bounded_str(ref.relevance_reason, MAX_EVIDENCE_REASON_LEN),
                )
                for ref in (hypo.supporting_evidence or [])[:3]
                if isinstance(ref, EvidenceReference)
            ]

            contradictory = [
                AgentEvidenceItem(
                    source_filename=_bounded_str(ref.source_filename, 100),
                    chunk_index=ref.chunk_index,
                    relevance_reason=_bounded_str(ref.relevance_reason, MAX_EVIDENCE_REASON_LEN),
                )
                for ref in (hypo.contradictory_evidence or [])[:3]
                if isinstance(ref, EvidenceReference)
            ]

            checks = [
                _bounded_str(c, 160)
                for c in (hypo.suggested_checks or [])[:MAX_SUGGESTED_CHECKS_COUNT]
                if isinstance(c, str) and c.strip()
            ]

            ver_val = hypo.verification_status.value if hasattr(hypo.verification_status, "value") else str(hypo.verification_status)
            str_val = hypo.evidence_strength.value if hasattr(hypo.evidence_strength, "value") else str(hypo.evidence_strength)

            findings.append(
                AgentDiagnosticFinding(
                    title=_bounded_str(hypo.title, 120),
                    description=_bounded_str(hypo.description, MAX_HYPOTHESIS_TEXT_LEN),
                    qualification=_bounded_str(hypo.qualification, 200, fallback="None"),
                    verification_status=ver_val,
                    evidence_strength=str_val,
                    supporting_reasons=supporting,
                    contradictory_reasons=contradictory,
                    suggested_checks=checks,
                )
            )

        return findings

    def _assemble_technical_references(
        self,
        incident_id: str,
        dataset_id: str,
        analysis_result: FaultAnalysisResult,
    ) -> List[AgentTechnicalReference]:
        """Query technical knowledge advisor with active dataset and bounded excerpts."""
        if self.knowledge_advisor is None:
            return []

        try:
            raw_refs = self.knowledge_advisor.retrieve_technical_references(
                incident_id=incident_id,
                dataset_id=dataset_id,
                analysis_result=analysis_result,
                max_references=MAX_TECH_REFS_COUNT,
            )
        except Exception:
            return []

        results: List[AgentTechnicalReference] = []
        for ref in raw_refs[:MAX_TECH_REFS_COUNT]:
            # Double check dataset isolation (only active dataset or global)
            if ref.dataset_id is not None and ref.dataset_id != dataset_id:
                continue

            results.append(
                AgentTechnicalReference(
                    document_id=ref.document_id,
                    chunk_id=ref.chunk_id,
                    title=_bounded_str(ref.title, 120),
                    publisher_author=_bounded_str(ref.publisher_author, 80),
                    source_type=ref.source_type.value if hasattr(ref.source_type, "value") else str(ref.source_type),
                    reliability_tier=ref.reliability_tier.value if hasattr(ref.reliability_tier, "value") else str(ref.reliability_tier),
                    match_level=ref.match_level.value if hasattr(ref.match_level, "value") else str(ref.match_level),
                    section_heading=_bounded_str(ref.section_heading, 100, fallback="") or None,
                    estimated_page=ref.estimated_page,
                    relevance_score=round(ref.relevance_score, 4),
                    snippet=_bounded_str(ref.text, MAX_TECH_REF_SNIPPET_LEN),
                )
            )

        return results

    def _assemble_historical_outcomes(
        self,
        incident_id: str,
        dataset_id: str,
        machine_id: str,
    ) -> List[AgentHistoricalOutcome]:
        """Fetch past outcomes for the same machine or previous runs in this dataset."""
        if self.history_store is None:
            return []

        outcomes: List[AgentHistoricalOutcome] = []
        try:
            # List past investigation runs in this dataset
            runs = self.history_store.list_runs(
                store=self.store,
                dataset_id=dataset_id,
                limit=20,
            )

            for run in runs:
                # Skip current run if matching
                run_id = run["run_id"]
                case = self.history_store.get_case_by_run_id(
                    store=self.store,
                    dataset_id=dataset_id,
                    run_id=run_id,
                )
                if not case:
                    continue

                full_case = self.history_store.get_case(
                    store=self.store,
                    dataset_id=dataset_id,
                    case_id=case["case_id"],
                )
                if not full_case:
                    continue

                # Verify case belongs to dataset
                if full_case.get("dataset_id") != dataset_id:
                    continue

                case_status = full_case.get("status")
                closed_at = full_case.get("closed_at")

                # Extract inspections and outcomes
                inspections = full_case.get("inspections", [])
                action_taken = None
                if inspections:
                    # Latest inspection action
                    action_taken = inspections[-1].get("physical_action_taken")

                hypothesis_outcomes = full_case.get("hypothesis_outcomes", [])
                if hypothesis_outcomes:
                    for ho in hypothesis_outcomes:
                        outcomes.append(
                            AgentHistoricalOutcome(
                                case_id=full_case["case_id"],
                                run_id=run_id,
                                case_status=_bounded_str(case_status, 32),
                                failure_mode=_bounded_str(ho.get("failure_mode"), 80),
                                determination=_bounded_str(ho.get("determination"), 40),
                                confidence=_bounded_str(ho.get("confidence"), 40),
                                physical_action_taken=_bounded_str(action_taken, 160, fallback=None),
                                closed_at=closed_at,
                            )
                        )
                elif action_taken:
                    outcomes.append(
                        AgentHistoricalOutcome(
                            case_id=full_case["case_id"],
                            run_id=run_id,
                            case_status=_bounded_str(case_status, 32),
                            failure_mode=None,
                            determination=None,
                            confidence=None,
                            physical_action_taken=_bounded_str(action_taken, 160, fallback=None),
                            closed_at=closed_at,
                        )
                    )

                if len(outcomes) >= MAX_OUTCOMES_COUNT:
                    break

        except Exception:
            return []

        return outcomes[:MAX_OUTCOMES_COUNT]

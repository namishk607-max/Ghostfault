"""Tests for Industrial Intelligence Agent context foundation (Milestone M9.5.2)."""

import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pytest

from src.ai_gateway import (
    GatewayConfig,
    GatewayConnectionError,
    GatewayError,
    GatewayResponseError,
    GatewayTimeoutError,
)
from src.equipment import EquipmentRegistry
from src.history import InvestigationHistoryStore
from src.industrial_agent import (
    AdvisoryDiagnosticExplanation,
    AdvisoryMaintenanceRecommendation,
    AdvisoryReliabilityAssessment,
    AdvisoryUncertaintyAndGaps,
    AdvisoryValidationCheck,
    AgentDiagnosticFinding,
    AgentEquipmentContext,
    AgentHistoricalOutcome,
    AgentIncidentContext,
    AgentTechnicalReference,
    IndustrialAdvisoryReport,
    IndustrialAgentContext,
    IndustrialAgentContextAssembler,
    MAX_HYPOTHESIS_COUNT,
    MAX_TECH_REFS_COUNT,
    build_safe_agent_payload,
    generate_industrial_advisory,
    _validate_structured_report_json,
)
from src.knowledge_advisor import TechnicalKnowledgeAdvisor
from src.knowledge_chunker import KnowledgeChunker
from src.knowledge_ingestion import KnowledgeIngestionService
from src.knowledge_models import ReliabilityTier, SourceType
from src.knowledge_search import KnowledgeSearchIndex
from src.knowledge_store import KnowledgeStore
from src.models import (
    AnalysisStatus,
    DatasetOwnershipError,
    EvidenceContextType,
    EvidenceCounts,
    EvidenceReference,
    EvidenceStrength,
    FaultAnalysisResult,
    InvestigativeHypothesis,
    PassageQuality,
    VerificationStatus,
)
from src.storage import EvidenceStore


def _make_dummy_hypothesis(title: str, desc: str) -> InvestigativeHypothesis:
    ref = EvidenceReference(
        document_id="DOC-1",
        source_filename="pump_manual.pdf",
        chunk_id="CHUNK-1",
        chunk_index=1,
        snippet="Bearing overheating was observed on drive-end bearing due to inadequate lubrication.",
        relevance_reason="Confirms elevated temperature signature on drive end.",
        context_type=EvidenceContextType.INCIDENT_RECORDED,
        quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
        dataset_id="DATASET-PLANT-1",
    )
    return InvestigativeHypothesis(
        title=title,
        description=desc,
        qualification="High probability based on temperature signatures.",
        supporting_evidence=[ref],
        contradictory_evidence=[],
        suggested_checks=["Check oil level", "Perform vibration FFT"],
        verification_status=VerificationStatus.SUPPORTED,
        evidence_strength=EvidenceStrength.STRONG,
    )


def _make_dummy_analysis_result(incident_id: str, hypotheses, dataset_id: str = "DATASET-PLANT-1") -> FaultAnalysisResult:
    return FaultAnalysisResult(
        incident_id=incident_id,
        incident_facts={
            "incident_id": incident_id,
            "dataset_id": dataset_id,
            "machine_id": "PUMP-101",
            "symptom": "High vibration and abnormal acoustic noise at drive end.",
            "maintenance_action": "Bearing inspected and lubricated.",
        },
        retrieved_evidence=[],
        hypotheses=hypotheses,
        missing_information=[],
        limitations=[],
        status=AnalysisStatus.COMPLETED,
        analyzed_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def agent_env(tmp_path: Path):
    store_dir = tmp_path / "store"
    store = EvidenceStore(store_dir)
    store.create_dataset("DATASET-PLANT-1", "Plant 1")
    store.create_dataset("DATASET-PLANT-2", "Plant 2")

    eq_db = tmp_path / "equipment.db"
    equipment_registry = EquipmentRegistry(store=store, db_path=eq_db)

    know_db = tmp_path / "knowledge.db"
    know_store = KnowledgeStore(store=store, db_path=know_db)
    ingestion_service = KnowledgeIngestionService(
        knowledge_store=know_store,
        base_dir=tmp_path / "knowledge_files",
    )
    search_index = KnowledgeSearchIndex(
        knowledge_store=know_store,
        ingestion_service=ingestion_service,
        db_path=tmp_path / "knowledge_fts.db",
    )
    ingestion_service.search_index = search_index
    knowledge_advisor = TechnicalKnowledgeAdvisor(
        equipment_registry=equipment_registry,
        search_index=search_index,
    )

    history_db = tmp_path / "history.db"
    history_store = InvestigationHistoryStore(db_path=history_db)

    assembler = IndustrialAgentContextAssembler(
        store=store,
        equipment_registry=equipment_registry,
        knowledge_advisor=knowledge_advisor,
        history_store=history_store,
    )

    return (
        store,
        equipment_registry,
        know_store,
        ingestion_service,
        search_index,
        knowledge_advisor,
        history_store,
        assembler,
    )


class TestIndustrialAgentContextAssembly:

    def test_assemble_context_with_reviewed_equipment_and_knowledge(self, agent_env):
        (
            store,
            equipment_registry,
            know_store,
            ingestion_service,
            search_index,
            knowledge_advisor,
            history_store,
            assembler,
        ) = agent_env

        # 1. Register equipment
        asset = equipment_registry.create_equipment(
            dataset_id="DATASET-PLANT-1",
            asset_name="Boiler Feed Pump P-101",
            category_code="pump",
            subtype_code="centrifugal",
            manufacturer="Flowserve",
            model="HPX-200",
            criticality="HIGH",
        )
        incidents_df = pd.DataFrame([{"incident_id": "INC-PUMP-001", "dataset_id": "DATASET-PLANT-1"}])
        assoc = equipment_registry.create_incident_association(
            dataset_id="DATASET-PLANT-1",
            incident_id="INC-PUMP-001",
            equipment_id=asset["equipment_id"],
            incidents=incidents_df,
            role="PRIMARY",
        )
        # Mark as REVIEWED
        equipment_registry.review_incident_association(
            dataset_id="DATASET-PLANT-1",
            association_id=assoc["association_id"],
            decision="REVIEWED",
            reviewer="chief_engineer",
        )

        # 2. Ingest engineering document
        doc_text = "Flowserve HPX Pump Manual. Bearing lubrication and shaft alignment tolerances."
        ingestion_service.ingest_document(
            filename="flowserve_manual.txt",
            content=doc_text.encode("utf-8"),
            title="Flowserve HPX Manual",
            publisher_author="Flowserve",
            source_type=SourceType.OEM_MANUAL,
            reliability_tier=ReliabilityTier.TIER_2_OEM,
            dataset_id="DATASET-PLANT-1",
            equipment_category="pump",
            equipment_subtype="centrifugal",
            manufacturer="Flowserve",
        )

        # 3. Create analysis result
        hypo = _make_dummy_hypothesis("Bearing Lubrication Breakdown", "Lack of lubrication caused overheating.")
        analysis = _make_dummy_analysis_result("INC-PUMP-001", [hypo], dataset_id="DATASET-PLANT-1")

        # 4. Assemble context
        context = assembler.assemble_context(
            incident_id="INC-PUMP-001",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )

        assert isinstance(context, IndustrialAgentContext)
        assert context.active_dataset_id == "DATASET-PLANT-1"

        # Verify incident context
        assert context.incident.incident_id == "INC-PUMP-001"
        assert context.incident.machine_id == "PUMP-101"
        assert "vibration" in context.incident.symptom.lower()

        # Verify equipment context (reviewed)
        assert context.equipment is not None
        assert context.equipment.asset_name == "Boiler Feed Pump P-101"
        assert context.equipment.category_code == "pump"
        assert context.equipment.manufacturer == "Flowserve"
        assert context.equipment.review_status == "REVIEWED"

        # Verify diagnostic findings
        assert len(context.findings) == 1
        finding = context.findings[0]
        assert finding.title == "Bearing Lubrication Breakdown"
        assert finding.verification_status == "supported"
        assert len(finding.supporting_reasons) == 1
        # Raw snippet must NOT be in the finding reason
        assert "Bearing overheating was observed" not in finding.supporting_reasons[0].relevance_reason
        assert "Confirms elevated temperature" in finding.supporting_reasons[0].relevance_reason

        # Verify technical references
        assert len(context.technical_references) > 0
        ref = context.technical_references[0]
        assert ref.title == "Flowserve HPX Manual"
        assert ref.match_level == "EXACT_MANUFACTURER_MODEL"
        assert len(ref.snippet) <= 400

    def test_pending_and_rejected_associations_are_ignored(self, agent_env):
        (
            store,
            equipment_registry,
            _,
            _,
            _,
            _,
            _,
            assembler,
        ) = agent_env

        asset_pending = equipment_registry.create_equipment(
            dataset_id="DATASET-PLANT-1",
            asset_name="Pending Compressor",
            category_code="compressor",
            subtype_code="reciprocating",
        )
        incidents_df = pd.DataFrame([{"incident_id": "INC-COMP-999", "dataset_id": "DATASET-PLANT-1"}])
        assoc = equipment_registry.create_incident_association(
            dataset_id="DATASET-PLANT-1",
            incident_id="INC-COMP-999",
            equipment_id=asset_pending["equipment_id"],
            incidents=incidents_df,
        )
        # Default is PENDING_REVIEW; ensure it is not included
        analysis = _make_dummy_analysis_result("INC-COMP-999", [], dataset_id="DATASET-PLANT-1")
        ctx = assembler.assemble_context(
            incident_id="INC-COMP-999",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )
        assert ctx.equipment is None

        # Now reject it and ensure it remains None
        equipment_registry.review_incident_association(
            dataset_id="DATASET-PLANT-1",
            association_id=assoc["association_id"],
            decision="REJECTED",
            reviewer="chief_engineer",
        )
        ctx2 = assembler.assemble_context(
            incident_id="INC-COMP-999",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )
        assert ctx2.equipment is None

    def test_dataset_isolation_enforced(self, agent_env):
        (
            store,
            equipment_registry,
            _,
            _,
            _,
            _,
            _,
            assembler,
        ) = agent_env

        # Attempt to assemble context for un-registered dataset
        analysis = _make_dummy_analysis_result("INC-PUMP-001", [], dataset_id="DATASET-PLANT-1")
        with pytest.raises(DatasetOwnershipError, match="not registered"):
            assembler.assemble_context(
                incident_id="INC-PUMP-001",
                dataset_id="UNREGISTERED-DATASET",
                analysis_result=analysis,
            )

        # Attempt to assemble context when analysis result belongs to a different dataset
        analysis_plant2 = _make_dummy_analysis_result("INC-PUMP-001", [], dataset_id="DATASET-PLANT-2")
        with pytest.raises(DatasetOwnershipError, match="Incident facts belong to 'DATASET-PLANT-2'"):
            assembler.assemble_context(
                incident_id="INC-PUMP-001",
                dataset_id="DATASET-PLANT-1",
                analysis_result=analysis_plant2,
            )

    def test_missing_information_marked_unknown(self, agent_env):
        (
            store,
            equipment_registry,
            _,
            _,
            _,
            _,
            _,
            assembler,
        ) = agent_env

        # Create equipment with minimal fields
        asset = equipment_registry.create_equipment(
            dataset_id="DATASET-PLANT-1",
            asset_name="Minimal Pump",
            category_code="pump",
            subtype_code="centrifugal",
            # manufacturer, model, serial_number not provided
        )
        incidents_df = pd.DataFrame([{"incident_id": "INC-PUMP-01", "dataset_id": "DATASET-PLANT-1"}])
        assoc = equipment_registry.create_incident_association(
            dataset_id="DATASET-PLANT-1",
            incident_id="INC-PUMP-01",
            equipment_id=asset["equipment_id"],
            incidents=incidents_df,
        )
        equipment_registry.review_incident_association(
            dataset_id="DATASET-PLANT-1",
            association_id=assoc["association_id"],
            decision="REVIEWED",
            reviewer="inspector",
        )

        analysis = _make_dummy_analysis_result("INC-PUMP-01", [], dataset_id="DATASET-PLANT-1")
        ctx = assembler.assemble_context(
            incident_id="INC-PUMP-01",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )

        assert ctx.equipment is not None
        assert ctx.equipment.manufacturer == "UNKNOWN"
        assert ctx.equipment.model == "UNKNOWN"
        assert ctx.equipment.serial_number == "UNKNOWN"

    def test_bounded_context_size(self, agent_env):
        (
            store,
            equipment_registry,
            _,
            _,
            _,
            _,
            _,
            assembler,
        ) = agent_env

        # Generate 10 hypotheses (exceeding MAX_HYPOTHESIS_COUNT)
        hypos = [
            _make_dummy_hypothesis(f"Hypothesis {i}", f"Description {i} " * 50)
            for i in range(10)
        ]
        analysis = _make_dummy_analysis_result("INC-TEST-BOUNDS", hypos, dataset_id="DATASET-PLANT-1")
        ctx = assembler.assemble_context(
            incident_id="INC-TEST-BOUNDS",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )

        # Must be bounded to MAX_HYPOTHESIS_COUNT
        assert len(ctx.findings) == MAX_HYPOTHESIS_COUNT
        for f in ctx.findings:
            assert len(f.description) <= 303  # 300 + "..."

    def test_no_database_mutations(self, agent_env):
        (
            store,
            equipment_registry,
            know_store,
            _,
            _,
            _,
            history_store,
            assembler,
        ) = agent_env

        # Count records before
        eq_count_before = len(equipment_registry.list_equipment("DATASET-PLANT-1"))
        doc_count_before = len(know_store.list_documents("DATASET-PLANT-1"))
        runs_before = len(history_store.list_runs(store, "DATASET-PLANT-1"))

        analysis = _make_dummy_analysis_result("INC-PUMP-001", [], dataset_id="DATASET-PLANT-1")
        _ = assembler.assemble_context(
            incident_id="INC-PUMP-001",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )

        # Count records after
        assert len(equipment_registry.list_equipment("DATASET-PLANT-1")) == eq_count_before
        assert len(know_store.list_documents("DATASET-PLANT-1")) == doc_count_before
        assert len(history_store.list_runs(store, "DATASET-PLANT-1")) == runs_before


# ---------------------------------------------------------------------------
# M9.5.3 Tests: Structured AI Reasoning & Validation
# ---------------------------------------------------------------------------

class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeChatCompletion:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeOpenAIClient:
    def __init__(self, response_content: str, should_raise: Exception = None):
        self.response_content = response_content
        self.should_raise = should_raise
        self.captured_messages = []
        self.captured_model = None

    class Chat:
        def __init__(self, outer):
            self.completions = self.Completions(outer)

        class Completions:
            def __init__(self, outer):
                self.outer = outer

            def create(self, model, messages, **kwargs):
                self.outer.captured_model = model
                self.outer.captured_messages = messages
                if self.outer.should_raise:
                    raise self.outer.should_raise
                return FakeChatCompletion(self.outer.response_content)

    @property
    def chat(self):
        return self.Chat(self)


VALID_AGENT_RESPONSE_JSON = json.dumps({
    "diagnostic_explanation": {
        "primary_hypothesis_title": "Bearing Lubrication Breakdown",
        "physical_mechanism_explanation": "Boundary lubrication failure generated excessive boundary friction.",
        "contributing_factors": ["High operating ambient temperature", "Extended grease re-lubrication interval"],
        "contradicting_factors": ["No initial seal degradation observed"]
    },
    "maintenance_recommendations": [
        {
            "action_type": "INSPECTION",
            "priority": "IMMEDIATE",
            "recommended_procedure": "Perform vibration analysis (FFT) on drive-end bearing housing.",
            "safety_precaution": "Lock out and tag out pump drive motor before physical inspection.",
            "authoritative_reference": "Flowserve HPX Operation Manual, Section 5"
        }
    ],
    "reliability_assessment": {
        "recurrence_risk": "MODERATE",
        "failure_pattern_analysis": "Consistent with thermal breakdown of NLGI-2 polyurea grease.",
        "fleet_impact_commentary": "Check other HPX pumps operating in the same process area."
    },
    "uncertainty_and_gaps": {
        "missing_information": ["Exact oil sample lab analysis", "Continuous vibration trend data"],
        "assumptions_made": ["Pump operated within rated flow envelope"],
        "unverified_parameters": ["Operating temperature at time of failure"]
    },
    "validation_checks": [
        {
            "target_component": "Drive-end rolling elements",
            "verification_method": "Inspect raceway for discoloration and spalling",
            "acceptance_criteria": "Verify radial internal clearance against ISO 5753-1 limits"
        }
    ]
})


class TestIndustrialAgentStructuredReasoning:

    def test_build_safe_agent_payload_privacy(self, agent_env):
        """Verify strict payload privacy guarantees (no IDs, no secrets, no raw paths)."""
        *_, assembler = agent_env

        hypo = _make_dummy_hypothesis("Bearing Overheating", "Winding degradation.")
        analysis = _make_dummy_analysis_result("INC-SENSITIVE-123", [hypo], dataset_id="DATASET-PLANT-1")
        ctx = assembler.assemble_context(
            incident_id="INC-SENSITIVE-123",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )

        payload = build_safe_agent_payload(ctx)

        payload_str = json.dumps(payload)

        # Assert no private identifiers in payload
        assert "DATASET-PLANT-1" not in payload_str
        assert "INC-SENSITIVE-123" not in payload_str
        assert "PUMP-101" not in payload_str  # machine identifier stripped
        assert "pump_manual.pdf" not in payload_str  # raw filename stripped
        assert "CHUNK-1" not in payload_str
        assert "data/knowledge" not in payload_str

        # Assert technical context is present
        assert "Bearing Overheating" in payload_str
        assert "incident_facts" in payload
        assert "deterministic_findings" in payload

    def test_generate_industrial_advisory_successful(self, agent_env):
        """End-to-end advisory generation with mocked LLM returning valid JSON."""
        *_, assembler = agent_env

        hypo = _make_dummy_hypothesis("Bearing Lubrication Breakdown", "Lack of lubrication caused overheating.")
        analysis = _make_dummy_analysis_result("INC-PUMP-001", [hypo], dataset_id="DATASET-PLANT-1")
        ctx = assembler.assemble_context(
            incident_id="INC-PUMP-001",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )

        config = GatewayConfig(base_url="http://localhost:20128/v1", api_key="secret-key", model="gpt-4o-mini")
        fake_client = FakeOpenAIClient(response_content=VALID_AGENT_RESPONSE_JSON)

        report = generate_industrial_advisory(
            config=config,
            context=ctx,
            client_factory=lambda cfg: fake_client,
        )

        assert isinstance(report, IndustrialAdvisoryReport)
        assert report.incident_id == "INC-PUMP-001"
        assert report.dataset_id == "DATASET-PLANT-1"
        assert "ADVISORY ONLY" in report.disclaimer

        # Check diagnostic explanation
        assert report.diagnostic_explanation.primary_hypothesis_title == "Bearing Lubrication Breakdown"
        assert "Boundary lubrication" in report.diagnostic_explanation.physical_mechanism_explanation
        assert len(report.diagnostic_explanation.contributing_factors) == 2

        # Check maintenance recommendations
        assert len(report.maintenance_recommendations) == 1
        rec = report.maintenance_recommendations[0]
        assert rec.action_type == "INSPECTION"
        assert rec.priority == "IMMEDIATE"
        assert "Lock out and tag out" in rec.safety_precaution
        assert "Flowserve HPX" in rec.authoritative_reference

        # Check uncertainty & validation checks
        assert len(report.uncertainty_and_gaps.missing_information) == 2
        assert len(report.validation_checks) == 1
        check = report.validation_checks[0]
        assert check.target_component == "Drive-end rolling elements"
        assert "ISO 5753-1" in check.acceptance_criteria

    def test_handles_markdown_wrapped_json(self):
        """Handle AI returning markdown ```json ... ``` code blocks."""
        wrapped = f"```json\n{VALID_AGENT_RESPONSE_JSON}\n```"
        report = _validate_structured_report_json(wrapped, "INC-01", "DATASET-1")
        assert report.diagnostic_explanation.primary_hypothesis_title == "Bearing Lubrication Breakdown"

    def test_rejects_malformed_json_with_gateway_response_error(self):
        """Handle malformed or truncated JSON from AI."""
        invalid_json = '{"diagnostic_explanation": { "primary_hypothesis_title": "Broken...'
        with pytest.raises(GatewayResponseError, match="not valid JSON"):
            _validate_structured_report_json(invalid_json, "INC-01", "DATASET-1")

    def test_rejects_missing_required_sections(self):
        """Handle JSON missing diagnostic_explanation or maintenance_recommendations."""
        missing_diag = json.dumps({"maintenance_recommendations": []})
        with pytest.raises(GatewayResponseError, match="Missing or invalid 'diagnostic_explanation'"):
            _validate_structured_report_json(missing_diag, "INC-01", "DATASET-1")

    def test_handles_gateway_connection_and_timeout_errors(self, agent_env):
        """Ensure connection and timeout errors bubble up as GatewayError types."""
        *_, assembler = agent_env
        analysis = _make_dummy_analysis_result("INC-01", [], dataset_id="DATASET-PLANT-1")
        ctx = assembler.assemble_context(
            incident_id="INC-01",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )

        config = GatewayConfig(base_url="http://localhost:20128/v1", api_key="secret-key", model="gpt-4o-mini")

        # Simulate timeout error
        timeout_client = FakeOpenAIClient(
            response_content="",
            should_raise=TimeoutError("Connection timed out"),
        )
        with pytest.raises(GatewayTimeoutError, match="timed out"):
            generate_industrial_advisory(
                config=config,
                context=ctx,
                client_factory=lambda cfg: timeout_client,
            )

        # Simulate connection error
        conn_client = FakeOpenAIClient(
            response_content="",
            should_raise=ConnectionError("Failed to connect"),
        )
        with pytest.raises(GatewayConnectionError, match="could not be reached"):
            generate_industrial_advisory(
                config=config,
                context=ctx,
                client_factory=lambda cfg: conn_client,
            )

    def test_no_database_mutations_during_advisory_generation(self, agent_env):
        """Ensure AI advisory generation is strictly read-only and mutates zero tables."""
        store, eq_reg, know_store, *_, hist_store, assembler = agent_env

        analysis = _make_dummy_analysis_result("INC-PUMP-001", [], dataset_id="DATASET-PLANT-1")
        ctx = assembler.assemble_context(
            incident_id="INC-PUMP-001",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )

        eq_count_before = len(eq_reg.list_equipment("DATASET-PLANT-1"))
        doc_count_before = len(know_store.list_documents("DATASET-PLANT-1"))
        runs_before = len(hist_store.list_runs(store, "DATASET-PLANT-1"))

        config = GatewayConfig(base_url="http://localhost:20128/v1", api_key="secret-key", model="gpt-4o-mini")
        fake_client = FakeOpenAIClient(response_content=VALID_AGENT_RESPONSE_JSON)

        _ = generate_industrial_advisory(
            config=config,
            context=ctx,
            client_factory=lambda cfg: fake_client,
        )

        assert len(eq_reg.list_equipment("DATASET-PLANT-1")) == eq_count_before
        assert len(know_store.list_documents("DATASET-PLANT-1")) == doc_count_before
        assert len(hist_store.list_runs(store, "DATASET-PLANT-1")) == runs_before

    def test_ai_cannot_introduce_ungrounded_primary_hypothesis(self, agent_env):
        """Verify that AI response proposing an ungrounded hypothesis is rejected."""
        *_, assembler = agent_env
        hypo = _make_dummy_hypothesis("Bearing Lubrication Breakdown", "Lack of lubrication.")
        analysis = _make_dummy_analysis_result("INC-PUMP-001", [hypo], dataset_id="DATASET-PLANT-1")
        ctx = assembler.assemble_context(
            incident_id="INC-PUMP-001",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis,
        )

        config = GatewayConfig(base_url="http://localhost:20128/v1", api_key="secret-key", model="gpt-4o-mini")

        # Fake response proposing a brand new hallucinated hypothesis not in findings
        hallucinated_json = json.dumps({
            "diagnostic_explanation": {
                "primary_hypothesis_title": "Severe Stator Core Lamination Fault",
                "physical_mechanism_explanation": "Eddy current breakdown.",
                "contributing_factors": [],
                "contradicting_factors": []
            },
            "maintenance_recommendations": [
                {
                    "action_type": "INSPECTION",
                    "priority": "ROUTINE",
                    "recommended_procedure": "Inspect stator.",
                    "safety_precaution": "LOTO required.",
                    "authoritative_reference": "NEMA MG-1"
                }
            ],
            "reliability_assessment": {"recurrence_risk": "LOW"},
            "uncertainty_and_gaps": {},
            "validation_checks": []
        })
        fake_client = FakeOpenAIClient(response_content=hallucinated_json)

        with pytest.raises(GatewayResponseError, match="does not match any deterministic finding"):
            generate_industrial_advisory(
                config=config,
                context=ctx,
                client_factory=lambda cfg: fake_client,
            )

    def test_payload_excludes_proprietary_restricted_documents(self):
        """Verify build_safe_agent_payload excludes documents marked proprietary."""
        ctx = IndustrialAgentContext(
            incident=AgentIncidentContext(
                incident_id="INC-1",
                dataset_id="DS-1",
                machine_id="M-1",
                symptom="Symptom text",
                maintenance_action="Action text",
            ),
            equipment=None,
            findings=[],
            technical_references=[
                AgentTechnicalReference(
                    document_id="DOC-PUBLIC",
                    chunk_id="CHK-1",
                    title="Public ISO Standard",
                    publisher_author="ISO",
                    source_type="ENGINEERING_STANDARD",
                    reliability_tier="TIER_1_STANDARD",
                    match_level="CATEGORY_LEVEL",
                    section_heading="Section 4",
                    estimated_page=12,
                    relevance_score=1.5,
                    snippet="Public excerpt for technical guidance.",
                    is_proprietary_restricted=False,
                ),
                AgentTechnicalReference(
                    document_id="DOC-PROPRIETARY",
                    chunk_id="CHK-2",
                    title="Confidential Plant IP Manual",
                    publisher_author="Vendor",
                    source_type="OEM_MANUAL",
                    reliability_tier="TIER_2_OEM",
                    match_level="EXACT_MANUFACTURER_MODEL",
                    section_heading="Confidential Secret Section",
                    estimated_page=45,
                    relevance_score=2.0,
                    snippet="TOP SECRET PROPRIETARY PROCESS",
                    is_proprietary_restricted=True,
                ),
            ],
            historical_outcomes=[],
            assembled_at="2026-11-18T00:00:00Z",
            active_dataset_id="DS-1",
        )

        payload = build_safe_agent_payload(ctx)
        payload_str = json.dumps(payload)

        # Public excerpt allowed
        assert "Public ISO Standard" in payload_str
        assert "Public excerpt for technical guidance" in payload_str

        # Proprietary excerpt completely excluded
        assert "Confidential Plant IP Manual" not in payload_str
        assert "TOP SECRET PROPRIETARY PROCESS" not in payload_str

    def test_safety_precaution_requires_qualified_engineer_and_site_verification(self):
        """Verify safety precautions automatically enforce qualified engineer verification."""
        resp = _validate_structured_report_json(
            VALID_AGENT_RESPONSE_JSON,
            incident_id="INC-1",
            dataset_id="DS-1",
            allowed_hypothesis_titles=["Bearing Lubrication Breakdown"],
        )
        rec = resp.maintenance_recommendations[0]
        assert "qualified engineer" in rec.safety_precaution.lower() or "site" in rec.safety_precaution.lower()

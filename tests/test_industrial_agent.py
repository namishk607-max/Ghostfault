"""Tests for Industrial Intelligence Agent context foundation (Milestone M9.5.2)."""

import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pytest

from src.equipment import EquipmentRegistry
from src.history import InvestigationHistoryStore
from src.industrial_agent import (
    AgentDiagnosticFinding,
    AgentEquipmentContext,
    AgentHistoricalOutcome,
    AgentIncidentContext,
    AgentTechnicalReference,
    IndustrialAgentContext,
    IndustrialAgentContextAssembler,
    MAX_HYPOTHESIS_COUNT,
    MAX_TECH_REFS_COUNT,
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

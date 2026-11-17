"""Focused tests for GhostFault M9.4.4 Equipment-Filtered Technical Retrieval."""

from pathlib import Path

import pandas as pd
import pytest

from src.equipment import EquipmentRegistry
from src.knowledge_advisor import (
    ContextualTechnicalReference,
    EquipmentMatchLevel,
    TechnicalKnowledgeAdvisor,
)
from src.knowledge_chunker import KnowledgeChunker
from src.knowledge_ingestion import KnowledgeIngestionService
from src.knowledge_models import EngineeringReference, ReliabilityTier, SourceType
from src.knowledge_search import KnowledgeSearchIndex
from src.knowledge_store import KnowledgeStore
from src.models import (
    AnalysisStatus,
    EvidenceContextType,
    EvidenceCounts,
    EvidenceReference,
    EvidenceStrength,
    FaultAnalysisResult,
    InvestigativeHypothesis,
    PassageQuality,
    VerificationStatus,
    current_iso_timestamp,
)
from src.storage import EvidenceStore


@pytest.fixture
def advisor_env(tmp_path: Path):
    """Fixture providing isolated EvidenceStore, EquipmentRegistry, KnowledgeStore, SearchIndex, and Advisor."""
    evidence_dir = tmp_path / "evidence"
    evidence_store = EvidenceStore(base_dir=evidence_dir)
    evidence_store.ensure_default_dataset()
    evidence_store.create_dataset("DATASET-PLANT-1", "Plant Unit 1")
    evidence_store.create_dataset("DATASET-PLANT-2", "Plant Unit 2")

    equipment_db = tmp_path / "equipment" / "equipment_registry.db"
    equipment_registry = EquipmentRegistry(store=evidence_store, db_path=equipment_db)

    knowledge_root = tmp_path / "knowledge"
    knowledge_db = knowledge_root / "knowledge_library.db"
    knowledge_store = KnowledgeStore(store=evidence_store, db_path=knowledge_db)

    index_db = knowledge_root / "knowledge_index.db"
    bootstrap_service = KnowledgeIngestionService(knowledge_store=knowledge_store, base_dir=knowledge_root)
    search_index = KnowledgeSearchIndex(
        knowledge_store=knowledge_store,
        ingestion_service=bootstrap_service,
        db_path=index_db,
    )
    ingestion_service = KnowledgeIngestionService(
        knowledge_store=knowledge_store,
        base_dir=knowledge_root,
        search_index=search_index,
    )
    search_index.ingestion_service = ingestion_service

    advisor = TechnicalKnowledgeAdvisor(
        search_index=search_index,
        equipment_registry=equipment_registry,
    )

    return evidence_store, equipment_registry, knowledge_store, ingestion_service, search_index, advisor


def _make_dummy_hypothesis(title: str, description: str) -> InvestigativeHypothesis:
    return InvestigativeHypothesis(
        title=title,
        description=description,
        qualification="High probability mechanical seal defect",
        supporting_evidence=[
            EvidenceReference(
                document_id="DOC-EV-1",
                source_filename="log.txt",
                chunk_id="CHK-1",
                chunk_index=0,
                snippet="Vibration match detected on bearing housing.",
                relevance_reason="Direct observation of high vibration",
                context_type=EvidenceContextType.INCIDENT_RECORDED,
                quality=PassageQuality.SUBSTANTIVE_TECHNICAL,
            )
        ],
        contradictory_evidence=[],
        suggested_checks=["Check seal flush pressure", "Inspect mechanical seal faces"],
        verification_status=VerificationStatus.SUPPORTED,
        evidence_strength=EvidenceStrength.STRONG,
    )


def _make_dummy_analysis_result(incident_id: str, hypotheses: list) -> FaultAnalysisResult:
    return FaultAnalysisResult(
        incident_id=incident_id,
        incident_facts={"machine_id": "TEST-MACHINE"},
        retrieved_evidence=[],
        hypotheses=hypotheses,
        missing_information=[],
        limitations=[],
        status=AnalysisStatus.COMPLETED,
        analyzed_at=current_iso_timestamp(),
    )


class TestTechnicalKnowledgeAdvisor:
    """Test conditioned retrieval based on equipment assets and deterministic hypotheses."""

    def test_retrieve_exact_oem_manual_match(self, advisor_env):
        _, equipment_registry, _, ingestion_service, _, advisor = advisor_env

        # 1. Register an equipment asset in DATASET-PLANT-1
        asset = equipment_registry.create_equipment(
            dataset_id="DATASET-PLANT-1",
            asset_name="Boiler Feed Water Pump P-101",
            category_code="pump",
            subtype_code="centrifugal",
            manufacturer="Flowserve",
            model="Mark 3",
        )

        # 2. Explicitly link incident to equipment
        incidents_df = pd.DataFrame([{"incident_id": "INC-PUMP-901", "dataset_id": "DATASET-PLANT-1"}])
        equipment_registry.create_incident_association(
            dataset_id="DATASET-PLANT-1",
            incident_id="INC-PUMP-901",
            equipment_id=asset["equipment_id"],
            incidents=incidents_df,
            role="PRIMARY",
        )

        # 3. Ingest Flowserve Mark 3 OEM manual (Dataset-scoped)
        text_oem = (
            "Flowserve Mark 3 Centrifugal Pump User Instructions.\n\n"
            "Section 5.3 Mechanical Seal Flush and Barrier Fluid Piping.\n\n"
            "Mechanical seal failure often stems from inadequate flush flow or barrier fluid loss. "
            "Inspect Plan 11 orifice for clogging or crystallization."
        )
        ingestion_service.ingest_document(
            filename="Flowserve_Mark3_Manual.txt",
            content=text_oem.encode("utf-8"),
            title="Flowserve Mark 3 Pump Maintenance Guide",
            publisher_author="Flowserve Corporation",
            source_type=SourceType.OEM_MANUAL,
            reliability_tier=ReliabilityTier.TIER_2_OEM,
            dataset_id="DATASET-PLANT-1",
            equipment_category="pump",
            equipment_subtype="centrifugal",
            manufacturer="Flowserve",
        )

        # 4. Ingest an ISO General Standard (Global)
        text_iso = (
            "ISO 10816-3 Mechanical Vibration Guidelines for Industrial Pumps.\n\n"
            "Clause 4.2 Velocity vibration thresholds for rigid support centrifugal pumps."
        )
        ingestion_service.ingest_document(
            filename="ISO_10816_3.txt",
            content=text_iso.encode("utf-8"),
            title="ISO 10816-3 Vibration Severity",
            publisher_author="ISO",
            source_type=SourceType.ENGINEERING_STANDARD,
            reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            dataset_id=None,  # Global
            equipment_category="pump",
            equipment_subtype="centrifugal",
        )

        # 5. Create hypothesis about mechanical seal failure
        hypo = _make_dummy_hypothesis(
            title="Mechanical Seal Failure",
            description="Loss of seal flush flow causing face overheating and leakage.",
        )
        analysis_result = _make_dummy_analysis_result("INC-PUMP-901", [hypo])

        # 6. Retrieve conditioned technical references
        refs = advisor.retrieve_technical_references(
            incident_id="INC-PUMP-901",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis_result,
        )

        assert len(refs) > 0
        top_ref = refs[0]
        assert top_ref.title == "Flowserve Mark 3 Pump Maintenance Guide"
        # Verify exact manufacturer match level classification
        assert top_ref.match_level in (
            EquipmentMatchLevel.EXACT_MANUFACTURER_MODEL,
            EquipmentMatchLevel.EXACT_MANUFACTURER,
        )
        assert top_ref.reliability_tier == ReliabilityTier.TIER_2_OEM
        assert "Mechanical seal failure" in top_ref.text

        # Verify conversion to formal EngineeringReference
        eng_ref = top_ref.to_engineering_reference()
        assert isinstance(eng_ref, EngineeringReference)
        assert "Contextual Reference Only" in eng_ref.reference_disclaimer

    def test_retrieve_category_level_standards_when_oem_not_available(self, advisor_env):
        _, equipment_registry, _, ingestion_service, _, advisor = advisor_env

        # Asset without manufacturer/model specified
        asset = equipment_registry.create_equipment(
            dataset_id="DATASET-PLANT-1",
            asset_name="Generic Induction Motor M-201",
            category_code="electric_motor",
            subtype_code="induction",
        )
        incidents_df = pd.DataFrame([{"incident_id": "INC-MOTOR-102", "dataset_id": "DATASET-PLANT-1"}])
        equipment_registry.create_incident_association(
            dataset_id="DATASET-PLANT-1",
            incident_id="INC-MOTOR-102",
            equipment_id=asset["equipment_id"],
            incidents=incidents_df,
        )

        # Global ISO standard for electric motors
        text_std = (
            "NEMA MG-1 Motors and Generators Standards.\n\n"
            "Section 12.53 Bearing Temperature and Stator Winding Insulation Limits."
        )
        ingestion_service.ingest_document(
            filename="NEMA_MG1.txt",
            content=text_std.encode("utf-8"),
            title="NEMA MG-1 Motor Standards",
            publisher_author="NEMA",
            source_type=SourceType.ENGINEERING_STANDARD,
            reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            dataset_id=None,
            equipment_category="electric_motor",
            equipment_subtype="induction",
        )

        hypo = _make_dummy_hypothesis(
            title="Stator Winding Overheating",
            description="High winding temperature observed during continuous operation.",
        )
        analysis_result = _make_dummy_analysis_result("INC-MOTOR-102", [hypo])

        refs = advisor.retrieve_technical_references(
            incident_id="INC-MOTOR-102",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis_result,
        )

        assert len(refs) > 0
        ref = refs[0]
        assert ref.title == "NEMA MG-1 Motor Standards"
        assert ref.match_level == EquipmentMatchLevel.CATEGORY_AND_SUBTYPE
        assert ref.dataset_id is None  # Global visibility

    def test_dataset_isolation_rejects_other_plant_references(self, advisor_env):
        _, equipment_registry, _, ingestion_service, _, advisor = advisor_env

        # Asset in Plant 1
        asset_1 = equipment_registry.create_equipment(
            dataset_id="DATASET-PLANT-1",
            asset_name="Compressor C-1",
            category_code="compressor",
            subtype_code="screw",
        )
        incidents_df = pd.DataFrame([{"incident_id": "INC-COMP-1", "dataset_id": "DATASET-PLANT-1"}])
        equipment_registry.create_incident_association(
            dataset_id="DATASET-PLANT-1",
            incident_id="INC-COMP-1",
            equipment_id=asset_1["equipment_id"],
            incidents=incidents_df,
        )

        # Proprietary document belonging strictly to Plant 2
        ingestion_service.ingest_document(
            filename="Plant2_Compressor_Bulletin.txt",
            content=b"Proprietary Plant 2 screw compressor rotor mesh inspection bulletin.",
            title="Plant 2 Compressor Bulletin",
            publisher_author="Plant 2 Engineering",
            source_type=SourceType.INDUSTRY_BULLETIN,
            reliability_tier=ReliabilityTier.TIER_3_CURATED_RESEARCH,
            dataset_id="DATASET-PLANT-2",
            equipment_category="compressor",
            equipment_subtype="screw",
        )

        # Retrieve for Plant 1 -> must NOT return Plant 2 document
        hypo = _make_dummy_hypothesis("Rotor Mesh Failure", "Abnormal contact in rotor mesh.")
        analysis_result = _make_dummy_analysis_result("INC-COMP-1", [hypo])

        refs = advisor.retrieve_technical_references(
            incident_id="INC-COMP-1",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis_result,
        )

        assert all(r.title != "Plant 2 Compressor Bulletin" for r in refs)
        assert all(r.dataset_id != "DATASET-PLANT-2" for r in refs)

    def test_rejected_associations_are_ignored(self, advisor_env):
        _, equipment_registry, _, ingestion_service, _, advisor = advisor_env

        # Asset linked but subsequently REJECTED
        asset = equipment_registry.create_equipment(
            dataset_id="DATASET-PLANT-1",
            asset_name="Wrongly Linked Gearbox G-99",
            category_code="gearbox",
            subtype_code="planetary",
        )
        incidents_df = pd.DataFrame([{"incident_id": "INC-REJECT-1", "dataset_id": "DATASET-PLANT-1"}])
        assoc = equipment_registry.create_incident_association(
            dataset_id="DATASET-PLANT-1",
            incident_id="INC-REJECT-1",
            equipment_id=asset["equipment_id"],
            incidents=incidents_df,
        )
        # Mark rejected
        equipment_registry.review_incident_association(
            dataset_id="DATASET-PLANT-1",
            association_id=assoc["association_id"],
            decision="REJECTED",
            reviewer="Lead Engineer",
            review_notes="Incorrect asset",
        )

        # Ingest planetary gearbox bulletin
        ingestion_service.ingest_document(
            filename="Gearbox_Doc.txt",
            content=b"Planetary gearbox gear tooth spalling diagnostic guide.",
            title="Gearbox Guide",
            publisher_author="AGMA",
            source_type=SourceType.GENERAL_REFERENCE,
            reliability_tier=ReliabilityTier.TIER_4_GENERAL,
            dataset_id="DATASET-PLANT-1",
            equipment_category="gearbox",
        )

        hypo = _make_dummy_hypothesis("Tooth Spalling", "Surface fatigue on teeth.")
        analysis_result = _make_dummy_analysis_result("INC-REJECT-1", [hypo])

        refs = advisor.retrieve_technical_references(
            incident_id="INC-REJECT-1",
            dataset_id="DATASET-PLANT-1",
            analysis_result=analysis_result,
        )

        # Even if text matches query, match_level should NOT be boosted as associated equipment
        for r in refs:
            if r.title == "Gearbox Guide":
                assert r.match_level == EquipmentMatchLevel.CATEGORY_LEVEL

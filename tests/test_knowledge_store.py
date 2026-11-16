"""Tests for GhostFault M9.4.1 Engineering Knowledge Foundation."""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from src.knowledge_models import EngineeringReference, KnowledgeDocument, ReliabilityTier, SourceType
from src.knowledge_store import KnowledgeStore, KnowledgeStoreError
from src.models import DatasetOwnershipError
from src.storage import EvidenceStore


@pytest.fixture
def isolated_env(tmp_path: Path):
    """Fixture providing isolated EvidenceStore and KnowledgeStore in tmp_path."""
    evidence_dir = tmp_path / "evidence"
    evidence_store = EvidenceStore(base_dir=evidence_dir)
    evidence_store.ensure_default_dataset()
    evidence_store.create_dataset("DATASET-TEST-A", "Diagnostics Alpha")
    evidence_store.create_dataset("DATASET-TEST-B", "Diagnostics Beta")

    knowledge_db_path = tmp_path / "knowledge" / "knowledge_library.db"
    knowledge_store = KnowledgeStore(store=evidence_store, db_path=knowledge_db_path)
    return evidence_store, knowledge_store, tmp_path


class TestKnowledgeModels:
    """Test domain model validation, immutability, and separate reliability vs relevance."""

    def test_knowledge_document_dataclass_valid(self):
        doc = KnowledgeDocument(
            document_id="KNOW-DOC-001",
            title="ISO 10816-3 Mechanical Vibration Evaluation",
            publisher_author="International Organization for Standardization",
            source_type=SourceType.ENGINEERING_STANDARD,
            reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            publication_year=2009,
            edition_revision="Second Edition",
            equipment_category="electric_motor",
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            copyright_license="ISO Copyrighted; Reference only",
        )
        assert doc.document_id == "KNOW-DOC-001"
        assert doc.source_type == SourceType.ENGINEERING_STANDARD
        assert doc.reliability_tier == ReliabilityTier.TIER_1_STANDARD
        assert doc.dataset_id is None  # Global scope default

    def test_knowledge_document_invalid_types_raise(self):
        with pytest.raises(ValueError, match="requires a non-empty document_id"):
            KnowledgeDocument(
                document_id="",
                title="Some Standard",
                publisher_author="Author",
                source_type=SourceType.ENGINEERING_STANDARD,
                reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            )

        with pytest.raises(TypeError, match="source_type must be an instance"):
            KnowledgeDocument(
                document_id="KNOW-1",
                title="Title",
                publisher_author="Author",
                source_type="NOT_AN_ENUM",  # type: ignore
                reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            )

        with pytest.raises(TypeError, match="reliability_tier must be an instance"):
            KnowledgeDocument(
                document_id="KNOW-1",
                title="Title",
                publisher_author="Author",
                source_type=SourceType.OEM_MANUAL,
                reliability_tier="INVALID",  # type: ignore
            )

    def test_engineering_reference_separates_relevance_from_reliability(self):
        ref = EngineeringReference(
            document_id="KNOW-DOC-001",
            title="ISO 10816-3 Mechanical Vibration Evaluation",
            citation="ISO 10816-3:2009, Section 4.2, Table A.1",
            source_type=SourceType.ENGINEERING_STANDARD,
            source_reliability=ReliabilityTier.TIER_1_STANDARD,
            retrieval_relevance=0.884,
            page_reference="14-16",
            section_reference="4.2",
        )
        # Verify separation of inherent source authority from query match score
        assert ref.source_reliability == ReliabilityTier.TIER_1_STANDARD
        assert ref.retrieval_relevance == 0.884
        # Verify mandatory disclaimer is present
        assert "Contextual Reference Only" in ref.reference_disclaimer
        assert "verified factory incident evidence" in ref.reference_disclaimer


class TestKnowledgeStoreSchemaAndInit:
    """Test schema creation, versioning, and safety checks."""

    def test_schema_created_with_expected_tables_and_version(self, isolated_env):
        _, knowledge_store, tmp_path = isolated_env
        assert knowledge_store.db_path.exists()
        assert knowledge_store.integrity_check() == "ok"
        assert knowledge_store.foreign_key_check() == []

        # Connect directly to inspect user_version and tables
        with sqlite3.connect(knowledge_store.db_path) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == KnowledgeStore.SCHEMA_VERSION

            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "knowledge_documents" in tables

    def test_refuses_newer_schema_version(self, isolated_env):
        evidence_store, _, tmp_path = isolated_env
        db_path = tmp_path / "future_knowledge.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE knowledge_documents (document_id TEXT PRIMARY KEY)")
            conn.execute("PRAGMA user_version = 999")

        with pytest.raises(KnowledgeStoreError, match="newer than supported"):
            KnowledgeStore(store=evidence_store, db_path=db_path)

    def test_refuses_unversioned_dirty_database(self, isolated_env):
        evidence_store, _, tmp_path = isolated_env
        db_path = tmp_path / "dirty_knowledge.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE unmanaged_table (id INT)")
            # user_version remains 0

        with pytest.raises(KnowledgeStoreError, match="unversioned tables"):
            KnowledgeStore(store=evidence_store, db_path=db_path)


class TestKnowledgeStoreCRUDAndValidation:
    """Test document creation, listing, duplicate rejection, and input validations."""

    def test_add_and_get_global_document(self, isolated_env):
        _, knowledge_store, _ = isolated_env

        sample_hash = hashlib.sha256(b"ISO standard binary content").hexdigest()
        doc = knowledge_store.add_document(
            title="ISO 15243:2017 Rolling Bearings Damage and Failures",
            publisher_author="ISO/TC 4",
            source_type=SourceType.ENGINEERING_STANDARD,
            reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            dataset_id=None,  # Global
            publication_year=2017,
            edition_revision="Second Edition",
            equipment_category="electric_motor",
            equipment_subtype="induction",
            manufacturer="SKF/ISO",
            sha256_hash=sample_hash,
            copyright_license="ISO Copyrighted; Diagnostic reference only",
            rights_attribution="ISO standard classification of rolling bearing damages",
            is_proprietary_restricted=False,
        )

        assert doc.document_id.startswith("KNOW-")
        assert doc.title == "ISO 15243:2017 Rolling Bearings Damage and Failures"
        assert doc.dataset_id is None
        assert doc.source_type == SourceType.ENGINEERING_STANDARD
        assert doc.reliability_tier == ReliabilityTier.TIER_1_STANDARD
        assert doc.sha256_hash == sample_hash

        fetched = knowledge_store.get_document(doc.document_id)
        assert fetched is not None
        assert fetched.title == doc.title
        assert fetched.publisher_author == doc.publisher_author

    def test_add_and_list_dataset_specific_document(self, isolated_env):
        _, knowledge_store, _ = isolated_env

        oem_hash = hashlib.sha256(b"Flowserve Pump Manual").hexdigest()
        doc_a = knowledge_store.add_document(
            title="Flowserve Mark 3 Centrifugal Pump User Instructions",
            publisher_author="Flowserve Pump Division",
            source_type=SourceType.OEM_MANUAL,
            reliability_tier=ReliabilityTier.TIER_2_OEM,
            dataset_id="DATASET-TEST-A",
            publication_year=2021,
            equipment_category="pump",
            equipment_subtype="centrifugal",
            manufacturer="Flowserve",
            sha256_hash=oem_hash,
            copyright_license="OEM proprietary maintenance manual",
            is_proprietary_restricted=True,
        )

        assert doc_a.dataset_id == "DATASET-TEST-A"

        # List with dataset_id="DATASET-TEST-A" returns doc_a
        results_a = knowledge_store.list_documents(dataset_id="DATASET-TEST-A", include_global=False)
        assert len(results_a) == 1
        assert results_a[0].document_id == doc_a.document_id

        # List with dataset_id="DATASET-TEST-B" must NOT return doc_a (Strict isolation)
        results_b = knowledge_store.list_documents(dataset_id="DATASET-TEST-B", include_global=False)
        assert len(results_b) == 0

    def test_global_documents_visible_in_dataset_when_include_global_true(self, isolated_env):
        _, knowledge_store, _ = isolated_env

        # 1 global standard
        knowledge_store.add_document(
            title="API 610 Centrifugal Pumps for Petroleum Industries",
            publisher_author="American Petroleum Institute",
            source_type=SourceType.ENGINEERING_STANDARD,
            reliability_tier=ReliabilityTier.TIER_1_STANDARD,
            dataset_id=None,
        )

        # 1 dataset-specific manual in DATASET-TEST-A
        knowledge_store.add_document(
            title="Custom Site Pump Service Bulletin",
            publisher_author="Site Reliability Group",
            source_type=SourceType.INDUSTRY_BULLETIN,
            reliability_tier=ReliabilityTier.TIER_3_CURATED_RESEARCH,
            dataset_id="DATASET-TEST-A",
        )

        # Query DATASET-TEST-A with include_global=True -> should return 2 documents
        docs_a = knowledge_store.list_documents("DATASET-TEST-A", include_global=True)
        assert len(docs_a) == 2
        titles_a = {d.title for d in docs_a}
        assert "API 610 Centrifugal Pumps for Petroleum Industries" in titles_a
        assert "Custom Site Pump Service Bulletin" in titles_a

        # Query DATASET-TEST-B with include_global=True -> should return only the 1 global document
        docs_b = knowledge_store.list_documents("DATASET-TEST-B", include_global=True)
        assert len(docs_b) == 1
        assert docs_b[0].title == "API 610 Centrifugal Pumps for Petroleum Industries"

        # Query without dataset_id -> returns only global documents
        docs_global = knowledge_store.list_documents(dataset_id=None)
        assert len(docs_global) == 1
        assert docs_global[0].title == "API 610 Centrifugal Pumps for Petroleum Industries"

    def test_reject_invalid_dataset_id(self, isolated_env):
        _, knowledge_store, _ = isolated_env
        with pytest.raises(DatasetOwnershipError, match="references unregistered dataset"):
            knowledge_store.add_document(
                title="Unknown Dataset Manual",
                publisher_author="Author",
                source_type=SourceType.OEM_MANUAL,
                reliability_tier=ReliabilityTier.TIER_2_OEM,
                dataset_id="DATASET-DOES-NOT-EXIST",
            )

    def test_reject_duplicate_sha256_in_same_scope(self, isolated_env):
        _, knowledge_store, _ = isolated_env
        sample_hash = hashlib.sha256(b"Identical Manual").hexdigest()

        knowledge_store.add_document(
            title="Manual Copy 1",
            publisher_author="Author",
            source_type=SourceType.OEM_MANUAL,
            reliability_tier=ReliabilityTier.TIER_2_OEM,
            dataset_id="DATASET-TEST-A",
            sha256_hash=sample_hash,
        )

        with pytest.raises(KnowledgeStoreError, match="duplicate or inconsistent"):
            knowledge_store.add_document(
                title="Manual Copy 2",
                publisher_author="Author",
                source_type=SourceType.OEM_MANUAL,
                reliability_tier=ReliabilityTier.TIER_2_OEM,
                dataset_id="DATASET-TEST-A",
                sha256_hash=sample_hash,
            )

    def test_reject_invalid_metadata_fields(self, isolated_env):
        _, knowledge_store, _ = isolated_env

        # Empty title
        with pytest.raises(KnowledgeStoreError, match="Document title must contain 1 to 300 valid characters"):
            knowledge_store.add_document(
                title="",
                publisher_author="Author",
                source_type=SourceType.OEM_MANUAL,
                reliability_tier=ReliabilityTier.TIER_2_OEM,
            )

        # Title containing NUL byte
        with pytest.raises(KnowledgeStoreError, match="valid characters"):
            knowledge_store.add_document(
                title="Bad\x00Title",
                publisher_author="Author",
                source_type=SourceType.OEM_MANUAL,
                reliability_tier=ReliabilityTier.TIER_2_OEM,
            )

        # Invalid publication year
        with pytest.raises(KnowledgeStoreError, match="Publication year must be between"):
            knowledge_store.add_document(
                title="Ancient Standard",
                publisher_author="Author",
                source_type=SourceType.ENGINEERING_STANDARD,
                reliability_tier=ReliabilityTier.TIER_1_STANDARD,
                publication_year=1492,
            )

        # Malformed SHA-256 hash
        with pytest.raises(KnowledgeStoreError, match="SHA-256 hash must be a 64-character"):
            knowledge_store.add_document(
                title="Standard",
                publisher_author="Author",
                source_type=SourceType.ENGINEERING_STANDARD,
                reliability_tier=ReliabilityTier.TIER_1_STANDARD,
                sha256_hash="not_a_valid_sha256_hex_string",
            )

    def test_parameterized_queries_resist_sql_injection(self, isolated_env):
        _, knowledge_store, _ = isolated_env
        injection_title = "Injection'; DROP TABLE knowledge_documents; --"
        doc = knowledge_store.add_document(
            title=injection_title,
            publisher_author="Test Author",
            source_type=SourceType.GENERAL_REFERENCE,
            reliability_tier=ReliabilityTier.TIER_4_GENERAL,
        )
        assert doc.title == injection_title

        # Verify table still exists and integrity is fine
        assert knowledge_store.integrity_check() == "ok"
        fetched = knowledge_store.get_document(doc.document_id)
        assert fetched is not None
        assert fetched.title == injection_title

    def test_transaction_rollback_on_error(self, isolated_env):
        _, knowledge_store, _ = isolated_env
        # Attempt an insert that triggers an error during transaction
        # by passing an invalid custom document_id already taken
        doc1 = knowledge_store.add_document(
            title="Original",
            publisher_author="Author",
            source_type=SourceType.GENERAL_REFERENCE,
            reliability_tier=ReliabilityTier.TIER_4_GENERAL,
            document_id="DOC-COLLISION-1",
        )
        assert doc1.document_id == "DOC-COLLISION-1"

        with pytest.raises(KnowledgeStoreError, match="duplicate or inconsistent"):
            knowledge_store.add_document(
                title="Collision",
                publisher_author="Author",
                source_type=SourceType.GENERAL_REFERENCE,
                reliability_tier=ReliabilityTier.TIER_4_GENERAL,
                document_id="DOC-COLLISION-1",
            )

        # Ensure database remains uncorrupted
        assert knowledge_store.integrity_check() == "ok"

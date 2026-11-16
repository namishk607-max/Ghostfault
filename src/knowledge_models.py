"""Domain models for GhostFault M9.4 external engineering knowledge."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class SourceType(str, Enum):
    """Classification of external technical documentation sources."""

    OEM_MANUAL = "OEM_MANUAL"
    ENGINEERING_STANDARD = "ENGINEERING_STANDARD"
    PEER_REVIEWED_RESEARCH = "PEER_REVIEWED_RESEARCH"
    INDUSTRY_BULLETIN = "INDUSTRY_BULLETIN"
    GENERAL_REFERENCE = "GENERAL_REFERENCE"


class ReliabilityTier(str, Enum):
    """Authority and verification tier for external technical documentation."""

    TIER_1_STANDARD = "TIER_1_STANDARD"
    TIER_2_OEM = "TIER_2_OEM"
    TIER_3_CURATED_RESEARCH = "TIER_3_CURATED_RESEARCH"
    TIER_4_GENERAL = "TIER_4_GENERAL"


@dataclass(frozen=True)
class KnowledgeDocument:
    """A recorded external engineering document, standard, or manual."""

    document_id: str
    title: str
    publisher_author: str
    source_type: SourceType
    reliability_tier: ReliabilityTier
    dataset_id: Optional[str] = None  # None indicates Global technical scope
    publication_year: Optional[int] = None
    edition_revision: Optional[str] = None
    equipment_category: Optional[str] = None
    equipment_subtype: Optional[str] = None
    manufacturer: Optional[str] = None
    sha256_hash: Optional[str] = None
    copyright_license: Optional[str] = None
    rights_attribution: Optional[str] = None
    is_proprietary_restricted: bool = False
    ingested_at: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.document_id or not isinstance(self.document_id, str):
            raise ValueError("KnowledgeDocument requires a non-empty document_id.")
        if not self.title or not isinstance(self.title, str):
            raise ValueError("KnowledgeDocument requires a non-empty title.")
        if not self.publisher_author or not isinstance(self.publisher_author, str):
            raise ValueError("KnowledgeDocument requires a non-empty publisher_author.")
        if not isinstance(self.source_type, SourceType):
            raise TypeError("source_type must be an instance of SourceType.")
        if not isinstance(self.reliability_tier, ReliabilityTier):
            raise TypeError("reliability_tier must be an instance of ReliabilityTier.")


@dataclass(frozen=True)
class EngineeringReference:
    """Passage or citation reference into external engineering knowledge.

    Separates source reliability (inherent document authority) from
    retrieval relevance (similarity/BM25 score to current query).
    Always carries an explicit disclaimer that external knowledge is not
    incident evidence.
    """

    document_id: str
    title: str
    citation: str
    source_type: SourceType
    source_reliability: ReliabilityTier
    retrieval_relevance: float  # Score indicating relevance to query
    page_reference: Optional[str] = None
    section_reference: Optional[str] = None
    dataset_id: Optional[str] = None  # None indicates Global technical scope
    reference_disclaimer: str = (
        "Contextual Reference Only: External engineering knowledge, standards, "
        "and manuals provide technical baseline context and do not constitute "
        "verified factory incident evidence or root-cause determinations."
    )

    def __post_init__(self) -> None:
        if not self.document_id or not isinstance(self.document_id, str):
            raise ValueError("EngineeringReference requires a non-empty document_id.")
        if not self.citation or not isinstance(self.citation, str):
            raise ValueError("EngineeringReference requires a non-empty citation.")
        if not isinstance(self.source_type, SourceType):
            raise TypeError("source_type must be an instance of SourceType.")
        if not isinstance(self.source_reliability, ReliabilityTier):
            raise TypeError("source_reliability must be an instance of ReliabilityTier.")
        if not isinstance(self.retrieval_relevance, (int, float)):
            raise TypeError("retrieval_relevance must be a numeric score.")

"""Dedicated technical knowledge retrieval and advisory module for GhostFault M9.4.4.

Retrieves relevant technical manuals, standards, and research references for an incident,
conditioned on associated equipment assets (category, subtype, manufacturer) and
deterministic fault hypotheses, without altering deterministic analysis outcomes or
conflating engineering references with incident evidence.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Set

from src.equipment import EquipmentRegistry
from src.knowledge_models import EngineeringReference, ReliabilityTier, SourceType
from src.knowledge_search import KnowledgeSearchIndex, KnowledgeSearchResult
from src.models import FaultAnalysisResult, InvestigativeHypothesis


class EquipmentMatchLevel(str, Enum):
    """Indicates degree of specificity between an engineering document and associated equipment."""

    EXACT_MANUFACTURER_MODEL = "EXACT_MANUFACTURER_MODEL"
    EXACT_MANUFACTURER = "EXACT_MANUFACTURER"
    CATEGORY_AND_SUBTYPE = "CATEGORY_AND_SUBTYPE"
    CATEGORY_LEVEL = "CATEGORY_LEVEL"
    GENERAL_ENGINEERING = "GENERAL_ENGINEERING"


@dataclass(frozen=True)
class ContextualTechnicalReference:
    """An external engineering reference retrieved for an incident and its equipment."""

    document_id: str
    chunk_id: str
    title: str
    publisher_author: str
    source_type: SourceType
    reliability_tier: ReliabilityTier
    text: str
    relevance_score: float
    match_level: EquipmentMatchLevel
    dataset_id: Optional[str] = None
    estimated_page: Optional[int] = None
    section_heading: Optional[str] = None
    publication_year: Optional[int] = None
    equipment_category: Optional[str] = None
    equipment_subtype: Optional[str] = None
    manufacturer: Optional[str] = None
    matched_hypothesis_title: Optional[str] = None
    matched_keywords: Optional[str] = None
    reference_disclaimer: str = (
        "Contextual Reference Only: External engineering knowledge, standards, "
        "and manuals provide technical baseline context and do not constitute "
        "verified factory incident evidence or root-cause determinations."
    )

    def to_engineering_reference(self) -> EngineeringReference:
        """Convert into a formal EngineeringReference."""
        citation_parts = [self.title]
        if self.section_heading:
            citation_parts.append(self.section_heading)
        if self.estimated_page:
            citation_parts.append(f"p. {self.estimated_page}")
        citation_str = ", ".join(citation_parts)

        return EngineeringReference(
            document_id=self.document_id,
            title=self.title,
            citation=citation_str,
            source_type=self.source_type,
            source_reliability=self.reliability_tier,
            retrieval_relevance=round(self.relevance_score, 4),
            page_reference=str(self.estimated_page) if self.estimated_page else None,
            section_reference=self.section_heading,
            dataset_id=self.dataset_id,
            reference_disclaimer=self.reference_disclaimer,
        )


class TechnicalKnowledgeAdvisor:
    """Evaluates equipment context and hypothesis mechanisms to retrieve technical literature."""

    def __init__(
        self,
        search_index: KnowledgeSearchIndex,
        equipment_registry: EquipmentRegistry,
    ) -> None:
        if not isinstance(search_index, KnowledgeSearchIndex):
            raise TypeError("TechnicalKnowledgeAdvisor requires a KnowledgeSearchIndex.")
        if not isinstance(equipment_registry, EquipmentRegistry):
            raise TypeError("TechnicalKnowledgeAdvisor requires an EquipmentRegistry.")
        self.search_index = search_index
        self.equipment_registry = equipment_registry

    @staticmethod
    def _determine_match_level(
        doc: KnowledgeSearchResult,
        eq_category: Optional[str],
        eq_subtype: Optional[str],
        eq_manufacturer: Optional[str],
        eq_model: Optional[str],
    ) -> EquipmentMatchLevel:
        """Classify how specifically the technical document corresponds to the equipment asset."""
        doc_cat = (doc.equipment_category or "").strip().lower()
        doc_mfg = (doc.manufacturer or "").strip().lower()
        ast_cat = (eq_category or "").strip().lower()
        ast_sub = (eq_subtype or "").strip().lower()
        ast_mfg = (eq_manufacturer or "").strip().lower()

        # Check manufacturer match
        if ast_mfg and doc_mfg and (ast_mfg in doc_mfg or doc_mfg in ast_mfg):
            # Check model or subtype
            doc_sub = (doc.equipment_subtype or "").strip().lower()
            if ast_sub and doc_sub and ast_sub == doc_sub:
                return EquipmentMatchLevel.EXACT_MANUFACTURER_MODEL
            return EquipmentMatchLevel.EXACT_MANUFACTURER

        # Check category and subtype
        doc_sub = (doc.equipment_subtype or "").strip().lower()
        if ast_cat and doc_cat and ast_cat == doc_cat:
            if ast_sub and doc_sub and ast_sub == doc_sub:
                return EquipmentMatchLevel.CATEGORY_AND_SUBTYPE
            return EquipmentMatchLevel.CATEGORY_LEVEL

        if doc_cat:
            return EquipmentMatchLevel.CATEGORY_LEVEL

        return EquipmentMatchLevel.GENERAL_ENGINEERING

    @staticmethod
    def extract_hypothesis_terms(hypothesis: InvestigativeHypothesis) -> List[str]:
        """Extract searchable technical keywords from a deterministic hypothesis."""
        text = f"{hypothesis.title} {hypothesis.description}"
        # Filter common stopwords and short tokens
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
            "been", "have", "has", "had", "not", "but", "into", "over", "under", "also",
            "may", "can", "could", "should", "due", "cause", "caused", "causing", "potential",
            "possible", "indicates", "indicated", "indicating", "observed", "evidence", "check",
        }
        tokens = []
        for word in text.replace(".", " ").replace(",", " ").replace(";", " ").split():
            clean = word.strip().lower()
            if len(clean) > 3 and clean not in stopwords and clean.isalpha():
                tokens.append(clean)
        return tokens[:8]

    def retrieve_technical_references(
        self,
        incident_id: str,
        dataset_id: str,
        analysis_result: Optional[FaultAnalysisResult] = None,
        max_references: int = 6,
    ) -> List[ContextualTechnicalReference]:
        """Conditioned technical retrieval for an incident and its associated equipment."""
        if not incident_id or not incident_id.strip():
            return []
        if not dataset_id or not dataset_id.strip():
            return []

        # 1. Fetch reviewed or pending incident-equipment associations
        associations = self.equipment_registry.list_incident_associations(
            dataset_id=dataset_id,
            incident_id=incident_id.strip(),
        )

        associated_assets: List[Dict[str, Any]] = []
        for assoc in associations:
            # Respect REJECTED status: do not condition retrieval on rejected links
            if assoc.get("review_status") == "REJECTED":
                continue
            eq_id = assoc.get("equipment_id")
            if eq_id:
                asset = self.equipment_registry.get_equipment(dataset_id, eq_id)
                if asset:
                    associated_assets.append(asset)

        # 2. Extract technical query terms from hypotheses or general failure mechanism
        queries: List[Dict[str, Any]] = []

        if analysis_result and analysis_result.hypotheses:
            for hypo in analysis_result.hypotheses[:3]:
                terms = self.extract_hypothesis_terms(hypo)
                if terms:
                    # Individual key failure terms and bigram combinations
                    # Primary title term pair
                    query_str = " ".join(terms[:2])
                    queries.append({
                        "query": query_str,
                        "hypothesis_title": hypo.title,
                        "weight": 1.0,
                    })
                    # Also add individual specific terms if available
                    for term in terms[:3]:
                        queries.append({
                            "query": term,
                            "hypothesis_title": hypo.title,
                            "weight": 0.85,
                        })

        # Fallback if no hypotheses
        if not queries:
            queries.append({
                "query": "failure inspection maintenance standard",
                "hypothesis_title": None,
                "weight": 0.5,
            })

        # 3. Perform FTS5 searches across scopes
        seen_chunks: Set[str] = set()
        candidates: List[ContextualTechnicalReference] = []

        primary_asset = associated_assets[0] if associated_assets else None
        eq_category = primary_asset.get("category_code") if primary_asset else None
        eq_subtype = primary_asset.get("subtype_code") if primary_asset else None
        eq_mfg = primary_asset.get("manufacturer") if primary_asset else None
        eq_model = primary_asset.get("model") if primary_asset else None

        for q_item in queries:
            search_text = q_item["query"]
            # We search for the failure mechanism keywords directly, and if equipment context
            # is available, also search with equipment context tokens.
            # In FTS5, searching `eq_category search_text` requires BOTH the category slug and
            # the text to appear as words in the document chunk if using AND queries.
            # Therefore, we query the search text, and optionally the category-enhanced query,
            # merging matches.
            query_variants = [search_text]
            if eq_category and eq_category != "unclassified":
                query_variants.append(f"{eq_category} {search_text}")
            if eq_mfg:
                query_variants.append(f"{eq_mfg} {search_text}")

            for q_str in query_variants:
                # Execute FTS5 search (scoped to active dataset + global)
                matches = self.search_index.search(
                    query=q_str,
                    dataset_id=dataset_id,
                    include_global=True,
                    limit=max_references * 2,
                )

                for match in matches:
                    if match.chunk_id in seen_chunks:
                        continue
                    seen_chunks.add(match.chunk_id)

                    match_lvl = self._determine_match_level(
                        match,
                        eq_category=eq_category,
                        eq_subtype=eq_subtype,
                        eq_manufacturer=eq_mfg,
                        eq_model=eq_model,
                    )

                    # Score boost based on equipment match level
                    boost = 1.0
                    if match_lvl == EquipmentMatchLevel.EXACT_MANUFACTURER_MODEL:
                        boost = 1.6
                    elif match_lvl == EquipmentMatchLevel.EXACT_MANUFACTURER:
                        boost = 1.4
                    elif match_lvl == EquipmentMatchLevel.CATEGORY_AND_SUBTYPE:
                        boost = 1.25
                    elif match_lvl == EquipmentMatchLevel.CATEGORY_LEVEL:
                        boost = 1.1

                    final_score = match.score * boost * q_item["weight"]

                    candidates.append(
                        ContextualTechnicalReference(
                            document_id=match.document_id,
                            chunk_id=match.chunk_id,
                            title=match.title,
                            publisher_author=match.publisher_author,
                            source_type=match.source_type,
                            reliability_tier=match.reliability_tier,
                            text=match.text,
                            relevance_score=final_score,
                            match_level=match_lvl,
                            dataset_id=match.dataset_id,
                            estimated_page=match.estimated_page,
                            section_heading=match.section_heading,
                            publication_year=match.publication_year,
                            equipment_category=match.equipment_category,
                            equipment_subtype=match.equipment_subtype,
                            manufacturer=match.manufacturer,
                            matched_hypothesis_title=q_item["hypothesis_title"],
                            matched_keywords=q_item["query"],
                        )
                    )

        # 4. Rank candidates by final_score descending
        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return candidates[:max_references]

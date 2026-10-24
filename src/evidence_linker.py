"""Cross-incident evidence linking engine for GhostFault (Milestone 5.3).

Links cross-incident symptom groups to relevant evidence passages available through
GhostFault's local evidence retrieval and SQLite FTS5 search index.
Preserves complete provenance, deduplicates passages, distinguishes general guidance
from direct incident evidence, and excludes low-value passages without inferring root causes.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple

from src.models import (
    SymptomGroup,
    CrossIncidentComparisonResult,
    SymptomGroupEvidenceLink,
    GroupEvidenceLinkResult,
    CrossIncidentEvidenceLinkingResult,
    EvidenceLinkStatus,
    PassageQuality,
    EvidenceContextType,
    MachineScope,
    ComparisonLimitation,
    SearchResult,
    current_iso_timestamp,
)
from src.retriever import EvidenceRetriever
from src.analyzer import (
    classify_passage_quality,
    classify_evidence_context,
    extract_sentence_bounded_snippet,
    normalize_text_for_similarity,
)


class CrossIncidentEvidenceLinker:
    """
    Deterministic cross-incident evidence linking service.
    Connects symptom groups to verified local evidence passages.
    """

    def __init__(self, retriever: EvidenceRetriever):
        self.retriever = retriever

    @staticmethod
    def extract_group_query_terms(group: SymptomGroup) -> List[str]:
        """
        Extract meaningful technical keywords from a symptom group's canonical label
        and original symptom descriptions, filtering domain stopwords.
        """
        stopwords = {
            "and", "the", "with", "from", "for", "during", "under", "over", "into",
            "after", "before", "across", "within", "without", "resolved", "verified",
            "taken", "reported", "observed", "high", "recurrent", "primary", "secondary",
            "cycles", "delay", "drift", "drop", "trip", "pitch", "error"
        }

        # Collect text from canonical label and original symptoms
        corpus_parts = [group.canonical_label, group.normalized_symptom]
        for item in group.original_symptoms:
            corpus_parts.append(item.get("symptom", ""))
        for action in group.maintenance_actions:
            corpus_parts.append(action.get("maintenance_action") or "")

        combined_text = " ".join(corpus_parts)
        tokens = re.findall(r"[a-zA-Z0-9_\-]+", combined_text.lower())

        # Keep terms with length >= 4 that are not stopwords
        meaningful: List[str] = []
        for t in tokens:
            cleaned = t.strip("-").strip("_")
            if len(cleaned) >= 4 and cleaned not in stopwords and cleaned not in meaningful:
                meaningful.append(cleaned)

        return meaningful

    @staticmethod
    def inspect_group_machine_context(
        group_machine_ids: List[str],
        chunk_text: str,
    ) -> Tuple[MachineScope, Optional[str]]:
        """
        Evaluate if a retrieved chunk specifically cites any machine associated with the symptom group,
        another machine (conflict), or no machine at all (general guidance).
        """
        chunk_machines = re.findall(r"\bM-\d{3,}\b", chunk_text)
        if not chunk_machines:
            return MachineScope.GENERAL_REFERENCE, None

        # Filter out empty or unassigned group machine IDs
        valid_group_machines = [m for m in group_machine_ids if m and str(m).strip()]

        # Check if any chunk machine matches any group machine
        matched = [m for m in chunk_machines if m in valid_group_machines]
        if matched:
            conflicts = [m for m in chunk_machines if m not in valid_group_machines]
            if conflicts:
                return MachineScope.CROSS_MACHINE_CONFLICT, conflicts[0]
            return MachineScope.EXACT_MACHINE, matched[0]

        # Chunk mentions machines, but none match the group machines
        return MachineScope.CROSS_MACHINE_CONFLICT, chunk_machines[0]

    @staticmethod
    def generate_link_explanation(
        canonical_label: str,
        scope: MachineScope,
        context_type: EvidenceContextType,
        chunk_text: str,
        referenced_machine: Optional[str],
    ) -> Tuple[str, str]:
        """
        Generate a conservative relevance reason and link explanation.
        Never claims proof of root cause.
        Returns: (relevance_reason, link_explanation)
        """
        lower = chunk_text.lower()

        if scope == MachineScope.EXACT_MACHINE and referenced_machine:
            scope_desc = f"Direct equipment maintenance record for asset {referenced_machine}."
        elif scope == MachineScope.CROSS_MACHINE_CONFLICT and referenced_machine:
            scope_desc = f"Cross-machine reference mentioning asset {referenced_machine}."
        else:
            scope_desc = "General technical maintenance documentation."

        if "maintenance history" in lower or "preventive maintenance" in lower or "action:" in lower:
            rel = "References historical maintenance procedures, corrective actions, or inspection intervals."
        elif "specifications" in lower or "nominal" in lower or "limit" in lower:
            rel = "Discusses technical specifications, allowable tolerance thresholds, or operating limits."
        elif "symptom" in lower or "distress" in lower or "damage" in lower or "wear" in lower:
            rel = f"Documents technical mechanisms and observations related to '{canonical_label}'."
        else:
            rel = f"Provides relevant technical background corresponding to observed symptoms."

        if context_type == EvidenceContextType.ILLUSTRATIVE_TEST_VALUE:
            caution = " Caution: Contains illustrative test readings; not confirmed physical operational telemetry."
        elif context_type == EvidenceContextType.VERIFIED_OEM_LIMIT:
            caution = " Contains manufacturer OEM threshold limits."
        else:
            caution = ""

        relevance_reason = f"{scope_desc} {rel}{caution}"
        link_explanation = (
            f"Passage retrieved based on symptom group profile '{canonical_label}'. "
            f"Linked as contextual technical reference ({scope.value}); does not prove common root cause."
        )

        return relevance_reason, link_explanation

    @classmethod
    def deduplicate_evidence_links(
        cls,
        links: List[SymptomGroupEvidenceLink],
    ) -> List[SymptomGroupEvidenceLink]:
        """Group duplicate and near-duplicate passages under primary link."""
        if not links:
            return []

        primary_links: List[SymptomGroupEvidenceLink] = []
        normalized_seen: Dict[str, SymptomGroupEvidenceLink] = {}

        for link in links:
            norm = normalize_text_for_similarity(link.snippet)
            matched_primary: Optional[SymptomGroupEvidenceLink] = None

            if norm in normalized_seen:
                matched_primary = normalized_seen[norm]
            else:
                for existing_norm, existing_link in normalized_seen.items():
                    min_len = min(len(norm), len(existing_norm))
                    if min_len > 60:
                        shared_prefix = 0
                        for c1, c2 in zip(norm, existing_norm):
                            if c1 == c2:
                                shared_prefix += 1
                            else:
                                break
                        if shared_prefix / min_len > 0.80 or norm in existing_norm or existing_norm in norm:
                            matched_primary = existing_link
                            break

            if matched_primary is not None:
                link.is_duplicate = True
                link.duplicate_of_chunk_id = matched_primary.chunk_id
                copy_info = {
                    "document_id": link.document_id,
                    "source_filename": link.source_filename,
                    "chunk_id": link.chunk_id,
                    "chunk_index": str(link.chunk_index),
                }
                if matched_primary.duplicate_source_copies is None:
                    matched_primary.duplicate_source_copies = []
                matched_primary.duplicate_source_copies.append(copy_info)
            else:
                normalized_seen[norm] = link
                primary_links.append(link)

        return primary_links

    def link_symptom_group(
        self,
        group: SymptomGroup,
        max_results: int = 8,
    ) -> GroupEvidenceLinkResult:
        """
        Link an individual SymptomGroup to substantive evidence passages.
        Filters out search-term metadata, extracts readable sentence-bounded snippets,
        and records provenance limitations.
        """
        query_terms = self.extract_group_query_terms(group)
        group_limitations: List[str] = [
            "Evidence links indicate technical relevance only and do not establish shared root causes across incidents."
        ]

        if not query_terms:
            return GroupEvidenceLinkResult(
                group_id=group.group_id,
                canonical_label=group.canonical_label,
                normalized_symptom=group.normalized_symptom,
                incident_ids=group.incident_ids,
                recurrence_scope=group.recurrence_scope,
                status=EvidenceLinkStatus.NO_SUPPORTING_EVIDENCE,
                supporting_evidence=[],
                excluded_evidence=[],
                query_used="",
                raw_retrieved_count=0,
                substantive_count=0,
                excluded_count=0,
                limitations=group_limitations + ["No meaningful technical query terms could be derived from this symptom group."],
            )

        # Build bounded query
        query_str = " ".join(query_terms[:4])
        seen_chunk_ids: Set[str] = set()
        search_results: List[SearchResult] = []

        try:
            initial_results = self.retriever.search(query=query_str, limit=max_results)
            for r in initial_results:
                if r.chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(r.chunk.chunk_id)
                    search_results.append(r)
        except Exception:
            pass

        # If initial conjunction search yielded few results, also query key 2-term subqueries or individual terms
        if len(search_results) < max_results and len(query_terms) >= 2:
            # Query pairs like (terms[0], terms[1])
            pair_query = f"{query_terms[0]} {query_terms[1]}"
            try:
                pair_results = self.retriever.search(query=pair_query, limit=max_results)
                for r in pair_results:
                    if r.chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(r.chunk.chunk_id)
                        search_results.append(r)
            except Exception:
                pass

        if len(search_results) < max_results and len(query_terms) > 0:
            for term in query_terms[:4]:
                try:
                    term_results = self.retriever.search(query=term, limit=3)
                    for r in term_results:
                        if r.chunk.chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(r.chunk.chunk_id)
                            search_results.append(r)
                except Exception:
                    continue

        search_results = search_results[:max_results]
        raw_retrieved_count = len(search_results)

        if not search_results:
            return GroupEvidenceLinkResult(
                group_id=group.group_id,
                canonical_label=group.canonical_label,
                normalized_symptom=group.normalized_symptom,
                incident_ids=group.incident_ids,
                recurrence_scope=group.recurrence_scope,
                status=EvidenceLinkStatus.NO_SUPPORTING_EVIDENCE,
                supporting_evidence=[],
                excluded_evidence=[],
                query_used=query_str,
                raw_retrieved_count=0,
                substantive_count=0,
                excluded_count=0,
                limitations=group_limitations + ["No matching evidence passages found in local index for derived query terms."],
            )

        substantive_candidates: List[SymptomGroupEvidenceLink] = []
        excluded_candidates: List[SymptomGroupEvidenceLink] = []

        for sr in search_results:
            chunk = sr.chunk
            quality, exc_reason = classify_passage_quality(chunk.text)
            context_type = classify_evidence_context(chunk.text)
            scope, referenced_mach = self.inspect_group_machine_context(group.machine_ids, chunk.text)

            snippet_text, is_excerpt = extract_sentence_bounded_snippet(chunk.text, max_length=350)
            rel_reason, link_expl = self.generate_link_explanation(
                canonical_label=group.canonical_label,
                scope=scope,
                context_type=context_type,
                chunk_text=chunk.text,
                referenced_machine=referenced_mach,
            )

            # Check provenance completeness
            prov_limits: List[str] = []
            if not chunk.document_id or not chunk.source_filename:
                prov_limits.append("Missing source document identifier or filename.")
            if not chunk.chunk_id:
                prov_limits.append("Missing chunk identifier.")
            if chunk.start_offset is None or chunk.end_offset is None:
                prov_limits.append("Character offsets not indexed.")

            link_item = SymptomGroupEvidenceLink(
                group_id=group.group_id,
                related_incident_ids=list(group.incident_ids),
                document_id=chunk.document_id,
                source_filename=chunk.source_filename,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                snippet=snippet_text,
                relevance_reason=rel_reason,
                link_explanation=link_expl,
                machine_scope=scope,
                context_type=context_type,
                quality=quality,
                referenced_machine_id=referenced_mach,
                retrieval_score=round(sr.score, 3) if sr.score is not None else None,
                matched_query_terms=sr.matched_terms,
                provenance_limitations=prov_limits if prov_limits else None,
            )

            if quality == PassageQuality.SUBSTANTIVE_TECHNICAL:
                substantive_candidates.append(link_item)
            else:
                excluded_candidates.append(link_item)

        # Deduplicate substantive evidence without losing source provenance
        deduped_substantive = self.deduplicate_evidence_links(substantive_candidates)

        if not deduped_substantive:
            if excluded_candidates:
                status = EvidenceLinkStatus.INSUFFICIENT_SUBSTANTIVE_EVIDENCE
                group_limitations.append(
                    "All retrieved passages were non-substantive (search-term lists or metadata) and were excluded from linking."
                )
            else:
                status = EvidenceLinkStatus.NO_SUPPORTING_EVIDENCE
                group_limitations.append("No substantive evidence found.")
        else:
            status = EvidenceLinkStatus.SUPPORTED

        return GroupEvidenceLinkResult(
            group_id=group.group_id,
            canonical_label=group.canonical_label,
            normalized_symptom=group.normalized_symptom,
            incident_ids=group.incident_ids,
            recurrence_scope=group.recurrence_scope,
            status=status,
            supporting_evidence=deduped_substantive,
            excluded_evidence=excluded_candidates,
            query_used=query_str,
            raw_retrieved_count=raw_retrieved_count,
            substantive_count=len(deduped_substantive),
            excluded_count=len(excluded_candidates),
            limitations=group_limitations,
        )

    def link_comparison_evidence(
        self,
        comparison_result: CrossIncidentComparisonResult,
        max_results_per_group: int = 8,
    ) -> CrossIncidentEvidenceLinkingResult:
        """
        Execute cross-incident evidence linking across all symptom groups in a comparison result.
        """
        linking_id = f"EVLINK-{hashlib.sha256(str(current_iso_timestamp()).encode('utf-8')).hexdigest()[:12].upper()}"

        group_results: List[GroupEvidenceLinkResult] = []
        supported_count = 0
        unsupported_count = 0

        for group in comparison_result.symptom_groups:
            res = self.link_symptom_group(group, max_results=max_results_per_group)
            group_results.append(res)
            if res.status == EvidenceLinkStatus.SUPPORTED:
                supported_count += 1
            else:
                unsupported_count += 1

        global_limitations: List[ComparisonLimitation] = [
            ComparisonLimitation(
                code="NO_SHARED_ROOT_CAUSE",
                description=(
                    "Evidence linked to cross-incident symptom groups provides technical background and mechanism context, "
                    "but does not prove or establish that different incidents share an identical root cause."
                ),
            ),
            ComparisonLimitation(
                code="GENERAL_GUIDANCE_BOUNDARY",
                description=(
                    "General maintenance manuals and guidelines describe OEM recommended procedures and possible failure "
                    "mechanisms; they must not be represented as physical telemetry from a specific incident."
                ),
            ),
        ]

        return CrossIncidentEvidenceLinkingResult(
            linking_id=linking_id,
            comparison_id=comparison_result.comparison_id,
            total_groups_evaluated=len(comparison_result.symptom_groups),
            supported_groups_count=supported_count,
            unsupported_groups_count=unsupported_count,
            group_results=group_results,
            global_limitations=global_limitations,
            linked_at=current_iso_timestamp(),
        )

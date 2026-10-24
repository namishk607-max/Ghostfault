"""Cross-incident comparison and recurrence analysis engine for GhostFault (Milestone 5.1).

Provides deterministic comparison of maintenance incident records to identify recurring
symptom patterns across machines or on the same machine, without inferring root causes
or fabricating missing fields.
"""

import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import pandas as pd

from src.models import (
    CrossIncidentComparisonResult,
    SymptomGroup,
    IncidentTimelineEntry,
    ComparisonLimitation,
    RecurrenceScope,
    current_iso_timestamp,
)


def normalize_symptom_text(text: Optional[str]) -> Tuple[str, List[str]]:
    """
    Conservatively normalize symptom text for deterministic grouping.

    Rules:
    - Lowercase and trim excess whitespace.
    - Tokenize composite descriptions split by connectors (" and ", " & ", ";", ",").
    - Strip punctuation and trailing periods.
    - Keep technical specifiers intact (e.g. 'high bearing temperature' vs 'bearing noise').
    - Sort tokens within atomic symptom phrases where word ordering is cosmetic
      (e.g., 'elevated temperature bearing' vs 'bearing elevated temperature'),
      while preserving distinct domain qualifiers.

    Returns:
        (canonical_key, list_of_sub_symptom_keys)
    """
    if not text or not str(text).strip():
        return ("", [])

    cleaned = str(text).strip().lower()

    # Split compound symptoms on natural delimiters
    parts = re.split(r"\s*(?:and|&|;|,)\s*", cleaned)
    sub_keys = []

    for part in parts:
        # Remove non-alphanumeric punctuation except hyphens inside words
        cleaned_part = re.sub(r"[^\w\s\-]", " ", part)
        tokens = [t.strip() for t in cleaned_part.split() if t.strip()]
        if not tokens:
            continue

        # Canonicalize common synonymous adjective modifiers without losing technical specificity
        synonym_map = {
            "elevated": "high",
            "excessive": "high",
            "overheating": "high",
            "abnormal": "unusual",
            "irregular": "unusual",
            "recurrent": "recurring",
            "intermittent": "periodic",
        }
        mapped_tokens = [synonym_map.get(tok, tok) for tok in tokens]

        # Form a stable ordered key for this sub-symptom phrase
        # E.g. 'high bearing temperature' -> 'bearing high temperature'
        sorted_tokens = sorted(mapped_tokens)
        sub_key = " ".join(sorted_tokens)
        sub_keys.append(sub_key)

    # Sort compound sub-keys to ensure order of 'A and B' matches 'B and A'
    unique_sub_keys = sorted(list(set(sub_keys)))
    canonical_key = " + ".join(unique_sub_keys)
    return (canonical_key, unique_sub_keys)


def parse_incident_date(date_val: Any) -> Tuple[Optional[str], bool]:
    """
    Safely parse and validate an incident date string into ISO YYYY-MM-DD.
    Preserves raw value without guessing or inventing dates.

    Returns:
        (parsed_iso_date_or_none, is_valid)
    """
    if date_val is None or pd.isna(date_val):
        return (None, False)

    date_str = str(date_val).strip()
    if not date_str:
        return (None, False)

    # Common valid formats: YYYY-MM-DD, YYYY/MM/DD, DD-MM-YYYY, etc.
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return (dt.strftime("%Y-%m-%d"), True)
        except ValueError:
            continue

    return (None, False)


def compute_recurrence_scope(machine_ids: List[Optional[str]]) -> RecurrenceScope:
    """
    Determine recurrence scope across a collection of machine IDs.
    Strictly distinguishes:
    - SINGLETON: Only 1 occurrence.
    - SAME_MACHINE: >= 2 occurrences all on the same known machine.
    - CROSS_MACHINE: >= 2 occurrences spanning >= 2 different known machines.
    - MIXED: Occurrences spanning known machine(s) and unknown/missing machine(s).
    - UNDETERMINED: All machine IDs missing/unknown.
    """
    if len(machine_ids) <= 1:
        return RecurrenceScope.SINGLETON

    known_machines = [m for m in machine_ids if m is not None and str(m).strip() and str(m).strip().lower() != "nan"]
    missing_count = len(machine_ids) - len(known_machines)
    unique_known = set(known_machines)

    if not unique_known:
        return RecurrenceScope.UNDETERMINED

    if len(unique_known) == 1:
        if missing_count > 0:
            return RecurrenceScope.MIXED
        return RecurrenceScope.SAME_MACHINE

    # More than 1 distinct known machine
    return RecurrenceScope.CROSS_MACHINE


class CrossIncidentComparisonEngine:
    """
    Deterministic comparison engine for cross-incident analysis.
    Identifies recurring symptoms, evaluates timeline-readiness, and reports data limitations.
    """

    def __init__(self):
        pass

    def compare_incidents(
        self,
        incidents: Union[List[Dict[str, Any]], pd.DataFrame],
    ) -> CrossIncidentComparisonResult:
        """
        Compare a list or DataFrame of incident records deterministically.
        """
        if isinstance(incidents, pd.DataFrame):
            records = incidents.to_dict(orient="records")
        elif isinstance(incidents, list):
            records = incidents
        else:
            records = []

        comparison_id = f"CMP-{hashlib.sha256(str(current_iso_timestamp()).encode('utf-8')).hexdigest()[:12].upper()}"

        # Initialize counters for missing fields
        missing_counts = {
            "missing_machine_id": 0,
            "missing_or_invalid_date": 0,
            "missing_symptom": 0,
            "missing_maintenance_action": 0,
            "missing_outcome": 0,
        }

        timeline_entries: List[IncidentTimelineEntry] = []
        compared_ids: List[str] = []

        # Intermediate grouping structure: normalized_key -> list of incident info
        symptom_buckets: Dict[str, List[Dict[str, Any]]] = {}

        for rec in records:
            inc_id = str(rec.get("incident_id") or "").strip()
            if not inc_id or inc_id.lower() == "nan":
                inc_id = f"UNKNOWN-INC-{len(timeline_entries) + 1}"
            compared_ids.append(inc_id)

            # Machine ID handling
            raw_machine = rec.get("machine_id")
            if raw_machine is None or pd.isna(raw_machine) or not str(raw_machine).strip() or str(raw_machine).strip().lower() == "nan":
                clean_machine = None
                missing_counts["missing_machine_id"] += 1
            else:
                clean_machine = str(raw_machine).strip()

            # Date handling
            raw_date = rec.get("date")
            parsed_date, is_valid_date = parse_incident_date(raw_date)
            if not is_valid_date:
                missing_counts["missing_or_invalid_date"] += 1

            # Symptom handling
            raw_symptom = rec.get("symptom")
            if raw_symptom is None or pd.isna(raw_symptom) or not str(raw_symptom).strip() or str(raw_symptom).strip().lower() == "nan":
                clean_symptom = "Unspecified symptom"
                missing_counts["missing_symptom"] += 1
            else:
                clean_symptom = str(raw_symptom).strip()

            # Maintenance action
            raw_action = rec.get("maintenance_action")
            if raw_action is None or pd.isna(raw_action) or not str(raw_action).strip() or str(raw_action).strip().lower() == "nan":
                clean_action = None
                missing_counts["missing_maintenance_action"] += 1
            else:
                clean_action = str(raw_action).strip()

            # Outcome
            raw_outcome = rec.get("outcome")
            if raw_outcome is None or pd.isna(raw_outcome) or not str(raw_outcome).strip() or str(raw_outcome).strip().lower() == "nan":
                clean_outcome = None
                missing_counts["missing_outcome"] += 0 # not strictly counted as core missing if optional, but let's record
                if raw_outcome is None or pd.isna(raw_outcome):
                    missing_counts["missing_outcome"] += 1
            else:
                clean_outcome = str(raw_outcome).strip()

            timeline_entry = IncidentTimelineEntry(
                incident_id=inc_id,
                machine_id=clean_machine,
                raw_date=str(raw_date) if raw_date is not None and not pd.isna(raw_date) else None,
                parsed_date=parsed_date,
                is_date_valid=is_valid_date,
                symptom=clean_symptom,
                maintenance_action=clean_action,
                outcome=clean_outcome,
                original_record=dict(rec),
            )
            timeline_entries.append(timeline_entry)

            # Normalize symptom
            norm_key, sub_keys = normalize_symptom_text(clean_symptom)
            if not norm_key:
                norm_key = "unspecified_symptom"

            if norm_key not in symptom_buckets:
                symptom_buckets[norm_key] = []

            symptom_buckets[norm_key].append({
                "incident_id": inc_id,
                "machine_id": clean_machine,
                "raw_symptom": clean_symptom,
                "maintenance_action": clean_action,
                "raw_record": rec,
            })

        # Sort timeline entries deterministically: valid parsed date first (asc), then raw date, then incident_id
        timeline_entries.sort(
            key=lambda t: (
                0 if t.is_date_valid else 1,
                t.parsed_date or "",
                t.incident_id,
            )
        )

        # Build SymptomGroup objects
        symptom_groups: List[SymptomGroup] = []
        for norm_key, items in symptom_buckets.items():
            inc_ids = [item["incident_id"] for item in items]
            mach_ids = [item["machine_id"] for item in items]
            orig_symptoms = [
                {"incident_id": item["incident_id"], "symptom": item["raw_symptom"]}
                for item in items
            ]
            maint_actions = [
                {"incident_id": item["incident_id"], "maintenance_action": item["maintenance_action"]}
                for item in items
            ]

            scope = compute_recurrence_scope(mach_ids)

            # Canonical label from most descriptive or first raw symptom
            canonical_label = items[0]["raw_symptom"]

            if len(items) > 1:
                basis = (
                    f"Normalized symptom equivalence: shared canonical symptom profile '{norm_key}' "
                    f"observed across {len(items)} incidents."
                )
            else:
                basis = f"Single incident observation with distinct symptom profile: '{norm_key}'."

            group_id = f"SYMGRP-{hashlib.md5(norm_key.encode('utf-8')).hexdigest()[:8].upper()}"

            group = SymptomGroup(
                group_id=group_id,
                normalized_symptom=norm_key,
                canonical_label=canonical_label,
                incident_ids=inc_ids,
                machine_ids=[m for m in mach_ids if m is not None],
                recurrence_scope=scope,
                occurrences_count=len(items),
                grouping_basis=basis,
                original_symptoms=orig_symptoms,
                maintenance_actions=maint_actions,
            )
            symptom_groups.append(group)

        # Sort symptom groups: recurrence count descending, then normalized symptom name
        symptom_groups.sort(key=lambda g: (-g.occurrences_count, g.normalized_symptom))

        # Compile explicit limitations
        limitations: List[ComparisonLimitation] = [
            ComparisonLimitation(
                code="NO_ROOT_CAUSE_INFERENCE",
                description=(
                    "Symptom co-occurrence and recurrence across incidents indicate shared operational distress "
                    "patterns, but do not confirm identical root causes without independent physical inspection."
                ),
            ),
            ComparisonLimitation(
                code="CONSERVATIVE_GROUPING",
                description=(
                    "Symptoms sharing single isolated keywords (e.g. 'bearing') are intentionally kept distinct "
                    "unless technical descriptors and failure modes are explicitly equivalent."
                ),
            ),
        ]

        if missing_counts["missing_machine_id"] > 0:
            limitations.append(
                ComparisonLimitation(
                    code="UNASSIGNED_MACHINE_RECORDS",
                    description=(
                        f"{missing_counts['missing_machine_id']} record(s) lack a verified machine identifier; "
                        "these cannot be attributed to same-machine recurrence."
                    ),
                )
            )

        if missing_counts["missing_or_invalid_date"] > 0:
            limitations.append(
                ComparisonLimitation(
                    code="UNVERIFIED_TEMPORAL_ORDERING",
                    description=(
                        f"{missing_counts['missing_or_invalid_date']} record(s) contain missing or unparseable date values; "
                        "timeline chronological sequence may be incomplete."
                    ),
                )
            )

        if missing_counts["missing_maintenance_action"] > 0:
            limitations.append(
                ComparisonLimitation(
                    code="INCOMPLETE_MAINTENANCE_HISTORY",
                    description=(
                        f"{missing_counts['missing_maintenance_action']} record(s) lack recorded maintenance actions."
                    ),
                )
            )

        return CrossIncidentComparisonResult(
            comparison_id=comparison_id,
            compared_incident_ids=compared_ids,
            total_incidents_analyzed=len(records),
            symptom_groups=symptom_groups,
            timeline_entries=timeline_entries,
            missing_fields_summary=missing_counts,
            limitations=limitations,
            compared_at=current_iso_timestamp(),
        )


def generate_incident_timeline(
    incidents: Union[List[Dict[str, Any]], pd.DataFrame]
) -> Tuple[List[IncidentTimelineEntry], List[IncidentTimelineEntry], Dict[str, int]]:
    """
    Deterministically generate chronological timeline entries from incident records.
    Separates dated incidents (sorted by date ascending, then incident ID) from undated incidents.

    Returns:
        (dated_entries, undated_entries, missing_counts_summary)
    """
    engine = CrossIncidentComparisonEngine()
    result = engine.compare_incidents(incidents)

    dated = [entry for entry in result.timeline_entries if entry.is_date_valid]
    undated = [entry for entry in result.timeline_entries if not entry.is_date_valid]

    # Explicit deterministic sort for dated: parsed_date ascending, then incident_id ascending
    dated.sort(key=lambda x: (x.parsed_date or "", x.incident_id))

    # Undated: sorted by raw_date (if present) then incident_id ascending
    undated.sort(key=lambda x: (x.raw_date or "", x.incident_id))

    return (dated, undated, result.missing_fields_summary)


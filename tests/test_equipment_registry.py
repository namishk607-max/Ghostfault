import sqlite3
import uuid
from datetime import date, datetime

import pandas as pd
import pytest

from src.equipment import EquipmentRegistry, EquipmentRegistryError
from src.equipment_taxonomy import (
    CLASSIFICATION_CANDIDATE,
    CLASSIFICATION_UNCLASSIFIED,
    EQUIPMENT_TAXONOMY,
    TAXONOMY_VERSION,
)
from src.models import Dataset, DatasetOwnershipError, DEFAULT_DATASET_ID
from src.storage import EvidenceStore


@pytest.fixture
def equipment_env(tmp_path):
    evidence_store = EvidenceStore(base_dir=tmp_path / "evidence")
    evidence_store.ensure_default_dataset()
    evidence_store.save_datasets(
        evidence_store.list_datasets()
        + [
            Dataset(
                dataset_id="DATASET-A",
                name="Dataset A",
                description="Isolated test dataset A",
                created_at="2026-01-01T00:00:00+00:00",
            ),
            Dataset(
                dataset_id="DATASET-B",
                name="Dataset B",
                description="Isolated test dataset B",
                created_at="2026-01-01T00:00:00+00:00",
            ),
        ]
    )
    registry = EquipmentRegistry(
        evidence_store,
        db_path=tmp_path / "equipment" / "equipment_registry.db",
    )
    incidents = pd.DataFrame(
        [
            {
                "incident_id": "INC-1",
                "machine_id": "M-104",
                "dataset_id": DEFAULT_DATASET_ID,
                "symptom": "Temperature increased",
            },
            {
                "incident_id": "INC-2",
                "machine_id": "M-205",
                "dataset_id": "DATASET-A",
                "symptom": "Vibration increased",
            },
        ]
    )
    return registry, evidence_store, incidents, tmp_path


def _create_asset(registry, dataset_id=DEFAULT_DATASET_ID, **overrides):
    values = {
        "dataset_id": dataset_id,
        "asset_name": "Drive motor",
        "category_code": "electric_motor",
        "subtype_code": "induction",
    }
    values.update(overrides)
    return registry.create_equipment(**values)


def test_asset_creation_and_retrieval_use_versioned_candidate_taxonomy(equipment_env):
    registry, _store, _incidents, tmp_path = equipment_env
    asset = _create_asset(
        registry,
        manufacturer="Example Works",
        model="EM-100",
        serial_number="SN-1",
        asset_tag="ASSET-1",
        legacy_machine_id="M-104",
        installation_date=date(2020, 5, 1),
        location="Line 4 / Bay 2",
        criticality="high",
    )

    assert str(uuid.UUID(asset["equipment_id"])) == asset["equipment_id"]
    assert asset["dataset_id"] == DEFAULT_DATASET_ID
    assert asset["installation_date"] == "2020-05-01"
    assert asset["criticality"] == "HIGH"
    assert asset["taxonomy_version"] == TAXONOMY_VERSION
    assert asset["classification_status"] == CLASSIFICATION_CANDIDATE
    assert registry.get_equipment(DEFAULT_DATASET_ID, asset["equipment_id"]) == asset
    assert registry.integrity_check() == "ok"
    assert registry.foreign_key_check() == []
    assert (tmp_path / "equipment" / "equipment_registry.db").exists()


def test_optional_metadata_can_be_absent_and_equipment_can_be_unclassified(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(
        registry,
        asset_name="Unclassified asset",
        category_code="unclassified",
        subtype_code=None,
    )

    assert asset["subtype_code"] is None
    assert asset["manufacturer"] is None
    assert asset["serial_number"] is None
    assert asset["installation_date"] is None
    assert asset["location"] is None
    assert asset["criticality"] is None
    assert asset["classification_status"] == CLASSIFICATION_UNCLASSIFIED


def test_equipment_update_changes_only_permitted_fields(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry, manufacturer="Maker")

    updated = registry.update_equipment(
        DEFAULT_DATASET_ID,
        asset["equipment_id"],
        {"model": "EM-200", "location": "Line 5", "serial_number": None},
    )

    assert updated["model"] == "EM-200"
    assert updated["location"] == "Line 5"
    assert updated["serial_number"] is None
    assert updated["manufacturer"] == "Maker"
    assert updated["updated_at"] >= asset["updated_at"]
    with pytest.raises(EquipmentRegistryError):
        registry.update_equipment(
            DEFAULT_DATASET_ID,
            asset["equipment_id"],
            {"dataset_id": "DATASET-A"},
        )


def test_asset_reads_and_writes_are_dataset_scoped(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry, dataset_id="DATASET-A")

    assert registry.list_equipment(DEFAULT_DATASET_ID) == []
    assert registry.get_equipment(DEFAULT_DATASET_ID, asset["equipment_id"]) is None
    with pytest.raises(EquipmentRegistryError):
        registry.update_equipment(
            DEFAULT_DATASET_ID,
            asset["equipment_id"],
            {"asset_name": "Cross-scope"},
        )
    with pytest.raises(DatasetOwnershipError):
        registry.list_equipment("DATASET-NOT-REGISTERED")


def test_duplicate_asset_tag_and_same_dataset_identity_are_rejected_but_other_dataset_isolated(
    equipment_env,
):
    registry, _store, _incidents, _tmp = equipment_env
    _create_asset(registry, dataset_id="DATASET-A", asset_tag="Tag-A")
    with pytest.raises(EquipmentRegistryError):
        _create_asset(registry, dataset_id="DATASET-A", asset_tag="tag-a")

    _create_asset(
        registry,
        dataset_id="DATASET-A",
        asset_name="Serial asset",
        asset_tag=None,
        manufacturer="Maker",
        serial_number="SERIAL-1",
    )
    with pytest.raises(EquipmentRegistryError):
        _create_asset(
            registry,
            dataset_id="DATASET-A",
            asset_name="Duplicate serial",
            asset_tag=None,
            manufacturer="maker",
            serial_number="serial-1",
        )

    other_dataset_asset = _create_asset(
        registry,
        dataset_id="DATASET-B",
        asset_name="Same physical-looking serial",
        manufacturer="Maker",
        serial_number="SERIAL-1",
    )
    assert other_dataset_asset["dataset_id"] == "DATASET-B"


@pytest.mark.parametrize(
    "values",
    [
        {"asset_name": " "},
        {"category_code": "unknown_category"},
        {"category_code": "pump", "subtype_code": "induction"},
        {"installation_date": "2026-02-30"},
        {"installation_date": datetime(2026, 1, 1)},
        {"criticality": "EXTREME"},
    ],
)
def test_invalid_equipment_fields_are_rejected(equipment_env, values):
    registry, _store, _incidents, _tmp = equipment_env
    base = {
        "asset_name": "Test equipment",
        "category_code": "electric_motor",
        "subtype_code": "induction",
    }
    base.update(values)
    with pytest.raises(EquipmentRegistryError):
        registry.create_equipment(DEFAULT_DATASET_ID, **base)


def test_candidate_categories_and_taxonomy_version_are_not_marked_validated(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env

    assert {
        code: definition["status"]
        for code, definition in EQUIPMENT_TAXONOMY.items()
        if code != "unclassified"
    } == {
        "electric_motor": CLASSIFICATION_CANDIDATE,
        "pump": CLASSIFICATION_CANDIDATE,
        "gearbox": CLASSIFICATION_CANDIDATE,
        "compressor": CLASSIFICATION_CANDIDATE,
    }
    pump = _create_asset(
        registry,
        dataset_id="DATASET-A",
        asset_name="Pump 1",
        category_code="pump",
        subtype_code="centrifugal",
    )
    assert pump["taxonomy_version"] == TAXONOMY_VERSION
    assert pump["classification_status"] == CLASSIFICATION_CANDIDATE
    with pytest.raises(EquipmentRegistryError):
        registry.create_equipment(
            "DATASET-A",
            "Unknown category",
            "fan",
        )


def test_component_creation_and_parent_child_hierarchy(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    parent = registry.create_component(
        DEFAULT_DATASET_ID,
        asset["equipment_id"],
        "Motor bearings",
        component_code="bearings",
    )
    child = registry.create_component(
        DEFAULT_DATASET_ID,
        asset["equipment_id"],
        "Drive-end bearing",
        parent_component_id=parent["component_id"],
        manufacturer="Bearing Co",
    )

    assert child["parent_component_id"] == parent["component_id"]
    assert child["classification_status"] == CLASSIFICATION_UNCLASSIFIED
    assert {row["component_id"] for row in registry.list_components(
        DEFAULT_DATASET_ID, asset["equipment_id"]
    )} == {parent["component_id"], child["component_id"]}


def test_component_taxonomy_rejects_unknown_type_but_allows_unclassified(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry)

    unknown = registry.create_component(
        DEFAULT_DATASET_ID,
        asset["equipment_id"],
        "Custom accessory",
    )
    assert unknown["component_code"] is None
    assert unknown["classification_status"] == CLASSIFICATION_UNCLASSIFIED
    with pytest.raises(EquipmentRegistryError):
        registry.create_component(
            DEFAULT_DATASET_ID,
            asset["equipment_id"],
            "Unsupported part",
            component_code="impeller",
        )


def test_component_parent_must_be_on_same_equipment_and_dataset(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    first = _create_asset(registry, dataset_id="DATASET-A")
    second = _create_asset(
        registry,
        dataset_id="DATASET-A",
        asset_name="Second motor",
        asset_tag="MOTOR-2",
    )
    foreign_dataset = _create_asset(registry, dataset_id="DATASET-B")
    parent = registry.create_component(
        "DATASET-A", first["equipment_id"], "Stator", component_code="stator"
    )

    with pytest.raises(EquipmentRegistryError):
        registry.create_component(
            "DATASET-A",
            second["equipment_id"],
            "Child of another asset",
            parent_component_id=parent["component_id"],
        )
    with pytest.raises(EquipmentRegistryError):
        registry.create_component(
            "DATASET-B",
            foreign_dataset["equipment_id"],
            "Cross-dataset parent",
            parent_component_id=parent["component_id"],
        )
    with pytest.raises(EquipmentRegistryError):
        registry.list_components("DATASET-B", first["equipment_id"])


def test_component_self_parent_and_cycles_are_rejected(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    root = registry.create_component(
        DEFAULT_DATASET_ID, asset["equipment_id"], "Stator", component_code="stator"
    )
    child = registry.create_component(
        DEFAULT_DATASET_ID,
        asset["equipment_id"],
        "Subassembly",
        parent_component_id=root["component_id"],
    )

    with pytest.raises(EquipmentRegistryError, match="own parent"):
        registry.update_component(
            DEFAULT_DATASET_ID,
            asset["equipment_id"],
            root["component_id"],
            {"parent_component_id": root["component_id"]},
        )
    with pytest.raises(EquipmentRegistryError, match="cycles"):
        registry.update_component(
            DEFAULT_DATASET_ID,
            asset["equipment_id"],
            root["component_id"],
            {"parent_component_id": child["component_id"]},
        )


def test_component_updates_revalidate_category_and_parent(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    stator = registry.create_component(
        DEFAULT_DATASET_ID, asset["equipment_id"], "Stator", component_code="stator"
    )
    updated = registry.update_component(
        DEFAULT_DATASET_ID,
        asset["equipment_id"],
        stator["component_id"],
        {"component_name": "Rewound stator", "manufacturer": "Maker"},
    )
    assert updated["component_name"] == "Rewound stator"
    assert updated["manufacturer"] == "Maker"
    with pytest.raises(EquipmentRegistryError):
        registry.update_component(
            DEFAULT_DATASET_ID,
            asset["equipment_id"],
            stator["component_id"],
            {"component_code": "impeller"},
        )


def test_deletion_is_restricted_for_assets_with_components_or_associations(equipment_env):
    registry, _store, incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    component = registry.create_component(
        DEFAULT_DATASET_ID, asset["equipment_id"], "Stator", component_code="stator"
    )
    with pytest.raises(EquipmentRegistryError, match="components"):
        registry.delete_equipment(DEFAULT_DATASET_ID, asset["equipment_id"])
    registry.delete_component(
        DEFAULT_DATASET_ID, asset["equipment_id"], component["component_id"]
    )
    registry.create_incident_association(
        DEFAULT_DATASET_ID, "INC-1", asset["equipment_id"], incidents
    )
    with pytest.raises(EquipmentRegistryError, match="associations"):
        registry.delete_equipment(DEFAULT_DATASET_ID, asset["equipment_id"])


def test_component_deletion_is_restricted_when_children_exist(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    parent = registry.create_component(
        DEFAULT_DATASET_ID, asset["equipment_id"], "Stator", component_code="stator"
    )
    registry.create_component(
        DEFAULT_DATASET_ID,
        asset["equipment_id"],
        "Stator winding",
        parent_component_id=parent["component_id"],
    )
    with pytest.raises(EquipmentRegistryError, match="child components"):
        registry.delete_component(
            DEFAULT_DATASET_ID, asset["equipment_id"], parent["component_id"]
        )


def test_legacy_machine_id_does_not_create_equipment_or_association_automatically(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry, legacy_machine_id="M-104")

    assert registry.list_equipment(DEFAULT_DATASET_ID) == [asset]
    assert registry.list_incident_associations(DEFAULT_DATASET_ID) == []


def test_incident_association_requires_explicit_selection_and_starts_pending(equipment_env):
    registry, _store, incidents, _tmp = equipment_env
    asset = _create_asset(registry)

    association = registry.create_incident_association(
        DEFAULT_DATASET_ID,
        "INC-1",
        asset["equipment_id"],
        incidents,
        role="PRIMARY",
    )

    assert association["incident_id"] == "INC-1"
    assert association["equipment_id"] == asset["equipment_id"]
    assert association["role"] == "PRIMARY"
    assert association["link_source"] == "USER_SELECTED"
    assert association["review_status"] == "PENDING_REVIEW"
    assert registry.list_incident_associations(DEFAULT_DATASET_ID, "INC-1") == [association]


def test_multiple_assets_can_be_explicitly_linked_to_one_incident(equipment_env):
    registry, _store, incidents, _tmp = equipment_env
    asset_one = _create_asset(registry, asset_tag="MOTOR-1")
    asset_two = _create_asset(
        registry,
        asset_name="Pump",
        category_code="pump",
        subtype_code="centrifugal",
        asset_tag="PUMP-1",
    )
    first = registry.create_incident_association(
        DEFAULT_DATASET_ID, "INC-1", asset_one["equipment_id"], incidents
    )
    second = registry.create_incident_association(
        DEFAULT_DATASET_ID, "INC-1", asset_two["equipment_id"], incidents
    )

    assert {row["association_id"] for row in registry.list_incident_associations(
        DEFAULT_DATASET_ID, "INC-1"
    )} == {first["association_id"], second["association_id"]}


def test_associations_are_dataset_scoped_and_cross_scope_links_rejected(equipment_env):
    registry, _store, incidents, _tmp = equipment_env
    asset_a = _create_asset(registry, dataset_id="DATASET-A")
    asset_b = _create_asset(registry, dataset_id="DATASET-B")

    association = registry.create_incident_association(
        "DATASET-A", "INC-2", asset_a["equipment_id"], incidents
    )
    assert registry.list_incident_associations("DATASET-B") == []
    assert registry.get_incident_association(
        "DATASET-B", association["association_id"]
    ) is None
    with pytest.raises(EquipmentRegistryError):
        registry.create_incident_association(
            "DATASET-A", "INC-2", asset_b["equipment_id"], incidents
        )


@pytest.mark.parametrize(
    ("incident_id", "frame"),
    [
        ("INC-MISSING", pd.DataFrame([{
            "incident_id": "INC-1",
            "dataset_id": DEFAULT_DATASET_ID,
        }])),
        ("INC-DUP", pd.DataFrame([
            {"incident_id": "INC-DUP", "dataset_id": DEFAULT_DATASET_ID},
            {"incident_id": "INC-DUP", "dataset_id": DEFAULT_DATASET_ID},
        ])),
        ("INC-2", pd.DataFrame([{
            "incident_id": "INC-2",
            "dataset_id": "DATASET-A",
        }])),
    ],
)
def test_invalid_or_ambiguous_incident_references_are_rejected(
    equipment_env, incident_id, frame
):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    with pytest.raises((EquipmentRegistryError, DatasetOwnershipError)):
        registry.create_incident_association(
            DEFAULT_DATASET_ID, incident_id, asset["equipment_id"], frame
        )


def test_association_rejects_missing_equipment_and_duplicate_links(equipment_env):
    registry, _store, incidents, _tmp = equipment_env
    missing_equipment_id = str(uuid.uuid4())
    with pytest.raises(EquipmentRegistryError):
        registry.create_incident_association(
            DEFAULT_DATASET_ID, "INC-1", missing_equipment_id, incidents
        )

    asset = _create_asset(registry)
    registry.create_incident_association(
        DEFAULT_DATASET_ID, "INC-1", asset["equipment_id"], incidents
    )
    with pytest.raises(EquipmentRegistryError):
        registry.create_incident_association(
            DEFAULT_DATASET_ID, "INC-1", asset["equipment_id"], incidents
        )


@pytest.mark.parametrize("decision", ["REVIEWED", "REJECTED"])
def test_pending_association_can_be_accepted_or_rejected(equipment_env, decision):
    registry, _store, incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    association = registry.create_incident_association(
        DEFAULT_DATASET_ID, "INC-1", asset["equipment_id"], incidents
    )

    reviewed = registry.review_incident_association(
        DEFAULT_DATASET_ID,
        association["association_id"],
        decision,
        reviewer="Reliability Engineer",
        review_notes="Checked against the incident record.",
    )

    assert reviewed["review_status"] == decision
    assert reviewed["reviewed_by"] == "Reliability Engineer"
    assert reviewed["reviewed_at"] is not None
    assert reviewed["review_notes"] == "Checked against the incident record."
    with pytest.raises(EquipmentRegistryError):
        registry.review_incident_association(
            DEFAULT_DATASET_ID,
            association["association_id"],
            "REVIEWED",
            reviewer="Second reviewer",
        )


def test_review_rejects_invalid_decisions_and_missing_reviewer(equipment_env):
    registry, _store, incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    association = registry.create_incident_association(
        DEFAULT_DATASET_ID, "INC-1", asset["equipment_id"], incidents
    )
    with pytest.raises(EquipmentRegistryError):
        registry.review_incident_association(
            DEFAULT_DATASET_ID,
            association["association_id"],
            "PENDING_REVIEW",
            reviewer="Engineer",
        )
    with pytest.raises(EquipmentRegistryError):
        registry.review_incident_association(
            DEFAULT_DATASET_ID,
            association["association_id"],
            "REJECTED",
            reviewer=" ",
        )


def test_missing_incident_diagnostics_are_scoped_and_do_not_delete_association(equipment_env):
    registry, _store, incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    association = registry.create_incident_association(
        DEFAULT_DATASET_ID, "INC-1", asset["equipment_id"], incidents
    )
    reduced_incidents = incidents[incidents["incident_id"] != "INC-1"]

    findings = registry.diagnose_missing_incidents(
        DEFAULT_DATASET_ID, reduced_incidents
    )
    assert len(findings) == 1
    assert findings[0]["association_id"] == association["association_id"]
    assert registry.get_incident_association(
        DEFAULT_DATASET_ID, association["association_id"]
    ) is not None
    assert registry.diagnose_missing_incidents("DATASET-A", reduced_incidents) == []


def test_unregistered_dataset_diagnostics_return_only_aggregate_findings(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    with sqlite3.connect(registry.db_path) as connection:
        connection.execute(
            "UPDATE equipment_assets SET dataset_id = ? WHERE equipment_id = ?",
            ("DATASET-REMOVED", asset["equipment_id"]),
        )

    findings = registry.diagnose_unregistered_dataset_references()
    assert findings == [
        {
            "code": "unregistered_dataset",
            "dataset_id": "DATASET-REMOVED",
            "entity_type": "equipment",
            "record_count": 1,
        }
    ]


def test_legacy_incident_frames_without_dataset_id_only_resolve_to_default(equipment_env):
    registry, _store, _incidents, _tmp = equipment_env
    asset = _create_asset(registry)
    legacy = pd.DataFrame([{"incident_id": "INC-LEGACY", "machine_id": "M-9"}])

    association = registry.create_incident_association(
        DEFAULT_DATASET_ID, "INC-LEGACY", asset["equipment_id"], legacy
    )
    assert association["incident_id"] == "INC-LEGACY"
    with pytest.raises(DatasetOwnershipError):
        registry.create_incident_association(
            "DATASET-A", "INC-LEGACY", _create_asset(
                registry,
                dataset_id="DATASET-A",
                asset_tag="A-1",
            )["equipment_id"], legacy
        )


def test_unknown_or_unversioned_database_schema_is_not_overwritten(tmp_path):
    evidence_store = EvidenceStore(base_dir=tmp_path / "evidence")
    evidence_store.ensure_default_dataset()
    db_path = tmp_path / "existing.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    with pytest.raises(EquipmentRegistryError, match="unversioned"):
        EquipmentRegistry(evidence_store, db_path=db_path)

    db_path_v2 = tmp_path / "newer.db"
    with sqlite3.connect(db_path_v2) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(EquipmentRegistryError, match="newer"):
        EquipmentRegistry(evidence_store, db_path=db_path_v2)

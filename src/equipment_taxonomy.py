"""Versioned candidate equipment taxonomy for GhostFault M9.3."""

from typing import Dict, Mapping, Tuple


TAXONOMY_VERSION = "m9.3-v1"
CLASSIFICATION_CANDIDATE = "CANDIDATE"
CLASSIFICATION_UNCLASSIFIED = "UNCLASSIFIED"
TAXONOMY_STATUS = CLASSIFICATION_CANDIDATE


EQUIPMENT_TAXONOMY: Dict[str, Mapping[str, object]] = {
    "electric_motor": {
        "label": "Electric motor",
        "status": TAXONOMY_STATUS,
        "subtypes": {
            "induction": "Induction motor",
            "synchronous": "Synchronous motor",
            "dc": "DC motor",
        },
        "components": {
            "stator": "Stator",
            "rotor": "Rotor",
            "bearings": "Bearings",
            "cooling_system": "Cooling system",
            "shaft": "Shaft",
            "terminal_box": "Terminal box",
        },
    },
    "pump": {
        "label": "Pump",
        "status": TAXONOMY_STATUS,
        "subtypes": {
            "centrifugal": "Centrifugal",
            "positive_displacement": "Positive displacement",
            "submersible": "Submersible",
        },
        "components": {
            "impeller": "Impeller",
            "casing": "Casing",
            "shaft": "Shaft",
            "bearings": "Bearings",
            "seals": "Seals",
            "coupling": "Coupling",
        },
    },
    "gearbox": {
        "label": "Gearbox",
        "status": TAXONOMY_STATUS,
        "subtypes": {
            "helical": "Helical",
            "bevel": "Bevel",
            "planetary": "Planetary",
            "worm": "Worm",
        },
        "components": {
            "gears": "Gears",
            "shafts": "Shafts",
            "bearings": "Bearings",
            "seals": "Seals",
            "lubrication_system": "Lubrication system",
            "housing": "Housing",
        },
    },
    "compressor": {
        "label": "Compressor",
        "status": TAXONOMY_STATUS,
        "subtypes": {
            "centrifugal": "Centrifugal",
            "screw": "Screw",
            "reciprocating": "Reciprocating",
            "scroll": "Scroll",
        },
        "components": {
            "compression_element": "Compression element",
            "rotor": "Rotor",
            "crankshaft": "Crankshaft",
            "valves": "Valves",
            "bearings": "Bearings",
            "seals": "Seals",
            "lubrication_system": "Lubrication system",
            "cooling_system": "Cooling system",
        },
    },
    "unclassified": {
        "label": "Unclassified equipment",
        "status": CLASSIFICATION_UNCLASSIFIED,
        "subtypes": {},
        "components": {},
    },
}


def category_options() -> Tuple[Tuple[str, str], ...]:
    """Return stable category codes and labels for forms."""
    return tuple(
        (code, str(definition["label"]))
        for code, definition in EQUIPMENT_TAXONOMY.items()
    )


def subtype_options(category_code: str) -> Tuple[Tuple[str, str], ...]:
    definition = EQUIPMENT_TAXONOMY.get(category_code)
    if definition is None:
        return ()
    subtypes = definition["subtypes"]
    return tuple(sorted(subtypes.items()))


def component_options(category_code: str) -> Tuple[Tuple[str, str], ...]:
    definition = EQUIPMENT_TAXONOMY.get(category_code)
    if definition is None:
        return ()
    components = definition["components"]
    return tuple(sorted(components.items()))

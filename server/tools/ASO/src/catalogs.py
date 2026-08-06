"""The vocabularies ASO works with: CBCT landmark names, IOS tooth names, IOS
landmark types, and the tooth name <-> universal ID table.

Ported from the Slicer module's `ASO_Method/CBCT.py::DicLandmark` and
`ASO_Method/IOS.py`, which is where they used to live -- inside the Qt widget,
duplicated between the UI (checkbox labels) and each CLI (the strings it
parsed). The server owns them now and publishes them through
`ArgSpec.choices`, so a client renders its checkboxes straight from
GET /tools and the two can no longer disagree.

Nothing here imports anything heavy: this module is read at registry
discovery time.
"""

# ---------------------------------------------------------------------------
# CBCT landmarks
# ---------------------------------------------------------------------------
# Grouped exactly as the Slicer UI grouped them, one tab per group. This is
# published through `ArgSpec.groups`, so a client renders the tabs from the
# schema: the alternative once considered here -- prefixing the option labels
# with their group -- was rejected because those labels ARE the names the
# server matches against a reference file. See ALI_PORT_CONTEXT.md section 3.1.
CBCT_LANDMARK_GROUPS = {
    "Cranial base": (
        "Ba", "C2", "C3", "C4", "LFZyg", "LPo", "N", "RFZyg", "RPo", "S",
    ),
    "Upper": (
        "A", "ANS", "IF", "LInfOr", "LMZyg", "LNC", "LOr", "LPF", "PNS",
        "RInfOr", "RMZyg", "RNC", "ROr", "RPF",
        "UL1O", "UL1R", "UL2O", "UL2R", "UL3O", "UL3R", "UL4O", "UL4R",
        "UL5O", "UL5R", "UL6DB", "UL6MB", "UL6MP", "UL6O", "UL6R",
        "UL7O", "UL7R",
        "UR1O", "UR1R", "UR2O", "UR2R", "UR3O", "UR3R", "UR4O", "UR4R",
        "UR5O", "UR5R", "UR6DB", "UR6MB", "UR6MP", "UR6O", "UR6R",
        "UR7O", "UR7R",
    ),
    "Lower": (
        "B", "Gn", "LAE", "LAF", "LARa", "LCo", "LGo", "LLCo", "LMCo",
        "LMeF", "LPRa", "LSig", "Me", "Pog", "PogL",
        "LL1O", "LL1R", "LL2O", "LL2R", "LL3O", "LL3R", "LL4O", "LL4R",
        "LL5O", "LL5R", "LL6DB", "LL6MB", "LL6O", "LL6R", "LL7O", "LL7R",
        "LR1O", "LR1R", "LR2O", "LR2R", "LR3O", "LR3R", "LR4O", "LR4R",
        "LR5O", "LR5R", "LR6DB", "LR6MB", "LR6O", "LR6R", "LR7O", "LR7R",
        "RAE", "RAF", "RARa", "RCo", "RGo", "RLCo", "RMCo", "RMeF",
        "RPRa", "RSig",
    ),
}

CBCT_LANDMARKS = tuple(
    name for group in CBCT_LANDMARK_GROUPS.values() for name in group
)

# What the Slicer module's `Suggest()` pre-checked: the seven points that
# define the orientation planes both published reference bundles are built on.
DEFAULT_CBCT_LANDMARKS = ("Ba", "S", "N", "RPo", "LPo", "ROr", "LOr")

CBCT_LANDMARK_CHOICES = {
    name: name in DEFAULT_CBCT_LANDMARKS for name in CBCT_LANDMARKS
}


# ---------------------------------------------------------------------------
# IOS teeth
# ---------------------------------------------------------------------------
# The universal numbering a segmented mesh carries in its label array, and the
# names a human uses. The original kept this table in five places (two CLIs,
# two Method classes, and an exception class); it lives here once.
TOOTH_IDS = {
    "UR8": 1, "UR7": 2, "UR6": 3, "UR5": 4, "UR4": 5, "UR3": 6, "UR2": 7,
    "UR1": 8,
    "UL1": 9, "UL2": 10, "UL3": 11, "UL4": 12, "UL5": 13, "UL6": 14,
    "UL7": 15, "UL8": 16,
    "LL8": 17, "LL7": 18, "LL6": 19, "LL5": 20, "LL4": 21, "LL3": 22,
    "LL2": 23, "LL1": 24,
    "LR1": 25, "LR2": 26, "LR3": 27, "LR4": 28, "LR5": 29, "LR6": 30,
    "LR7": 31, "LR8": 32,
}

TOOTH_NAMES = {number: name for name, number in TOOTH_IDS.items()}

UPPER_TEETH = tuple(name for name, number in TOOTH_IDS.items() if number <= 16)
LOWER_TEETH = tuple(name for name, number in TOOTH_IDS.items() if number > 16)

# One row per arch, teeth left to right in universal-numbering order -- the
# dental chart a clinician reads, published so a client lays the check boxes
# out that way instead of stacking 32 of them in a column. Deriving it from
# TOOTH_IDS rather than writing it out again is what keeps the chart and the
# label array in step: a tooth added to TOOTH_IDS appears in its own arch.
TOOTH_GROUPS = {"Upper": UPPER_TEETH, "Lower": LOWER_TEETH}

# Auto_IOS.Suggest(): three teeth per jaw, spread left/middle/right, which is
# what the coarse pre-alignment needs (see ios/pre_icp.py).
DEFAULT_TEETH = ("UR6", "UL1", "UL6", "LL6", "LR1", "LR6")

TOOTH_CHOICES = {name: name in DEFAULT_TEETH for name in TOOTH_IDS}


# ---------------------------------------------------------------------------
# IOS landmark types
# ---------------------------------------------------------------------------
# Semi-automated IOS registers on <tooth><type> keys read from the user's own
# markups file and the reference's, so every type the original UI offered is
# legitimately selectable here -- unlike ALI's IOS mode, where three of them
# had no model behind them and were published anyway.
IOS_LANDMARK_TYPES = ("O", "MB", "DB", "CB", "CL", "OIP", "R", "RIP")

DEFAULT_IOS_LANDMARK_TYPES = ("O",)

IOS_LANDMARK_TYPE_CHOICES = {
    name: name in DEFAULT_IOS_LANDMARK_TYPES for name in IOS_LANDMARK_TYPES
}


# ---------------------------------------------------------------------------
# Jaws and occlusion
# ---------------------------------------------------------------------------
JAWS = ("Upper", "Lower")

JAW_CHOICES = {"Upper": True, "Lower": True}

# The original had this as two arguments that could contradict each other: an
# `occlusion` boolean and a `jaw` string the client built by joining the
# checked jaws with "/", which the CLI then read as `jaw[0] == "Upper"`. One
# argument, three states, no illegal combination.
OCCLUSION_INDEPENDENT = "Orient each jaw independently"
OCCLUSION_UPPER_DRIVES = "Upper drives Lower"
OCCLUSION_LOWER_DRIVES = "Lower drives Upper"

OCCLUSION_CHOICES = {
    OCCLUSION_INDEPENDENT: True,
    OCCLUSION_UPPER_DRIVES: False,
    OCCLUSION_LOWER_DRIVES: False,
}

# Which jaw's transform is applied to the other one, per occlusion mode.
DRIVING_JAW = {
    OCCLUSION_INDEPENDENT: None,
    OCCLUSION_UPPER_DRIVES: "Upper",
    OCCLUSION_LOWER_DRIVES: "Lower",
}


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
MODALITY_CBCT = "CBCT"
MODALITY_IOS = "IOS"
MODALITY_CHOICES = {MODALITY_CBCT: True, MODALITY_IOS: False}

AUTOMATION_SEMI = "Semi-Automated"
AUTOMATION_FULLY = "Fully-Automated"
AUTOMATION_CHOICES = {AUTOMATION_SEMI: True, AUTOMATION_FULLY: False}


def teeth_to_ids(names) -> tuple:
    """Universal IDs for a list of tooth names, in the order given.

    Raises KeyError on an unknown name, which cannot happen through the schema
    (validate() rejects an option outside `choices` with a 422 first) but keeps
    a direct call to the src API honest.
    """
    return tuple(TOOTH_IDS[name] for name in names)


def split_by_jaw(tooth_names) -> dict:
    """{"Upper": [...], "Lower": [...]} for a flat list of tooth names."""
    return {
        "Upper": [name for name in tooth_names if name in UPPER_TEETH],
        "Lower": [name for name in tooth_names if name in LOWER_TEETH],
    }


def landmark_keys_by_jaw(tooth_names, landmark_types) -> dict:
    """The <tooth><type> keys semi-automated IOS registers on, per jaw.

    "UR6" x ("O", "MB") -> "UR6O", "UR6MB". Declaration order is preserved so
    the caller can report exactly what it asked for.
    """
    per_jaw = split_by_jaw(tooth_names)
    return {
        jaw: [f"{tooth}{kind}" for tooth in teeth for kind in landmark_types]
        for jaw, teeth in per_jaw.items()
    }

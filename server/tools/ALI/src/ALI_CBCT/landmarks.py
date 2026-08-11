"""The CBCT landmark vocabulary: which landmarks exist, and how they group.

This is the single source of truth the schema publishes through `choices` and
the engine validates against. The original had two copies that disagreed on the
impacted-canine spelling (`UR3OI` vs `UR3OIP`), and the cost was a lost patient
rather than a mislabelled point: an unguarded `LABEL_GROUPS[...]` in the save
loop raised a KeyError caught far above, so nothing at all was written for that
scan. Both spellings are accepted here (see `_ALIASES`) and `group_of()` never
raises.
"""

# Anatomical region -> the landmarks it contains. Region codes are the ones
# the original wrote into its output file names (CB/U/L/CI); the display names
# beside them are what the schema publishes and the client renders.
GROUP_LABELS = {
    "CB": [
        "Ba", "S", "N", "RPo", "LPo", "RFZyg", "LFZyg", "C2", "C3", "C4",
    ],
    "U": [
        "RInfOr", "LInfOr", "LMZyg", "RPF", "LPF", "PNS", "ANS", "A", "UR3O",
        "UR1O", "UL3O", "UR6DB", "UR6MB", "UL6MB", "UL6DB", "IF", "ROr", "LOr",
        "RMZyg", "RNC", "LNC", "UR7O", "UR5O", "UR4O", "UR2O", "UL1O", "UL2O",
        "UL4O", "UL5O", "UL7O", "UL7R", "UL5R", "UL4R", "UL2R", "UL1R", "UR2R",
        "UR4R", "UR5R", "UR7R", "UR6MP", "UL6MP", "UL6R", "UR6R", "UR6O",
        "UL6O", "UL3R", "UR3R", "UR1R",
    ],
    "L": [
        "RCo", "RGo", "Me", "Gn", "Pog", "PogL", "B", "LGo", "LCo", "LR1O",
        "LL6MB", "LL6DB", "LR6MB", "LR6DB", "LAF", "LAE", "RAF", "RAE", "LMCo",
        "LLCo", "RMCo", "RLCo", "RMeF", "LMeF", "RSig", "RPRa", "RARa", "LSig",
        "LARa", "LPRa", "LR7R", "LR5R", "LR4R", "LR3R", "LL3R", "LL4R", "LL5R",
        "LL7R", "LL7O", "LL5O", "LL4O", "LL3O", "LL2O", "LL1O", "LR2O", "LR3O",
        "LR4O", "LR5O", "LR7O", "LL6R", "LR6R", "LL6O", "LR6O", "LR1R", "LL1R",
        "LL2R", "LR2R",
    ],
    "CI": ["UR3OIP", "UL3OIP", "UR3RIP", "UL3RIP"],
}

# What the schema publishes: display name -> region code. Order is the order
# the client renders the check boxes in.
REGION_NAMES = {
    "Cranial base": "CB",
    "Upper": "U",
    "Lower": "L",
    "Impacted canine": "CI",
}

REGION_CODES = tuple(REGION_NAMES.values())

# Every region on by default: a landmark whose weights are absent from the
# chosen bundle costs nothing but a line in the run report, whereas a region
# left off by default is one the user never discovers.
REGION_CHOICES = {display_name: True for display_name in REGION_NAMES}

# The UI spelling of the impacted-canine landmarks, mapped to the spelling the
# weights are packaged under. Accepted, never emitted.
_ALIASES = {
    "UR3OI": "UR3OIP",
    "UL3OI": "UL3OIP",
    "UR3RI": "UR3RIP",
    "UL3RI": "UL3RIP",
}

LABEL_GROUPS = {
    label: code for code, labels in GROUP_LABELS.items() for label in labels
}
LABEL_GROUPS.update({alias: LABEL_GROUPS[canonical] for alias, canonical in _ALIASES.items()})

LABELS = tuple(label for labels in GROUP_LABELS.values() for label in labels)

# Every catalog landmark, all off. "All off" is what "not specified" looks like
# for a multichoice, and it hands the decision back to `cbct_regions`. Off
# rather than on, unlike REGION_CHOICES: 119 landmarks all on would mean every
# panel that never touches this field silently overrides the region selection.
LANDMARK_CHOICES = {label: False for label in LABELS}

# The two spacings the agent walks, coarse first. The scale key is the spacing
# with its decimal point replaced, because that is what the shipped weight
# folders are named (`<landmark>/1/` and `<landmark>/0-3/`).
SCALE_SPACINGS = (1.0, 0.3)

# Region code -> a name a human reads. Used for the run report only.
REGION_DISPLAY_NAMES = {code: display for display, code in REGION_NAMES.items()}

# The tabs a client lays the 119 `landmarks` check boxes out in: the SAME
# grouping the engine names its output files by, published rather than
# restated, so a landmark added to GROUP_LABELS gets its tab with no client
# release. In REGION_NAMES order, so the tabs read like the check boxes above.
LANDMARK_GROUPS = {
    display: tuple(GROUP_LABELS[code]) for display, code in REGION_NAMES.items()
}

UNGROUPED = "Other"


def scale_key(spacing: float) -> str:
    """1.0 -> '1', 0.3 -> '0-3'. The name of the weight folder for that scale."""
    text = f"{spacing:g}"
    return text.replace(".", "-")


SCALE_KEYS = tuple(scale_key(spacing) for spacing in SCALE_SPACINGS)


def canonical(label: str) -> str:
    """The spelling the vocabulary uses for `label`, aliases resolved."""
    return _ALIASES.get(label, label)


def group_of(label: str) -> str:
    """The region a landmark belongs to, or "Other" for an unknown name. Never
    raises: the unguarded lookup this replaces cost a patient every one of
    their landmarks when a bundle carried an unfamiliar name."""
    return LABEL_GROUPS.get(label, UNGROUPED)


def region_codes(selection) -> tuple:
    """Turn what `run()` received for `cbct_regions` into region codes.

    Accepts a `base.Selection` (the HTTP path), None for an omitted optional
    argument, or a plain sequence of codes, so the engine stays callable by
    another server-side tool that does not know the display names.
    """
    if selection is None:
        return REGION_CODES
    if isinstance(selection, dict):
        return tuple(
            REGION_NAMES[name]
            for name, enabled in selection.items()
            if enabled and name in REGION_NAMES
        )
    return tuple(selection)


def landmark_names(selection) -> tuple:
    """Turn what `run()` received for `landmarks` into canonical label names.

    Empty is the normal case and means "not specified": `cbct_regions` decides.
    Same accepted shapes as `region_codes`. Aliases are resolved here
    (`UR3OI` -> `UR3OIP`), the weights only ever using the canonical spelling.
    """
    if selection is None:
        return ()
    if isinstance(selection, dict):
        chosen = [name for name, enabled in selection.items() if enabled]
    else:
        chosen = list(selection)
    # Declaration order, not the caller's: it is the order the report and the
    # per-group output files already use.
    wanted = {canonical(name) for name in chosen}
    return tuple(label for label in LABELS if label in wanted)


def landmarks_in(codes) -> tuple:
    """Every known landmark belonging to one of these region codes."""
    wanted = set(codes)
    return tuple(
        label for code, labels in GROUP_LABELS.items() if code in wanted for label in labels
    )

"""The IOS landmark vocabulary: teeth, landmark types, and the networks.

Three networks exist, and each predicts a fixed set of landmark types on a
tooth it is pointed at:

* **Occlusal** (`O`) -- the occlusal point plus the mesio- and disto-buccal
  cusps (`O`, `MB`, `DB`);
* **Cervical** (`C`) -- the cervical lingual and buccal points (`CL`, `CB`);
* **Mucogingival** (`MG`) -- one point per lower tooth, on the gingival margin
  rather than on the crown. Added upstream in 2026-08 and ported here so AREG
  can register a mandible on the band around the mucogingival line without the
  caller predicting the landmarks elsewhere first.

The original Slicer UI also offered `R`, `RIP` and `OIP` under "Cervical".
They are deliberately absent: no shipped model predicts them, they are not in
the type list either CLI uses, and ticking them did nothing at all -- neither
selecting a network nor producing a label. An option that cannot work is worse
than an option that is not offered.
"""

# Teeth in Universal numbering, per jaw. Upper is 2..15 right-to-left, lower
# is 18..31 left-to-right; the ALIDDM weights index their label tables this
# way, so the order is part of the models' contract, not a presentation
# choice.
UPPER_TEETH = ["UL7", "UL6", "UL5", "UL4", "UL3", "UL2", "UL1",
               "UR1", "UR2", "UR3", "UR4", "UR5", "UR6", "UR7"]
LOWER_TEETH = ["LL7", "LL6", "LL5", "LL4", "LL3", "LL2", "LL1",
               "LR1", "LR2", "LR3", "LR4", "LR5", "LR6", "LR7"]

# Universal number of every tooth, per jaw.
UNIVERSAL_NUMBERS = {
    "Upper": {name: 15 - index for index, name in enumerate(UPPER_TEETH)},
    "Lower": {name: 18 + index for index, name in enumerate(LOWER_TEETH)},
}

# Landmark types, in the order the label tables below index them.
TYPE_LM = ["O", "MB", "DB", "CL", "CB", "MG"]

# Network -> the landmark types it predicts, mapped to the channel of its
# output the type comes out on.
NETWORKS = {
    "O": {"O": 0, "MB": 1, "DB": 2},
    "C": {"CL": 0, "CB": 1},
    "MG": {"MG": 0},
}

# What the schema publishes: display name -> network code.
NETWORK_NAMES = {"Occlusal": "O", "Cervical": "C", "Mucogingival": "MG"}
NETWORK_CODES = tuple(NETWORK_NAMES.values())
# Mucogingival is OFF by default: it is one point per lower tooth on the
# gingival margin, wanted by AREG's lower-arch registration and by nobody who
# asks for crown landmarks. Turning it on for everyone would add a third pass
# over every mesh of every existing request.
NETWORK_CHOICES = {display_name: code != "MG" for display_name, code in NETWORK_NAMES.items()}
NETWORK_DISPLAY_NAMES = {code: display for display, code in NETWORK_NAMES.items()}

# The networks that only exist for one jaw. MG was trained on the mandible
# alone, so asking for it on a maxilla is not a missing model, it is a question
# with no answer.
NETWORK_JAWS = {"MG": ("Lower",)}

# Radius at which the agent's cameras orbit the tooth, per network. The
# cervical points sit lower on the crown and need a wider view than the
# occlusal ones; MG shares the occlusal radius but aims its cameras elsewhere
# (see MG_AIM_OFFSET).
CAMERA_RADIUS = {"O": 0.2, "C": 0.3, "MG": 0.2}

# ---------------------------------------------------------------------------
# Mucogingival
# ---------------------------------------------------------------------------

# Names written in the predicted MG file, assigned positionally to the 13
# trained teeth taken in arch order (universal ids 19 -> 31). Tooth 25 carries
# the midline name L0MG, so the right side is shifted by one against the tooth
# numbers: LR1MG sits on tooth 26, not 25. Tooth 18 has no MG label (excluded
# from training).
#
# Six of these names collide with the TRAINING name of a different tooth (LR1MG
# is the training name of tooth 25 and the output name of tooth 26), which is
# why the table is positional and why nothing here translates a label on its
# own -- upstream's `TradLabelMG` exists for exactly that reason.
MG_OUTPUT_NAME = (
    "LL6MG", "LL5MG", "LL4MG", "LL3MG", "LL2MG", "LL1MG", "L0MG",
    "LR1MG", "LR2MG", "LR3MG", "LR4MG", "LR5MG", "LR6MG",
)

MG_TEETH = tuple(range(19, 32))

# Where the MG landmark sits relative to the centre of its tooth, in
# unit-sphere space, in the tooth's local frame: (buccal, along-the-arch,
# vertical). Median over upstream's 155 training scans, spread 0.02-0.03 on the
# buccal and vertical axes -- a stable anatomical prior, not a per-scan fit.
#
# The cameras aim here instead of at a flat "0.2 below the tooth centre", which
# only ever matched the incisors: on the molars the landmark is ~0.15 further
# buccal, which is why it fell outside the render entirely.
MG_AIM_OFFSET = {
    19: (0.149, -0.154, -0.125),   # LL6
    20: (0.130, -0.099, -0.156),   # LL5
    21: (0.077, -0.106, -0.164),   # LL4
    22: (0.035, -0.110, -0.182),   # LL3
    23: (0.012, -0.071, -0.200),   # LL2
    24: (-0.008, -0.063, -0.202),  # LL1
    25: (-0.014, -0.057, -0.189),  # L0
    26: (-0.008, -0.059, -0.199),  # LR1
    27: (0.003, -0.074, -0.190),   # LR2
    28: (0.058, -0.073, -0.186),   # LR3
    29: (0.096, -0.082, -0.171),   # LR4
    30: (0.135, -0.124, -0.168),   # LR5
    31: (0.160, -0.107, -0.156),   # LR6
}

# {network: {universal number as str: [landmark label per channel]}}.
# `UR1O`, `UR1MB`, ... -- the tooth name followed by the type.
LABELS = {
    network: {
        str(number): [f"{tooth}{lm_type}" for lm_type in types]
        for jaw, teeth in UNIVERSAL_NUMBERS.items()
        for tooth, number in teeth.items()
    }
    for network, types in (("O", ("O", "MB", "DB")), ("C", ("CL", "CB")))
}

# MG is positional rather than "<tooth><type>", for the shift described above.
LABELS["MG"] = {
    str(number): [MG_OUTPUT_NAME[index]] for index, number in enumerate(MG_TEETH)
}

# Jaw a Universal tooth number belongs to.
JAW_OF_NUMBER = {
    number: jaw for jaw, teeth in UNIVERSAL_NUMBERS.items() for number in teeth.values()
}

JAWS = ("Upper", "Lower")


def network_codes(selection) -> tuple:
    """Turn what `run()` received for `ios_networks` into network codes.

    Accepts a `base.Selection` (the normal HTTP path), None for an omitted
    optional argument, or a plain sequence of codes so the engine stays
    callable by another server-side tool.
    """
    if selection is None:
        return NETWORK_CODES
    if isinstance(selection, dict):
        return tuple(
            NETWORK_NAMES[name]
            for name, enabled in selection.items()
            if enabled and name in NETWORK_NAMES
        )
    return tuple(selection)

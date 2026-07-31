"""The IOS landmark vocabulary: teeth, landmark types, and the networks.

Two networks exist, and each predicts a fixed set of landmark types on a tooth
it is pointed at:

* **Occlusal** (`O`) -- the occlusal point plus the mesio- and disto-buccal
  cusps (`O`, `MB`, `DB`);
* **Cervical** (`C`) -- the cervical lingual and buccal points (`CL`, `CB`).

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
TYPE_LM = ["O", "MB", "DB", "CL", "CB"]

# Network -> the landmark types it predicts, mapped to the channel of its
# output the type comes out on.
NETWORKS = {
    "O": {"O": 0, "MB": 1, "DB": 2},
    "C": {"CL": 0, "CB": 1},
}

# What the schema publishes: display name -> network code, both on by default.
NETWORK_NAMES = {"Occlusal": "O", "Cervical": "C"}
NETWORK_CODES = tuple(NETWORK_NAMES.values())
NETWORK_CHOICES = {display_name: True for display_name in NETWORK_NAMES}
NETWORK_DISPLAY_NAMES = {code: display for display, code in NETWORK_NAMES.items()}

# Radius at which the agent's cameras orbit the tooth, per network. The
# cervical points sit lower on the crown and need a wider view than the
# occlusal ones.
CAMERA_RADIUS = {"O": 0.2, "C": 0.3}

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

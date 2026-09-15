"""What a tool gets without anyone configuring it.

A new tool should be servable by dropping it in one of the directories
`TOOLS_DIR` names, with no edit to this repository. These rules derive from the schema what `deployment.toml`
used to have to state; that file remains, as an override for the exceptions.

    argument named `model`, `*_model`, `*_reference`   picked from DATA/<tool>/models/
    any other `path` argument                          may be filled from DATA/<tool>/testfiles/
    argument named in TECHNICAL                        not rendered to a clinician
    tool `Batch_Dental_Seg`                            reads DATA/BatchDentalSeg/
    no `section` declared                              one derived from the name
    no `label` declared                                the argument name, written out

The one rule that is a safety property rather than a convenience: a model is
published as a name, never as a file argument, so a clinician cannot upload
weights from their laptop. See schema_tool's `selectable == "model"` branch.
"""

from __future__ import annotations

from typing import Optional

from . import deployment as deployment_module
from .deployment import ToolDeployment

# Suffixes that mean "the server hosts this, the caller names it".
MODEL_NAMES = ("model", "reference")

# The output directory every tool declares and no caller ever supplies: the
# server fills it in with the job's own output/ and takes it out of the
# published schema entirely (schema_tool imports this name from here). Written
# down in this file because the rules below read the RAW schema, where it is
# still a required `path` like any other -- which is exactly how a first version
# of batch_axis_for came to nominate it as the argument to divide.
OUTPUT_DIR_ARGUMENT = "output_dir"

# Arguments a clinician is never asked: device placement, tiling, worker
# counts, search budgets, mesh tuning. The tool still declares them and still
# applies its own defaults -- they are the deployment's business.
TECHNICAL = frozenset(
    {
        "device",
        "gpu_resampling",
        "tile_step_size",
        "num_workers",
        "n_workers",
        "batch_size",
        "threads",
        "search_seconds",
        "seed",
        "max_triplets",
        "surface_smoothing",
        "surface_decimation",
    }
)


# --- the automatic panel ---------------------------------------------------
#
# A tool that declares nothing still gets a panel someone can read. These two
# rules are what a `layout.py` OVERRIDES rather than what it has to restate: a
# declared `section` or `label` always wins, per argument.
#
# The section names are the ones the hand-written Slicer panels used, so a tool
# that declares nothing lands in the same boxes a clinician already knows:
# AMASSS's own .ui reads Inputs / Segmentation selection / Outputs / Advanced.

SECTION_INPUTS = "Inputs"
SECTION_MODEL = "Model"
SECTION_OPTIONS = "Options"
SECTION_OUTPUTS = "Outputs"
SECTION_ADVANCED = "Advanced"

# Name fragments that put an argument in Outputs. Matched as whole
# underscore-separated tokens, never as substrings -- `output` inside
# `output_dir` is a token, `put` inside `input` is not.
_OUTPUT_TOKENS = frozenset({"output", "outputs", "suffix", "prediction", "naming"})

# Abbreviations a label reads better spelled out. `num_workers` is "Number of
# workers"; nobody says "num".
_EXPANSIONS = {"num": "number of", "nb": "number of", "max": "maximum", "min": "minimum"}

# A timepoint, a jaw, a tooth number: a short token of letters then digits that
# is a name rather than a word. `t1` is "T1", not "T1" sentence-cased to "T1"
# by accident of being first.
_CODE_TOKEN = __import__("re").compile(r"^[a-z]{1,3}\d{1,2}$")


def section_for(argument_name: str, declaration: dict) -> str:
    """The collapsible box an argument lands in when it declares none.

    Order of the tests is the order of the rules: a model is a model wherever
    its name puts it, a technical knob is advanced whatever its type, and only
    then does the type decide.
    """
    if is_model(argument_name):
        return SECTION_MODEL
    if argument_name in TECHNICAL:
        return SECTION_ADVANCED
    tokens = set(argument_name.lower().split("_"))
    if tokens & _OUTPUT_TOKENS:
        return SECTION_OUTPUTS
    if isinstance(declaration, dict) and declaration.get("type") == "path":
        return SECTION_INPUTS
    if isinstance(declaration, dict) and declaration.get("required"):
        return SECTION_INPUTS
    return SECTION_OPTIONS


def label_for(argument_name: str) -> str:
    """The argument name written out for a clinician.

    Sentence case, not title case: "Tile step size", the way the hand-written
    panels wrote their labels. Short codes keep their own shape, and an
    abbreviation nobody says out loud is spelled out.

    **No vocabulary lives here, and that is the whole rule.** This used to hold
    a table casing `cbct` as "CBCT" and `areg` as "AREG" -- which made the
    server carry the list of the tools it serves, the one thing it is built not
    to know (`scripts/domain_coupling.py` fails the build over it). The table
    now lives in the client's `formgen.label_for`: the client IS the dental
    extension, so a dental word is its to know. What is left here is derivation
    -- splitting, casing, spelling out -- and none of it names anything.

    So `cbct_regions` comes off this function as "Cbct regions" and reaches a
    clinician as "CBCT regions". A tool that wants a word no rule can produce
    ("Scan / Landmark Folder") declares its own `label`, which nothing here or
    in the client overwrites.
    """
    words = []
    for index, token in enumerate(argument_name.split("_")):
        if not token:
            continue
        lowered = token.lower()
        if lowered in _EXPANSIONS and index == 0:
            words.append(_EXPANSIONS[lowered].capitalize())
        elif lowered in _EXPANSIONS:
            words.append(_EXPANSIONS[lowered])
        elif token.isupper():
            words.append(token)
        elif _CODE_TOKEN.match(lowered):
            words.append(token.upper())
        elif not words:
            words.append(lowered.capitalize())
        else:
            words.append(lowered)
    return " ".join(words) or argument_name


def is_model(argument_name: str) -> bool:
    return any(argument_name == name or argument_name.endswith("_" + name) for name in MODEL_NAMES)


# --- splitting a cohort ----------------------------------------------------


def batch_axis_for(arguments: dict) -> Optional[str]:
    """The argument a cohort is split on, or None if this tool cannot be split.

    The rule is "exactly one REQUIRED `path` argument", and the tools it
    EXCLUDES are the reason it is written this way rather than "the first path
    argument". A tool taking two required folders pairs them per patient --
    AREG's `t1`/`t2`, AutoCrop3D's `scans`/`roi`, AutoMatrix's
    `files`/`transforms`, GreedyReg's `t1`/`t2`. Splitting one of them without
    splitting the other by the same key changes which scan is registered
    against which, and that does not fail: it returns a plausible wrong result.

    So a tool of that shape is never offered for batching by convention, and
    the day a new one is added it is excluded without anyone having had to
    notice. Naming its axis in deployment.toml stays possible and is then a
    deliberate act, made by someone who has decided how the pairing survives.
    """
    required_paths = [
        name
        for name, declaration in arguments.items()
        if isinstance(declaration, dict)
        and declaration.get("type") == "path"
        and declaration.get("required")
        # Two required `path` arguments no caller ever sends a folder for, and
        # both invisible in what GET /tools publishes. Counting them gets the
        # answer wrong in BOTH directions, measured on the real tools: AMASSS
        # reads as three required folders (scans, model, output_dir) and is
        # excluded from batching altogether, while a tool whose only other path
        # is optional reads as exactly one and has its OUTPUT DIRECTORY
        # nominated as the thing to divide.
        and name != OUTPUT_DIR_ARGUMENT
        and not is_model(name)
    ]
    return required_paths[0] if len(required_paths) == 1 else None


def batch_plan(arguments: dict, declared: ToolDeployment, defaults: tuple) -> Optional[dict]:
    """How a client should split a cohort for this tool, or None to send it whole.

    `defaults` is this server's `(max MB, max files)`. A tool may override
    either -- the same escape hatch `max_upload_mb` and `timeout_seconds` have
    -- but the normal case is that it does not, batching being a property of
    the deployment and not of the tool.
    """
    if declared.batch_enabled is False:
        return None
    axis = declared.batch_axis or batch_axis_for(arguments)
    if not axis:
        return None

    max_mb, max_files = defaults
    if declared.batch_max_mb is not None:
        max_mb = declared.batch_max_mb
    if declared.batch_max_files is not None:
        max_files = declared.batch_max_files
    # Either number alone is a usable plan; both off is how a deployment turns
    # batching off without editing a tool table.
    if max_mb <= 0 and max_files <= 0:
        return None
    return {"axis": axis, "max_mb": max_mb, "max_files": max_files}


def derive(arguments: dict, declared: ToolDeployment, batch_defaults: tuple) -> ToolDeployment:
    """`declared` (from deployment.toml) merged over these conventions.

    Anything stated explicitly wins, per argument, so an exception costs one
    line rather than restating everything the conventions already got right.
    """
    selectable = {}
    for name, declaration in arguments.items():
        if not isinstance(declaration, dict) or declaration.get("type") != "path":
            continue
        selectable[name] = "model" if is_model(name) else "testfile"
    selectable.update(declared.server_selectable)
    # "none" is a removal, not a kind: it is how a deployment opts an argument
    # out of a convention its NAME would otherwise put it in.
    for name, kind in list(selectable.items()):
        if kind == deployment_module.SERVER_SELECTABLE_NONE:
            del selectable[name]

    hidden = {name for name in arguments if name in TECHNICAL}
    hidden.update(declared.hidden)

    return ToolDeployment(
        server_selectable=selectable,
        max_upload_mb=declared.max_upload_mb,
        data_dir=declared.data_dir,
        hidden=tuple(sorted(hidden)),
        batch=batch_plan(arguments, declared, batch_defaults),
    )

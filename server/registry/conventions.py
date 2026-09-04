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

from . import deployment as deployment_module
from .deployment import ToolDeployment

# Suffixes that mean "the server hosts this, the caller names it".
MODEL_NAMES = ("model", "reference")

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

# Words a clinician reads as one unit, kept in their own case rather than
# sentence-cased into nonsense. `cbct_regions` is "CBCT regions", not "Cbct
# regions"; `prediction_ID` is "Prediction ID", not "Prediction Id".
_ACRONYMS = {
    "cbct": "CBCT", "ios": "IOS", "mri": "MRI", "ct": "CT", "roi": "ROI",
    "id": "ID", "gpu": "GPU", "cpu": "CPU", "vram": "VRAM", "dicom": "DICOM",
    "vtk": "VTK", "stl": "STL", "nifti": "NIfTI", "tmj": "TMJ", "llm": "LLM",
    "3d": "3D", "2d": "2D", "fdi": "FDI", "icp": "ICP", "aso": "ASO",
    "ali": "ALI", "areg": "AREG", "amasss": "AMASSS",
}

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
    panels wrote their labels. Acronyms and short codes keep their own shape.
    """
    words = []
    for index, token in enumerate(argument_name.split("_")):
        if not token:
            continue
        lowered = token.lower()
        if lowered in _ACRONYMS:
            words.append(_ACRONYMS[lowered])
        elif lowered in _EXPANSIONS and index == 0:
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


def derive(arguments: dict, declared: ToolDeployment) -> ToolDeployment:
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
    )

"""The seams between AREG and the three tools its automated modes need.

AREG is the first tool that is mostly *other tools*. Its Fully-Automated CBCT
mode is "segment the T1 to get a mask, then register"; its Oriented mode adds
"orient the T1 first"; its Fully-Automated IOS mode is "label the crowns, orient
both timepoints, then register". Segmenting, orienting and labelling are AMASSS,
ASO and CrownSeg -- all three already on this server, none of them AREG's job to
contain.

### Why the calls are in-process and not HTTP requests to our own /run

Exactly the reason `ASO/src/ali_client.py` gives, and it bites harder here
because AREG chains up to three of them. `main.py` runs every tool inside an
`anyio.CapacityLimiter` capped at `MAX_CONCURRENT_TOOLS` (4 by default), and an
AREG request holds one of those slots for its whole run. If it then blocked on
an HTTP request to this same server asking for a slot of its own, four
concurrent AREG runs would hold all four slots and each wait for a fifth that
can never be freed -- the server would stop answering anything, `/health`
included, until they timed out.

`Tool.invoke` is the same entry point `main.py` uses, validation included, and
for a tool whose `output_kind` is `"files"` it returns the output DIRECTORY
(main.py is what zips it), so there is no archive round trip either.

### Why `registry.TOOLS[...]` rather than importing the logic module

Both patterns exist in this repo: `ali_client` goes through the registry,
AMASSS's docstring invites a direct `from ...AMASSSLogic import segment`. The
registry is the right one here because AREG needs the *availability* answer as
much as the result. A deployment may legitimately not carry AMASSS, and AREG
must then say "this mode needs AMASSS, use Semi-Automated and send your own
masks" -- a 422 the caller can act on -- rather than fail at import time and
take AREG out of the registry along with it.
"""

import logging
import os

from base import ResolvedPath, ToolArgumentError

logger = logging.getLogger("AREG")

_NOT_AVAILABLE = (
    "{mode} needs the '{tool}' tool, which is not deployed on this server. {advice}"
)

_ADVICE = {
    "AMASSS": (
        "Use Semi-Automated mode and send your own T1 segmentation masks, or ask the "
        "administrator to deploy 'AMASSS'."
    ),
    "ASO": (
        "Orient the T1 scans yourself and use Fully-Automated mode, or ask the "
        "administrator to deploy 'ASO'."
    ),
    "CrownSeg": (
        "Segment the crowns yourself (the meshes need a per-point tooth-label array) "
        "and use Semi-Automated mode, or ask the administrator to deploy 'CrownSeg'."
    ),
    "ALI": (
        "Send the 13 mucogingival landmarks per lower scan in 'mgl_landmarks' instead, "
        "or ask the administrator to deploy 'ALI'."
    ),
}


def is_available(tool_name: str) -> bool:
    """Whether a tool is registered and loaded.

    Imported inside the function: `registry` imports every tool at startup, so
    importing it at AREG's module level would be a cycle.
    """
    from registry import TOOLS

    return tool_name in TOOLS


def require(tool_name: str, mode: str) -> None:
    """Refuse a mode whose helper tool this deployment does not carry.

    Called before any file is read, so an unusable request comes back as a 422
    in a second rather than after minutes of work.
    """
    if not is_available(tool_name):
        raise ToolArgumentError(
            _NOT_AVAILABLE.format(
                mode=mode, tool=tool_name, advice=_ADVICE.get(tool_name, "")
            ).strip()
        )


def _invoke(tool_name: str, args: dict, what: str) -> str:
    """Run a tool and return the directory holding its results."""
    from registry import TOOLS

    tool = TOOLS[tool_name]
    missing = [name for name in args if name not in tool.arguments]
    if missing:
        raise ToolArgumentError(
            f"The '{tool_name}' tool on this server does not take "
            f"{', '.join(sorted(missing))}, so AREG cannot drive it. This is a server "
            f"configuration problem, not a problem with your request."
        )

    logger.info("AREG: asking '%s' for %s", tool_name, what)
    result = tool.invoke(args)
    if isinstance(result, (list, tuple)):
        result = result[0] if result else None
    if not result or not os.path.isdir(str(result)):
        raise ToolArgumentError(f"The '{tool_name}' tool produced no {what}.")
    return str(result)


# ---------------------------------------------------------------------------
# AMASSS -- the T1 masks the registration is confined to
# ---------------------------------------------------------------------------

def segment_masks(
    scan_dir: str, model_path: str, mask_structures, tool_name: str = "AMASSS"
) -> str:
    """Segment every scan under `scan_dir` into the requested mask structures.

    Returns the directory holding AMASSS's output, which
    `cbct.pipeline.find_masks` then reads exactly as it reads a mask folder the
    caller sent -- the automated and semi-automated paths differ only in where
    the masks came from.

    `mask_structures` are AMASSS structure CODES (CBMASK/MANDMASK/MAXMASK).
    `segment()` speaks codes and `Tool.invoke` speaks the display names the
    schema publishes, so the codes are translated back through AMASSS's own
    table rather than being spelled out here -- a structure renamed in AMASSS
    must not silently stop matching.
    """
    args = {
        "input": ResolvedPath(scan_dir, "folder"),
        "model": model_path,
        "structures": {
            _option_named(tool_name, "structures", code): True for code in mask_structures
        },
        # One binary file per structure: `find_masks` looks each region's mask
        # up by name, and a merged multi-label volume would make every region
        # resolve to the same file.
        "merge": {_option_named(tool_name, "merge", "SEPARATE"): True},
        "prediction_ID": "seg",
        "generate_surface": False,
    }
    return _invoke(tool_name, args, "T1 masks")


def _option_named(tool_name: str, argument: str, code: str) -> str:
    """The schema option name AMASSS publishes for one of its internal codes.

    The wire carries the DISPLAY names -- "Cranial base (Mask)", "Separated
    segmentation files" -- and `validate()` rejects anything else with a 422
    before `run()` sees it, so AREG cannot send codes. Writing those display
    strings into AREG instead would make a rename in AMASSS an unexplained 422
    here; asking AMASSS's own table makes it the message below.

    Imported inside the function, and only on a path that has already checked
    AMASSS is registered: `registry` imports every tool at startup, and AREG has
    to load on a server that does not carry AMASSS.
    """
    from registry import TOOLS
    from tools.AMASSS.src import AMASSSLogic

    tables = {
        "structures": AMASSSLogic.STRUCTURE_NAMES,
        "merge": AMASSSLogic.MERGE_MODE_NAMES,
    }
    published = TOOLS[tool_name].arguments[argument].choices
    for name, published_code in tables[argument].items():
        if published_code == code and name in published:
            return name
    raise ToolArgumentError(
        f"The '{tool_name}' tool on this server does not offer '{code}' under "
        f"'{argument}', which AREG needs for this registration."
    )


# ---------------------------------------------------------------------------
# ASO -- orienting the T1 scans before registering onto them
# ---------------------------------------------------------------------------

def orient_scans(
    scan_dir: str,
    reference_path: str,
    modality: str,
    tool_name: str = "ASO",
    **extra,
) -> str:
    """Orient every case under `scan_dir` onto `reference_path`.

    Fully-Automated on both modalities: for CBCT that is ASO predicting the
    landmarks through ALI, for IOS it is the tooth-centroid alignment. Either
    way AREG hands over a folder and gets an oriented folder back.
    """
    args = {
        "modality": modality,
        "automation": "Fully-Automated",
        "input": ResolvedPath(scan_dir, "folder"),
        # Tagged with what it actually is: a reference resolved from the data
        # store can be a directory or a .zip, and ASO branches on `.kind`.
        "reference": ResolvedPath(
            reference_path, "folder" if os.path.isdir(reference_path) else "zip_file"
        ),
        "output_suffix": "Or",
    }
    args.update(extra)
    return _invoke(tool_name, args, f"oriented {modality} scans")


# ---------------------------------------------------------------------------
# CrownSeg -- the per-tooth labels the IOS engine needs
# ---------------------------------------------------------------------------

def predict_mucogingival(
    mesh_dir: str, model_path: str = None, tool_name: str = "ALI"
) -> str:
    """Predict the 13 mucogingival landmarks on every lower arch under `mesh_dir`.

    Returns the directory holding ALI's markups files, which `MGLPainter` then
    reads exactly as it reads a folder the caller sent -- the two paths differ
    only in where the landmarks came from.

    `ios_networks` is sent as the COMPLETE selection with Mucogingival alone
    enabled: a multichoice is read as everything it mentions and nothing else,
    so leaving the crown networks out is what keeps this from also running two
    more passes over every mesh for landmarks nobody asked for.

    `model` is omitted when the caller named nothing, which makes ALI pick the
    hosted bundle matching the input itself -- the same default ASO relies on.
    """
    args = {
        "input": ResolvedPath(mesh_dir, "folder"),
        "ios_networks": {_ali_network_named(tool_name, "MG"): True},
        "prediction_ID": "MG_Pred",
    }
    if model_path:
        args["model"] = model_path
    return _invoke(tool_name, args, "mucogingival landmarks")


def _ali_network_named(tool_name: str, code: str) -> str:
    """ALI's schema option name for one of its IOS network codes."""
    from registry import TOOLS
    from tools.ALI.src.ALI_IOS import landmarks as ios_catalog

    published = TOOLS[tool_name].arguments["ios_networks"].choices
    for name, network_code in ios_catalog.NETWORK_NAMES.items():
        if network_code == code and name in published:
            return name
    raise ToolArgumentError(
        f"The '{tool_name}' tool on this server does not offer the '{code}' network. "
        f"Registering on the mucogingival line needs it -- send the landmarks in "
        f"'mgl_landmarks' instead, or update '{tool_name}'."
    )


def label_crowns(mesh_dir: str, model_name: str = None, tool_name: str = "CrownSeg") -> str:
    """Label the crowns of every mesh under `mesh_dir`.

    `skip_segmented` is left at its default: a mesh that already carries a
    tooth-label array passes through untouched, so a batch mixing segmented and
    raw meshes costs network time only for the ones that need it. The original
    did the same thing by hand (`__BypassCrownseg__` copied files into two
    directories and merged them afterwards), which is where the bypass belongs
    -- inside the tool that does the segmenting.
    """
    args = {"input": ResolvedPath(mesh_dir, "folder"), "suffix": "Seg"}
    if model_name:
        args["model"] = model_name
    return _invoke(tool_name, args, "tooth-labelled meshes")

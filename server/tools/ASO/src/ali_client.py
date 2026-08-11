"""The seam between ASO's fully-automated CBCT mode and the landmark
identification it needs.

Fully-automated means "orient a scan nobody has placed landmarks on", so the
landmarks are predicted first -- which is what ALI does. ASO calls it rather
than containing it. The availability check stays even though ALI is deployed
here: a deployment may legitimately not carry it, and must then refuse with a
message pointing at the mode that does work.

The call is IN-PROCESS, not an HTTP request to our own /run/ALI, and that is
load-bearing. main.py runs every tool inside a CapacityLimiter capped at
MAX_CONCURRENT_TOOLS, and an ASO request holds one slot for its whole run: four
concurrent fully-automated runs each waiting on a fifth slot would deadlock the
server, /health included. `Tool.invoke` is the same entry point main.py uses,
validation included. The whole transport is `_invoke_ali` below, so pointing it
at a SEPARATE ALI deployment over HTTP is a change to that one function.
"""

import logging
import os

from base import ResolvedPath, ToolArgumentError

from . import markups

logger = logging.getLogger("ASO")

# What ASO needs ALI's schema to expose. Checked before the call so a contract
# drift is one clear message rather than a 422 from inside another tool.
_REQUIRED_ARGUMENTS = ("input", "model", "landmarks")

_NOT_AVAILABLE = (
    "Fully-Automated CBCT predicts the landmarks with the '{tool}' tool, which is "
    "not deployed on this server. Use Semi-Automated mode and send your own "
    "landmark files (.mrk.json) alongside the scans, or ask the administrator to "
    "deploy '{tool}'."
)


def is_available(tool_name: str) -> bool:
    """Whether the landmark tool is registered and loaded.

    Imported inside the function: `registry` imports every tool at startup, so
    importing it at ASO's module level would be a cycle.
    """
    from registry import TOOLS

    return tool_name in TOOLS


def predict_landmarks(
    input_dir: str,
    tool_name: str,
    model_path: str,
    landmarks: list,
    work_dir: str,
) -> dict:
    """Predict landmarks for every scan under `input_dir`.

    `model_path` is optional and None is the ordinary case: the argument is
    then left out of the call entirely, and the landmark tool picks the bundle
    matching the input from the models hosted for IT. That is the better
    default, since ASO's own folder also holds the reference bundles and
    nothing in a flat list of names says which entries are landmark weights.
    When given, it is already a local path resolved by main.py.

    Returns `{patient key: {landmark: np.ndarray}}` keyed like
    `cbct.pipeline.discover`. The results never touch the caller's input tree:
    the per-group files are merged in memory.
    """
    if not is_available(tool_name):
        raise ToolArgumentError(_NOT_AVAILABLE.format(tool=tool_name))

    output_dir = _invoke_ali(input_dir, tool_name, model_path, landmarks, work_dir)
    return _collect(output_dir)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _invoke_ali(
    input_dir: str, tool_name: str, model_path: str, landmarks: list, work_dir: str
) -> str:
    """Run the landmark tool and return the directory holding its results."""
    from registry import TOOLS

    tool = TOOLS[tool_name]
    missing = [name for name in _REQUIRED_ARGUMENTS if name not in tool.arguments]
    if missing:
        raise ToolArgumentError(
            f"The '{tool_name}' tool on this server does not take "
            f"{', '.join(missing)}, so ASO cannot drive it. This is a server "
            f"configuration problem, not a problem with your request."
        )

    args = {
        "input": ResolvedPath(input_dir, "folder"),
        # Sent as the complete selection: an option left out counts as off,
        # whatever its declared default (see base.Tool._coerce_multichoice).
        "landmarks": {name: True for name in landmarks},
    }
    # OMITTED, not sent empty, when the caller named no bundle: `model` is
    # optional on the landmark tool, and leaving it out is what makes it pick
    # the hosted bundle whose layout matches the input. Sending "" instead
    # would be a named bundle that does not exist.
    if model_path:
        args["model"] = model_path
    if "prediction_ID" in tool.arguments:
        args["prediction_ID"] = "Pred"

    logger.info(
        "Requesting %d landmark(s) from '%s' (%s)",
        len(landmarks),
        tool_name,
        f"bundle '{os.path.basename(str(model_path).rstrip(os.sep))}'"
        if model_path
        else "bundle chosen by the tool",
    )
    try:
        result = tool.invoke(args)
    except ToolArgumentError as exc:
        # The landmark tool judges the bundle in its own vocabulary and cannot
        # know ASO offered that name. The caller picked it from ASO's model
        # list, which also holds the reference bundles, so the likeliest
        # mistake is having chosen a reference: name the field to clear.
        if model_path:
            raise ToolArgumentError(
                f"{exc} -- 'landmark_models' named "
                f"'{os.path.basename(str(model_path).rstrip(os.sep))}', which is one "
                f"of the entries hosted for ASO but not a landmark model bundle "
                f"(the reference bundles are in that same list). Leave "
                f"'landmark_models' empty to let the server pick the right one."
            ) from exc
        raise

    if isinstance(result, (list, tuple)):
        result = result[0] if result else None
    if not result or not os.path.isdir(str(result)):
        raise ToolArgumentError(
            f"The '{tool_name}' tool returned no landmark files for these scans."
        )
    return str(result)


def _collect(output_dir: str) -> dict:
    """Merge every markups file the landmark tool produced, per patient.

    Keys mirror `cbct.pipeline.discover`: the patient's path relative to the
    results root. ALI wrote one file per landmark GROUP, so several files can
    belong to one patient and are merged.
    """
    # Imported here rather than at module level: cbct/ pulls in SimpleITK and
    # VTK, and this module is also on the IOS path.
    from .cbct.pipeline import patient_stem

    predictions: dict = {}
    for directory, _, file_names in os.walk(output_dir):
        relative = os.path.relpath(directory, output_dir)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if not markups.is_markups_file(file_name) or file_name.startswith("."):
                continue
            key = os.path.join(prefix, patient_stem(file_name))
            try:
                found = markups.load_landmarks(os.path.join(directory, file_name))
            except (ValueError, OSError) as exc:
                logger.warning("Skipping '%s': %s", file_name, exc)
                continue
            predictions.setdefault(key, {}).update(found)
    return predictions

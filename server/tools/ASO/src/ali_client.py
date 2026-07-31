"""The seam between ASO's fully-automated CBCT mode and the landmark
identification it needs.

ASO's fully-automated mode is "orient a scan nobody has placed landmarks on".
That works by predicting the landmarks first, which is exactly what ALI does --
one deep-RL agent per landmark, one `.pth` weight set per (landmark, scale).
Porting that engine into ASO would duplicate a tool that is on this server's
roadmap in its own right and that AREG will need too, so ASO **calls** it
instead of containing it.

**`tools/ALI/` does not exist on this server yet.** Until it does, this module
does exactly one thing: refuse the request with a message that says so and
points at the mode that does work. Everything after the availability check is
written against the server's ordinary tool contract, so the day ALI is
registered, ASO needs no change.

### Why the call is in-process and not an HTTP request to our own /run/ALI

The decoupled-by-HTTP version is tempting and is a **deadlock**. `main.py`
runs every tool inside an `anyio.CapacityLimiter` capped at
`MAX_CONCURRENT_TOOLS` (4 by default). An ASO request holds one of those slots
for its whole run. If it then blocks on an HTTP request to this same server
asking for a slot of its own, four concurrent fully-automated ASO runs hold all
four slots and each waits for a fifth that can never be freed -- the server
stops serving anything, including `/health`, until they time out. It would also
mean re-uploading the scans to ourselves and re-downloading the results.

`Tool.invoke` is the same entry point `main.py` uses, validation included, so
calling it directly gives the identical contract without any of that. The whole
transport is `_invoke_ali` below: swapping it for an HTTP call to a *separate*
ALI deployment (where the deadlock does not apply, because the slots are on
another machine) is a change to that one function.
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

    `model_path` is already a local path: `landmark_models` is declared
    `server_selectable` on ASO's schema, so main.py resolved the name the client
    picked against `DATA/ASO/models/` before `run()` was called. It is forwarded
    as-is, which is exactly the shape the landmark tool would have received had
    it been called over HTTP.

    Returns `{patient key: {landmark: np.ndarray}}` keyed exactly like
    `cbct.pipeline.discover`, so the caller can look a patient up without
    caring where the landmarks came from.

    The results never touch the caller's input tree. `MergeJson` used to merge
    ALI's per-group files by writing the merged one into the input folder and
    **deleting the sources**; the merge happens in memory here.
    """
    if not is_available(tool_name):
        raise ToolArgumentError(_NOT_AVAILABLE.format(tool=tool_name))
    if not model_path:
        raise ToolArgumentError(
            "Fully-Automated CBCT needs 'landmark_models': the name of a landmark "
            "model bundle hosted on this server (see GET /tools/ASO/data)."
        )

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
        "model": model_path,
        # Sent as the complete selection: an option left out counts as off,
        # whatever its declared default (see base.Tool._coerce_multichoice).
        "landmarks": {name: True for name in landmarks},
    }
    if "prediction_ID" in tool.arguments:
        args["prediction_ID"] = "Pred"

    logger.info("Requesting %d landmark(s) from '%s'", len(landmarks), tool_name)
    result = tool.invoke(args)

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

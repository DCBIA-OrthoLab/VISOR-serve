"""What a tool gets without anyone configuring it.

A new tool should be servable by dropping it in one of the directories
`TOOLS_DIR` names, with no edit to this repository. These rules derive from the schema what `deployment.toml`
used to have to state; that file remains, as an override for the exceptions.

    argument named `model`, `*_model`, `*_reference`   picked from DATA/<tool>/models/
    any other `path` argument                          may be filled from DATA/<tool>/testfiles/
    argument named in TECHNICAL                        not rendered to a clinician
    tool `Batch_Dental_Seg`                            reads DATA/BatchDentalSeg/

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

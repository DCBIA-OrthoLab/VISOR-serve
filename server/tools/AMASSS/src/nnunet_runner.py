"""nnUNet v2 inference, isolated from the rest of the AMASSS pipeline.

Four things this module exists to get right. The first three are defects the
original Slicer CLI had, in ways that only hurt on a shared server; the fourth
is where the run time actually goes.

1. **No `nnUNet_results` environment variable.** The CLI set
   `os.environ['nnUNet_results']` before spawning `nnUNetv2_predict`. On this
   server tools run concurrently in worker threads (see MAX_CONCURRENT_TOOLS
   in config.py), and `os.environ` is process-global: two AMASSS requests
   overlapping would silently overwrite each other's model path and predict
   the wrong structure. `initialize_from_trained_model_folder` takes an
   explicit path, so the race cannot happen by construction.

2. **No output-file polling heuristic.** The CLI watched the output file's
   size and killed the predictor process once it stopped growing for three
   seconds, which could interrupt nnUNet mid-postprocessing. Calling the
   Python API means the call simply returns when it is done.

3. **GPU work is serialized.** MAX_CONCURRENT_TOOLS allows several tools to
   run at once, but a single GPU cannot hold several concurrent 3d_fullres
   inferences. A semaphore caps AMASSS's own GPU usage; extra requests wait
   for a slot instead of pushing the card into an out-of-memory failure.

4. **The resampling runs on the GPU too.** Profiling one structure on a
   512x512x365 CBCT: 14.6s resampling the input to the model's grid, 4.5s of
   inference, 6.9s resampling the logits back -- the network is an eighth of
   the run and the card is idle for the rest. Both resamplings are scipy
   splines pinned to one core; nnUNet ships torch versions of them, and
   selecting those (see `_enable_gpu_resampling`) took the default
   five-structure run on that scan from 195.9s to 77.0s, for a documented cost
   in mask agreement. Bigger batches do NOT help here and were measured not to: at a
   128^3 patch the network already saturates the SMs at batch 1, so throughput
   is flat from batch 1 to 12 while using 2.7GB of a 48GB card. The idle memory
   is not convertible into speed; the idle *time* is.

torch and nnunetv2 are imported lazily, inside the functions that need them:
registry.py imports every tool at server startup, and a heavy (or missing)
deep-learning stack must not prevent the whole server from booting. If they
are absent, only AMASSS fails, with an actionable message.
"""

import glob
import inspect
import logging
import os
import threading

from base import ToolUnavailableError
from config import settings

logger = logging.getLogger("AMASSS.nnunet")

CHECKPOINT_NAME = "checkpoint_final.pth"
PLANS_FOLDER_PATTERN = "*__nnUNetPlans__3d_fullres"

# How many AMASSS inferences may touch the GPU at the same time -- see
# settings.AMASSS_MAX_GPU_JOBS for what a higher value does and does not buy.
# Read through config like every other setting rather than straight from
# os.getenv, so the whole server configuration stays discoverable in one file
# (and documented in .env.example).
_MAX_GPU_JOBS = max(1, int(settings.AMASSS_MAX_GPU_JOBS))
_GPU_SEMAPHORE = threading.BoundedSemaphore(_MAX_GPU_JOBS)

_INSTALL_HINT = (
    "AMASSS needs the nnUNet v2 inference stack. Install it with "
    "`pip install -r requirements.txt` (see server/README.md)."
)


class ModelNotFoundError(FileNotFoundError):
    """No usable nnUNet model folder for a requested structure."""


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: torch)") from exc
    return torch


def _import_predictor():
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: nnunetv2)") from exc
    return nnUNetPredictor


def resolve_device(requested: str) -> str:
    """Return the device to actually use, falling back to CPU when needed."""
    torch = _import_torch()
    wanted = (requested or "cpu").strip().lower()
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("DEVICE=%s requested but CUDA is unavailable; falling back to CPU", requested)
        return "cpu"
    return wanted


def find_model_folder(model_root: str, structure_code: str):
    """Locate the trained nnUNet folder for one structure, or None.

    Layout expected under the model bundle:
        <model_root>/<CODE>/**/<Dataset...>__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth

    Unlike the original CLI -- which took the first matching plans folder and
    only then discovered its checkpoint was missing -- a candidate is only
    accepted once its fold_0 checkpoint is confirmed present. A bundle
    containing a half-copied model therefore degrades to "this structure is
    unavailable" (reported to the caller) rather than crashing the run.
    """
    structure_root = os.path.join(model_root, structure_code)
    if not os.path.isdir(structure_root):
        return None

    pattern = os.path.join(structure_root, "**", PLANS_FOLDER_PATTERN)
    for candidate in sorted(glob.glob(pattern, recursive=True)):
        if os.path.isfile(os.path.join(candidate, "fold_0", CHECKPOINT_NAME)):
            return candidate

    # Also accept the plans folder being the structure folder itself.
    if os.path.isfile(os.path.join(structure_root, "fold_0", CHECKPOINT_NAME)):
        return structure_root
    return None


def _build_predictor(device: str):
    """Instantiate an nnUNetPredictor, tolerating nnUNet's renamed kwargs.

    nnUNet 2.x renamed `perform_everything_on_gpu` to
    `perform_everything_on_device` mid-series. Passing whichever the installed
    version actually declares keeps this working across the range instead of
    pinning the server to one nnUNet release.
    """
    torch = _import_torch()
    nnUNetPredictor = _import_predictor()

    options = {
        # See settings.AMASSS_TILE_STEP_SIZE: this is the one knob here that
        # changes the segmentation, so it stays configurable rather than tuned.
        "tile_step_size": float(settings.AMASSS_TILE_STEP_SIZE),
        "use_gaussian": True,
        # Equivalent to the CLI's --disable_tta: no test-time mirroring.
        "use_mirroring": False,
        "device": torch.device(device),
        "verbose": False,
        "verbose_preprocessing": False,
        "allow_tqdm": False,
    }
    accepted = set(inspect.signature(nnUNetPredictor.__init__).parameters)
    for name in ("perform_everything_on_device", "perform_everything_on_gpu"):
        if name in accepted:
            options[name] = device.startswith("cuda")
            break

    return nnUNetPredictor(**{k: v for k, v in options.items() if k in accepted})


# The resampler nnUNet's own plans name by default, and the only one we are
# willing to substitute. A bundle asking for anything else (no_resampling, a
# custom function) was configured that way deliberately and its geometry is not
# ours to reinterpret.
_STOCK_RESAMPLER = "resample_data_or_seg_to_shape"
_RESAMPLING_KEYS = ("resampling_fn_data", "resampling_fn_probabilities")


def _enable_gpu_resampling(predictor, device: str) -> bool:
    """Point this predictor's resamplers at the GPU. Returns whether it applied.

    Resampling, not inference, is what makes AMASSS slow: nnUNet's default
    resamplers are scipy splines running single-threaded on one core, and on a
    CBCT they outweigh the network by roughly seven to one (the numbers are in
    settings.AMASSS_GPU_RESAMPLING). nnUNet already ships torch equivalents, so
    there is nothing to reimplement -- only to select.

    It is selected by NAME: nnUNet resolves `resampling_fn_data` and
    `resampling_fn_probabilities` out of the configuration dict through
    `recursive_find_resampling_fn_by_name`, so rewriting those two names is
    enough to redirect both ends of the pipeline. No monkeypatching, and no
    reaching into nnUNet internals that a point release could rename.

    Mutating that dict is safe on two counts worth stating, because both would
    be silent bugs: `PlansManager._internal_resolve_configuration_inheritance`
    hands out a `deepcopy`, so this touches neither the shared plans nor another
    concurrent request's predictor -- and in particular the `torch.device` put
    in here never reaches the `plans.json` nnUNet writes next to its output,
    which `json.dump` could not serialize.
    """
    if not device.startswith("cuda"):
        return False

    try:
        from nnunetv2.preprocessing.resampling.resample_torch import (  # noqa: F401
            resample_torch_fornnunet,
        )
    except ImportError:
        logger.info("This nnUNet has no torch resampler; keeping the scipy one")
        return False

    torch = _import_torch()
    configuration_manager = predictor.configuration_manager
    configuration = configuration_manager.configuration

    if any(configuration.get(key) != _STOCK_RESAMPLER for key in _RESAMPLING_KEYS):
        logger.info("Model plans request a non-default resampler; leaving it alone")
        return False

    for key in _RESAMPLING_KEYS:
        configuration[key] = "resample_torch_fornnunet"
        # 'linear' is order 1, which is already what the plans ask for on the
        # probabilities. The input data is what changes: order 3 down to order 1,
        # because torch has no 3D cubic interpolation. That is the whole of the
        # numerical difference, and it is not nothing -- Dice against the scipy
        # pipeline ran 0.998 on the mandible but 0.978 on the cervical vertebra.
        # settings.AMASSS_GPU_RESAMPLING carries the full table.
        configuration[f"{key}_kwargs"] = {
            "is_seg": False,
            "device": torch.device(device),
            "mode": "linear",
        }

    # Both are `@property @lru_cache`, so a value read before this point would
    # otherwise outlive the swap.
    manager_class = type(configuration_manager)
    for key in _RESAMPLING_KEYS:
        getattr(manager_class, key).fget.cache_clear()

    return True


def predict_folder(model_folder: str, input_dir: str, output_dir: str, device: str) -> None:
    """Segment every `*_0000.nii.gz` in `input_dir`, writing masks to `output_dir`.

    Predicting a whole folder in one call is deliberate: the model is loaded
    once per structure instead of once per (scan x structure) as the original
    CLI did, which is the difference between N*S and S checkpoint loads on a
    batch run.
    """
    os.makedirs(output_dir, exist_ok=True)

    with _GPU_SEMAPHORE:
        predictor = _build_predictor(device)
        # Explicit path: no nnUNet_results env var, hence no cross-request race.
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(0,),
            checkpoint_name=CHECKPOINT_NAME,
        )

        on_gpu = bool(settings.AMASSS_GPU_RESAMPLING) and _enable_gpu_resampling(predictor, device)

        if on_gpu:
            # `predict_from_files` fans preprocessing and export out to *spawned*
            # processes. Each would have to build its own CUDA context to run a
            # GPU resampler -- more VRAM and more start-up than the resampling
            # itself -- so the GPU path runs everything in this process, where
            # the model already lives. That costs the CPU/GPU overlap
            # `predict_from_files` gets across a multi-scan batch; the
            # resampling win is several times larger, but recovering the overlap
            # (a reader thread feeding the GPU) is the obvious next step.
            predictor.predict_from_files_sequential(
                input_dir,
                output_dir,
                save_probabilities=False,
                overwrite=True,
            )
        else:
            predictor.predict_from_files(
                input_dir,
                output_dir,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
            )
        # TODO (deliberately not done here, see claude.md "out of scope"):
        # cache predictors across requests keyed by (model_folder, device).
        # Loading a checkpoint costs seconds and VRAM; a cache would need an
        # explicit eviction policy and GPU-memory accounting, which is the
        # model/GPU memory management work noted for later.

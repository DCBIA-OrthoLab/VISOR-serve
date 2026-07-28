"""nnUNet v2 inference, isolated from the rest of the AMASSS pipeline.

Three things this module exists to get right, all of which the original
Slicer CLI got wrong in ways that only hurt on a shared server:

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

logger = logging.getLogger("AMASSS.nnunet")

CHECKPOINT_NAME = "checkpoint_final.pth"
PLANS_FOLDER_PATTERN = "*__nnUNetPlans__3d_fullres"

# How many AMASSS inferences may touch the GPU at the same time. One by
# default: a 3d_fullres model plus its sliding-window buffers already fills a
# typical card. Raise it only on hardware you have actually measured.
_MAX_GPU_JOBS = max(1, int(os.getenv("AMASSS_MAX_GPU_JOBS", "1")))
_GPU_SEMAPHORE = threading.BoundedSemaphore(_MAX_GPU_JOBS)

_INSTALL_HINT = (
    "AMASSS needs the nnUNet v2 inference stack. Install it with "
    "`pip install -r requirements-amasss.txt` (see server/README.md)."
)


class ModelNotFoundError(FileNotFoundError):
    """No usable nnUNet model folder for a requested structure."""


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise RuntimeError(f"{_INSTALL_HINT} (missing: torch)") from exc
    return torch


def _import_predictor():
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise RuntimeError(f"{_INSTALL_HINT} (missing: nnunetv2)") from exc
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
        "tile_step_size": 0.5,
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

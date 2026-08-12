"""nnUNet v2 inference for BatchDentalSeg, isolated from the pipeline.

The upstream widget drove nnUNet through `SlicerNNUNetLib.Parameter` and a
`QProcess`, which is why most of that file is process management: killing a
crashed inference tree, reclaiming stray workers, a RAM watchdog. None of it
applies here -- the Python API returns when it is done, and the server already
bounds concurrency (see `settings.MAX_CONCURRENT_TOOLS` and the semaphore
below).

AMASSS carries a near-identical module. They are deliberately separate copies:
`registry.py` imports every tool at startup, so importing another tool's module
would make one tool's missing dependency take both out of the registry. The
same reason ASO and AREG each carry their own dicom.py.

torch and nnunetv2 are imported lazily, so a deployment without them still
boots and only a BatchDentalSeg *run* fails, with a message naming what is
missing.
"""

import inspect
import logging
import os
import threading

from base import ToolUnavailableError
from config import settings

logger = logging.getLogger("BatchDentalSeg.nnunet")

CHECKPOINT_NAME = "checkpoint_final.pth"

# A bundle is the directory holding these three, whatever it is called and
# however deeply the archive nested it. Discovered rather than assumed: the
# four bundles do not share one layout -- three are three flat files plus a
# fold_0/, and DentalSegmentator arrives as a zip with its own Dataset<n>/
# tree inside.
_REQUIRED_FILES = ("dataset.json", "plans.json")
# What BatchDentalSegLogic names an input case: one modality, index 0000.
_CASE_SUFFIX = "_0000.nii.gz"
_FOLD_DIR = "fold_0"

# See settings.BATCHDENTALSEG_MAX_GPU_JOBS. One by default, like AMASSS: these
# are 3d_fullres models and a single inference plus its sliding-window buffers
# already fills a typical card.
_GPU_SEMAPHORE = threading.BoundedSemaphore(
    max(1, int(settings.BATCHDENTALSEG_MAX_GPU_JOBS))
)

_INSTALL_HINT = (
    "BatchDentalSeg needs the nnUNet v2 inference stack. Install it with "
    "`pip install -r requirements.txt` (see server/README.md)."
)


class ModelNotFoundError(FileNotFoundError):
    """No usable nnUNet bundle for the requested model."""


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


def check_dependencies() -> None:
    """Import the whole stack once, before any scan is read.

    A missing dependency belongs to the server, not to one scan: discovered in
    the per-scan loop it would be reported as if the patient's data were at
    fault, once per scan.
    """
    _import_torch()
    _import_predictor()


def resolve_device(requested: str = None) -> str:
    """The device to actually use, falling back to CPU when CUDA is absent."""
    torch = _import_torch()
    wanted = (requested or settings.DEVICE or "cpu").strip().lower()
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("DEVICE=%s requested but CUDA is unavailable; falling back to CPU", wanted)
        return "cpu"
    return wanted


def find_model_folder(model_root: str):
    """The nnUNet bundle inside `model_root`, or None.

    Walks for the directory holding dataset.json, plans.json and
    fold_0/checkpoint_final.pth, all three confirmed before a candidate is
    accepted. A half-downloaded bundle therefore reports "this model is not
    installed" rather than failing inside nnUNet's loader.
    """
    if not os.path.isdir(model_root):
        return None

    for directory, _dirs, _files in os.walk(model_root):
        if not all(os.path.isfile(os.path.join(directory, name)) for name in _REQUIRED_FILES):
            continue
        if os.path.isfile(os.path.join(directory, _FOLD_DIR, CHECKPOINT_NAME)):
            return directory
    return None


def _build_predictor(device: str):
    """Instantiate an nnUNetPredictor, tolerating nnUNet's renamed kwargs.

    nnUNet 2.x renamed `perform_everything_on_gpu` to
    `perform_everything_on_device` mid-series; passing whichever the installed
    version declares keeps this working across the range.
    """
    torch = _import_torch()
    nnUNetPredictor = _import_predictor()

    options = {
        "tile_step_size": float(settings.BATCHDENTALSEG_TILE_STEP_SIZE),
        "use_gaussian": True,
        # No test-time mirroring, matching upstream's inference settings.
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

    return nnUNetPredictor(**{key: value for key, value in options.items() if key in accepted})


def predict_folder(model_folder: str, input_dir: str, output_dir: str, device: str,
                   on_case_start=None, on_case_done=None) -> None:
    """Segment every `*_0000.nii.gz` in `input_dir`, writing masks to `output_dir`.

    The checkpoint is loaded ONCE for the batch either way. What the callbacks
    change is the granularity of the nnUNet call:

    * without it, one `predict_from_files` over the whole folder, which lets
      nnUNet overlap the preprocessing of scan N+1 with the inference of scan N
      in its worker processes;
    * with it, one call per scan, so the caller learns which scan just finished
      and can post-process, report and SHIP it while the rest of the batch is
      still running.

    That is the trade: a streamed run gives up nnUNet's cross-scan pipelining
    to gain per-scan boundaries. It is the caller's choice rather than a
    setting, because it is a property of the request (did the client ask to be
    told?) and not of the deployment. Both callbacks take
    `(case_id, index, total)`; `on_case_done` fires once the mask is on disk,
    which is what lets the caller post-process and ship that patient while the
    next one is on the GPU.

    NOTE: AMASSS additionally redirects nnUNet's resamplers to the GPU
    (settings.AMASSS_GPU_RESAMPLING), which is worth ~2.5x there. It is
    deliberately not done here yet: it drops the input resampling from spline
    order 3 to order 1, and nothing has measured what that costs THESE models.
    """
    os.makedirs(output_dir, exist_ok=True)

    with _GPU_SEMAPHORE:
        predictor = _build_predictor(device)
        # An explicit path, never nnUNet's `nnUNet_results` environment
        # variable: tools run concurrently in worker threads and os.environ is
        # process-global, so two overlapping requests would swap model paths.
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(0,),
            checkpoint_name=CHECKPOINT_NAME,
        )

        if on_case_start is None and on_case_done is None:
            predictor.predict_from_files(
                input_dir,
                output_dir,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
            )
            return

        cases = sorted(
            name for name in os.listdir(input_dir) if name.endswith(_CASE_SUFFIX)
        )
        for index, name in enumerate(cases):
            case_id = name[: -len(_CASE_SUFFIX)]
            if on_case_start is not None:
                on_case_start(case_id, index, len(cases))
            # A list of lists is nnUNet's "one case, one modality" shape; the
            # predictor object -- and therefore the loaded checkpoint -- is the
            # same one across the whole loop.
            #
            # The output path is TRUNCATED: nnUNet appends the file ending from
            # its own dataset.json (`isfile(i + file_ending)` in
            # _manage_input_and_output_lists), so passing "case_0000.nii.gz"
            # here writes "case_0000.nii.gz.nii.gz" and the caller finds
            # nothing where it looked. The folder form above hides this,
            # because nnUNet builds the names itself from the case ids.
            predictor.predict_from_files(
                [[os.path.join(input_dir, name)]],
                [os.path.join(output_dir, case_id)],
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
            )
            if on_case_done is not None:
                # The caller post-processes and ships this patient HERE, while
                # the loop moves on to the next one.
                on_case_done(case_id, index, len(cases))

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

# What nnUNet's plans name when they ask for the stock (scipy) resampler, and
# the two configuration keys that select one. See _enable_gpu_resampling.
_STOCK_RESAMPLER = "resample_data_or_seg_to_shape"
_RESAMPLING_KEYS = ("resampling_fn_data", "resampling_fn_probabilities")
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


def _largest_scan_voxels(input_dir: str) -> int:
    """Voxels in the biggest scan of the batch, read from the HEADERS only.

    The resampled array lives on the original scan's grid, so this is what
    decides whether the GPU resampler fits. Reading headers costs milliseconds;
    loading the pixels would cost what we are trying to save.
    """
    try:
        import SimpleITK as sitk
    except ImportError:  # pragma: no cover - SimpleITK is a hard dependency
        return 0

    largest = 0
    reader = sitk.ImageFileReader()
    for name in os.listdir(input_dir):
        if not name.endswith(_CASE_SUFFIX):
            continue
        reader.SetFileName(os.path.join(input_dir, name))
        try:
            reader.ReadImageInformation()
        except Exception:  # noqa: BLE001 - an unreadable scan is reported later, per scan
            continue
        voxels = 1
        for extent in reader.GetSize():
            voxels *= int(extent)
        largest = max(largest, voxels)
    return largest


def _gpu_resampling_fits(predictor, device: str, input_dir: str) -> bool:
    """Whether the torch resampler's biggest allocation fits in free VRAM.

    **This guard is why the port is not a copy of AMASSS's.** The array being
    resampled is `(classes, Z, Y, X)` float32, so the cost scales with the
    number of classes -- and UniversalLab emits 55 of them. Measured on a
    512x512x365 CBCT: nnUNet asked for a single 11.2 GiB allocation, and around
    20 GiB peak. AMASSS never had to think about this because its models emit
    two or three classes; here the very model that suffers most from the CPU
    resampler is also the one that strains the card.

    Without the guard a 16 or 24 GB card would take the whole run down with a
    CUDA OOM deep inside nnUNet, minutes in. Falling back to the (slow, correct)
    scipy path is always better than failing.
    """
    torch = _import_torch()
    try:
        free_bytes, _total = torch.cuda.mem_get_info(torch.device(device))
    except Exception:  # noqa: BLE001 - an exotic device; assume we cannot tell
        logger.info("Cannot read free VRAM; keeping the scipy resampler")
        return False

    classes = int(getattr(predictor.label_manager, "num_segmentation_heads", 0) or 0)
    voxels = _largest_scan_voxels(input_dir)
    if classes <= 0 or voxels <= 0:
        return False

    # Calibrated against the measured run rather than guessed. On a 512x512x365
    # CBCT with UniversalLab's 55 classes, the output array alone is 19.6 GiB
    # (torch asked for exactly that) and the model-grid input another 11.2 GiB,
    # so the real peak is around 31 GiB -- and it ran on a card with 47 GiB
    # free. A factor of 2 on the output array covers that with slack; 2.5
    # would have refused the very card the measurement was taken on, which is
    # the failure mode a guard must not have.
    needed = int(classes * voxels * 4 * 2)
    fits = needed < free_bytes
    logger.info(
        "BatchDentalSeg GPU resampling: %d class(es) x %.1f Mvox needs ~%.1f GiB, "
        "%.1f GiB free -> %s",
        classes, voxels / 1e6, needed / 2**30, free_bytes / 2**30,
        "on" if fits else "off (falling back to the CPU resampler)",
    )
    return fits


def _enable_gpu_resampling(predictor, device: str) -> bool:
    """Point this predictor's resamplers at the GPU. Returns whether it applied.

    Ported from AMASSS, where it was worth 2.5x, and for the same reason: the
    network is not what makes these runs long. Measured here on a 512x512x365
    CBCT with UniversalLab, the GPU spends **40 seconds** on three scans and
    nnUNet then spends **8 minutes 19** resampling the logits back on CPU --
    scipy splines on one core, over a (55, Z, Y, X) float32 array. nnUNet ships
    torch equivalents, so there is nothing to reimplement, only to select.

    Selected by NAME: nnUNet resolves both resampling functions out of the
    configuration dict via `recursive_find_resampling_fn_by_name`, so rewriting
    the two names redirects both ends. No monkeypatching.

    Mutating that dict is safe because PlansManager hands out a `deepcopy`: it
    touches neither the shared plans nor a concurrent request, and the
    `torch.device` put in here never reaches the `plans.json` nnUNet writes
    beside its output (which `json.dump` could not serialize).
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
        # A bundle whose plans pin their own resampler opts itself out: we have
        # no idea what it was trained to expect.
        logger.info("Model plans request a non-default resampler; leaving it alone")
        return False

    for key in _RESAMPLING_KEYS:
        configuration[key] = "resample_torch_fornnunet"
        # 'linear' is order 1, already what the plans ask for on the
        # probabilities. The input data drops from order 3 to order 1 (torch
        # has no 3D cubic interpolation): that is the whole numerical
        # difference, and settings.BATCHDENTALSEG_GPU_RESAMPLING carries its
        # measured Dice.
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

    **What that split costs depends entirely on the path.** On the CPU
    resampler it is expensive: nnUNet overlaps the export of scan N with the
    inference of N+1 across worker processes, and splitting the call loses that
    AND respawns those processes per scan -- measured at +62% wall clock on a
    three-scan cohort (601s -> 972s). On the GPU path (see
    _enable_gpu_resampling) it is nearly free, because that path is sequential
    and in-process to begin with: there is no cross-scan overlap left to lose.

    Both callbacks take `(case_id, index, total)`; `on_case_done` fires once
    the mask is on disk, which is what lets the caller post-process and ship
    that patient while the next one is still running.
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

        on_gpu = (
            bool(settings.BATCHDENTALSEG_GPU_RESAMPLING)
            and device.startswith("cuda")
            and _gpu_resampling_fits(predictor, device, input_dir)
            and _enable_gpu_resampling(predictor, device)
        )

        def predict(inputs, outputs) -> None:
            """One nnUNet call, on whichever path this run is taking.

            The GPU path must run everything IN THIS PROCESS:
            `predict_from_files` fans preprocessing and export out to spawned
            processes, and each would need its own CUDA context to run a GPU
            resampler.
            """
            if on_gpu:
                predictor.predict_from_files_sequential(
                    inputs, outputs, save_probabilities=False, overwrite=True
                )
            else:
                predictor.predict_from_files(
                    inputs,
                    outputs,
                    save_probabilities=False,
                    overwrite=True,
                    num_processes_preprocessing=2,
                    num_processes_segmentation_export=2,
                )

        if on_case_start is None and on_case_done is None:
            predict(input_dir, output_dir)
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
            predict([[os.path.join(input_dir, name)]], [os.path.join(output_dir, case_id)])
            if on_case_done is not None:
                # The caller post-processes and ships this patient HERE, while
                # the loop moves on to the next one.
                on_case_done(case_id, index, len(cases))

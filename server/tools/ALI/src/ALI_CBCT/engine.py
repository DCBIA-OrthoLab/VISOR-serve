"""The CBCT landmark pipeline: preprocess, then one agent per landmark.

Ported from ALI_CBCT/ALI_CBCT.py. What the CLI envelope did -- argparse,
`ast.literal_eval` on stringified lists, `sys.exit()`, the
`<filter-progress>` protocol, the caller-supplied temp and output folders --
is gone: on this server scratch space and cleanup belong to main.py, and a
failure has to be an exception, since a `SystemExit` raised inside a worker
thread does not surface as a clean 500.

Beyond that, three behaviours changed on purpose:

* **Weights are discovered, landmarks are requested.** The bundle's folder
  tree says which landmarks it can predict; the caller's region selection says
  which are wanted. What is wanted but absent is reported, not silently
  skipped.
* **One markups file per scan**, holding every landmark found. The original
  wrote one file per anatomical region, so every downstream tool (ASO, AREG,
  AutoMatrix) had to recombine them by hand.
* **GPU work is serialized** by ALI's own semaphore, independently of
  MAX_CONCURRENT_TOOLS: several concurrent requests each holding a DenseNet
  and a padded volume do not fit on one card.
"""

import logging
import os
import shutil
import sys
import threading
import time

import numpy as np

from config import settings

from ..markups import MARKUPS_EXTENSION
from ..markups import write as write_markups
from . import landmarks as catalog
from . import preprocess
from .agent import AGENT_FOV, MOVEMENT_COUNT, Agent, NotFound
from .brain import Brain, import_torch

logger = logging.getLogger("ALI.cbct")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")
    )
    logger.addHandler(_handler)

_GPU_SEMAPHORE = threading.BoundedSemaphore(max(1, int(settings.ALI_MAX_GPU_JOBS)))

# Per-landmark search budgets when ALI_SEARCH_MAX_SECONDS is unset. Each search
# step is a forward pass, so CPU-only inference needs several times longer than
# a GPU to reach the same place.
_DEFAULT_BUDGET_SECONDS = {"cuda": 15.0, "cpu": 60.0}

COMPOUND_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz")


def search_budget(device: str) -> float:
    if settings.ALI_SEARCH_MAX_SECONDS is not None:
        return float(settings.ALI_SEARCH_MAX_SECONDS)
    return _DEFAULT_BUDGET_SECONDS["cuda" if device.startswith("cuda") else "cpu"]


def resolve_device(requested: str = None) -> str:
    """The device to actually use, falling back to CPU when CUDA is absent.

    Read from settings rather than decided by `torch.cuda.is_available()` deep
    in the code -- which the original did independently in five modules, so a
    server configured for CPU still used a card that happened to be present.
    """
    torch = import_torch()
    wanted = (requested or settings.DEVICE or "cpu").strip().lower()
    if wanted.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("DEVICE=%s requested but CUDA is unavailable; falling back to CPU", wanted)
        return "cpu"
    return wanted


def check_dependencies() -> None:
    """Import the whole lazy stack once, before any scan is touched.

    A missing dependency is a property of the SERVER, not of one scan. Without
    this, the per-scan `except` below catches it as if a single patient's data
    were at fault: every scan fails identically -- each only after a complete
    histogram correction, so a 200-scan cohort spends real minutes discovering
    the same thing 200 times -- and the run then ends on "ALI produced no
    landmarks for any scan", which buries the one line that says what to
    install.

    Raising here instead surfaces the install message itself, immediately.
    """
    from .environment import import_transforms

    import_torch()
    preprocess.import_itk()
    import_transforms()


def scan_stem(filename: str) -> str:
    """'patient01.nii.gz' -> 'patient01', compound extensions preserved."""
    lower = filename.lower()
    for suffix in COMPOUND_EXTENSIONS:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return os.path.splitext(filename)[0]


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

def discover_weights(model_path: str) -> dict:
    """{landmark: {scale key: checkpoint path}} for a CBCT model bundle.

    The layout is the one every published packaging of these weights uses:
    `<...>/<landmark>/<scale>/<anything>.pth`, where the scale folder is named
    after the spacing (`1`, `0-3`). Walked recursively, so it does not matter
    how deeply the region folders nest the landmarks -- which is what lets the
    eight separately published region archives be unpacked side by side into
    a single bundle.

    Aliased landmark spellings are folded onto the canonical one, so a bundle
    naming the impacted canines the way the old Slicer UI did resolves to the
    same landmark as one naming them the way the CLI did.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"CBCT model bundle is not a directory: {model_path}")

    weights: dict = {}
    for root, _dirs, files in os.walk(model_path):
        checkpoints = [name for name in sorted(files) if name.endswith(".pth")]
        scale = os.path.basename(root)
        if not checkpoints or scale not in catalog.SCALE_KEYS:
            continue
        label = catalog.canonical(os.path.basename(os.path.dirname(root)))
        weights.setdefault(label, {})[scale] = os.path.join(root, checkpoints[0])

    # A landmark needs a checkpoint at EVERY scale: the agent walks the coarse
    # one and then the fine one. Filtering here means a half-copied bundle is
    # reported up front rather than failing in the middle of the run.
    return {
        label: scales
        for label, scales in weights.items()
        if all(scale in scales for scale in catalog.SCALE_KEYS)
    }


def requested_landmarks(weights: dict, regions):
    """(runnable, without_model, ungrouped) for a bundle and a region selection.

    "Requested" is every catalog landmark of the selected regions, plus any
    landmark the bundle itself provides whose region is selected -- so weights
    published for a landmark this catalog has not heard of are not silently
    ignored: they surface as `ungrouped`.
    """
    regions = set(regions)
    wanted = set(catalog.landmarks_in(regions))
    ungrouped = []

    for label in weights:
        group = catalog.group_of(label)
        if group == catalog.UNGROUPED:
            ungrouped.append(label)
        elif group in regions:
            wanted.add(label)

    runnable = tuple(sorted(label for label in wanted if label in weights))
    without_model = sorted(label for label in wanted if label not in weights)
    return runnable, without_model, sorted(ungrouped)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _prepare_scan(scan_path: str, work_dir: str) -> dict:
    """Histogram-correct one scan and resample it to every scale.

    Returns {scale key: path}, written under `work_dir` -- always the request's
    own scratch directory, never next to the input, which is either read-only
    server-side data or an extracted upload.
    """
    os.makedirs(work_dir, exist_ok=True)

    base = os.path.basename(scan_path)
    corrected = os.path.join(work_dir, base)
    preprocess.correct_histogram(scan_path, corrected)

    stem = scan_stem(base)
    resampled = {}
    for spacing in catalog.SCALE_SPACINGS:
        key = catalog.scale_key(spacing)
        # Written as NIfTI whatever the input was: this is a real read/write
        # conversion, not the rename the original relied on for NRRD and GIPL.
        destination = os.path.join(work_dir, f"{stem}_sp{key}.nii.gz")
        preprocess.set_spacing(corrected, spacing, destination)
        resampled[key] = destination
    return resampled


def predict_landmarks(
    scans: list,
    model_path: str,
    regions=None,
    prediction_ID: str = "Pred",
    output_dir: str = None,
    scratch_dir: str = None,
    device: str = None,
) -> dict:
    """Place landmarks on every scan; return the run report.

    `scans` is a list of `(absolute path, key)` pairs, where the key is the
    scan's path relative to the input root. ALILogic owns discovery and hands
    the keys over; this owns inference. Keying by relative path rather than by
    base name is what stops two patients called `scan.nii.gz` in different
    subfolders from overwriting each other -- silently, in the original, both
    in the working dictionary and in the flat output folder.

    The loop is scan-outer, landmark-inner: one scan's volumes and one
    landmark's networks are in memory at a time. Inverting it would load each
    checkpoint once instead of once per scan, but would need either every scan
    resident at both spacings, or all 112 possible landmarks' networks resident
    on the card. Neither fits a cohort.
    """
    started_at = time.monotonic()

    check_dependencies()
    device = resolve_device(device)
    regions = tuple(regions) if regions is not None else catalog.REGION_CODES
    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"
    budget = search_budget(device)

    weights = discover_weights(model_path)
    if not weights:
        raise FileNotFoundError(
            f"No CBCT landmark weights found in '{os.path.basename(model_path)}'. Expected "
            f"<bundle>/**/<landmark>/<scale>/*.pth, with scale folders named "
            f"{' and '.join(catalog.SCALE_KEYS)}."
        )

    runnable, without_model, ungrouped = requested_landmarks(weights, regions)
    if not runnable:
        region_names = ", ".join(catalog.REGION_DISPLAY_NAMES.get(code, code) for code in regions)
        raise FileNotFoundError(
            f"'{os.path.basename(model_path)}' has no weights for any landmark of the selected "
            f"region(s) ({region_names}). It provides: {', '.join(sorted(weights)) or 'nothing'}."
        )

    preprocessed_dir = os.path.join(scratch_dir, "preprocessed")
    logger.info(
        "ALI CBCT: %d scan(s), %d landmark(s), device=%s", len(scans), len(runnable), device
    )

    scan_reports = {}
    for scan_index, (scan_path, key) in enumerate(scans, start=1):
        record = {
            "input": os.path.basename(scan_path),
            "status": "pending",
            "landmarks_found": [],
            "landmarks_failed": {},
            "files": [],
        }
        scan_reports[key] = record
        scan_started = time.monotonic()
        # Position in the batch, never the scan's name: a file name is patient
        # metadata and this server does not write it to a log.
        logger.info("scan %d/%d: preprocessing", scan_index, len(scans))

        try:
            _predict_one_scan(
                scan_path=scan_path,
                key=key,
                record=record,
                weights=weights,
                runnable=runnable,
                device=device,
                budget=budget,
                preprocessed_dir=preprocessed_dir,
                output_dir=output_dir,
                prediction_ID=prediction_ID,
                scan_index=scan_index,
                scan_total=len(scans),
            )
            record["status"] = "ok"
        except Exception as exc:
            # One unreadable or hopeless scan must not cost the other 199.
            logger.exception("ALI CBCT failed on one scan")
            record["status"] = "failed"
            record["error"] = str(exc)
        record["duration_seconds"] = round(time.monotonic() - scan_started, 2)
        logger.info(
            "scan %d/%d: %s -- %d found, %d failed, %.0fs",
            scan_index,
            len(scans),
            record["status"],
            len(record["landmarks_found"]),
            len(record["landmarks_failed"]),
            record["duration_seconds"],
        )

    processed = [record for record in scan_reports.values() if record["status"] == "ok"]
    if not processed:
        first_error = next(
            (record.get("error") for record in scan_reports.values() if record.get("error")),
            "unknown",
        )
        raise RuntimeError(f"ALI produced no landmarks for any scan. First error: {first_error}")

    written = sum(len(record["landmarks_found"]) for record in scan_reports.values())
    never_found = sorted(
        {label for record in scan_reports.values() for label in record["landmarks_failed"]}
    )
    logger.info(
        "ALI CBCT done: %d/%d scan(s), %d landmark(s) written, %.0fs",
        len(processed), len(scan_reports), written, time.monotonic() - started_at,
    )
    if without_model:
        logger.info("  not in this bundle (%d): %s", len(without_model), ", ".join(without_model))
    if never_found:
        logger.info("  never converged (%d): %s", len(never_found), ", ".join(never_found))

    return {
        "mode": "CBCT",
        "device": device,
        "prediction_ID": prediction_ID,
        "regions": [catalog.REGION_DISPLAY_NAMES.get(code, code) for code in regions],
        "landmarks_requested": list(runnable),
        # Named after AMASSS's `structures_without_model` and read by the
        # Slicer module: a landmark listed here means "use another bundle",
        # whereas one in a scan's `landmarks_failed` means "this scan is hard".
        # Kept apart because the fix differs.
        "landmarks_without_model": without_model,
        "landmarks_ungrouped": ungrouped,
        "scans": scan_reports,
        "summary": {
            "total": len(scan_reports),
            "processed": len(processed),
            "failed": len(scan_reports) - len(processed),
        },
        "duration_seconds": round(time.monotonic() - started_at, 2),
    }


def _predict_one_scan(scan_path, key, record, weights, runnable, device, budget,
                      preprocessed_dir, output_dir, prediction_ID,
                      scan_index: int = 1, scan_total: int = 1) -> None:
    """Preprocess one scan, run every requested landmark on it, write its file.

    Logs progress as it goes. The search is the long part -- 119 landmarks is
    minutes of it -- and without a line in between, a run is indistinguishable
    from a hang. Everything logged here is a COUNT or an anatomical label:
    never a file name, which is patient metadata.
    """
    from .environment import Environment

    scan_work_dir = os.path.join(preprocessed_dir, key.replace(os.sep, "_"))
    started_at = time.monotonic()
    images = _prepare_scan(scan_path, scan_work_dir)
    logger.info(
        "scan %d/%d: preprocessed in %.0fs, searching %d landmark(s)",
        scan_index, scan_total, time.monotonic() - started_at, len(runnable),
    )

    # About ten progress lines per scan, whatever the number of landmarks:
    # one per landmark would be 119 lines here and 23 800 on a 200-scan batch.
    progress_every = max(1, len(runnable) // 10)

    environment = Environment(
        patient_id=key,
        # Half a field of view plus one, so a box centred anywhere inside the
        # volume is still complete once the borders are padded.
        padding=np.array(AGENT_FOV) / 2 + 1,
        device=device,
    )

    positions = {}
    try:
        environment.load_images(images)

        search_started = time.monotonic()
        for index, label in enumerate(runnable, start=1):
            brain = Brain(catalog.SCALE_KEYS, device, out_channels=MOVEMENT_COUNT)
            try:
                brain.load(weights[label])
                agent = Agent(
                    target=label,
                    scale_keys=catalog.SCALE_KEYS,
                    brain=brain,
                    environment=environment,
                )
                with _GPU_SEMAPHORE:
                    voxel_position = agent.search(budget)
            except NotFound as exc:
                # At INFO, not DEBUG: these are rare (6 of 119 on the reference
                # scan) and they are exactly what someone watching the log wants
                # to see, without having to open the archive to find out.
                logger.info("  %s: not found -- %s", label, exc)
                record["landmarks_failed"][label] = str(exc)
                continue
            except Exception as exc:
                # A broken checkpoint, an out-of-memory, anything: this
                # landmark is lost, the rest of the scan is not.
                logger.exception("Landmark search raised for '%s'", label)
                record["landmarks_failed"][label] = f"{type(exc).__name__}: {exc}"
                continue
            finally:
                brain.release()

            positions[label] = environment.physical_position(
                catalog.SCALE_KEYS[-1], voxel_position
            )
            record["landmarks_found"].append(label)
            logger.debug("  %s: found", label)

            if index % progress_every == 0 or index == len(runnable):
                elapsed = time.monotonic() - search_started
                remaining = (elapsed / index) * (len(runnable) - index)
                logger.info(
                    "scan %d/%d: %d/%d landmarks (%d found, %d failed), "
                    "%.0fs elapsed, ~%.0fs left",
                    scan_index, scan_total, index, len(runnable),
                    len(record["landmarks_found"]), len(record["landmarks_failed"]),
                    elapsed, remaining,
                )
    finally:
        environment.release()
        # Deleted per scan rather than left to the end-of-request cleanup:
        # three volumes per scan is gigabytes across a cohort, and holding all
        # of them until the response streams is how TEMP_DIR fills up.
        shutil.rmtree(scan_work_dir, ignore_errors=True)

    if not positions:
        raise RuntimeError("no landmark converged on this scan")

    # ONE file per scan, holding every region, in the input's own tree.
    destination = os.path.join(
        output_dir,
        os.path.dirname(key),
        f"{scan_stem(os.path.basename(scan_path))}_lm_{prediction_ID}{MARKUPS_EXTENSION}",
    )
    record["files"].append(write_markups(positions, destination))

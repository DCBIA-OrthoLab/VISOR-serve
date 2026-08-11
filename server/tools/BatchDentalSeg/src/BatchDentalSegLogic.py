"""BatchDentalSeg -- dental CT/CBCT segmentation with the DentalSegmentator
family of nnUNet models.

Ported from `BATCHDENTALSEG/BATCHDENTALSEGLib/SegmentationWidget.py`. That file
is 2940 lines, and most of it is not this pipeline: a queue table, a RAM
watchdog, killing nnUNet processes a crashed scan left behind, a "free memory"
button, a cool-down between scans, restoring the queue from disk after a
crash. All of it exists because the widget runs inside Slicer on a clinician's
laptop and has to survive being out of memory. On this server the queue is a
folder argument, concurrency is bounded by MAX_CONCURRENT_TOOLS and the GPU
semaphore, and a failure is an exception. None of it is ported.

Two entry points, following the AMASSSLogic precedent:

* `segment(...)` -> `SegmentationRun`, the reusable API: the produced files
  plus a report, zipping nothing. This is what another server-side tool calls.
* `main(...)` -> path to the output directory, the schema adapter
  BatchDentalSeg.py uses.

Deliberately NOT ported, each for a stated reason:

* the runtime model download from GitHub releases. A server holding patient
  data does not make outbound calls mid-request; bundles are staged with
  `scripts/setup-models.sh --tool BatchDentalSeg` and a missing one is an
  error naming that command.
* the auto-crop. Upstream applies it only when its RAM preflight fails, as a
  mitigation for a laptop, and it changes what the network sees. It is not a
  clinical step and this server has the memory.
* the mirroring resolution (`onResolveMirroring`). It is a button the user
  presses after looking at the result, not part of the automatic pipeline.
* the mesh exports (STL/OBJ/GLTF/VTK). The segmentation is the deliverable
  every downstream tool consumes; surfaces are the obvious next addition.
"""

import json
import logging
import os
import time
import zipfile

import numpy as np
import SimpleITK as sitk

import file_utils
from base import ToolArgumentError
from config import settings

from . import catalogs, nnunet_runner

logger = logging.getLogger("BatchDentalSeg")

# What nnUNet expects an input case to be called: one modality, index 0000.
_NNUNET_SUFFIX = "_0000.nii.gz"


class SegmentationRun:
    """Result of `segment()`: where the files are, and what actually happened.

    Returned instead of a bare path so a calling module gets the per-scan
    outputs directly and can tell a partial run from a complete one.
    """

    def __init__(self, output_dir: str, report: dict, scans: list):
        self.output_dir = output_dir
        self.report = report
        self.scans = scans

    @property
    def segmentation_files(self) -> list:
        return [path for scan in self.scans for path in scan.get("segmentations", [])]


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def is_previous_output(filename: str, suffix: str) -> bool:
    """True if this file looks like something a previous run wrote.

    Without it, a second run on the same folder feeds the first run's
    segmentations back in as input scans.
    """
    base, _extension = file_utils.split_scan_extension(os.path.basename(filename))
    if suffix and base.endswith(f"_{suffix}"):
        return True
    # The per-segment files, whose names end in the segment they hold.
    return any(base.endswith(f"_{suffix}_{name.replace(' ', '-')}")
               for model in catalogs.MODELS.values()
               for name in model.labels)


def discover_scans(input_path: str, suffix: str, scratch_dir: str) -> list:
    """Resolve the `input` argument into a list of scan files.

    Accepts the three shapes one schema argument can carry: a single scan, a
    zip of a folder of them, or a folder served from the read-only data store.
    Folder scanning is RECURSIVE, so a nested cohort is processed whole.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path) and input_path.lower().endswith(".zip"):
        if not zipfile.is_zipfile(input_path):
            raise ValueError(f"Input has a .zip extension but is not a zip archive: {input_path}")
        # The batch arrives as an archive rather than pre-extracted by main.py
        # (the schema declares "volume_or_zip_file" first), so the zip-bomb cap
        # main.py would have applied is applied here instead.
        input_path = file_utils.extract_zip(
            input_path,
            os.path.join(scratch_dir, "input_extracted"),
            strip_single_root=True,
            max_total_bytes=settings.MAX_EXTRACTED_MB * 1024 * 1024,
        )

    if os.path.isfile(input_path):
        return [input_path]

    scans = []
    for root, _dirs, files in os.walk(input_path):
        for name in sorted(files):
            if not name.lower().endswith(file_utils.SCAN_EXTENSIONS):
                continue
            if is_previous_output(name, suffix):
                continue
            scans.append(os.path.join(root, name))
    return sorted(scans)


def resolve_model(model_path: str) -> tuple:
    """`(the model this bundle is, the nnUNet folder inside it)`.

    The bundle the caller picked identifies the model: its directory name is
    what the manifest downloaded it as, and what
    GET /tools/BatchDentalSeg/data offered. Both failures are 422 rather than
    500 -- nothing about the request is malformed, the deployment's data is.
    """
    name = os.path.basename(str(model_path).rstrip(os.sep))
    try:
        model = catalogs.get(name)
    except KeyError:
        raise ToolArgumentError(
            f"'{name}' is not a BatchDentalSeg model. This tool knows: "
            f"{catalogs.describe_all()}."
        )

    folder = nnunet_runner.find_model_folder(str(model_path))
    if folder is None:
        raise ToolArgumentError(
            f"The '{name}' bundle on this server holds no usable nnUNet model "
            f"(expected dataset.json, plans.json and fold_0/{nnunet_runner.CHECKPOINT_NAME}). "
            f"Re-fetch it with `scripts/setup-models.sh --tool BatchDentalSeg`."
        )
    return model, folder


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _convert_to_nifti(scan_path: str, destination: str) -> None:
    """Real format conversion into the NIfTI file nnUNet reads.

    A read plus a write, never a rename: nnUNet's reader picks its format from
    the extension, so an NRRD renamed to .nii.gz is read as garbage. The voxel
    type is left alone -- nnUNet casts to float32 itself.
    """
    sitk.WriteImage(sitk.ReadImage(scan_path), destination)


def _match_reference_geometry(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Put a predicted mask back onto the original scan's exact grid.

    nnUNet resamples to the model's spacing and back, which can leave a mask
    whose origin differs from its scan in the last float digits -- enough for a
    viewer to draw it offset from the anatomy it describes.
    """
    if (
        mask.GetSize() == reference.GetSize()
        and np.allclose(mask.GetSpacing(), reference.GetSpacing(), atol=1e-4)
    ):
        mask.CopyInformation(reference)
        return mask

    return sitk.Resample(
        mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID()
    )


def _write_segmentation(image: sitk.Image, destination: str) -> str:
    """Write a label volume, always compressed: these are long runs of one
    value, so gzip takes them roughly a hundred times down."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    sitk.WriteImage(image, destination, useCompression=True)
    return destination


def _split_segments(labels: sitk.Image, model: catalogs.Model, base: str,
                    extension: str, output_dir: str, suffix: str) -> list:
    """One binary file per label the network actually emitted.

    Only the labels PRESENT in this scan are written: a full UniversalLab run
    would otherwise produce 55 files per patient, most of them empty, and an
    empty mask is indistinguishable from a structure the model failed on.
    """
    array = sitk.GetArrayViewFromImage(labels)
    present = set(int(value) for value in np.unique(array) if value != 0)

    written = []
    for name, value in model.labels.items():
        if value not in present:
            continue
        binary = sitk.GetImageFromArray((array == value).astype(np.uint8))
        binary.CopyInformation(labels)
        safe_name = name.replace(" ", "-").replace("/", "-")
        destination = os.path.join(
            output_dir, f"{base}_{suffix}_{safe_name}{extension}"
        )
        written.append(_write_segmentation(binary, destination))
    return written


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def segment(
    input_path: str,
    model_path: str,
    separate_segments: bool = False,
    prediction_ID: str = "Seg",
    device: str = None,
    scratch_dir: str = None,
) -> SegmentationRun:
    """Segment every scan under `input_path` with one hosted model bundle.

    `model_path` is the resolved path of the bundle the caller picked; its
    directory name is which of catalogs.MODELS it is.
    """
    started = time.monotonic()
    scratch_dir = scratch_dir or file_utils.make_scratch_dir("batchdentalseg_")

    # Before any scan is read: a missing dependency and an unusable bundle are
    # both properties of the server, and discovering either inside the loop
    # would report them once per patient.
    nnunet_runner.check_dependencies()
    model, model_folder = resolve_model(model_path)
    device = nnunet_runner.resolve_device(device)

    scans = discover_scans(input_path, prediction_ID, scratch_dir)
    if not scans:
        raise ToolArgumentError(
            "No scan found in the input. Expected one of "
            f"{', '.join(file_utils.SCAN_EXTENSIONS)}, or a .zip of a folder of them."
        )

    input_root = input_path if os.path.isdir(input_path) else os.path.dirname(scans[0])
    output_dir = os.path.join(scratch_dir, f"BatchDentalSeg_{prediction_ID}")
    os.makedirs(output_dir, exist_ok=True)

    # One nnUNet call for the whole batch, so the checkpoint is loaded once.
    # The case ids are positional, never derived from the patient's file name:
    # nnUNet writes its output beside its input under the same id, and two
    # scans called scan.nii.gz in different subfolders would collide.
    nnunet_input = os.path.join(scratch_dir, "nnunet_in")
    nnunet_output = os.path.join(scratch_dir, "nnunet_out")
    os.makedirs(nnunet_input, exist_ok=True)

    def _describe(path: str) -> str:
        return os.path.relpath(path, input_root) if os.path.isdir(input_root) else os.path.basename(path)

    cases = {}
    failed_conversions = []
    for index, scan in enumerate(scans):
        case_id = f"case_{index:04d}"
        try:
            _convert_to_nifti(scan, os.path.join(nnunet_input, f"{case_id}{_NNUNET_SUFFIX}"))
        except Exception as exc:  # noqa: BLE001 - one unreadable scan must not end the batch
            # Guarded per scan, and deliberately: this loop runs BEFORE
            # inference, so without it one corrupt file in a cohort of forty
            # would abort the whole run before a single scan was segmented.
            logger.exception("BatchDentalSeg: could not read a scan")
            failed_conversions.append(
                {
                    "case_id": case_id,
                    "input": _describe(scan),
                    "status": "failed",
                    "error": f"could not be read ({type(exc).__name__}: {exc})",
                }
            )
            continue
        cases[case_id] = scan

    if not cases:
        raise ToolArgumentError(
            "None of the input scans could be read. Check the files are valid "
            "medical volumes."
        )

    logger.info("BatchDentalSeg: %d scan(s), model=%s, device=%s", len(cases), model.name, device)
    nnunet_runner.predict_folder(model_folder, nnunet_input, nnunet_output, device)

    report_scans = list(failed_conversions)
    for case_id, scan in cases.items():
        entry = {"case_id": case_id, "input": _describe(scan)}
        predicted = os.path.join(nnunet_output, f"{case_id}.nii.gz")
        if not os.path.isfile(predicted):
            # Reported per scan rather than raised: one unreadable patient in a
            # cohort of forty must not lose the other thirty-nine.
            entry.update(status="failed", error="nnUNet produced no output for this scan")
            report_scans.append(entry)
            continue

        try:
            reference = sitk.ReadImage(scan)
            labels = _match_reference_geometry(sitk.ReadImage(predicted), reference)

            base, extension = file_utils.split_scan_extension(os.path.basename(scan))
            extension = file_utils.compressed_extension(extension)
            # The output mirrors the input tree, so two patients whose scans
            # share a file name stay apart.
            relative = (
                os.path.relpath(os.path.dirname(scan), input_root)
                if os.path.isdir(input_root)
                else "."
            )
            scan_output_dir = os.path.normpath(os.path.join(output_dir, relative))

            produced = [
                _write_segmentation(
                    labels, os.path.join(scan_output_dir, f"{base}_{prediction_ID}{extension}")
                )
            ]
            if separate_segments:
                produced.extend(
                    _split_segments(labels, model, base, extension, scan_output_dir, prediction_ID)
                )
            entry.update(status="ok", segmentations=produced)
        except Exception as exc:  # noqa: BLE001 - one bad scan must not end the batch
            logger.exception("BatchDentalSeg: scan failed")
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")

        report_scans.append(entry)

    succeeded = [entry for entry in report_scans if entry.get("status") == "ok"]
    report = {
        "tool": "BatchDentalSeg",
        "model": model.name,
        "model_description": model.description,
        # Published with the results: the segmentation is a label volume, and
        # without this table its integers mean nothing to whoever opens it.
        "labels": model.labels,
        "device": device,
        "prediction_ID": prediction_ID,
        "separate_segments": separate_segments,
        "tile_step_size": settings.BATCHDENTALSEG_TILE_STEP_SIZE,
        "scans": report_scans,
        "summary": f"{len(succeeded)}/{len(report_scans)} scan(s) segmented",
        "duration_seconds": round(time.monotonic() - started, 2),
    }

    report_path = os.path.join(output_dir, "BatchDentalSeg_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return SegmentationRun(output_dir, report, report_scans)


def main(
    input: str,
    model: str,
    separate_segments: bool = False,
    prediction_ID: str = "Seg",
) -> str:
    """The schema adapter: returns the output directory, which main.py zips."""
    run = segment(
        input_path=str(input),
        model_path=str(model),
        separate_segments=separate_segments,
        prediction_ID=prediction_ID,
    )
    return run.output_dir

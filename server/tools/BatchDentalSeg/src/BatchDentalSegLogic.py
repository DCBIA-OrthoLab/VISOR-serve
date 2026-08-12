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

from . import catalogs, nnunet_runner, surface_export

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

    @property
    def surface_files(self) -> list:
        return [path for scan in self.scans for path in scan.get("surfaces", [])]


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

def _requested_formats(export_formats) -> list:
    """The mesh formats a caller asked for, in the schema's own order.

    Accepts what `validate()` produces (a `Selection`, i.e. every option mapped
    to a boolean) as well as a plain list or a single name, so a server-side
    caller of `segment()` is not forced to build a Selection by hand.
    """
    if not export_formats:
        return []
    if isinstance(export_formats, dict):
        chosen = {name for name, enabled in export_formats.items() if enabled}
    elif isinstance(export_formats, str):
        chosen = {export_formats}
    else:
        chosen = set(export_formats)
    return [name for name in surface_export.ALL_FORMATS if name in chosen]


def _emit_failures(emit, failed_conversions: list, total: int) -> None:
    """Announce the scans that never made it as far as the GPU.

    They are already in the report, but a client watching a queue would show
    them as pending forever otherwise -- the run would end with rows that never
    moved and no explanation on screen.
    """
    for entry in failed_conversions:
        emit({
            "event": "item",
            "name": entry["input"],
            "status": "failed",
            "error": entry.get("error"),
            "total": total,
        })


def _scan_artifact(entry: dict, scratch_dir: str) -> str:
    """Bundle ONE scan's outputs so they can be shipped on their own.

    A zip rather than one event per file: a run with `separate_segments` and
    three mesh formats produces dozens of files per patient, and each would
    otherwise be its own stored result and its own round trip. The archive is
    flat and the event carries `relative_dir`, so the client rebuilds the same
    tree without a path from this server ever travelling.
    """
    produced = list(entry.get("segmentations") or []) + list(entry.get("surfaces") or [])
    if not produced:
        return ""
    staging = os.path.join(scratch_dir, "artifacts")
    os.makedirs(staging, exist_ok=True)
    return file_utils.make_zip(
        produced, os.path.join(staging, f"{entry['case_id']}.zip")
    )


def _stream_batch(emit, runner, model_folder, nnunet_input, nnunet_output, device,
                  cases, finish, collected, scratch_dir) -> None:
    """Run the batch one scan at a time, reporting and shipping as it goes.

    nnUNet's own loop drives this through `on_case_start`/`on_case_done`, so
    the checkpoint is still loaded once -- what is given up is its cross-scan
    pipelining, which is the price of knowing where the run is.
    """
    case_ids = list(cases)

    def started(case_id, index, total):
        emit({
            "event": "item",
            "index": index + 1,
            "total": total,
            "name": _basename_of(cases[case_id]),
            "status": "running",
        })

    def done(case_id, index, total):
        entry = finish(case_id, index)
        collected.append(entry)
        emit({
            "event": "item",
            "index": index + 1,
            "total": total,
            "name": entry["input"],
            "status": entry["status"],
            "error": entry.get("error"),
        })
        if entry.get("status") != "ok":
            return
        # **This is the point of the whole exercise**: the patient's files
        # leave the server now, so a failure on scan 27 cannot cost the
        # twenty-six that already worked.
        archive = _scan_artifact(entry, scratch_dir)
        if archive:
            emit({
                "event": "artifact",
                "name": entry["input"],
                "relative_dir": entry.get("relative_dir") or ".",
                "path": archive,
            })

    runner.predict_folder(
        model_folder, nnunet_input, nnunet_output, device,
        on_case_start=started, on_case_done=done,
    )

    # A runner that ignored the callbacks would otherwise leave the report
    # missing exactly the scans it ran. Finish whatever the loop did not.
    reported = {entry.get("case_id") for entry in collected}
    for index, case_id in enumerate(case_ids):
        if case_id not in reported:
            collected.append(finish(case_id, index))


def _basename_of(path: str) -> str:
    return os.path.basename(path)


def segment(
    input_path: str,
    model_path: str,
    separate_segments: bool = False,
    prediction_ID: str = "Seg",
    export_formats=None,
    surface_smoothing: int = 5,
    surface_decimation: int = 90,
    device: str = None,
    scratch_dir: str = None,
    emit=None,
) -> SegmentationRun:
    """Segment every scan under `input_path` with one hosted model bundle.

    `model_path` is the resolved path of the bundle the caller picked; its
    directory name is which of catalogs.MODELS it is.

    `emit`, when given, turns the run into a reported one: each scan is
    announced as it starts and again as it finishes, and its files are handed
    over the moment they exist so a later failure cannot cost them. Passing it
    changes the nnUNet call granularity (see nnunet_runner.predict_folder) and
    nothing else -- the outputs, the tree and the report are identical.
    """
    started = time.monotonic()
    scratch_dir = scratch_dir or file_utils.make_scratch_dir("batchdentalseg_")

    # Before any scan is read: a missing dependency and an unusable bundle are
    # both properties of the server, and discovering either inside the loop
    # would report them once per patient.
    nnunet_runner.check_dependencies()
    model, model_folder = resolve_model(model_path)
    device = nnunet_runner.resolve_device(device)

    # Same rule, one step further: a deployment without VTK cannot make meshes,
    # and finding that out after an hour of inference would waste the whole run.
    formats = _requested_formats(export_formats)
    if formats and not surface_export.is_available():
        raise ToolArgumentError(
            "Mesh export was requested but VTK is not installed on this server. "
            "Install requirements.txt, or run with no export format selected."
        )

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

    # Scratch for the mesh writer's intermediate .nrrd files, inside the
    # request's own directory so cleanup takes them.
    surface_temp = os.path.join(scratch_dir, "surface_tmp")
    if formats:
        os.makedirs(surface_temp, exist_ok=True)

    report_scans = list(failed_conversions)
    case_ids = list(cases)
    # Reported to a streaming caller as the denominator, so "3/40" counts the
    # scans that are actually going to run -- a file that could not even be
    # read is already reported as failed above and will never produce an event.
    total = len(case_ids)

    def finish(case_id: str, index: int) -> dict:
        """Post-process one predicted scan: write it, mesh it, report it.

        Called from inside nnUNet's loop on a streamed run (so each patient is
        written and shipped while the next one is still on the GPU) and after
        the whole batch otherwise. One implementation either way, because two
        would drift on exactly the details that decide what a clinician gets.
        """
        scan = cases[case_id]
        entry = {"case_id": case_id, "input": _describe(scan), "index": index + 1, "total": total}
        predicted = os.path.join(nnunet_output, f"{case_id}.nii.gz")
        if not os.path.isfile(predicted):
            # Reported per scan rather than raised: one unreadable patient in a
            # cohort of forty must not lose the other thirty-nine.
            entry.update(status="failed", error="nnUNet produced no output for this scan")
            return entry

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
            entry.update(status="ok", segmentations=produced, relative_dir=relative)

            if formats:
                # After the segmentation is on disk, never before: the label
                # volume is what every downstream tool consumes, and a mesh
                # that fails must not take it with it.
                entry["surfaces"] = surface_export.write_surfaces(
                    labels=labels,
                    model=model,
                    formats=formats,
                    base=base,
                    output_dir=scan_output_dir,
                    suffix=prediction_ID,
                    temp_dir=surface_temp,
                    smoothing=surface_smoothing,
                    decimation=surface_decimation,
                )
        except Exception as exc:  # noqa: BLE001 - one bad scan must not end the batch
            logger.exception("BatchDentalSeg: scan failed")
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")

        return entry

    if emit is None:
        # The blocking contract, unchanged: one nnUNet call for the whole
        # batch, so it can overlap the preprocessing of one scan with the
        # inference of the previous one.
        nnunet_runner.predict_folder(model_folder, nnunet_input, nnunet_output, device)
        for index, case_id in enumerate(case_ids):
            report_scans.append(finish(case_id, index))
    else:
        _emit_failures(emit, failed_conversions, total)
        _stream_batch(
            emit=emit,
            runner=nnunet_runner,
            model_folder=model_folder,
            nnunet_input=nnunet_input,
            nnunet_output=nnunet_output,
            device=device,
            cases=cases,
            finish=finish,
            collected=report_scans,
            scratch_dir=scratch_dir,
        )

    succeeded = [entry for entry in report_scans if entry.get("status") == "ok"]
    report = {
        "tool": "BatchDentalSeg",
        "model": model.name,
        "model_description": model.description,
        # Published with the results: the segmentation is a label volume, and
        # without this table its integers mean nothing to whoever opens it.
        "labels": model.labels,
        # And what colour each structure is. A surface export bakes the colour
        # into the .vtk it writes, so a client naming its own would disagree
        # with the mesh from the same run; one table serves both.
        "label_colors": model.label_colors,
        "device": device,
        "prediction_ID": prediction_ID,
        "separate_segments": separate_segments,
        "export_formats": formats,
        "surface_smoothing": int(surface_smoothing) if formats else None,
        "surface_decimation": int(surface_decimation) if formats else None,
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
    export_formats=None,
    surface_smoothing: int = 5,
    surface_decimation: int = 90,
    emit=None,
) -> str:
    """The schema adapter: returns the output directory, which main.py zips.

    The directory is returned in every case, streamed or not: a streamed run
    has already delivered each scan, but the report is written at the end and
    the archive is what a client that ignored the events still gets.
    """
    run = segment(
        input_path=str(input),
        model_path=str(model),
        separate_segments=separate_segments,
        prediction_ID=prediction_ID,
        export_formats=export_formats,
        surface_smoothing=surface_smoothing,
        surface_decimation=surface_decimation,
        emit=emit,
    )
    return run.output_dir

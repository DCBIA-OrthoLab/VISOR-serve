"""AMASSS -- Automatic Multi-Anatomical Skull Structure Segmentation.

Ported from AMASSS_CLI.py. The business logic (nnUNet per structure, binary
masks, optional multi-label merge, optional surfaces) is preserved; the CLI
envelope is not -- `sys.argv` parsing, `sys.exit()`, the `<filter-progress>`
protocol and the caller-supplied temp/output folders are gone. Scratch space
and cleanup belong to main.py, and a failure must be an exception, since a
`SystemExit` raised in a worker thread does not surface as a clean 500.

Two entry points, on purpose:

* `segment(...)` -> `SegmentationRun`, the real API. Returns the output
  directory plus a structured report, with no zip round trip. This is what
  other server-side tools call.
* `main(...)` -> path to the output directory, the schema adapter AMASSS.py
  uses.

See the comments marked "FIX:" for the original CLI's defects corrected here.
"""

import glob
import json
import logging
import os
import re
import shutil
import sys
import time
import zipfile

import numpy as np
import SimpleITK as sitk

import file_utils
from base import ToolArgumentError

from . import nnunet_runner, vtk_export

logger = logging.getLogger("AMASSS")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")
    )
    logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# Structure catalog -- the single source of truth, published to clients
# ---------------------------------------------------------------------------
# The server owns both the grouping and the human-readable names, so the
# Slicer module (and any other client) renders its checkboxes straight
# from GET /tools instead of keeping its own copy in sync. This is exported
# through ArgSpec.choices in AMASSS.py (see STRUCTURE_CHOICES below).
#
# FIX: "Teeth" (TEETH), "Root canal" (RC) and "Mandibular canal" (MCAN) are
# deliberately ABSENT. They were offered by the Slicer UI while sitting in
# UNAVAILABLE_MODELS, had no entry in the LARGE label table, and produced
# either a KeyError during surface export or a silent collision onto label 1
# (= mandible) during merge. Offering a structure with no model is worse than
# not offering it; they come back here the day a model ships, in one place.
STRUCTURE_GROUPS = {
    "Bones": {
        "Mandible": "MAND",
        "Maxilla": "MAX",
        "Cranial base": "CB",
        "Cervical vertebra": "CV",
    },
    "Soft tissue": {
        "Upper airway": "UAW",
        "Skin": "SKIN",
    },
    "Masks": {
        "Cranial base (Mask)": "CBMASK",
        "Mandible (Mask)": "MANDMASK",
        "Maxilla (Mask)": "MAXMASK",
    },
}

STRUCTURE_CODES = tuple(
    code for group in STRUCTURE_GROUPS.values() for code in group.values()
)

DEFAULT_STRUCTURES = ("MAND", "MAX", "CB", "CV", "UAW")

# Label values in the merged multi-label volume. Unchanged from the original
# LABELS["LARGE"], because AREG and existing datasets depend on them.
LABELS = {
    "MAND": 1,
    "CB": 2,
    "UAW": 3,
    "MAX": 4,
    "CV": 5,
    "SKIN": 6,
    "CBMASK": 7,
    "MANDMASK": 8,
    "MAXMASK": 9,
}
NAMES_FROM_LABELS = {value: code for code, value in LABELS.items()}

# FIX: the original color table stopped at label 6, so the three mask
# structures fell back to white, and label 3 (upper airway) was pure black --
# invisible against a dark 3D view.
LABEL_COLORS = {
    1: (216, 101, 79),
    2: (128, 174, 128),
    3: (0, 151, 206),
    4: (230, 220, 70),
    5: (111, 184, 210),
    6: (172, 122, 101),
    7: (144, 190, 144),
    8: (230, 130, 110),
    9: (240, 232, 120),
}

# Order in which structures are painted into the merged volume: later entries
# overwrite earlier ones where they overlap.
# FIX: the original list contained "CAN", a code that exists nowhere else
# (the mandibular canal is "MCAN") -- so that entry could never match. Dead
# entries removed rather than left as decoration.
MERGING_ORDER = (
    "SKIN",
    "CV",
    "UAW",
    "CB",
    "MAX",
    "MAND",
    "CBMASK",
    "MANDMASK",
    "MAXMASK",
)

MERGE_MODES = ("MERGED", "SEPARATE")
DEFAULT_MERGE_MODES = ("MERGED",)

MERGE_MODE_NAMES = {
    "One merged segmentation file": "MERGED",
    "Separated segmentation files": "SEPARATE",
}


# ---------------------------------------------------------------------------
# What the schema publishes -- {option name: on by default}
# ---------------------------------------------------------------------------
# A "multichoice" argument declares its options as human-readable names, and
# those names are what the client renders as check-box labels and sends back.
# Built from the catalog above so the two can never disagree, and in group
# order (Bones, then Soft tissue, then Masks) since a Selection preserves
# declaration order and the client renders them in it.
#
# NOTE: the client renders one flat list -- the "choice" / "multichoice" schema
# has no notion of option groups, so the former Bones / Soft tissue / Masks
# tabs are only preserved as ordering here. Restoring real group boxes would
# need a grouping key in the schema plus formgen support; it is deliberately
# NOT faked with prefixed labels, which would end up inside the option names
# the server validates against.

STRUCTURE_NAMES = {
    display_name: code
    for group in STRUCTURE_GROUPS.values()
    for display_name, code in group.items()
}

STRUCTURE_CHOICES = {
    display_name: code in DEFAULT_STRUCTURES for display_name, code in STRUCTURE_NAMES.items()
}

MERGE_CHOICES = {
    display_name: code in DEFAULT_MERGE_MODES for display_name, code in MERGE_MODE_NAMES.items()
}


def _codes_from(selection, name_to_code: dict, fallback: tuple) -> tuple:
    """Turn what `run()` received for a multichoice argument into tool codes.

    Accepts the four shapes that legitimately reach here:

    * a `base.Selection` / dict `{option name: bool}` -- the normal HTTP path;
    * `None` -- an omitted optional argument, which must fall back to the
      declared defaults (see the note in AMASSS.py about ArgSpec.default);
    * a single option name or code as a plain string -- what a "choice"-typed
      schema hands run(). Without its own branch a string fell through to the
      iterable one below and was split into a tuple of its CHARACTERS, so no
      code ever matched: `merge` shipped that way once, and every run came
      back as a report-only zip with zero segmentation files in it;
    * an iterable of codes (or display names) -- so `segment()` stays directly
      callable by another server-side tool without going through the schema.
    """
    if selection is None:
        return fallback
    if isinstance(selection, str):
        selection = (selection,)
    if isinstance(selection, dict):
        return tuple(
            name_to_code[name] for name, enabled in selection.items() if enabled and name in name_to_code
        )
    # Display names are translated, codes pass through; anything unknown is
    # kept as-is for segment() to reject loudly rather than silently drop.
    return tuple(name_to_code.get(item, item) for item in selection)


def structure_codes(selection) -> tuple:
    return _codes_from(selection, STRUCTURE_NAMES, DEFAULT_STRUCTURES)


def merge_modes(selection) -> tuple:
    return _codes_from(selection, MERGE_MODE_NAMES, DEFAULT_MERGE_MODES)

# Shared with the other tools that read volumes; see file_utils. .gipl and
# .gipl.gz are genuinely supported: every input goes through a real SimpleITK
# read/write conversion (see _convert_to_nifti) instead of being renamed.
SCAN_EXTENSIONS = file_utils.SCAN_EXTENSIONS
split_scan_extension = file_utils.split_scan_extension

# A segmentation output keeps its input's FORMAT but is always written
# compressed. The masks are label volumes -- long runs of a single value -- so
# gzip takes them roughly 100x down: uncompressed, a 0.33mm CBCT gives a 191 MB
# file PER structure, and a nine-structure run wrote 1.75 GB against 14 MB
# compressed. The scan the masks came from is untouched.
compressed_extension = file_utils.compressed_extension


class SegmentationRun:
    """Result of `segment()`: where the files are, and what actually happened.

    Returned instead of a bare path so a calling module gets the per-scan
    outputs directly (no zip to unpack) and can tell a partial run from a
    complete one -- which the original CLI made impossible, since a structure
    whose model was missing was skipped with nothing but a log line.
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

def is_previous_output(filename: str, prediction_id: str) -> bool:
    """True if this file looks like a segmentation AMASSS itself produced.

    Without this, running twice on the same folder feeds the first run's
    outputs back in as input scans. Outputs are recognized by the suffixes
    AMASSS actually writes.
    """
    base, _extension = split_scan_extension(os.path.basename(filename))

    if prediction_id and f"_{prediction_id}_" in f"{base}_":
        return True
    if base.endswith("_MERGED"):
        return True
    for code in STRUCTURE_CODES:
        if base.endswith(f"_{code}"):
            return True
    return bool(re.search(r"_Seg(Out)?$", base))


def discover_scans(input_path: str, prediction_id: str, scratch_dir: str) -> list:
    """Resolve the `input` argument into a list of scan files.

    Accepts the three shapes a single schema argument can carry: one scan
    file, a zip archive of a folder of scans, or a folder served straight
    from the read-only data store.

    FIX: folder scanning is RECURSIVE. The Slicer UI counted scans
    recursively while the CLI listed only the top level, so on any nested
    dataset the UI announced N scans and the CLI silently processed a subset.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if os.path.isfile(input_path) and input_path.lower().endswith(".zip"):
        if not zipfile.is_zipfile(input_path):
            raise ValueError(f"Input has a .zip extension but is not a zip archive: {input_path}")
        # A batch arrives here as an archive rather than pre-extracted by
        # main.py (the schema declares "volume_or_zip_file" first, see
        # AMASSS.py), so the zip-bomb cap main.py would have applied is applied
        # here instead -- this is untrusted client input either way.
        from config import settings

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
        for name in files:
            if not name.lower().endswith(SCAN_EXTENSIONS):
                continue
            if is_previous_output(name, prediction_id):
                logger.info("Skipping %s: looks like a previous AMASSS output", name)
                continue
            scans.append(os.path.join(root, name))

    if not scans:
        # FIX: the original called sys.exit(1) here. Inside a server worker
        # thread that does not produce a clean error response.
        raise FileNotFoundError(
            f"No scan found in {input_path}. Supported extensions: "
            f"{', '.join(SCAN_EXTENSIONS)}"
        )
    return sorted(scans)


def resolve_models(model_path: str, structures, scratch_dir: str):
    """Map each requested structure to its nnUNet model folder.

    Returns (available, missing). A model bundle is normally a folder served
    from the data store; a zip archive is extracted first, mirroring how
    SurgMovPred handles its own model argument.
    """
    if os.path.isfile(model_path) and model_path.lower().endswith(".zip"):
        model_path = file_utils.extract_zip(
            model_path, os.path.join(scratch_dir, "model_extracted")
        )

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model bundle is not a directory: {model_path}")

    # A bundle may be wrapped in a single top-level folder (a zip of
    # "AMASSS_Models/" rather than of its contents). Descend into it so both
    # layouts work.
    if not any(os.path.isdir(os.path.join(model_path, code)) for code in STRUCTURE_CODES):
        entries = [
            os.path.join(model_path, name)
            for name in sorted(os.listdir(model_path))
            if os.path.isdir(os.path.join(model_path, name)) and name != "__MACOSX"
        ]
        if len(entries) == 1:
            model_path = entries[0]

    available, missing = {}, []
    for code in structures:
        model_folder = nnunet_runner.find_model_folder(model_path, code)
        if model_folder is None:
            logger.warning("No usable model for structure '%s' in %s", code, model_path)
            missing.append(code)
        else:
            available[code] = model_folder

    if not available:
        raise nnunet_runner.ModelNotFoundError(
            f"No nnUNet model found in '{os.path.basename(model_path)}' for any of the "
            f"requested structures ({', '.join(structures)}). Expected "
            f"<bundle>/<CODE>/**/*__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
        )
    return available, missing


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _convert_to_nifti(scan_path: str, destination: str) -> None:
    """Real format conversion into the NIfTI file nnUNet will read.

    FIX: the original did `shutil.copy(volume_file, "p_XXX_0000.nii.gz")` --
    an NRRD renamed to .nii.gz, handed to a reader that picks its format from
    the extension. NRRD is Slicer's own default format, so "supported" input
    was in practice only reliable for NIfTI. A read + write actually converts.

    The voxel type is left alone. Casting to float32 first, as this did, was
    not what made the conversion real -- the read and write are -- and on a CBCT
    it doubled the bytes to gzip on the way out and to gunzip on the way back
    in, for 2.4s + 0.4s per scan buying nothing: nnUNet's reader casts to
    float32 itself, and int16 CBCT values are exact in float32 either way.
    """
    image = sitk.ReadImage(scan_path)
    sitk.WriteImage(image, destination)


def _match_reference_geometry(mask_image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Put a predicted mask back onto the original scan's exact grid."""
    same_geometry = (
        mask_image.GetSize() == reference.GetSize()
        and np.allclose(mask_image.GetSpacing(), reference.GetSpacing())
        and np.allclose(mask_image.GetOrigin(), reference.GetOrigin())
        and np.allclose(mask_image.GetDirection(), reference.GetDirection())
    )
    if not same_geometry:
        mask_image = sitk.Resample(
            mask_image, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0,
            mask_image.GetPixelID(),
        )
    return sitk.Cast(mask_image, sitk.sitkInt16)


def _write_segmentation(array: np.ndarray, reference: sitk.Image, output_path: str) -> str:
    image = sitk.GetImageFromArray(array.astype(np.int16))
    image.CopyInformation(reference)
    # useCompression covers the formats whose extension does not already imply
    # it (.nrrd); for a .nii.gz the gz is applied from the name either way.
    # Harmless where compression is already on, so it is not conditional.
    sitk.WriteImage(
        _match_reference_geometry(image, reference), output_path, useCompression=True
    )
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def segment(
    input_path: str,
    model_path: str,
    structures=None,
    merge=None,
    prediction_ID: str = "Pred",
    generate_surface: bool = False,
    surface_smoothing: int = 5,
    surface_decimation: int = 90,
    device: str = None,
    scratch_dir: str = None,
) -> SegmentationRun:
    """Segment one scan or a batch, and return where the results are.

    This is the reusable API: another server-side tool imports this function
    and gets the produced files directly. `main()` below is only the HTTP
    adapter that packs the same output into a zip.
    """
    from config import settings

    started_at = time.monotonic()

    # None means "not specified" and takes the defaults; an EMPTY selection is
    # a different thing entirely -- the caller unchecked every box -- and must
    # be reported rather than silently turned back into the defaults.
    structures = tuple(DEFAULT_STRUCTURES if structures is None else structures)
    merge = tuple(DEFAULT_MERGE_MODES if merge is None else merge)
    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"

    unknown = [code for code in structures if code not in STRUCTURE_CODES]
    if unknown:
        raise ValueError(
            f"Unknown structure code(s): {', '.join(unknown)}. "
            f"Known: {', '.join(STRUCTURE_CODES)}"
        )

    # Constraints the schema cannot express: every box unchecked is a perfectly
    # valid Selection. ToolArgumentError is what makes main.py answer 422 with
    # these messages rather than a blank 500.
    if not structures:
        raise ToolArgumentError("Select at least one structure to segment.")
    if not merge:
        raise ToolArgumentError(
            "Select at least one output form (merged and/or separated segmentations)."
        )
    # Same treatment as unknown structure codes above: an unrecognised merge
    # mode must fail HERE, before minutes of inference. Left unvalidated, it
    # matches neither branch of _assemble_scan_outputs and the run ends "ok"
    # with a report and no segmentation files at all.
    unknown_merge = [code for code in merge if code not in MERGE_MODES]
    if unknown_merge:
        raise ValueError(
            f"Unknown merge mode(s): {', '.join(unknown_merge)}. "
            f"Known: {', '.join(MERGE_MODES)}"
        )

    if scratch_dir is None:
        scratch_dir = file_utils.make_scratch_dir("AMASSS_")
    os.makedirs(scratch_dir, exist_ok=True)

    if generate_surface and not vtk_export.is_available():
        raise RuntimeError(
            "generate_surface=true but VTK is not installed on the server. "
            "Install requirements.txt, or run with generate_surface=false."
        )

    device = nnunet_runner.resolve_device(device or settings.DEVICE)

    scans = discover_scans(input_path, prediction_ID, scratch_dir)
    models, missing_structures = resolve_models(model_path, structures, scratch_dir)
    logger.info(
        "AMASSS: %d scan(s), %d structure(s) available, device=%s",
        len(scans), len(models), device,
    )

    # Named after the run, not "output": with output_kind = "files", main.py
    # names the archive it streams back after this directory, so the client
    # receives AMASSS_Pred.zip rather than output.zip.
    output_dir = os.path.join(scratch_dir, f"AMASSS_{prediction_ID}")
    os.makedirs(output_dir, exist_ok=True)

    # Convert every scan once into the single folder nnUNet reads. Predicting
    # per structure over the whole folder loads each checkpoint once, instead
    # of once per (scan x structure) as the original did.
    nnunet_input = os.path.join(scratch_dir, "nnunet_input")
    os.makedirs(nnunet_input, exist_ok=True)

    scan_records = []
    for index, scan_path in enumerate(scans):
        case_id = f"p_{index:03d}"
        record = {
            "case_id": case_id,
            "input": os.path.basename(scan_path),
            "input_path": scan_path,
            "status": "pending",
            "predicted_structures": [],
            "segmentations": [],
            "surfaces": [],
        }
        try:
            _convert_to_nifti(scan_path, os.path.join(nnunet_input, f"{case_id}_0000.nii.gz"))
        except Exception as exc:
            logger.exception("Could not read scan %s", scan_path)
            record["status"] = "failed"
            record["error"] = f"Unreadable input: {exc}"
        scan_records.append(record)

    readable = [record for record in scan_records if record["status"] != "failed"]
    if not readable:
        raise ValueError("None of the input scans could be read as a medical volume.")

    # --- Inference: one model load per structure --------------------------
    predictions = {}
    failed_structures = {}
    for code, model_folder in models.items():
        structure_output = os.path.join(scratch_dir, f"pred_{code}")
        try:
            logger.info("Predicting %s on %s", code, device)
            nnunet_runner.predict_folder(model_folder, nnunet_input, structure_output, device)
            predictions[code] = structure_output
        except Exception as exc:
            # One structure failing must not lose the others.
            logger.exception("Prediction failed for structure %s", code)
            failed_structures[code] = str(exc)

    if not predictions:
        raise RuntimeError(
            "Every structure failed to predict: "
            + "; ".join(f"{code}: {error}" for code, error in failed_structures.items())
        )

    # --- Assemble per-scan outputs ----------------------------------------
    for record in readable:
        scan_started = time.monotonic()
        try:
            _assemble_scan_outputs(
                record=record,
                predictions=predictions,
                output_dir=output_dir,
                scratch_dir=scratch_dir,
                prediction_ID=prediction_ID,
                merge=merge,
                generate_surface=generate_surface,
                surface_smoothing=surface_smoothing,
                surface_decimation=surface_decimation,
            )
            record["status"] = "ok"
        except Exception as exc:
            # FIX: the original re-raised on the LAST scan only, so a batch
            # could abort at the very end and lose everything already
            # produced. Every failure is recorded; the run continues.
            logger.exception("Failed to assemble outputs for %s", record["input"])
            record["status"] = "failed"
            record["error"] = str(exc)
        record["duration_seconds"] = round(time.monotonic() - scan_started, 2)
        record.pop("input_path", None)

    processed = [record for record in scan_records if record["status"] == "ok"]
    if not processed:
        raise RuntimeError(
            "AMASSS produced no output for any scan. First error: "
            + str(next((r.get("error") for r in scan_records if r.get("error")), "unknown"))
        )

    report = {
        "tool": "AMASSS",
        "device": device,
        # Both of these move the segmentation, so a mask is only reproducible
        # alongside them (see settings.AMASSS_GPU_RESAMPLING for how much).
        # This records what was ASKED for: a model bundle whose plans pin a
        # non-default resampler opts itself out, and says so in the log.
        "gpu_resampling": bool(settings.AMASSS_GPU_RESAMPLING) and device.startswith("cuda"),
        "tile_step_size": float(settings.AMASSS_TILE_STEP_SIZE),
        "prediction_ID": prediction_ID,
        "merge_modes": list(merge),
        "generate_surface": bool(generate_surface),
        "surface_decimation": int(surface_decimation) if generate_surface else None,
        "requested_structures": list(structures),
        "predicted_structures": sorted(predictions, key=lambda c: STRUCTURE_CODES.index(c)),
        # FIX: a structure whose model was missing used to vanish with nothing
        # but a logger.warning -- invisible in a 200-scan batch. It is now
        # reported explicitly, next to the results.
        "structures_without_model": missing_structures,
        "structures_failed": failed_structures,
        "scans": scan_records,
        "summary": {
            "total": len(scan_records),
            "processed": len(processed),
            "failed": len(scan_records) - len(processed),
        },
        "duration_seconds": round(time.monotonic() - started_at, 2),
    }

    with open(os.path.join(output_dir, "AMASSS_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info(
        "AMASSS finished: %d/%d scan(s) in %.1fs",
        len(processed), len(scan_records), report["duration_seconds"],
    )
    return SegmentationRun(output_dir=output_dir, report=report, scans=scan_records)


def _assemble_scan_outputs(record, predictions, output_dir, scratch_dir, prediction_ID,
                           merge, generate_surface, surface_smoothing,
                           surface_decimation) -> None:
    """Turn one scan's per-structure nnUNet masks into its final files."""
    scan_path = record["input_path"]
    case_id = record["case_id"]
    base, input_extension = split_scan_extension(os.path.basename(scan_path))
    # Never the input's own spelling: a scan sent as an uncompressed .nii must
    # not produce nine uncompressed masks (see compressed_extension).
    extension = compressed_extension(input_extension)

    reference = sitk.ReadImage(scan_path)

    masks = {}
    for code, structure_output in predictions.items():
        predicted_file = os.path.join(structure_output, f"{case_id}.nii.gz")
        if not os.path.isfile(predicted_file):
            logger.warning("No %s prediction for %s", code, record["input"])
            continue
        array = sitk.GetArrayFromImage(sitk.ReadImage(predicted_file))
        masks[code] = (array > 0).astype(np.uint8)

    if not masks:
        raise FileNotFoundError(f"nnUNet produced no prediction for {record['input']}")

    record["predicted_structures"] = sorted(masks, key=lambda c: STRUCTURE_CODES.index(c))

    scan_dir = os.path.join(output_dir, f"{base}_{prediction_ID}_SegOut")
    os.makedirs(scan_dir, exist_ok=True)
    surface_temp = os.path.join(scratch_dir, "surface_tmp")
    os.makedirs(surface_temp, exist_ok=True)

    # Separate files: requested, or unavoidable when there is only one
    # structure (a "merged" volume of one structure is just that structure).
    if "SEPARATE" in merge or len(masks) == 1:
        for code, mask in masks.items():
            output_path = os.path.join(scan_dir, f"{base}_{prediction_ID}_{code}{extension}")
            record["segmentations"].append(
                _write_segmentation(mask, reference, output_path)
            )
            if generate_surface:
                record["surfaces"].append(
                    vtk_export.write_separate_surface(
                        mask=mask,
                        reference=reference,
                        # FIX: the code is passed explicitly instead of being
                        # parsed back out of the file name (original KeyError).
                        structure_code=code,
                        label_colors=LABEL_COLORS,
                        labels=LABELS,
                        temp_dir=surface_temp,
                        smoothing=surface_smoothing,
                        decimation=surface_decimation,
                        output_path=os.path.join(
                            scan_dir, f"{base}_{prediction_ID}_{code}.vtk"
                        ),
                    )
                )

    if "MERGED" in merge and len(masks) > 1:
        shape = next(iter(masks.values())).shape
        merged = np.zeros(shape, dtype=np.int16)
        for code in MERGING_ORDER:
            if code in masks:
                merged = np.where(masks[code] == 1, LABELS[code], merged)

        output_path = os.path.join(scan_dir, f"{base}_{prediction_ID}_MERGED{extension}")
        record["segmentations"].append(_write_segmentation(merged, reference, output_path))
        if generate_surface:
            surface = vtk_export.write_merged_surface(
                merged=merged,
                reference=reference,
                names_from_labels=NAMES_FROM_LABELS,
                label_colors=LABEL_COLORS,
                temp_dir=surface_temp,
                smoothing=surface_smoothing,
                decimation=surface_decimation,
                output_path=os.path.join(scan_dir, f"{base}_{prediction_ID}_MERGED.vtk"),
            )
            if surface:
                record["surfaces"].append(surface)

    # Unreachable while the merge codes are validated in segment(): SEPARATE
    # writes per structure, MERGED writes above, and a lone mask falls back to
    # the separate branch. Kept anyway -- the one time this happened, the run
    # was reported "ok" and the client received a zip holding nothing but the
    # report. A scan with no output must never count as processed.
    if not record["segmentations"]:
        raise RuntimeError(
            f"No segmentation was written for {record['input']} (merge modes: {', '.join(merge)})."
        )


def main(
    input: str,
    model: str,
    structures=None,
    merge=None,
    prediction_ID: str = "Pred",
    generate_surface: bool = False,
    surface_smoothing: int = 5,
    surface_decimation: int = 90,
) -> str:
    """Schema adapter: translate the declared option names into tool codes, run
    `segment()`, and return the folder holding the results.

    `structures` and `merge` arrive as `base.Selection` mappings keyed by the
    human-readable names the schema published; `segment()` speaks structure
    codes, and keeps doing so, because that is the API another server-side tool
    calls (see the module docstring). `_codes_from` bridges the two and also
    accepts plain code sequences, so a direct caller never has to know the
    display names exist.

    Returning a DIRECTORY rather than an archive is deliberate: with
    `output_kind = "files"`, main.py bundles it and streams the zip, so no zip
    code lives in this tool. Scratch space always comes from
    `file_utils.make_scratch_dir`, which registers it for cleanup even if the
    run raises half-way -- patient data must never survive a crash.
    """
    scratch_dir = file_utils.make_scratch_dir("AMASSS_")

    run = segment(
        input_path=input,
        model_path=model,
        structures=structure_codes(structures),
        merge=merge_modes(merge),
        prediction_ID=prediction_ID,
        generate_surface=generate_surface,
        surface_smoothing=surface_smoothing,
        surface_decimation=surface_decimation,
        scratch_dir=scratch_dir,
    )

    # The intermediate nnUNet folders can be large (one predicted volume per
    # scan and per structure); free them before the response is streamed
    # rather than holding the disk until main.py's cleanup runs. They sit
    # beside output_dir, never inside it, so the archive is unaffected.
    for temporary in ("nnunet_input", "surface_tmp", "input_extracted", "model_extracted"):
        shutil.rmtree(os.path.join(scratch_dir, temporary), ignore_errors=True)
    for structure_output in glob.glob(os.path.join(scratch_dir, "pred_*")):
        shutil.rmtree(structure_output, ignore_errors=True)

    return run.output_dir

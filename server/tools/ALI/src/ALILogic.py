"""ALI's dispatcher: work out what the input is, then run the right engine.

ALI is one tool with two engines that share nothing but their output format.
Which one applies is decided **here, from the data**, not from an argument the
caller sets:

* a `.zip` can hold either kind, so its extension says nothing;
* a DICOM series has no extension at all;
* and a user who has to tell the server what they just gave it can tell it
  wrong, at which point the error arrives several minutes into a GPU run
  rather than at the door.

Everything before inference lives here -- unpacking, DICOM conversion, mode
detection, crown segmentation for unlabelled meshes, and the run report --
so that `ALI_CBCT/` and `ALI_IOS/` only have to know how to place landmarks.

The one thing the schema cannot say is "this argument only applies in mode X".
Both selections are therefore optional and both are always shown by the
client; an empty selection for the mode that actually ran is a
`ToolArgumentError`, which main.py turns into a 422 naming the argument to
fill in. That 422 is how a mode mismatch explains itself.
"""

import json
import logging
import os
import shutil
import sys
import time
import zipfile

import file_utils
from base import ToolArgumentError
from config import settings

from .ALI_CBCT import landmarks as cbct_catalog
from .ALI_IOS import landmarks as ios_catalog

logger = logging.getLogger("ALI")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")
    )
    logger.addHandler(_handler)


VOLUME_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")
SURFACE_EXTENSIONS = (".vtk", ".stl")

CBCT = "CBCT"
IOS = "IOS"

REPORT_NAME = "run_report.json"


class Input:
    """What the input turned out to hold, and how to run it.

    `scans` is a list of `(absolute path, key)` pairs. The key is the path
    relative to the input root and is what identifies a scan everywhere
    afterwards -- in the report and in the output tree. Keying by BASE NAME,
    as the original did, meant two patients called `scan.nii.gz` in different
    subfolders silently overwrote each other twice over: once in the working
    dictionary, once in the flat output folder.
    """

    def __init__(self, mode: str, scans: list, converted_dicom: int = 0,
                 unsegmented: int = 0):
        self.mode = mode
        self.scans = scans
        self.converted_dicom = converted_dicom
        self.unsegmented = unsegmented


def scan_key(path: str, root: str) -> str:
    """A scan's identity: its path relative to the input root."""
    if not root:
        return os.path.basename(path)
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return os.path.basename(path)


# ---------------------------------------------------------------------------
# Discovery and mode detection
# ---------------------------------------------------------------------------

def _extracted(input_path: str, scratch_dir: str) -> str:
    """The directory to walk: the input itself, or the archive unpacked."""
    if os.path.isdir(input_path):
        return input_path
    if not input_path.lower().endswith(".zip"):
        return input_path
    if not zipfile.is_zipfile(input_path):
        raise ToolArgumentError(
            f"'{os.path.basename(input_path)}' has a .zip extension but is not a zip archive."
        )
    # A batch reaches run() as an archive rather than pre-extracted, because
    # the schema declares two FILE types and no "folder" (see ALI.py), so the
    # zip-bomb cap main.py would have applied is applied here instead. This is
    # untrusted client input either way.
    return file_utils.extract_zip(
        input_path,
        os.path.join(scratch_dir, "input_extracted"),
        strip_single_root=True,
        max_total_bytes=settings.MAX_EXTRACTED_MB * 1024 * 1024,
    )


def _dicom_directories(root: str) -> list:
    """Directories under `root` holding a DICOM series.

    Only directories with no volume or surface file of their own are probed:
    a folder that already contains NIfTI is a folder of scans, not a series,
    and asking GDCM about every directory of a large cohort is slow.
    """
    from .ALI_CBCT import preprocess

    found = []
    for directory, _subdirs, files in os.walk(root):
        if any(name.lower().endswith(VOLUME_EXTENSIONS + SURFACE_EXTENSIONS) for name in files):
            continue
        if not files:
            continue
        if preprocess.is_dicom_series(directory):
            found.append(directory)
    return sorted(found)


def detect(input_path: str, scratch_dir: str) -> Input:
    """Work out whether this input is CBCT or IOS, and list what to process.

    Raises `ToolArgumentError` -- a 422 -- when the input holds neither kind,
    or both. "Both" is refused rather than guessed at: running one engine and
    silently ignoring half the archive is the failure that looks like success.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    root = _extracted(input_path, scratch_dir)

    if os.path.isfile(root):
        lower = root.lower()
        if lower.endswith(VOLUME_EXTENSIONS):
            return Input(CBCT, [(root, os.path.basename(root))])
        if lower.endswith(SURFACE_EXTENSIONS):
            return Input(IOS, [(root, os.path.basename(root))])
        raise ToolArgumentError(
            f"'{os.path.basename(root)}' is neither a CBCT volume "
            f"({', '.join(VOLUME_EXTENSIONS)}) nor an intraoral surface "
            f"({', '.join(SURFACE_EXTENSIONS)})."
        )

    volumes, surfaces = [], []
    for directory, _subdirs, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(directory, name)
            if name.lower().endswith(VOLUME_EXTENSIONS):
                volumes.append(path)
            elif name.lower().endswith(SURFACE_EXTENSIONS):
                surfaces.append(path)

    dicom_dirs = _dicom_directories(root) if not surfaces else []

    if (volumes or dicom_dirs) and surfaces:
        raise ToolArgumentError(
            f"This input mixes {len(volumes) + len(dicom_dirs)} CBCT scan(s) and "
            f"{len(surfaces)} intraoral surface(s). ALI runs one engine per request; "
            f"send them as two batches."
        )

    if surfaces:
        return Input(IOS, [(path, scan_key(path, root)) for path in sorted(surfaces)])

    if volumes or dicom_dirs:
        scans = [(path, scan_key(path, root)) for path in sorted(volumes)]
        converted = _convert_dicom(dicom_dirs, root, scratch_dir)
        scans.extend(converted)
        return Input(CBCT, sorted(scans, key=lambda item: item[1]), converted_dicom=len(converted))

    raise ToolArgumentError(
        f"No CBCT scan ({', '.join(VOLUME_EXTENSIONS)}), DICOM series, or intraoral surface "
        f"({', '.join(SURFACE_EXTENSIONS)}) found in the input."
    )


def _convert_dicom(directories: list, root: str, scratch_dir: str) -> list:
    """Convert each DICOM series to NIfTI; return (path, key) pairs.

    Written into the scratch directory, never into the input. The original
    created `<input>/NIFTI/` inside the folder the user had selected -- so it
    modified their data, and a second run re-discovered its own output as
    input scans.
    """
    if not directories:
        return []

    from .ALI_CBCT import preprocess

    destination_root = os.path.join(scratch_dir, "dicom_converted")
    converted = []
    for directory in directories:
        key = scan_key(directory, root)
        destination = os.path.join(destination_root, f"{key.replace(os.sep, '_')}.nii.gz")
        try:
            preprocess.convert_dicom_series(directory, destination)
        except Exception:
            # A folder GDCM claimed as a series but could not read. Skipped
            # with a log line naming nothing: the folder name is the patient.
            logger.exception("Could not convert a DICOM series")
            continue
        converted.append((destination, f"{key}.nii.gz"))
    logger.info("ALI: converted %d DICOM series", len(converted))
    return converted


# ---------------------------------------------------------------------------
# Crown segmentation for IOS input
# ---------------------------------------------------------------------------

def ensure_segmented(scans: list, scratch_dir: str) -> tuple:
    """Make sure every mesh carries tooth labels; segment the ones that do not.

    Returns (scans, number segmented). This is the seam that makes ALI a
    single button: a raw intraoral scan is segmented server-side through
    CrownSeg and then goes straight into landmark identification, with no
    second request and no Slicer-side conda environment.

    CrownSeg is imported HERE rather than at module level, and by tool folder
    rather than by package: it is a sibling tool, it may itself have failed to
    load, and ALI's CBCT engine must keep working when it has.
    """
    from tools.CrownSeg.src import CrownSegLogic

    unsegmented = [(path, key) for path, key in scans if not CrownSegLogic.is_segmented(path)]
    if not unsegmented:
        return scans, 0

    logger.info("ALI: %d mesh(es) carry no tooth labels; segmenting", len(unsegmented))

    segmented_root = os.path.join(scratch_dir, "crownseg")
    replacements = {}
    for path, key in unsegmented:
        run = CrownSegLogic.segment_crowns(
            input_path=path,
            scratch_dir=scratch_dir,
            output_dir=os.path.join(segmented_root, os.path.dirname(key)),
            skip_segmented=False,
        )
        if not run.meshes:
            raise RuntimeError(f"Crown segmentation produced no labelled mesh for '{key}'")
        # The key stays the ORIGINAL one, so the run report and the output tree
        # name the file the caller sent, not CrownSeg's intermediate copy.
        replacements[key] = run.meshes[0]

    return [(replacements.get(key, path), key) for path, key in scans], len(unsegmented)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def identify(
    input_path: str,
    model_path: str,
    cbct_regions=None,
    ios_networks=None,
    prediction_ID: str = "Pred",
    scratch_dir: str = None,
    device: str = None,
) -> dict:
    """Place landmarks on whatever this input holds; return the run report.

    The reusable API, in the shape `AMASSSLogic.segment()` established: it
    returns the report, and the produced files are named in it. `main()` below
    is the thin adapter the schema calls.
    """
    started_at = time.monotonic()

    if scratch_dir is None:
        scratch_dir = file_utils.make_scratch_dir("ALI_")
    os.makedirs(scratch_dir, exist_ok=True)

    prediction_ID = (prediction_ID or "Pred").strip() or "Pred"

    # Unpacking a cohort and converting DICOM are minutes of work on a large
    # archive, and used to happen in complete silence -- so a run looked hung
    # before it had even started. Counts only, never a file name.
    logger.info("ALI: inspecting the input")
    detected = detect(input_path, scratch_dir)
    logger.info(
        "ALI: %s input, %d scan(s)%s",
        detected.mode,
        len(detected.scans),
        f", {detected.converted_dicom} converted from DICOM" if detected.converted_dicom else "",
    )

    output_dir = os.path.join(scratch_dir, f"ALI_{prediction_ID}")
    os.makedirs(output_dir, exist_ok=True)

    if detected.mode == CBCT:
        regions = cbct_catalog.region_codes(cbct_regions)
        if not regions:
            # The cross-argument rule the schema cannot express. Naming the
            # detected mode is the point: it is what tells a caller who ticked
            # the other group's boxes what actually happened.
            raise ToolArgumentError(
                f"This input is a CBCT batch: select at least one region under 'cbct_regions' "
                f"({', '.join(cbct_catalog.REGION_NAMES)})."
            )
        from .ALI_CBCT import engine as cbct_engine

        report = cbct_engine.predict_landmarks(
            scans=detected.scans,
            model_path=model_path,
            regions=regions,
            prediction_ID=prediction_ID,
            output_dir=output_dir,
            scratch_dir=scratch_dir,
            device=device,
        )
        report["dicom_series_converted"] = detected.converted_dicom
    else:
        networks = ios_catalog.network_codes(ios_networks)
        if not networks:
            raise ToolArgumentError(
                f"This input is a batch of intraoral surfaces: select at least one landmark "
                f"family under 'ios_networks' ({', '.join(ios_catalog.NETWORK_NAMES)})."
            )
        scans, segmented = ensure_segmented(detected.scans, scratch_dir)

        from .ALI_IOS import engine as ios_engine

        report = ios_engine.predict_landmarks(
            meshes=scans,
            model_path=model_path,
            networks=networks,
            prediction_ID=prediction_ID,
            output_dir=output_dir,
            scratch_dir=scratch_dir,
            device=device,
        )
        report["meshes_segmented"] = segmented

    report["tool"] = "ALI"
    report["output_dir"] = output_dir
    report["duration_seconds"] = round(time.monotonic() - started_at, 2)

    # Named `run_report.json` because that is what the Slicer module reads to
    # tell "the model bundle has no such landmark" from "the agent did not
    # converge on this scan" -- two failures that look identical in the scene
    # and need opposite fixes.
    with open(os.path.join(output_dir, REPORT_NAME), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    logger.info(
        "ALI finished in %s mode: %d/%d scan(s) in %.1fs",
        report["mode"],
        report["summary"]["processed"],
        report["summary"]["total"],
        report["duration_seconds"],
    )
    return report


def main(
    input: str,
    model: str,
    cbct_regions=None,
    ios_networks=None,
    prediction_ID: str = "Pred",
) -> str:
    """Schema adapter: run `identify` and return the folder of results.

    Returns a DIRECTORY: with `output_kind = "files"`, main.py bundles it and
    streams the archive, so no zip code lives in this tool.
    """
    scratch_dir = file_utils.make_scratch_dir("ALI_")

    report = identify(
        input_path=input,
        model_path=model,
        cbct_regions=cbct_regions,
        ios_networks=ios_networks,
        prediction_ID=prediction_ID,
        scratch_dir=scratch_dir,
    )

    # The intermediates are large -- an extracted cohort, its converted DICOM,
    # its preprocessed volumes at two spacings. Freed before the response is
    # streamed rather than held until main.py's cleanup runs. All of them sit
    # beside output_dir, never inside it, so the archive is unaffected.
    for intermediate in ("input_extracted", "dicom_converted", "preprocessed", "crownseg"):
        shutil.rmtree(os.path.join(scratch_dir, intermediate), ignore_errors=True)

    return report["output_dir"]

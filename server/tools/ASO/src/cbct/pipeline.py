"""The CBCT half of ASO: discover scans, recentre them, register on a
reference, write the oriented volume, its landmarks and its transform.

Ported from `ASO_CBCT/PRE_ASO_CBCT/PRE_ASO_CBCT.py` (the recentring) and
`ASO_CBCT/SEMI_ASO_CBCT/SEMI_ASO_CBCT.py` (the registration driver). Both
CLIs' envelopes are gone: no `<filter-progress>` prints, no `time.sleep`
progress theatre, no `sys.exit`, no writing into the caller's input tree.

The two automation modes share this whole file. Semi-automated reads the
landmarks the caller sent; fully-automated is handed predicted ones. That is
the only difference, and it is a parameter.
"""

import logging
import os

import numpy as np
import SimpleITK as sitk

from .. import markups
from . import icp

logger = logging.getLogger("ASO")

SCAN_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")

# See AMASSSLogic.compressed_extension for why this table is what it is: NIfTI
# and GIPL take an external .gz, NRRD compresses inside the file and ITK has no
# ".nrrd.gz" WRITER at all, so that spelling maps back down to ".nrrd".
_COMPRESSED_EXTENSIONS = {".nii": ".nii.gz", ".gipl": ".gipl.gz", ".nrrd.gz": ".nrrd"}

# Suffixes previous runs (of ASO, AMASSS or ALI) leave on a file name, stripped
# to recover the patient they belong to. Longest first so "_Scanreg" is not cut
# short by "_Scan".
#
# "_T1"/"_T2" are deliberately NOT here, although `GetPatients` stripped them:
# they mark two timepoints of the same subject, which are two scans to orient,
# not one. Collapsing them dropped the second scan and merged both landmark
# sets into the survivor.
_PATIENT_SUFFIXES = (
    "_lm_Pred", "_Scanreg", "_MERGED", "_scan", "_Scan", "_Or", "_OR", "_lm",
)


def split_scan_extension(filename: str) -> tuple:
    """('scan.nii.gz') -> ('scan', '.nii.gz'), compound extensions preserved."""
    lower = filename.lower()
    for extension in SCAN_EXTENSIONS:
        if lower.endswith(extension):
            return filename[: -len(extension)], filename[-len(extension):]
    return os.path.splitext(filename)


def compressed_extension(extension: str) -> str:
    return _COMPRESSED_EXTENSIONS.get(extension.lower(), extension)


def patient_stem(filename: str) -> str:
    """The patient a file belongs to, from its name alone.

    Only ever compared against other files in the SAME directory (see
    `discover`), so two patients whose names collapse to the same stem in
    different folders stay separate.
    """
    stem = filename
    for extension in SCAN_EXTENSIONS:
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
            break
    else:
        for extension in markups.MARKUPS_EXTENSIONS:
            if stem.lower().endswith(extension):
                stem = stem[: -len(extension)]
                break
        else:
            stem = os.path.splitext(stem)[0]

    for suffix in _PATIENT_SUFFIXES:
        index = stem.find(suffix)
        if index > 0:
            stem = stem[:index]
    return stem


def is_previous_output(filename: str, suffix: str) -> bool:
    """True if this file looks like something a previous ASO run wrote.

    Running twice on the same folder must not orient an already-oriented scan.
    `patient1_Or.nii.gz` sorts BEFORE `patient1_scan.nii.gz`, so without this a
    second run would silently pick the first run's output as its input -- and a
    previous run's `patient1_lm_Or.mrk.json` would overwrite the caller's own
    landmarks during the merge.
    """
    stem = filename
    for extension in SCAN_EXTENSIONS + markups.MARKUPS_EXTENSIONS + (".tfm",):
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
            break
    return stem.endswith(f"_{suffix}") or stem.endswith(f"_{suffix}_transform")


def discover(input_root: str, output_suffix: str = "Or") -> dict:
    """{patient key: {"scan": path, "markups": [paths]}} for one input tree.

    The key is the patient's path RELATIVE to the input root, not its base
    name. `GetPatients` keyed on the base name, so two scans called `scan.nii.gz`
    in different subfolders silently became one patient -- in the working dict
    and again in the flat output folder. Here they stay apart, and the output
    keeps the input's tree.

    A patient may have several markups files: fully-automated runs used to
    produce one per landmark group, which `MergeJson` merged by rewriting the
    caller's input folder and DELETING the sources. They are merged in memory
    instead (see `load_landmarks`).

    Files that look like a previous run's output are set aside and used only if
    a patient has nothing else -- so re-running on an output folder still works,
    while a folder holding both an original and its orientation uses the
    original.
    """
    patients: dict = {}
    for directory, _, file_names in os.walk(input_root):
        relative = os.path.relpath(directory, input_root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if file_name.startswith("."):
                continue
            key = os.path.join(prefix, patient_stem(file_name))
            entry = patients.setdefault(
                key, {"scans": [], "old_scans": [], "markups": [], "old_markups": []}
            )
            path = os.path.join(directory, file_name)
            previous = is_previous_output(file_name, output_suffix)
            if file_name.lower().endswith(SCAN_EXTENSIONS):
                entry["old_scans" if previous else "scans"].append(path)
            elif markups.is_markups_file(file_name):
                entry["old_markups" if previous else "markups"].append(path)

    found = {}
    for key, entry in patients.items():
        # A patient with several scans in one folder keeps the first in sorted
        # order; the rest would be re-orientations of the same subject and are
        # not silently mixed in.
        scans = entry["scans"] or entry["old_scans"]
        if not scans:
            continue
        found[key] = {
            "scan": scans[0],
            "markups": entry["markups"] or entry["old_markups"],
        }
    return found


def load_landmarks(paths: list) -> dict:
    """Merge every markups file a patient has into one landmark dict.

    Later files win on a repeated label, which only happens when a caller sends
    overlapping groups. Unreadable files are skipped with a warning naming the
    file, never the data.
    """
    merged: dict = {}
    for path in paths:
        try:
            merged.update(markups.load_landmarks(path))
        except (ValueError, OSError) as exc:
            logger.warning("Skipping '%s': %s", os.path.basename(path), exc)
    return merged


def load_reference(reference_dir: str) -> dict:
    """The reference landmark set, from the first markups file in the bundle.

    The reference SCAN is not read. The original loaded it
    (`gold_image = sitk.ReadImage(gold_file)`) and never used the result, while
    requiring one to be present -- so a reference bundle holding just the
    landmarks it actually registers against was rejected with an IndexError.
    """
    for directory, _, file_names in sorted(os.walk(reference_dir)):
        for file_name in sorted(file_names):
            if markups.is_markups_file(file_name) and not file_name.startswith("."):
                landmarks = markups.load_landmarks(os.path.join(directory, file_name))
                if landmarks:
                    return landmarks
    raise ValueError(
        "The reference bundle holds no readable landmark file (.mrk.json). A CBCT "
        "reference is one already-oriented case's landmarks."
    )


def recenter(image: sitk.Image) -> tuple:
    """Move a volume's field-of-view centre onto the physical origin.

    Returns `(recentred image, the translation that was applied)`. This is the
    whole of what `PRE_ASO_CBCT` still does -- its `model_folder`, `SmallFOV`
    and `temp_folder` arguments were read and never used once the learned
    orientation step was removed.
    """
    center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(
            (np.array(image.GetSize()) / 2.0).tolist()
        )
    )
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((-center).tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(translation.GetInverse())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetSize([int(size) for size in image.GetSize()])
    resampler.SetOutputOrigin((np.array(image.GetOrigin()) - center).tolist())
    return resampler.Execute(image), translation


def prepare(scan_path: str, destination: str = None) -> tuple:
    """Read a scan and recentre it. Returns `(centred image, translation)`.

    With `destination` set the centred volume is also written there -- which is
    what the fully-automated mode needs, since the landmark tool runs on the
    centred scans, not the originals.
    """
    centered, translation = recenter(sitk.ReadImage(scan_path))
    if destination:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        sitk.WriteImage(centered, destination, useCompression=True)
    return centered, translation


def center_landmarks(landmarks: dict, translation: sitk.Transform) -> dict:
    """Move landmarks into the centred volume's space.

    Recentring resamples the volume so a feature at physical point q ends up at
    q + offset. The landmarks have to follow, or the registration would compare
    a centred scan against uncentred points -- which is the state the original
    semi-automated CLI ran in, since it recentred nothing and then read a `.tfm`
    that only the fully-automated chain ever produced.
    """
    offset = np.array(translation.GetOffset())
    return {name: point + offset for name, point in landmarks.items()}


def orient_patient(
    centered: sitk.Image,
    pre_transform: sitk.Transform,
    source_landmarks: dict,
    reference_landmarks: dict,
    requested: list,
    output_dir: str,
    relative_key: str,
    suffix: str,
    extension: str,
    max_triplets: int,
    seed: int,
) -> dict:
    """Orient one centred CBCT and write its three outputs. Returns a report entry.

    Raises `icp.RegistrationError` when this patient cannot be registered; the
    caller records that and moves on to the next one.
    """
    registration = icp.register(
        source_landmarks,
        reference_landmarks,
        requested,
        pre_transform=pre_transform,
        max_triplets=max_triplets,
        seed=seed,
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(centered)
    resampler.SetTransform(registration.resample_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    oriented = resampler.Execute(centered)

    relative_dir, patient = os.path.split(relative_key)
    destination = os.path.join(output_dir, relative_dir)
    os.makedirs(destination, exist_ok=True)

    scan_output = os.path.join(destination, f"{patient}_{suffix}{compressed_extension(extension)}")
    sitk.WriteImage(oriented, scan_output, useCompression=True)

    landmark_output = os.path.join(destination, f"{patient}_lm_{suffix}.mrk.json")
    markups.write_landmarks(registration.landmarks, landmark_output)

    transform_output = os.path.join(destination, f"{patient}_{suffix}_transform.tfm")
    sitk.WriteTransform(registration.full_transform, transform_output)

    return {
        "status": "ok",
        "landmarks_used": registration.used,
        "landmarks_dropped": registration.dropped,
        "outputs": sorted(
            os.path.relpath(path, output_dir)
            for path in (scan_output, landmark_output, transform_output)
        ),
    }

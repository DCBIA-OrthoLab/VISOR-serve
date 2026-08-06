"""DICOM -> NIfTI conversion for CBCT input.

Ported from `ASO_CBCT_utils/utils.py::convertdicom2nifti`, with the three
things that made it unusable on a server removed:

* it wrote its output into `<input>/NIFTI/`, i.e. into the caller's own data,
  which a later run then re-discovered as input scans. Everything written here
  goes to a scratch directory instead;
* it only looked one level down, so a nested export was invisible;
* its fallback path ran `dicom2nifti.convert_directory`, then took
  `search(output_folder, "nii.gz")[0]` and RENAMED it -- with more than one
  patient already converted, that renames an arbitrary earlier file onto the
  current patient's name. The fallback writes to its own empty directory here,
  so there is nothing else to pick up.
"""

import logging
import os
import shutil
import tempfile

import SimpleITK as sitk

from ..markups import is_markups_file

logger = logging.getLogger("ASO")

_INSTALL_HINT = (
    "Reading this DICOM series needs the dicom2nifti package. Install it with "
    "`pip install -r requirements.txt` (see server/README.md)."
)


def convert_tree(input_root: str, output_root: str) -> str:
    """Convert every DICOM series under `input_root` into `output_root`.

    A directory holding a series becomes `<output_root>/<its relative path>.nii.gz`,
    so a nested export keeps its structure and two patients with the same folder
    name in different parents cannot overwrite each other. Markups files found
    anywhere in the tree are copied across at the same relative position, which
    is what lets semi-automated mode take DICOM input.

    Returns `output_root`. Raises RuntimeError when the tree holds no series.
    """
    os.makedirs(output_root, exist_ok=True)
    converted = 0

    for directory, _, file_names in os.walk(input_root):
        relative = os.path.relpath(directory, input_root)
        for file_name in file_names:
            if is_markups_file(file_name):
                destination = os.path.join(output_root, relative, file_name)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy2(os.path.join(directory, file_name), destination)

        series = _series_in(directory)
        if not series:
            continue
        name = os.path.basename(directory) if relative != "." else "scan"
        destination = os.path.join(
            output_root, os.path.dirname(relative) if relative != "." else "", f"{name}.nii.gz"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        _convert_series(directory, series, destination)
        converted += 1

    if converted == 0:
        raise RuntimeError(
            "No DICOM series found in the input. Send a zip of one folder per "
            "patient, or turn dicom_input off if the input is already NIfTI/NRRD/GIPL."
        )
    logger.info("Converted %d DICOM series", converted)
    return output_root


def _series_in(directory: str) -> tuple:
    reader = sitk.ImageSeriesReader()
    try:
        return tuple(reader.GetGDCMSeriesFileNames(directory))
    except RuntimeError:
        return ()


def _convert_series(directory: str, series: tuple, destination: str) -> None:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series)
    # A CBCT export routinely carries tags ITK complains about without the read
    # actually failing; the warnings are noise in the server log and could carry
    # patient identifiers.
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    try:
        image = reader.Execute()
    except RuntimeError:
        _convert_with_dicom2nifti(directory, destination)
        return
    finally:
        sitk.ProcessObject_SetGlobalWarningDisplay(True)
    sitk.WriteImage(image, destination, useCompression=True)


def _convert_with_dicom2nifti(directory: str, destination: str) -> None:
    """Fallback for series SimpleITK's GDCM reader refuses.

    Imported here rather than at module level: registry.py imports every tool at
    startup, and a server with no DICOM stack must still be able to serve ASO's
    other modes.
    """
    try:
        import dicom2nifti
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise RuntimeError(f"{_INSTALL_HINT} (missing: dicom2nifti)") from exc

    staging = tempfile.mkdtemp(prefix="dicom2nifti_", dir=os.path.dirname(destination))
    try:
        dicom2nifti.convert_directory(directory, staging, compression=True)
        produced = sorted(
            name for name in os.listdir(staging) if name.lower().endswith(".nii.gz")
        )
        if not produced:
            raise RuntimeError(
                f"Could not read the DICOM series in '{os.path.basename(directory)}'."
            )
        shutil.move(os.path.join(staging, produced[0]), destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

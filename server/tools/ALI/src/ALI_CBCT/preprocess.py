"""Volume preparation for the CBCT engine: histogram correction, resampling,
and DICOM series conversion.

Ported from ALI_CBCT_utils/preprocess.py. Two things changed, both because
this now runs on a server rather than on the user's own machine:

* **DICOM conversion never writes into the input.** The original created
  `<input>/NIFTI/` inside the folder the user had selected -- modifying their
  data, and leaving files that the next run re-discovered as input scans.
  Everything is written to the caller's scratch directory instead.
* **A failure is an exception.** The original logged and continued, so a run
  could end reporting success having silently skipped half the cohort.
"""

import logging
import os
import sys

import numpy as np
import SimpleITK as sitk

from base import ToolUnavailableError

logger = logging.getLogger("ALI.cbct.preprocess")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")
    )
    logger.addHandler(_handler)

_ITK_HINT = (
    "ALI's CBCT engine needs itk for resampling. Install it with "
    "`pip install -r requirements.txt` (see server/README.md)."
)


def import_itk():
    """itk is imported lazily: registry.py imports every tool at startup, and
    a missing imaging stack must not keep the whole tool out of the registry.
    """
    try:
        import itk
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_ITK_HINT} (missing: itk)") from exc
    return itk


def correct_histogram(
    input_path: str,
    output_path: str,
    min_percent: float = 0.01,
    max_percent: float = 0.99,
    intensity_min: int = -1500,
    intensity_max: int = 4000,
) -> str:
    """Clamp a scan's intensities to its own percentile range.

    The agents were trained on volumes normalized this way, so this is part of
    the model's contract, not a cosmetic step.
    """
    image = sitk.Cast(sitk.ReadImage(input_path), sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(image)

    array_min = float(np.min(array))
    array_max = float(np.max(array))
    span = array_max - array_min

    if span == 0:
        # A constant volume has no histogram to speak of. Writing it through
        # unchanged keeps the pipeline going; the agents will simply not
        # converge on it, which the run report says.
        logger.warning("Scan has a single intensity value; histogram correction is a no-op")
        clamped = array
    else:
        definition = 1000
        histogram = np.histogram(array, definition)
        cumulative = np.cumsum(histogram[0]).astype(np.float64)
        cumulative -= cumulative.min()
        cumulative /= cumulative.max()

        high = int(np.argmax(cumulative > max_percent))
        low = int(np.argmax(cumulative > min_percent))
        upper = min((high * span) / definition + array_min, intensity_max)
        lower = max((low * span) / definition + array_min, intensity_min)

        clamped = np.clip(array, lower, upper)

    output = sitk.GetImageFromArray(clamped)
    output.SetSpacing(image.GetSpacing())
    output.SetDirection(image.GetDirection())
    output.SetOrigin(image.GetOrigin())

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sitk.WriteImage(sitk.Cast(output, sitk.sitkInt16), output_path)
    return output_path


def set_spacing(input_path: str, spacing: float, output_path: str) -> str:
    """Resample a volume to an isotropic `spacing` mm grid."""
    itk = import_itk()

    image = itk.imread(input_path)
    current = np.array(image.GetSpacing())
    wanted = np.array([spacing, spacing, spacing])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if np.array_equal(current, wanted):
        itk.imwrite(image, output_path)
        return output_path

    size = itk.size(image)
    scale = current / wanted
    output_size = (np.array(size) * scale).astype(int).tolist()

    # Keep the physical centre fixed: resizing the grid around the old origin
    # would shift the volume, and the agents navigate in voxel coordinates.
    output_physical_size = np.array(output_size) * wanted
    input_physical_size = np.array(size) * current
    output_origin = np.array(image.GetOrigin()) - (output_physical_size - input_physical_size) / 2.0

    pixel_type, dimension = itk.template(image)[1]
    image_type = itk.Image[pixel_type, dimension]
    interpolator = itk.LinearInterpolateImageFunction[image_type, itk.D].New()

    resampler = itk.ResampleImageFilter[image_type, image_type].New()
    resampler.SetOutputSpacing(wanted.tolist())
    resampler.SetOutputOrigin(output_origin.tolist())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetSize(output_size)
    resampler.SetInput(image)
    resampler.Update()

    itk.imwrite(resampler.GetOutput(), output_path)
    return output_path


def is_dicom_series(directory: str) -> bool:
    """True when this directory holds a readable DICOM series.

    Asks GDCM rather than looking at extensions: DICOM slices routinely carry
    no extension at all, which is also why a folder is the only way to send
    one and why the client offers a directory picker for ALI's input.
    """
    reader = sitk.ImageSeriesReader()
    try:
        return bool(reader.GetGDCMSeriesFileNames(directory))
    except RuntimeError:
        # GDCM raises on a directory it cannot even scan; that is "no series",
        # not a server error.
        return False


def convert_dicom_series(directory: str, output_path: str) -> str:
    """Convert one DICOM series directory into a single NIfTI volume.

    `output_path` is always under the caller's scratch directory. The original
    wrote `<input>/NIFTI/*.nii.gz` straight into the user's own folder, which
    a later run then picked up again as input scans.
    """
    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(directory)
    if not file_names:
        raise ValueError(f"No DICOM series found in {directory}")

    reader.SetFileNames(file_names)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sitk.WriteImage(reader.Execute(), output_path)
    return output_path

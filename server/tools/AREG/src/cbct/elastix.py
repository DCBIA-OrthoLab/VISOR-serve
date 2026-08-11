"""The voxel-based rigid registration itself: elastix, and the conversion of
its result into a SimpleITK transform.

Ported from `AREG_CBCT/AREG_CBCT_utils/utils.py`
(`make_rigid_param_map_deterministic` / `ElastixReg` / `MaskedImage` /
`MatrixRetrieval` / `ComputeFinalMatrix`).

**The correction that matters is in `retrieve_transform`.** elastix reports a
rigid result as three Euler angles, a translation, AND a *centre of rotation*:
the transform is `y = R(x - c) + c + t`. `MatrixRetrieval` read the angles and
the translation and dropped `c`, building a SimpleITK `Euler3DTransform` whose
centre is the origin -- a different transform by exactly `(I - R)c`, i.e.
proportional to how far the fixed image sits from the physical origin.

That is not a rounding difference. Measured against a known ground-truth rigid
transform on a phantom whose origin is at (-140, -90, 60) mm (itk-elastix
0.25.4, the tuned map below):

| | max error vs. ground truth |
|---|---|
| centre dropped (what shipped) | **8.371 mm** |
| centre honoured (here) | **0.025 mm** |

It is invisible on data that happens to be centred -- which is exactly what the
oriented mode produces, since ASO recentres -- and silently catastrophic on data
that is not. When `c` is the origin the two agree to the float, so this fixes
the bad cases without touching the good ones.
"""

import logging

import numpy as np
import SimpleITK as sitk

from base import ToolUnavailableError

logger = logging.getLogger("AREG")

_INSTALL_HINT = (
    "CBCT registration needs the itk-elastix package. Install it with "
    "`pip install -r requirements.txt` and recreate the container "
    "(`docker compose up -d --force-recreate inference`)."
)


class RegistrationError(Exception):
    """One patient could not be registered. Reported, and the batch goes on."""


def check_dependencies() -> None:
    """Import the whole stack once, before any scan is read.

    A dependency belongs to the server, not to one input: discovering it inside
    the per-patient loop means a 40-patient batch fails 40 times identically,
    each time only after that patient's mask has been built, and the run ends on
    a summary that buries the one line saying what to install.
    """
    _import_elastix()


def _import_elastix():
    try:
        import itk
    except ImportError as exc:  # pragma: no cover - depends on the deployment
        raise ToolUnavailableError(f"{_INSTALL_HINT} (missing: itk)") from exc
    if not hasattr(itk, "ElastixRegistrationMethod"):  # pragma: no cover
        raise ToolUnavailableError(
            f"{_INSTALL_HINT} (the itk package is installed but carries no elastix module)"
        )
    return itk


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# The tuned rigid map, unchanged from the original except for two lines.
#
# REMOVED -- `ImagePyramidSchedule = 8,8, 4,4, 2,2`. That is six values for a
# THREE-dimensional image over THREE resolutions, where elastix wants nine. It
# does not error: a schedule whose length does not match is discarded and the
# default one is used, so the line has never had any effect. Verified by
# running the same registration three ways (2026-08-10, itk-elastix 0.25.4):
# the original six-value schedule and no schedule at all give bit-identical
# results (0.025 mm from ground truth), a corrected nine-value 8/4/2 schedule
# gives a different one (0.113 mm). Deleting the dead line therefore keeps the
# behaviour every published result was produced with; "fixing" it would have
# changed a validated pipeline for no measured gain.
#
# ADDED -- `AutomaticTransformInitializationMethod = GeometricalCenter`, which
# is already elastix's default. It is spelled out because it is what makes the
# T2 pre-centring step removable: the original resampled every T2 onto a
# centred grid before registering (an extra interpolation pass per patient,
# and a written .tfm expressed in a coordinate space the caller never
# received), and this line is the guarantee that elastix aligns the two
# volumes' centres itself.
_RIGID_PARAMETERS = {
    # Determinism. Two runs of the same request on the same data must give the
    # same transform -- this is patient data being resampled.
    "NumberOfThreads": ["1"],
    "UseDirectionCosines": ["true"],
    "ImageSampler": ["Grid"],
    "NewSamplesEveryIteration": ["false"],
    # Multi-resolution pyramid, coarse to fine.
    "NumberOfResolutions": ["3"],
    "FixedImagePyramid": ["FixedSmoothingImagePyramid"],
    "MovingImagePyramid": ["MovingSmoothingImagePyramid"],
    # Metric and interpolation.
    "Metric": ["AdvancedMattesMutualInformation"],
    "NumberOfHistogramBins": ["64"],
    "NormalizeGradient": ["true"],
    "Interpolator": ["LinearInterpolator"],
    # Optimizer.
    "Optimizer": ["ConjugateGradient"],
    "MaximumNumberOfIterations": ["1500"],
    "MaximumStepLength": ["2.0"],
    "MinimumStepLength": ["0.001"],
    "ValueTolerance": ["1e-6"],
    "GradientTolerance": ["1e-6"],
    # Initialization and scale estimation.
    "AutomaticTransformInitialization": ["true"],
    "AutomaticTransformInitializationMethod": ["GeometricalCenter"],
    "AutomaticScalesEstimation": ["true"],
    # Output. `ErodeMask` is kept for fidelity but is inert: it applies to an
    # elastix fixed MASK, and none is set -- the region is imposed by zeroing
    # the fixed image instead (see `apply_mask`).
    "ErodeMask": ["true"],
    "WriteResultImage": ["false"],
    # Ignored with the Grid sampler, harmless.
    "NumberOfSpatialSamples": ["30000"],
}


def rigid_parameter_map():
    itk = _import_elastix()
    parameters = itk.ParameterObject.New()
    parameter_map = parameters.GetDefaultParameterMap("rigid")
    for key, value in _RIGID_PARAMETERS.items():
        parameter_map[key] = value
    parameters.AddParameterMap(parameter_map)
    return parameters


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def apply_mask(image: sitk.Image, mask: sitk.Image, label: int = None) -> tuple:
    """Zero everything outside the mask. Returns `(masked image, note)`.

    `note` is None when nothing worth reporting happened, and a sentence
    otherwise -- it ends up in the run report rather than in a log line nobody
    reads.

    Two things the original did silently and this does not:

    * it forced `mask.SetOrigin(image.GetOrigin())` unconditionally. That
      papers over the float drift an AMASSS mask can pick up, and it also
      papers over a genuinely different geometry -- a mask cropped or resampled
      relative to its scan is then applied several millimetres off, and nothing
      says so. The geometries are compared first: identical sampling means the
      drift is copied across, anything else is a per-patient failure;
    * a `label` that does not exist in the mask fell through to using the WHOLE
      mask, so asking for label 4 of a mask that only has 1 and 2 registered on
      everything and reported success.
    """
    if mask.GetSize() != image.GetSize():
        raise RegistrationError(
            f"the mask is {mask.GetSize()} voxels and the scan is {image.GetSize()}: "
            f"they are not the same sampling of the same patient."
        )
    if not np.allclose(mask.GetSpacing(), image.GetSpacing(), atol=1e-4):
        raise RegistrationError(
            f"the mask's spacing {tuple(round(v, 4) for v in mask.GetSpacing())} differs "
            f"from the scan's {tuple(round(v, 4) for v in image.GetSpacing())}."
        )
    # Same sampling: any remaining origin/direction difference is the float
    # drift a round trip through a segmentation tool leaves behind.
    mask.CopyInformation(image)

    array = sitk.GetArrayViewFromImage(mask)
    note = None
    if label:
        present = np.unique(array)
        if label not in present:
            raise RegistrationError(
                f"the mask holds no label {label} (it has "
                f"{', '.join(str(int(value)) for value in present)}). Set "
                f"'segmentation_label' to 0 to use the whole mask."
            )
        binary = sitk.GetImageFromArray((array == label).astype(np.uint8))
        binary.CopyInformation(image)
        mask = binary
    elif len(np.unique(array)) > 2:
        note = (
            "the mask holds several labels and 'segmentation_label' is 0, so the "
            "union of all of them was used"
        )

    return sitk.Mask(image, sitk.Cast(mask, sitk.sitkUInt8)), note


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """Rigidly register `moving` onto `fixed`; return the resampling transform.

    The returned transform maps a point of the FIXED image's space to the
    corresponding point of the MOVING image's space, which is the direction
    `sitk.ResampleImageFilter` consumes -- so resampling `moving` with it
    produces the moving image in the fixed image's frame.

    Neither image touches the disk. `MaskedImage` wrote the masked fixed image
    to `<temp_folder>/fixed_image_masked.nii.gz`, a name FIXED across every
    patient of a run and every concurrent request on this server, so two
    overlapping runs registered against each other's anatomy. The conversion is
    in memory.
    """
    itk = _import_elastix()

    registration = itk.ElastixRegistrationMethod.New(_to_itk(fixed), _to_itk(moving))
    registration.SetParameterObject(rigid_parameter_map())
    registration.SetLogToConsole(False)
    registration.UpdateLargestPossibleRegion()
    return retrieve_transform(registration.GetTransformParameterObject())


def retrieve_transform(parameter_object) -> sitk.Transform:
    """The SimpleITK transform equivalent to elastix's result.

    See this module's docstring for why `CenterOfRotationPoint` is read.
    """
    parameters = parameter_object.GetParameterMap(0)
    kind = parameters["Transform"][0]
    values = [float(value) for value in parameters["TransformParameters"]]
    # Absent only for a transform kind that has no centre; falling back to the
    # origin then is exact rather than a guess.
    center = (
        [float(value) for value in parameters["CenterOfRotationPoint"]]
        if "CenterOfRotationPoint" in parameters
        else [0.0, 0.0, 0.0]
    )

    if kind == "EulerTransform":
        transform = sitk.Euler3DTransform()
        transform.SetCenter(center)
        transform.SetRotation(angleX=values[0], angleY=values[1], angleZ=values[2])
        transform.SetTranslation(values[3:6])
        return transform

    if kind == "AffineTransform":
        transform = sitk.AffineTransform(3)
        transform.SetCenter(center)
        transform.SetMatrix(values[:9])
        transform.SetTranslation(values[9:12])
        return transform

    # `ComputeFinalMatrix` used to compose a LIST of transforms by multiplying
    # the rotations and ADDING the translations, which is not composition at
    # all; it was only ever handed one transform, so the error never showed.
    # There is one transform here too, and an unexpected kind says so instead of
    # producing a plausible wrong one.
    raise RegistrationError(
        f"elastix returned a '{kind}', which this tool does not know how to convert. "
        f"Expected EulerTransform (rigid) or AffineTransform."
    )


def _to_itk(image: sitk.Image):
    """A float32 itk.Image sharing this image's geometry, built in memory."""
    itk = _import_elastix()

    array = sitk.GetArrayFromImage(image).astype(np.float32)
    itk_image = itk.GetImageFromArray(np.ascontiguousarray(array))
    itk_image.SetOrigin(tuple(float(value) for value in image.GetOrigin()))
    itk_image.SetSpacing(tuple(float(value) for value in image.GetSpacing()))
    itk_image.SetDirection(
        itk.matrix_from_array(np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3))
    )
    return itk_image

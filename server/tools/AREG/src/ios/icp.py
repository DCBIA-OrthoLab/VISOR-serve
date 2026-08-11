"""Rigidly aligning two palatal patches, and writing the result as a transform.

Ported from `AREG_IOS/AREG_IOS_utils/ICP.py` (ICP / vtkICP) and the
`saveMatrixAsTfm` / `read_matrix` half of `transformation.py`.

Two things the originals got wrong, both invisible:

* **`vtkICP.__call__` returned its source unmoved.** It built a
  `vtkTransformPolyDataFilter`, ran it, and then returned the *input* polydata
  plus the matrix, throwing the filter's output away. `ICP.run` fed that return
  value into the next entry of `list_icp` and multiplied the matrices together
  -- so with more than one method in the list, every one after the first ran on
  unaligned points while its matrix was composed as if it had not. Only one
  method was ever configured, which is why it never showed. There is one
  alignment here and no list to compose;
* **`read_matrix` read `GetTranslation()`**, which is only the whole
  translation when the transform's centre of rotation is the origin. A
  transform about any other centre maps `y = R(x - c) + c + t`, i.e. an offset
  of `t + c - Rc` -- SimpleITK spells that `GetOffset()`. It happens to be
  equal here (ASO writes its transforms centred on the origin), so this is the
  same correction the CBCT engine needed for elastix's output, applied before
  it can bite rather than after.
"""

import numpy as np
import SimpleITK as sitk
import vtk

from . import surfaces

MAX_ITERATIONS = 1000


class RegistrationError(Exception):
    """One patient could not be registered. Reported, and the batch goes on."""


def align(source: vtk.vtkPolyData, target: vtk.vtkPolyData) -> np.ndarray:
    """The 4x4 matrix moving `source`'s points onto `target`'s.

    Both are the vertex-only patch clouds `butterfly.patch_cloud` builds.
    """
    if source.GetNumberOfPoints() == 0 or target.GetNumberOfPoints() == 0:
        raise RegistrationError("one of the two patches holds no point")

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(MAX_ITERATIONS)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    matrix = surfaces.matrix_from_vtk(icp.GetMatrix())
    if not np.all(np.isfinite(matrix)):
        raise RegistrationError("the alignment did not converge to a usable transform")
    return matrix


# ---------------------------------------------------------------------------
# Transform files
# ---------------------------------------------------------------------------

def matrix_of(transform: sitk.Transform) -> np.ndarray:
    """A 4x4 matrix from a SimpleITK linear transform, centre included.

    A centred transform maps `y = R(x - c) + c + t`, so the 4x4 translation
    column is `t + c - Rc`, not `t`. SimpleITK has no accessor for it -- there
    is `GetTranslation()` and `GetCenter()` and nothing that returns the
    composition -- which is exactly why `read_matrix` reading `GetTranslation()`
    into `matrix[:3, 3]` looks correct and is not.
    """
    try:
        linear = np.asarray(transform.GetMatrix(), dtype=np.float64).reshape(3, 3)
        translation = np.asarray(transform.GetTranslation(), dtype=np.float64)
        center = np.asarray(transform.GetCenter(), dtype=np.float64)
    except AttributeError as exc:
        raise RegistrationError(
            f"'{transform.GetName()}' is not a linear transform, so it cannot be "
            f"composed with the registration."
        ) from exc

    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation + center - linear @ center
    return matrix


def write_transform(matrix: np.ndarray, path: str, prior_path: str = None) -> str:
    """Write the resampling transform for a point-moving matrix.

    `matrix` moves the moving mesh's POINTS onto the fixed one. A `.tfm` is
    consumed the other way round -- it maps a point of the result back to where
    to sample it -- so the inverse is what is written, exactly as the CBCT
    engine writes elastix's output.

    `prior_path` is a transform already applied to the mesh before AREG saw it,
    which is the case in the automated modes: ASO oriented both timepoints
    first, so a transform relating AREG's inputs would refer to meshes the
    caller did not send. Composing the two gives one file mapping the caller's
    ORIGINAL mesh to the registered result.
    """
    combined = matrix
    if prior_path:
        # original --prior--> oriented --matrix--> registered
        combined = matrix @ np.linalg.inv(matrix_of(sitk.ReadTransform(prior_path)))

    inverted = np.linalg.inv(combined)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(inverted[:3, :3].flatten().tolist())
    transform.SetTranslation(inverted[:3, 3].tolist())
    sitk.WriteTransform(transform, path)
    return path

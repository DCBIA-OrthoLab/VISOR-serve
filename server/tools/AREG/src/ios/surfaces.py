"""Reading, writing and transforming intra-oral surface meshes, plus the jaw a
file name claims to be.

Ported from `AREG_IOS/AREG_IOS_utils/utils.py` (ReadSurf/WriteSurf/
VTKMatrixToNumpy), `transformation.py` (TransformSurf) and the jaw half of
`dataset.py` (isLowerUpper/removeLowerUpper).

Deliberately AREG's own copy rather than an import of `tools/ASO/src/ios/
surfaces.py`, for the reason ASO and ALI each carry their own markups writer:
importing another tool's module at load time makes one tool's missing
dependency take both out of the registry. The two files are close but not
identical -- this one has the jaw vocabulary AREG needs and no .off reader,
because AREG's IOS engine never sees a .off.

Two behaviours differ from the original:

* **meshes are written BINARY.** `vtkPolyDataWriter` defaults to ASCII, and the
  difference is not cosmetic: binary is smaller, roughly a hundred times faster
  to parse, and the MORE accurate of the two -- it round-trips float32
  coordinates exactly while ASCII prints about six significant digits and moves
  points on read-back (see this repository's 2026-07-30 changelog entry);
* **a mesh whose name does not say its jaw is refused**, instead of being
  treated as a lower arch. `isLowerUpper(file, "Upper")` returning False was
  taken to mean "lower" by `Sort`, so a maxillary mesh named `patient1.vtk` was
  registered against the mandibular timepoint and returned as a success.
"""

import os
import re

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .. import catalogs

SURFACE_EXTENSIONS = (".vtk", ".vtp", ".stl", ".obj")

# The point arrays a crown-segmented intra-oral scan may carry, most specific
# first. Slicer's tools have used all three over time.
LABEL_ARRAY_NAMES = ("Universal_ID", "PredictedID", "UniversalID")

_SEPARATORS = re.compile(r"[_\-.\s]+")


class SurfaceError(Exception):
    """A mesh could not be read, or does not carry what the mode needs."""


def is_surface_file(filename: str) -> bool:
    return filename.lower().endswith(SURFACE_EXTENSIONS)


def read_surface(path: str) -> vtk.vtkPolyData:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllFieldsOn()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
    elif extension == ".obj":
        reader = vtk.vtkOBJReader()
    else:
        raise SurfaceError(
            f"'{os.path.basename(path)}': unsupported surface format. Expected one of "
            f"{', '.join(SURFACE_EXTENSIONS)}."
        )

    reader.SetFileName(path)
    reader.Update()
    surface = reader.GetOutput()
    if surface is None or surface.GetNumberOfPoints() == 0:
        raise SurfaceError(f"'{os.path.basename(path)}' holds no geometry.")
    return surface


def write_surface(surface: vtk.vtkPolyData, path: str) -> str:
    """Write a mesh, keeping its per-point arrays. Returns the path written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".vtp"):
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()
    writer.SetFileName(path)
    writer.SetInputData(surface)
    writer.Update()
    return path


def output_extension(input_path: str) -> str:
    """The format a result mesh is written in.

    `.vtp` in gives `.vtp` out; everything else becomes `.vtk`. The tooth-label
    array is what makes an intra-oral result usable downstream and `.stl` cannot
    hold it, so `.stl` is read but never written back.
    """
    return ".vtp" if input_path.lower().endswith(".vtp") else ".vtk"


def label_array_name(surface: vtk.vtkPolyData) -> str:
    """The per-point tooth-label array this mesh carries, or None."""
    point_data = surface.GetPointData()
    present = {point_data.GetArrayName(index) for index in range(point_data.GetNumberOfArrays())}
    for name in LABEL_ARRAY_NAMES:
        if name in present:
            return name
    return None


def jaw_of(filename: str) -> str:
    """'Upper', 'Lower', or None when the name does not say.

    Matched on whole tokens of the stem. The original used substrings, and its
    Upper vocabulary included the bare `_U` and `U_` -- so a patient identifier
    like `P_U12` was an upper arch whatever the file actually held, and a
    subject folder named `Mdx` made every mesh in it a mandible.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    for token in _SEPARATORS.split(stem):
        jaw = catalogs.JAW_TOKENS.get(token.lower())
        if jaw:
            return jaw
    return None


def transform_surface(surface: vtk.vtkPolyData, matrix: np.ndarray) -> vtk.vtkPolyData:
    """Apply a 4x4 matrix to a mesh, returning a new one."""
    copy = vtk.vtkPolyData()
    copy.DeepCopy(surface)

    transform = vtk.vtkTransform()
    transform.SetMatrix(np.asarray(matrix, dtype=np.float64).reshape(16).tolist())

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(copy)
    transform_filter.Update()
    return transform_filter.GetOutput()


def points_of(surface: vtk.vtkPolyData) -> np.ndarray:
    return vtk_to_numpy(surface.GetPoints().GetData())


def matrix_from_vtk(matrix) -> np.ndarray:
    return np.array(
        [[matrix.GetElement(row, column) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def compute_normals(surface: vtk.vtkPolyData) -> vtk.vtkPolyData:
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def scale_to_unit(surface: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Centre a mesh on its bounding box and scale it into the unit sphere.

    What the network was trained on. Rewritten in numpy: the original walked
    every vertex twice through `GetPoint`/`SetPoint`, which on a 200k-vertex
    intra-oral scan is the slowest step of the whole preprocessing.
    """
    copy = vtk.vtkPolyData()
    copy.DeepCopy(surface)
    points = copy.GetPoints()

    bounds = np.asarray(points.GetBounds(), dtype=np.float64)
    center = np.array(
        [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
    )
    extreme = np.array([max(bounds[0], bounds[1]), max(bounds[2], bounds[3]), max(bounds[4], bounds[5])])
    span = np.linalg.norm(extreme - center)
    if span == 0:
        raise SurfaceError("mesh has zero extent, nothing to scale")

    scaled = (vtk_to_numpy(points.GetData()) - center) / span
    from vtk.util.numpy_support import numpy_to_vtk

    points.SetData(numpy_to_vtk(np.ascontiguousarray(scaled, dtype=np.float32), deep=True))
    copy.SetPoints(points)
    return copy

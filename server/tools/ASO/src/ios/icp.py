"""IOS registration: tooth centroids or landmarks in, a 4x4 matrix out.

Ported from `ASO_IOS/ASO_IOS_utils/icp.py` (the `ICP`, `vtkICP`, `InitIcp`,
`vtkMeanTeeth` and `SelectKey` classes). Three things changed.

**The triplet search no longer goes through the filesystem.** `InitIcp` wrote
`source.npy` and `target.npy` into `ASO_IOS_utils/cache/` -- a directory it
created inside its own installed package -- and reloaded `source.npy` on every
one of up to 2500 search iterations. On a server that is a write into the
install tree, thousands of pointless round trips per patient, and, since the
path is fixed, two concurrent requests overwriting each other's landmarks. The
search is pure and in memory now (see `..geometry.best_triplet`).

**The composition order is fixed.** The original built its final matrix as
`M_init @ M_icp`, but the ICP is computed on points the initialisation has
already moved, so the two must compose the other way round: `M_icp @ M_init`.
The CBCT engine always did it correctly (`TransformMatrixBis @ TransformMatrix`
in `SEMI_ASO_CBCT`), which is what makes this a transcription slip rather than
a deliberate choice. It stayed invisible because the ICP that follows a good
initialisation is nearly the identity.

**Point sets must correspond.** `SameNumberPoint` silently subsampled the
longer of the two with the global `np.random` and re-keyed both by index, so a
patient missing one tooth was registered against a random correspondence
instead of failing. A mismatch raises here.
"""

import numpy as np
import vtk

from .. import geometry
from . import surfaces


class RegistrationError(Exception):
    """One jaw could not be registered; the reason goes into the run report."""


def mean_teeth(surface: vtk.vtkPolyData, tooth_ids, array_name: str) -> dict:
    """{tooth id as str: centroid} for each requested tooth.

    Raises RegistrationError naming the tooth when the mesh carries no point
    for it -- the mesh is segmented, but that tooth is absent or unlabelled.
    """
    labels = surfaces.labels_of(surface, array_name)
    points = surfaces.points_of(surface)

    centroids = {}
    for tooth_id in tooth_ids:
        selected = points[labels == tooth_id]
        if selected.size == 0:
            raise RegistrationError(
                f"tooth {tooth_id} is not present in the mesh's '{array_name}' labels"
            )
        centroids[str(tooth_id)] = np.mean(selected, axis=0)
    return centroids


def select_keys(landmarks: dict, keys) -> dict:
    """The named landmarks, failing with the list of missing ones rather than
    on the first KeyError."""
    missing = [key for key in keys if key not in landmarks]
    if missing:
        raise RegistrationError(f"missing landmark(s): {', '.join(missing)}")
    return {key: landmarks[key] for key in keys}


def register(
    source: dict,
    target: dict,
    max_triplets: int = 2500,
    seed: int = 0,
) -> np.ndarray:
    """The 4x4 matrix taking `source` points onto `target` points.

    Both dicts must be keyed identically -- tooth ids for the fully-automated
    mode, `<tooth><landmark type>` labels for the semi-automated one.
    """
    if set(source) != set(target):
        only_source = sorted(set(source) - set(target))
        only_target = sorted(set(target) - set(source))
        raise RegistrationError(
            "the input and the reference do not describe the same points "
            f"(only in the input: {', '.join(only_source) or 'none'}; "
            f"only in the reference: {', '.join(only_target) or 'none'})"
        )
    if len(source) < geometry_min_points():
        raise RegistrationError(
            f"only {len(source)} point(s), {geometry_min_points()} are needed"
        )

    ordered_source = {key: source[key] for key in sorted(source)}
    ordered_target = {key: target[key] for key in sorted(target)}

    picks = geometry.best_triplet(
        ordered_source,
        ordered_target,
        include_translation=True,
        max_triplets=max_triplets,
        seed=seed,
    )
    coarse, coarse_matrix, _ = geometry.init_icp(
        ordered_source, ordered_target, picks, include_translation=True
    )
    return _icp_matrix(coarse, ordered_target) @ coarse_matrix


def geometry_min_points() -> int:
    """Three: the coarse alignment picks three distinct points."""
    return 3


def _icp_matrix(source: dict, target: dict) -> np.ndarray:
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(_to_polydata(source))
    icp.SetTarget(_to_polydata(target))
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    matrix = icp.GetMatrix()
    return np.array(
        [[matrix.GetElement(row, col) for col in range(4)] for row in range(4)]
    )


def _to_polydata(points: dict) -> vtk.vtkPolyData:
    vtk_points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    for position in points.values():
        point_id = vtk_points.InsertNextPoint([float(value) for value in position])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point_id)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetVerts(vertices)
    return polydata

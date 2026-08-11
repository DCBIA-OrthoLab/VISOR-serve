"""The lower arch's registration patch: a band of surface around the
mucogingival line.

Ported from `AREG_IOS/AREG_IOS_utils/mgl_patch.py` in the upstream
SlicerAutomatedDentalTools ("ADD: MGL registration mode for the lower arch in
AREG_IOS", 2026-08), which the Cloud fork the rest of AREG came from predates.

**Why it exists.** The upper arch is registered on the palate, a plateau
orthodontic treatment does not move, painted by a network (see `butterfly.py`).
The mandible has no such plateau -- but it has the mucogingival line, where
attached gingiva meets alveolar mucosa. ALI's MG model places 13 landmarks along
it; joined into a curve and grown into a band, they play exactly the same role,
and the patch is written as a 0/1 point array of the same shape so the ICP reads
it identically.

**It needs no network of its own**, which is the operational point on this
server: the palatal patch needs pytorch3d, and this one is a spline, a
shortest-path walk and a label lookup. A deployment without pytorch3d answers
501 for the upper arch and registers lower arches perfectly well.

Two properties the original singles out, and both are load-bearing:

* every sample of the curve is **snapped onto the mesh**, because a curve
  interpolated between landmarks floats off the surface in the concavities
  between teeth;
* the band grows **along the surface** (geodesic), never through space, so a
  buccal patch cannot leak onto the lingual side wherever the ridge is thinner
  than the radius.

What is different here is the implementation of the second one, not its
meaning: `_adjacency` built a Python `set` per vertex by walking every cell of
the mesh through `vtkIdList`, which on a 200k-vertex intraoral scan is millions
of interpreter-level operations for something `postprocess.Adjacency` already
computes in numpy. The Dijkstra walk itself is unchanged.
"""

import heapq
import logging

import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from . import postprocess, surfaces

logger = logging.getLogger("AREG")

# Landmark names of the MG model, in arch order. L0MG is the midline (tooth 25),
# so the right side is shifted by one against the tooth numbers.
MGL_ORDER = (
    "LL6MG", "LL5MG", "LL4MG", "LL3MG", "LL2MG", "LL1MG", "L0MG",
    "LR1MG", "LR2MG", "LR3MG", "LR4MG", "LR5MG", "LR6MG",
)

# Names written by predictions made before the MG suffix was added.
MGL_ORDER_LEGACY = tuple(name[:-2] for name in MGL_ORDER)

# The palatal patch is called "Butterfly" after its shape; the band along the
# mucogingival line is neither butterfly-shaped nor on the same arch, so it
# carries its own name -- a mandible is never labelled after the palate.
MGL_ARRAY_NAME = "Bottom_MGL"

DEFAULT_HEIGHT = 5.0     # mm, half-height of the band around the curve
DEFAULT_SAMPLES = 300    # samples along the spline

# Universal_ID labels the band is kept OFF: the crowns are what moves between
# the two timepoints, so a patch overlapping them would register the change
# instead of measuring it.
#
# 18..31 is the upstream range, verbatim. It leaves the two lower third molars
# (17 and 32) in the band if it ever reaches them, which reads like an
# off-by-one on both ends of "the lower teeth" -- but it is exactly the span
# ALI's MG model is trained on (teeth 19..31, plus 18), so it may equally be
# deliberate. Kept as-is: widening it changes which vertices drive a clinical
# registration, and that is the upstream author's call, not this port's.
LOWER_TOOTH_LABELS = tuple(range(18, 32))


class PatchError(Exception):
    """The MGL patch could not be built for this mesh."""


def ordered_landmarks(landmarks: dict) -> np.ndarray:
    """The MG landmark positions in arch order, as an (N, 3) array.

    Accepts both the current names (LL6MG...) and the older suffix-less ones,
    and tolerates missing teeth: a scan where ALI could not place every point
    still yields a usable curve as long as three remain.
    """
    for order in (MGL_ORDER, MGL_ORDER_LEGACY):
        points = [np.asarray(landmarks[name], dtype=float) for name in order if name in landmarks]
        if len(points) >= 3:
            missing = [name for name in order if name not in landmarks]
            if missing:
                logger.warning("AREG MGL: %d MG landmark(s) missing from the prediction", len(missing))
            return np.array(points), missing
    raise PatchError(
        f"fewer than 3 mucogingival landmarks in this file (expected names such as "
        f"{', '.join(MGL_ORDER[:3])})"
    )


def _spline_through(points: np.ndarray, samples: int) -> np.ndarray:
    """Sample a B-spline passing through `points`.

    The landmarks are sparse (one per tooth), so the curve between them is an
    interpolation, not a measurement: it only places the band.
    """
    import vtk

    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(*point)

    spline = vtk.vtkParametricSpline()
    spline.SetPoints(vtk_points)
    spline.ClosedOff()

    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(spline)
    source.SetUResolution(samples)
    source.Update()
    return vtk_to_numpy(source.GetOutput().GetPoints().GetData())


def _snap_to_surface(surface, samples: np.ndarray) -> list:
    """The id of the closest vertex of `surface` for each sample.

    Duplicates are removed: consecutive samples often land on one vertex.
    """
    import vtk

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    return sorted({locator.FindClosestPoint(sample) for sample in samples})


def grow_band(surface, seeds, radius: float, adjacency=None) -> np.ndarray:
    """Boolean mask of the vertices within `radius` mm of a seed, along the mesh.

    Dijkstra over the edge graph, exactly as upstream; the adjacency it walks is
    the numpy one `postprocess.Adjacency` builds rather than a Python set per
    vertex.
    """
    adjacency = adjacency or postprocess.Adjacency(surface)
    points = surfaces.points_of(surface)

    distance = np.full(adjacency.count, np.inf)
    queue = []
    for seed in seeds:
        distance[seed] = 0.0
        heapq.heappush(queue, (0.0, int(seed)))

    while queue:
        walked, point_id = heapq.heappop(queue)
        if walked > distance[point_id]:
            continue
        neighbours = adjacency.neighbours(point_id)
        steps = np.linalg.norm(points[neighbours] - points[point_id], axis=1)
        for neighbour, step in zip(neighbours, steps):
            reached = walked + float(step)
            if reached < distance[neighbour] and reached <= radius:
                distance[neighbour] = reached
                heapq.heappush(queue, (reached, int(neighbour)))

    return distance <= radius


def _tooth_mask(surface) -> np.ndarray:
    """True where a vertex belongs to a lower crown, False on the gingiva.

    All-False when the mesh carries no segmentation, so the caller keeps the
    whole band rather than losing the patch entirely.
    """
    array_name = surfaces.label_array_name(surface)
    if array_name is None:
        logger.warning(
            "AREG MGL: this mesh carries no tooth labels, so the band is not kept off the crowns"
        )
        return np.zeros(surface.GetNumberOfPoints(), dtype=bool)
    labels = vtk_to_numpy(surface.GetPointData().GetArray(array_name))
    return np.isin(labels, LOWER_TOOTH_LABELS)


def build_patch(
    surface,
    landmarks: dict,
    height: float = DEFAULT_HEIGHT,
    samples: int = DEFAULT_SAMPLES,
    array_name: str = MGL_ARRAY_NAME,
    exclude_teeth: bool = True,
) -> tuple:
    """Paint the band around the mucogingival line into `array_name`.

    Returns `(surface, note)`. `note` is None when nothing worth reporting
    happened and a sentence otherwise, so a partial prediction reaches the run
    report rather than a log line nobody reads.

    A `height` of 0 leaves no band and no curve either: the array then holds the
    snapped landmarks alone and the ICP runs on those points only. Upstream
    keeps that as the control case, to measure on real scans what the surface
    around the mucogingival line buys over the points that carry it.
    """
    surface = postprocess.triangulate(surface)
    points, missing = ordered_landmarks(landmarks)

    if height == 0:
        seeds = _snap_to_surface(surface, points)
        inside = np.zeros(surface.GetNumberOfPoints(), dtype=bool)
        inside[seeds] = True
    else:
        seeds = _snap_to_surface(surface, _spline_through(points, samples))
        inside = grow_band(surface, seeds, height)

    dropped = 0
    if exclude_teeth:
        on_teeth = _tooth_mask(surface) & inside
        dropped = int(on_teeth.sum())
        inside = inside & ~on_teeth

    if not inside.any():
        raise PatchError(
            "the mucogingival patch came out empty -- the landmarks may not belong to "
            "this scan, or every point of the band fell on a crown"
        )

    array = numpy_to_vtk(np.ascontiguousarray(inside.astype(np.int64)), deep=True)
    array.SetName(array_name)
    surface.GetPointData().AddArray(array)
    surface.GetPointData().SetActiveScalars(array_name)

    logger.info(
        "AREG MGL: %d of %d vertices in the patch (height %g mm, %d dropped as crown)",
        int(inside.sum()), surface.GetNumberOfPoints(), height, dropped,
    )
    note = None
    if missing:
        note = (
            f"{len(missing)} of the {len(MGL_ORDER)} mucogingival landmarks were absent "
            f"({', '.join(missing)}); the curve was interpolated through the rest"
        )
    return surface, note

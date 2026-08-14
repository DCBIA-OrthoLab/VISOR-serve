"""Cleaning up the predicted palatal patch: drop specks, close pinholes.

Ported from `AREG_IOS/AREG_IOS_utils/post_process.py` (RemoveIslands /
DilateLabel / ErodeLabel / GetNeighbors / ConnectedRegion / NeighborLabel).

The semantics are unchanged; the implementation is not. The original asked VTK
for a point's neighbours one `vtkIdList` at a time, inside loops over every
point, once per morphological iteration -- a few million Python-level VTK calls
on a 200k-vertex scan for what is one sparse adjacency traversal. The adjacency
is built once here, in numpy.

One call is removed rather than ported: `RemoveIslands(surf, labels, 33, 500)`.
The label array is binary, so nothing was ever equal to 33 -- it is the first
of the four post-processing steps and it has never run.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def triangulate(surface: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """A triangle-only copy of a mesh.

    Everything downstream -- the network's face tensor, the adjacency below --
    reads `GetPolys()` as a flat `[3, i, j, k, 3, i, j, k, ...]` array and
    reshapes it to `(-1, 4)`. On a mesh holding anything but triangles that
    reshape does not fail, it silently reads the wrong indices, so the shape
    is guaranteed here instead of assumed.
    """
    if _is_all_triangles(surface):
        return surface
    filter_ = vtk.vtkTriangleFilter()
    filter_.SetInputData(surface)
    filter_.PassLinesOff()
    filter_.PassVertsOff()
    filter_.Update()
    return filter_.GetOutput()


def _is_all_triangles(surface: vtk.vtkPolyData) -> bool:
    polys = surface.GetPolys()
    if polys.GetNumberOfCells() == 0:
        return False
    return polys.GetData().GetNumberOfTuples() == 4 * polys.GetNumberOfCells()


def faces_of(surface: vtk.vtkPolyData) -> np.ndarray:
    """(n, 3) point indices. The mesh must already be triangulated."""
    return vtk_to_numpy(surface.GetPolys().GetData()).reshape(-1, 4)[:, 1:]


class Adjacency:
    """Point-to-point adjacency in CSR form, built once per mesh.

    `neighbours(pid)` is every point sharing a triangle with `pid`, which is
    what the original's `GetNeighbors` returned.
    """

    def __init__(self, surface: vtk.vtkPolyData):
        faces = faces_of(surface)
        left = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]] * 2)
        right = np.concatenate(
            [faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 2], faces[:, 0], faces[:, 1]]
        )
        count = surface.GetNumberOfPoints()

        order = np.argsort(left, kind="stable")
        self._indices = right[order].astype(np.int64)
        self._indptr = np.zeros(count + 1, dtype=np.int64)
        np.cumsum(np.bincount(left, minlength=count), out=self._indptr[1:])
        self.count = count

    def neighbours(self, pid: int) -> np.ndarray:
        return self._indices[self._indptr[pid]: self._indptr[pid + 1]]

    def dilate_frontier(self, mask: np.ndarray) -> np.ndarray:
        """Points adjacent to `mask` but not in it, as a boolean array."""
        touched = np.zeros(self.count, dtype=bool)
        source = np.flatnonzero(mask)
        for pid in source:
            touched[self.neighbours(pid)] = True
        return touched & ~mask

    def boundary(self, mask: np.ndarray, target: int = None, labels: np.ndarray = None) -> np.ndarray:
        """Points of `mask` having at least one neighbour outside it.

        With `target` given, only a neighbour carrying that label counts --
        which is what `ErodeLabel(target=...)` meant.
        """
        result = np.zeros(self.count, dtype=bool)
        for pid in np.flatnonzero(mask):
            neighbours = self.neighbours(pid)
            outside = ~mask[neighbours]
            if target is not None:
                outside &= labels[neighbours] == target
            if outside.any():
                result[pid] = True
        return result


def remove_islands(labels: np.ndarray, adjacency: Adjacency, label: int, min_count: int) -> None:
    """Relabel connected components of `label` smaller than `min_count`.

    Each undersized component takes the label most common among the points
    touching it. A component with no neighbour of any other label is left
    alone, exactly as the original's `neighbor_label != -1` guard did.

    Mutates `labels` in place.
    """
    mask = labels == label
    visited = np.zeros(adjacency.count, dtype=bool)

    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        component = _component(start, mask, visited, adjacency)
        if len(component) >= min_count:
            continue
        replacement = _surrounding_label(component, labels, label, adjacency)
        if replacement is not None:
            labels[component] = replacement


def _component(start: int, mask: np.ndarray, visited: np.ndarray, adjacency: Adjacency) -> np.ndarray:
    """Every point connected to `start` through points of `mask`.

    Iterative rather than the original's recursive-ish `while` over a growing
    Python list with `np.append` inside it, which is quadratic in the component
    size -- the palate patch is tens of thousands of points.
    """
    visited[start] = True
    stack = [start]
    component = [start]
    while stack:
        pid = stack.pop()
        for neighbour in adjacency.neighbours(pid):
            if mask[neighbour] and not visited[neighbour]:
                visited[neighbour] = True
                stack.append(neighbour)
                component.append(neighbour)
    return np.array(component, dtype=np.int64)


def _surrounding_label(component: np.ndarray, labels: np.ndarray, label: int, adjacency: Adjacency):
    neighbouring: list = []
    for pid in component:
        neighbours = adjacency.neighbours(pid)
        neighbouring.append(neighbours[labels[neighbours] != label])
    if not neighbouring:
        return None
    candidates = np.concatenate(neighbouring)
    if candidates.size == 0:
        return None
    values, counts = np.unique(labels[candidates], return_counts=True)
    return values[np.argmax(counts)]


def dilate(labels: np.ndarray, adjacency: Adjacency, label: int, iterations: int = 2) -> None:
    """Grow `label` by `iterations` rings. Mutates `labels` in place."""
    for _ in range(iterations):
        frontier = adjacency.dilate_frontier(labels == label)
        if not frontier.any():
            return
        labels[frontier] = label


def erode(labels: np.ndarray, adjacency: Adjacency, label: int, iterations: int = 2,
          target: int = None) -> None:
    """Shrink `label` by `iterations` rings, each boundary point taking a
    neighbouring label. Mutates `labels` in place.
    """
    for _ in range(iterations):
        mask = labels == label
        boundary = adjacency.boundary(mask, target=target, labels=labels)
        if not boundary.any():
            return
        # Read from a snapshot: a point relabelled earlier in this pass must not
        # become a donor for its own neighbours, or one ring of erosion would
        # cascade across the whole patch. The original relabelled the entire
        # boundary in one go after collecting it, for the same reason.
        before = labels.copy()
        for pid in np.flatnonzero(boundary):
            neighbours = adjacency.neighbours(pid)
            outside = neighbours[before[neighbours] != label]
            if target is not None:
                outside = outside[before[outside] == target]
            if outside.size:
                labels[pid] = before[outside[0]]


def clean_patch(labels: np.ndarray, adjacency: Adjacency) -> np.ndarray:
    """The original's post-processing chain, minus its one dead call.

    Specks of either class smaller than 200 points are absorbed, then the patch
    is closed (dilate 2, erode 2) so a pinhole in the middle of the palate does
    not become a hole in the point cloud the ICP runs on.
    """
    labels = labels.astype(np.int64, copy=True)
    for label in (0, 1):
        remove_islands(labels, adjacency, label, 200)
    dilate(labels, adjacency, 1, iterations=2)
    erode(labels, adjacency, 1, iterations=2)
    return labels

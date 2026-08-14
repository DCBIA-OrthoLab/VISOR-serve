"""Putting a labelled intra-oral mesh into the canonical frame the patch
network was trained in.

Ported from `AREG_IOS/AREG_IOS_utils/orientation.py` and the `vtkMeanTeeth`
half of `vtkSegTeeth.py`. Four tooth centroids -- UR6, UR1, UL1, UL6 (universal
ids 3, 8, 9, 14) -- are aligned onto a fixed reference triangle, which points
the arch the same way for every patient so the seven cameras above it always
look at the palate.

This is a preprocessing step, not a result: the transform is used to render, and
the registration the tool returns is computed and reported in the mesh's own
original coordinates.

Two latent bugs are fixed, both silent:

* `np.arccos` was clamped at +1 only, so a dot product rounding just past -1
  gave NaN, which propagated through the rotation matrix into every vertex;
* `RotationMatrix` normalised its axis without checking it, so two already
  parallel vectors divided by zero. The rotation there is the identity.
"""

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from . import surfaces

# Universal ids of the four teeth the canonical frame is built on, and the
# reference triangle they are aligned onto, both straight from the original.
REFERENCE_TEETH = (3, 8, 9, 14)
REFERENCE_TRIANGLE = np.array([[-0.5, -0.5, 0.0], [0.0, 0.0, 0.0], [0.5, -0.5, 0.0]])


class OrientationError(Exception):
    """The mesh does not carry what the canonical frame is built from."""


def rotation_matrix(axis, theta: float) -> np.ndarray:
    """Counter-clockwise rotation of `theta` radians about `axis`."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0 or not np.isfinite(norm):
        # Anti/parallel vectors have no rotation axis; the rotation is the
        # identity. The original divided by zero here and returned NaNs.
        return np.eye(3)
    axis = axis / norm

    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    """The angle between two unit vectors, clamped at BOTH ends."""
    return float(np.arccos(np.clip(float(np.dot(first, second)), -1.0, 1.0)))


def tooth_centroids(surface, teeth=REFERENCE_TEETH) -> dict:
    """{universal id: centroid} for the requested teeth.

    Raises `OrientationError` naming what is missing, rather than the
    original's two separate exception types caught two frames apart.
    """
    array_name = surfaces.label_array_name(surface)
    if array_name is None:
        raise OrientationError(
            f"the mesh carries no tooth-label array (expected one of "
            f"{', '.join(surfaces.LABEL_ARRAY_NAMES)})"
        )

    labels = vtk_to_numpy(surface.GetPointData().GetScalars(array_name))
    points = surfaces.points_of(surface)

    centroids = {}
    missing = []
    for tooth in teeth:
        selected = points[labels == tooth]
        if selected.size == 0:
            missing.append(str(tooth))
            continue
        centroids[tooth] = selected.mean(axis=0)
    if missing:
        raise OrientationError(
            f"the mesh has no points labelled {', '.join(missing)} "
            f"(universal ids {', '.join(str(t) for t in teeth)} are needed)"
        )
    return centroids


def _frame(edge_points, apex) -> tuple:
    """The (normal, in-plane direction) frame of a triangle."""
    along = edge_points[1] - edge_points[0]
    along = along / np.linalg.norm(along)

    first = edge_points[0] - apex
    second = edge_points[1] - apex
    normal = np.cross(first / np.linalg.norm(first), second / np.linalg.norm(second))
    normal = normal / np.linalg.norm(normal)

    direction = np.cross(normal, along)
    return normal, direction / np.linalg.norm(direction)


def canonical_transform(surface) -> np.ndarray:
    """The 4x4 matrix putting this mesh into the canonical frame."""
    centroids = tooth_centroids(surface)
    left, middle_1, middle_2, right = (centroids[tooth] for tooth in REFERENCE_TEETH)
    middle_source = (middle_1 + middle_2) / 2.0

    left_target, middle_target, right_target = REFERENCE_TRIANGLE

    normal_source, direction_source = _frame([right, left], middle_source)
    normal_target, direction_target = _frame([right_target, left_target], middle_target)

    align_normal = rotation_matrix(
        np.cross(normal_source, normal_target), _angle_between(normal_source, normal_target)
    )
    direction_source = align_normal @ direction_source
    direction_source = direction_source / np.linalg.norm(direction_source)

    align_direction = rotation_matrix(
        np.cross(direction_source, direction_target),
        _angle_between(direction_source, direction_target),
    )
    rotation = align_direction @ align_normal

    rotated = np.array([rotation @ point for point in (left, middle_source, right)])
    offset = np.array([left_target, middle_target, right_target]).mean(axis=0) - rotated.mean(axis=0)

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = offset
    return matrix


def to_canonical(surface):
    """`(oriented mesh, 4x4 matrix)`. Raises OrientationError if it cannot."""
    matrix = canonical_transform(surface)
    return surfaces.transform_surface(surface, matrix), matrix

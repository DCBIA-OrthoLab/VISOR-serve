"""Coarse jaw alignment from three or four tooth centroids.

Ported from `ASO_IOS/ASO_IOS_utils/pre_icp.py::PrePreAso`. A dental arch is
nearly planar, so three well-spread teeth define both the occlusal plane's
normal and an in-plane direction; matching those two on the input and on the
reference brings a jaw close enough for the ICP that follows to converge.

Unchanged except for the arccos clamping, which is now handled once in
`..geometry.angle_and_axis` (the original clamped `dt > 1.0` and left
`dt < -1.0` to produce NaN, which propagated into the rotation matrix and came
out as a mesh at the origin).
"""

import numpy as np

from .. import catalogs, geometry
from . import icp as ios_icp


def align(source_surface, target_surface, tooth_names: list, array_name: str) -> np.ndarray:
    """The 4x4 matrix roughly aligning one jaw onto the reference's.

    `tooth_names` must hold three or four teeth of the SAME jaw; the schema's
    own description says so and `pipeline` enforces it before any mesh is read.
    """
    if len(tooth_names) not in (3, 4):
        raise ios_icp.RegistrationError(
            f"the coarse alignment needs 3 or 4 teeth per jaw, got {len(tooth_names)}"
        )

    left, middle, right = _organize(tooth_names)
    tooth_ids = catalogs.teeth_to_ids(tooth_names)
    source = ios_icp.mean_teeth(source_surface, tooth_ids, array_name)
    target = ios_icp.mean_teeth(target_surface, tooth_ids, array_name)

    source_points = _triangle(source, left, middle, right)
    target_points = _triangle(target, left, middle, right)

    matrix = _rotation_between(source_points, target_points)

    rotated = [matrix @ point for point in source_points]
    offset = np.mean(np.array(target_points), axis=0) - np.mean(np.array(rotated), axis=0)

    full = np.eye(4)
    full[:3, :3] = matrix
    full[:3, 3] = offset
    return full


def _organize(tooth_names: list) -> tuple:
    """Split the selected teeth into (left, middle, right) by universal id.

    Universal numbering runs right-to-left across the upper arch (1..16) and
    left-to-right across the lower one (17..32), so which end is "left" depends
    on the jaw.
    """
    numbers = sorted(catalogs.teeth_to_ids(tooth_names))
    lowest, highest = numbers[0], numbers[-1]
    middle = [str(number) for number in numbers[1:-1]]
    if catalogs.TOOTH_NAMES[lowest] in catalogs.UPPER_TEETH:
        return str(highest), middle, str(lowest)
    return str(lowest), middle, str(highest)


def _triangle(centroids: dict, left: str, middle: list, right: str) -> list:
    """(left, middle, right) points; the two middle teeth of a four-tooth
    selection are averaged into one."""
    middle_point = np.mean(
        np.array([centroids[name] for name in middle]), axis=0
    )
    return [centroids[left], middle_point, centroids[right]]


def _rotation_between(source_points: list, target_points: list) -> np.ndarray:
    left_source, middle_source, right_source = source_points
    left_target, middle_target, right_target = target_points

    normal_source, direction_source = _plane_frame(
        right_source, left_source, middle_source
    )
    normal_target, direction_target = _plane_frame(
        right_target, left_target, middle_target
    )

    angle = np.arccos(np.clip(np.dot(normal_source, normal_target), -1.0, 1.0))
    to_normal = geometry.rotation_matrix(np.cross(normal_source, normal_target), angle)

    direction_source = to_normal @ direction_source
    norm = np.linalg.norm(direction_source)
    if norm:
        direction_source = direction_source / norm

    angle = np.arccos(np.clip(np.dot(direction_source, direction_target), -1.0, 1.0))
    to_direction = geometry.rotation_matrix(
        np.cross(direction_source, direction_target), angle
    )
    return to_direction @ to_normal


def _plane_frame(first, second, apex) -> tuple:
    """Normal of the plane through three points, and an in-plane direction
    perpendicular to the first-to-second edge."""
    edge = _unit(second - first)
    normal = _unit(np.cross(_unit(first - apex), _unit(second - apex)))
    return normal, _unit(np.cross(normal, edge))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm

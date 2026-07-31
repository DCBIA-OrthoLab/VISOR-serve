"""The landmark-registration maths shared by both ASO engines.

`ASO_CBCT_utils/utils.py` and `ASO_IOS_utils/{icp,transformation}.py` carried
two copies of all of this -- `RotationMatrix`, `AngleAndAxisVectors`,
`InitICP`, `FindOptimalLandmarks`, `ComputeMeanDistance` -- that had drifted
apart in ways nobody could have wanted (the IOS copy cached its input to a
`.npy` file on disk and reloaded it on every search iteration; the CBCT copy
did not). One implementation, parameterised where the two genuinely differ.

Pure numpy: no VTK, no SimpleITK, no I/O. That is what makes it testable
without either heavy library, and what keeps the triplet search free of the
global state it used to depend on.
"""

import itertools
import logging

import numpy as np

logger = logging.getLogger("ASO")

# Steps init_icp reports, so a caller that needs SimpleITK transforms (the CBCT
# engine composes them to resample the volume) can rebuild them in the same
# order without this module importing SimpleITK.
STEP_TRANSLATION = "translation"
STEP_ROTATION = "rotation"


def rotation_matrix(axis, theta: float) -> np.ndarray:
    """Rotation matrix for a counterclockwise rotation of `theta` radians
    about `axis` (Euler-Rodrigues).

    A zero-length axis means the two vectors were already parallel, so there is
    nothing to rotate: the original divided by `np.linalg.norm(axis)` anyway and
    produced a matrix of NaN, which then propagated silently into the final
    transform and came out as a scan full of zeros.
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0 or not np.isfinite(norm):
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


def angle_and_axis(v1, v2) -> tuple:
    """Angle and rotation axis taking `v1` onto `v2`.

    Two guards the original lacked, both of which produced NaN rather than an
    error:

    * `np.amax(v)` is used to scale the vectors (it is what the original did,
      and the scaling cancels out of the angle) -- but it is zero for a null
      vector and negative for an all-negative one. Callers pass componentwise
      absolute values, so a zero only happens for coincident landmarks;
      returning "no rotation" is the honest answer.
    * `np.arccos` of a dot product that floating-point rounding pushed to
      1.0000000002 is NaN. `pre_icp.py` clamped the upper end only, leaving
      antiparallel vectors to produce NaN.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    scale1, scale2 = np.amax(v1), np.amax(v2)
    if scale1 == 0 or scale2 == 0:
        return 0.0, np.zeros(3)
    v1_u = v1 / scale1
    v2_u = v2 / scale2
    norm1, norm2 = np.linalg.norm(v1_u), np.linalg.norm(v2_u)
    if norm1 == 0 or norm2 == 0:
        return 0.0, np.zeros(3)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u) / (norm1 * norm2), -1.0, 1.0))
    return angle, np.cross(v1_u, v2_u)


def translate_landmarks(landmarks: dict, offset) -> dict:
    return {name: point + offset for name, point in landmarks.items()}


def transform_landmarks(landmarks: dict, matrix: np.ndarray) -> dict:
    """Apply a 4x4 homogeneous matrix to every point of a landmark dict."""
    return {
        name: (matrix @ np.append(point, 1.0))[:3] for name, point in landmarks.items()
    }


def mean_distance(source: dict, target: dict) -> float:
    """Mean point-to-point distance over the labels the two dicts share."""
    shared = [name for name in source if name in target]
    if not shared:
        return float("inf")
    return float(
        np.mean([np.linalg.norm(source[name] - target[name]) for name in shared])
    )


def init_icp(source: dict, target: dict, picks, include_translation: bool) -> tuple:
    """Coarse alignment from three landmarks: one translation then two rotations.

    Returns `(transformed_source, matrix, steps)`.

    `include_translation` is the one real difference between the two engines,
    and it is deliberate on both sides. The CBCT engine leaves it out: both
    volumes have already been recentred on the physical origin, so re-adding a
    landmark-derived offset would push the scan back off centre -- which is
    also why it drops the ICP translation afterwards. The IOS engine keeps it:
    two intra-oral scans share no common origin.
    """
    first, second, third = picks
    steps = []
    matrix = np.eye(4)

    # ----- translation onto the first pick
    offset = target[first] - source[first]
    translation = np.eye(4)
    translation[:3, 3] = offset
    steps.append((STEP_TRANSLATION, offset))
    source = translate_landmarks(source, offset)
    if include_translation:
        matrix = translation

    # ----- rotation aligning the first->second direction
    angle, axis = angle_and_axis(
        np.absolute(target[second] - target[first]),
        np.absolute(source[second] - source[first]),
    )
    rotation = np.eye(4)
    rotation[:3, :3] = rotation_matrix(axis, angle)
    steps.append((STEP_ROTATION, rotation[:3, :3]))
    source = transform_landmarks(source, rotation)
    matrix = rotation @ matrix

    # ----- rotation aligning the first->third direction, about the first->second
    # axis so the alignment just obtained is not undone.
    angle, axis = angle_and_axis(
        np.absolute(target[third] - target[first]),
        np.absolute(source[third] - source[first]),
    )
    rotation = np.eye(4)
    rotation[:3, :3] = rotation_matrix(
        np.absolute(source[second] - source[first]), angle
    )
    steps.append((STEP_ROTATION, rotation[:3, :3]))
    source = transform_landmarks(source, rotation)
    matrix = rotation @ matrix

    return source, matrix, steps


def best_triplet(
    source: dict,
    target: dict,
    include_translation: bool,
    max_triplets: int,
    seed: int,
) -> tuple:
    """The three landmarks whose coarse alignment leaves the smallest mean
    distance, searched deterministically.

    The original drew triplets with the GLOBAL `np.random`, so the same request
    gave a different orientation each time it ran, and two concurrent requests
    consumed each other's random state. An orientation applied to patient data
    has to be reproducible, so:

    * every ordered triplet is evaluated when there are at most `max_triplets`
      of them -- which covers any realistic selection (7 landmarks is 210,
      14 is 2184) and is both faster and better than sampling;
    * above that the candidates are sampled from a LOCAL generator seeded from
      configuration, so a rerun repeats the same search.
    """
    labels = list(source)
    total = len(labels) * (len(labels) - 1) * (len(labels) - 2)

    if total <= max_triplets:
        candidates = itertools.permutations(labels, 3)
    else:
        rng = np.random.default_rng(seed)
        candidates = _sampled_triplets(labels, max_triplets, rng)

    best, best_distance = None, float("inf")
    for picks in candidates:
        transformed, _, _ = init_icp(source, target, picks, include_translation)
        distance = mean_distance(transformed, target)
        if distance < best_distance:
            best, best_distance = picks, distance

    if best is None:  # fewer than three landmarks; callers check first
        raise ValueError("Need at least three landmarks to search for a triplet")
    logger.debug("Best triplet leaves a mean distance of %.3f", best_distance)
    return best


def _sampled_triplets(labels: list, count: int, rng) -> list:
    seen, picks = set(), []
    while len(picks) < count:
        candidate = tuple(rng.choice(labels, size=3, replace=False))
        if candidate not in seen:
            seen.add(candidate)
            picks.append(candidate)
    return picks

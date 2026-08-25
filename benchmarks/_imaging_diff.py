#!/usr/bin/env python3
"""Voxelwise distance between two images, printed as JSON.

Executed by a TOOL's interpreter, never by the harness's own: the tools carry
SimpleITK and numpy in their virtualenvs, and the harness deliberately does not.
See artifacts.numeric_distance.

    <tool>/.venv/bin/python benchmarks/_imaging_diff.py LEFT RIGHT

Standard library plus SimpleITK and numpy, and nothing from this repository:
it has to import inside an environment that knows nothing about the server.
Exits non-zero, with the reason on stderr, when it cannot read a pair.
"""

import json
import sys


def main(argv):
    if len(argv) != 3:
        print("usage: _imaging_diff.py LEFT RIGHT", file=sys.stderr)
        return 2
    left_path, right_path = argv[1], argv[2]

    try:
        import numpy
        import SimpleITK
    except ImportError as error:
        print(f"this interpreter cannot read images: {error}", file=sys.stderr)
        return 3

    left = SimpleITK.ReadImage(left_path)
    right = SimpleITK.ReadImage(right_path)

    geometry = {
        "size_equal": left.GetSize() == right.GetSize(),
        "spacing_equal": _close(left.GetSpacing(), right.GetSpacing()),
        "origin_equal": _close(left.GetOrigin(), right.GetOrigin()),
        "direction_equal": _close(left.GetDirection(), right.GetDirection()),
        "left_size": list(left.GetSize()),
        "right_size": list(right.GetSize()),
        "left_pixel_type": left.GetPixelIDTypeAsString(),
        "right_pixel_type": right.GetPixelIDTypeAsString(),
    }

    if not geometry["size_equal"]:
        # No voxelwise comparison is possible, and resampling one onto the other
        # would compare an interpolation rather than the two outputs.
        print(json.dumps({"geometry": geometry, "voxelwise": None,
                          "reason": "the two images do not have the same size"}))
        return 0

    left_array = SimpleITK.GetArrayFromImage(left).astype("float64")
    right_array = SimpleITK.GetArrayFromImage(right).astype("float64")
    difference = numpy.abs(left_array - right_array)
    differing = int(numpy.count_nonzero(difference))

    print(json.dumps({
        "geometry": geometry,
        "voxelwise": {
            "voxels": int(difference.size),
            "differing_voxels": differing,
            "differing_fraction": differing / difference.size if difference.size else 0.0,
            "max_abs_difference": float(difference.max()) if difference.size else 0.0,
            "mean_abs_difference": float(difference.mean()) if difference.size else 0.0,
            "rms_difference": float(numpy.sqrt((difference ** 2).mean()))
            if difference.size else 0.0,
            "left_range": [float(left_array.min()), float(left_array.max())]
            if left_array.size else None,
            "right_range": [float(right_array.min()), float(right_array.max())]
            if right_array.size else None,
        },
    }))
    return 0


def _close(one, other, tolerance=1e-9):
    if len(one) != len(other):
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(one, other))


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Reading and writing Slicer markups files (.mrk.json), shared by both engines.

Ported from `ASO_CBCT_utils/utils.py` (LoadJsonLandmarks / GenControlePoint /
WriteJson / WriteJsonLandmarks) and its near-duplicate in
`ASO_IOS_utils/utils.py`. The two copies had drifted -- the CBCT one skipped a
control point with a short `position`, the IOS one raised; the CBCT one wrote
`.mrk.json`, the IOS one wrote `.json` for identical content. One
implementation, one extension.
"""

import json
import os

import numpy as np

# The point arrays a segmented intra-oral scan may carry, most specific first.
# Kept next to the markups helpers because both answer "what does this file
# claim about teeth".
LABEL_ARRAY_NAMES = ("Universal_ID", "PredictedID", "UniversalID")

# Slicer writes a markups file under one of these two names for the same
# content. Both are read; only the first is ever written.
MARKUPS_EXTENSIONS = (".mrk.json", ".json")

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/"
    "Markups/Resources/Schema/markups-schema-v1.0.0.json#"
)


def is_markups_file(path: str) -> bool:
    return path.lower().endswith(MARKUPS_EXTENSIONS)


def load_landmarks(path: str, keep=None) -> dict:
    """{landmark label: np.array([x, y, z])} from a Slicer markups file.

    `keep` restricts the result to those labels, in the file's own order; a
    label in `keep` that the file does not have is simply absent, never a
    KeyError. That is the difference that matters: the original built the
    restricted dict with `{key: landmarks[key] for key in ldmk_list}` in one
    place and tolerated the miss in another, so which one you hit decided
    whether a patient survived.
    """
    with open(path) as handle:
        data = json.load(handle)

    try:
        control_points = data["markups"][0]["controlPoints"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"{os.path.basename(path)} is not a Slicer markups file") from exc

    landmarks = {}
    for point in control_points:
        position = point.get("position")
        label = point.get("label")
        if not label or position is None or len(position) < 3:
            continue
        landmarks[label] = np.array(position[:3], dtype=np.float64)

    if keep is not None:
        wanted = set(keep)
        return {name: value for name, value in landmarks.items() if name in wanted}
    return landmarks


def write_landmarks(landmarks: dict, output_path: str) -> str:
    """Write a markups file from scratch. Returns the path written."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    document = {
        "@schema": _SCHEMA_URL,
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": False,
                "labelFormat": "%N-%d",
                "controlPoints": _control_points(landmarks),
                "measurements": [],
                "display": _display_settings(),
            }
        ],
    }
    with open(output_path, "w") as handle:
        json.dump(document, handle, indent=4)
    return output_path


def rewrite_landmarks(landmarks: dict, template_path: str, output_path: str) -> str:
    """Write a markups file keeping the input file's own display settings.

    Used when the caller supplied the landmarks: their colours, glyph sizes and
    any extra fields are theirs to keep. Only the positions change, matched by
    LABEL -- the original matched by INDEX, so any reordering (or a landmark
    dropped as an outlier) moved the coordinates onto the wrong points.
    """
    with open(template_path) as handle:
        document = json.load(handle)

    kept = []
    for point in document.get("markups", [{}])[0].get("controlPoints", []):
        position = landmarks.get(point.get("label"))
        if position is None:
            continue  # dropped upstream (outlier, or absent from the reference)
        point["position"] = [float(position[0]), float(position[1]), float(position[2])]
        kept.append(point)
    document["markups"][0]["controlPoints"] = kept

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(document, handle, indent=4)
    return output_path


def _control_points(landmarks: dict) -> list:
    points = []
    for index, (label, position) in enumerate(landmarks.items(), start=1):
        points.append(
            {
                "id": str(index),
                "label": label,
                "description": "",
                "associatedNodeID": "",
                "position": [float(position[0]), float(position[1]), float(position[2])],
                "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "selected": True,
                "locked": True,
                "visibility": True,
                "positionStatus": "defined",
            }
        )
    return points


def _display_settings() -> dict:
    return {
        "visibility": False,
        "opacity": 1.0,
        "color": [0.5, 0.5, 0.5],
        "selectedColor": [0.2666666666666667, 0.6745098039215687, 0.39215686274509806],
        "propertiesLabelVisibility": False,
        "pointLabelsVisibility": True,
        "textScale": 2.0,
        "glyphType": "Sphere3D",
        "glyphScale": 2.0,
        "glyphSize": 5.0,
        "useGlyphScale": True,
        "sliceProjection": False,
        "sliceProjectionUseFiducialColor": True,
        "sliceProjectionOutlinedBehindSlicePlane": False,
        "sliceProjectionColor": [1.0, 1.0, 1.0],
        "sliceProjectionOpacity": 0.6,
        "lineThickness": 0.2,
        "lineColorFadingStart": 1.0,
        "lineColorFadingEnd": 10.0,
        "lineColorFadingSaturation": 1.0,
        "lineColorFadingHueOffset": 0.0,
        "handlesInteractive": False,
        "snapMode": "toVisibleSurface",
    }

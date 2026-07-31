"""Writing Slicer markups files (`.mrk.json`).

Shared by ALI's two engines, which had a copy each and disagreed: the CBCT CLI
wrote `.mrk.json`, the IOS one wrote `.json` for byte-identical content --
and Slicer only associates the first with a markups node, so half of ALI's
output had to be imported by hand. One writer, one extension.

Lives beside `ALI_CBCT/` and `ALI_IOS/` rather than in a tool of its own
because no other tool needs it yet. The day one does (ASO and AREG both read
and write these files), this is the module to promote.
"""

import json
import os

# Slicer's own schema URL. Written verbatim; a markups file without it loads
# but is not recognized by version-aware readers.
_SCHEMA = (
    "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/"
    "Markups/Resources/Schema/markups-schema-v1.0.0.json#"
)

# ALI's two engines both produce coordinates in LPS, and both original CLIs
# declared LPS. Stated here once so a future engine cannot quietly write RAS
# into a file that claims otherwise.
COORDINATE_SYSTEM = "LPS"

MARKUPS_EXTENSION = ".mrk.json"

_DISPLAY = {
    # TRUE, and this is the one value in this block that is not cosmetic.
    # Both original CLIs wrote `false` here, which switches the markups
    # DISPLAY node off: Slicer loads the file, builds the node, lists it in the
    # Markups module -- and draws nothing. Inside the old Slicer module that
    # went unnoticed, because the module loaded the nodes itself and the panel
    # could toggle them back on. Opening a result file directly, which is what
    # anyone does with a returned archive, showed an empty scene.
    #
    # Note this is independent of each control point's own `visibility`, which
    # was already true: a point can be visible in a node that is not displayed.
    "visibility": True,
    "opacity": 1.0,
    "color": [0.5, 0.5, 0.5],
    "selectedColor": [0.26666666666666669, 0.6745098039215687, 0.39215686274509806],
    "propertiesLabelVisibility": False,
    "pointLabelsVisibility": True,
    "textScale": 2.0,
    "glyphType": "Sphere3D",
    "glyphScale": 2.0,
    "glyphSize": 5.0,
    "useGlyphScale": True,
    # FALSE: a landmark is drawn only on the slice it actually sits on.
    # Turning it on was tried and rejected -- with 113 points, projecting each
    # one onto its neighbouring slices crowds the very view used to judge
    # placement, and a point that shows everywhere is a point you can no longer
    # locate. Blinking in and out as you scroll IS the signal here.
    #
    # Unlike `visibility` above, this one is a preference: it is a checkbox in
    # the Markups module's Display section, changeable per node without
    # re-running anything.
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


def control_points(landmarks: dict) -> list:
    """One markups control point per landmark, from {label: (x, y, z)}.

    Coordinates are cast to float: they arrive as numpy scalars, which
    `json.dump` cannot serialize -- and the failure would land at the very end
    of a run, after all the inference.
    """
    return [
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
        for index, (label, position) in enumerate(landmarks.items(), start=1)
    ]


def write(landmarks: dict, output_path: str) -> str:
    """Write {label: (x, y, z)} as a Slicer markups file; return its path."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    content = {
        "@schema": _SCHEMA,
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": COORDINATE_SYSTEM,
                "locked": False,
                "labelFormat": "%N-%d",
                "controlPoints": control_points(landmarks),
                "measurements": [],
                "display": _DISPLAY,
            }
        ],
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=4)
    return output_path

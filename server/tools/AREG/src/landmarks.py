"""Reading Slicer markups files, and matching one to the scan it belongs to.

AREG only ever READS landmarks -- the mucogingival ones ALI predicts for the
lower arch (see `ios/mgl.py`) -- so this is a reader, not the full markups
module ASO and ALI each carry. Same reason those two are separate: importing
another tool's module at load time makes one tool's missing dependency take
both out of the registry.

The matching is the part worth reading. Upstream's `FindLandmarkFile` tried
`<scan stem>_Lower_MG_Pred.json` and then fell back to *any* json whose name
merely CONTAINS the scan's stem, taking `sorted(...)[0]` of the result with a
warning when several matched. That substring test is the one this codebase has
already had to fix twice (`vtk_name in json_name` paired patient `1` with
patient `10`; `"cb" in basename` made every CBCT a cranial base): here it would
hand patient `P1`'s landmarks to patient `P10` whenever `P1`'s own file is
missing, and register a mandible against another patient's mucogingival line
while reporting success. Files are matched through `pairing.patient_stem`, the
same rule that paired the scans, and an ambiguity is an error naming the
candidates rather than a warning and a guess.
"""

import json
import os
import re

import numpy as np

from . import pairing

MARKUPS_EXTENSIONS = (".mrk.json", ".json")


class LandmarkError(Exception):
    """No usable landmark file for this scan."""


def is_markups_file(filename: str) -> bool:
    return filename.lower().endswith(MARKUPS_EXTENSIONS)


def load(path: str) -> dict:
    """{landmark label: np.array([x, y, z])} from a Slicer markups file."""
    with open(path) as handle:
        document = json.load(handle)

    try:
        control_points = document["markups"][0]["controlPoints"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LandmarkError(
            f"'{os.path.basename(path)}' is not a Slicer markups file"
        ) from exc

    found = {}
    for point in control_points:
        label = point.get("label")
        position = point.get("position")
        if not label or position is None or len(position) < 3:
            continue
        found[label] = np.array(position[:3], dtype=np.float64)
    if not found:
        raise LandmarkError(f"'{os.path.basename(path)}' holds no usable control point")
    return found


# Tokens a landmark file's name carries that its scan's does not:
# `P1_T1_Lower_MG_Pred.json` has to reduce to what `P1_T1_Lower.vtk` reduces to.
_LANDMARK_TOKENS = ("mg", "pred", "lm", "landmarks")


def scan_key(filename: str, also_drop=()) -> str:
    """The key identifying ONE SCAN -- patient AND timepoint.

    Deliberately not the patient key the scans are paired on. There is one
    landmark file per scan, and both timepoints' files routinely sit in the same
    folder (upstream points `lm_T1` and `lm_T2` at one directory), so a key that
    collapsed `P1_T1_...` and `P1_T2_...` would make a patient's own two files
    indistinguishable -- which is exactly what it did before an end-to-end run
    showed both matching and neither being usable.
    """
    return pairing.patient_stem(
        filename,
        also_drop=set(also_drop) | set(_LANDMARK_TOKENS),
        drop_timepoint=False,
    )


def index(root: str, also_drop=()) -> dict:
    """{scan key: [markups paths]} for every landmark file under `root`.

    Keyed through the same name rules as the scans, so a landmark file finds its
    scan whatever the folder layout -- one flat directory holding both
    timepoints, or a subfolder per timepoint, which are the two shapes upstream
    produces and users send.
    """
    found: dict = {}
    for directory, _, file_names in os.walk(root):
        for file_name in sorted(file_names):
            if file_name.startswith(".") or not is_markups_file(file_name):
                continue
            found.setdefault(scan_key(file_name, also_drop), []).append(
                os.path.join(directory, file_name)
            )
    return found


def _tokens(key: str) -> tuple:
    return tuple(part for part in re.split(r"[_\-.\s]+", key.lower()) if part)


def for_scan(indexed: dict, scan_path: str, also_drop=()) -> str:
    """The one landmark file belonging to the scan at `scan_path`.

    Two rules, in order: the keys are equal, or **the scan's tokens are a
    PREFIX of the landmark file's**. The second is what makes real names work.
    People name a landmark file after the scan and append something --
    `H10_T1_L_MG_edited.mrk.json` beside `H10_T1_L.vtk`, or `_corrected`,
    `_v2`, `_manual`, their initials -- and a rule that only strips a fixed
    list of known words (`mg`, `pred`, `lm`) can never cover what people
    actually type. It rejected the caller's own hand-edited files.

    Whole tokens, never a substring, which is the whole reason not to do what
    upstream's `FindLandmarkFile` did: it fell back to any json whose name
    merely CONTAINS the scan's stem, so with `P1`'s file missing `P1` took
    `P10`'s and registered a mandible against another patient's mucogingival
    line while reporting success. Comparing `('p1',)` against `('p10', ...)`
    cannot do that.

    An ambiguity is an error naming the candidates, never a pick.
    """
    key = scan_key(os.path.basename(scan_path), also_drop)
    candidates = indexed.get(key)

    if candidates is None:
        wanted = _tokens(key)
        extended = {
            other: paths
            for other, paths in indexed.items()
            if _tokens(other)[: len(wanted)] == wanted
        }
        if len(extended) == 1:
            candidates = next(iter(extended.values()))
        elif len(extended) > 1:
            raise LandmarkError(
                f"{len(extended)} landmark files could belong to "
                f"'{os.path.basename(scan_path)}' ({', '.join(sorted(extended))}): their "
                f"names all start with '{key}' and nothing says which one is this scan's"
            )

    if not candidates:
        raise LandmarkError(
            f"no landmark file for '{os.path.basename(scan_path)}'. They are matched to "
            f"their scan by name: '{key}' followed by anything at all, so "
            f"'{key}_MG_Pred.json' and '{key}_L_MG_edited.mrk.json' both work. Found: "
            f"{', '.join(sorted(indexed)) or 'none'}"
        )
    if len(candidates) > 1:
        raise LandmarkError(
            f"{len(candidates)} landmark files match '{os.path.basename(scan_path)}' "
            f"({', '.join(sorted(os.path.basename(path) for path in candidates))}): "
            f"leave one per scan so which points registered it is not a coin toss"
        )
    return candidates[0]

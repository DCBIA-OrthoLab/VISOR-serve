"""What two runs produced, compared file by file.

B5's claim is parity: the artifacts a tool produces through the API are the
artifacts it produces without one. This module is how that claim is checked, and
it is deliberately unforgiving -- a difference is NAMED, never softened.

Three levels, applied in order, and every level that applies is reported:

1. **Bytes.** sha256 per file, keyed by path RELATIVE to each side's output
   root: absolute paths differ by construction (one is a job directory inside a
   container, the other a download folder) and comparing them would report a
   difference in every file.
2. **Structure.** For a text or JSON artifact that differs, which keys or which
   lines. A run report holding a timestamp differs from itself, and saying so
   is more useful than a hash mismatch.
3. **Numbers.** For an imaging artifact that differs, a numeric distance --
   max, mean and RMS of the voxelwise difference, plus how many voxels moved at
   all, plus whether the geometry (shape, spacing, origin, direction) is
   identical.

Level 3 needs a reader for medical image formats, which this harness
deliberately does not depend on: the tools already carry SimpleITK in their own
virtualenvs, so the comparison is executed BY one of those interpreters, exactly
the way a tool is. When no such interpreter is configured, level 3 is reported
as unavailable with the reason -- never quietly skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

_READ_CHUNK = 1024 * 1024

# What the imaging comparison can read, when an interpreter is available.
IMAGE_EXTENSIONS = (".nii", ".nii.gz", ".nrrd", ".nrrd.gz", ".gipl", ".gipl.gz", ".mha", ".mhd")
TEXT_EXTENSIONS = (".txt", ".csv", ".log", ".md", ".mrk", ".fcsv")
JSON_EXTENSIONS = (".json", ".mrk.json")

_DIFF_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_imaging_diff.py")
_DIFF_TIMEOUT_SECONDS = 900.0


def digest(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_READ_CHUNK)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def snapshot(root: str) -> dict:
    """{relative path: {sha256, size}} for every file under `root`.

    Empty for a root that does not exist, which is itself a finding the caller
    records rather than an error: "the local run produced nothing" is a parity
    result.
    """
    found: dict = {}
    if not os.path.isdir(root):
        if os.path.isfile(root):
            found[os.path.basename(root)] = {
                "sha256": digest(root), "size": os.path.getsize(root)
            }
        return found
    for directory, _subdirs, names in os.walk(root):
        for name in sorted(names):
            absolute = os.path.join(directory, name)
            relative = os.path.relpath(absolute, root)
            try:
                found[relative] = {"sha256": digest(absolute), "size": os.path.getsize(absolute)}
            except OSError as error:
                found[relative] = {"sha256": None, "size": None, "unreadable": str(error)}
    return found


@dataclass
class ParityReport:
    identical: list = field(default_factory=list)
    differing: list = field(default_factory=list)
    only_left: list = field(default_factory=list)
    only_right: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.differing or self.only_left or self.only_right)

    def as_dict(self) -> dict:
        return {
            "identical": self.identical,
            "differing": self.differing,
            "only_left": self.only_left,
            "only_right": self.only_right,
            "details": self.details,
            "ok": self.ok,
            "identical_count": len(self.identical),
            "differing_count": len(self.differing),
        }


def compare(left: dict, right: dict) -> ParityReport:
    """Two snapshots, by relative name. `left` is the local side by convention."""
    report = ParityReport()
    for name in sorted(left):
        if name not in right:
            report.only_left.append(name)
        elif left[name]["sha256"] == right[name]["sha256"]:
            report.identical.append(name)
        else:
            report.differing.append(name)
    report.only_right = sorted(set(right) - set(left))
    return report


# ----------------------------------------------------------------------
# Saying WHAT differs
# ----------------------------------------------------------------------

def byte_difference(left_path: str, right_path: str, limit: int = 64 * 1024 * 1024) -> dict:
    """Where two files first differ, and how much of them does.

    Reads up to `limit` bytes of each. Over that it reports the sizes and the
    first differing offset only, because holding two 200 MB volumes in memory to
    count differing bytes is not worth what it tells you.
    """
    left_size = os.path.getsize(left_path)
    right_size = os.path.getsize(right_path)
    summary = {
        "left_size": left_size,
        "right_size": right_size,
        "same_size": left_size == right_size,
        "first_differing_offset": None,
        "differing_bytes": None,
        "compared_bytes": 0,
        "truncated": max(left_size, right_size) > limit,
    }
    differing = 0
    offset = 0
    with open(left_path, "rb") as left, open(right_path, "rb") as right:
        while offset < limit:
            left_block = left.read(_READ_CHUNK)
            right_block = right.read(_READ_CHUNK)
            if not left_block and not right_block:
                break
            span = min(len(left_block), len(right_block))
            for index in range(span):
                if left_block[index] != right_block[index]:
                    if summary["first_differing_offset"] is None:
                        summary["first_differing_offset"] = offset + index
                    differing += 1
            if len(left_block) != len(right_block) and summary["first_differing_offset"] is None:
                summary["first_differing_offset"] = offset + span
            offset += span
            summary["compared_bytes"] = offset
            if not left_block or not right_block:
                break
    if not summary["truncated"]:
        summary["differing_bytes"] = differing
        summary["differing_fraction"] = (
            differing / offset if offset else 0.0
        )
    return summary


def json_difference(left_path: str, right_path: str) -> dict:
    """Which KEYS differ between two JSON artifacts, not which bytes.

    A landmark file whose only difference is a `timestamp` field is a different
    finding from one whose coordinates moved, and a byte diff cannot tell them
    apart.
    """
    try:
        with open(left_path, encoding="utf-8") as handle:
            left = json.load(handle)
        with open(right_path, encoding="utf-8") as handle:
            right = json.load(handle)
    except (OSError, ValueError) as error:
        return {"readable": False, "reason": str(error)}
    return {"readable": True, "differing_keys": sorted(_differing_keys(left, right, ""))}


def _differing_keys(left, right, prefix: str) -> set:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                keys.add(path)
            else:
                keys |= _differing_keys(left[key], right[key], path)
        return keys
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {f"{prefix}[length]"}
        keys = set()
        for index, (one, other) in enumerate(zip(left, right)):
            keys |= _differing_keys(one, other, f"{prefix}[{index}]")
        return keys
    return set() if left == right else {prefix or "<root>"}


def text_difference(left_path: str, right_path: str, max_lines: int = 40) -> dict:
    try:
        with open(left_path, encoding="utf-8", errors="replace") as handle:
            left = handle.read().splitlines()
        with open(right_path, encoding="utf-8", errors="replace") as handle:
            right = handle.read().splitlines()
    except OSError as error:
        return {"readable": False, "reason": str(error)}
    changed = []
    for number, (one, other) in enumerate(zip(left, right), start=1):
        if one != other:
            changed.append({"line": number, "left": one[:200], "right": other[:200]})
            if len(changed) >= max_lines:
                break
    return {
        "readable": True,
        "left_lines": len(left),
        "right_lines": len(right),
        "changed_lines": changed,
        "truncated": len(changed) >= max_lines,
    }


def numeric_distance(left_path: str, right_path: str, interpreter: Optional[str]) -> dict:
    """Voxelwise distance between two images, computed by a TOOL's interpreter.

    The harness itself carries no SimpleITK: adding one would mean pinning an
    imaging stack in a benchmark harness, and whichever one was chosen would not
    be the one that wrote the file. So the reader used is the tool's own.
    """
    if not interpreter:
        return {
            "available": False,
            "reason": "no interpreter configured for imaging comparison "
                      "(campaigns.b5.imaging_interpreter)",
        }
    if not os.path.exists(interpreter):
        return {"available": False, "reason": f"interpreter does not exist: {interpreter}"}
    try:
        completed = subprocess.run(
            [interpreter, _DIFF_SCRIPT, left_path, right_path],
            capture_output=True,
            text=True,
            timeout=_DIFF_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": f"comparison exited {completed.returncode}: {completed.stderr[-600:]}",
        }
    try:
        return dict(json.loads(completed.stdout), available=True)
    except ValueError:
        return {"available": False, "reason": f"unreadable output: {completed.stdout[:300]}"}


def rename_substitutions(files: dict, chunked_arguments) -> dict:
    """`{local stem: remote stem}`, from the SERVER's own staging rule.

    A file that travels does not keep its name. `server/main.py` stages an
    uploaded input under the ARGUMENT it was sent as, and by two different
    rules depending on how it travelled:

        multipart (small, inline)   ->  "<argument>_<stem><extension>"
        chunked (>= min_chunked)    ->  "<argument><extension>"

    Every tool here names its outputs after its input, so the remote side's
    artifacts are spelled differently from the local side's for reasons that
    have nothing to do with what the tool computed. This function reconstructs
    exactly that substitution -- from the argument names in `config.yaml` and
    the harness's own record of which arguments were chunked -- so the pairing
    below is a rule, not a guess. A DIRECTORY input is excluded: it travels as a
    zip and the server unpacks it, so the members keep their names.
    """
    chunked = set(chunked_arguments or ())
    substitutions = {}
    for argument, path in (files or {}).items():
        if os.path.isdir(path):
            continue
        local_stem = _stem(os.path.basename(path.rstrip(os.sep)))
        remote_stem = argument if argument in chunked else f"{argument}_{local_stem}"
        if remote_stem != local_stem:
            substitutions[local_stem] = remote_stem
    return substitutions


def _stem(name: str) -> str:
    """`scan.nii.gz` -> `scan`. Compound extensions are not split twice."""
    lowered = name.lower()
    for extension in (".nii.gz", ".nrrd.gz", ".gipl.gz", ".tar.gz"):
        if lowered.endswith(extension):
            return name[: -len(extension)]
    return os.path.splitext(name)[0]


def pair_renamed(
    report: "ParityReport",
    substitutions: dict,
    left_root: str,
    right_root: str,
    imaging_interpreter: Optional[str] = None,
) -> dict:
    """Compare the files that exist on both sides under DIFFERENT names.

    The strict comparison above stays strict: a file whose name differs is
    reported as `only_left` / `only_right` and stays reported that way, because
    a differently-named artifact IS a difference and the paper has to say so.
    This is a SECOND, separately labelled pass answering the other question a
    reviewer asks immediately afterwards -- are the CONTENTS the same? -- and it
    pairs names only by the server's own documented staging rule.
    """
    outcome = {
        "substitutions": dict(substitutions),
        "pairs": [],
        "unpaired_left": [],
        "unpaired_right": list(report.only_right),
    }
    if not substitutions:
        outcome["unpaired_left"] = list(report.only_left)
        outcome["content_ok"] = not (
            report.differing or report.only_left or report.only_right
        )
        return outcome

    remaining_right = set(report.only_right)
    for name in report.only_left:
        candidate = name
        for local_stem, remote_stem in substitutions.items():
            candidate = candidate.replace(local_stem, remote_stem)
        if candidate == name or candidate not in remaining_right:
            outcome["unpaired_left"].append(name)
            continue
        remaining_right.discard(candidate)
        left_path = os.path.join(left_root, name)
        right_path = os.path.join(right_root, candidate)
        identical = digest(left_path) == digest(right_path)
        pair = {"local": name, "remote": candidate, "identical": identical}
        if not identical:
            pair["detail"] = describe_difference_paths(
                left_path, right_path, name, imaging_interpreter
            )
        outcome["pairs"].append(pair)
    outcome["unpaired_right"] = sorted(remaining_right)
    outcome["content_ok"] = (
        not report.differing
        and not outcome["unpaired_left"]
        and not outcome["unpaired_right"]
        and all(pair["identical"] for pair in outcome["pairs"])
    )
    return outcome


def describe_difference(
    name: str, left_root: str, right_root: str, imaging_interpreter: Optional[str] = None
) -> dict:
    """Everything that can be said about ONE differing artifact."""
    return describe_difference_paths(
        os.path.join(left_root, name),
        os.path.join(right_root, name),
        name,
        imaging_interpreter,
    )


def describe_difference_paths(
    left_path: str, right_path: str, name: str, imaging_interpreter: Optional[str] = None
) -> dict:
    """The same, for two files that do not share a name. `name` chooses the
    reader: it is the artifact's own spelling, and the two sides' extensions
    agree even when their stems do not."""
    lowered = name.lower()
    detail = {"bytes": byte_difference(left_path, right_path)}
    if lowered.endswith(JSON_EXTENSIONS):
        detail["json"] = json_difference(left_path, right_path)
    elif lowered.endswith(TEXT_EXTENSIONS):
        detail["text"] = text_difference(left_path, right_path)
    if lowered.endswith(IMAGE_EXTENSIONS):
        detail["numeric"] = numeric_distance(left_path, right_path, imaging_interpreter)
    return detail

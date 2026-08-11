"""Pairing a T1 folder with a T2 folder, and finding the masks that go with T1.

AREG's whole input is two timepoints of the same subjects, sent as two folders.
Everything downstream keys off the pairing this module produces, so it is one
implementation shared by both engines -- `AREG_CBCT_utils.GetPatients` and the
near-identical copy inside `AREG_Method/CBCT.py` had already drifted into two
different signatures with two different mask rules.

Three defects of those copies are fixed here by construction:

* **the patient key was a base name**, so `scanT1.nii.gz` in two different
  subfolders became one patient -- in the working dict and again in the flat
  output folder. The key is the path RELATIVE to the input root, so two
  subjects with the same file name stay apart and the output mirrors the input
  tree;
* **`.split(".")[0]`** truncated a name at its first dot, so `P1.2_scan.nii.gz`
  became patient `P1`. Extensions are split off properly, compound ones
  (`.nii.gz`) included;
* **mask regions were matched as substrings**: `"cb" in basename.lower()` makes
  every file whose name contains **CBCT** a cranial-base mask, `"max"` matches
  a patient named MAX_01 and `"md"` matches almost anything. Matching is on
  whole tokens of the stem.
"""

import os
import re

from . import catalogs

SCAN_EXTENSIONS = (".nii.gz", ".nrrd.gz", ".gipl.gz", ".nii", ".nrrd", ".gipl")

# See AMASSSLogic.compressed_extension: NIfTI and GIPL take an external .gz,
# NRRD compresses inside the file and ITK has no ".nrrd.gz" writer at all.
_COMPRESSED_EXTENSIONS = {".nii": ".nii.gz", ".gipl": ".gipl.gz", ".nrrd.gz": ".nrrd"}

_SEPARATORS = re.compile(r"([_\-.\s]+)")


def split_scan_extension(filename: str) -> tuple:
    """('scan.nii.gz') -> ('scan', '.nii.gz'), compound extensions preserved."""
    lower = filename.lower()
    for extension in SCAN_EXTENSIONS:
        if lower.endswith(extension):
            return filename[: -len(extension)], filename[-len(extension):]
    return os.path.splitext(filename)


def compressed_extension(extension: str) -> str:
    return _COMPRESSED_EXTENSIONS.get(extension.lower(), extension)


def is_scan_file(filename: str) -> bool:
    return filename.lower().endswith(SCAN_EXTENSIONS)


def tokens(stem: str) -> tuple:
    """The lowercase words of a file stem, split on _ - . and whitespace.

    Token matching rather than substring matching is the whole point: see this
    module's docstring for what `"cb" in "P1_CBCT_seg"` used to cost.
    """
    return tuple(
        part.lower()
        for part in _SEPARATORS.split(stem)
        if part and not _SEPARATORS.fullmatch(part)
    )


def has_token(stem: str, wanted) -> bool:
    present = set(tokens(stem))
    return any(token in present for token in wanted)


def _drop_tokens(stem: str, unwanted) -> str:
    """`stem` without the given tokens, separators collapsed.

    Case is preserved for what survives: the key ends up in output paths, and a
    patient folder should keep the name its owner gave it.
    """
    parts = _SEPARATORS.split(stem)
    kept = [
        part
        for part in parts
        if not (part and not _SEPARATORS.fullmatch(part) and part.lower() in unwanted)
    ]
    return re.sub(r"[_\-.\s]+", "_", "".join(kept)).strip("_-. ")


def patient_stem(filename: str, also_drop=(), drop_timepoint: bool = True) -> str:
    """The subject a file belongs to, from its name alone.

    Strips the extension, any suffix a previous ASO/AMASSS/ALI/AREG run left,
    and the timepoint token -- which is what pairs `P1_T1_scan.nii.gz` in the
    T1 folder with `P1_T2.nii.gz` in the T2 folder. `also_drop` adds tokens to
    remove, used for masks so `P1_T1_MAND_seg.nii.gz` keys to the same patient
    as the scan it belongs to.

    Unlike ASO's equivalent, the timepoint token IS stripped here, and for the
    opposite reason: there both timepoints of a subject sit in one folder and
    are two separate scans to orient, so collapsing them lost one. Here they
    arrive in two folders and collapsing them is the pairing.

    `drop_timepoint=False` keeps it, which identifies ONE SCAN rather than a
    subject. That is what the mucogingival landmarks need: there is one landmark
    file per scan, both timepoints' files routinely sit in one folder, and
    collapsing them would make a patient's two files indistinguishable.
    """
    stem, extension = split_scan_extension(filename)
    if extension.lower() not in SCAN_EXTENSIONS:
        stem = os.path.splitext(filename)[0]
        # .mrk.json and friends: strip the inner extension too.
        if stem.lower().endswith(".mrk"):
            stem = stem[: -len(".mrk")]

    for suffix in catalogs.PATIENT_SUFFIXES:
        index = stem.find(suffix)
        if index > 0:
            stem = stem[:index]

    unwanted = set(also_drop)
    if drop_timepoint:
        unwanted |= set(catalogs.TIMEPOINT_TOKENS)
    return _drop_tokens(stem, unwanted)


def is_previous_output(filename: str, suffix: str) -> bool:
    """True if this file looks like something a previous AREG run wrote.

    Running twice on the same folder must not register an already-registered
    scan: `P1_CB_Reg.nii.gz` sorts before `P1_scan.nii.gz`, so without this the
    second run would silently take the first run's output as its input.

    Matched on a whole trailing token, not with `suffix in stem`: with the
    default suffix "Reg" that substring test would also exclude a patient
    called `Regina`.
    """
    if not suffix:
        return False
    stem, _ = split_scan_extension(filename)
    stem = stem[: -len("_transform")] if stem.endswith("_transform") else stem
    return stem.endswith(f"_{suffix}")


def discover(root: str, suffix: str, accept=is_scan_file) -> dict:
    """{patient key: path} for one timepoint folder.

    The key is `<relative directory>/<patient stem>`. A patient with several
    matching files in one directory keeps the first in sorted order; files a
    previous run wrote are set aside and used only when a patient has nothing
    else, so re-running on an output folder still works while a folder holding
    both an original and its registration uses the original.
    """
    fresh: dict = {}
    previous: dict = {}
    for directory, _, file_names in os.walk(root):
        relative = os.path.relpath(directory, root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if file_name.startswith(".") or not accept(file_name):
                continue
            key = os.path.join(prefix, patient_stem(file_name))
            target = previous if is_previous_output(file_name, suffix) else fresh
            target.setdefault(key, os.path.join(directory, file_name))

    for key, path in previous.items():
        fresh.setdefault(key, path)
    return fresh


class Pairing:
    """What `pair()` found: the matched subjects, and what was left over.

    The leftovers are carried rather than dropped because "34 of your 40
    patients were registered" and "the other 6 are in the T2 folder under names
    nothing in T1 matched" are the same sentence, and only the second half is
    actionable. The original logged `Error: There is no patient to process.
    Check the files names.` and returned nothing.
    """

    def __init__(self, matched: dict, t1_only: list, t2_only: list):
        self.matched = matched
        self.t1_only = t1_only
        self.t2_only = t2_only

    def __len__(self) -> int:
        return len(self.matched)

    def unmatched_report(self) -> dict:
        return {"t1_without_t2": self.t1_only, "t2_without_t1": self.t2_only}


def pair(t1_root: str, t2_root: str, suffix: str, accept=is_scan_file) -> Pairing:
    """Match the subjects of two timepoint folders by name."""
    t1 = discover(t1_root, suffix, accept)
    t2 = discover(t2_root, suffix, accept)

    matched = {
        key: {"t1": t1[key], "t2": t2[key]} for key in sorted(set(t1) & set(t2))
    }
    return Pairing(
        matched=matched,
        t1_only=sorted(set(t1) - set(t2)),
        t2_only=sorted(set(t2) - set(t1)),
    )


def discover_masks(root: str, region: str) -> dict:
    """{patient key: mask path} for one registration region.

    A file counts as a mask for `region` when its stem carries BOTH a
    segmentation token (mask/seg/pred) and one of that region's own tokens --
    or when it is the only candidate in a folder that holds nothing but masks,
    which is what AMASSS's own output looks like (`P1_seg_CBMASK.nii.gz`).
    """
    found: dict = {}
    wanted = catalogs.REGION_TOKENS[region]
    # Every region's tokens, so the patient key of `P1_MAND_seg.nii.gz` is the
    # `P1` its scan keys to rather than `P1_MAND`.
    anatomy = {token for group in catalogs.REGION_TOKENS.values() for token in group}
    anatomy |= set(catalogs.MASK_TOKENS)

    for directory, _, file_names in os.walk(root):
        relative = os.path.relpath(directory, root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if file_name.startswith(".") or not is_scan_file(file_name):
                continue
            stem, _ = split_scan_extension(file_name)
            if not has_token(stem, catalogs.MASK_TOKENS):
                continue
            if not has_token(stem, wanted):
                continue
            key = os.path.join(prefix, patient_stem(file_name, also_drop=anatomy))
            found.setdefault(key, os.path.join(directory, file_name))
    return found

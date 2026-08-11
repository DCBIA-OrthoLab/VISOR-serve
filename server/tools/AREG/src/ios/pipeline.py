"""The IOS half of AREG: pair two timepoints of intra-oral scans, paint the
stable region on each, align them, move the whole model with it.

Ported from `AREG_IOS/AREG_IOS.py` (the driver, both of its modes) and the
`Sort`/`SortLower` half of `AREG_IOS_utils/dataset.py`.

**Which region is stable depends on the arch, and that is the whole design.**
The maxilla has the palate -- a plateau orthodontic treatment does not move,
painted by a network (`butterfly.py`). The mandible has none, but it has the
mucogingival line, and the band around it does the same job (`mgl.py`). Picking
the patch therefore also picks which arch is registered:

* **palate** -- the upper arches are registered and the lower ones are carried
  along by the upper's transform, so the occlusion the pair was captured in
  survives;
* **mucogingival line** -- the lower arches are registered on their own. The
  upper arches are left untouched, as upstream leaves them: the MG model covers
  the mandible only, and a maxilla is not rigidly attached to it.

The pairing is rewritten rather than transcribed. `Sort` matched two files by
`os.path.basename(name).replace("T1", "")` -- so it paired on base names alone
(two subjects called `Upper.vtk` in different folders became one), a patient
identifier containing the literal `T1` was mangled, and its jaw split defaulted
every file that did not say "Upper" to the LOWER arch. Here both timepoints go
through `pairing.patient_stem`, and a mesh whose name does not say its jaw is
reported rather than guessed.
"""

import logging
import os

from .. import catalogs, landmarks as landmark_files, pairing
from . import butterfly, icp, mgl, surfaces

logger = logging.getLogger("AREG")


def discover(root: str, suffix: str) -> dict:
    """{patient key: {jaw: path}} for one timepoint folder.

    Files whose name does not name a jaw are collected under `None` so the
    caller can report them; they are never assigned to an arch.
    """
    found: dict = {}
    jaw_tokens = set(catalogs.JAW_TOKENS)

    for directory, _, file_names in os.walk(root):
        relative = os.path.relpath(directory, root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if file_name.startswith(".") or not surfaces.is_surface_file(file_name):
                continue
            if pairing.is_previous_output(file_name, suffix):
                continue
            key = os.path.join(prefix, pairing.patient_stem(file_name, also_drop=jaw_tokens))
            found.setdefault(key, {}).setdefault(
                surfaces.jaw_of(file_name), os.path.join(directory, file_name)
            )
    return found


class Pairing:
    """Matched subjects plus everything that could not be matched, per reason."""

    def __init__(self, registered_jaw: str):
        self.registered_jaw = registered_jaw
        self.matched: dict = {}
        self.no_jaw: list = []
        self.missing_jaw: list = []
        self.unpaired: list = []

    def report(self) -> dict:
        return {
            "registered_jaw": self.registered_jaw,
            "meshes_without_a_jaw_in_their_name": sorted(self.no_jaw),
            f"patients_without_a_{self.registered_jaw.lower()}_arch": sorted(self.missing_jaw),
            "patients_present_at_only_one_timepoint": sorted(self.unpaired),
        }


def pair(t1_root: str, t2_root: str, suffix: str, registered_jaw: str, carry_other: bool) -> Pairing:
    """Match the subjects of two timepoint folders.

    `registered_jaw` is the arch the patch lives on, and it is the one a subject
    must have at both timepoints to be processed at all. `carry_other` decides
    whether the opposite arch travels with it.

    Pairing the registered jaw ON ITS OWN is what upstream added `SortLower`
    for: `Sort` only kept a lower pair when the matching upper pair existed,
    since the palatal registration always starts from the maxilla -- so a folder
    of mandibles alone, which is exactly what an MGL run may hold, paired to
    nothing.
    """
    other_jaw = catalogs.JAW_LOWER if registered_jaw == catalogs.JAW_UPPER else catalogs.JAW_UPPER
    t1, t2 = discover(t1_root, suffix), discover(t2_root, suffix)
    result = Pairing(registered_jaw)

    for key in sorted(set(t1) | set(t2)):
        for side in (side for side in (t1.get(key), t2.get(key)) if side):
            for jaw, path in side.items():
                if jaw is None:
                    result.no_jaw.append(os.path.basename(path))
        if key not in t1 or key not in t2:
            result.unpaired.append(key)
            continue
        if registered_jaw not in t1[key] or registered_jaw not in t2[key]:
            result.missing_jaw.append(key)
            continue

        wanted = [registered_jaw] + ([other_jaw] if carry_other else [])
        result.matched[key] = {
            jaw: {"t1": t1[key][jaw], "t2": t2[key][jaw]}
            for jaw in wanted
            if jaw in t1[key] and jaw in t2[key]
        }
    return result


# ---------------------------------------------------------------------------
# Painting the patch
# ---------------------------------------------------------------------------
# Both modes hand `register_patient` the same thing: something callable with a
# mesh AND THE PATH IT CAME FROM, returning `(mesh carrying a 0/1 patch array,
# note)`, plus the name of the array it wrote. Everything after that -- the ICP,
# the transform, the writing -- is identical, which is the point: the two arches
# differ in where their stable region is, not in what is done with it.
#
# The path, not the patient key: a patient has ONE key and TWO scans, and the
# mucogingival landmarks are per scan.

# Jaw tokens plus the arch words, dropped when matching a landmark file to its
# scan so `P1_T1_Lower_MG_Pred.json` reduces to what `P1_T1_Lower.vtk` does.
_MATCHING_TOKENS = set(catalogs.JAW_TOKENS)


class MGLPainter:
    """Builds the mucogingival band from landmarks the caller supplied.

    Needs no model and no GPU: a spline, a shortest-path walk over the mesh, and
    a label lookup. A server without pytorch3d cannot run the palatal mode at
    all and runs this one at full speed.
    """

    array_name = mgl.MGL_ARRAY_NAME

    def __init__(self, landmark_root: str, height: float = mgl.DEFAULT_HEIGHT):
        self.height = height
        self.index = landmark_files.index(landmark_root, also_drop=_MATCHING_TOKENS)
        if not self.index:
            raise landmark_files.LandmarkError(
                "the mucogingival landmarks folder holds no Slicer markups file "
                "(.json/.mrk.json)"
            )

    def __call__(self, surface, scan_path: str) -> tuple:
        path = landmark_files.for_scan(self.index, scan_path, also_drop=_MATCHING_TOKENS)
        return mgl.build_patch(surface, landmark_files.load(path), height=self.height)


class PalatePainter:
    """Adapter putting `butterfly.PatchPredictor` behind the same call."""

    array_name = butterfly.PATCH_ARRAY_NAME

    def __init__(self, predictor):
        self.predictor = predictor

    def __call__(self, surface, scan_path: str) -> tuple:
        return self.predictor(surface)


# ---------------------------------------------------------------------------
# One patient
# ---------------------------------------------------------------------------

def register_patient(
    jaws: dict,
    painter,
    registered_jaw: str,
    output_dir: str,
    relative_key: str,
    suffix: str,
    prior_transforms: dict = None,
) -> dict:
    """Register one patient's two timepoints. Returns a report entry.

    Raises `icp.RegistrationError`, `surfaces.SurfaceError`, `mgl.PatchError` or
    `landmarks.LandmarkError` when this patient cannot be registered; the caller
    records that and moves on.
    """
    registered = jaws[registered_jaw]
    t1_surface, t1_note = painter(surfaces.read_surface(registered["t1"]), registered["t1"])
    t2_surface, t2_note = painter(surfaces.read_surface(registered["t2"]), registered["t2"])

    matrix = icp.align(
        butterfly.patch_cloud(t2_surface, painter.array_name),
        butterfly.patch_cloud(t1_surface, painter.array_name),
    )

    relative_dir, patient = os.path.split(relative_key)
    destination = os.path.join(output_dir, relative_dir)
    os.makedirs(destination, exist_ok=True)

    written = [
        _write(t1_surface, registered["t1"], destination, suffix),
        _write(surfaces.transform_surface(t2_surface, matrix), registered["t2"], destination, suffix),
    ]

    for jaw, files in jaws.items():
        if jaw == registered_jaw:
            continue
        # Moved by the REGISTERED arch's transform, not one of its own: the
        # point of registering on a region that does not move is that the whole
        # model follows it, so the occlusion the two arches were captured in is
        # preserved.
        written.append(_write(surfaces.read_surface(files["t1"]), files["t1"], destination, suffix))
        written.append(
            _write(
                surfaces.transform_surface(surfaces.read_surface(files["t2"]), matrix),
                files["t2"],
                destination,
                suffix,
            )
        )

    prior = (prior_transforms or {}).get(relative_key)
    written.append(
        icp.write_transform(
            matrix,
            os.path.join(destination, f"{patient}_{suffix}_transform.tfm"),
            prior_path=prior,
        )
    )

    entry = {
        "status": "ok",
        "registered_on": painter.array_name,
        "registered_jaw": registered_jaw,
        "jaws": sorted(jaws),
        "transform_maps": (
            "the T2 mesh you sent -> T1 space"
            if prior
            else "the registered T2 mesh -> the T1 mesh AREG was given"
        ),
        "outputs": sorted(os.path.relpath(path, output_dir) for path in written),
    }
    notes = [note for note in (t1_note, t2_note) if note]
    if notes:
        entry["notes"] = notes
    return entry


def _write(surface, source_path: str, destination: str, suffix: str) -> str:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return surfaces.write_surface(
        surface,
        os.path.join(destination, f"{stem}_{suffix}{surfaces.output_extension(source_path)}"),
    )

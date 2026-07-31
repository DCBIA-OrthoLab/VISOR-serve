"""The IOS half of ASO: pair meshes with their landmarks per jaw, register each
jaw onto the reference, write the oriented meshes and transforms.

Ported from `ASO_IOS/PRE_ASO_IOS/PRE_ASO_IOS.py` (fully-automated, registering
on tooth centroids) and `ASO_IOS/SEMI_ASO_IOS/SEMI_ASO_IOS.py` (semi-automated,
registering on landmarks). Both drove the same `ICP` class with a different
`option` callable, so they are one function here with that callable as the
difference.

Crown segmentation is NOT part of this. A fully-automated run needs meshes that
already carry a per-point tooth-label array; `segment_unlabelled` marks the seam
where a future `tools/CrownSeg` plugs in (see ALI_PORT_CONTEXT.md section 3.2).
"""

import logging
import os
import re

import numpy as np
import SimpleITK as sitk

from .. import catalogs, markups
from . import icp as ios_icp
from . import pre_icp, surfaces

logger = logging.getLogger("ASO")

# Tokens naming a jaw inside a file name. The original tested only for "_U_" and
# the substring "upper", and DEFAULTED TO LOWER when neither was found -- so a
# maxillary scan named `patient1.vtk` was quietly registered against the
# mandibular reference and returned as a success.
_JAW_TOKENS = {
    "u": "Upper", "up": "Upper", "upper": "Upper", "maxilla": "Upper", "max": "Upper",
    "l": "Lower", "low": "Lower", "lower": "Lower", "mandible": "Lower", "mand": "Lower",
}

_SPLIT = re.compile(r"[_\-.]+")


class JawError(Exception):
    """A file's jaw could not be determined from its name."""


def patient_and_jaw(filename: str) -> tuple:
    """('P1_U_Seg.vtk') -> ('P1', 'Upper').

    Everything from the jaw token onwards is decoration a previous step added
    (`_Seg`, `_Or`, `_lm`, ...), so the patient is what comes before it. Raises
    JawError when no token names a jaw, rather than guessing.
    """
    stem = _strip_extension(filename)
    tokens = _SPLIT.split(stem)
    for index, token in enumerate(tokens):
        jaw = _JAW_TOKENS.get(token.lower())
        if jaw is not None and index > 0:
            return "_".join(tokens[:index]), jaw
    raise JawError(
        f"'{filename}': cannot tell which jaw this is. Name the files so a "
        f"token says it, e.g. 'P1_U_Seg.vtk' / 'P1_Lower.vtk'."
    )


def is_previous_output(filename: str, suffix: str) -> bool:
    """True if this file looks like something a previous ASO run wrote.

    Same reason as the CBCT engine's: a second run on the same folder must
    orient the original mesh, not the first run's result.
    """
    return _strip_extension(filename).endswith(f"_{suffix}")


def discover(input_root: str, output_suffix: str = "Or") -> dict:
    """{patient key: {jaw: {"surface": path, "markups": path}}}.

    The key is the patient's path relative to the input root, so two patients
    with the same name in different folders stay apart -- and files are paired
    only within one directory, by exact stem, never by the `vtk_name in
    json_name` substring test that made patient `1` match patient `10`.

    A previous run's outputs are used only when a patient has nothing else.
    """
    patients: dict = {}
    unnamed: list = []

    for directory, _, file_names in os.walk(input_root):
        relative = os.path.relpath(directory, input_root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if file_name.startswith("."):
                continue
            is_surface = surfaces.is_surface_file(file_name)
            is_markups = markups.is_markups_file(file_name)
            if not (is_surface or is_markups):
                continue
            try:
                stem, jaw = patient_and_jaw(file_name)
            except JawError as exc:
                unnamed.append(str(exc))
                continue
            entry = patients.setdefault(os.path.join(prefix, stem), {})
            jaw_entry = entry.setdefault(
                jaw, {"surface": [], "old_surface": [], "markups": [], "old_markups": []}
            )
            kind = "surface" if is_surface else "markups"
            if is_previous_output(file_name, output_suffix):
                kind = f"old_{kind}"
            jaw_entry[kind].append(os.path.join(directory, file_name))

    for message in unnamed:
        logger.warning("%s", message)

    return {
        key: {
            jaw: {
                "surface": _first(entry["surface"], entry["old_surface"]),
                "markups": _first(entry["markups"], entry["old_markups"]),
            }
            for jaw, entry in jaws.items()
        }
        for key, jaws in patients.items()
    }


def _first(preferred: list, fallback: list):
    candidates = preferred or fallback
    return candidates[0] if candidates else None


def load_reference(reference_dir: str, need_surfaces: bool) -> dict:
    """{jaw: {"surface": path, "markups": path}} for the reference bundle.

    Fully-automated registers meshes against the reference's meshes, so it needs
    the surfaces; semi-automated registers landmarks against landmarks and needs
    only the markups.
    """
    reference = discover(reference_dir)
    merged: dict = {}
    for jaws in reference.values():
        for jaw, entry in jaws.items():
            target = merged.setdefault(jaw, {"surface": None, "markups": None})
            target["surface"] = target["surface"] or entry["surface"]
            target["markups"] = target["markups"] or entry["markups"]

    wanted = "surface" if need_surfaces else "markups"
    if not any(entry[wanted] for entry in merged.values()):
        raise ValueError(
            f"The reference bundle holds no {'mesh' if need_surfaces else 'landmark file'} "
            f"whose name says which jaw it is (e.g. 'Gold_Upper.vtk')."
        )
    return merged


def orient_patient(
    jaws: dict,
    reference: dict,
    automation: str,
    selected_teeth: dict,
    landmark_keys: dict,
    wanted_jaws: list,
    driving_jaw: str,
    output_dir: str,
    relative_key: str,
    suffix: str,
    max_triplets: int,
    seed: int,
) -> dict:
    """Orient one patient's jaws. Returns a report entry.

    With `driving_jaw` set, that jaw's transform is applied to the other one as
    well -- occlusion is preserved by moving both halves rigidly together, which
    only makes sense if the two meshes were in occlusion to begin with.
    """
    entry: dict = {"status": "ok", "jaws": {}, "outputs": []}
    matrices: dict = {}

    order = _ordered_jaws(wanted_jaws, driving_jaw)
    for jaw in order:
        available = jaws.get(jaw)
        if not available or not available["surface"]:
            entry["jaws"][jaw] = {"status": "skipped", "reason": "no mesh for this jaw"}
            continue
        try:
            matrices[jaw] = _matrix_for(
                jaw,
                available,
                reference,
                automation,
                selected_teeth,
                landmark_keys,
                driving_jaw,
                matrices,
                max_triplets,
                seed,
            )
        except (ios_icp.RegistrationError, surfaces.SurfaceError, ValueError) as exc:
            entry["jaws"][jaw] = {"status": "failed", "reason": str(exc)}
            continue

        written = _write_jaw(
            available, matrices[jaw], output_dir, relative_key, jaw, suffix
        )
        entry["jaws"][jaw] = {
            "status": "ok",
            "registered_on": (
                f"the {driving_jaw} jaw's transform"
                if driving_jaw and jaw != driving_jaw
                else ("tooth centroids" if automation == catalogs.AUTOMATION_FULLY
                      else "landmarks")
            ),
        }
        entry["outputs"].extend(written)

    if not any(jaw.get("status") == "ok" for jaw in entry["jaws"].values()):
        entry["status"] = "failed"
        entry["reason"] = "; ".join(
            f"{jaw}: {detail.get('reason', detail['status'])}"
            for jaw, detail in entry["jaws"].items()
        ) or "no jaw could be oriented"
    entry["outputs"].sort()
    return entry


def segment_unlabelled(surface_path: str) -> None:
    """Seam for server-side crown segmentation, not implemented yet.

    A mesh with no tooth-label array cannot be oriented by the fully-automated
    mode. Producing that array means running `shapeaxi`'s `dental_model_seg`,
    which needs pytorch3d -- absent from the deployment image (see
    ALI_PORT_CONTEXT.md section 4). It belongs in a `tools/CrownSeg` of its own
    because ALI, AREG and FlexReg need it too, so this function is where ASO
    will call it, and nothing else here changes when it lands.
    """
    raise ios_icp.RegistrationError(
        f"'{os.path.basename(surface_path)}' carries no tooth labels. The "
        f"fully-automated mode needs a mesh with a per-point array named one of "
        f"{', '.join(markups.LABEL_ARRAY_NAMES)}. Segment it first, or use the "
        f"semi-automated mode with landmark files."
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _ordered_jaws(wanted_jaws: list, driving_jaw: str) -> list:
    """The driving jaw first: the other one reuses its matrix."""
    if driving_jaw and driving_jaw in wanted_jaws:
        return [driving_jaw] + [jaw for jaw in wanted_jaws if jaw != driving_jaw]
    return list(wanted_jaws)


def _matrix_for(
    jaw: str,
    available: dict,
    reference: dict,
    automation: str,
    selected_teeth: dict,
    landmark_keys: dict,
    driving_jaw: str,
    matrices: dict,
    max_triplets: int,
    seed: int,
) -> np.ndarray:
    if driving_jaw and jaw != driving_jaw:
        if driving_jaw not in matrices:
            raise ios_icp.RegistrationError(
                f"the {driving_jaw} jaw, which drives this one, was not oriented"
            )
        return matrices[driving_jaw]

    reference_entry = reference.get(jaw)
    if not reference_entry:
        raise ios_icp.RegistrationError(f"the reference has no {jaw} jaw")

    if automation == catalogs.AUTOMATION_FULLY:
        return _fully_automated_matrix(
            available, reference_entry, selected_teeth[jaw], max_triplets, seed
        )
    return _semi_automated_matrix(
        available, reference_entry, landmark_keys[jaw], max_triplets, seed
    )


def _fully_automated_matrix(
    available: dict, reference_entry: dict, teeth: list, max_triplets: int, seed: int
) -> np.ndarray:
    if not reference_entry["surface"]:
        raise ios_icp.RegistrationError("the reference has no mesh for this jaw")

    source = surfaces.read_surface(available["surface"])
    array_name = surfaces.label_array_name(source)
    if array_name is None:
        segment_unlabelled(available["surface"])

    target = surfaces.read_surface(reference_entry["surface"])
    reference_array = surfaces.label_array_name(target)
    if reference_array is None:
        raise ios_icp.RegistrationError(
            "the reference mesh carries no tooth labels, so there is nothing to "
            "register the tooth centroids against"
        )

    coarse = pre_icp.align(source, target, teeth, array_name)
    aligned = surfaces.transform_surface(source, coarse)

    tooth_ids = catalogs.teeth_to_ids(teeth)
    fine = ios_icp.register(
        ios_icp.mean_teeth(aligned, tooth_ids, array_name),
        ios_icp.mean_teeth(target, tooth_ids, reference_array),
        max_triplets=max_triplets,
        seed=seed,
    )
    return fine @ coarse


def _semi_automated_matrix(
    available: dict, reference_entry: dict, keys: list, max_triplets: int, seed: int
) -> np.ndarray:
    if not available["markups"]:
        raise ios_icp.RegistrationError(
            "no landmark file for this jaw; the semi-automated mode registers on "
            "landmarks you provide"
        )
    if not reference_entry["markups"]:
        raise ios_icp.RegistrationError("the reference has no landmark file for this jaw")

    source = ios_icp.select_keys(markups.load_landmarks(available["markups"]), keys)
    target = ios_icp.select_keys(markups.load_landmarks(reference_entry["markups"]), keys)
    return ios_icp.register(source, target, max_triplets=max_triplets, seed=seed)


def _write_jaw(
    available: dict,
    matrix: np.ndarray,
    output_dir: str,
    relative_key: str,
    jaw: str,
    suffix: str,
) -> list:
    relative_dir, patient = os.path.split(relative_key)
    destination = os.path.join(output_dir, relative_dir)
    os.makedirs(destination, exist_ok=True)
    written = []

    surface = surfaces.read_surface(available["surface"])
    oriented = surfaces.transform_surface(surface, matrix)
    stem = _strip_extension(os.path.basename(available["surface"]))
    extension = surfaces.output_extension(available["surface"])
    written.append(
        surfaces.write_surface(oriented, os.path.join(destination, f"{stem}_{suffix}{extension}"))
    )

    if available["markups"]:
        landmarks = markups.load_landmarks(available["markups"])
        moved = {
            name: (matrix @ np.append(point, 1.0))[:3] for name, point in landmarks.items()
        }
        landmark_stem = _strip_extension(os.path.basename(available["markups"]))
        written.append(
            markups.rewrite_landmarks(
                moved,
                available["markups"],
                os.path.join(destination, f"{landmark_stem}_{suffix}.mrk.json"),
            )
        )

    # Named per jaw. The original wrote `<patient>_SegOr.tfm` for both, so the
    # second jaw silently overwrote the first one's transform.
    written.append(
        _write_transform(
            matrix, os.path.join(destination, f"{patient}_{jaw}_{suffix}.tfm")
        )
    )
    return [os.path.relpath(path, output_dir) for path in written]


def _write_transform(matrix: np.ndarray, path: str) -> str:
    inverted = np.linalg.inv(matrix)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(inverted[:3, :3].flatten().tolist())
    transform.SetTranslation(inverted[:3, 3].tolist())
    sitk.WriteTransform(transform, path)
    return path


def _strip_extension(filename: str) -> str:
    lower = filename.lower()
    for extension in markups.MARKUPS_EXTENSIONS + surfaces.SURFACE_EXTENSIONS:
        if lower.endswith(extension):
            return filename[: -len(extension)]
    return os.path.splitext(filename)[0]

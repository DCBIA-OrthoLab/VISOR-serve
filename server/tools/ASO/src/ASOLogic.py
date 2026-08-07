"""ASO -- Automated Standardized Orientation.

Ported from the Slicer extension's `ASO/` module and its four CLI modules
(`ASO_CBCT/{PRE,SEMI}_ASO_CBCT`, `ASO_IOS/{PRE,SEMI}_ASO_IOS`). One tool, two
engines, four modes:

|            | Semi-Automated                        | Fully-Automated                       |
|------------|---------------------------------------|---------------------------------------|
| **CBCT**   | landmarks you send, ICP onto a gold   | landmarks predicted first (see        |
|            | landmark set                          | `ali_client`), then the same ICP      |
| **IOS**    | landmarks you send, ICP per jaw       | tooth centroids of an already         |
|            |                                       | segmented mesh, ICP per jaw           |

The Slicer envelope is gone entirely: no `slicer.modules.*` chaining, no
`<filter-progress>` prints, no `time.sleep` progress theatre, no `sys.exit`, no
log file the client polls, no `*Error.txt` written next to the results, and
nothing whatsoever written into the caller's input tree.

Two entry points, for the same reason AMASSS has two:

* `orient(...)` -> `OrientationRun`. The real API, returning the output
  directory plus a structured report. **This is what other server-side tools
  should call** (AREG needs oriented scans): it hands back the files, with no
  zip round trip.
* `main(...)` -> the output directory's path. The thin schema adapter used by
  `ASO.py`; with `output_kind = "files"`, main.py zips that directory and
  streams it, so no zip code lives here.
"""

import json
import logging
import os
import shutil
import sys
import zipfile

import file_utils
from base import ToolArgumentError
from config import settings

from . import ali_client, catalogs
from .cbct import dicom
from .cbct import pipeline as cbct_pipeline
from .ios import pipeline as ios_pipeline

logger = logging.getLogger("ASO")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")
    )
    logger.addHandler(_handler)

REPORT_NAME = "ASO_report.json"


class OrientationRun:
    """Result of `orient()`: where the files are, and what actually happened.

    A run is reported per patient, so a batch where three scans out of forty
    could not be registered says which three and why -- the original wrote a
    `<name>Error.txt` into the output folder and carried on, which read as a
    successful run with a few odd text files in it.
    """

    def __init__(self, output_dir: str, report: dict):
        self.output_dir = output_dir
        self.report = report

    @property
    def patients(self) -> dict:
        return self.report["patients"]

    @property
    def succeeded(self) -> list:
        return [
            key for key, entry in self.patients.items() if entry.get("status") == "ok"
        ]

    @property
    def output_files(self) -> list:
        return [
            os.path.join(self.output_dir, relative)
            for entry in self.patients.values()
            for relative in entry.get("outputs", [])
        ]


# ---------------------------------------------------------------------------
# The reusable API
# ---------------------------------------------------------------------------

def orient(
    input_path: str,
    reference_path: str,
    modality: str,
    automation: str,
    cbct_landmarks=None,
    ios_teeth=None,
    ios_landmark_types=None,
    ios_jaws=None,
    ios_occlusion: str = catalogs.OCCLUSION_INDEPENDENT,
    landmark_models: str = None,
    dicom_input: bool = False,
    output_suffix: str = "Or",
    scratch_dir: str = None,
    max_triplets: int = None,
    seed: int = None,
) -> OrientationRun:
    """Orient every case under `input_path` onto `reference_path`.

    `input_path` and `reference_path` are each a file, a `.zip`, or a directory.
    Selections are sequences of the names declared in `catalogs`.
    """
    scratch_dir = scratch_dir or file_utils.make_scratch_dir("ASO_")
    max_triplets = settings.ASO_ICP_MAX_TRIPLETS if max_triplets is None else max_triplets
    seed = settings.ASO_ICP_SEED if seed is None else seed

    output_dir = os.path.join(scratch_dir, f"ASO_{output_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    input_root = _as_directory(input_path, os.path.join(scratch_dir, "input_extracted"))
    reference_root = _as_directory(
        reference_path, os.path.join(scratch_dir, "reference_extracted")
    )

    report = {
        "modality": modality,
        "automation": automation,
        "output_suffix": output_suffix,
        "reference": os.path.basename(str(reference_path).rstrip(os.sep)),
        "patients": {},
    }

    if modality == catalogs.MODALITY_CBCT:
        _run_cbct(
            input_root=input_root,
            reference_root=reference_root,
            automation=automation,
            requested=list(cbct_landmarks or ()),
            landmark_models=landmark_models,
            dicom_input=dicom_input,
            output_dir=output_dir,
            scratch_dir=scratch_dir,
            suffix=output_suffix,
            max_triplets=max_triplets,
            seed=seed,
            report=report,
        )
    else:
        _run_ios(
            input_root=input_root,
            reference_root=reference_root,
            automation=automation,
            teeth=list(ios_teeth or ()),
            landmark_types=list(ios_landmark_types or ()),
            jaws=list(ios_jaws or ()),
            occlusion=ios_occlusion,
            output_dir=output_dir,
            suffix=output_suffix,
            max_triplets=max_triplets,
            seed=seed,
            report=report,
        )

    _summarize(report)
    with open(os.path.join(output_dir, REPORT_NAME), "w") as handle:
        json.dump(report, handle, indent=2)

    return OrientationRun(output_dir, report)


# ---------------------------------------------------------------------------
# The schema adapter
# ---------------------------------------------------------------------------

def main(
    input,
    modality,
    automation,
    reference,
    landmark_models=None,
    cbct_landmarks=None,
    ios_teeth=None,
    ios_landmark_types=None,
    ios_jaws=None,
    ios_occlusion=None,
    dicom_input=False,
    output_suffix="Or",
) -> str:
    """Translate the schema's arguments into `orient()` and return its output
    directory, which main.py zips and streams.

    Every cross-argument rule is checked HERE, before any file is read: a
    request that cannot work must come back as a 422 in a second, not after
    minutes of registration.
    """
    modality, automation = str(modality), str(automation)
    occlusion = str(ios_occlusion or catalogs.OCCLUSION_INDEPENDENT)
    suffix = (output_suffix or "Or").strip() or "Or"
    if os.sep in suffix or (os.altsep and os.altsep in suffix):
        raise ToolArgumentError("'output_suffix' is a name fragment, not a path.")

    if modality == catalogs.MODALITY_CBCT:
        landmarks = _selected(cbct_landmarks, catalogs.CBCT_LANDMARK_CHOICES)
        _check_cbct(automation, landmarks)
        selection = {"cbct_landmarks": landmarks}
    else:
        teeth = _selected(ios_teeth, catalogs.TOOTH_CHOICES)
        types = _selected(ios_landmark_types, catalogs.IOS_LANDMARK_TYPE_CHOICES)
        jaws = _selected(ios_jaws, catalogs.JAW_CHOICES)
        _check_ios(automation, teeth, types, jaws, occlusion)
        selection = {"ios_teeth": teeth, "ios_landmark_types": types, "ios_jaws": jaws}

    scratch_dir = file_utils.make_scratch_dir("ASO_")
    run = orient(
        input_path=str(input),
        reference_path=str(reference),
        modality=modality,
        automation=automation,
        ios_occlusion=occlusion,
        landmark_models=str(landmark_models) if landmark_models else None,
        dicom_input=bool(dicom_input),
        output_suffix=suffix,
        scratch_dir=scratch_dir,
        **selection,
    )

    # The archive should hold results, not the working copies they came from.
    for intermediate in ("input_extracted", "reference_extracted", "dicom_nifti", "centered"):
        shutil.rmtree(os.path.join(scratch_dir, intermediate), ignore_errors=True)
    return run.output_dir


# ---------------------------------------------------------------------------
# Argument rules
# ---------------------------------------------------------------------------

def _selected(value, choices: dict) -> list:
    """The enabled options of a multichoice argument, in declaration order.

    Accepts the `Selection` validate() produces, a plain dict, or a sequence --
    so `orient()` can be called directly with `["Ba", "S", "N"]`.
    """
    if value is None:
        return [name for name, on in choices.items() if on]
    if isinstance(value, dict):
        return [name for name in choices if value.get(name)]
    wanted = set(value)
    return [name for name in choices if name in wanted]


def _check_cbct(automation: str, landmarks: list) -> None:
    if len(landmarks) < 3:
        raise ToolArgumentError(
            f"CBCT orientation registers on at least 3 landmarks; "
            f"{len(landmarks)} are selected in 'cbct_landmarks'."
        )
    if automation != catalogs.AUTOMATION_FULLY:
        return
    # The landmark tool is checked before the input is even extracted. Neither
    # order changes what fails, but this one changes what the caller is told:
    # with the tool absent, "deploy ALI or use Semi-Automated" is the answer
    # whatever they put in 'landmark_models'.
    #
    # 'landmark_models' itself is NOT checked here any more. It is optional:
    # omitted, the landmark tool picks the bundle matching the input from its
    # OWN hosted models (ALILogic.select_bundle), which is both the right
    # default and the only one a client can express -- a combo box selects its
    # first entry the moment it is filled, so "name a bundle or else" made the
    # first entry of a list ASO does not control into the de-facto answer. That
    # list is DATA/ASO/models/, which holds the reference bundles too, so the
    # de-facto answer was routinely a reference: 'No CBCT landmark weights
    # found in CBCT_Gold_Frankfurt_Horizontal_Midsagittal_Plane'.
    if not ali_client.is_available(settings.ASO_LANDMARK_TOOL):
        raise ToolArgumentError(
            ali_client._NOT_AVAILABLE.format(tool=settings.ASO_LANDMARK_TOOL)
        )


def _no_landmarks_reason(key: str, markups_paths: list, orphans: list) -> str:
    """Why a Semi-Automated patient has no landmarks -- three situations the
    one message used to cover with the only wording that could be wrong.

    "No landmark file alongside this scan" was emitted whenever the merged
    dict came out empty, which happens when the file is absent, when it was
    matched but unreadable (`load_landmarks` only logs that), and when a file
    IS in the folder but under a name that put it in another bucket. Telling
    a user their file is missing while it sits next to the scan is the worst
    of the three, and it was the likeliest.
    """
    if markups_paths:
        names = ", ".join(sorted(os.path.basename(path) for path in markups_paths))
        return (
            f"the landmark file(s) found for this scan ({names}) hold no usable "
            f"control point -- check the server log for the file that was skipped"
        )
    if orphans:
        return (
            f"no landmark file matched this scan. {len(orphans)} landmark file(s) in "
            f"the input matched no scan at all ({', '.join(orphans)}): a scan and its "
            f"landmarks are paired by name, up to a trailing "
            f"{', '.join(cbct_pipeline.PATIENT_SUFFIXES)}. Rename them so both sides "
            f"share the same stem -- e.g. '{key}_scan.nii.gz' with '{key}_lm.mrk.json'"
        )
    return (
        "no landmark file (.mrk.json) alongside this scan. Semi-Automated mode "
        "registers on landmarks you provide, so they must travel with the scans -- "
        "send the whole FOLDER rather than the single scan file, or use "
        "Fully-Automated mode to have them predicted"
    )


def _check_selection_against_reference(requested: list, reference_landmarks: dict) -> None:
    """Refuse a selection the reference cannot support, before reading a scan.

    A reference defines the target frame through the landmarks it carries, and
    the two published bundles carry DISJOINT sets: Frankfurt Horizontal +
    Midsagittal has Ba/S/N/RPo/LPo/ROr/LOr, Occlusal + Midsagittal has
    ANS/IF/PNS/UL6O/UR1O/UR6O. The schema's defaults are the first set, so
    picking the second one without changing the selection means every landmark
    is dropped as "not in the reference" and every patient fails separately with
    "0 usable landmarks" -- a batch of forty identical failures for one wrong
    choice made in one place.

    The server cannot know which reference will be picked when it publishes
    `choices`, but it knows both the moment the request arrives. So it says so
    once, and names what the reference actually offers.
    """
    usable = [name for name in requested if name in reference_landmarks]
    if len(usable) >= cbct_pipeline.icp.MIN_LANDMARKS:
        return
    raise ToolArgumentError(
        f"Only {len(usable)} of the selected landmarks exist in this reference, and "
        f"{cbct_pipeline.icp.MIN_LANDMARKS} are needed. The reference defines: "
        f"{', '.join(sorted(reference_landmarks))}. Select landmarks from that list in "
        f"'cbct_landmarks', or pick a reference built on the ones you selected."
    )


def _check_ios(automation: str, teeth: list, types: list, jaws: list, occlusion: str) -> None:
    if not jaws:
        raise ToolArgumentError("Select at least one jaw in 'ios_jaws'.")
    if occlusion not in catalogs.DRIVING_JAW:
        raise ToolArgumentError(
            f"Unknown 'ios_occlusion' mode '{occlusion}'. Expected one of: "
            f"{', '.join(catalogs.DRIVING_JAW)}"
        )
    driving = catalogs.DRIVING_JAW[occlusion]
    if driving and driving not in jaws:
        raise ToolArgumentError(
            f"'ios_occlusion' asks the {driving} jaw to drive the other one, but "
            f"{driving} is not selected in 'ios_jaws'."
        )

    per_jaw = catalogs.split_by_jaw(teeth)
    # Only the jaws that are actually registered need teeth. With one jaw
    # driving the other, the driven one is moved by the driver's transform and
    # needs none of its own.
    registered = [driving] if driving else list(jaws)

    if automation == catalogs.AUTOMATION_FULLY:
        for jaw in registered:
            count = len(per_jaw[jaw])
            if count not in (3, 4):
                raise ToolArgumentError(
                    f"Fully-Automated IOS aligns each jaw from 3 or 4 teeth spread "
                    f"across the arch; {count} are selected for the {jaw} jaw in "
                    f"'ios_teeth'."
                )
        return

    if not types:
        raise ToolArgumentError(
            "Semi-Automated IOS registers on landmarks; select at least one in "
            "'ios_landmark_types'."
        )
    keys = catalogs.landmark_keys_by_jaw(teeth, types)
    for jaw in registered:
        if len(keys[jaw]) < 3:
            raise ToolArgumentError(
                f"Semi-Automated IOS registers on at least 3 landmarks per jaw; "
                f"the {jaw} jaw has {len(keys[jaw])} "
                f"({len(per_jaw[jaw])} teeth x {len(types)} landmark types)."
            )


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def _run_cbct(
    input_root, reference_root, automation, requested, landmark_models, dicom_input,
    output_dir, scratch_dir, suffix, max_triplets, seed, report,
) -> None:
    if dicom_input:
        input_root = dicom.convert_tree(
            input_root, os.path.join(scratch_dir, "dicom_nifti")
        )

    reference_landmarks = cbct_pipeline.load_reference(reference_root)
    report["requested_landmarks"] = list(requested)
    report["reference_landmarks"] = sorted(reference_landmarks)

    patients = cbct_pipeline.discover(input_root, suffix)
    if not patients:
        raise ToolArgumentError(
            "No CBCT scan found in the input. Expected one of "
            f"{', '.join(cbct_pipeline.SCAN_EXTENSIONS)}, sent as a file or a "
            f"zipped folder."
        )

    # After discovery, before a single scan is read: with nothing to orient,
    # "there are no scans here" is the useful answer, and complaining about the
    # landmark selection first would bury it.
    _check_selection_against_reference(requested, reference_landmarks)

    fully = automation == catalogs.AUTOMATION_FULLY

    # Landmark files that matched no scan. Collected even on a run that
    # succeeds: "39 of your 40 patients were oriented" and "the 40th one's
    # landmarks are in the folder under a name nothing matched" are the same
    # sentence, and only this makes the second half sayable.
    orphans = (
        [] if fully
        else sorted(
            os.path.basename(path)
            for paths in cbct_pipeline.orphan_markups(input_root, suffix).values()
            for path in paths
        )
    )
    report["unmatched_markups"] = orphans

    centered_root = os.path.join(scratch_dir, "centered")

    # Phase 1 -- recentre. The landmark tool runs on centred scans (that is what
    # the Slicer chain did, PRE_ASO_CBCT before ALI_CBCT), so in fully-automated
    # mode they have to reach disk before it is called.
    prepared = {}
    for key, entry in sorted(patients.items()):
        _, extension = cbct_pipeline.split_scan_extension(os.path.basename(entry["scan"]))
        destination = (
            os.path.join(
                centered_root,
                f"{key}{cbct_pipeline.compressed_extension(extension)}",
            )
            if fully
            else None
        )
        try:
            image, translation = cbct_pipeline.prepare(entry["scan"], destination)
        except RuntimeError as exc:
            report["patients"][key] = {"status": "failed", "reason": str(exc)}
            continue
        prepared[key] = {
            "image": None if fully else image,  # kept in RAM only when it is used next
            "translation": translation,
            "extension": extension,
            "centered_path": destination,
            "landmarks": None,
        }

    # Phase 2 -- landmarks, either the caller's (moved into the centred space)
    # or predicted ones (already in it).
    if fully:
        predictions = ali_client.predict_landmarks(
            centered_root,
            settings.ASO_LANDMARK_TOOL,
            landmark_models,
            requested,
            scratch_dir,
        )
        for key, entry in prepared.items():
            entry["landmarks"] = predictions.get(key, {})
    else:
        for key, entry in prepared.items():
            entry["landmarks"] = cbct_pipeline.center_landmarks(
                cbct_pipeline.load_landmarks(patients[key]["markups"]),
                entry["translation"],
            )

    # Phase 3 -- register and write.
    for key, entry in sorted(prepared.items()):
        if not entry["landmarks"]:
            report["patients"][key] = {
                "status": "failed",
                "reason": (
                    "no predicted landmarks for this scan"
                    if fully
                    else _no_landmarks_reason(key, patients[key]["markups"], orphans)
                ),
            }
            continue
        image = entry["image"]
        if image is None:
            import SimpleITK as sitk  # local: only the fully-automated path re-reads

            image = sitk.ReadImage(entry["centered_path"])
        try:
            report["patients"][key] = cbct_pipeline.orient_patient(
                centered=image,
                pre_transform=entry["translation"],
                source_landmarks=entry["landmarks"],
                reference_landmarks=reference_landmarks,
                requested=requested,
                output_dir=output_dir,
                relative_key=key,
                suffix=suffix,
                extension=entry["extension"],
                max_triplets=max_triplets,
                seed=seed,
            )
        except cbct_pipeline.icp.RegistrationError as exc:
            report["patients"][key] = {"status": "failed", "reason": str(exc)}
        entry["image"] = None  # a batch must not hold every volume in RAM


def _run_ios(
    input_root, reference_root, automation, teeth, landmark_types, jaws, occlusion,
    output_dir, suffix, max_triplets, seed, report,
) -> None:
    fully = automation == catalogs.AUTOMATION_FULLY
    reference = ios_pipeline.load_reference(reference_root, need_surfaces=fully)

    patients = ios_pipeline.discover(input_root, suffix)
    patients = {key: entry for key, entry in patients.items() if _has_surface(entry)}
    if not patients:
        raise ToolArgumentError(
            "No intra-oral mesh found in the input. Expected one of "
            f"{', '.join(ios_pipeline.surfaces.SURFACE_EXTENSIONS)}, named so a "
            f"token says which jaw it is (e.g. 'P1_U_Seg.vtk')."
        )

    report["requested_teeth"] = list(teeth)
    if not fully:
        report["requested_landmark_types"] = list(landmark_types)

    selected_teeth = catalogs.split_by_jaw(teeth)
    landmark_keys = catalogs.landmark_keys_by_jaw(teeth, landmark_types)
    driving = catalogs.DRIVING_JAW[occlusion]

    for key, entry in sorted(patients.items()):
        report["patients"][key] = ios_pipeline.orient_patient(
            jaws=entry,
            reference=reference,
            automation=automation,
            selected_teeth=selected_teeth,
            landmark_keys=landmark_keys,
            wanted_jaws=jaws,
            driving_jaw=driving,
            output_dir=output_dir,
            relative_key=key,
            suffix=suffix,
            max_triplets=max_triplets,
            seed=seed,
        )

    if fully:
        _reject_if_nothing_was_labelled(report["patients"])


def _has_surface(entry: dict) -> bool:
    return any(jaw["surface"] for jaw in entry.values())


# What `ios_pipeline.segment_unlabelled` puts in the reason, matched to tell a
# wrong-mode request apart from a bad mesh.
_UNLABELLED = "carries no tooth labels"


def _reject_if_nothing_was_labelled(patients: dict) -> None:
    """Turn "not one of your meshes is segmented" into a 422.

    The distinction matters, and only these two cases are treated differently.
    No labelled mesh at all means the caller chose the wrong mode, and saying so
    is more use than an empty archive. Some meshes labelled and others not is a
    data problem: those are recorded per patient and the rest of the batch is
    kept, because "one of your forty meshes was bad" is not a reason to return
    nothing.

    Checked after the loop rather than before it, so no mesh is read twice: an
    unlabelled one fails the moment its label array is probed, before any
    registration work.
    """
    reasons = [
        jaw.get("reason", "")
        for entry in patients.values()
        for jaw in entry.get("jaws", {}).values()
    ]
    if not reasons or not all(_UNLABELLED in reason for reason in reasons):
        return
    raise ToolArgumentError(
        "Fully-Automated IOS orients a mesh by its tooth labels, and none of the "
        "meshes sent carries a per-point array named one of "
        f"{', '.join(ios_pipeline.markups.LABEL_ARRAY_NAMES)}. Segment them first, "
        "or use Semi-Automated mode with landmark files."
    )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _as_directory(path: str, destination: str) -> str:
    """A directory holding the input, whatever shape it arrived in.

    A single uploaded file is linked into a directory of its own rather than
    used from where it landed: main.py streams every upload of a request into
    ONE work directory, so treating the file's parent as the input root would
    make the reference archive part of the input.
    """
    path = str(path)
    if os.path.isdir(path):
        return path

    if zipfile.is_zipfile(path):
        return file_utils.extract_zip(
            path,
            extract_dir=destination,
            strip_single_root=True,
            max_total_bytes=settings.MAX_EXTRACTED_MB * 1024 * 1024,
        )

    os.makedirs(destination, exist_ok=True)
    linked = os.path.join(destination, os.path.basename(path))
    try:
        os.link(path, linked)
    except OSError:
        shutil.copy2(path, linked)
    return destination


def _summarize(report: dict) -> None:
    statuses = [entry.get("status") for entry in report["patients"].values()]
    report["summary"] = {
        "patients": len(statuses),
        "oriented": statuses.count("ok"),
        "failed": statuses.count("failed"),
    }
    logger.info(
        "%s %s: %d/%d oriented",
        report["modality"],
        report["automation"],
        report["summary"]["oriented"],
        report["summary"]["patients"],
    )

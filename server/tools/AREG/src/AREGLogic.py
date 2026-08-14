"""AREG -- Automated REGistration of two timepoints.

Ported from the Slicer extension's `AREG/` module and its CLI modules
(`AREG_CBCT`, `AREG_IOS`). One tool, two engines, five modes:

|          | Semi-Automated                | Fully-Automated              | Oriented + Fully-Automated |
|----------|-------------------------------|------------------------------|----------------------------|
| **CBCT** | your T1 masks, masked Elastix | AMASSS segments the T1 masks | ASO orients the T1 first   |
| **IOS**  | your segmented meshes         | CrownSeg labels + ASO orients| --                         |

The Slicer envelope is gone entirely: no `<filter-progress>` prints, no
`time.sleep(0.2)` progress theatre, no `sys.exit`, no log file the client
polls, and nothing written into the caller's input tree.

Two entry points, for the same reason AMASSS and ASO have two:

* `register(...)` -> `RegistrationRun`, the real API: the output directory plus
  a structured report. This is what another server-side tool calls.
* `main(...)` -> the output directory's path, the schema adapter `AREG.py` uses.

The Slicer widget built a list of CLI invocations per mode and ran them in
order, passing folders between them. That structure survives, but the steps are
the server's own tools -- see `tools_client.py` for why they are called
in-process and through the registry.
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

from . import catalogs, dicom, pairing, tools_client

logger = logging.getLogger("AREG")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")
    )
    logger.addHandler(_handler)

REPORT_NAME = "AREG_report.json"


class RegistrationRun:
    """Result of `register()`: where the files are, and what actually happened.

    Reported per patient AND per region, because a CBCT run registering on the
    cranial base and the mandible is two registrations of every patient and one
    of them can fail on its own. The original caught each per-patient exception
    into a log line and finished by printing a count; the archive said nothing.
    """

    def __init__(self, output_dir: str, report: dict):
        self.output_dir = output_dir
        self.report = report

    @property
    def patients(self) -> dict:
        return self.report["patients"]

    @property
    def succeeded(self) -> list:
        return [key for key, entry in self.patients.items() if entry.get("status") == "ok"]


# ---------------------------------------------------------------------------
# The reusable API
# ---------------------------------------------------------------------------

def register(
    t1_path: str,
    t2_path: str,
    modality: str,
    automation: str,
    regions=None,
    t1_masks_path: str = None,
    segmentation_model: str = None,
    segmentation_label: int = 0,
    orientation_reference: str = None,
    registration_model: str = None,
    ios_patch: str = catalogs.PATCH_PALATE,
    mgl_landmarks_path: str = None,
    mgl_patch_height: float = None,
    dicom_input: bool = False,
    output_suffix: str = "Reg",
    scratch_dir: str = None,
) -> RegistrationRun:
    """Register every T2 under `t2_path` onto its T1 under `t1_path`.

    Each path is a directory or a `.zip`. `regions` are the display names
    declared in `catalogs.REGION_CHOICES` (CBCT only).
    """
    scratch_dir = scratch_dir or file_utils.make_scratch_dir("AREG_")
    output_dir = os.path.join(scratch_dir, f"AREG_{output_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    t1_root = _as_directory(t1_path, os.path.join(scratch_dir, "t1_input"))
    t2_root = _as_directory(t2_path, os.path.join(scratch_dir, "t2_input"))

    report = {
        "modality": modality,
        "automation": automation,
        "output_suffix": output_suffix,
        "patients": {},
    }

    if modality == catalogs.MODALITY_CBCT:
        _run_cbct(
            t1_root=t1_root,
            t2_root=t2_root,
            t1_masks_path=t1_masks_path,
            automation=automation,
            regions=list(regions or ()),
            segmentation_model=segmentation_model,
            segmentation_label=int(segmentation_label or 0),
            orientation_reference=orientation_reference,
            dicom_input=dicom_input,
            output_dir=output_dir,
            scratch_dir=scratch_dir,
            suffix=output_suffix,
            report=report,
        )
    else:
        from .ios import mgl

        _run_ios(
            t1_root=t1_root,
            t2_root=t2_root,
            automation=automation,
            registration_model=registration_model,
            orientation_reference=orientation_reference,
            ios_patch=ios_patch,
            mgl_landmarks_path=mgl_landmarks_path,
            mgl_patch_height=(
                mgl.DEFAULT_HEIGHT if mgl_patch_height is None else float(mgl_patch_height)
            ),
            output_dir=output_dir,
            scratch_dir=scratch_dir,
            suffix=output_suffix,
            report=report,
        )

    _summarize(report)
    with open(os.path.join(output_dir, REPORT_NAME), "w") as handle:
        json.dump(report, handle, indent=2)
    return RegistrationRun(output_dir, report)


# ---------------------------------------------------------------------------
# The schema adapter
# ---------------------------------------------------------------------------

def main(
    modality,
    automation,
    t1,
    t2,
    t1_masks=None,
    cbct_regions=None,
    segmentation_label=0,
    segmentation_model=None,
    cbct_reference=None,
    ios_reference=None,
    registration_model=None,
    ios_patch=None,
    mgl_landmarks=None,
    mgl_patch_height=None,
    dicom_input=False,
    output_suffix="Reg",
) -> str:
    """Translate the schema's arguments into `register()` and return its output
    directory, which main.py zips and streams.

    Every cross-argument rule is checked HERE, before any file is read: a
    request that cannot work must come back as a 422 in a second, not after an
    hour of registration.
    """
    modality, automation = str(modality), str(automation)
    suffix = (output_suffix or "Reg").strip() or "Reg"
    if os.sep in suffix or (os.altsep and os.altsep in suffix):
        raise ToolArgumentError("'output_suffix' is a name fragment, not a path.")

    allowed = catalogs.AUTOMATION_BY_MODALITY.get(modality, ())
    if automation not in allowed:
        raise ToolArgumentError(
            f"'{automation}' is not a mode {modality} has. {modality} offers: "
            f"{', '.join(allowed)}."
        )

    patch = str(ios_patch or catalogs.PATCH_PALATE)
    if modality == catalogs.MODALITY_CBCT:
        regions = _selected(cbct_regions, catalogs.REGION_CHOICES)
        reference = cbct_reference
        _check_cbct(automation, regions, t1_masks, reference)
    else:
        regions = []
        reference = ios_reference
        _check_ios(automation, patch, registration_model, reference, mgl_landmarks,
                   mgl_patch_height)

    scratch_dir = file_utils.make_scratch_dir("AREG_")
    run = register(
        t1_path=str(t1),
        t2_path=str(t2),
        modality=modality,
        automation=automation,
        regions=regions,
        t1_masks_path=str(t1_masks) if t1_masks else None,
        segmentation_model=str(segmentation_model) if segmentation_model else None,
        segmentation_label=int(segmentation_label or 0),
        orientation_reference=str(reference) if reference else None,
        registration_model=str(registration_model) if registration_model else None,
        ios_patch=patch,
        mgl_landmarks_path=str(mgl_landmarks) if mgl_landmarks else None,
        mgl_patch_height=mgl_patch_height,
        dicom_input=bool(dicom_input),
        output_suffix=suffix,
        scratch_dir=scratch_dir,
    )

    # The archive should hold results, not the working copies they came from.
    for intermediate in (
        "t1_input", "t2_input", "masks_input", "dicom_t1", "dicom_t2", "mgl_landmarks"
    ):
        shutil.rmtree(os.path.join(scratch_dir, intermediate), ignore_errors=True)
    return run.output_dir


# ---------------------------------------------------------------------------
# Argument rules
# ---------------------------------------------------------------------------

def _selected(value, choices: dict) -> list:
    """The enabled options of a multichoice argument, in declaration order.

    Accepts the `Selection` validate() produces, a plain dict, or a sequence --
    so `register()` stays directly callable with `["Mandible"]`.
    """
    if value is None:
        return [name for name, on in choices.items() if on]
    if isinstance(value, dict):
        return [name for name in choices if value.get(name)]
    wanted = set(value)
    return [name for name in choices if name in wanted]


def _check_cbct(automation: str, regions: list, t1_masks, reference) -> None:
    if not regions:
        raise ToolArgumentError(
            "Select at least one anatomical region to register on in 'cbct_regions' "
            f"({', '.join(catalogs.REGION_CHOICES)}). Each one is a separate "
            "registration with its own output folder."
        )

    if automation == catalogs.AUTOMATION_SEMI:
        if not t1_masks:
            raise ToolArgumentError(
                "Semi-Automated CBCT registers inside masks you provide: send the T1 "
                "segmentations in 't1_masks', or use Fully-Automated mode to have "
                "them produced server-side."
            )
        return

    # Both automated modes need the segmentation; the oriented one also needs
    # the orientation. Checked before the input is extracted -- with the tool
    # absent, the answer is the same whatever the rest of the request says.
    tools_client.require("AMASSS", f"{automation} CBCT registration")
    if automation == catalogs.AUTOMATION_ORIENTED:
        tools_client.require("ASO", "Oriented + Fully-Automated CBCT registration")
        if not reference:
            raise ToolArgumentError(
                "Oriented + Fully-Automated CBCT orients the T1 scans before "
                "registering onto them, which needs an orientation reference: name "
                "one in 'cbct_reference' (see GET /tools/AREG/data)."
            )


def _check_ios(automation, patch, registration_model, reference, mgl_landmarks, height) -> None:
    if patch not in catalogs.PATCH_CHOICES:
        raise ToolArgumentError(
            f"Unknown 'ios_patch' {patch!r}. Expected one of: "
            f"{', '.join(catalogs.PATCH_CHOICES)}."
        )

    if patch == catalogs.PATCH_MGL:
        # The mucogingival band is built from landmarks, not predicted by a
        # network, so the palatal checkpoint is not involved at all -- asking
        # for one here is how a user comes to believe this mode needs a model.
        #
        # The landmarks themselves are optional: absent, they are predicted by
        # the landmark tool, which is the whole point of running this on a
        # server. Sending them is for a folder that already has them, which also
        # lets a run be repeated without paying for the prediction again.
        if not mgl_landmarks:
            tools_client.require(
                settings.AREG_LANDMARK_TOOL, "Registering on the mucogingival line"
            )
        if height is not None and float(height) < 0:
            raise ToolArgumentError(
                "'mgl_patch_height' is a half-height in millimetres and cannot be "
                "negative. 0 registers on the landmarks alone, without any band."
            )
    elif not registration_model:
        raise ToolArgumentError(
            "Registering on the palate needs its patch-prediction checkpoint: name "
            "one in 'registration_model' (see GET /tools/AREG/data)."
        )

    if automation != catalogs.AUTOMATION_FULLY:
        return
    tools_client.require("CrownSeg", "Fully-Automated IOS registration")
    tools_client.require("ASO", "Fully-Automated IOS registration")
    if not reference:
        raise ToolArgumentError(
            "Fully-Automated IOS orients both timepoints before registering, which "
            "needs an orientation reference: name one in 'ios_reference' (see "
            "GET /tools/AREG/data)."
        )


# ---------------------------------------------------------------------------
# CBCT
# ---------------------------------------------------------------------------

def _run_cbct(
    t1_root, t2_root, t1_masks_path, automation, regions, segmentation_model,
    segmentation_label, orientation_reference, dicom_input, output_dir, scratch_dir,
    suffix, report,
) -> None:
    # Imported here rather than at module level: the CBCT engine pulls in
    # SimpleITK and itk-elastix, and AREG must load on a server without them so
    # its schema is still published and its IOS mode still runs.
    from .cbct import elastix
    from .cbct import pipeline as cbct_pipeline

    elastix.check_dependencies()

    if dicom_input:
        t1_root = dicom.convert_tree(t1_root, os.path.join(scratch_dir, "dicom_t1"))
        t2_root = dicom.convert_tree(t2_root, os.path.join(scratch_dir, "dicom_t2"))

    codes = [catalogs.region_code(name) for name in regions]
    report["regions"] = list(regions)
    report["segmentation_label"] = segmentation_label or None

    # Step 1 -- orient the T1 scans, when the mode asks for it. The T2 is NOT
    # oriented: it is about to be resampled into the T1's frame anyway, and
    # orienting it first would be one more interpolation of the same data.
    if automation == catalogs.AUTOMATION_ORIENTED:
        oriented = tools_client.orient_scans(
            t1_root, orientation_reference, catalogs.MODALITY_CBCT
        )
        report["oriented_t1"] = True
        t1_root = oriented

    # Step 2 -- the masks the registration is confined to.
    mask_roots = []
    if t1_masks_path:
        mask_roots.append(_as_directory(t1_masks_path, os.path.join(scratch_dir, "masks_input")))
    if automation == catalogs.AUTOMATION_SEMI:
        # Where the original looked when no mask folder was given.
        mask_roots.append(t1_root)
    else:
        structures = [catalogs.REGION_MASK_STRUCTURES[code] for code in codes]
        mask_roots.append(
            tools_client.segment_masks(t1_root, segmentation_model, structures)
        )
        report["segmented_t1"] = sorted(structures)

    # Step 3 -- pair the timepoints, then register once per region.
    matched = pairing.pair(t1_root, t2_root, suffix)
    report["unmatched"] = matched.unmatched_report()
    if not matched:
        raise ToolArgumentError(
            "No subject appears in both the T1 and the T2 folder. They are paired by "
            "name, up to the timepoint token and a trailing "
            f"{', '.join(catalogs.PATIENT_SUFFIXES[:4])}... -- so 'P1_T1_scan.nii.gz' "
            f"in one folder pairs with 'P1_T2.nii.gz' in the other. Found "
            f"{len(matched.t1_only)} T1-only and {len(matched.t2_only)} T2-only subject(s)."
        )

    for code in codes:
        masks = cbct_pipeline.find_masks(mask_roots, code, scan_keys=matched.matched)
        for key, entry in sorted(matched.matched.items()):
            record = report["patients"].setdefault(key, {"status": "ok", "regions": {}})
            mask_path = masks.get(key)
            if not mask_path:
                record["regions"][code] = {
                    "status": "failed",
                    "reason": _no_mask_reason(automation, code),
                }
                continue
            try:
                record["regions"][code] = cbct_pipeline.register_patient(
                    t1_path=entry["t1"],
                    t2_path=entry["t2"],
                    mask_path=mask_path,
                    region=code,
                    output_dir=output_dir,
                    relative_key=key,
                    suffix=suffix,
                    segmentation_label=segmentation_label or None,
                )
            except elastix.RegistrationError as exc:
                record["regions"][code] = {"status": "failed", "reason": str(exc)}
            except RuntimeError as exc:
                record["regions"][code] = {"status": "failed", "reason": f"registration failed: {exc}"}

    _roll_up_regions(report["patients"])


def _no_mask_reason(automation: str, code: str) -> str:
    region = catalogs.region_name(code)
    if automation == catalogs.AUTOMATION_SEMI:
        return (
            f"no {region} mask for this subject. A mask is matched to its scan by name "
            f"and has to say both that it is a segmentation (mask/seg/pred) and which "
            f"structure it covers ({'/'.join(catalogs.REGION_TOKENS[code][:2])}) -- "
            f"e.g. 'P1_T1_{code}_seg.nii.gz' next to 'P1_T1_scan.nii.gz'"
        )
    return (
        f"the segmentation step produced no {region} mask for this subject -- see the "
        f"AMASSS report if one was included in this archive"
    )


def _roll_up_regions(patients: dict) -> None:
    """A patient is 'ok' when at least one of its regions registered."""
    for entry in patients.values():
        statuses = [region.get("status") for region in entry["regions"].values()]
        entry["status"] = "ok" if "ok" in statuses else "failed"


# ---------------------------------------------------------------------------
# IOS
# ---------------------------------------------------------------------------

def _run_ios(
    t1_root, t2_root, automation, registration_model, orientation_reference,
    ios_patch, mgl_landmarks_path, mgl_patch_height,
    output_dir, scratch_dir, suffix, report,
) -> None:
    # Imported here rather than at module level: the IOS engine pulls in torch,
    # monai and pytorch3d, and AREG must load (and register CBCT scans) on a
    # server without them.
    from . import landmarks as landmark_files
    from .ios import butterfly, icp, mgl, net
    from .ios import pipeline as ios_pipeline
    from .ios import surfaces

    registered_jaw = catalogs.PATCH_JAW[ios_patch]
    on_palate = ios_patch == catalogs.PATCH_PALATE
    report["patch"] = ios_patch
    report["registered_jaw"] = registered_jaw

    # Only the palatal patch is predicted by a network. The mucogingival band is
    # a spline and a walk over the mesh, so the whole torch/pytorch3d stack is
    # never imported for it -- which is what lets a deployment without pytorch3d
    # register lower arches at full speed while answering 501 for upper ones.
    if on_palate:
        net.check_dependencies()

    prior_transforms: dict = {}
    if automation == catalogs.AUTOMATION_FULLY:
        # Label the crowns, then orient -- the order the Slicer chain used, and
        # the necessary one: ASO's fully-automated IOS mode aligns a mesh by its
        # tooth centroids, so the labels have to exist first.
        t1_root = tools_client.label_crowns(t1_root)
        t2_root = tools_client.label_crowns(t2_root)
        t1_root = tools_client.orient_scans(
            t1_root, orientation_reference, catalogs.MODALITY_IOS
        )
        t2_root = tools_client.orient_scans(
            t2_root, orientation_reference, catalogs.MODALITY_IOS
        )
        report["labelled_and_oriented"] = True
        prior_transforms = _collect_transforms(t2_root, suffix="Or")

    matched = ios_pipeline.pair(
        t1_root, t2_root, suffix, registered_jaw=registered_jaw, carry_other=on_palate
    )
    report["unmatched"] = matched.report()
    if not matched.matched:
        raise ToolArgumentError(
            f"No subject has a {registered_jaw.lower()} arch at both timepoints, and "
            f"the {ios_patch} patch lives on that arch. Meshes are paired by name, and "
            f"each one has to say which jaw it is with a token in its name (e.g. "
            f"'P1_T1_{registered_jaw}.vtk' / 'P1_T2_{registered_jaw[0]}.vtk'). "
            f"{len(matched.no_jaw)} mesh(es) named no jaw, "
            f"{len(matched.unpaired)} subject(s) appear at one timepoint only."
        )

    if on_palate:
        predictor = butterfly.PatchPredictor(registration_model)
        painter = ios_pipeline.PalatePainter(predictor)
        report["device"] = predictor.device
        # Which weights placed this patch has to have an answer next to the
        # result: the bundle is a folder and the checkpoint inside it is found,
        # not named.
        report["model_checkpoint"] = os.path.basename(predictor.checkpoint)
    else:
        if mgl_landmarks_path:
            landmark_root = _as_directory(
                mgl_landmarks_path, os.path.join(scratch_dir, "mgl_landmarks")
            )
            report["mgl_landmarks"] = "sent with the request"
        else:
            # Both timepoints into ONE folder: `landmarks.for_scan` keys on the
            # scan's own name, timepoint included, so T1's and T2's files coexist
            # -- and one call is one model load instead of two.
            landmark_root = os.path.join(scratch_dir, "mgl_predicted")
            os.makedirs(landmark_root, exist_ok=True)
            for root in (t1_root, t2_root):
                # No model named: ALI picks the hosted bundle matching the input
                # from the models hosted for IT, which is the right default and
                # the only one a caller can express -- AREG's own model list
                # holds the palatal checkpoint and the orientation references,
                # none of which is a landmark bundle.
                _merge_into(
                    tools_client.predict_mucogingival(root, tool_name=settings.AREG_LANDMARK_TOOL),
                    landmark_root,
                )
            report["mgl_landmarks"] = f"predicted by '{settings.AREG_LANDMARK_TOOL}'"

        painter = ios_pipeline.MGLPainter(landmark_root, height=mgl_patch_height)
        report["mgl_patch_height_mm"] = mgl_patch_height

    for key, jaws in sorted(matched.matched.items()):
        try:
            report["patients"][key] = ios_pipeline.register_patient(
                jaws=jaws,
                painter=painter,
                registered_jaw=registered_jaw,
                output_dir=output_dir,
                relative_key=key,
                suffix=suffix,
                prior_transforms=prior_transforms,
            )
        except (
            icp.RegistrationError,
            surfaces.SurfaceError,
            mgl.PatchError,
            landmark_files.LandmarkError,
        ) as exc:
            report["patients"][key] = {"status": "failed", "reason": str(exc)}


def _collect_transforms(oriented_root: str, suffix: str) -> dict:
    """{patient key: path} of the transforms ASO wrote for the upper arches.

    They are what lets the `.tfm` AREG returns refer to the mesh the CALLER
    sent rather than to the oriented copy AREG made -- see
    `ios.icp.write_transform`.
    """
    from .ios import surfaces  # for the jaw vocabulary only

    found: dict = {}
    for directory, _, file_names in os.walk(oriented_root):
        relative = os.path.relpath(directory, oriented_root)
        prefix = "" if relative == "." else relative
        for file_name in sorted(file_names):
            if not file_name.endswith(".tfm"):
                continue
            if surfaces.jaw_of(file_name) != catalogs.JAW_UPPER:
                continue
            key = os.path.join(
                prefix, pairing.patient_stem(file_name, also_drop=set(catalogs.JAW_TOKENS))
            )
            found.setdefault(key, os.path.join(directory, file_name))
    return found


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _merge_into(source: str, destination: str) -> None:
    """Copy every file of `source` under `destination`, keeping its tree.

    The two timepoints' landmarks end up in ONE folder on purpose: they are
    matched to their scan by a key that carries the timepoint, so they cannot
    collide, and one folder is one index for the painter to search.
    """
    for directory, _, file_names in os.walk(source):
        relative = os.path.relpath(directory, source)
        target = os.path.join(destination, "" if relative == "." else relative)
        os.makedirs(target, exist_ok=True)
        for file_name in file_names:
            shutil.copy2(os.path.join(directory, file_name), os.path.join(target, file_name))


def _as_directory(path: str, destination: str) -> str:
    """A directory holding the input, whatever shape it arrived in.

    A single uploaded file is linked into a directory of its own rather than
    used from where it landed: main.py streams every upload of a request into
    ONE work directory, so treating a file's parent as an input root would make
    the T2 folder part of the T1 one.
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
        "registered": statuses.count("ok"),
        "failed": statuses.count("failed"),
    }
    logger.info(
        "AREG %s %s: %d/%d registered",
        report["modality"],
        report["automation"],
        report["summary"]["registered"],
        report["summary"]["patients"],
    )

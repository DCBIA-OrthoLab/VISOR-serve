"""AREG -- Automated REGistration of two timepoints onto each other.

Schema only. Every line of pipeline lives in src/; see src/AREGLogic.py for what
the five modes do and which defects of the original CLIs are fixed here.
"""

from base import ArgSpec, Tool

from .src import AREGLogic, catalogs

# The collapsible boxes a client lays this tool's panel out in. AREG's five
# modes share ONE schema, and the two modalities have almost nothing in common
# -- a panel showing every argument offers an intra-oral checkpoint next to a
# CBCT segmentation label. `visible_when` on every mode-specific argument is the
# old widget's mode-switching expressed as data instead of as widget code.
_INPUTS = "Inputs"
_CBCT = "CBCT Registration"
_IOS = "IOS Registration"
_OUTPUTS = "Outputs"

_CBCT_ONLY = {"modality": catalogs.MODALITY_CBCT}
_IOS_ONLY = {"modality": catalogs.MODALITY_IOS}


class AREGTool(Tool):
    name = "AREG"
    arguments = {
        # Never inferred from the input: a .zip can hold either kind of data,
        # and guessing wrong means running the wrong engine on a patient's
        # follow-up scan.
        "modality": ArgSpec(
            label="Input Type",
            type="choice",
            required=True,
            choices=catalogs.MODALITY_CHOICES,
            description="CBCT: cone-beam CT volumes. IOS: intra-oral surface scans",
            section=_INPUTS,
        ),
        "automation": ArgSpec(
            label="Mode",
            type="choice",
            required=True,
            choices=catalogs.AUTOMATION_CHOICES,
            description=(
                "Semi-Automated: you send what the registration needs (CBCT masks, "
                "or already segmented and oriented meshes). Fully-Automated: the "
                "server produces it. Oriented + Fully-Automated (CBCT only): the T1 "
                "scans are oriented first"
            ),
            section=_INPUTS,
        ),
        # The FILE type is declared FIRST, as in AMASSS and CrownSeg:
        # GET /tools publishes types[0] as `type`, and a client keys its file
        # picker -- and its own schema check -- off it, so leading with "folder"
        # makes the argument look like a non-file one client-side and it renders
        # as a bare text field. A .zip therefore reaches run() as an archive and
        # AREGLogic._as_directory unpacks it.
        "t1": ArgSpec(
            label="T1 Folder",
            type=("zip_file", "folder"),
            required=True,
            server_selectable="testfile",
            description=(
                "The first timepoint -- the scans everything is registered ONTO. A "
                "folder sent as a .zip, or the name of a hosted test set (see "
                "GET /tools/AREG/data)"
            ),
            section=_INPUTS,
        ),
        "t2": ArgSpec(
            label="T2 Folder",
            type=("zip_file", "folder"),
            required=True,
            server_selectable="testfile",
            description=(
                "The second timepoint -- the scans that get moved. Paired with T1 by "
                "name up to the timepoint token: 'P1_T1_scan.nii.gz' pairs with "
                "'P1_T2.nii.gz'"
            ),
            section=_INPUTS,
        ),
        "dicom_input": ArgSpec(
            label="DICOM Input",
            type=bool,
            required=False,
            initial=False,
            description=(
                "CBCT only: both folders are zips of DICOM folders, one per patient, "
                "to convert server-side before registering"
            ),
            section=_INPUTS,
            visible_when=_CBCT_ONLY,
        ),
        # ------------------------------------------------------------------
        # CBCT
        # ------------------------------------------------------------------
        "cbct_regions": ArgSpec(
            label="Regions of Reference",
            type="multichoice",
            required=False,
            choices=catalogs.REGION_CHOICES,
            description=(
                "CBCT only: the anatomy the registration is confined to. Each region "
                "is a SEPARATE registration with its own output folder -- registering "
                "on the cranial base and on the mandible answer two different "
                "clinical questions"
            ),
            section=_CBCT,
            visible_when=_CBCT_ONLY,
            ui="inline",
        ),
        "t1_masks": ArgSpec(
            label="T1 Masks",
            type=("zip_file", "folder"),
            required=False,
            description=(
                "Semi-Automated CBCT only: the T1 segmentations to register inside, "
                "sent as a .zip. A mask is matched to its scan by name and has to say "
                "both that it is a segmentation (mask/seg/pred) and which structure it "
                "covers (cb/mand/max), e.g. 'P1_T1_CB_seg.nii.gz'"
            ),
            section=_CBCT,
            visible_when={
                "modality": catalogs.MODALITY_CBCT,
                "automation": catalogs.AUTOMATION_SEMI,
            },
        ),
        "segmentation_label": ArgSpec(
            label="Mask Label",
            type=int,
            required=False,
            initial=0,
            description=(
                "Semi-Automated CBCT only: which label value of a multi-label mask to "
                "register inside. 0 uses the whole mask. A label the mask does not "
                "hold is refused rather than silently falling back to the whole mask"
            ),
            section=_CBCT,
            visible_when={
                "modality": catalogs.MODALITY_CBCT,
                "automation": catalogs.AUTOMATION_SEMI,
            },
        ),
        "segmentation_model": ArgSpec(
            label="Segmentation Models",
            type=str,
            required=False,
            server_selectable="model",
            description=(
                "Fully-Automated CBCT only: the AMASSS model bundle used to produce "
                "the T1 masks (see GET /tools/AREG/data)"
            ),
            section=_CBCT,
            visible_when={
                "modality": catalogs.MODALITY_CBCT,
                "automation": (catalogs.AUTOMATION_FULLY, catalogs.AUTOMATION_ORIENTED),
            },
        ),
        "cbct_reference": ArgSpec(
            label="Orientation Reference",
            type=str,
            required=False,
            server_selectable="model",
            description=(
                "Oriented + Fully-Automated CBCT only: the already-oriented case "
                "defining the frame the T1 scans are put in before registration (see "
                "GET /tools/AREG/data)"
            ),
            section=_CBCT,
            visible_when={
                "modality": catalogs.MODALITY_CBCT,
                "automation": catalogs.AUTOMATION_ORIENTED,
            },
        ),
        # ------------------------------------------------------------------
        # IOS
        # ------------------------------------------------------------------
        # Picking the patch also picks the ARCH: the palate exists only on the
        # maxilla and the mucogingival line only matters on the mandible. A
        # "choice", not a "multichoice", because that is what the CLI's
        # `reg_type` is -- one run registers one arch.
        "ios_patch": ArgSpec(
            label="Register On",
            type="choice",
            required=False,
            choices=catalogs.PATCH_CHOICES,
            description=(
                "IOS only: the region that does not move between the two timepoints. "
                "The palate is predicted by a network and registers the UPPER arches "
                "(the lower ones follow). The mucogingival line is built from 13 "
                "landmarks you supply and registers the LOWER arches on their own"
            ),
            section=_IOS,
            visible_when=_IOS_ONLY,
        ),
        "registration_model": ArgSpec(
            label="Registration Model",
            type=str,
            required=False,
            server_selectable="model",
            description=(
                "Palate patch only: the checkpoint that predicts it (see "
                "GET /tools/AREG/data). The mucogingival patch needs no model"
            ),
            section=_IOS,
            visible_when={
                "modality": catalogs.MODALITY_IOS,
                "ios_patch": catalogs.PATCH_PALATE,
            },
        ),
        "mgl_landmarks": ArgSpec(
            label="Mucogingival Landmarks",
            type=("zip_file", "folder"),
            required=False,
            description=(
                "Mucogingival patch only. Leave empty -- the server predicts the 13 MG "
                "landmarks itself, which is the ordinary case. Send a .zip of Slicer "
                "markups files only to reuse landmarks you already have (one per lower "
                "scan, matched by name: 'P1_T1_Lower_MG_Pred.json' goes with "
                "'P1_T1_Lower.vtk'), which also skips paying for the prediction twice"
            ),
            section=_IOS,
            visible_when={
                "modality": catalogs.MODALITY_IOS,
                "ios_patch": catalogs.PATCH_MGL,
            },
        ),
        "mgl_patch_height": ArgSpec(
            label="Patch Height (mm)",
            type=float,
            required=False,
            initial=5.0,
            description=(
                "Mucogingival patch only: how far the band reaches on each side of "
                "the line, measured ALONG the surface so it cannot leak to the "
                "lingual side. 0 registers on the landmarks alone, with no band at "
                "all -- the control case for measuring what the surface adds"
            ),
            section=_IOS,
            visible_when={
                "modality": catalogs.MODALITY_IOS,
                "ios_patch": catalogs.PATCH_MGL,
            },
        ),
        "ios_reference": ArgSpec(
            label="Orientation Reference",
            type=str,
            required=False,
            server_selectable="model",
            description=(
                "Fully-Automated IOS only: the reference meshes both timepoints are "
                "oriented onto before the patch is predicted (see GET /tools/AREG/data)"
            ),
            section=_IOS,
            visible_when={
                "modality": catalogs.MODALITY_IOS,
                "automation": catalogs.AUTOMATION_FULLY,
            },
        ),
        "output_suffix": ArgSpec(
            label="Suffix",
            type=str,
            required=False,
            initial="Reg",
            description="Added to every output file name, e.g. patient1_CB_Reg.nii.gz",
            section=_OUTPUTS,
        ),
    }
    # One folder per region (CBCT) or per patient (IOS), plus a run report:
    # main.py zips what run() returns and streams the archive, so no zip code
    # lives in this tool.
    output_kind = "files"

    def run(
        self,
        modality: str,
        automation: str,
        t1: str,
        t2: str,
        t1_masks: str = None,
        cbct_regions: dict = None,
        segmentation_label: int = 0,
        segmentation_model: str = None,
        cbct_reference: str = None,
        ios_patch: str = None,
        registration_model: str = None,
        mgl_landmarks: str = None,
        mgl_patch_height: float = 5.0,
        ios_reference: str = None,
        dicom_input: bool = False,
        output_suffix: str = "Reg",
    ) -> str:
        return AREGLogic.main(
            modality=modality,
            automation=automation,
            t1=t1,
            t2=t2,
            t1_masks=t1_masks,
            cbct_regions=cbct_regions,
            segmentation_label=segmentation_label,
            segmentation_model=segmentation_model,
            cbct_reference=cbct_reference,
            ios_patch=ios_patch,
            registration_model=registration_model,
            mgl_landmarks=mgl_landmarks,
            mgl_patch_height=mgl_patch_height,
            ios_reference=ios_reference,
            dicom_input=dicom_input,
            output_suffix=output_suffix,
        )

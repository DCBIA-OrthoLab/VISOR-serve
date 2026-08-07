"""ASO -- Automated Standardized Orientation.

Schema only. Every line of pipeline lives in src/; see src/ASOLogic.py for what
the four modes do and which defects of the original CLIs are fixed here.
"""

from base import ArgSpec, Tool

from .src import catalogs
from .src import ASOLogic


# The collapsible boxes a client lays this tool's panel out in, and the reason
# they are declared at all: ASO's four modes share ONE schema, so a panel that
# shows every argument shows 130 CBCT landmarks next to 32 teeth when a run
# uses one or the other. The two selection sections below are mutually
# exclusive by construction (`visible_when` on every argument they hold), which
# is the old Slicer module's four-page QStackedWidget expressed as data instead
# of as widget code -- see SADT_tools_analysis/ASO.md.
_INPUTS = "Inputs"
_CBCT_SELECTION = "Landmark Reference"
_IOS_SELECTION = "Teeth & Landmarks"
_OUTPUTS = "Outputs"

_CBCT_ONLY = {"modality": catalogs.MODALITY_CBCT}
_IOS_ONLY = {"modality": catalogs.MODALITY_IOS}


class ASOTool(Tool):
    name = "ASO"
    arguments = {
        # Never inferred from the input's extension: a .zip can hold either kind
        # of data, and guessing wrong means orienting a patient against the
        # wrong reference and calling it a success.
        "modality": ArgSpec(
            label="Input Type",
            type="choice",
            required=True,
            choices=catalogs.MODALITY_CHOICES,
            description=(
                "CBCT: cone-beam CT volumes. IOS: intra-oral surface scans"
            ),
            section=_INPUTS,
        ),
        "automation": ArgSpec(
            label="Mode",
            type="choice",
            required=True,
            choices=catalogs.AUTOMATION_CHOICES,
            description=(
                "Semi-Automated: you send the landmarks (.mrk.json) alongside "
                "each scan. Fully-Automated: CBCT landmarks are predicted "
                "server-side, IOS meshes are oriented from their tooth labels"
            ),
            section=_INPUTS,
        ),
        "input": ArgSpec(
            label="Scan / Landmark Folder",
            type=("volume_or_zip_file", "surface_file", "folder"),
            required=True,
            description=(
                "One CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz) or "
                "one intra-oral mesh (.vtk/.vtp/.stl/.obj/.off), or a folder of "
                "them for batch orientation (sent as a .zip). In Semi-Automated "
                "mode the landmark files go in the same folder as the scans they "
                "belong to"
            ),
            section=_INPUTS,
        ),
        "reference": ArgSpec(
            label="Reference",
            type=("zip_file", "folder"),
            required=True,
            server_selectable="model",
            description=(
                "The already-oriented case defining the target coordinate frame. "
                "Pick one hosted on this server (see GET /tools/ASO/data) or send "
                "your own. CBCT: one landmark file. IOS: one file per jaw, named "
                "so a token says which jaw it is (e.g. 'Gold_Upper.vtk')"
            ),
            section=_INPUTS,
        ),
        "landmark_models": ArgSpec(
            label="Landmark Models",
            type=str,
            required=False,
            server_selectable="model",
            description=(
                "Fully-Automated CBCT only. Leave empty -- the server then picks "
                "the bundle matching the scans from the ones hosted for the "
                "landmark tool, which is almost always what you want. Name one "
                "(see GET /tools/ASO/data) only to force a particular vintage; "
                "note that list also holds the reference bundles, which are not "
                "landmark weights"
            ),
            section=_INPUTS,
            # The one argument that is neither CBCT-wide nor IOS-wide: it is
            # read by exactly one of the four modes, and offering it to the
            # other three is how a user comes to believe Semi-Automated needs
            # a model bundle.
            visible_when={
                "modality": catalogs.MODALITY_CBCT,
                "automation": catalogs.AUTOMATION_FULLY,
            },
        ),
        "cbct_landmarks": ArgSpec(
            label="Landmarks",
            type="multichoice",
            required=False,
            choices=catalogs.CBCT_LANDMARK_CHOICES,
            description=(
                "CBCT only: the landmarks to register on. At least 3 are needed; "
                "the defaults are the points the published reference planes are "
                "built on. A landmark missing from your file or from the "
                "reference is reported in ASO_report.json, not silently dropped"
            ),
            section=_CBCT_SELECTION,
            visible_when=_CBCT_ONLY,
            ui="tabs",
            groups=catalogs.CBCT_LANDMARK_GROUPS,
        ),
        "ios_teeth": ArgSpec(
            label="Teeth",
            type="multichoice",
            required=False,
            choices=catalogs.TOOTH_CHOICES,
            description=(
                "IOS only: the teeth to register on. Fully-Automated needs "
                "exactly 3 or 4 per jaw, spread across the arch"
            ),
            section=_IOS_SELECTION,
            visible_when=_IOS_ONLY,
            # "spread across the arch" is a spatial instruction, and a column
            # of 32 check boxes is the one layout that cannot show whether a
            # selection is spread or clustered.
            ui="grid",
            groups=catalogs.TOOTH_GROUPS,
        ),
        "ios_landmark_types": ArgSpec(
            label="Landmark Types",
            type="multichoice",
            required=False,
            choices=catalogs.IOS_LANDMARK_TYPE_CHOICES,
            description=(
                "Semi-Automated IOS only: which landmark per tooth to register "
                "on. Combined with the selected teeth into the labels looked up "
                "in your landmark file, e.g. UR6 + O -> 'UR6O'"
            ),
            section=_IOS_SELECTION,
            visible_when={
                "modality": catalogs.MODALITY_IOS,
                "automation": catalogs.AUTOMATION_SEMI,
            },
            ui="inline",
        ),
        "ios_jaws": ArgSpec(
            label="Jaws",
            type="multichoice",
            required=False,
            choices=catalogs.JAW_CHOICES,
            description="IOS only: which jaws to orient",
            section=_IOS_SELECTION,
            visible_when=_IOS_ONLY,
            ui="inline",
        ),
        "ios_occlusion": ArgSpec(
            label="Occlusion",
            type="choice",
            required=False,
            choices=catalogs.OCCLUSION_CHOICES,
            description=(
                "IOS only: orienting each jaw on its own is the default. Letting "
                "one drive the other applies a single transform to both, which "
                "preserves the occlusion but only makes sense if the two meshes "
                "were in occlusion to begin with"
            ),
            section=_IOS_SELECTION,
            visible_when=_IOS_ONLY,
        ),
        "dicom_input": ArgSpec(
            label="DICOM Input",
            type=bool,
            required=False,
            initial=False,
            description=(
                "CBCT only: the input is a zip of DICOM folders, one per patient, "
                "to convert server-side before orienting"
            ),
            section=_INPUTS,
            visible_when=_CBCT_ONLY,
        ),
        "output_suffix": ArgSpec(
            label="Suffix",
            type=str,
            required=False,
            initial="Or",
            description="Added to every output file name, e.g. patient1_Or.nii.gz",
            section=_OUTPUTS,
        ),
    }
    output_kind = "files"

    def run(
        self,
        modality: str,
        automation: str,
        input: str,
        reference: str,
        landmark_models: str = None,
        cbct_landmarks: dict = None,
        ios_teeth: dict = None,
        ios_landmark_types: dict = None,
        ios_jaws: dict = None,
        ios_occlusion: str = None,
        dicom_input: bool = False,
        output_suffix: str = "Or",
    ) -> str:
        return ASOLogic.main(
            input=input,
            modality=modality,
            automation=automation,
            reference=reference,
            landmark_models=landmark_models,
            cbct_landmarks=cbct_landmarks,
            ios_teeth=ios_teeth,
            ios_landmark_types=ios_landmark_types,
            ios_jaws=ios_jaws,
            ios_occlusion=ios_occlusion,
            dicom_input=dicom_input,
            output_suffix=output_suffix,
        )

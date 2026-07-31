"""ASO -- Automated Standardized Orientation.

Schema only. Every line of pipeline lives in src/; see src/ASOLogic.py for what
the four modes do and which defects of the original CLIs are fixed here.
"""

from base import ArgSpec, Tool

from .src import catalogs
from .src import ASOLogic


class ASOTool(Tool):
    name = "ASO"
    arguments = {
        # Never inferred from the input's extension: a .zip can hold either kind
        # of data, and guessing wrong means orienting a patient against the
        # wrong reference and calling it a success.
        "modality": ArgSpec(
            type="choice",
            required=True,
            choices=catalogs.MODALITY_CHOICES,
            description=(
                "CBCT: cone-beam CT volumes. IOS: intra-oral surface scans"
            ),
        ),
        "automation": ArgSpec(
            type="choice",
            required=True,
            choices=catalogs.AUTOMATION_CHOICES,
            description=(
                "Semi-Automated: you send the landmarks (.mrk.json) alongside "
                "each scan. Fully-Automated: CBCT landmarks are predicted "
                "server-side, IOS meshes are oriented from their tooth labels"
            ),
        ),
        "input": ArgSpec(
            type=("volume_or_zip_file", "surface_file", "folder"),
            required=True,
            description=(
                "One CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz) or "
                "one intra-oral mesh (.vtk/.vtp/.stl/.obj/.off), or a folder of "
                "them for batch orientation (sent as a .zip). In Semi-Automated "
                "mode the landmark files go in the same folder as the scans they "
                "belong to"
            ),
        ),
        "reference": ArgSpec(
            type=("zip_file", "folder"),
            required=True,
            server_selectable="model",
            description=(
                "The already-oriented case defining the target coordinate frame. "
                "Pick one hosted on this server (see GET /tools/ASO/data) or send "
                "your own. CBCT: one landmark file. IOS: one file per jaw, named "
                "so a token says which jaw it is (e.g. 'Gold_Upper.vtk')"
            ),
        ),
        "landmark_models": ArgSpec(
            type=str,
            required=False,
            server_selectable="model",
            description=(
                "Fully-Automated CBCT only: name of the landmark model bundle to "
                "predict with (see GET /tools/ASO/data), laid out as one folder "
                "per landmark holding its per-scale weights"
            ),
        ),
        "cbct_landmarks": ArgSpec(
            type="multichoice",
            required=False,
            choices=catalogs.CBCT_LANDMARK_CHOICES,
            description=(
                "CBCT only: the landmarks to register on. At least 3 are needed; "
                "the defaults are the points the published reference planes are "
                "built on. A landmark missing from your file or from the "
                "reference is reported in ASO_report.json, not silently dropped"
            ),
        ),
        "ios_teeth": ArgSpec(
            type="multichoice",
            required=False,
            choices=catalogs.TOOTH_CHOICES,
            description=(
                "IOS only: the teeth to register on. Fully-Automated needs "
                "exactly 3 or 4 per jaw, spread across the arch"
            ),
        ),
        "ios_landmark_types": ArgSpec(
            type="multichoice",
            required=False,
            choices=catalogs.IOS_LANDMARK_TYPE_CHOICES,
            description=(
                "Semi-Automated IOS only: which landmark per tooth to register "
                "on. Combined with the selected teeth into the labels looked up "
                "in your landmark file, e.g. UR6 + O -> 'UR6O'"
            ),
        ),
        "ios_jaws": ArgSpec(
            type="multichoice",
            required=False,
            choices=catalogs.JAW_CHOICES,
            description="IOS only: which jaws to orient",
        ),
        "ios_occlusion": ArgSpec(
            type="choice",
            required=False,
            choices=catalogs.OCCLUSION_CHOICES,
            description=(
                "IOS only: orienting each jaw on its own is the default. Letting "
                "one drive the other applies a single transform to both, which "
                "preserves the occlusion but only makes sense if the two meshes "
                "were in occlusion to begin with"
            ),
        ),
        "dicom_input": ArgSpec(
            type=bool,
            required=False,
            initial=False,
            description=(
                "CBCT only: the input is a zip of DICOM folders, one per patient, "
                "to convert server-side before orienting"
            ),
        ),
        "output_suffix": ArgSpec(
            type=str,
            required=False,
            initial="Or",
            description="Added to every output file name, e.g. patient1_Or.nii.gz",
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

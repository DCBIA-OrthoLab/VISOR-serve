"""AMASSS -- Automatic Multi-Anatomical Skull Structure Segmentation.

Segments skull structures (mandible, maxilla, cranial base, cervical
vertebrae, upper airway, skin, and the three masks consumed by AREG) from
oriented CBCT scans, using one nnUNet v2 model per structure.

The whole schema is declared here; the pipeline lives in src/AMASSSLogic.py.
Note that the structure catalog (`choice_groups`) is served FROM THE SERVER:
a client renders its grouped checkboxes from GET /tools and never hardcodes
the list, so adding a structure the day its model ships is a one-line change
in AMASSSLogic.STRUCTURE_GROUPS with no client release.

AMASSS is also meant to be called by other modules, not just by the Slicer
GUI. Over HTTP that is simply POST /run/AMASSS. Server-side, another tool
should import `AMASSSLogic.segment()` directly: it returns the produced files
and a report, with no zip round trip.
"""

from typing import Optional

from base import SELECTION_TYPE, ArgSpec, Tool

from .src import AMASSSLogic


class AMASSSTool(Tool):
    name = "AMASSS"
    arguments = {
        # One argument serves both use cases: a single scan, or a zip of a
        # folder of scans for a batch. A server-side test file may also be a
        # bare scan or a folder (see GET /tools/AMASSS/data).
        "input": ArgSpec(
            type="volume_or_zip_file",
            required=True,
            server_selectable="testfile",
            description=(
                "A CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz), or a zip "
                "archive of a folder of scans for batch segmentation."
            ),
        ),
        # Server-side only: the client sends the NAME of a model bundle
        # hosted on the server, never the models themselves. A bundle is a
        # folder holding one subfolder per structure code (MAND/, MAX/, ...).
        "model": ArgSpec(
            type=str,
            required=True,
            server_selectable="model",
            description=(
                "Name of a model bundle hosted on the server (see GET /tools/AMASSS/data): "
                "a folder containing one subfolder per structure code, each holding an "
                "nnUNet v2 model (<CODE>/**/*__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth)."
            ),
        ),
        "structures": ArgSpec(
            type=SELECTION_TYPE,
            required=True,
            multiple=True,
            choices=AMASSSLogic.STRUCTURE_CODES,
            choice_groups=AMASSSLogic.STRUCTURE_GROUPS,
            default=AMASSSLogic.DEFAULT_STRUCTURES,
            description=(
                "Anatomical structures to segment. Send either a list of codes "
                "(\"MAND,MAX\") or a {display name: true/false} mapping matching the "
                "groups published in choice_groups."
            ),
        ),
        "merge": ArgSpec(
            type=SELECTION_TYPE,
            required=False,
            multiple=True,
            choices=AMASSSLogic.MERGE_MODES,
            choice_groups=AMASSSLogic.MERGE_MODE_GROUPS,
            default=AMASSSLogic.DEFAULT_MERGE_MODES,
            description=(
                "MERGED: one multi-label file per scan. SEPARATE: one binary file per "
                "structure. Both may be selected. Defaults to MERGED."
            ),
        ),
        "prediction_ID": ArgSpec(
            type=str,
            required=False,
            default="Pred",
            description="Suffix used in output file names, e.g. scan_Pred_MAND.nii.gz.",
        ),
        "generate_surface": ArgSpec(
            type=bool,
            required=False,
            default=False,
            description="Also export a 3D surface (.vtk) alongside each segmentation.",
        ),
        "surface_smoothing": ArgSpec(
            type=int,
            required=False,
            default=5,
            description="Smoothing iterations for the surfaces (0-95). Ignored unless generate_surface is set.",
        ),
    }
    output_kind = "file"

    def run(
        self,
        input: str,
        model: str,
        structures: list,
        merge: Optional[list] = None,
        prediction_ID: str = "Pred",
        generate_surface: bool = False,
        surface_smoothing: int = 5,
    ) -> str:
        return AMASSSLogic.main(
            input=input,
            model=model,
            structures=structures,
            merge=merge,
            prediction_ID=prediction_ID,
            generate_surface=generate_surface,
            surface_smoothing=surface_smoothing,
        )

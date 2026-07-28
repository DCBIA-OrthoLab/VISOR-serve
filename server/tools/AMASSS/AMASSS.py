"""AMASSS -- Automatic Multi-Anatomical Skull Structure Segmentation.

Segments skull structures (mandible, maxilla, cranial base, cervical
vertebrae, upper airway, skin, and the three masks consumed by AREG) from
oriented CBCT scans, using one nnUNet v2 model per structure.

The whole schema is declared here; the pipeline lives in src/AMASSSLogic.py.
The structure catalog is served FROM THE SERVER as a "multichoice" argument:
a client renders its check boxes from GET /tools and never hardcodes the list,
so adding a structure the day its model ships is a one-line change in
AMASSSLogic.STRUCTURE_GROUPS with no client release.

AMASSS is also meant to be called by other modules, not just by the Slicer
GUI. Over HTTP that is simply POST /run/AMASSS. Server-side, another tool
should import `AMASSSLogic.segment()` directly: it takes structure codes,
returns the produced files and a report, and never zips anything.
"""

from typing import Optional

from base import ArgSpec, Selection, Tool

from .src import AMASSSLogic


class AMASSSTool(Tool):
    name = "AMASSS"
    arguments = {
        # One argument, two use cases: a single scan, or a whole folder of them
        # for a batch. "folder" first so that a .zip -- which both types accept
        # -- resolves as a folder and is unpacked by main.py before run(), the
        # tool then receiving a real directory. Anything else falls through to
        # the volume types.
        # "input": ArgSpec(
        #     type=("folder", "volume_or_zip_file"),
        #     required=True,
        #     server_selectable="testfile",
        #     description=(
        #         "A CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz), or a folder of "
        #         "scans for batch segmentation (sent as a .zip archive)."
        #     ),
        # ),
        # Server-side only: the client sends the NAME of a model bundle hosted
        # on the server, never the models themselves. A bundle is a folder
        # holding one subfolder per structure code (MAND/, MAX/, ...).
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
        # Check boxes. The option names are the human-readable structure names,
        # and their declared booleans are the defaults -- both come from
        # AMASSSLogic's catalog, so the list is written down exactly once and
        # the client shows whatever the server currently has models for.
        "structures": ArgSpec(
            type="multichoice",
            required=True,
            choices=AMASSSLogic.STRUCTURE_CHOICES,
            description="Anatomical structures to segment.",
        ),
        "merge": ArgSpec(
            type="multichoice",
            required=False,
            choices=AMASSSLogic.MERGE_CHOICES,
            description=(
                "Merged: one multi-label file per scan. Separated: one binary file per "
                "structure. Both may be selected."
            ),
        ),
        "prediction_ID": ArgSpec(
            type=str,
            required=False,
            description="Suffix used in output file names, e.g. scan_Pred_MAND.nii.gz.",
        ),
        "generate_surface": ArgSpec(
            type=bool,
            required=False,
            description="Also export a 3D surface (.vtk) alongside each segmentation.",
        ),
        "surface_smoothing": ArgSpec(
            type=int,
            required=False,
            description="Smoothing iterations for the surfaces (0-95). Ignored unless generate_surface is set.",
        ),
    }
    # One folder per scan plus a run report: several files, which main.py zips
    # and streams back. No zip code lives in this tool.
    output_kind = "files"

    def run(
        self,
        input: str,
        model: str,
        structures: Selection,
        merge: Optional[Selection] = None,
        prediction_ID: str = "Pred",
        generate_surface: bool = False,
        surface_smoothing: int = 5,
    ) -> str:
        # `merge` is declared optional, so it may legitimately be absent. It
        # should then arrive as its declared defaults (ArgSpec.default), but a
        # None is tolerated here too -- see the note in AMASSSLogic.main and
        # the report that goes with this change.
        return AMASSSLogic.main(
            input=input,
            model=model,
            structures=structures,
            merge=merge,
            prediction_ID=prediction_ID,
            generate_surface=generate_surface,
            surface_smoothing=surface_smoothing,
        )

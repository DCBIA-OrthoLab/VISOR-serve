"""AMASSS -- Automatic Multi-Anatomical Skull Structure Segmentation.

Segments skull structures (mandible, maxilla, cranial base, cervical vertebrae,
upper airway, skin, and the three masks AREG consumes) from oriented CBCT
scans, with one nnUNet v2 model per structure.

Only the schema lives here; the pipeline is in src/AMASSSLogic.py. Another
server-side tool should import `AMASSSLogic.segment()` instead of going through
this wrapper: it speaks structure codes, returns the produced files plus a
report, and zips nothing.
"""

from base import ArgSpec, Tool

from .src import AMASSSLogic


class AMASSSTool(Tool):
    name = "AMASSS"
    arguments = {
        # One argument, two use cases: a single scan, or a whole folder of them
        # for a batch (sent as a .zip). The FILE type is declared first, like
        # example_tool's ("csv_file", "folder"): GET /tools publishes types[0]
        # as `type`, and a client keys its file picker -- and its own schema
        # check -- off it, so leading with "folder" makes the argument look like
        # a non-file one client-side. A .zip therefore reaches run() as an
        # archive; discover_scans unpacks it.
        "input": ArgSpec(
            type=("volume_or_zip_file", "folder"),
            required=True,
            description=(
                "A CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz), or a folder of "
                "scans for batch segmentation (sent as a .zip archive)"
            ),
        ),
        # Server-side only: the client sends the NAME of a model bundle hosted
        # on the server, never the models themselves.
        "model": ArgSpec(
            type=str,
            required=True,
            server_selectable="model",
            description=(
                "Name of a model bundle hosted on the server (see GET /tools/AMASSS/data): "
                "one subfolder per structure code (MAND/, MAX/, ...), each holding an "
                "nnUNet v2 model"
            ),
        ),
        # Check boxes. Option names and their declared booleans both come from
        # AMASSSLogic's catalog, so the structure list is written down exactly
        # once and the client shows whatever the server has models for.
        "structures": ArgSpec(
            type="multichoice",
            required=True,
            choices=AMASSSLogic.STRUCTURE_CHOICES,
            description="Anatomical structures to segment",
        ),
        "merge": ArgSpec(
            type="multichoice",
            required=False,
            choices=AMASSSLogic.MERGE_CHOICES,
            description=(
                "Merged: one multi-label file per scan. Separated: one binary file per "
                "structure. Both may be selected"
            ),
        ),
        "prediction_ID": ArgSpec(
            type=str,
            required=False,
            initial="Pred",
            description="Suffix used in output file names, e.g. scan_Pred_MAND.nii.gz",
        ),
        "generate_surface": ArgSpec(
            type=bool,
            required=False,
            initial=False,
            description="Also export a 3D surface (.vtk) alongside each segmentation",
        ),
        # `initial` is what the client's spin box starts at. Without it the box
        # starts at 0 and, since a form always sends its widgets, run()'s own
        # default of 5 was never reached -- every surface came out unsmoothed.
        # Keep the two in step.
        "surface_smoothing": ArgSpec(
            type=int,
            required=False,
            initial=5,
            description="Smoothing iterations for the surfaces (0-95), ignored without generate_surface",
        ),
    }
    # One folder per scan plus a run report: main.py zips what run() returns and
    # streams the archive back, so no zip code lives in this tool.
    output_kind = "files"

    def run(
        self,
        input: str,
        model: str,
        structures: dict,
        merge: dict,
        prediction_ID: str = "Pred",
        generate_surface: bool = False,
        surface_smoothing: int = 5,
    ) -> str:
        # `structures` and `merge` are base.Selection mappings keyed by the
        # display names the schema published; AMASSSLogic.main translates them
        # into the structure codes segment() speaks.
        return AMASSSLogic.main(
            input=input,
            model=model,
            structures=structures,
            merge=merge,
            prediction_ID=prediction_ID,
            generate_surface=generate_surface,
            surface_smoothing=surface_smoothing,
        )

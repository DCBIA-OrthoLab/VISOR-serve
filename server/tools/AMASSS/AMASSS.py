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
        # One argument, two use cases: a single scan, or a folder of them for a
        # batch (sent as a .zip). The FILE type is declared FIRST because
        # GET /tools publishes types[0] as `type` and a client keys its file
        # picker off it -- leading with "folder" makes the argument look like a
        # non-file one. A .zip therefore reaches run() as an archive, which
        # discover_scans unpacks.
        "input": ArgSpec(
            type=("volume_or_zip_file", "folder"),
            required=True,
            # Lets the client run against (or download) the reference scan
            # hosted under DATA/AMASSS/testfiles/ instead of uploading its own.
            server_selectable="testfile",
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
        # "multichoice", NOT "choice": both forms may be selected at once, and
        # the two types hand run() different values (a Selection dict here, a
        # bare string for "choice"). This shipped as "choice" once, and the
        # string fell through _codes_from's iterable branch as a tuple of
        # CHARACTERS: every run "succeeded" with zero segmentation files.
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
        # Marching cubes runs on the original scan grid, so a 0.33mm CBCT gives
        # a triangle per voxel face: 3.5M across a five-structure run. That is
        # minutes of parsing in a viewer for detail the mask does not have, it
        # being accurate to about half a voxel. 90 drops nine triangles in ten
        # and moves the cranial-base surface by 0.059mm on average (max
        # 0.692mm); 0 keeps the raw mesh. `initial` must match run()'s default.
        "surface_decimation": ArgSpec(
            type=int,
            required=False,
            initial=90,
            description=(
                "Percentage of surface triangles to drop (0-99). 90 keeps the shape to "
                "well under a voxel while making the .vtk usable; 0 keeps every triangle. "
                "Ignored without generate_surface"
            ),
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
        surface_decimation: int = 90,
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
            surface_decimation=surface_decimation,
        )

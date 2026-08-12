"""BatchDentalSeg -- dental CT/CBCT segmentation, one scan or a whole cohort.

Segments teeth and jaw structures with the DentalSegmentator family of nnUNet
v2 models. Four models are offered and they do not label the same things: see
`dental_model`'s choices, and `labels` in the run report, which says what the
integers in the returned volume mean.

Only the schema lives here; the pipeline is in src/BatchDentalSegLogic.py.
Another server-side tool should call `BatchDentalSegLogic.segment()` instead of
going through this wrapper: it returns the produced files plus a report, and
zips nothing.
"""

from base import ArgSpec, Tool

from .src import BatchDentalSegLogic, surface_export

_INPUTS = "Inputs"
_OUTPUTS = "Outputs"


class BatchDentalSegTool(Tool):
    name = "BatchDentalSeg"
    arguments = {
        # One argument, two use cases: a single scan or a folder of them for a
        # batch (sent as a .zip). The FILE type is declared FIRST because
        # GET /tools publishes types[0] as `type` and a client keys its file
        # picker off it -- leading with "folder" makes the argument look like a
        # non-file one. A .zip therefore reaches run() as an archive, which
        # discover_scans unpacks.
        "input": ArgSpec(
            label="Scan or Folder",
            section=_INPUTS,
            type=("volume_or_zip_file", "folder"),
            required=True,
            server_selectable="testfile",
            description=(
                "A dental CT/CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz), or a "
                "folder of scans for batch segmentation (sent as a .zip archive)"
            ),
        ),
        # Server-side only: the client sends the NAME of a hosted bundle, never
        # the weights. That name IS the model -- it selects the weights and the
        # label table together. A second "which labels" argument would let a
        # caller pair one bundle with another's table, and the result would be
        # a plausible volume with every structure named wrong.
        "model": ArgSpec(
            label="Model",
            section=_INPUTS,
            type=str,
            required=True,
            server_selectable="model",
            description=(
                "Name of a model hosted on the server (see "
                "GET /tools/BatchDentalSeg/data). DentalSegmentator and "
                "PediatricDentalSeg label 5 segments (the maxilla is inside Upper "
                "Skull); NasoMaxillaDentSeg separates the maxilla; UniversalLab labels "
                "every tooth individually. The run report says what the values mean"
            ),
        ),
        "separate_segments": ArgSpec(
            label="Also write one file per segment",
            section=_OUTPUTS,
            type=bool,
            required=False,
            initial=False,
            description=(
                "In addition to the multi-label volume, write a binary mask per segment "
                "the model actually found. Empty segments are not written"
            ),
        ),
        "prediction_ID": ArgSpec(
            label="Prediction ID",
            section=_OUTPUTS,
            type=str,
            required=False,
            initial="Seg",
            description="Suffix used in output file names, e.g. scan_Seg.nii.gz",
        ),
        # The label volume is always written; these are meshes built from it.
        # Every option off by default: a UniversalLab scan is 55 structures,
        # and meshing them all is minutes of CPU and hundreds of MB nobody
        # asked for. glTF is deliberately absent -- see src/surface_export.py.
        "export_formats": ArgSpec(
            label="Also export meshes",
            section=_OUTPUTS,
            type="multichoice",
            required=False,
            ui="inline",
            choices={name: False for name in surface_export.ALL_FORMATS},
            description=(
                "Surface meshes built from the segmentation, one file per structure "
                "(plus one whole-scan file for 'VTK (merged)'). Only VTK carries the "
                "structure colours; STL has no colour field and OBJ would need a "
                "companion .mtl. The segmentation volume is written either way"
            ),
        ),
        # `initial` is what the client's spin box starts at. Without it the box
        # starts at Qt's 0 and, since a form always sends its widgets, run()'s
        # own default is never reached -- every mesh would come out unsmoothed.
        # Keep the two in step.
        "surface_smoothing": ArgSpec(
            label="Mesh smoothing",
            section=_OUTPUTS,
            type=int,
            required=False,
            initial=5,
            description=(
                "Smoothing iterations for the meshes (0-95). Ignored when no export "
                "format is selected"
            ),
        ),
        # Marching cubes runs on the original scan grid, so a 0.33mm CBCT gives
        # a triangle per voxel face. Measured on AMASSS's cranial base against
        # a 0.33mm voxel: 90% drops nine triangles in ten and moves the surface
        # by 0.059mm on average (p95 0.171mm), a fifth of a voxel. `initial`
        # must match run()'s default.
        "surface_decimation": ArgSpec(
            label="Mesh decimation (%)",
            section=_OUTPUTS,
            type=int,
            required=False,
            initial=90,
            description=(
                "Percentage of mesh triangles to drop (0-99). 90 keeps the shape to "
                "well under a voxel while making the files usable; 0 keeps every "
                "triangle. Ignored when no export format is selected"
            ),
        ),
    }
    # One segmentation per scan in the input's own tree, plus a run report:
    # main.py zips what run() returns and streams the archive.
    output_kind = "files"
    # A cohort is minutes to hours of work whose only visible sign used to be
    # elapsed time, and a failure on scan 27 used to cost the twenty-six that
    # had already worked. With `X-Result-Delivery: stream` each scan is
    # reported as it starts and finishes, and its files leave the server the
    # moment they exist. A client that does not ask for it gets exactly the
    # single archive it always got.
    streaming = True

    def run(
        self,
        input: str,
        model: str,
        separate_segments: bool = False,
        prediction_ID: str = "Seg",
        export_formats=None,
        surface_smoothing: int = 5,
        surface_decimation: int = 90,
        emit=None,
    ) -> str:
        return BatchDentalSegLogic.main(
            input=input,
            model=model,
            separate_segments=separate_segments,
            prediction_ID=prediction_ID,
            export_formats=export_formats,
            surface_smoothing=surface_smoothing,
            surface_decimation=surface_decimation,
            emit=emit,
        )

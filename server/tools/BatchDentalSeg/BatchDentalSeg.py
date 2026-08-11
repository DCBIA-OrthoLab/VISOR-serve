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

from .src import BatchDentalSegLogic

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
    }
    # One segmentation per scan in the input's own tree, plus a run report:
    # main.py zips what run() returns and streams the archive.
    output_kind = "files"

    def run(
        self,
        input: str,
        model: str,
        separate_segments: bool = False,
        prediction_ID: str = "Seg",
    ) -> str:
        return BatchDentalSegLogic.main(
            input=input,
            model=model,
            separate_segments=separate_segments,
            prediction_ID=prediction_ID,
        )

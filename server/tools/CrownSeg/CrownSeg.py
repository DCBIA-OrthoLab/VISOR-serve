"""CrownSeg -- per-tooth labelling of intraoral surface scans.

Adds a point-data array naming the tooth each vertex belongs to. That array is
the precondition for ALI's IOS landmark identification and for the IOS modes
of ASO, AREG and FlexReg, which is why it is a tool of its own rather than a
private helper of any one of them: burying it inside ALI would make those
three depend on ALI, and ALI's IOS half needs pytorch3d -- so one absent
dependency would take four tools out of the registry instead of one.

Only the schema lives here; the work is in src/CrownSegLogic.py. Another
server-side tool should import `CrownSegLogic.segment_crowns()` rather than
going through this wrapper: it returns the produced meshes directly, and zips
nothing.
"""

from base import ArgSpec, Tool

from .src import CrownSegLogic


class CrownSegTool(Tool):
    name = "CrownSeg"
    arguments = {
        # One argument, two use cases: a single mesh, or a folder of them for
        # a batch (sent as a .zip). The file type is declared FIRST, as in
        # AMASSS: GET /tools publishes types[0] as `type`, and a client keys
        # its file picker off it, so leading with "folder" would make the
        # argument look like a non-file one client-side.
        "input": ArgSpec(
            type=("surface_or_zip_file", "folder"),
            required=True,
            server_selectable="testfile",
            description=(
                "An intraoral surface scan (.vtk/.stl), or a folder of them for batch "
                "segmentation (sent as a .zip archive)"
            ),
        ),
        # Optional, unlike AMASSS's: a caller that names nothing gets the
        # server's configured checkpoint (settings.CROWNSEG_MODEL), which is
        # what makes crown segmentation a service another tool can simply ask
        # for. Still server-side only -- the weights never travel.
        "model": ArgSpec(
            type=str,
            required=False,
            server_selectable="model",
            description=(
                "Name of a segmentation checkpoint hosted on the server (see "
                "GET /tools/CrownSeg/data). Leave empty to use the server's default"
            ),
        ),
        "array_name": ArgSpec(
            type=str,
            required=False,
            initial=CrownSegLogic.DEFAULT_ARRAY_NAME,
            description=(
                "Name of the point-data array the tooth labels are written to. Leave as "
                "Universal_ID unless a downstream tool expects another name"
            ),
        ),
        "numbering": ArgSpec(
            type="choice",
            required=False,
            choices={"Universal": True, "FDI": False},
            description="Tooth numbering system used for the labels",
        ),
        "suffix": ArgSpec(
            type=str,
            required=False,
            initial=CrownSegLogic.DEFAULT_SUFFIX,
            description="Suffix used in output file names, e.g. scan_Seg.vtk",
        ),
        "skip_segmented": ArgSpec(
            type=bool,
            required=False,
            initial=True,
            description=(
                "Pass a mesh that already carries tooth labels through unchanged instead "
                "of re-running the network on it"
            ),
        ),
    }
    # One mesh per input plus a run report: main.py zips what run() returns.
    output_kind = "files"

    def run(
        self,
        input: str,
        model: str = None,
        array_name: str = CrownSegLogic.DEFAULT_ARRAY_NAME,
        numbering: str = "Universal",
        suffix: str = CrownSegLogic.DEFAULT_SUFFIX,
        skip_segmented: bool = True,
    ) -> str:
        return CrownSegLogic.main(
            input=input,
            model=model,
            array_name=array_name,
            suffix=suffix,
            numbering=numbering,
            skip_segmented=skip_segmented,
        )

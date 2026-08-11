"""ALI -- Automatic Landmark Identification, on CBCT scans or intraoral scans.

Places anatomical landmarks and returns Slicer markups files. One tool, two
engines that share nothing but their output format:

* **CBCT** -- one deep-RL agent per landmark walks the volume at 1 mm and then
  at 0.3 mm until it converges on the point;
* **IOS** -- per tooth, the mesh is rendered from a dozen viewpoints and a 2D
  UNet predicts masks that are projected back onto the surface.

Only the schema lives here; everything else is in src/, where `ALILogic`
decides which engine applies. Another server-side tool should call
`ALILogic.identify()` rather than this wrapper: it returns the run report and
zips nothing.

There is no `mode` argument, on purpose: a `.zip` can hold either kind of data
and a DICOM series has no extension at all, so only the data distinguishes
them. The server looks, and an input holding both kinds is a 422, not a guess.

The cost is that the schema cannot say "this argument only applies in mode X":
both selections are optional, both are always rendered, and one is inert on any
given run. Emptying the selection for the mode that actually ran is a 422
naming the argument to fill in (see `ALILogic.identify`).
"""

from base import ArgSpec, Tool

from .src import ALILogic
from .src.ALI_CBCT import landmarks as cbct_catalog
from .src.ALI_IOS import landmarks as ios_catalog

# The collapsible boxes a client lays this panel out in. Both engines'
# selections are always shown, so the least a panel can do is not interleave
# them: a CBCT user reads one box and ignores the other.
_INPUTS = "Inputs"
_CBCT = "CBCT landmarks"
_IOS = "IOS landmarks"
_OUTPUTS = "Outputs"


class ALITool(Tool):
    name = "ALI"
    arguments = {
        # Two FILE types and no "folder", so a batch reaches run() as an
        # archive that ALILogic unpacks. Declaring "folder" would have main.py
        # extract it, but the client would then have to know which of the two
        # kinds it is sending in order to pick a file filter. A cohort, and any
        # DICOM series, is a directory: the Slicer client offers a folder
        # picker here and zips the selection before uploading.
        "input": ArgSpec(
            label="Scan or Folder",
            section=_INPUTS,
            type=("volume_or_zip_file", "surface_or_zip_file"),
            required=True,
            server_selectable="testfile",
            description=(
                "A CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz), an IOS surface "
                "(.vtk/.stl), or a .zip archive of a folder of either -- DICOM series are "
                "recognised inside the archive and converted automatically"
            ),
        ),
        # Server-side only: the client sends the NAME of a hosted bundle, never
        # the weights. Optional -- the server detects CBCT vs IOS from the data
        # and picks the hosted bundle whose layout matches. Naming one is only
        # needed to disambiguate several bundles of the same kind, and a name
        # that does not match the detected mode is a 422, not a guess.
        "model": ArgSpec(
            label="Model Bundle",
            section=_INPUTS,
            type=str,
            required=False,
            server_selectable="model",
            description=(
                "Name of a model bundle hosted on the server (see GET /tools/ALI/data). "
                "Leave empty to let the server pick the bundle matching the detected mode. "
                "CBCT bundles hold <landmark>/<scale>/*.pth; IOS bundles hold checkpoints "
                "named with an 'O' or 'C' token and an 'Upper' or 'Lower' one, "
                "e.g. Upper_O_model.pth"
            ),
        ),
        # Optional because the OTHER mode's request must not be blocked by it.
        # Every option is on by default: a landmark whose weights the chosen
        # bundle lacks costs a line in the run report, whereas a region left
        # off by default is one the user never finds.
        "cbct_regions": ArgSpec(
            label="Regions",
            section=_CBCT,
            type="multichoice",
            required=False,
            choices=cbct_catalog.REGION_CHOICES,
            ui="inline",
            description="CBCT only: anatomical regions to predict",
        ),
        # The counterpart of `cbct_regions`, and how another server-side tool
        # drives this engine. A region is the right granularity for a human
        # placing a full set of points and the wrong one for a caller needing a
        # named few: ASO registers on seven landmarks straddling two regions,
        # so asking by region would run 58 agents to use 7.
        #
        # All options off by default, unlike the regions: "off" is what an
        # omitted multichoice arrives as, so the default state means "nothing
        # said here, the regions decide".
        "landmarks": ArgSpec(
            label="Individual landmarks",
            section=_CBCT,
            type="multichoice",
            required=False,
            choices=cbct_catalog.LANDMARK_CHOICES,
            ui="tabs",
            groups=cbct_catalog.LANDMARK_GROUPS,
            description=(
                "CBCT only: predict exactly these landmarks. Leave every box unchecked "
                "to select by region instead -- naming any landmark here REPLACES the "
                "region selection rather than narrowing it"
            ),
        ),
        "ios_networks": ArgSpec(
            label="Landmark families",
            section=_IOS,
            type="multichoice",
            required=False,
            choices=ios_catalog.NETWORK_CHOICES,
            ui="inline",
            description="IOS only: landmark families to predict",
        ),
        # No `initial`: an optional field left empty is dropped from the
        # request, so the default lives once, in run()'s signature. The Slicer
        # module shows "Pred" as a placeholder rather than pre-filling it, for
        # the same reason.
        "prediction_ID": ArgSpec(
            label="Prediction ID",
            section=_OUTPUTS,
            type=str,
            required=False,
            description="Suffix used in output file names, e.g. scan_lm_Pred.mrk.json",
        ),
    }
    # One markups file per scan, in the input's own tree, plus a run report:
    # main.py zips what run() returns and streams the archive.
    output_kind = "files"

    def run(
        self,
        input: str,
        model: str = None,
        cbct_regions: dict = None,
        landmarks: dict = None,
        ios_networks: dict = None,
        prediction_ID: str = "Pred",
    ) -> str:
        # `cbct_regions`, `landmarks` and `ios_networks` are base.Selection
        # mappings keyed by the display names the schema published; ALILogic
        # translates them into the codes each engine speaks.
        return ALILogic.main(
            input=input,
            model=model,
            cbct_regions=cbct_regions,
            landmarks=landmarks,
            ios_networks=ios_networks,
            prediction_ID=prediction_ID,
        )

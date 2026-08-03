"""ALI -- Automatic Landmark Identification, on CBCT scans or intraoral scans.

Places anatomical landmarks and returns Slicer markups files. One tool, two
engines that share nothing but their output format:

* **CBCT** -- one deep-RL agent per landmark walks the volume at 1 mm and then
  at 0.3 mm until it converges on the point;
* **IOS** -- per tooth, the mesh is rendered from a dozen viewpoints and a 2D
  UNet predicts masks that are projected back onto the surface.

Only the schema lives here; everything else is in src/, where `ALILogic`
decides which engine applies and `ALI_CBCT/` and `ALI_IOS/` implement them.
Another server-side tool should call `ALILogic.identify()` rather than going
through this wrapper: it returns the run report, with the produced files named
in it, and zips nothing.

**There is no `mode` argument, on purpose.** A `.zip` can hold either kind of
data and a DICOM series has no extension at all, so nothing in the request
distinguishes them -- only the data does. The server looks, and an input
holding both kinds is a 422 rather than a guess.

The cost of one tool for two engines is that the schema cannot say "this
argument only applies in mode X": both selections are optional, both are
always rendered by the client, and one of them is inert on any given run. That
is stated in each description, and emptying the selection for the mode that
actually ran is a 422 naming the argument to fill in -- see
`ALILogic.identify`.
"""

from base import ArgSpec, Tool

from .src import ALILogic
from .src.ALI_CBCT import landmarks as cbct_catalog
from .src.ALI_IOS import landmarks as ios_catalog


class ALITool(Tool):
    name = "ALI"
    arguments = {
        # Two FILE types and no "folder": a batch therefore reaches run() as
        # an archive, which ALILogic unpacks. Declaring "folder" here instead
        # would have main.py extract it, but the client would then have to
        # guess which of the two kinds it is sending to pick a file filter --
        # and that is exactly what it cannot know.
        #
        # A cohort, and *any* DICOM series, is a directory: the Slicer client
        # offers a folder picker for this argument and zips the selection
        # before uploading.
        "input": ArgSpec(
            type=("volume_or_zip_file", "surface_or_zip_file"),
            required=True,
            server_selectable="testfile",
            description=(
                "A CBCT scan (.nii/.nii.gz/.nrrd/.nrrd.gz/.gipl/.gipl.gz), an IOS surface "
                "(.vtk/.stl), or a .zip archive of a folder of either -- DICOM series are "
                "recognised inside the archive and converted automatically"
            ),
        ),
        # Server-side only: the client sends the NAME of a bundle hosted on
        # the server, never the weights. Optional, like the mode it depends
        # on: the server already detects CBCT vs IOS from the data, so when
        # no name is sent it also picks the hosted bundle whose CONTENT
        # matches the detected mode (each engine recognises its own layout).
        # Naming one is only needed to disambiguate when several bundles of
        # the same kind are hosted -- and a name that does not match the
        # detected mode is a 422, not a guess.
        "model": ArgSpec(
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
            type="multichoice",
            required=False,
            choices=cbct_catalog.REGION_CHOICES,
            description="CBCT only: anatomical regions to predict",
        ),
        "ios_networks": ArgSpec(
            type="multichoice",
            required=False,
            choices=ios_catalog.NETWORK_CHOICES,
            description="IOS only: landmark families to predict",
        ),
        # No `initial`: an optional field left empty is dropped from the
        # request, so the default lives once, in run()'s signature. The Slicer
        # module shows "Pred" as a placeholder rather than pre-filling it, for
        # the same reason.
        "prediction_ID": ArgSpec(
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
        ios_networks: dict = None,
        prediction_ID: str = "Pred",
    ) -> str:
        # `cbct_regions` and `ios_networks` are base.Selection mappings keyed
        # by the display names the schema published; ALILogic translates them
        # into the codes each engine speaks.
        return ALILogic.main(
            input=input,
            model=model,
            cbct_regions=cbct_regions,
            ios_networks=ios_networks,
            prediction_ID=prediction_ID,
        )

from base import Tool, ArgSpec
from .src.SurgMovPredLogic import SurgMovPred as smp


class SurgMovPredTool(Tool):
    name = "SurgMovPred"
    arguments = {
        # The model is server-side only: the client sends the *name* of a model
        # hosted on the server (list them via GET /tools/SurgMovPred/data),
        # never uploads one. main.py resolves the name through data_store, so
        # run() receives a path to the model folder (or legacy zip archive)
        # served directly from the data store -- no zipping required.
        "input": ArgSpec(type="zip_file", required=True, description="Input data: a zip archive of a folder containing CSV / XLSX / ODS files; a server-side test file may also be a single such file", server_selectable="testfile"),
        "model": ArgSpec(type=str, required=True, description="Name of a model hosted on the server (see GET /tools/SurgMovPred/data); a folder (or zip archive) containing one or more stacking_package.pkl files", server_selectable="model"),
    }
    output_kind = "file"

    def run(self, model: str, input: str) -> str:
        return smp.main(model, input)



from base import Tool, ArgSpec
from .src.surg_mov_pred_logic import Surg_mov_pred as smp


class SurgMovPredTool(Tool):
    name = "surg_mov_pred"
    arguments = {
        # The model is server-side only: the client sends the *name* of a model
        # hosted on the server (list them via GET /tools/surg_mov_pred/data),
        # never uploads one. main.py resolves the name through data_store, so
        # run() still receives a path to the model zip archive.
        "model": ArgSpec(type=str, required=True, description="Name of a model hosted on the server (see GET /tools/surg_mov_pred/data); a zip archive containing one or more stacking_package.pkl files", server_selectable="model"),
        "input": ArgSpec(type="zip_file", required=True, description="Input data: a zip archive of a folder containing CSV / XLSX / ODS files", server_selectable="testfile"),
    }
    output_kind = "file"

    def run(self, model: str, input: str) -> str:
        return smp.main(model, input)



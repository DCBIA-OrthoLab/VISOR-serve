from base import Tool, ArgSpec
from .src.surg_mov_pred_logic import Surg_mov_pred as smp


class SurgMovPredTool(Tool):
    name = "surg_mov_pred"
    arguments = {
        "model": ArgSpec(type="zip_file", required=True, description="Model package: a zip archive containing one or more stacking_package.pkl files"),
        "input": ArgSpec(type="zip_file", required=True, description="Input data: a zip archive of a folder containing CSV / XLSX / ODS files"),
    }
    output_kind = "file"

    def run(self, model: str, input: str) -> str:
        return smp.main(model, input)



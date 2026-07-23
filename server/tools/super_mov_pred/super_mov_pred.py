from base import Tool, ArgSpec
from .src.supermovpred_logic import Super_mov_pred as smp


class SuperMovPredTool(Tool):
    name = "super_mov_pred"
    arguments = {
        "model": ArgSpec(type="file", required=True, description="Path to the model package (zip file)"),
        "input": ArgSpec(type="file", required=True, description="Path to the input data (folder of CSV / Excel / ODS files)"),
    }
    output_kind = "file"

    def run(self, model: str, input: str) -> str:
        return smp.main(model, input)



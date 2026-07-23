from base import Tool, ArgSpec
from src.supermovpred import Super_mov_pred as smp


class SuperMovPredTool(Tool):
    name = "super_mov_pred"
    arguments = {
        "model": ArgSpec(type="file", required=True, description="Path to the model package (zip file)"),
        "input": ArgSpec(type="file", required=True, description="Path to the input data (folder of CSV / Excel / ODS files)"),
        "output": ArgSpec(type="file", required=True, description="Path to the output folder where predictions will be saved"),
    }
    output_kind = "file"

    def run(self, model: str, input: str, output: str) -> str:
        return smp.main(model, input, output)



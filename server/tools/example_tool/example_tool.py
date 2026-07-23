"""Example tool: demonstrates a mix of argument types (str, file, numbers).

Shows how to declare a "file" argument (its value is the path where main.py
already streamed the upload to disk) alongside plain scalar arguments, and how
optional arguments with a default work. Copy this file as a starting point
for a real tool.

# NOTE: the file extension whitelist is enforced centrally in config.py
# (ALLOWED_EXTENSIONS), not per-tool. If this tool needs to accept mesh
# formats (.stl, .obj, .ply, ...) rather than just .nii/.nii.gz, extend
# ALLOWED_EXTENSIONS accordingly.
"""

import os
from typing import Optional

from base import ArgSpec, Tool


class ExampleTool(Tool):
    name = "example_tool"
    arguments = {
        "label": ArgSpec(type=str, required=True, description="Free-text label for this run"),
        "file": ArgSpec(type="file", required=True, description="3D file to process (e.g. .nii.gz)"),
        "threshold": ArgSpec(type=float, required=True, description="Numeric threshold parameter"),
        "iterations": ArgSpec(type=int, required=False, description="Optional number of iterations"),
    }
    output_kind = "text"

    def run(self, label: str, file: str, threshold: float, iterations: Optional[int] = None) -> str:
        file_size_bytes = os.path.getsize(file)
        return (
            f"label={label} file_size={file_size_bytes}B "
            f"threshold={threshold} iterations={iterations}"
        )

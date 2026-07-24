"""Example tool: demonstrates a mix of argument types (str, file, numbers).

Shows how to declare a specific file-typed argument (its value is the path
where main.py already streamed the upload to disk) alongside plain scalar
arguments, and how optional arguments with a default work. Copy this whole
folder as a starting point for a real tool.

# NOTE: "nifti_file" is one of the file types declared in base.FILE_TYPES,
# which maps it to its accepted extension(s) (.nii/.nii.gz here). This is
# what the extension check (main.py) and the client (GET /tools) both use --
# see base.py to add a new file type, or use the generic "file" type to fall
# back to config.ALLOWED_EXTENSIONS instead.
"""

import os
from typing import Optional

from base import ArgSpec, Tool


class ExampleTool(Tool):
    name = "example_tool"
    arguments = {
        "label": ArgSpec(type=str, required=True, description="Free-text label for this run"),
        "file": ArgSpec(type="nifti_file", required=True, description="3D volume to process (.nii or .nii.gz)"),
        "threshold": ArgSpec(type=float, required=True, description="Numeric threshold parameter"),
        "iterations": ArgSpec(type=int, required=False, description="Optional number of iterations"),
    }
    output_kind = "text"

    def run(self, label: str, file: str, threshold: float, iterations: Optional[int] = None) -> str:
        file_size_bytes = os.path.getsize(file)
        with open(file, "rb") as f:
            preview_bytes = f.read(200)
        content_preview = preview_bytes.decode("utf-8", errors="replace")
        return (
            f"label={label} file_size={file_size_bytes}B "
            f"threshold={threshold} iterations={iterations}\n"
            f"content_preview={content_preview!r}"
        )

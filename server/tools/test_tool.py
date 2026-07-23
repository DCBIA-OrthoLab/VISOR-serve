"""Minimal tool: proves the client -> server -> client round trip.

# TODO: to add a new tool, copy this file, rename the class, give it a
# unique `name`, declare its `arguments`, and implement `run`. Drop it in
# this tools/ package -- registry.py will pick it up automatically at
# startup. No changes needed anywhere else (no new route, no manual list).
"""

from base import ArgSpec, Tool


class TestTool(Tool):
    name = "test_tool"
    arguments = {
        "text_1": ArgSpec(type=str, required=True, description="First string"),
        "text_2": ArgSpec(type=str, required=True, description="Second string"),
    }
    output_kind = "text"

    def run(self, text_1: str, text_2: str) -> str:
        return f"{text_1} {text_2}"

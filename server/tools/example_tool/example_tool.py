"""Example tool: a multi-type input, declared option sets, a multi-file output.

Three things worth copying from here:

1. `"input"` accepts EITHER a single .csv OR a whole folder of tabular files.
   The client zips the folder (HTTP has no notion of a directory) and the
   server unpacks it before `run()` is called, so `run()` always receives a
   real local path -- and `input.kind` says which of the declared types it
   got, instead of it having to guess from the extension. See base.FILE_TYPES
   for the available types and base.ResolvedPath for `.kind`.

2. `"outputs"` and `"preview_format"` pick from a fixed set of options the
   tool declares in `choices`. "multichoice" is a group of check boxes and
   reaches run() as a base.Selection (a dict holding every option); "choice"
   is a combo box and reaches run() as the chosen name. The declared booleans
   double as the defaults, so they are written down exactly once.

3. `output_kind = "files"` lets `run()` return SEVERAL paths. main.py zips
   them and streams the archive back; no zip code lives in the tool. Returning
   the output directory itself works too -- see the end of run().

# NOTE: to accept a file kind that doesn't exist yet (e.g. ".vtk"), add one
# entry to base.FILE_TYPES -- that is the only core edit a new tool should
# ever need. Use the generic "file" type to fall back to
# config.ALLOWED_EXTENSIONS instead.
"""

import os
from typing import Optional

import file_utils
from base import ArgSpec, Tool, ToolArgumentError


class ExampleTool(Tool):
    name = "example_tool"
    arguments = {
        "label": ArgSpec(type=str, required=True, description="Free-text label for this run"),
        "input": ArgSpec(
            # A tuple = "any of these". Declaration order breaks ties when two
            # types share an extension, so put the one that should win first.
            type=("csv_file", "folder"),
            required=True,
            description="A single .csv file, or a folder of .csv/.xlsx/.ods files sent as a .zip archive",
        ),
        "threshold": ArgSpec(type=float, required=True, description="Numeric threshold parameter"),
        "iterations": ArgSpec(type=int, required=False, description="Optional number of iterations"),
        "outputs": ArgSpec(
            # Check boxes: any number of them, each declared with its initial
            # state. run() receives every option as a base.Selection, so the
            # defaults below are the ONLY place they are written down.
            type="multichoice",
            required=False,
            choices={"summary": True, "preview": True, "columns": False},
            description="Which result files to produce",
        ),
        "preview_format": ArgSpec(
            # Combo box: exactly one option, the True one being the default.
            type="choice",
            required=False,
            choices={"csv": True, "json": False},
            description="Format of the preview file",
        ),
    }
    output_kind = "files"

    def run(
        self,
        label: str,
        input: str,
        threshold: float,
        outputs: dict,
        preview_format: str,
        iterations: Optional[int] = None,
    ) -> list:
        # `input` is a base.ResolvedPath: a plain path string that also knows
        # which declared type it was resolved as. That is the whole point of a
        # multi-type argument -- no os.path.isdir() guessing needed here.
        if input.kind == "folder":
            table = file_utils.load_tabular_directory(input)
            source = f"folder, {len(os.listdir(input))} entries"
        else:
            table = file_utils.load_tabular_file(input)
            source = "single file"

        # Outputs go in a scratch dir under TEMP_DIR, which main.py removes
        # once the response has been streamed (see file_utils.make_scratch_dir).
        output_dir = file_utils.make_scratch_dir("example_tool_")

        numeric = table.select_dtypes("number")
        above_threshold = int((numeric > threshold).sum().sum()) if not numeric.empty else 0

        # `outputs` is a base.Selection: a dict holding EVERY declared option,
        # so `outputs["summary"]` is always safe -- no .get(name, False). Use
        # `outputs.selected` to loop over the enabled ones instead.
        if not outputs.selected:
            # A constraint spanning several arguments can't be expressed in the
            # schema. Raising ToolArgumentError (rather than any other
            # exception) is what makes main.py answer 422 with this message
            # instead of a blank 500.
            raise ToolArgumentError("Select at least one output to produce.")

        produced = []
        if outputs["summary"]:
            summary_path = os.path.join(output_dir, "summary.txt")
            with open(summary_path, "w") as summary_file:
                summary_file.write(
                    f"label={label}\n"
                    f"input_kind={input.kind} ({source})\n"
                    f"rows={len(table)} columns={len(table.columns)}\n"
                    f"threshold={threshold} values_above_threshold={above_threshold}\n"
                    f"iterations={iterations}\n"
                    f"outputs={','.join(outputs.selected)}\n"
                )
            produced.append(summary_path)

        if outputs["preview"]:
            # `preview_format` is one of the names declared in its choices, so
            # this needs no fallback branch: validate() rejected anything else.
            preview_path = os.path.join(output_dir, f"preview.{preview_format}")
            head = table.head(iterations or 5)
            if preview_format == "json":
                head.to_json(preview_path, orient="records", indent=2)
            else:
                head.to_csv(preview_path, index=False)
            produced.append(preview_path)

        if outputs["columns"]:
            columns_path = os.path.join(output_dir, "columns.txt")
            with open(columns_path, "w") as columns_file:
                columns_file.write("\n".join(str(column) for column in table.columns) + "\n")
            produced.append(columns_path)

        # Returning a list of paths -> main.py bundles them into
        # example_tool_output.zip. Returning `output_dir` instead would zip the
        # whole folder with its contents at the archive root; either is valid.
        return produced

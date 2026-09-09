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

# To accept a file kind that doesn't exist yet (e.g. ".vtk"), add one
# entry to base.FILE_TYPES -- that is the only core edit a new tool should
# ever need. Use the generic "file" type to fall back to
# config.ALLOWED_EXTENSIONS instead.
"""

import csv
import glob
import json
import os
from typing import Optional

import file_utils
from base import ArgSpec, Tool, ToolArgumentError


# ---------------------------------------------------------------------------
# Reading the input, with the standard library and nothing else
# ---------------------------------------------------------------------------
# An IN-PROCESS tool runs inside the API's own virtualenv, and that venv holds
# fastapi, uvicorn, python-multipart, pydantic-settings and NOTHING heavier --
# `server/requirements-api.txt` is what it installs, and a test pins it there.
# So this tool may use the standard library and the server's own modules, and
# nothing else. It used to call `file_utils.load_tabular_*`, which reach for
# pandas: absent from that venv, so every call answered 500 with
# `ModuleNotFoundError: No module named 'pandas'` -- at run time only, the
# import being lazy, which is why it survived startup and the schema.
#
# A PACKAGED tool has no such limit: it brings its own interpreter and pins
# whatever it likes. That is the trade this file demonstrates from the other
# side, and the reason to keep it small.


class Table:
    """A CSV as columns and rows of text, plus which columns hold numbers.

    Deliberately not a DataFrame. A column counts as numeric when every value
    in it that is not blank parses as a float, which is the rule pandas'
    type inference applies to a CSV -- a blank becoming NaN and comparing
    false against any threshold, as it does here.
    """

    def __init__(self, columns: list, rows: list):
        self.columns = columns
        self.rows = rows
        self.numeric_columns = [
            index
            for index, _ in enumerate(columns)
            if self._is_numeric(index)
        ]

    def _is_numeric(self, index: int) -> bool:
        seen = False
        for row in self.rows:
            value = row[index].strip() if index < len(row) else ""
            if not value:
                continue
            try:
                float(value)
            except ValueError:
                return False
            seen = True
        return seen

    def count_above(self, threshold: float) -> int:
        total = 0
        for row in self.rows:
            for index in self.numeric_columns:
                value = row[index].strip() if index < len(row) else ""
                if value and float(value) > threshold:
                    total += 1
        return total

    def head(self, count: int) -> list:
        return self.rows[:count]

    def cell(self, row: list, index: int):
        """One value, as a number where its column is numeric."""
        value = row[index].strip() if index < len(row) else ""
        if not value:
            return None
        if index not in self.numeric_columns:
            return value
        number = float(value)
        return int(number) if number.is_integer() and "." not in value else number

    def __len__(self) -> int:
        return len(self.rows)


def read_csv(path: str) -> Table:
    """One `.csv` as a Table. The first line names the columns."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration:
            raise ToolArgumentError(f"{os.path.basename(path)} is empty.")
        return Table(columns, [row for row in reader if any(cell.strip() for cell in row)])


def read_csv_directory(path: str) -> Table:
    """Every `.csv` directly in a folder, stacked, sorted by name.

    Only `.csv`: the single-file half of this argument declares `csv_file`,
    and one vocabulary is better than two that disagree. A folder holding
    `.xlsx` is refused by name rather than silently skipped.
    """
    files = sorted(
        set(glob.glob(os.path.join(path, "*.csv")) + glob.glob(os.path.join(path, "*.CSV")))
    )
    if not files:
        raise ToolArgumentError(f"No .csv file found directly in {os.path.basename(path)}.")

    first = read_csv(files[0])
    for other in files[1:]:
        table = read_csv(other)
        if table.columns != first.columns:
            raise ToolArgumentError(
                f"{os.path.basename(other)} has different columns from "
                f"{os.path.basename(files[0])}: stacking them would misalign the rows."
            )
        first.rows.extend(table.rows)
    return Table(first.columns, first.rows)


class ExampleTool(Tool):
    name = "Example_Tool"
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
            table = read_csv_directory(input)
            source = f"folder, {len(os.listdir(input))} entries"
        else:
            table = read_csv(input)
            source = "single file"

        # Outputs go in a scratch dir under TEMP_DIR, which main.py removes
        # once the response has been streamed (see file_utils.make_scratch_dir).
        output_dir = file_utils.make_scratch_dir("example_tool_")

        above_threshold = table.count_above(threshold)

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
                records = [
                    {name: table.cell(row, index) for index, name in enumerate(table.columns)}
                    for row in head
                ]
                with open(preview_path, "w") as preview_file:
                    json.dump(records, preview_file, indent=2)
            else:
                with open(preview_path, "w", newline="") as preview_file:
                    writer = csv.writer(preview_file)
                    writer.writerow(table.columns)
                    writer.writerows(head)
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

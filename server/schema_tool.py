"""A tool the server has never imported.

    /tools/<name>/
    ├── .schema.json     what run() takes, and the hash of the src/ it was read from
    ├── .venv/           the tool's own interpreter and dependencies
    └── src/             the tool's code -- which this process never touches

`SchemaTool` turns that JSON into the same `Tool` object the registry has
always held, so `GET /tools`, `validate()`, `main.py`'s upload handling and
`data_store` resolution all work on it unchanged. The only thing it cannot do
is run in-process: there is nothing to import, so `invoke` always dispatches.

**The two declarations are kept apart on purpose.** `.schema.json` is
generated from the tool's source and is the same wherever the tool is
installed; `deployment.toml` is what THIS server does with it (see
deployment.py). Which arguments can be filled from this server's DATA_DIR is
not a property of the tool.

The type vocabulary is narrow by design -- `path`, `str`, `int`, `float`,
`bool`, `list[str]` -- and maps onto the server's existing one:

| schema      | ArgSpec                | GET /tools                          |
|-------------|------------------------|-------------------------------------|
| `path`      | `"file"`               | extensions null -> ALLOWED_EXTENSIONS |
| `str` ...   | `str`, `int`, ...      | unchanged                           |
| `list[str]` | `base.LIST_TYPE`       | a new type string on the wire       |

A schema cannot express what the ported tools declare today -- a specific file
type and its extensions, a catalog of choices, a section, a label, a
`visible_when`. Those arguments are published as a generic file or a plain
scalar, which is honest but plainer: a client renders a file picker with no
extension filter instead of one that only offers .nii.gz.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from base import LIST_TYPE, ArgSpec, Tool, ToolSchemaError
from deployment import ToolDeployment
from schema_hash import hash_source_tree

logger = logging.getLogger("inference_server")

SCHEMA_FILE = ".schema.json"
SRC_DIR_NAME = "src"

# schema type -> the type an ArgSpec declares. "path" becomes the GENERIC file
# type: the schema says an argument is a path and nothing about which
# extensions are acceptable, so the server falls back to ALLOWED_EXTENSIONS
# rather than inventing a restriction the tool never asked for.
ARGUMENT_TYPES = {
    "path": "file",
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list[str]": LIST_TYPE,
}

# What `returns` means for main.py's response handling.
#   "path"  -> one output file, streamed
#   "paths" -> several, or a directory: zipped into an archive
#   "text"  -> any JSON-serializable value, returned as JSON
RETURN_KINDS = {
    "path": "file",
    "paths": "files",
    "text": "text",
    "json": "text",
}
DEFAULT_RETURN_KIND = "text"

# Per-argument keys this server reads. `description` is not in the frozen
# contract's example but is read if present -- a client shows it, and the
# alternative is a panel of unexplained fields.
_ARGUMENT_KEYS = ("type", "required", "default", "description")

_TOP_LEVEL_KEYS = ("name", "description", "arguments", "returns", "source_hash")


class SchemaError(Exception):
    """The schema cannot be used: unreadable, malformed, or declaring a type
    this server does not know. The tool is skipped and reported, the way any
    tool that fails to load is."""


class SourceHashMismatch(Exception):
    """The schema no longer describes the source next to it. Fatal, and
    deliberately so: every other failure costs one tool, while a stale schema
    means the server validates requests against a signature that has changed
    under it -- silently accepting arguments run() no longer takes, and
    refusing ones it now does."""


def read_schema(folder: str) -> dict:
    path = os.path.join(folder, SCHEMA_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            schema = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SchemaError(f"Cannot read {path}: {exc}")
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: expected a JSON object.")
    return schema


def verify_source_hash(folder: str, schema: dict) -> None:
    """Check the schema against the source tree it claims to describe.

    Raises SourceHashMismatch (fatal) when the two disagree, and SchemaError
    (skip this tool) when there is nothing to compare -- an unverifiable
    schema must not serve either, but it endangers only itself.
    """
    declared = schema.get("source_hash")
    if not declared:
        raise SchemaError(
            f"{os.path.join(folder, SCHEMA_FILE)}: no 'source_hash'. It cannot be checked "
            f"against src/, so it cannot be trusted to describe it."
        )

    src_dir = os.path.join(folder, SRC_DIR_NAME)
    try:
        actual = hash_source_tree(src_dir)
    except FileNotFoundError as exc:
        raise SchemaError(str(exc))

    if actual != declared:
        raise SourceHashMismatch(
            f"Tool '{schema.get('name', os.path.basename(folder))}': {SCHEMA_FILE} was generated "
            f"from a different {SRC_DIR_NAME}/ than the one installed "
            f"(schema {declared[:12]}..., source {actual[:12]}...). Regenerate the schema where "
            f"the tool is packaged, or reinstall the source it was generated from."
        )


def _argument_spec(
    tool_name: str, argument_name: str, declaration, deployment: ToolDeployment
) -> ArgSpec:
    where = f"Tool '{tool_name}', argument '{argument_name}'"
    if not isinstance(declaration, dict):
        raise SchemaError(f"{where}: expected an object with at least a 'type'.")

    unknown = sorted(set(declaration) - set(_ARGUMENT_KEYS))
    if unknown:
        # Warned, not refused: this is the seam between two repositories, and a
        # field one side adds must not stop the other from starting. It still
        # has to be visible, because a silently dropped field is a feature that
        # simply never appears.
        logger.warning(
            "%s: ignoring unknown schema key(s) %s -- this server reads %s.",
            where,
            unknown,
            list(_ARGUMENT_KEYS),
        )

    declared_type = declaration.get("type")
    if declared_type not in ARGUMENT_TYPES:
        raise SchemaError(
            f"{where}: unknown type {declared_type!r}. Expected one of {list(ARGUMENT_TYPES)}."
        )

    required = declaration.get("required", True)
    if not isinstance(required, bool):
        raise SchemaError(f"{where}: 'required' must be true or false.")

    selectable = deployment.server_selectable.get(argument_name)
    if selectable is not None and declared_type != "path":
        # A server-side file standing in for a str/int argument is exactly the
        # SurgMovPred case and is legitimate -- but only the deployment can say
        # so, and only for an argument that ends up as a path in run(). Anything
        # else means the two files disagree about what this argument is.
        raise SchemaError(
            f"{where}: deployment.toml marks it server_selectable, but the tool declares it "
            f"as {declared_type!r} rather than 'path'."
        )

    return ArgSpec(
        type=ARGUMENT_TYPES[declared_type],
        required=required,
        description=declaration.get("description", ""),
        server_selectable=selectable,
        # Advisory, exactly as for a ported tool: the value a client pre-fills
        # its widget with. Not applied server-side -- an omitted optional
        # argument is left out of job.json entirely, so the tool's own Python
        # default applies and stays the single source of truth. Dropped for a
        # path, where there is a file picker to pre-fill and nothing to put in
        # it; a schema declaring one is not an error.
        initial=None if declared_type == "path" else declaration.get("default"),
    )


class SchemaTool(Tool):
    """A tool declared by its `.schema.json` and run in its own interpreter."""

    def __init__(self, folder: str, schema: dict, deployment: ToolDeployment):
        name = schema.get("name")
        if not name or not isinstance(name, str):
            raise SchemaError(f"{os.path.join(folder, SCHEMA_FILE)}: no 'name'.")

        unknown = sorted(set(schema) - set(_TOP_LEVEL_KEYS))
        if unknown:
            logger.warning(
                "Tool '%s': ignoring unknown schema key(s) %s -- this server reads %s.",
                name,
                unknown,
                list(_TOP_LEVEL_KEYS),
            )

        arguments = schema.get("arguments", {})
        if not isinstance(arguments, dict):
            raise SchemaError(f"Tool '{name}': 'arguments' must be an object.")

        undeclared = sorted(set(deployment.server_selectable) - set(arguments))
        if undeclared:
            raise SchemaError(
                f"Tool '{name}': deployment.toml marks {undeclared} as server_selectable, but "
                f"the tool declares no such argument(s)."
            )

        self.name = name
        self.folder = folder
        # Read and kept, but NOT published: GET /tools has no tool-level
        # description field, and adding one changes the response shape the
        # Slicer client is built against. It waits for a client release.
        self.description = schema.get("description", "")
        self.arguments = {
            argument_name: _argument_spec(name, argument_name, declaration, deployment)
            for argument_name, declaration in arguments.items()
        }
        self.output_kind = RETURN_KINDS.get(schema.get("returns"), DEFAULT_RETURN_KIND)
        self.source_hash = schema.get("source_hash", "")

    def invoke(self, args: dict) -> Any:
        """Validate, then run in the tool's own interpreter.

        SADT_DISPATCH_MODE is not consulted: it decides how a tool the server
        IMPORTED is executed, and this one was never imported. There is no
        in-process path to fall back to.
        """
        cleaned = self.validate(args)
        from dispatch import dispatch

        return dispatch(self, cleaned)

    def run(self, **kwargs):
        raise RuntimeError(
            f"Tool '{self.name}' is declared by its {SCHEMA_FILE} and has no in-process "
            f"implementation; it runs through dispatch (see invoke)."
        )


def load_tool(folder: str, config) -> SchemaTool:
    """Build the tool declared by `folder`, after checking its schema is the
    one its source produced.

    `config` is the whole DeploymentConfig rather than one tool's entry,
    because which entry applies is only known once the schema has been read:
    deployment.toml is keyed by TOOL name, the name the client sends and the
    contract's `[tools.amasss]` uses.
    """
    schema = read_schema(folder)
    verify_source_hash(folder, schema)

    name = schema.get("name")
    if isinstance(name, str) and name and name != os.path.basename(folder.rstrip(os.sep)):
        # Not cosmetic: dispatch.py finds the interpreter at
        # <TOOLS_DIR>/<tool name>/.venv/bin/python, so a folder named anything
        # else is a tool that registers and then cannot be run.
        raise SchemaError(
            f"Tool '{name}' is installed in a folder named "
            f"'{os.path.basename(folder.rstrip(os.sep))}'. The folder must be named after the "
            f"tool: its interpreter is looked up by tool name."
        )

    tool = SchemaTool(folder, schema, config.for_tool(name))
    try:
        tool.check_schema()
    except ToolSchemaError as exc:
        raise SchemaError(str(exc))
    return tool


def has_schema(folder: str) -> bool:
    return os.path.isfile(os.path.join(folder, SCHEMA_FILE))

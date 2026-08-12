"""Discovery of the tools this server serves, from two sources.

**A tool declared by a `.schema.json`** (the one this is moving to). A folder
under TOOLS_DIR holding `.schema.json`, `.venv/` and `src/`. The server reads
the JSON, checks it against the hash of the source next to it, and builds the
tool from it -- it imports NOTHING. That is the whole point: a tool's
dependencies then have nothing to agree with the server's, and two tools
wanting incompatible versions of numpy or torch are no longer each other's
problem.

**A tool imported into this process** (what every tool here still is). A
folder under tools/ containing exactly one recognized file,
tools/<name>/<name>.py -- the file name must match the folder name -- defining
one or more Tool subclasses. Any other file in the folder is ignored by
discovery, though the recognized file is free to import from them.

Dropping a `.schema.json` into a folder is what moves a tool from the second
to the first: a folder that has one is never imported.

**A tool that will not load is SKIPPED, never fatal** -- missing dependency,
syntax error, a schema that cannot be generated. With 15+ tools, one
unavailable model must not block all the others. The failure is reported as
loudly as possible (see _record_failure), kept in FAILED_TOOLS, and surfaced
again by get_tool().

A schema whose `source_hash` no longer matches the source beside it is not a
failure at all: it is a stale cache, and it is regenerated (see
schema_tool.resolve_schema). Only a tool that can neither be read nor
described is skipped.
"""

import importlib
import logging
import os
import traceback

try:
    import tools as tools_package
except ModuleNotFoundError:
    # There is no tools/ package at all, which is not a failure: the
    # deployment image ships the server WITHOUT the in-process tools (its
    # virtualenv holds fastapi and nothing heavier), and serves everything
    # from TOOLS_DIR instead. This is also what the end state looks like once
    # every tool has been repackaged.
    tools_package = None

from base import Tool
from config import settings
from deployment import deployment_config
from schema_tool import SchemaTool, is_packaged, load_tool

# This module runs at import time, BEFORE main.py reaches its own
# logging.basicConfig call: without this one, the INFO summary below would be
# swallowed entirely. basicConfig is a no-op once handlers exist, so main.py's
# identical call later changes nothing.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("inference_server")

_BANNER = "=" * 78
_RULE = "-" * 78

# Tool FOLDER name -> one-line reason it is missing from TOOLS. Keyed by folder
# rather than by tool name because a module that fails to import never gets far
# enough to tell us the name it would have declared.
FAILED_TOOLS: dict = {}


def _record_failure(folder: str, exc: Exception) -> None:
    """Report a tool that won't be registered. Must be called from an `except`
    block: the full traceback is what makes the failure actionable."""
    reason = f"{type(exc).__name__}: {exc}"
    FAILED_TOOLS[folder] = reason
    logger.error(
        "\n%s\n"
        "  TOOL FAILED TO LOAD:  %s\n"
        "  %s\n\n"
        "  This tool is NOT registered. Every other tool still works.\n"
        "%s\n%s%s",
        _BANNER,
        folder,
        reason,
        _RULE,
        traceback.format_exc(),
        _BANNER,
    )


def _tool_folders(root: str) -> list:
    """Candidate tool folders directly under `root`, in a stable order.

    A leading underscore or dot excludes a folder from discovery, which is how
    a fixture like tools/_dispatch_probe/ stays out of GET /tools.
    """
    if not os.path.isdir(root):
        return []
    return [
        entry
        for entry in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, entry))
        and not entry.startswith("_")
        and not entry.startswith(".")
    ]


def _discover_schema_tools(root: str) -> list:
    """(folder name, SchemaTool) for every folder under `root` declaring one.

    Nothing here imports the tool: the schema is read, checked against the
    source it claims to describe, and turned into a Tool object. A mismatch is
    re-raised -- it is the one failure that takes the server with it.
    """
    discovered = []
    for entry in _tool_folders(root):
        folder = os.path.join(root, entry)
        if not is_packaged(folder):
            continue
        try:
            discovered.append((entry, load_tool(folder, deployment_config)))
        except Exception as exc:
            _record_failure(entry, exc)
    return discovered


def _discover_tool_classes(root: str) -> list:
    """(folder name, Tool subclass) pairs found under tools/.

    The folder name is carried along so a failure further down -- during
    instantiation or schema validation -- can still name the tool it came from.
    Empty when the image ships no in-process tools at all.
    """
    discovered = []
    if tools_package is None:
        return discovered

    for entry in _tool_folders(root):
        folder_path = os.path.join(root, entry)
        # A tool that declares a schema is served from it and never imported.
        if is_packaged(folder_path):
            continue

        try:
            expected_file = os.path.join(folder_path, f"{entry}.py")
            if not os.path.isfile(expected_file):
                raise RuntimeError(
                    f"Tool folder 'tools/{entry}/' is missing its 'tools/{entry}/{entry}.py' file."
                )
            module = importlib.import_module(f"{tools_package.__name__}.{entry}.{entry}")
        except Exception as exc:
            # Anything at all: a missing dependency, a syntax error, a
            # module-level statement that raises. None of it is worth a dead
            # server.
            _record_failure(entry, exc)
            continue

        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, Tool) and attr is not Tool:
                discovered.append((entry, attr))

    return discovered


def _instantiate(folder: str, cls, registry: dict):
    instance = cls()
    if not instance.name:
        raise RuntimeError(f"Tool class '{cls.__name__}' has no 'name' set.")
    _reject_duplicate(instance.name, registry)
    # Catch a malformed argument declaration here, at import time, rather
    # than on the first request that happens to reach that tool.
    instance.check_schema()
    return instance


def _reject_duplicate(name: str, registry: dict) -> None:
    if name in registry:
        raise RuntimeError(f"Duplicate tool name detected: '{name}'")


def _check_deployment_config(registry: dict) -> None:
    """Every [tools.X] in deployment.toml must name a tool this server carries.

    A typo'd or leftover section is otherwise dead config that looks live: the
    dropdown it was meant to add simply never appears, and nothing says why.
    """
    unknown = [name for name in deployment_config.configured_tools if name not in registry]
    if unknown:
        logger.warning(
            "deployment.toml configures %s, which this server does not serve. "
            "Check the tool name against GET /tools.",
            unknown,
        )
    for name in deployment_config.configured_tools:
        tool = registry.get(name)
        if tool is None or not deployment_config.for_tool(name).server_selectable:
            continue
        if not isinstance(tool, SchemaTool):
            # An imported tool declares server_selectable in its own ArgSpec,
            # where it travels with the code. Honouring it from here as well
            # would make GET /tools depend on a file the tool knows nothing
            # about.
            logger.warning(
                "deployment.toml sets server_selectable for '%s', which is an imported tool: "
                "it declares that in its own ArgSpec and this entry is ignored.",
                name,
            )


def _build_registry() -> dict:
    registry: dict = {}

    for folder, tool in _discover_schema_tools(settings.TOOLS_DIR):
        try:
            _reject_duplicate(tool.name, registry)
        except Exception as exc:
            _record_failure(folder, exc)
            continue
        registry[tool.name] = tool

    schema_tools = len(registry)

    packaged = {name.casefold(): name for name in registry}

    legacy_root = tools_package.__path__[0] if tools_package is not None else ""
    for folder, cls in _discover_tool_classes(legacy_root):
        try:
            instance = _instantiate(folder, cls, registry)
        except Exception as exc:
            _record_failure(folder, exc)
            continue
        # The packaged tool wins over the one this server still imports, and
        # the comparison is case-insensitive because that is the only thing
        # separating them: a tool is named after its folder, and packaging it
        # lowercased the name -- `AMASSS` became `amasss`.
        #
        # Without this both are served, they differ by nothing a person reads,
        # and the imported one fails on any deployment that does not carry the
        # whole inference stack in the SERVER's environment -- which is the
        # situation packaging exists to end. The failure is a 500 several
        # frames deep in the tool ("missing: nnunetv2"), and nothing tells the
        # caller they picked the superseded one.
        superseded = packaged.get(instance.name.casefold())
        if superseded is not None:
            logger.info(
                "Not serving imported tool '%s': superseded by the packaged '%s'. "
                "Delete server/tools/%s once that one is merged.",
                instance.name, superseded, instance.name,
            )
            continue
        registry[instance.name] = instance

    _check_deployment_config(registry)

    logger.info(
        "Tool registry: %d loaded, %d of them from a schema (%s)",
        len(registry),
        schema_tools,
        ", ".join(sorted(registry)) or "none",
    )
    if FAILED_TOOLS:
        # Repeated here, at the very end of startup, so it survives the wall of
        # uvicorn/framework output that follows and is still on screen.
        logger.error(
            "\n%s\n"
            "  %d TOOL(S) FAILED TO LOAD: %s\n"
            "  Scroll up for the full traceback of each one.\n"
            "%s",
            _BANNER,
            len(FAILED_TOOLS),
            ", ".join(sorted(FAILED_TOOLS)),
            _BANNER,
        )
    return registry


TOOLS: dict = _build_registry()


def get_tool(name: str):
    try:
        return TOOLS[name]
    except KeyError:
        if name in FAILED_TOOLS:
            # The tool exists in the source tree but didn't load. Say so rather
            # than "unknown tool", which reads like a typo. The reason stays in
            # the server logs, where it may name internal paths.
            raise KeyError(
                f"Tool '{name}' failed to load at server startup and is unavailable. "
                f"See the server logs."
            )
        raise KeyError(f"Unknown tool: '{name}'")

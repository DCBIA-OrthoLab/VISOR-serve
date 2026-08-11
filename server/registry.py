"""Auto-discovery of Tool subclasses under the tools/ package.

Each tool is a folder under tools/ (e.g. tools/example_tool/) containing
exactly one recognized file: tools/example_tool/example_tool.py -- the file
name must match the folder name. That file defines one or more Tool
subclasses. Any other file in the folder (helpers, data, ...) is ignored by
discovery, though the recognized file is free to import from them.

Adding a tool = dropping a new folder in tools/ with its <name>/<name>.py
file. No central list to edit, no route to add.

A tool that fails to load -- missing dependency, syntax error, bad schema --
is SKIPPED, not fatal: it would otherwise take the whole server down with it,
and with 15+ tools that means one unavailable model blocks all the others. The
failure is reported as loudly as possible instead (see _record_failure), kept
in FAILED_TOOLS, and surfaced again by get_tool().
"""

import importlib
import logging
import os
import traceback

import tools as tools_package
from base import Tool

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


def _discover_tool_classes() -> list:
    """(folder name, Tool subclass) pairs found under tools/.

    The folder name is carried along so a failure further down -- during
    instantiation or schema validation -- can still name the tool it came from.
    """
    discovered = []
    package_dir = tools_package.__path__[0]

    for entry in sorted(os.listdir(package_dir)):
        folder_path = os.path.join(package_dir, entry)
        if not os.path.isdir(folder_path) or entry.startswith("_") or entry.startswith("."):
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


def _build_registry() -> dict:
    registry: dict = {}

    for folder, cls in _discover_tool_classes():
        try:
            instance = cls()
            if not instance.name:
                raise RuntimeError(f"Tool class '{cls.__name__}' has no 'name' set.")
            if instance.name in registry:
                raise RuntimeError(f"Duplicate tool name detected: '{instance.name}'")
            # Catch a malformed argument declaration here, at import time, rather
            # than on the first request that happens to reach that tool.
            instance.check_schema()
        except Exception as exc:
            _record_failure(folder, exc)
            continue
        registry[instance.name] = instance

    logger.info(
        "Tool registry: %d loaded (%s)", len(registry), ", ".join(sorted(registry)) or "none"
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

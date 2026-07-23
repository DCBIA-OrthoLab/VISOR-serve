"""Auto-discovery of Tool subclasses under the tools/ package.

Each tool is a folder under tools/ (e.g. tools/example_tool/) containing
exactly one recognized file: tools/example_tool/example_tool.py -- the file
name must match the folder name. That file defines one or more Tool
subclasses. Any other file in the folder (helpers, data, ...) is ignored by
discovery, though the recognized file is free to import from them.

Adding a tool = dropping a new folder in tools/ with its <name>/<name>.py
file. No central list to edit, no route to add.
"""

import importlib
import os

import tools as tools_package
from base import Tool


def _discover_tool_classes() -> list:
    classes = []
    package_dir = tools_package.__path__[0]

    for entry in sorted(os.listdir(package_dir)):
        folder_path = os.path.join(package_dir, entry)
        if not os.path.isdir(folder_path) or entry.startswith("_") or entry.startswith("."):
            continue

        expected_file = os.path.join(folder_path, f"{entry}.py")
        if not os.path.isfile(expected_file):
            raise RuntimeError(
                f"Tool folder 'tools/{entry}/' is missing its 'tools/{entry}/{entry}.py' file."
            )

        module_name = f"{tools_package.__name__}.{entry}.{entry}"
        module = importlib.import_module(module_name)
        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, Tool) and attr is not Tool:
                classes.append(attr)

    return classes


def _build_registry() -> dict:
    registry: dict = {}
    for cls in _discover_tool_classes():
        instance = cls()
        if not instance.name:
            raise RuntimeError(f"Tool class '{cls.__name__}' has no 'name' set.")
        if instance.name in registry:
            raise RuntimeError(f"Duplicate tool name detected: '{instance.name}'")
        registry[instance.name] = instance
    return registry


TOOLS: dict = _build_registry()


def get_tool(name: str):
    try:
        return TOOLS[name]
    except KeyError:
        raise KeyError(f"Unknown tool: '{name}'")

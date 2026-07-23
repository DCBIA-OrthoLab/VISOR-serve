"""Auto-discovery of Tool subclasses under the tools/ package.

Each tool is a folder under tools/ (e.g. tools/example_tool/), with an
__init__.py (can be empty) and one or more .py files defining a Tool
subclass. Discovery recurses into every subpackage and is agnostic to the
internal file name -- only the Tool subclass itself matters.

Adding a tool = dropping a new folder in tools/. No central list to edit,
no route to add.
"""

import importlib
import pkgutil

import tools as tools_package
from base import Tool


def _discover_tool_classes() -> list:
    seen = set()
    for _, module_name, _ in pkgutil.walk_packages(tools_package.__path__, tools_package.__name__ + "."):
        module = importlib.import_module(module_name)
        for attr in vars(module).values():
            if isinstance(attr, type) and issubclass(attr, Tool) and attr is not Tool:
                seen.add(attr)
    return list(seen)


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

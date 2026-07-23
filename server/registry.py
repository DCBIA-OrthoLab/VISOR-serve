"""Auto-discovery of Tool subclasses under the tools/ package.

Adding a tool = dropping a new file in tools/ that defines a Tool subclass.
No central list to edit, no route to add.
"""

import importlib
import pkgutil

import tools as tools_package
from base import Tool


def _discover_tool_classes() -> list:
    classes = []
    for _, module_name, _ in pkgutil.iter_modules(tools_package.__path__, tools_package.__name__ + "."):
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

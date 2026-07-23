"""Tool base class: every tool declares a typed argument schema and a run()
method. Arguments are validated against the schema before run() is called,
so run() can always trust its inputs.

# TODO to add a new tool: create a folder tools/<name>/ with an __init__.py
# (can be empty) and a tools/<name>/<name>.py file (must match the folder
# name) subclassing Tool; set `name` and `arguments`, implement `run`.
# Nothing else needs to change -- see tools/test_tool/ for a minimal example
# and registry.py for how it gets picked up automatically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union


@dataclass
class ArgSpec:
    type: Union[type, str]  # str, int, float, bool, or "file"
    required: bool = True
    description: str = ""


class ToolArgumentError(Exception):
    """Raised when arguments passed to a tool don't match its declared schema."""


class Tool(ABC):
    name: str = ""
    arguments: dict = {}
    output_kind: str = "text"  # "text" | "file" | "segmentation" | ...

    def validate(self, args: dict) -> dict:
        """Check args against self.arguments; return cleaned/coerced args.

        Raises ToolArgumentError on missing required args, unknown args, or
        a type mismatch that can't be sensibly coerced.
        """
        unknown = set(args) - set(self.arguments)
        if unknown:
            raise ToolArgumentError(
                f"Unexpected argument(s) for tool '{self.name}': {', '.join(sorted(unknown))}"
            )

        cleaned: dict[str, Any] = {}
        for arg_name, spec in self.arguments.items():
            if arg_name not in args or args[arg_name] is None:
                if spec.required:
                    raise ToolArgumentError(
                        f"Missing required argument '{arg_name}' for tool '{self.name}'"
                    )
                continue
            cleaned[arg_name] = self._coerce(arg_name, args[arg_name], spec)

        return cleaned

    def _coerce(self, arg_name: str, value: Any, spec: ArgSpec) -> Any:
        if spec.type == "file":
            # The file has already been streamed to disk by main.py; value is its path.
            return value
        if spec.type is str:
            if isinstance(value, str):
                return value
            raise ToolArgumentError(f"Argument '{arg_name}' must be a string")
        if spec.type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
                return value.lower() in ("true", "1")
            raise ToolArgumentError(f"Argument '{arg_name}' must be a boolean")
        if spec.type in (int, float):
            try:
                return spec.type(value)
            except (TypeError, ValueError):
                raise ToolArgumentError(
                    f"Argument '{arg_name}' must be a {spec.type.__name__}"
                )
        return value

    def invoke(self, args: dict):
        """Validate args, then run the tool. This is what the server calls."""
        cleaned = self.validate(args)
        return self.run(**cleaned)

    @abstractmethod
    def run(self, **kwargs):
        """Do the actual work. Can trust that kwargs already match the schema."""
        ...

"""Tool base class: every tool declares a typed argument schema and a run()
method. Arguments are validated against the schema before run() is called,
so run() can always trust its inputs.

# TODO to add a new tool: create a folder tools/<name>/ with an __init__.py
# (can be empty) and a tools/<name>/<name>.py file (must match the folder
# name) subclassing Tool; set `name` and `arguments`, implement `run`.
# Nothing else needs to change -- see tools/test_tool/ for a minimal example
# and registry.py for how it gets picked up automatically.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

# File-typed arguments declare a specific kind here instead of a generic
# "file", so both the server (extension check) and the client (GET /tools)
# know exactly what's expected for that argument -- no shared global
# whitelist to keep in sync across unrelated tools.
# "file" is kept as a generic passthrough: None means "fall back to the
# server-wide config.ALLOWED_EXTENSIONS whitelist" instead of a fixed list.
FILE_TYPES: dict = {
    "file": None,
    "zip_file": (".zip",),
    "csv_file": (".csv",),
    "xlsx_file": (".xlsx",),
    "ods_file": (".ods",),
    "nifti_file": (".nii", ".nii.gz"),
    # A medical volume OR a zip archive of a folder of them: lets a single
    # argument serve both "one scan" and "a batch" without the schema having
    # to express "exactly one of these two arguments" (which it can't).
    # The tool dispatches on what it actually received (file / zip / folder).
    "volume_or_zip_file": (
        ".nii",
        ".nii.gz",
        ".nrrd",
        ".nrrd.gz",
        ".gipl",
        ".gipl.gz",
        ".zip",
    ),
}

# Argument type for "pick one or several values from a server-defined list".
# The server owns both the valid values (`choices`) and how they should be
# presented (`choice_groups`: group label -> {display name: value}), so a
# client can render grouped checkboxes without hardcoding anything --
# see ArgSpec below and Tool._coerce_selection for the accepted wire formats.
SELECTION_TYPE = "selection"

_TRUE_TOKENS = ("true", "1", "yes", "on", "checked")
_FALSE_TOKENS = ("false", "0", "no", "off", "unchecked", "")


@dataclass
class ArgSpec:
    type: Union[type, str]  # str, int, float, bool, SELECTION_TYPE, or a FILE_TYPES key
    required: bool = True
    description: str = ""
    # Lets the caller pick a file already present on the server (see
    # data_store.py) by sending its name as a plain form value.
    # "model" -> DATA_DIR/<tool_name>/models/, "testfile" ->
    # DATA_DIR/<tool_name>/testfiles/. None (default) means upload-only.
    # On a file-typed argument the caller may still upload its own file
    # instead; on a scalar (str) argument the server-side file is the ONLY
    # option -- uploads for non-file arguments are rejected (see main.py).
    # Either way run() receives a local path to the resolved file.
    server_selectable: Optional[str] = None

    # --- SELECTION_TYPE arguments only (ignored otherwise) ------------------
    # The canonical values this argument accepts, e.g. ("MAND", "MAX", "CB").
    # Anything else is rejected with a ToolArgumentError naming what's valid.
    choices: Optional[tuple] = None
    # Presentation metadata owned by the SERVER, so the client never hardcodes
    # a structure list: {group label: {human-readable name: canonical value}}.
    # e.g. {"Bones": {"Mandible": "MAND", "Maxilla": "MAX"}, "Masks": {...}}.
    # A client renders one box per group with one checkbox per entry, and may
    # send back {"Mandible": true, "Maxilla": false} -- display names are
    # accepted as aliases of their value on the way in.
    choice_groups: Optional[dict] = None
    # True -> run() receives a list of values; False -> exactly one value.
    multiple: bool = False

    # Advisory default, exposed through GET /tools so a client can pre-fill
    # the widget. The server does NOT apply it: an omitted optional argument
    # simply falls through to run()'s own Python default, which stays the
    # single source of truth for what actually happens.
    default: Any = None


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
        if spec.type in FILE_TYPES:
            # The file has already been streamed to disk by main.py; value is its path.
            return value
        if spec.type == SELECTION_TYPE:
            return self._coerce_selection(arg_name, value, spec)
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

    # ------------------------------------------------------------------
    # SELECTION_TYPE
    # ------------------------------------------------------------------
    # Multipart form fields are always strings, so the same logical value can
    # legitimately arrive in several shapes. All of these are accepted and
    # normalized to the same canonical list:
    #   '{"Mandible": true, "Maxilla": false}'  (checkbox dict, display names)
    #   '{"MAND": true, "MAX": false}'          (checkbox dict, values)
    #   '["MAND", "MAX"]'                       (JSON list)
    #   'MAND,MAX'  /  'MAND MAX'               (plain separated list)
    #   ["MAND", "MAX"]                         (already a Python list)
    # The dict form is the one a grouped-checkbox UI produces naturally, and
    # is why the server publishes `choice_groups`: the client never has to
    # know the canonical values, only echo back what the server named.

    def _coerce_selection(self, arg_name: str, value: Any, spec: ArgSpec) -> Any:
        selected = self._selection_values(arg_name, value, spec)
        aliases = self._selection_aliases(spec)

        resolved: list = []
        unknown: list = []
        for raw in selected:
            key = str(raw).strip().lower()
            if key in aliases:
                if aliases[key] not in resolved:
                    resolved.append(aliases[key])
            else:
                unknown.append(str(raw))

        if unknown:
            raise ToolArgumentError(
                f"Invalid value(s) for argument '{arg_name}' of tool '{self.name}': "
                f"{', '.join(sorted(unknown))}. Allowed: {', '.join(spec.choices or ())}"
            )

        # Return values in the order the tool declared them, not the order the
        # caller happened to send them, so downstream behavior is deterministic.
        if spec.choices:
            resolved.sort(key=lambda item: list(spec.choices).index(item))

        if spec.multiple:
            if spec.required and not resolved:
                raise ToolArgumentError(
                    f"Argument '{arg_name}' of tool '{self.name}' requires at least one "
                    f"selected value. Allowed: {', '.join(spec.choices or ())}"
                )
            return resolved

        if len(resolved) != 1:
            raise ToolArgumentError(
                f"Argument '{arg_name}' of tool '{self.name}' expects exactly one value, "
                f"got {len(resolved)}. Allowed: {', '.join(spec.choices or ())}"
            )
        return resolved[0]

    def _selection_values(self, arg_name: str, value: Any, spec: ArgSpec) -> list:
        """Normalize any accepted wire shape into a flat list of raw labels."""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") or text.startswith("["):
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ToolArgumentError(
                        f"Argument '{arg_name}' of tool '{self.name}' looks like JSON but "
                        f"could not be parsed: {exc}"
                    )
            else:
                return [part for part in re.split(r"[,;\s]+", text) if part]

        if isinstance(value, dict):
            # {name: true/false} -- keep only what is checked.
            return [key for key, flag in value.items() if self._is_truthy(arg_name, flag)]

        if isinstance(value, (list, tuple, set)):
            return [item for item in value]

        raise ToolArgumentError(
            f"Argument '{arg_name}' of tool '{self.name}' must be a list of values or a "
            f"{{name: true/false}} mapping. Allowed: {', '.join(spec.choices or ())}"
        )

    def _is_truthy(self, arg_name: str, flag: Any) -> bool:
        if isinstance(flag, bool):
            return flag
        if isinstance(flag, (int, float)):
            return bool(flag)
        if isinstance(flag, str):
            token = flag.strip().lower()
            if token in _TRUE_TOKENS:
                return True
            if token in _FALSE_TOKENS:
                return False
        raise ToolArgumentError(
            f"Argument '{arg_name}' of tool '{self.name}': '{flag}' is not a valid "
            f"true/false value."
        )

    @staticmethod
    def _selection_aliases(spec: ArgSpec) -> dict:
        """Lowercased {accepted label -> canonical value}.

        Both the canonical values and the human-readable names published in
        `choice_groups` are accepted, so a client can send back exactly the
        labels the server gave it without any translation table of its own.
        """
        aliases: dict = {}
        for value in spec.choices or ():
            aliases[str(value).strip().lower()] = value
        for group in (spec.choice_groups or {}).values():
            for label, value in group.items():
                aliases[str(label).strip().lower()] = value
        return aliases

    def invoke(self, args: dict):
        """Validate args, then run the tool. This is what the server calls."""
        cleaned = self.validate(args)
        return self.run(**cleaned)

    @abstractmethod
    def run(self, **kwargs):
        """Do the actual work. Can trust that kwargs already match the schema."""
        ...

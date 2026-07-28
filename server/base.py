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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

# File-typed arguments declare a specific kind here instead of a generic
# "file", so both the server (extension check) and the client (GET /tools)
# know exactly what's expected for that argument -- no shared global
# whitelist to keep in sync across unrelated tools.
# "file" is kept as a generic passthrough: None means "fall back to the
# server-wide config.ALLOWED_EXTENSIONS whitelist" instead of a fixed list.
# "folder" is the one type whose value reaches run() as a DIRECTORY: HTTP has
# no notion of a folder, so the client sends it as a .zip and main.py extracts
# it before calling run() (see FOLDER_TYPE below).
FILE_TYPES: dict = {
    "file": None,
    "folder": (".zip",),
    "zip_file": (".zip",),
    "csv_file": (".csv",),
    "xlsx_file": (".xlsx",),
    "ods_file": (".ods",),
    "nifti_file": (".nii", ".nii.gz"),
}

# The type whose resolved path is a directory rather than a file.
FOLDER_TYPE = "folder"

SCALAR_TYPES = (str, int, float, bool)

# Types picking from a fixed set of named options the tool declares in
# ArgSpec.choices. They exist so the client can render the right widget without
# any tool-specific code, and so an out-of-range value is caught by validate()
# instead of reaching run():
#   "choice"      -> exactly one option        -> combo box  -> run() gets a str
#   "multichoice" -> any number of options     -> check boxes -> run() gets a Selection
CHOICE_TYPE = "choice"
MULTICHOICE_TYPE = "multichoice"
CHOICE_TYPES = (CHOICE_TYPE, MULTICHOICE_TYPE)


class Selection(dict):
    """What run() receives for a "multichoice" argument: every option the tool
    declared, mapped to True/False, in declaration order.

    Being a plain dict means `selection["mandible"]` works and no option is
    ever missing, so a tool never needs `.get(name, False)`. `.selected` is
    there for the common case of looping over what's enabled.
    """

    @property
    def selected(self) -> tuple:
        """The enabled option names, in declaration order."""
        return tuple(name for name, enabled in self.items() if enabled)


class ResolvedPath(str):
    """Local path handed to Tool.run() for a file/folder argument, tagged with
    the declared type it was actually resolved as.

    This is what makes an argument accepting SEVERAL types usable: a tool
    declaring `type=("csv_file", "folder")` branches on `input.kind` instead
    of guessing from the extension or probing the filesystem.

    It subclasses `str`, so it stays a plain path everywhere else -- `open()`,
    `os.path.*`, `pandas.read_csv()` all work unchanged, and a tool that
    declares a single type can keep ignoring `.kind` entirely.
    """

    kind: str

    def __new__(cls, path: str, kind: str) -> "ResolvedPath":
        resolved = super().__new__(cls, path)
        resolved.kind = kind
        return resolved

    @property
    def is_folder(self) -> bool:
        return self.kind == FOLDER_TYPE


@dataclass
class ArgSpec:
    # A scalar type (str, int, float, bool), one of FILE_TYPES's keys, or a
    # TUPLE of FILE_TYPES keys when the argument accepts several -- typically
    # ("csv_file", "folder") for "one file or a whole folder of them". run()
    # then reads `<arg>.kind` to know which one it got (see ResolvedPath).
    # Mixing a scalar with a file type is rejected at startup (check_schema).
    type: Union[type, str, tuple]
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
    # Required by "choice"/"multichoice", forbidden elsewhere: the available
    # options, each mapped to whether it is on by default --
    # {"mandible": True, "maxilla": True, "skull": False}. One declaration
    # gives the client its widget AND supplies the default when the caller
    # doesn't send the argument, so defaults never live in two places.
    # "choice" declares exactly one True: the initially selected option.
    choices: Optional[dict] = None

    @property
    def types(self) -> tuple:
        """The declared type(s), always as a tuple."""
        return self.type if isinstance(self.type, tuple) else (self.type,)

    @property
    def is_file(self) -> bool:
        """True when this argument's value reaches run() as a path."""
        return all(declared in FILE_TYPES for declared in self.types)

    @property
    def is_choice(self) -> bool:
        """True when this argument picks from the options in self.choices."""
        return self.types[0] in CHOICE_TYPES

    @property
    def default(self) -> Any:
        """Value handed to run() when an optional choice argument is absent."""
        if self.types[0] == MULTICHOICE_TYPE:
            return Selection(self.choices)
        return next(name for name, on in self.choices.items() if on)

    @property
    def extensions(self) -> Optional[tuple]:
        """Every extension accepted across the declared file types, in
        declaration order. None means "no specific type declared" -- fall back
        to the server-wide config.ALLOWED_EXTENSIONS whitelist.
        """
        accepted: list = []
        for declared in self.types:
            allowed = FILE_TYPES[declared]
            if allowed is None:  # generic "file" widens the argument to the whitelist
                return None
            accepted.extend(extension for extension in allowed if extension not in accepted)
        return tuple(accepted)

    def match_type(self, extension: str) -> str:
        """Which of the declared types an upload with this extension resolves to.

        Declaration order breaks ties: with type=("zip_file", "folder") a .zip
        is handed over as an archive, with ("folder", "zip_file") it is
        extracted first. Only meaningful on a file-typed argument.
        """
        extension = extension.lower()
        for declared in self.types:
            allowed = FILE_TYPES[declared]
            if allowed is None or extension in allowed:
                return declared
        return self.types[0]


class ToolArgumentError(Exception):
    """Raised when arguments passed to a tool don't match its declared schema."""


class ToolSchemaError(Exception):
    """Raised at startup when a tool's own `arguments` declaration is invalid."""


class Tool(ABC):
    name: str = ""
    arguments: dict = {}
    # "text"              -> run() returns any JSON-serializable value
    # "file"/"segmentation" -> run() returns the path of ONE output file
    # "files"             -> run() returns a list of paths, or one directory
    #                        path; main.py zips them and streams the archive
    output_kind: str = "text"

    def check_schema(self) -> None:
        """Reject an invalid `arguments` declaration. Called by registry.py at
        startup, so a malformed tool fails loudly on boot instead of on the
        first request that happens to hit it.
        """
        for arg_name, spec in self.arguments.items():
            where = f"Tool '{self.name}', argument '{arg_name}'"
            if not spec.types:
                raise ToolSchemaError(f"{where}: no type declared.")
            for declared in spec.types:
                if (
                    declared not in FILE_TYPES
                    and declared not in SCALAR_TYPES
                    and declared not in CHOICE_TYPES
                ):
                    raise ToolSchemaError(
                        f"{where}: unknown type {declared!r}. Use a scalar type, one of "
                        f"{sorted(FILE_TYPES)}, or one of {sorted(CHOICE_TYPES)}."
                    )
            if len(spec.types) > 1 and not spec.is_file:
                raise ToolSchemaError(
                    f"{where}: only file types can be combined, a scalar or choice type "
                    f"must stand alone (got {spec.types})."
                )
            self._check_choices(where, spec)

    @staticmethod
    def _check_choices(where: str, spec: ArgSpec) -> None:
        if not spec.is_choice:
            if spec.choices is not None:
                raise ToolSchemaError(
                    f"{where}: 'choices' only applies to {sorted(CHOICE_TYPES)} arguments."
                )
            return

        if not isinstance(spec.choices, dict) or not spec.choices:
            raise ToolSchemaError(
                f"{where}: a {spec.types[0]!r} argument must declare a non-empty "
                f"choices dict, e.g. choices={{'mandible': True, 'skull': False}}."
            )
        for name, enabled in spec.choices.items():
            if not isinstance(name, str) or not name:
                raise ToolSchemaError(f"{where}: choice names must be non-empty strings.")
            if not isinstance(enabled, bool):
                raise ToolSchemaError(
                    f"{where}: choice '{name}' must map to True or False, got {enabled!r}."
                )
        # A combo box has exactly one selected entry, and that entry is also
        # what run() gets when the caller omits an optional argument.
        if spec.types[0] == CHOICE_TYPE and sum(spec.choices.values()) != 1:
            raise ToolSchemaError(
                f"{where}: a 'choice' argument declares exactly one option as True "
                f"(the default); use 'multichoice' for several."
            )

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
                # A choice argument already declared its default in `choices`,
                # so hand that over rather than making the tool repeat it in
                # run()'s signature, where the two would eventually drift.
                if spec.is_choice:
                    cleaned[arg_name] = spec.default
                continue
            cleaned[arg_name] = self._coerce(arg_name, args[arg_name], spec)

        return cleaned

    def _coerce(self, arg_name: str, value: Any, spec: ArgSpec) -> Any:
        if spec.is_file:
            # main.py already streamed the upload (or resolved the server-side
            # file) to disk and tagged the path with the type it matched. The
            # fallback only covers a tool invoked directly in a unit test.
            if isinstance(value, ResolvedPath):
                return value
            return ResolvedPath(value, spec.types[0])

        declared = spec.types[0]
        if declared == CHOICE_TYPE:
            return self._coerce_choice(arg_name, value, spec)
        if declared == MULTICHOICE_TYPE:
            return self._coerce_multichoice(arg_name, value, spec)
        if declared is str:
            # A ResolvedPath is a str, so a server-selectable scalar keeps its
            # .kind here instead of being flattened back to a plain string.
            if isinstance(value, str):
                return value
            raise ToolArgumentError(f"Argument '{arg_name}' must be a string")
        if declared is bool:
            return self._coerce_bool(arg_name, value)
        if declared in (int, float):
            try:
                return declared(value)
            except (TypeError, ValueError):
                raise ToolArgumentError(
                    f"Argument '{arg_name}' must be a {declared.__name__}"
                )
        return value

    @staticmethod
    def _coerce_bool(arg_name: str, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
            return value.lower() in ("true", "1")
        raise ToolArgumentError(f"Argument '{arg_name}' must be a boolean")

    @staticmethod
    def _coerce_choice(arg_name: str, value: Any, spec: ArgSpec) -> str:
        """One option name, sent as a plain form value."""
        if not isinstance(value, str):
            raise ToolArgumentError(f"Argument '{arg_name}' must be one option name")
        if value not in spec.choices:
            raise ToolArgumentError(
                f"Argument '{arg_name}': unknown option '{value}'. "
                f"Expected one of: {', '.join(spec.choices)}"
            )
        return value

    def _coerce_multichoice(self, arg_name: str, value: Any, spec: ArgSpec) -> Selection:
        """A subset of the declared options. Accepted on the wire as either a
        JSON object -- {"mandible": true, "skull": false} -- or the shorthand
        comma-separated list of the enabled ones -- "mandible,maxilla".

        Whatever arrives is the COMPLETE selection: an option that isn't
        mentioned is off, regardless of the default declared in `choices`.
        Omitting the argument entirely is what falls back to those defaults.
        """
        if isinstance(value, dict):
            provided = value
        elif isinstance(value, (list, tuple, set)):
            provided = {str(name): True for name in value}
        elif isinstance(value, str):
            text = value.strip()
            if text.startswith("{"):
                try:
                    provided = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ToolArgumentError(f"Argument '{arg_name}': invalid JSON ({exc})")
                if not isinstance(provided, dict):
                    raise ToolArgumentError(
                        f"Argument '{arg_name}': expected a JSON object of option -> true/false"
                    )
            else:
                provided = {name.strip(): True for name in text.split(",") if name.strip()}
        else:
            raise ToolArgumentError(
                f"Argument '{arg_name}': expected an option -> true/false mapping"
            )

        unknown = [name for name in provided if name not in spec.choices]
        if unknown:
            raise ToolArgumentError(
                f"Argument '{arg_name}': unknown option(s) {', '.join(sorted(unknown))}. "
                f"Expected any of: {', '.join(spec.choices)}"
            )
        # Rebuilt from the declaration, so run() always sees every option in
        # declaration order and never has to guard a missing key.
        return Selection(
            (name, self._coerce_bool(arg_name, provided[name]) if name in provided else False)
            for name in spec.choices
        )

    def invoke(self, args: dict):
        """Validate args, then run the tool. This is what the server calls."""
        cleaned = self.validate(args)
        return self.run(**cleaned)

    @abstractmethod
    def run(self, **kwargs):
        """Do the actual work. Can trust that kwargs already match the schema."""
        ...

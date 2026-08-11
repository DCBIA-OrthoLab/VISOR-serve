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

# A file-typed argument declares a specific kind here rather than a generic
# "file", so both the extension check and the client know what is expected.
# "file" is the generic passthrough: None means "fall back to
# config.ALLOWED_EXTENSIONS". "folder" is the one type whose value reaches
# run() as a DIRECTORY -- HTTP has no notion of a folder, so the client sends
# a .zip and main.py extracts it first.
FILE_TYPES: dict = {
    "file": None,
    "folder": (".zip",),
    "zip_file": (".zip",),
    "csv_file": (".csv",),
    "xlsx_file": (".xlsx",),
    "ods_file": (".ods",),
    "nifti_file": (".nii", ".nii.gz"),
    # One volume OR a zip of a folder of them, since the schema cannot express
    # "exactly one of these two arguments". The tool dispatches on what it got.
    "volume_or_zip_file": (
        ".nii",
        ".nii.gz",
        ".nrrd",
        ".nrrd.gz",
        ".gipl",
        ".gipl.gz",
        ".zip",
    ),
    # A 3D surface mesh. Every extension listed is one a tool declaring this
    # type must be able to READ: advertising a format and then only handling
    # .vtk is what made ALI accept .stl files it never processed.
    "surface_file": (".vtk", ".vtp", ".stl", ".obj", ".off"),
    # One mesh or a zipped folder of them. Deliberately shorter than
    # surface_file: these are the formats ALI's discovery actually walks.
    "surface_or_zip_file": (".vtk", ".stl", ".zip"),
}

# The type whose resolved path is a directory rather than a file.
FOLDER_TYPE = "folder"

SCALAR_TYPES = (str, int, float, bool)

# Types picking from the fixed set of options declared in ArgSpec.choices. The
# client renders the right widget with no tool-specific code, and an
# out-of-range value is caught by validate() instead of reaching run():
#   "choice"      -> exactly one option    -> combo box   -> run() gets a str
#   "multichoice" -> any number of options -> check boxes -> run() gets a Selection
CHOICE_TYPE = "choice"
MULTICHOICE_TYPE = "multichoice"
CHOICE_TYPES = (CHOICE_TYPE, MULTICHOICE_TYPE)

# How a client lays a "multichoice" argument's options out (ArgSpec.ui). None
# is the single-column stack.
#   "tabs"   -> one tab per `groups` entry, for a catalog too long to scroll.
#   "grid"   -> one ROW per `groups` entry, for options whose position carries
#               meaning (ASO's 32 teeth, upper arch above lower arch).
#   "inline" -> a single horizontal row, for a handful of short options.
UI_LAYOUTS = ("tabs", "grid", "inline")

# The layouts that are meaningless without ArgSpec.groups.
_GROUPED_LAYOUTS = ("tabs", "grid")


class Selection(dict):
    """What run() receives for a "multichoice" argument: every declared option
    mapped to True/False, in declaration order.

    Being a plain dict, `selection["mandible"]` always works and no option is
    ever missing, so a tool never needs `.get(name, False)`.
    """

    @property
    def selected(self) -> tuple:
        """The enabled option names, in declaration order."""
        return tuple(name for name, enabled in self.items() if enabled)


class ResolvedPath(str):
    """Local path handed to run() for a file/folder argument, tagged with the
    declared type it was resolved as.

    This is what makes a multi-type argument usable: a tool declaring
    `type=("csv_file", "folder")` branches on `input.kind` instead of guessing
    from the extension. It subclasses `str`, so it stays a plain path
    everywhere else and a single-type tool can ignore `.kind` entirely.
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
    # then reads `<arg>.kind` to know which it got (see ResolvedPath). Mixing a
    # scalar with a file type is rejected at startup by check_schema.
    type: Union[type, str, tuple]
    required: bool = True
    description: str = ""

    # Lets the caller pick a file already on the server (see data_store.py) by
    # sending its name as a plain form value. "model" ->
    # DATA_DIR/<tool>/models/, "testfile" -> DATA_DIR/<tool>/testfiles/. None
    # means upload-only. On a file-typed argument the caller may still upload
    # its own file; on a scalar argument the server-side file is the ONLY
    # option (main.py rejects uploads for non-file arguments). run() receives a
    # local path either way.
    server_selectable: Optional[str] = None

    # Required by "choice"/"multichoice", forbidden elsewhere: the available
    # options, each mapped to whether it is on by default. One declaration
    # gives the client its widget AND supplies the value used when the caller
    # omits the argument, so defaults never live in two places. "choice"
    # declares exactly one True.
    choices: Optional[dict] = None

    # SCALAR arguments only: the value a client pre-fills its widget with, so a
    # spin box starts at the tool's default rather than at Qt's 0. Advisory and
    # NOT applied server-side -- an omitted optional argument still falls
    # through to run()'s own Python default, which stays the source of truth.
    # Keep the two equal. Forbidden on choice types, whose initial state is
    # already in `choices`.
    initial: Any = None

    # ------------------------------------------------------------------
    # Presentation hints
    # ------------------------------------------------------------------
    # Published through GET /tools and read by the client's form generator;
    # validate() and run() ignore every one of them. They exist because a
    # schema that only says WHAT an argument is produces an unusable panel past
    # a certain size: ASO declares 130 landmarks, 32 teeth, 8 landmark types
    # and 2 jaws, which a generic client renders as one column of ~180 check
    # boxes. Nothing here names an anatomical concept: `groups` says what to
    # group, `ui` how to lay it out, `visible_when` when it applies.

    # The field label. None means "prettify the argument name", which cannot
    # produce "Scan / Landmark Folder" from `input` and renders an acronym as
    # "Cbct landmarks". The words a user reads belong with the tool.
    label: Optional[str] = None

    # The collapsible box this argument is rendered in. None is the client's
    # default section.
    section: Optional[str] = None

    # Show this argument only while other arguments hold given values:
    # {"modality": "CBCT", "automation": "Fully-Automated"} -- every entry must
    # match, and a tuple/list of values means "any of these". The named
    # arguments must be "choice" arguments of the same tool, enforced by
    # check_schema so a typo fails at boot instead of hiding a field forever.
    #
    # Presentation only: a hidden argument is simply not sent, so its declared
    # default applies. It does NOT replace the tool's own cross-argument
    # validation, which still has to hold for a direct API call.
    visible_when: Optional[dict] = None

    # How a "multichoice" argument's check boxes are laid out (see UI_LAYOUTS).
    ui: Optional[str] = None

    # {group name: (option, ...)} for the layouts that need it. Every option
    # must exist in `choices`; options left out of every group are rendered
    # after the groups rather than dropped.
    groups: Optional[dict] = None

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
        to config.ALLOWED_EXTENSIONS.
        """
        accepted: list = []
        for declared in self.types:
            allowed = FILE_TYPES[declared]
            if allowed is None:  # generic "file" widens the argument to the whitelist
                return None
            accepted.extend(extension for extension in allowed if extension not in accepted)
        return tuple(accepted)

    def match_type(self, extension: str) -> str:
        """Which declared type an upload with this extension resolves to.

        Declaration order breaks ties: with ("zip_file", "folder") a .zip is
        handed over as an archive, with ("folder", "zip_file") it is extracted.
        """
        extension = extension.lower()
        for declared in self.types:
            allowed = FILE_TYPES[declared]
            if allowed is None or extension in allowed:
                return declared
        return self.types[0]

    # NOTE: do not declare a `default` FIELD here. `default` is the @property
    # above, which derives the value from `choices`; a field of the same name
    # silently shadows it, and every optional choice argument then reaches
    # run() as None instead of its declared default.


class ToolArgumentError(Exception):
    """Raised when arguments passed to a tool don't match its declared schema."""


class ToolUnavailableError(Exception):
    """Raised when this SERVER cannot perform an otherwise valid request --
    typically a dependency the deployment image does not carry (see the
    lazy-import rule in ADDING_A_TOOL.md 7).

    Distinct from a generic 500: a crash inside a tool is rightly opaque, but
    "this server has no pytorch3d" is the one thing the caller needs to be
    told, names nothing sensitive, and no retry will help. main.py maps it to
    501 Not Implemented with the message.
    """


class ToolSchemaError(Exception):
    """Raised at startup when a tool's own `arguments` declaration is invalid."""


class Tool(ABC):
    name: str = ""
    arguments: dict = {}
    # "text"                -> run() returns any JSON-serializable value
    # "file"/"segmentation" -> run() returns the path of ONE output file
    # "files"               -> run() returns a list of paths, or one directory
    #                          path; main.py zips them and streams the archive
    output_kind: str = "text"

    def check_schema(self) -> None:
        """Reject an invalid `arguments` declaration. Called by registry.py at
        startup, so a malformed tool fails on boot instead of on the first
        request that happens to hit it.
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
            self._check_presentation(where, spec)

    def _check_presentation(self, where: str, spec: ArgSpec) -> None:
        """Reject a presentation hint that cannot be honored.

        Checked at startup precisely BECAUSE these are cosmetic: a wrong
        `visible_when` hides a field for good, and a client cannot tell that
        from a field the tool never declared.
        """
        if spec.label is not None and (not isinstance(spec.label, str) or not spec.label.strip()):
            raise ToolSchemaError(
                f"{where}: 'label' must be a non-empty string, or None to let the client "
                f"fall back to the argument name."
            )

        if spec.ui is not None:
            if spec.types[0] != MULTICHOICE_TYPE:
                raise ToolSchemaError(
                    f"{where}: 'ui' lays a multichoice argument's check boxes out, and this "
                    f"argument is a {spec.types[0]!r}."
                )
            if spec.ui not in UI_LAYOUTS:
                raise ToolSchemaError(
                    f"{where}: unknown ui layout {spec.ui!r}. Expected one of "
                    f"{sorted(UI_LAYOUTS)}."
                )

        if spec.groups is not None:
            if spec.ui not in _GROUPED_LAYOUTS:
                raise ToolSchemaError(
                    f"{where}: 'groups' only applies to the {sorted(_GROUPED_LAYOUTS)} "
                    f"layouts (ui={spec.ui!r})."
                )
            if not isinstance(spec.groups, dict) or not spec.groups:
                raise ToolSchemaError(f"{where}: 'groups' must be a non-empty dict.")
            for group_name, options in spec.groups.items():
                unknown = [option for option in options if option not in spec.choices]
                if unknown:
                    raise ToolSchemaError(
                        f"{where}: group '{group_name}' names {unknown}, which are not in "
                        f"this argument's choices."
                    )
        elif spec.ui in _GROUPED_LAYOUTS:
            raise ToolSchemaError(
                f"{where}: the {spec.ui!r} layout needs 'groups' to say what to group."
            )

        if spec.visible_when is None:
            return
        if not isinstance(spec.visible_when, dict) or not spec.visible_when:
            raise ToolSchemaError(f"{where}: 'visible_when' must be a non-empty dict.")
        for other_name, expected in spec.visible_when.items():
            other = self.arguments.get(other_name)
            if other is None:
                raise ToolSchemaError(
                    f"{where}: 'visible_when' refers to '{other_name}', which this tool "
                    f"does not declare."
                )
            if other.types[0] != CHOICE_TYPE:
                raise ToolSchemaError(
                    f"{where}: 'visible_when' can only test a 'choice' argument, and "
                    f"'{other_name}' is a {other.types[0]!r}. Only a choice argument has a "
                    f"fixed set of values to compare against."
                )
            wanted = expected if isinstance(expected, (tuple, list)) else (expected,)
            unknown = [value for value in wanted if value not in other.choices]
            if unknown:
                raise ToolSchemaError(
                    f"{where}: 'visible_when' expects '{other_name}' to be {unknown}, which "
                    f"is not among its choices ({sorted(other.choices)})."
                )

    @staticmethod
    def _check_choices(where: str, spec: ArgSpec) -> None:
        if not spec.is_choice:
            if spec.choices is not None:
                raise ToolSchemaError(
                    f"{where}: 'choices' only applies to {sorted(CHOICE_TYPES)} arguments."
                )
            if spec.initial is not None and spec.is_file:
                raise ToolSchemaError(
                    f"{where}: 'initial' only applies to scalar arguments, not file ones."
                )
            return

        # A choice argument's initial state is the True entry of `choices`; a
        # second declaration would put one default in two places.
        if spec.initial is not None:
            raise ToolSchemaError(
                f"{where}: 'initial' does not apply to {sorted(CHOICE_TYPES)} arguments -- "
                f"declare the initial state in 'choices' instead."
            )

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
        # A combo box has exactly one selected entry, which is also what run()
        # gets when the caller omits an optional argument.
        if spec.types[0] == CHOICE_TYPE and sum(spec.choices.values()) != 1:
            raise ToolSchemaError(
                f"{where}: a 'choice' argument declares exactly one option as True "
                f"(the default); use 'multichoice' for several."
            )

    def validate(self, args: dict) -> dict:
        """Check args against self.arguments; return cleaned/coerced args.

        Raises ToolArgumentError on missing required args, unknown args, or a
        type mismatch that can't be sensibly coerced.
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
                # so hand that over rather than making run()'s signature repeat
                # it, where the two would eventually drift.
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
        """A subset of the declared options, sent either as a JSON object --
        {"mandible": true, "skull": false} -- or as the comma-separated
        shorthand of the enabled ones -- "mandible,maxilla".

        Whatever arrives is the COMPLETE selection: an option not mentioned is
        off, whatever `choices` declares. Omitting the argument entirely is
        what falls back to those defaults.
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

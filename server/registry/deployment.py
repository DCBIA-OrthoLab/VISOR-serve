"""Per-tool server-side configuration: `deployment.toml`.

The split with `.schema.json` is deliberate and worth stating once. A tool's
schema says what its `run()` takes -- the same everywhere the tool is ever
installed, generated from its source, and hashed. This file says what THIS
deployment does with it: which arguments may be satisfied by a file already on
this server, and how large an upload this server accepts for this tool. Move
either of those into the schema and every deployment inherits one server's
paths and limits.

    [tools.amasss]
    server_selectable = { model = "model", scan = "testfile" }
    max_upload_mb = 500

Absent is the normal case: with no file at all, every `path` argument is
upload-only and MAX_UPLOAD_MB applies. Nothing here is required for a tool to
work.

The file is validated at startup rather than on first use: an entry naming an
argument that does not exist, or a `server_selectable` kind this server does not
know, would otherwise be a dropdown that silently never appears.

It is also MOUNTED into the container rather than baked into the image, so that
changing one line does not cost a 23 GB rebuild. The cost of that choice is that
the file can move ahead of the server reading it; `_unknown_value` is what makes
that legible when it happens.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

try:  # tomllib is standard from 3.11; the server targets the newest Python
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on 3.10 and older
    import tomli as tomllib

from config import settings

logger = logging.getLogger("inference_server")

# The two kinds data_store.py serves (DATA_DIR/<tool>/{models,testfiles}/).
# "none" is how a deployment says an argument is upload-only DESPITE its name.
# The conventions read any `path` called `reference` as a hosted bundle, which
# is right for ASO and AREG and wrong for FlexReg, whose `reference` is the
# patient's own other timepoint. Without a way to say no, the panel offers an
# empty dropdown and reports "no model available on the server" for a tool that
# needs none.
SERVER_SELECTABLE_NONE = "none"
SERVER_SELECTABLE_KINDS = ("model", "testfile", SERVER_SELECTABLE_NONE)

# Put in the message of every "this file asks for a value this server does not
# know" refusal, so `scripts/server_ctl.py` can recognise it in a container's log
# and explain it in one sentence instead of leaving an operator reading a
# traceback. Same device as its DEPENDENCY-INSTALL-FATAL marker, and here for a
# sharper reason: see `_unknown_value`.
CONFIG_AHEAD_MARKER = "DEPLOYMENT-CONFIG-AHEAD-OF-SERVER"


_TOOL_KEYS = ("server_selectable", "max_upload_mb", "data_dir", "hidden",
              "timeout_seconds", "dispatch", "batch", "width_from")

# What a [tools.X] `batch` table may say. `axis` names the argument a cohort is
# split on when the convention cannot derive it; the two caps override this
# server's own numbers for this tool alone.
_BATCH_KEYS = ("axis", "max_mb", "max_files")


class DeploymentConfigError(Exception):
    """Raised at startup when deployment.toml cannot be trusted."""


def _unknown_value(where: str, subject: str, known) -> "DeploymentConfigError":
    """The refusal for a value this server does not recognise, and the ADVICE.

    The advice is the point, and it was learned the hard way. `deployment.toml`
    is MOUNTED into the container from the installation, precisely so a config
    change does not cost a 23 GB rebuild -- which means the file can move ahead
    of the server reading it, and stay ahead silently: it is read once, at
    startup, so a container already running never notices. The mismatch surfaces
    at the next restart, or after a reboot, on a machine nobody is watching, as a
    crash loop.

    That happened on 2026-09-14: an image built on 2026-08-27 met a file using
    `server_selectable = "none"`, added on 2026-08-20 and first used today.

    The old message said only "expected one of ['model', 'testfile']", which
    reads as "your file is wrong" and invites the one repair that must not
    happen: deleting the value. Every value here was added to fix something --
    that `none` stops ALI being handed ASO's model folder and predicting with
    the wrong weights, silently, which was measured in a real run. Editing it out
    restores the defect and the server starts, so nothing says it came back.

    So the message names the likely cause and the correct repair: update the
    server.
    """
    return DeploymentConfigError(
        f"{where}: {subject}, which this server does not know "
        f"(it knows {list(known)}).\n"
        f"deployment.toml is mounted from the installation, so it can be NEWER "
        f"than the server reading it -- that is the usual reason for this. "
        f"Rebuild or update the server rather than editing the file: the value "
        f"was added to fix something, and removing it brings that back without "
        f"saying so. [{CONFIG_AHEAD_MARKER}]"
    )



@dataclass(frozen=True)
class ToolDeployment:
    """What this deployment says about one tool. All-defaults means "nothing
    said", which is the same as having no entry at all."""

    # {argument name: "model" | "testfile"}
    server_selectable: dict = field(default_factory=dict)
    # None falls back to settings.MAX_UPLOAD_MB.
    max_upload_mb: Optional[int] = None

    # The folder under DATA_DIR holding this tool's models and test files, when
    # it is not named after the tool. Packaged tools are lowercase (`amasss`)
    # while the data staged by scripts/setup-models.sh is not (`AMASSS/`), and
    # a case-insensitive lookup would be a guess -- on a case-sensitive
    # filesystem both can exist.
    data_dir: Optional[str] = None

    # {mode label shown to a clinician: the tool that mode runs}. Declaring one
    # publishes a FACADE: a tool of this name whose schema is composed from its
    # targets' and which forwards a run to whichever the caller picked.
    #
    # Here rather than in a package of its own, because the facade holds no copy
    # of anything. A packaged dispatcher would have to restate its targets'
    # arguments in its own run() signature, and the day one of them gains an
    # argument the dispatcher stops forwarding it -- silently, which is the
    # failure this repository keeps finding. Composed at startup, there is
    # nothing to keep in sync: the published schema IS the targets'.
    dispatch: dict = field(default_factory=dict)

    # Argument names a client must not render. The tool still declares them and
    # still applies its own defaults; this only says a clinician has no
    # business being asked. A deployment decision, which is why it lives here
    # and not in the tool: the tool knows nothing about who is looking at it.
    hidden: tuple = ()

    # How long this tool may run before it is killed, in seconds. None falls
    # back to settings.TOOL_TIMEOUT_SECONDS, and 0 there means "no limit".
    #
    # Per tool rather than global because the right number differs by two orders
    # of magnitude: a SurgMovPred prediction is seconds, an AMASSS cohort is
    # hours. One global value has to be set for the slowest tool, which means
    # every fast tool that hangs holds a slot until then.
    timeout_seconds: Optional[float] = None

    # --- splitting a cohort, as deployment.toml declared it ---------------
    #
    # `batch = false` on a tool whose model load is what a run costs. CNE holds
    # a 4.4 GB GGUF and loads it once per call: splitting a cohort of notes into
    # five batches is five loads of 4.4 GB to save nothing, the notes themselves
    # being kilobytes. The server cannot see that from a schema -- what a tool
    # pays to start is the one input to this decision that is genuinely the
    # tool's -- so it is declared.
    batch_enabled: Optional[bool] = None
    # `batch = { axis = "scans" }` where the convention derives none. See
    # conventions.batch_axis_for for what declaring this takes responsibility
    # for: a tool with two required folders pairs them per patient.
    batch_axis: Optional[str] = None
    # Overrides of this server's own caps, for this tool alone. Normally unset:
    # how much this deployment sends at once is the deployment's business, not
    # the tool's, which is the whole reason the numbers live in config.py.
    batch_max_mb: Optional[int] = None
    batch_max_files: Optional[int] = None

    # The argument whose ITEM COUNT bounds how many channels this tool can use.
    #
    # A tool that can process several things at once can never usefully open
    # more channels than it has things: one landmark does not need four
    # workers, and giving it four spends a process launch -- measured at ~3 s
    # for ALI_CBCT -- to do one landmark's work. Worse, the run is then
    # measured at a width it never reached, and the cost table learns a
    # per-channel figure that is too low.
    #
    # Absent, the batch axis is used -- the argument a cohort is already split
    # on, which for most tools is the same thing: AMASSS's channels are scans
    # and its axis is `scans`. Declared, it overrides that, which is for the
    # tools whose channel is a SUB-item: ALI_CBCT's axis is `input` (a folder
    # of scans) but its channels are the landmarks searched within one scan.
    #
    # `false` opts out entirely, for a tool whose width cannot be counted from
    # the request at all. CLIC is the case: its channel is a slice of a volume,
    # and how many slices there are is not known until the volume has been
    # read, which is after admission has decided.
    width_from: Optional[object] = None

    # The RESOLVED plan a client is handed -- `{axis, max_mb, max_files}`, or
    # None for a cohort that travels whole. Filled by conventions.derive, never
    # read from the file: the fields above are what was declared, this is what
    # those declarations came to once the conventions and the server's numbers
    # had their say.
    batch: Optional[dict] = None


_NOTHING_DECLARED = ToolDeployment()


class DeploymentConfig:
    """What this deployment says about each tool: the file, and the conventions.

    Two maps, deliberately not one. `_tools` is what `deployment.toml`
    DECLARED, and the startup checks read exactly that: a `[tools.X]` naming a
    tool this server does not serve is dead config, which only means anything
    while "configured" means "written in the file". `_resolved` is what those
    declarations came to once `conventions.derive` had merged them over the
    conventions, and it is filled by the registry as each tool loads.

    Keeping the resolution HERE is what lets the run path read it without
    knowing what a tool is. `execution/concurrency.py` needs one derived field
    -- the batch axis, which bounds how many channels a run may open -- and its
    only import from this package is this module: a leaf that imports tomllib
    and `config` and nothing else. The alternative was to reach the derived
    deployment through the `Tool` that carries it, which puts the tool registry
    (and with it schema loading, facades and every tool folder) on the import
    path of a module admission calls on every run, to read one string.
    """

    def __init__(self, tools: dict, batch: Optional[dict] = None):
        self._tools = tools
        self._batch = batch or {}
        # {tool name: the ToolDeployment conventions.derive settled on}. Empty
        # until the registry loads a tool, which is why every read of it falls
        # back to the declaration.
        self._resolved: dict = {}

    def for_tool(self, tool_name: str) -> ToolDeployment:
        """What the FILE declared for this tool, before any convention ran."""
        return self._tools.get(tool_name, _NOTHING_DECLARED)

    def record_resolved(self, tool_name: str, deployment: ToolDeployment) -> None:
        """Keep what the conventions made of this tool's declaration.

        Called once per tool by `schema_tool.load_tool`, which is the only
        place the derived object exists: it derives, hands the result to the
        `Tool`, and -- until this existed -- dropped it. Anything else asking
        this config about a derived field therefore got the raw declaration and
        could not tell, which is precisely how the batch axis came to bound no
        tool's width for as long as that rule existed.

        The last write for a name wins, so a rediscovery always leaves the
        newest answer. The one case where that is not the registry's own rule
        is two catalogues carrying the same tool name: the registry serves the
        first and refuses the second, while this keeps the second's. Nothing is
        served from it either way -- a duplicate name is a startup failure,
        named in the banner and in FAILED_TOOLS -- and the declarations that
        actually decide a width (`width_from`) are keyed by name, so they are
        the same for both.
        """
        self._resolved[tool_name] = deployment

    def resolved(self, tool_name: str) -> ToolDeployment:
        """What this deployment came to for this tool: the file AND the conventions.

        The declaration is the fallback, not a second lookup: `derive` merges
        over it, so a resolved entry carries every declared field unchanged.
        A tool nothing resolved -- an imported one, whose ArgSpecs the
        conventions never ran on, or a facade, which runs nothing itself --
        falls back to what the file said about it, which for both is the whole
        of what this server knows.
        """
        return self._resolved.get(tool_name) or self.for_tool(tool_name)

    @property
    def known_tools(self) -> tuple:
        """Every tool this config can answer for, declared or resolved.

        Distinct from `configured_tools`, and the difference is the point: the
        startup checks want the FILE's entries, while anything reading a
        derived field wants every tool the registry resolved -- which is most
        of the catalogue, none of it written in the file.
        """
        return tuple(sorted(set(self._tools) | set(self._resolved)))

    @property
    def batch_defaults(self) -> tuple:
        """`(max MB, max files)` this server asks a client to split a cohort on.

        Here as well as in config.py because this file is MOUNTED while the
        settings come from the environment: changing the number in the
        environment costs a container recreate, which drops whatever was
        running. This is the knob to turn while looking for the right value.
        """
        max_mb = self._batch.get("max_mb")
        max_files = self._batch.get("max_files")
        return (
            settings.BATCH_MAX_MB if max_mb is None else max_mb,
            settings.BATCH_MAX_FILES if max_files is None else max_files,
        )

    @property
    def configured_tools(self) -> tuple:
        return tuple(sorted(self._tools))

    def data_slug(self, tool_name: str) -> str:
        """The DATA_DIR folder to look this tool's models up in.

        The naming convention spells words out (`Batch_Dental_Seg`) while the
        data was staged run-together (`DATA/BatchDentalSeg/`). The literal name
        wins wherever it exists, so a folder that really does carry underscores
        is never mis-resolved.
        """
        declared = self.for_tool(tool_name).data_dir
        if declared:
            return declared
        if os.path.isdir(os.path.join(settings.DATA_DIR, tool_name)):
            return tool_name
        return tool_name.replace("_", "")

    def upload_limit_mb(self, tool_name: str) -> int:
        """The upload limit for this tool, in MB. Per-tool config wins; the
        global MAX_UPLOAD_MB is the default, not a ceiling."""
        limit = self.for_tool(tool_name).max_upload_mb
        return settings.MAX_UPLOAD_MB if limit is None else limit

    def timeout_seconds(self, tool_name: str) -> float:
        """How long this tool may run, in seconds; 0 means no limit.

        Per-tool config wins, and the global TOOL_TIMEOUT_SECONDS is the
        default rather than a ceiling -- an AMASSS cohort legitimately runs
        longer than anything else here, and capping it at whatever suits
        SurgMovPred would kill real work.
        """
        declared = self.for_tool(tool_name).timeout_seconds
        return settings.TOOL_TIMEOUT_SECONDS if declared is None else declared


def _tool_deployment(tool_name: str, table) -> ToolDeployment:
    where = f"deployment.toml, [tools.{tool_name}]"
    if not isinstance(table, dict):
        raise DeploymentConfigError(f"{where}: expected a table.")

    unknown = sorted(set(table) - set(_TOOL_KEYS))
    if unknown:
        # Two causes, and the message has to carry both. A typo is silent
        # otherwise: `server_selectible` would simply leave every argument
        # upload-only, with no dropdown and no error. A key a NEWER file adds is
        # the other, and it is the one an operator cannot guess -- see
        # `_unknown_value`.
        raise _unknown_value(where, f"unknown key(s) {unknown}", _TOOL_KEYS)

    selectable = table.get("server_selectable", {})
    if not isinstance(selectable, dict):
        raise DeploymentConfigError(
            f"{where}: 'server_selectable' must be a table of argument name -> "
            f"{' | '.join(SERVER_SELECTABLE_KINDS)}."
        )
    for argument, kind in selectable.items():
        if kind not in SERVER_SELECTABLE_KINDS:
            raise _unknown_value(
                where,
                f"argument '{argument}' is declared server_selectable as {kind!r}",
                SERVER_SELECTABLE_KINDS,
            )

    dispatch = table.get("dispatch", {})
    if dispatch and not isinstance(dispatch, dict):
        raise DeploymentConfigError(
            f"{where}: 'dispatch' must be a table of {{mode = \"tool name\"}}."
        )
    for mode, target in (dispatch or {}).items():
        if not isinstance(target, str) or not target:
            raise DeploymentConfigError(
                f"{where}: dispatch mode '{mode}' must name a tool, got {target!r}."
            )

    data_dir = table.get("data_dir")
    if data_dir is not None and (not isinstance(data_dir, str) or not data_dir.strip()):
        raise DeploymentConfigError(f"{where}: 'data_dir' must be a non-empty string.")

    limit = table.get("max_upload_mb")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise DeploymentConfigError(f"{where}: 'max_upload_mb' must be a positive integer.")

    hidden = table.get("hidden", ())
    if not isinstance(hidden, (list, tuple)) or not all(
        isinstance(argument, str) and argument.strip() for argument in hidden
    ):
        raise DeploymentConfigError(
            f"{where}: 'hidden' must be a list of argument names."
        )

    timeout = table.get("timeout_seconds")
    if timeout is not None and (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0
    ):
        raise DeploymentConfigError(
            f"{where}: 'timeout_seconds' must be a non-negative number (0 means no limit)."
        )

    bound = table.get("width_from")
    if bound is not None and not (
        bound is False or (isinstance(bound, str) and bound.strip())
    ):
        raise DeploymentConfigError(
            f"{where}: 'width_from' must name an argument, or be false to say "
            f"this tool's width cannot be counted from the request."
        )

    batch_enabled, batch_axis, batch_max_mb, batch_max_files = _batch_declaration(where, table)

    return ToolDeployment(
        server_selectable=dict(selectable),
        max_upload_mb=limit,
        data_dir=data_dir,
        dispatch=dict(dispatch or {}),
        hidden=tuple(hidden),
        timeout_seconds=float(timeout) if timeout is not None else None,
        width_from=bound,
        batch_enabled=batch_enabled,
        batch_axis=batch_axis,
        batch_max_mb=batch_max_mb,
        batch_max_files=batch_max_files,
    )


def _positive_cap(where: str, key: str, value) -> Optional[int]:
    """A batch cap: a non-negative integer, 0 meaning "this one does not bind"."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeploymentConfigError(
            f"{where}: '{key}' must be a non-negative integer (0 means no limit on this axis)."
        )
    return value


def _batch_declaration(where: str, table: dict) -> tuple:
    """`batch` on a [tools.X] table, as the four things it can say.

    Two spellings, because the two things an operator wants to say are of
    different kinds: `batch = false` turns batching off for this tool, and
    `batch = { ... }` configures it. A bare `true` is accepted and means
    "nothing more than the conventions already decided" -- so that writing it
    down, to record that someone looked, costs nothing.
    """
    declared = table.get("batch")
    if declared is None:
        return (None, None, None, None)
    if isinstance(declared, bool):
        return (declared, None, None, None)
    if not isinstance(declared, dict):
        raise DeploymentConfigError(
            f"{where}: 'batch' must be false, or a table of "
            f"{{{', '.join(_BATCH_KEYS)}}}."
        )

    unknown = sorted(set(declared) - set(_BATCH_KEYS))
    if unknown:
        raise _unknown_value(where, f"unknown batch key(s) {unknown}", _BATCH_KEYS)

    axis = declared.get("axis")
    if axis is not None and (not isinstance(axis, str) or not axis.strip()):
        raise DeploymentConfigError(
            f"{where}: batch 'axis' must name the argument a cohort is split on."
        )
    return (
        None,
        axis,
        _positive_cap(where, "batch.max_mb", declared.get("max_mb")),
        _positive_cap(where, "batch.max_files", declared.get("max_files")),
    )


def load(path: Optional[str] = None) -> DeploymentConfig:
    path = settings.DEPLOYMENT_CONFIG if path is None else path
    if not path or not os.path.isfile(path):
        # The normal case. Explicitly not an error: a server with no
        # deployment.toml serves every tool with upload-only inputs.
        return DeploymentConfig({})

    try:
        with open(path, "rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DeploymentConfigError(f"Cannot read {path}: {exc}")

    unknown = sorted(set(document) - {"tools", "batch"})
    if unknown:
        raise DeploymentConfigError(
            f"{path}: unknown top-level table(s) {unknown}. Expected [tools] or [batch]."
        )

    tools = document.get("tools", {})
    if not isinstance(tools, dict):
        raise DeploymentConfigError(f"{path}: [tools] must be a table of tool name -> settings.")

    batch = document.get("batch", {})
    if not isinstance(batch, dict):
        raise DeploymentConfigError(f"{path}: [batch] must be a table of max_mb / max_files.")
    unknown = sorted(set(batch) - {"max_mb", "max_files"})
    if unknown:
        raise _unknown_value(f"{path}, [batch]", f"unknown key(s) {unknown}", ("max_mb", "max_files"))
    for key in ("max_mb", "max_files"):
        _positive_cap(f"{path}, [batch]", key, batch.get(key))

    configured = {name: _tool_deployment(name, table) for name, table in tools.items()}
    logger.info("Deployment config: %d tool(s) configured (%s)", len(configured), path)
    return DeploymentConfig(configured, batch)


deployment_config: DeploymentConfig = load()

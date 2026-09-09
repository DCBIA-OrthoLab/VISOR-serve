"""The config file, loaded and validated once.

Adding a tool to a campaign is an edit to config.yaml and nothing else. That is
the whole point of this module: every name, path, argument and repetition count
a campaign needs comes from data, so a reviewer can point the harness at their
own tools without reading any Python.

Validation is strict and happens BEFORE anything runs, because the alternative
is discovering a typo forty minutes into a GPU campaign. `--dry-run` is exactly
this module plus the plan builders, with no execution.

The API token is the one value that never appears here. It is read from the
environment or from the server repo's .env, which .gitignore already covers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

# The repository root, two levels up from this file: benchmarks/ sits at the top
# of the server repo, so relative paths in config.yaml resolve against it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARKS_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BENCHMARKS_ROOT, "config.yaml")

TOKEN_ENVIRONMENT_VARIABLE = "API_TOKEN"

PATH_LOCAL = "local"
PATH_LOOPBACK = "loopback"
PATH_LAN = "lan"
PATHS = (PATH_LOCAL, PATH_LOOPBACK, PATH_LAN)

LOCAL_MODE_CONTAINER = "container"
LOCAL_MODE_HOST = "host"
LOCAL_MODES = (LOCAL_MODE_CONTAINER, LOCAL_MODE_HOST)


class ConfigError(ValueError):
    """The config file says something the harness cannot act on.

    Always names the key, because a benchmark config is a nested document and
    "invalid value" without a path into it is not actionable.
    """


def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return mapping[key]


def _as_dict(value, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _as_list(value, where: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
    return value


def resolve_path(path: str) -> str:
    """A config path, absolute. Relative ones are against the SERVER REPO root,
    not the current directory, so the same config works from anywhere."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


# ----------------------------------------------------------------------
# The pieces
# ----------------------------------------------------------------------

@dataclass
class ServerSpec:
    base_url: str
    lan_base_url: Optional[str]
    request_timeout_seconds: float
    verify_tls: bool


@dataclass
class TransferSpec:
    """The wire protocol the real Slicer client uses, restated as data.

    The defaults are that client's own constants (ServerToolsCoreLib/config.py
    and transfer.py): 8 MB parts, 4 concurrent transfers, gzip on anything not
    already compressed, results delivered by reference above 16 MB. A benchmark
    that used different numbers would be measuring a client nobody runs.
    """

    chunk_bytes: int
    parallelism: int
    connection_pool: int
    gzip_parts: bool
    result_delivery_reference: bool
    min_chunked_bytes: int


@dataclass
class LocalToolSpec:
    """Where this tool's own interpreter is, for the no-HTTP path."""

    folder: str  # relative to the tools dir, e.g. "AREG/AREG_IOSCBCT"
    package: Optional[str] = None  # importable package, for the import-cost split
    supported: bool = True
    reason: str = ""


@dataclass
class LocalSpec:
    """How the `local` path invokes runner.py. See NOTES-local-path.md."""

    mode: str
    container: str
    container_user: str
    container_tools_dir: str
    container_runner: str
    container_jobs_dir: str
    container_data_dir: str
    host_tools_dir: str
    host_runner: str
    host_jobs_dir: str
    host_data_dir: str

    @property
    def in_container(self) -> bool:
        return self.mode == LOCAL_MODE_CONTAINER

    @property
    def tools_dir(self) -> str:
        return self.container_tools_dir if self.in_container else self.host_tools_dir

    @property
    def runner(self) -> str:
        return self.container_runner if self.in_container else self.host_runner

    @property
    def jobs_dir(self) -> str:
        return self.container_jobs_dir if self.in_container else self.host_jobs_dir

    @property
    def data_dir(self) -> str:
        return self.container_data_dir if self.in_container else self.host_data_dir


@dataclass
class ToolSpec:
    """One tool as a campaign runs it: what to send, and what it costs.

    `files` are inputs that travel over the wire (and are handed to the local
    path as absolute paths on the machine that runs it). `server_files` are
    inputs the server already holds -- a model bundle, a staged test file --
    named rather than uploaded, exactly as a client names them.
    """

    name: str
    args: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    server_files: dict = field(default_factory=dict)
    output_kind: str = "files"
    # The DATA/ folder this tool reads, which is NOT always its own name --
    # deployment.toml maps AREG_CBCT, AREG_IOS and AREG_IOSCBCT all onto
    # DATA/AREG. The local path has no server to ask, so it is written here.
    data_slug: str = ""
    # Every packaged tool takes `output_dir` and the SERVER fills it in. The
    # local path has to fill it in too, or the tool writes nowhere.
    wants_output_dir: bool = True
    local: Optional[LocalToolSpec] = None
    estimated_seconds: float = 60.0
    estimated_output_mb: float = 50.0
    payload_label: Optional[str] = None
    notes: str = ""

    @property
    def supports_local(self) -> bool:
        return self.local is not None and self.local.supported


@dataclass
class GuardSpec:
    min_free_gb: float
    margin_gb: float
    scratch_dir: str


@dataclass
class Config:
    server: ServerSpec
    transfer: TransferSpec
    local: LocalSpec
    guards: GuardSpec
    repos: dict
    tools: dict  # {name: ToolSpec}
    campaigns: dict  # raw per-campaign mappings, validated by each campaign
    source_path: str

    def tool(self, name: str) -> ToolSpec:
        if name not in self.tools:
            raise ConfigError(
                f"tools: '{name}' is used by a campaign but not defined. "
                f"Defined tools: {', '.join(sorted(self.tools)) or '(none)'}"
            )
        return self.tools[name]

    def campaign(self, name: str) -> dict:
        if name not in self.campaigns:
            raise ConfigError(
                f"campaigns: no section '{name}'. "
                f"Present: {', '.join(sorted(self.campaigns)) or '(none)'}"
            )
        return self.campaigns[name]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

# A reviewer running against their own deployment can point the harness at
# their own file without editing anything: --config on the CLI, or this.
CONFIG_ENVIRONMENT_VARIABLE = "BENCHMARKS_CONFIG"


def load(path: Optional[str] = None) -> Config:
    path = path or os.environ.get(CONFIG_ENVIRONMENT_VARIABLE) or DEFAULT_CONFIG
    if not os.path.isfile(path):
        raise ConfigError(f"No config file at {path}")
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    return parse(document, path)


def parse(document: dict, source_path: str = "<memory>") -> Config:
    """Validate a already-loaded document. Separated from `load` so the tests
    can exercise validation without a file on disk."""
    server_section = _as_dict(document.get("server"), "server")
    server = ServerSpec(
        base_url=str(_require(server_section, "base_url", "server")).rstrip("/"),
        lan_base_url=(
            str(server_section["lan_base_url"]).rstrip("/")
            if server_section.get("lan_base_url")
            else None
        ),
        request_timeout_seconds=float(server_section.get("request_timeout_seconds", 3600)),
        verify_tls=bool(server_section.get("verify_tls", True)),
    )

    transfer_section = _as_dict(document.get("transfer"), "transfer")
    chunk_bytes = int(transfer_section.get("chunk_bytes", 8 * 1024 * 1024))
    transfer = TransferSpec(
        chunk_bytes=chunk_bytes,
        parallelism=int(transfer_section.get("parallelism", 4)),
        connection_pool=int(transfer_section.get("connection_pool", 16)),
        gzip_parts=bool(transfer_section.get("gzip_parts", True)),
        result_delivery_reference=bool(transfer_section.get("result_delivery_reference", True)),
        min_chunked_bytes=int(transfer_section.get("min_chunked_bytes", chunk_bytes * 2)),
    )
    if transfer.parallelism < 1:
        raise ConfigError("transfer.parallelism must be at least 1")

    local_section = _as_dict(document.get("local"), "local")
    mode = str(local_section.get("mode", LOCAL_MODE_CONTAINER))
    if mode not in LOCAL_MODES:
        raise ConfigError(f"local.mode must be one of {LOCAL_MODES}, got {mode!r}")
    local = LocalSpec(
        mode=mode,
        container=str(local_section.get("container", "")),
        container_user=str(local_section.get("container_user", "sadt")),
        container_tools_dir=str(local_section.get("container_tools_dir", "/tools")),
        container_runner=str(
            local_section.get("container_runner", "/opt/sadt/server/execution/runner.py")
        ),
        container_jobs_dir=str(local_section.get("container_jobs_dir", "/jobs")),
        container_data_dir=str(local_section.get("container_data_dir", "/DATA")),
        host_tools_dir=resolve_path(
            str(local_section.get("host_tools_dir", "../SADT-VISOR/tools"))
        ),
        host_runner=resolve_path(
            str(local_section.get("host_runner", "server/execution/runner.py"))
        ),
        host_jobs_dir=resolve_path(
            str(local_section.get("host_jobs_dir", "/tmp/sadt-benchmarks"))
        ),
        host_data_dir=resolve_path(str(local_section.get("host_data_dir", "DATA"))),
    )
    if mode == LOCAL_MODE_CONTAINER and not local.container:
        raise ConfigError("local.mode is 'container' but local.container names no container")

    guards_section = _as_dict(document.get("guards"), "guards")
    guards = GuardSpec(
        min_free_gb=float(guards_section.get("min_free_gb", 10)),
        margin_gb=float(guards_section.get("margin_gb", 5)),
        scratch_dir=resolve_path(
            str(guards_section.get("scratch_dir", "benchmarks/results/scratch"))
        ),
    )

    repos = {
        label: resolve_path(str(path))
        for label, path in _as_dict(document.get("repos"), "repos").items()
    }

    tools = {}
    for name, raw in _as_dict(document.get("tools"), "tools").items():
        tools[name] = _parse_tool(name, _as_dict(raw, f"tools.{name}"))

    campaigns = _as_dict(document.get("campaigns"), "campaigns")

    return Config(
        server=server,
        transfer=transfer,
        local=local,
        guards=guards,
        repos=repos,
        tools=tools,
        campaigns=campaigns,
        source_path=source_path,
    )


def _parse_tool(name: str, raw: dict) -> ToolSpec:
    where = f"tools.{name}"
    local_raw = raw.get("local")
    local: Optional[LocalToolSpec] = None
    if isinstance(local_raw, dict):
        local = LocalToolSpec(
            folder=str(_require(local_raw, "folder", f"{where}.local")),
            package=str(local_raw["package"]) if local_raw.get("package") else None,
            supported=bool(local_raw.get("supported", True)),
            reason=str(local_raw.get("reason", "")),
        )
    elif local_raw is not None:
        # `local: false` (or any scalar) is the explicit "this tool has no
        # no-HTTP path", and the reason belongs next to it rather than in a
        # comment nobody reads at analysis time.
        local = LocalToolSpec(
            folder="",
            supported=False,
            reason=str(raw.get("local_reason", "")) or str(local_raw),
        )

    files = {
        argument: resolve_path(str(path))
        for argument, path in _as_dict(raw.get("files"), f"{where}.files").items()
    }
    server_files = _as_dict(raw.get("server_files"), f"{where}.server_files")
    for argument, hosted in server_files.items():
        if not isinstance(hosted, dict) or "kind" not in hosted or "name" not in hosted:
            raise ConfigError(
                f"{where}.server_files.{argument}: expected a mapping with 'kind' "
                f"('model' or 'testfile') and 'name'. The kind is not guessable and the "
                f"local path has no server to ask."
            )
        if hosted["kind"] not in ("model", "testfile"):
            raise ConfigError(
                f"{where}.server_files.{argument}.kind must be 'model' or 'testfile', "
                f"got {hosted['kind']!r}"
            )

    return ToolSpec(
        name=name,
        args=_as_dict(raw.get("args"), f"{where}.args"),
        files=files,
        server_files=server_files,
        output_kind=str(raw.get("output_kind", "files")),
        data_slug=str(raw.get("data_slug", name)),
        wants_output_dir=bool(raw.get("wants_output_dir", True)),
        local=local,
        estimated_seconds=float(raw.get("estimated_seconds", 60)),
        estimated_output_mb=float(raw.get("estimated_output_mb", 50)),
        payload_label=str(raw["payload_label"]) if raw.get("payload_label") else None,
        notes=str(raw.get("notes", "")),
    )


# ----------------------------------------------------------------------
# The token
# ----------------------------------------------------------------------

def read_token(env_file: Optional[str] = None) -> Optional[str]:
    """The API token, from the environment first and a .env file second.

    Never from config.yaml, and never with a default: a token compiled into a
    published repository is a token that has leaked. Returns None when there is
    none, so a caller can say which of the two places it looked in.
    """
    from_environment = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE)
    if from_environment:
        return from_environment

    path = env_file or os.path.join(REPO_ROOT, ".env")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() == TOKEN_ENVIRONMENT_VARIABLE:
                    return value.strip().strip("'\"") or None
    except OSError:
        return None
    return None


def require_token(env_file: Optional[str] = None) -> str:
    token = read_token(env_file)
    if not token:
        raise ConfigError(
            f"No API token. Set {TOKEN_ENVIRONMENT_VARIABLE} in the environment, or put it in "
            f"the server repo's .env (which .gitignore excludes). It is never read from "
            f"config.yaml."
        )
    return token

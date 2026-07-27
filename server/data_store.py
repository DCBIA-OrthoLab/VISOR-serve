"""Storage backend for server-side data (AI models, test files) that tools
can use without the client uploading them on every request.

Everything above this module (main.py, individual Tool subclasses) only
talks to the `DataStore` interface -- never to the filesystem, a database,
or an object store directly. That is the extension point for swapping the
backend later.

# TODO to plug in an external database / object store instead of the local
# DATA/ folder: add a new DataStore subclass here (e.g. `DatabaseDataStore`)
# implementing list_models/list_testfiles/resolve_model/resolve_testfile,
# then return it from build_data_store() when settings.DATA_BACKEND selects
# it. Nothing in main.py, base.py, or any tool needs to change.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import settings


class DataNotFoundError(Exception):
    """Raised when a requested model/test file doesn't exist for a tool."""


@dataclass
class ResolvedFile:
    """A server-side data file, ready to be read from `path`.

    `is_temporary` tells the caller whether `path` must be deleted once the
    request is done. Local files under DATA_DIR are persistent reference
    data and must never be deleted (`is_temporary=False`, the default). A
    backend that materializes a remote blob to a local temp copy (e.g.
    downloaded from a database or object store) sets `is_temporary=True` so
    main.py cleans it up like any other per-request temp file.
    """

    path: str
    is_temporary: bool = False


class DataStore(ABC):
    """Read-only access to server-side models and test files, per tool."""

    @abstractmethod
    def list_models(self, tool_name: str) -> list:
        ...

    @abstractmethod
    def list_testfiles(self, tool_name: str) -> list:
        ...

    @abstractmethod
    def resolve_model(self, tool_name: str, filename: str) -> ResolvedFile:
        ...

    @abstractmethod
    def resolve_testfile(self, tool_name: str, filename: str) -> ResolvedFile:
        ...


class LocalDataStore(DataStore):
    """Reads models/test files from DATA_DIR/<tool_name>/{models,testfiles}/.

    An entry can be a single file (e.g. a zip archive) or a whole folder
    (e.g. an unpacked model directory): both are listed and resolved the
    same way, and the tool receives the resolved path either way.

    DATA_DIR is expected to be mounted read-only (see docker-compose.yml):
    this store never writes to it.
    """

    # Junk that archive tools and macOS leave behind; never listed as data.
    _IGNORED_NAMES = ("__MACOSX",)

    def __init__(self, root: str):
        self._root = root

    def list_models(self, tool_name: str) -> list:
        return self._list(tool_name, "models")

    def list_testfiles(self, tool_name: str) -> list:
        return self._list(tool_name, "testfiles")

    def resolve_model(self, tool_name: str, filename: str) -> ResolvedFile:
        return ResolvedFile(path=self._resolve(tool_name, "models", filename))

    def resolve_testfile(self, tool_name: str, filename: str) -> ResolvedFile:
        return ResolvedFile(path=self._resolve(tool_name, "testfiles", filename))

    def _list(self, tool_name: str, kind: str) -> list:
        directory = os.path.join(self._root, tool_name, kind)
        if not os.path.isdir(directory):
            return []
        return sorted(
            entry
            for entry in os.listdir(directory)
            if not entry.startswith(".") and entry not in self._IGNORED_NAMES
        )

    def _resolve(self, tool_name: str, kind: str, filename: str) -> str:
        # First line of defense: filename must be a bare name, not a path
        # (rejects "../../etc/passwd" style traversal before touching disk).
        if not filename or os.path.basename(filename) != filename:
            raise DataNotFoundError(f"Invalid file name: '{filename}'")

        directory = os.path.realpath(os.path.join(self._root, tool_name, kind))
        candidate = os.path.realpath(os.path.join(directory, filename))

        # Second line of defense: a bare name could still resolve outside
        # `directory` via a symlink planted under DATA_DIR. Confirm the
        # resolved path stays contained in the expected directory.
        if os.path.commonpath([directory, candidate]) != directory or not os.path.exists(candidate):
            raise DataNotFoundError(f"No such {kind[:-1]} '{filename}' for tool '{tool_name}'")
        return candidate


def build_data_store() -> DataStore:
    if settings.DATA_BACKEND == "local":
        return LocalDataStore(settings.DATA_DIR)
    raise ValueError(f"Unknown DATA_BACKEND: '{settings.DATA_BACKEND}'")


data_store: DataStore = build_data_store()

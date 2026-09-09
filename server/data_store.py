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

    `is_temporary` says whether `path` must be deleted once the request is
    done. Local files under DATA_DIR are persistent reference data and must
    never be deleted (the default). A backend that materializes a remote blob
    to a local temp copy sets it True so main.py cleans the copy up.
    """

    path: str
    is_temporary: bool = False


def _size_on_disk(path: str):
    """Bytes an entry holds, walking a folder. None if it cannot be read.

    A folder's size is the sum of its files, not its own inode: what a client
    downloads is everything inside it. Symlinks are counted by their target's
    size only when the target is inside -- an unreadable entry reports None
    rather than a wrong number, because the number is shown to a user deciding
    whether to start a transfer.
    """
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for directory, _subdirs, names in os.walk(path):
            for name in names:
                try:
                    total += os.path.getsize(os.path.join(directory, name))
                except OSError:
                    continue
        return total
    except OSError:
        return None


class DataStore(ABC):
    """Read-only access to server-side models and test files, per tool."""

    @abstractmethod
    def list_models(self, tool_name: str) -> list:
        ...

    @abstractmethod
    def list_testfiles(self, tool_name: str) -> list:
        ...

    def describe(self, tool_name: str, kind: str) -> list:
        """`[{"name", "kind", "size"}]` for one of "models"/"testfiles".

        A name alone cannot say whether an entry is one scan or a cohort of
        forty, and the client now downloads what a user picks -- so the picker
        has to show what the click costs. `kind` is "file" or "folder", which a
        bare name never said either: a folder simply looked like a file whose
        extension was missing.

        Default implementation so a backend that cannot cheaply size its
        entries still lists them; it reports `None` for both fields rather
        than guessing.
        """
        return [{"name": name, "kind": None, "size": None}
                for name in (self.list_models(tool_name) if kind == "models"
                             else self.list_testfiles(tool_name))]

    @abstractmethod
    def resolve_model(self, tool_name: str, filename: str) -> ResolvedFile:
        ...

    @abstractmethod
    def resolve_testfile(self, tool_name: str, filename: str) -> ResolvedFile:
        ...


class LocalDataStore(DataStore):
    """Reads models/test files from DATA_DIR/<tool_name>/{models,testfiles}/.

    An entry can be a single file or a whole folder; both are listed and
    resolved the same way. DATA_DIR is mounted read-only (see
    docker-compose.yml) and this store never writes to it.
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

    def describe(self, tool_name: str, kind: str) -> list:
        directory = os.path.join(self._root, tool_name, kind)
        described = []
        for name in self._list(tool_name, kind):
            path = os.path.join(directory, name)
            described.append({
                "name": name,
                "kind": "folder" if os.path.isdir(path) else "file",
                "size": _size_on_disk(path),
            })
        return described

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

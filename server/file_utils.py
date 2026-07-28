"""Shared helpers for tools that need to unzip an upload and/or load tabular
data files (CSV/XLSX/ODS). Factored out so multiple tools can reuse the same
logic instead of each reimplementing zip extraction and tabular loading.
"""

import contextvars
import glob
import os
import tempfile
import zipfile

import pandas as pd

from config import settings


# Scratch dirs handed out during the request being served, so main.py can
# remove them all even when run() raises before returning any path -- a crash
# mid-inference must not leave patient data behind. A ContextVar (rather than a
# module global) keeps concurrent requests from seeing each other's list; the
# list is mutated in place, never reassigned, so the copy anyio hands to the
# worker thread stays the same object the request handler holds.
_scratch_dirs: contextvars.ContextVar = contextvars.ContextVar("scratch_dirs", default=None)


def track_scratch_dirs() -> list:
    """Start recording scratch dirs for this request; returns the live list."""
    created: list = []
    _scratch_dirs.set(created)
    return created


def make_scratch_dir(prefix: str = "tool_") -> str:
    """Fresh writable scratch dir under settings.TEMP_DIR for one request.

    For tools whose inputs all come from the read-only data store: there is
    no upload work dir to write next to, so extraction/output files go here
    instead. main.py deletes this directory once the request is over --
    whether it succeeded or run() raised.
    """
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    scratch_dir = tempfile.mkdtemp(prefix=prefix, dir=settings.TEMP_DIR)
    tracked = _scratch_dirs.get()
    if tracked is not None:  # None when a tool is called outside a request
        tracked.append(scratch_dir)
    return scratch_dir


class BadArchiveError(Exception):
    """Raised when a zip archive can't be trusted or can't be read.

    main.py maps this to a 400: it always means the *client's* archive is at
    fault, never a server bug.
    """


# Junk that archive tools and macOS leave next to the real content; ignored
# when deciding whether an archive has a single root folder.
_IGNORED_ENTRIES = ("__MACOSX",)

_SYMLINK_MODE = 0o120000


def extract_zip(
    zip_path: str,
    extract_dir: str = None,
    strip_single_root: bool = False,
    max_total_bytes: int = None,
) -> str:
    """Extract a zip archive, by default into an "extracted" folder next to it.

    Returns the path to the directory holding the extracted content.

    Archives uploaded by a client are untrusted input, so members are checked
    before anything is written to disk:

    - a member resolving outside `extract_dir` is refused (zip slip, e.g. a
      member literally named "../../etc/cron.d/x");
    - symlink members are refused (they would otherwise let a later member,
      or the tool itself, read/write through the link);
    - `max_total_bytes` caps the *uncompressed* total, so a zip bomb can't
      fill the server's disk.

    `strip_single_root=True` returns the inner folder when the archive holds
    exactly one — zipping "patients/" on any OS produces "patients/<files>",
    and callers almost always want the files, not the wrapper.
    """
    if extract_dir is None:
        extract_dir = os.path.join(os.path.dirname(zip_path), "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    root = os.path.realpath(extract_dir)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            total_bytes = 0
            for member in zf.infolist():
                target = os.path.realpath(os.path.join(root, member.filename))
                if target != root and os.path.commonpath([root, target]) != root:
                    raise BadArchiveError(
                        f"Refusing to extract '{member.filename}': it points outside the archive."
                    )
                if (member.external_attr >> 16) & 0o170000 == _SYMLINK_MODE:
                    raise BadArchiveError(
                        f"Refusing to extract '{member.filename}': symbolic links are not allowed."
                    )
                total_bytes += member.file_size
                if max_total_bytes is not None and total_bytes > max_total_bytes:
                    raise BadArchiveError(
                        f"Archive expands to more than {max_total_bytes // (1024 * 1024)} MB."
                    )
            zf.extractall(root)
    except zipfile.BadZipFile as exc:
        raise BadArchiveError(f"Not a readable zip archive: {exc}")

    if strip_single_root:
        entries = [
            entry
            for entry in os.listdir(root)
            if entry not in _IGNORED_ENTRIES and not entry.startswith(".")
        ]
        if len(entries) == 1 and os.path.isdir(os.path.join(root, entries[0])):
            return os.path.join(root, entries[0])
    return root


def make_zip(sources, destination: str) -> str:
    """Bundle `sources` into the zip archive at `destination`; return its path.

    `sources` is either a single directory path (its *contents* land at the
    root of the archive) or a list of file/directory paths (each kept under
    its own base name). This is what backs `output_kind = "files"`: a tool
    returns the paths it produced and never writes zip code of its own.
    """
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    sources = [str(source) for source in sources]
    if not sources:
        raise ValueError("Nothing to zip: no output path given.")

    # A lone directory is the "here is my whole output folder" case: put its
    # contents at the archive root rather than nesting them under its name.
    flatten_root = len(sources) == 1 and os.path.isdir(sources[0])

    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    written: set = set()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as zf:
        for source in sources:
            if not os.path.exists(source):
                raise FileNotFoundError(f"Output path does not exist: {source}")
            if os.path.isdir(source):
                base = source if flatten_root else os.path.dirname(source.rstrip(os.sep))
                for dir_path, _, file_names in os.walk(source):
                    for file_name in sorted(file_names):
                        file_path = os.path.join(dir_path, file_name)
                        _add_to_zip(zf, file_path, os.path.relpath(file_path, base), written)
            else:
                _add_to_zip(zf, source, os.path.basename(source), written)
    return destination


def _add_to_zip(zf: zipfile.ZipFile, file_path: str, arcname: str, written: set) -> None:
    # Two outputs sharing a base name would silently overwrite each other
    # inside the archive, so the client would receive fewer files than the
    # tool produced. Fail loudly instead.
    if arcname in written:
        raise ValueError(f"Duplicate name in output archive: '{arcname}'")
    written.add(arcname)
    zf.write(file_path, arcname)


def zip_directory(source_dir: str, zip_path: str) -> str:
    """Zip the whole contents of `source_dir` into `zip_path`.

    Paths inside the archive are relative to `source_dir`, so unzipping it
    reproduces the folder's contents (not the folder itself nested one level
    deeper). Used by tools whose result is several files but whose HTTP
    response can only carry one blob.
    """
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(source_dir):
            for name in sorted(files):
                absolute = os.path.join(root, name)
                zf.write(absolute, os.path.relpath(absolute, source_dir))
    return zip_path


def load_tabular_file(file_path: str) -> pd.DataFrame:
    """Load a single CSV, XLSX, or ODS file into a DataFrame."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    if ext == ".xlsx":
        return pd.read_excel(file_path)
    if ext == ".ods":
        return pd.read_excel(file_path, engine="odf")
    raise ValueError(f"Unsupported file extension '{ext}' for tabular file: {file_path}")


def load_tabular_directory(directory_path: str) -> pd.DataFrame:
    """Load and concatenate every CSV/XLSX/ODS file found directly in a directory."""
    extensions = ["*.csv", "*.xlsx", "*.ods"]
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(directory_path, ext)))
        all_files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
    all_files = sorted(set(all_files))

    if not all_files:
        raise FileNotFoundError(f"No valid CSV, XLSX, or ODS files found in: {directory_path}")

    df_list = [load_tabular_file(file_path) for file_path in all_files]
    return pd.concat(df_list, ignore_index=True)

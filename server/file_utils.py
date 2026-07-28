"""Shared helpers for tools that need to unzip an upload and/or load tabular
data files (CSV/XLSX/ODS). Factored out so multiple tools can reuse the same
logic instead of each reimplementing zip extraction and tabular loading.
"""

import glob
import os
import tempfile
import zipfile

import pandas as pd

from config import settings


def make_scratch_dir(prefix: str = "tool_") -> str:
    """Fresh writable scratch dir under settings.TEMP_DIR for one request.

    For tools whose inputs all come from the read-only data store: there is
    no upload work dir to write next to, so extraction/output files go here
    instead. main.py deletes this directory once the response has been
    streamed (it cleans up the TEMP_DIR folder containing the tool's output).
    """
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=settings.TEMP_DIR)


def extract_zip(zip_path: str, extract_dir: str = None) -> str:
    """Extract a zip archive, by default into an "extracted" folder next to it.

    Returns the path to the extraction directory.
    """
    if extract_dir is None:
        extract_dir = os.path.join(os.path.dirname(zip_path), "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


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

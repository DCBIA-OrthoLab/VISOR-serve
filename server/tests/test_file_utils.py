"""Unit tests for file_utils.make_zip's per-member compression choice.

Run with: cd server && pytest tests/test_file_utils.py

Why this exists (2026-08-07): make_zip deflated every member at level 6, which
compresses at ~30 MB/s on one core. Most of what this server ships is already
gzip-compressed (.nii.gz volumes and segmentations), so a result archive paid
seconds of CPU per request to shrink by ~0% -- and the client then paid the
same tax twice more, CRC-checking and extracting members that decompress ~5x
faster when STORED. Storing the already-compressed members and deflating the
rest at settings.ZIP_COMPRESSLEVEL keeps the archive an ordinary zip that any
reader (the Slicer client included) handles without change.
"""

import os
import zipfile

os.environ.setdefault("API_TOKEN", "test-token")

import pytest

from file_utils import extract_zip, make_zip


@pytest.fixture
def outputs(tmp_path):
    """A tool-output-shaped folder: one already-compressed volume, one
    compressible binary mesh."""
    folder = tmp_path / "run_output"
    folder.mkdir()
    # Random enough not to deflate, and .gz says it is already compressed.
    (folder / "patient_seg.nii.gz").write_bytes(os.urandom(64 * 1024))
    # Repetitive bytes, the shape of a binary .vtk: worth deflating.
    (folder / "patient_surface.vtk").write_bytes(b"\x00\x01\x02\x03" * (64 * 1024))
    return folder


def test_already_compressed_members_are_stored_not_redeflated(tmp_path, outputs):
    """DEFLATE on a .nii.gz was measured gaining ~0% for 3.3s of CPU per 94 MB;
    such members must be stored as-is, whatever the case of their extension."""
    (outputs / "REPORT.XLSX").write_bytes(os.urandom(1024))
    archive = make_zip(str(outputs), str(tmp_path / "out.zip"))

    with zipfile.ZipFile(archive) as zf:
        assert zf.getinfo("patient_seg.nii.gz").compress_type == zipfile.ZIP_STORED
        assert zf.getinfo("REPORT.XLSX").compress_type == zipfile.ZIP_STORED


def test_compressible_members_are_still_deflated(tmp_path, outputs):
    """Binary .vtk still compresses ~2.7:1, so storing EVERYTHING would trade
    real wire bytes for the CPU saved on the .gz members. It must keep deflating."""
    archive = make_zip(str(outputs), str(tmp_path / "out.zip"))

    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo("patient_surface.vtk")
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert info.compress_size < info.file_size


def test_mixed_archive_round_trips_bytes_exactly(tmp_path, outputs):
    """A mixed stored/deflated archive is an ordinary zip: extraction must
    return every member byte-identical, through our own extract_zip."""
    archive = make_zip(str(outputs), str(tmp_path / "out.zip"))

    extracted = extract_zip(archive, str(tmp_path / "extracted"))
    for name in ("patient_seg.nii.gz", "patient_surface.vtk"):
        assert (
            open(os.path.join(extracted, name), "rb").read()
            == open(os.path.join(str(outputs), name), "rb").read()
        )

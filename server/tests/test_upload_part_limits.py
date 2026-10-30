"""A part upload is bounded before it is read whole, and before it is inflated."""

import gzip

from fastapi.testclient import TestClient

import main
from config import settings
from wire import transfer

client = TestClient(main.app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}


def _session(size: int) -> str:
    response = client.post("/uploads", headers=AUTH,
                           json={"filename": "scan.nii", "size": size, "chunk_size": 1024 * 1024})
    assert response.status_code == 200, response.text
    return response.json()["upload_id"]


def test_a_gzip_bomb_is_refused_without_being_inflated(monkeypatch):
    monkeypatch.setattr(transfer, "MAX_CHUNK_BYTES", 1024 * 1024)
    upload_id = _session(1024 * 1024)
    bomb = gzip.compress(b"\0" * (8 * 1024 * 1024))
    response = client.put(f"/uploads/{upload_id}/parts/0", headers={**AUTH, "Content-Encoding": "gzip"},
                          content=bomb)
    assert response.status_code == 400
    assert "inflates past" in response.json()["detail"]


def test_a_body_larger_than_any_part_is_refused(monkeypatch):
    monkeypatch.setattr(transfer, "MAX_CHUNK_BYTES", 1024 * 1024)
    upload_id = _session(1024 * 1024)
    response = client.put(f"/uploads/{upload_id}/parts/0", headers=AUTH,
                          content=b"x" * (3 * 1024 * 1024))
    assert response.status_code == 413


def test_an_ordinary_gzip_part_still_lands():
    upload_id = _session(1024 * 1024)
    response = client.put(f"/uploads/{upload_id}/parts/0", headers={**AUTH, "Content-Encoding": "gzip"},
                          content=gzip.compress(b"y" * (1024 * 1024)))
    assert response.status_code == 200, response.text


def test_a_part_is_held_in_memory_once_on_its_way_to_disk(tmp_path, monkeypatch):
    """A part was copied whole twice between the socket and the disk -- once
    into `bytes`, once by the slice handed to `pwrite` -- so a 64 MB part held
    three times that, per part, with parts arriving in parallel."""
    import asyncio
    import hashlib
    import tracemalloc

    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    size = 16 * 1024 * 1024
    session = transfer.create_upload("scan.nii", size, chunk_size=size)
    piece = b"x" * 65536
    digest = hashlib.sha256(piece * (size // len(piece))).hexdigest()

    class _Request:
        async def stream(self):
            for _ in range(size // len(piece)):
                yield piece

    async def receive():
        # Measured inside the coroutine, the way the endpoint awaits it: what
        # a test harness does with the return value is not the server's.
        tracemalloc.start()
        try:
            body = await main._read_bounded(_Request(), size + 1024 * 1024)
            transfer.write_part(session, 0, body, digest)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    peak = asyncio.run(receive())
    assert set(session.received_parts()) == {0}
    assert peak < 1.3 * size, f"peak {peak / size:.2f}x the part"

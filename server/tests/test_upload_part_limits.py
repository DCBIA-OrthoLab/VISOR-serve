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

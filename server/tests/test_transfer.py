"""Chunked upload sessions and range-served results (transfer.py + its endpoints).

Run with: cd server && ./venv/bin/pytest tests/test_transfer.py
"""

import hashlib
import gzip
import json
import os
import time

os.environ["API_TOKEN"] = "test-token"

import pytest
from fastapi.testclient import TestClient

import main
import transfer
from config import settings
from main import app

client = TestClient(app)
TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _open_session(payload: bytes, chunk_size: int = 1024 * 1024, filename: str = "scan.nii.gz"):
    response = client.post(
        "/uploads",
        headers=AUTH,
        json={"filename": filename, "size": len(payload), "chunk_size": chunk_size},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _put_parts(upload_id: str, payload: bytes, chunk_size: int, indices=None, checksum=True):
    """Send parts, defaulting to all of them. `indices` is what makes a
    half-finished upload testable, the same thing a dropped connection does."""
    sent = []
    total = max(1, (len(payload) + chunk_size - 1) // chunk_size)
    for index in range(total) if indices is None else indices:
        chunk = payload[index * chunk_size : (index + 1) * chunk_size]
        headers = dict(AUTH)
        if checksum:
            headers["X-Part-SHA256"] = hashlib.sha256(chunk).hexdigest()
        response = client.put(f"/uploads/{upload_id}/parts/{index}", headers=headers, content=chunk)
        assert response.status_code == 200, response.text
        sent.append(response.json())
    return sent


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------

def test_upload_endpoints_require_a_token():
    assert client.post("/uploads", json={"filename": "a.nii.gz", "size": 1}).status_code == 401
    assert client.get("/uploads/whatever").status_code == 401
    assert client.put("/uploads/whatever/parts/0", content=b"x").status_code == 401
    assert client.get("/results/whatever").status_code == 401


def test_parts_reassemble_into_the_original_bytes():
    payload = os.urandom(300_000)
    session = _open_session(payload, chunk_size=1024 * 1024)
    assert session["part_count"] == 1
    _put_parts(session["upload_id"], payload, session["chunk_size"])

    claimed = os.path.join(settings.TEMP_DIR, "claimed.bin")
    transfer.claim_upload(session["upload_id"], claimed)
    assert open(claimed, "rb").read() == payload
    os.remove(claimed)


def test_parts_may_arrive_in_any_order():
    """The point of the whole design: parts go out on several connections at
    once, so nothing may depend on the order they land in."""
    chunk_size = 1024 * 1024
    payload = os.urandom(chunk_size * 3 + 1234)
    session = _open_session(payload, chunk_size=chunk_size)
    assert session["part_count"] == 4
    _put_parts(session["upload_id"], payload, chunk_size, indices=[3, 0, 2, 1])

    claimed = os.path.join(settings.TEMP_DIR, "shuffled.bin")
    transfer.claim_upload(session["upload_id"], claimed)
    assert open(claimed, "rb").read() == payload
    os.remove(claimed)


def test_status_reports_what_is_missing_so_a_client_can_resume():
    chunk_size = 1024 * 1024
    payload = os.urandom(chunk_size * 3)
    session = _open_session(payload, chunk_size=chunk_size)
    _put_parts(session["upload_id"], payload, chunk_size, indices=[0, 2])

    status_response = client.get(f"/uploads/{session['upload_id']}", headers=AUTH)
    assert status_response.json()["missing_parts"] == [1]

    # Resuming sends only the gap, and re-sending a part already held is
    # allowed (a client that never saw the first response retries it).
    _put_parts(session["upload_id"], payload, chunk_size, indices=[1, 0])
    assert client.get(f"/uploads/{session['upload_id']}", headers=AUTH).json()["missing_parts"] == []


def test_an_incomplete_upload_is_never_handed_to_a_tool():
    chunk_size = 1024 * 1024
    payload = os.urandom(chunk_size * 2)
    session = _open_session(payload, chunk_size=chunk_size)
    _put_parts(session["upload_id"], payload, chunk_size, indices=[0])

    with pytest.raises(transfer.TransferError, match="incomplete"):
        transfer.claim_upload(session["upload_id"], os.path.join(settings.TEMP_DIR, "nope.bin"))


def test_a_part_failing_its_checksum_is_not_written():
    payload = os.urandom(50_000)
    session = _open_session(payload)
    response = client.put(
        f"/uploads/{session['upload_id']}/parts/0",
        headers={**AUTH, "X-Part-SHA256": hashlib.sha256(b"something else").hexdigest()},
        content=payload,
    )
    assert response.status_code == 400
    assert "checksum" in response.json()["detail"]
    # Rejected, not half-applied: the client retries this part alone.
    assert client.get(f"/uploads/{session['upload_id']}", headers=AUTH).json()["missing_parts"] == [0]


def test_a_part_of_the_wrong_length_is_refused():
    """Only the last part is short, and only by a known amount. Anything else
    means the two sides disagree about the layout, and writing it would leave
    a hole or an overlap in the file."""
    payload = os.urandom(1024 * 1024 * 2)
    session = _open_session(payload, chunk_size=1024 * 1024)
    response = client.put(
        f"/uploads/{session['upload_id']}/parts/0", headers=AUTH, content=payload[:100]
    )
    assert response.status_code == 400
    assert "expected" in response.json()["detail"]


def test_a_part_outside_the_declared_layout_is_refused():
    payload = os.urandom(1000)
    session = _open_session(payload)
    assert client.put(
        f"/uploads/{session['upload_id']}/parts/7", headers=AUTH, content=payload
    ).status_code == 400


def test_a_gzipped_part_is_stored_decompressed():
    payload = b"NIfTI-ish uncompressed bytes, very repetitive. " * 5000
    session = _open_session(payload, chunk_size=1024 * 1024, filename="scan.nii")
    compressed = gzip.compress(payload, 1)
    assert len(compressed) < len(payload)  # the reason the client bothers
    response = client.put(
        f"/uploads/{session['upload_id']}/parts/0",
        headers={
            **AUTH,
            "Content-Encoding": "gzip",
            # The checksum covers what LANDS on disk, not what travelled.
            "X-Part-SHA256": hashlib.sha256(payload).hexdigest(),
        },
        content=compressed,
    )
    assert response.status_code == 200, response.text

    claimed = os.path.join(settings.TEMP_DIR, "gz.bin")
    transfer.claim_upload(session["upload_id"], claimed)
    assert open(claimed, "rb").read() == payload
    os.remove(claimed)


def test_a_file_over_the_limit_is_refused_before_any_byte_travels():
    over = (settings.MAX_UPLOAD_MB + 1) * 1024 * 1024
    response = client.post(
        "/uploads", headers=AUTH, json={"filename": "huge.nii.gz", "size": over}
    )
    assert response.status_code == 413


def test_an_unknown_or_malformed_id_is_a_404_not_a_path_lookup():
    assert client.get("/uploads/does-not-exist-but-well-formed", headers=AUTH).status_code == 404
    # Traversal never reaches the filesystem: the id is matched against the
    # token alphabet first.
    assert client.get("/uploads/..%2F..%2Fetc", headers=AUTH).status_code == 404
    assert client.get("/results/..%2F..%2Fetc", headers=AUTH).status_code == 404


def test_deleting_a_session_drops_it():
    payload = os.urandom(1000)
    session = _open_session(payload)
    assert client.delete(f"/uploads/{session['upload_id']}", headers=AUTH).status_code == 200
    assert client.get(f"/uploads/{session['upload_id']}", headers=AUTH).status_code == 404


def _age(directory: str, seconds: int) -> None:
    """Pretend nothing has touched `directory` for `seconds`."""
    when = os.path.getmtime(directory) - seconds
    os.utime(directory, (when, when))


def test_the_reaper_removes_only_expired_directories():
    fresh = _open_session(os.urandom(100))
    stale = _open_session(os.urandom(100))
    stale_dir = os.path.join(settings.TEMP_DIR, "uploads", stale["upload_id"])
    _age(stale_dir, settings.TRANSFER_TTL_SECONDS + 60)

    transfer.reap_expired()
    assert not os.path.isdir(stale_dir)
    assert os.path.isdir(os.path.join(settings.TEMP_DIR, "uploads", fresh["upload_id"]))


def test_a_transfer_still_moving_is_never_reaped():
    """The TTL is an IDLE timeout, not an age limit. A slow upload that keeps
    sending parts must survive however long it takes, or the bound could never
    be short enough to be a useful confidentiality guarantee."""
    chunk_size = 1024 * 1024
    payload = os.urandom(chunk_size * 2)
    session = _open_session(payload, chunk_size=chunk_size)
    directory = os.path.join(settings.TEMP_DIR, "uploads", session["upload_id"])

    _age(directory, settings.TRANSFER_TTL_SECONDS + 60)   # a long stall...
    _put_parts(session["upload_id"], payload, chunk_size, indices=[0])   # ...then a part lands

    transfer.reap_expired()
    assert os.path.isdir(directory), "an upload that just made progress was reaped"


def test_reading_a_range_keeps_the_result_alive():
    """Same rule on the way down: a big result on a slow link is many minutes
    of range requests, and none of them may race the reaper."""
    stored = _stored_result(os.urandom(50_000))
    _age(stored.directory, settings.TRANSFER_TTL_SECONDS + 60)

    response = client.get(
        f"/results/{stored.result_id}", headers={**AUTH, "Range": "bytes=0-99"}
    )
    assert response.status_code == 206

    transfer.reap_expired()
    assert os.path.isdir(stored.directory), "a result being downloaded was reaped"


def test_an_abandoned_result_does_expire():
    """The other half of the guarantee: a client that never came back (Slicer
    closed mid-download, machine asleep) must not leave patient data on the
    server. This is the bound that DELETE only makes rarer, never redundant."""
    stored = _stored_result(os.urandom(50_000))
    _age(stored.directory, settings.TRANSFER_TTL_SECONDS + 60)

    transfer.reap_expired()
    assert not os.path.isdir(stored.directory)
    assert client.get(f"/results/{stored.result_id}", headers=AUTH).status_code == 404


def test_the_idle_timeout_is_short_enough_to_be_a_real_bound():
    """A guard on the setting itself. The whole argument for reference delivery
    resting on 'the reaper catches it' only holds while this is minutes, not
    hours: it is the worst-case time patient data can sit undownloaded."""
    assert settings.TRANSFER_TTL_SECONDS <= 3600
    assert settings.TRANSFER_SWEEP_SECONDS <= 300


# ----------------------------------------------------------------------
# /run through an upload session
# ----------------------------------------------------------------------

def test_run_accepts_an_upload_reference_in_place_of_a_multipart_file():
    payload = b"col\n1\n2\n"
    session = _open_session(payload, filename="input.csv")
    _put_parts(session["upload_id"], payload, session["chunk_size"])

    response = client.post(
        "/run/Example_Tool",
        headers=AUTH,
        data={
            "label": "x",
            "threshold": "0.5",
            "__uploads__": json.dumps({"input": session["upload_id"]}),
        },
    )
    assert response.status_code == 200, response.text
    # And the session is gone: its blob was renamed into the run's work dir.
    assert client.get(f"/uploads/{session['upload_id']}", headers=AUTH).status_code == 404


def test_run_validates_an_uploaded_extension_the_same_way_either_route():
    payload = b"not a csv"
    session = _open_session(payload, filename="input.exe")
    _put_parts(session["upload_id"], payload, session["chunk_size"])

    response = client.post(
        "/run/Example_Tool",
        headers=AUTH,
        data={"label": "x", "__uploads__": json.dumps({"input": session["upload_id"]})},
    )
    assert response.status_code == 400
    assert "extension" in response.json()["detail"]
    # The refused session does not linger on disk holding patient data.
    assert client.get(f"/uploads/{session['upload_id']}", headers=AUTH).status_code == 404


def test_run_rejects_a_malformed_uploads_field():
    response = client.post(
        "/run/Example_Tool", headers=AUTH, data={"label": "x", "__uploads__": "{not json"}
    )
    assert response.status_code == 400


def test_run_rejects_an_unknown_upload_id():
    response = client.post(
        "/run/Example_Tool",
        headers=AUTH,
        data={"label": "x", "__uploads__": json.dumps({"input": "0123456789abcdefghij"})},
    )
    assert response.status_code == 404


# ----------------------------------------------------------------------
# Range-served results
# ----------------------------------------------------------------------

def _stored_result(payload: bytes):
    path = os.path.join(settings.TEMP_DIR, "to_store.bin")
    with open(path, "wb") as handle:
        handle.write(payload)
    return transfer.store_result(path, "application/zip")


def test_a_result_can_be_pulled_down_in_pieces_and_reassembles_exactly():
    payload = os.urandom(250_000)
    stored = _stored_result(payload)

    spans = [(0, 99_999), (100_000, 199_999), (200_000, len(payload) - 1)]
    rebuilt = bytearray(len(payload))
    for start, end in spans:
        response = client.get(
            f"/results/{stored.result_id}",
            headers={**AUTH, "Range": f"bytes={start}-{end}"},
        )
        assert response.status_code == 206
        assert response.headers["Content-Range"] == f"bytes {start}-{end}/{len(payload)}"
        rebuilt[start : end + 1] = response.content
    assert bytes(rebuilt) == payload


def test_a_result_without_a_range_is_served_whole():
    payload = os.urandom(10_000)
    stored = _stored_result(payload)
    response = client.get(f"/results/{stored.result_id}", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["Accept-Ranges"] == "bytes"
    assert response.content == payload


def test_an_unsatisfiable_range_reports_the_real_size():
    payload = os.urandom(1000)
    stored = _stored_result(payload)
    response = client.get(
        f"/results/{stored.result_id}", headers={**AUTH, "Range": "bytes=5000-6000"}
    )
    assert response.status_code == 416
    assert response.headers["Content-Range"] == "bytes */1000"


def test_a_suffix_range_reads_from_the_end():
    payload = os.urandom(1000)
    stored = _stored_result(payload)
    response = client.get(f"/results/{stored.result_id}", headers={**AUTH, "Range": "bytes=-100"})
    assert response.status_code == 206
    assert response.content == payload[-100:]


def test_deleting_a_result_frees_it():
    stored = _stored_result(os.urandom(100))
    assert client.delete(f"/results/{stored.result_id}", headers=AUTH).status_code == 200
    assert client.get(f"/results/{stored.result_id}", headers=AUTH).status_code == 404


def test_the_reaper_runs_on_a_timer_with_no_request_at_all():
    """The hole this closes: reaping only when a new session is created means
    an idle server never reaps, and an idle server is exactly where an
    abandoned result sits longest."""
    stored = _stored_result(os.urandom(1000))
    _age(stored.directory, settings.TRANSFER_TTL_SECONDS + 60)
    original = settings.TRANSFER_SWEEP_SECONDS
    settings.TRANSFER_SWEEP_SECONDS = 0.05
    try:
        # Entering the context runs the app lifespan, which starts the loop.
        with TestClient(app):
            deadline = time.monotonic() + 10
            while os.path.isdir(stored.directory) and time.monotonic() < deadline:
                time.sleep(0.05)
    finally:
        settings.TRANSFER_SWEEP_SECONDS = original

    assert not os.path.isdir(stored.directory), "the timed sweep never ran"


def test_a_small_result_is_streamed_and_deleted_server_side():
    """Below RESULT_REFERENCE_MIN_MB the client's header is ignored and the old
    path is used, which deletes the file when the response ends with no
    dependency on the client ever coming back. Parallel ranges would buy
    nothing at this size, so there is no reason to give that up."""
    before = set(os.listdir(os.path.join(settings.TEMP_DIR, "results")))

    response = client.post(
        "/run/Example_Tool",
        headers={**AUTH, "X-Result-Delivery": "reference"},
        data={"label": "x", "threshold": "0.5", "input": "", "iterations": "1"},
        files={"input": ("input.csv", b"a,b\n1,2\n")},
    )
    assert response.status_code == 200, response.text
    # The bytes themselves, not a pointer to them.
    assert "application/json" not in response.headers.get("Content-Type", "")
    assert set(os.listdir(os.path.join(settings.TEMP_DIR, "results"))) == before


def test_a_large_result_takes_the_reference_route():
    original = main._RESULT_REFERENCE_MIN_BYTES
    main._RESULT_REFERENCE_MIN_BYTES = 0      # every result counts as large
    try:
        response = client.post(
            "/run/Example_Tool",
            headers={**AUTH, "X-Result-Delivery": "reference"},
            data={"label": "x", "threshold": "0.5"},
            files={"input": ("input.csv", b"a,b\n1,2\n")},
        )
    finally:
        main._RESULT_REFERENCE_MIN_BYTES = original

    assert response.status_code == 200, response.text
    reference = response.json()["result_ref"]
    assert reference["size"] > 0

    # And it really is fetchable, then really is gone once released.
    assert client.get(f"/results/{reference['result_id']}", headers=AUTH).status_code == 200
    assert client.delete(f"/results/{reference['result_id']}", headers=AUTH).status_code == 200
    assert client.get(f"/results/{reference['result_id']}", headers=AUTH).status_code == 404


def test_run_returns_a_reference_when_the_client_asks_for_one():
    """The delivery header is opt-in, and everything else about the run is
    unchanged, which is what lets an old client keep working against a new
    server, and a new client against an old one."""
    response = client.post(
        "/run/Test_Tool",
        headers={**AUTH, "X-Result-Delivery": "reference"},
        data={"text_1": "a", "text_2": "b"},
    )
    # test_tool is a "text" tool: reference delivery only applies to file
    # results, so this must come back exactly as it always did.
    assert response.status_code == 200
    assert "result" in response.json()


def test_parse_range_ignores_forms_it_does_not_serve():
    # Multi-range and unit-less forms fall back to "send the whole thing"
    # rather than being answered with a body that does not match the header.
    assert transfer.parse_range("bytes=0-10,20-30", 100) is None
    assert transfer.parse_range("items=0-10", 100) is None
    assert transfer.parse_range(None, 100) is None
    assert transfer.parse_range("bytes=0-", 100) == (0, 99)
    assert transfer.parse_range("bytes=90-500", 100) == (90, 99)

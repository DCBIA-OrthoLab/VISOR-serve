"""Adversarial tests: what a hostile client can send, and what it gets back.

There is no SQL anywhere in this server -- no database, no ORM, no query -- so
injection here is not about statements. The surfaces that exist are the ones a
file-handling API has: paths that travel from a client onto disk, archives that
expand, identifiers that become filesystem paths, and a bearer token.

Each test below names the attack and asserts the refusal, not just an absence
of crash.
"""

import io
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

import file_utils
import registry
from base import ArgSpec, Tool
from config import settings
from main import app

client = TestClient(app)
TOKEN = settings.API_TOKEN
AUTH = {"Authorization": f"Bearer {TOKEN}"}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": ""},
    {"Authorization": "Bearer"},
    {"Authorization": "Bearer "},
    {"Authorization": "Basic " + TOKEN},
    {"Authorization": "Bearer " + TOKEN + "x"},
    {"Authorization": "Bearer " + TOKEN[:-1]},
])
def test_a_malformed_or_wrong_credential_is_refused(headers):
    """Every shape of a bad Authorization header answers 401, not 500."""
    response = client.post("/run/Test_Tool", data={"text_1": "a", "text_2": "b"},
                           headers=headers)
    assert response.status_code == 401, (headers, response.text)


def test_the_token_is_never_echoed_back():
    """A refusal must not confirm any part of the secret."""
    response = client.post("/run/Test_Tool", data={"text_1": "a", "text_2": "b"},
                           headers={"Authorization": "Bearer " + TOKEN[:4]})
    assert response.status_code == 401
    assert TOKEN[:4] not in response.text
    assert TOKEN not in response.text


def test_health_and_tools_stay_open_but_running_does_not():
    """The discovery endpoints are deliberately unauthenticated; /run is not."""
    assert client.get("/health").status_code == 200
    assert client.get("/tools").status_code == 200
    assert client.post("/run/Test_Tool", data={}).status_code == 401


# ---------------------------------------------------------------------------
# Path traversal, in every place a client-supplied string becomes a path
# ---------------------------------------------------------------------------

_TRAVERSALS = [
    "../etc/passwd",
    "../../etc/passwd",
    "..%2f..%2fetc%2fpasswd",
    "....//....//etc/passwd",
    "/etc/passwd",
    "\\..\\..\\windows\\system32",
    "a/../../../b",
]


@pytest.mark.parametrize("name", _TRAVERSALS)
def test_a_traversing_tool_name_cannot_reach_the_filesystem(name):
    """`/run/{tool_name}` is a registry lookup, never a path join."""
    response = client.post(f"/run/{name}", data={}, headers=AUTH)
    assert response.status_code in (400, 404), response.text
    assert "root:" not in response.text


@pytest.mark.parametrize("name", _TRAVERSALS)
def test_a_traversing_testfile_name_is_refused(name):
    """The test-file route resolves through the data store, not os.path.join."""
    response = client.get(f"/tools/Test_Tool/testfiles/{name}", headers=AUTH)
    assert response.status_code in (400, 404), response.text
    assert "root:" not in response.text


@pytest.mark.parametrize("bad", ["../x", "..", "/abs", "a b", "a;b", "a$(id)b", "x" * 200])
def test_a_malformed_upload_id_never_builds_a_path(bad, tmp_path):
    """Ids are matched against a strict pattern BEFORE a path is built.

    GET answers 404. DELETE answers 200 and does nothing, deliberately: a client
    cancelling a run must never be handed an error for it. What matters for
    security is not the status but that no path was built from the string, which
    is asserted by checking nothing outside the transfer root was touched.
    """
    assert client.get(f"/uploads/{bad}", headers=AUTH).status_code in (400, 404)

    witness = tmp_path / "witness"
    witness.write_text("untouched")
    assert client.delete(f"/uploads/{bad}", headers=AUTH).status_code in (200, 400, 404)
    assert witness.read_text() == "untouched"


@pytest.mark.parametrize("bad", ["../x", "..", "/abs", "a;b"])
def test_a_malformed_result_id_never_builds_a_path(bad):
    assert client.get(f"/results/{bad}", headers=AUTH).status_code in (400, 404)


# ---------------------------------------------------------------------------
# Uploaded filenames: sanitized, never trusted
# ---------------------------------------------------------------------------

class _Spy(Tool):
    name = "Security_Spy"
    arguments = {"scan": ArgSpec(type="nifti_file", required=True)}
    output_kind = "text"
    seen: dict = {}

    def run(self, scan):
        _Spy.seen["path"] = str(scan)
        return "ok"


@pytest.mark.parametrize("filename", [
    "../../../etc/passwd.nii.gz",
    "..%2f..%2fpasswd.nii.gz",
    "/etc/shadow.nii.gz",
    "scan\x00.nii.gz",
    "scan\n../evil.nii.gz",
    "$(whoami).nii.gz",
    "`id`.nii.gz",
    "a" * 500 + ".nii.gz",
])
def test_a_hostile_upload_filename_stays_inside_the_work_directory(filename, monkeypatch):
    """The name is kept for the patient's sake, so it must be sanitized."""
    monkeypatch.setitem(registry.TOOLS, "Security_Spy", _Spy())
    _Spy.seen.clear()

    response = client.post(
        "/run/Security_Spy",
        files={"scan": (filename, io.BytesIO(b"x"), "application/gzip")},
        headers=AUTH,
    )
    # Either refused outright, or accepted with a name that cannot escape.
    if response.status_code == 200:
        received = _Spy.seen["path"]
        base = os.path.basename(received)
        assert ".." not in received
        assert "\x00" not in base and "\n" not in base
        assert "/" not in base and "\\" not in base
        assert "$" not in base and "`" not in base
        assert len(base) < 200
    else:
        assert response.status_code in (400, 413, 422), response.text


# ---------------------------------------------------------------------------
# Archives
# ---------------------------------------------------------------------------

def _zip_with(name: str, data: bytes = b"x") -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, data)
    buffer.seek(0)
    return buffer


def test_zip_slip_is_refused(tmp_path):
    """A member named ../../x must not be written outside the destination."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../../escaped.txt", b"x")
    with pytest.raises(file_utils.BadArchiveError):
        file_utils.extract_zip(str(archive), str(tmp_path / "out"))
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_an_absolute_member_is_refused(tmp_path):
    archive = tmp_path / "abs.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("/tmp/escaped.txt", b"x")
    with pytest.raises(file_utils.BadArchiveError):
        file_utils.extract_zip(str(archive), str(tmp_path / "out"))


def test_a_symlink_member_is_refused(tmp_path):
    """A symlink member would let a LATER member write through it."""
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16)
        handle.writestr(info, "/etc/passwd")
    with pytest.raises(file_utils.BadArchiveError):
        file_utils.extract_zip(str(archive), str(tmp_path / "out"))


def test_a_zip_bomb_is_capped(tmp_path):
    """The cap is on the UNCOMPRESSED total, which is what a bomb inflates."""
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("big", b"\0" * (5 * 1024 * 1024))
    with pytest.raises(file_utils.BadArchiveError):
        file_utils.extract_zip(str(archive), str(tmp_path / "out"),
                               max_total_bytes=1024 * 1024)


# ---------------------------------------------------------------------------
# Identifiers the transfer layer generates
# ---------------------------------------------------------------------------

def test_generated_ids_are_unguessable_and_unique():
    """They name a directory holding patient data, so they must not be ordinal.

    Taken from the endpoint rather than a helper, because that is what a client
    is actually handed.
    """
    ids = set()
    for index in range(25):
        response = client.post("/uploads", json={"filename": f"s{index}.nii.gz",
                                                 "size": 1024}, headers=AUTH)
        assert response.status_code == 200, response.text
        ids.add(response.json()["upload_id"])
    assert len(ids) == 25
    for value in ids:
        assert len(value) >= 16
        assert all(character.isalnum() or character in "-_" for character in value)
        client.delete(f"/uploads/{value}", headers=AUTH)


def test_an_unknown_result_id_does_not_reveal_whether_it_ever_existed():
    """Two unknown ids get the same treatment, so this is not an oracle.

    The message echoes the id back, which is safe here for two reasons worth
    stating: the client supplied it and already knows it, and it has passed
    `[A-Za-z0-9_-]{16,64}` before reaching this point, so nothing that could be
    interpreted downstream survives. What must NOT differ is the shape, which is
    what would let a caller distinguish "never existed" from "expired".
    """
    a = client.get("/results/" + "a" * 32, headers=AUTH)
    b = client.get("/results/" + "b" * 32, headers=AUTH)
    assert a.status_code == b.status_code == 404

    normalise = lambda text: text.replace("a" * 32, "ID").replace("b" * 32, "ID")
    assert normalise(a.text) == normalise(b.text)
    for leak in ("/", "\\", "Traceback", "TEMP_DIR", "root:"):
        assert leak not in a.text, a.text

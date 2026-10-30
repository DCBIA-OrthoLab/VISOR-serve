"""A file argument arrives as bytes, never as a path on the server."""

import io
import json
import os

import pytest
from fastapi.testclient import TestClient

import registry
from base import ArgSpec, PATH_TYPE, Tool
from config import settings
from main import app
from registry import conventions, deployment

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}
FORM = {"label": "x", "threshold": "0.5", "iterations": "1"}


def test_a_path_named_in_a_file_argument_is_refused(tmp_path):
    secret = tmp_path / "someone_else.csv"
    secret.write_text("a,b\n1,2\n")
    response = client.post("/run/Example_Tool", headers=AUTH,
                           data={**FORM, "input": str(secret)})
    assert response.status_code == 422
    assert "upload it" in response.json()["detail"]
    assert "1,2" not in response.text


def test_the_forms_empty_placeholder_beside_an_upload_still_works():
    response = client.post("/run/Example_Tool", headers=AUTH,
                           data={**FORM, "input": ""},
                           files={"input": ("input.csv", b"a,b\n1,2\n")})
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# A model is never uploaded
# ---------------------------------------------------------------------------
#
# A deployment opts a model argument out of the hosted-name convention with
# "none" when the tool finds its own weights (deployment.toml does it for 21 of
# them). That leaves the argument a plain `path` -- an uploadable file -- and a
# model loader that unpickles what it is given then runs whatever was uploaded.

class _OptedOutModelTool(Tool):
    """What `server_selectable = { seg_model = "none", t1_reference = "none" }`
    makes of a tool: both arguments plain file paths."""

    name = "Opted_Out_Model_Probe"
    arguments = {
        "scan": ArgSpec(type=PATH_TYPE, required=False),
        "seg_model": ArgSpec(type=PATH_TYPE, required=False),
        "t1_reference": ArgSpec(type=PATH_TYPE, required=False),
    }
    output_kind = "text"
    ran = []

    def run(self, scan=None, seg_model=None, t1_reference=None):
        _OptedOutModelTool.ran.append({"seg_model": seg_model, "t1_reference": t1_reference})
        return "ran"


@pytest.fixture
def opted_out(monkeypatch):
    _OptedOutModelTool.ran = []
    monkeypatch.setitem(registry.TOOLS, _OptedOutModelTool.name, _OptedOutModelTool())
    return f"/run/{_OptedOutModelTool.name}"


def _staged():
    """What the requests have left in TEMP_DIR, beside the long-lived roots."""
    return set(os.listdir(settings.TEMP_DIR)) - {"runs", "uploads", "results"}


def test_an_uploaded_model_is_refused_and_nothing_is_staged(opted_out):
    before = _staged()
    response = client.post(opted_out, headers=AUTH,
                           files={"seg_model": ("weights.pth", io.BytesIO(b"\x80\x04 pickle"))})
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "'seg_model' is a model: the server uses its own weights, they cannot be uploaded.")
    assert _OptedOutModelTool.ran == []
    assert _staged() == before


def test_a_model_sent_through_a_chunked_upload_is_refused_and_the_upload_dropped(opted_out):
    payload = b"\x80\x04 pickle"
    session = client.post("/uploads", headers=AUTH,
                          json={"filename": "weights.pth", "size": len(payload)}).json()
    client.put(f"/uploads/{session['upload_id']}/parts/0", headers=AUTH, content=payload)
    response = client.post(opted_out, headers=AUTH,
                           data={"__uploads__": json.dumps({"seg_model": session["upload_id"]})})
    assert response.status_code == 422
    assert "is a model" in response.json()["detail"]
    assert _OptedOutModelTool.ran == []
    # A refused session does not linger on disk.
    assert client.get(f"/uploads/{session['upload_id']}", headers=AUTH).status_code == 404


def test_a_reference_is_still_the_clinician_s_own_upload(opted_out):
    """`*_reference` is a scan or a mesh -- FlexReg's other timepoint,
    AutoMatrix's output grid -- and is the caller's to send."""
    response = client.post(opted_out, headers=AUTH,
                           files={"t1_reference": ("baseline.nii", io.BytesIO(b"volume"))})
    assert response.status_code == 200, response.text
    assert _OptedOutModelTool.ran and _OptedOutModelTool.ran[0]["t1_reference"]


def test_a_model_that_was_opted_out_is_not_offered_to_a_panel():
    """A panel renders an upload row for every file argument it is shown; the
    server would now refuse what that row sends. Hidden, nobody is asked."""
    arguments = {"seg_model": {"type": "path"}, "t1_reference": {"type": "path"},
                 "scan": {"type": "path"}}
    declared = deployment.ToolDeployment(server_selectable={
        "seg_model": deployment.SERVER_SELECTABLE_NONE,
        "t1_reference": deployment.SERVER_SELECTABLE_NONE,
    })
    merged = conventions.derive(arguments, declared,
                                (settings.BATCH_MAX_MB, settings.BATCH_MAX_FILES))
    assert "seg_model" in merged.hidden
    assert "t1_reference" not in merged.hidden
    assert "scan" not in merged.hidden

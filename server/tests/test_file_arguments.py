"""A file argument arrives as bytes, never as a path on the server."""

from fastapi.testclient import TestClient

from config import settings
from main import app

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

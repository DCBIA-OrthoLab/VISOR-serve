"""scripts/image_release.py without a network: references, the registry's
anonymous token dance, and reading a local image -- each failing quietly."""

import email.message
import importlib.util
import io
import json
import os
import subprocess
import urllib.error

import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")


@pytest.fixture(scope="module")
def release():
    path = os.path.join(_SCRIPTS, "image_release.py")
    if not os.path.isfile(path):
        pytest.skip("scripts/image_release.py is not mounted here")
    spec = importlib.util.spec_from_file_location("image_release_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reference, expected", [
    ("ghcr.io/lab/tools:latest", ("ghcr.io", "lab/tools", "latest")),
    ("ghcr.io/lab/tools", ("ghcr.io", "lab/tools", "latest")),
    ("ghcr.io/lab/tools:tools-2026.10.12-1", ("ghcr.io", "lab/tools", "tools-2026.10.12-1")),
    ("localhost:5000/a/b:c", ("localhost:5000", "a/b", "c")),
    ("python:3.11-slim", ("registry-1.docker.io", "library/python", "3.11-slim")),
    ("ghcr.io/lab/tools@sha256:abc", ("ghcr.io", "lab/tools", "latest")),
])
def test_references_are_read_as_docker_reads_them(release, reference, expected):
    assert release.split_reference(reference) == expected


class _Response:
    def __init__(self, headers=None, body=b""):
        self.headers = headers or {}
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _unauthorized(url, challenge):
    headers = email.message.Message()
    headers["WWW-Authenticate"] = challenge
    return urllib.error.HTTPError(url, 401, "Unauthorized", headers, io.BytesIO())


def test_the_digest_is_read_after_the_anonymous_token(release, monkeypatch):
    seen = []

    def opener(request, timeout=0):
        seen.append(request.full_url)
        if request.get_header("Authorization") != "Bearer tok":
            raise _unauthorized(request.full_url,
                                'Bearer realm="https://ghcr.io/token",service="ghcr.io",'
                                'scope="repository:lab/tools:pull"')
        return _Response({"Docker-Content-Digest": "sha256:feed"})

    monkeypatch.setattr(release.urllib.request, "urlopen",
                        lambda request, timeout=0: _Response(body=json.dumps({"token": "tok"}).encode()))
    assert release.remote_digest("ghcr.io/lab/tools:latest", opener=opener) == "sha256:feed"
    assert seen[0] == "https://ghcr.io/v2/lab/tools/manifests/latest"


@pytest.mark.parametrize("failure", [OSError("offline"), urllib.error.HTTPError("u", 404, "no", {}, io.BytesIO())])
def test_an_unreachable_or_unknown_tag_answers_empty(release, failure):
    def opener(request, timeout=0):
        raise failure
    assert release.remote_digest("ghcr.io/lab/tools:nope", opener=opener) == ""


def test_the_local_digest_is_the_one_of_this_repository(release):
    def run(command, **kwargs):
        out = json.dumps(["other.io/x/y@sha256:nope", "ghcr.io/lab/tools@sha256:mine"]) + "\t" + \
            json.dumps({"org.opencontainers.image.version": "tools-1"})
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr="")
    local = release.local_image("ghcr.io/lab/tools:latest", run=run)
    assert local == {"present": True, "digest": "sha256:mine",
                     "labels": {"org.opencontainers.image.version": "tools-1"}}


def test_an_absent_image_is_not_present(release):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="No such image")
    assert release.local_image("ghcr.io/lab/tools:latest", run=run)["present"] is False


def test_a_github_source_is_recognised(release):
    assert release.github_repository("https://github.com/DCBIA-OrthoLab/SADT-VISOR") == ("DCBIA-OrthoLab", "SADT-VISOR")
    assert release.github_repository("https://example.org/x") is None

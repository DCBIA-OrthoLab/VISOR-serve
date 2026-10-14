"""The server only ever listens over HTTPS, and nothing can make it do otherwise."""

import os
import socket
import ssl
import subprocess
import sys
import time
import urllib.request

import pytest
from fastapi.testclient import TestClient

import main
import tls
from config import settings
from execution import dispatch

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(SERVER_DIR), "scripts"))
import server_ctl  # noqa: E402


@pytest.fixture
def tls_dir(tmp_path, monkeypatch):
    directory = tmp_path / "tls"
    monkeypatch.setattr(settings, "TLS_DIR", str(directory))
    monkeypatch.setattr(settings, "TLS_CERT_FILE", "")
    monkeypatch.setattr(settings, "TLS_KEY_FILE", "")
    return directory


def test_a_certificate_is_generated_once_private_and_kept(tls_dir):
    cert, key = tls.ensure_certificate()
    assert os.stat(key).st_mode & 0o777 == 0o600
    assert os.stat(tls_dir).st_mode & 0o777 == 0o700
    with open(cert, "rb") as handle:
        first = handle.read()
    assert tls.ensure_certificate() == (cert, key)
    with open(cert, "rb") as handle:
        assert handle.read() == first, "a pinned client must keep trusting it"


def test_half_an_operator_certificate_is_refused(tls_dir, monkeypatch):
    monkeypatch.setattr(settings, "TLS_CERT_FILE", "/somewhere/cert.pem")
    with pytest.raises(tls.TLSConfigurationError, match="both"):
        tls.ensure_certificate()


def test_an_unusable_certificate_stops_the_start(tls_dir, monkeypatch, tmp_path):
    cert, key = tls.ensure_certificate()
    other = tmp_path / "other"
    monkeypatch.setattr(settings, "TLS_DIR", str(other))
    _other_cert, other_key = tls.ensure_certificate()
    monkeypatch.setattr(settings, "TLS_CERT_FILE", cert)
    monkeypatch.setattr(settings, "TLS_KEY_FILE", other_key)
    with pytest.raises(tls.TLSConfigurationError, match="cannot be used"):
        tls.ensure_certificate()


def test_the_app_refuses_to_start_unless_serve_py_gave_it_tls(monkeypatch):
    monkeypatch.delenv(tls.ACTIVE_ENV, raising=False)
    with pytest.raises(tls.TLSConfigurationError, match="HTTPS"):
        with TestClient(main.app):
            pass


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_serve_py_answers_https_and_never_plain_http(tls_dir, tmp_path):
    port = _free_port()
    environment = dict(os.environ, TLS_DIR=str(tls_dir), TEMP_DIR=str(tmp_path / "temp"),
                       HISTORY_DIR="", API_TOKEN="test-token")
    environment.pop(tls.ACTIVE_ENV, None)
    process = subprocess.Popen([sys.executable, "serve.py", "--port", str(port)],
                               cwd=SERVER_DIR, env=environment,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"https://localhost:{port}/health"
        deadline = time.monotonic() + 60
        while not server_ctl.health(f"https://localhost:{port}", timeout=2):
            assert process.poll() is None, "serve.py exited"
            assert time.monotonic() < deadline, "serve.py never answered over HTTPS"
            time.sleep(0.3)
        # Verified against the certificate on disk, host name included.
        context = ssl.create_default_context(cafile=str(tls_dir / tls.CERT_NAME))
        with urllib.request.urlopen(url, context=context, timeout=5) as response:
            assert response.status == 200
        # Plain HTTP gets no HTTP answer at all: the bytes hit a TLS handshake.
        with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
            raw.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            raw.settimeout(5)
            try:
                reply = raw.recv(64)
            except (ConnectionResetError, socket.timeout):
                reply = b""
        assert not reply.startswith(b"HTTP/"), reply
    finally:
        process.terminate()
        process.wait(timeout=30)


def test_server_ctl_refuses_a_plain_http_url():
    with pytest.raises(ValueError, match="HTTPS"):
        server_ctl.tls_context("http://localhost:8000")
    assert server_ctl.health("http://localhost:8000") is False


def test_a_remote_server_is_verified_against_the_public_authorities():
    context = server_ctl.tls_context("https://visor.example.ts.net")
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED


def test_a_tool_inherits_none_of_the_servers_secrets(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "admin")
    monkeypatch.setenv("TLS_KEY_FILE", "/key.pem")
    environment = dispatch.child_environment_base()
    assert not set(dispatch.SERVER_SECRETS) & set(environment)

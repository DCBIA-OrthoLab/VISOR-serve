"""Benchmarks answer to the ADMIN token, never to the API token alone.

A campaign is the developer's and the operator's reading of how far this
machine can be pushed, and a battery spends the machine on purpose. Both belong
with the admin panel; the API token every workstation holds opens none of it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main
from config import settings

client = TestClient(main.app)
AUTH = {"Authorization": f"Bearer {settings.API_TOKEN}"}

# Every benchmark route that reads or acts, with a body where one is needed.
ROUTES = [
    ("get", "/benchmarks", None),
    ("get", "/benchmark/presets", None),
    ("get", "/benchmark/status", None),
    ("post", "/benchmark/run", {"preset": "no-such-preset"}),
    ("delete", "/benchmark/run", None),
]


def _call(method, path, body, headers):
    if body is None:
        return getattr(client, method)(path, headers=headers)
    return getattr(client, method)(path, headers=headers, json=body)


@pytest.fixture
def admin(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "operator-secret")
    return {"X-Admin-Token": "operator-secret"}


@pytest.mark.parametrize("method,path,body", ROUTES)
def test_the_api_token_alone_opens_no_benchmark_route(admin, method, path, body):
    assert _call(method, path, body, AUTH).status_code == 401


@pytest.mark.parametrize("method,path,body", ROUTES)
def test_a_wrong_admin_token_is_refused(admin, method, path, body):
    assert _call(method, path, body, {**AUTH, "X-Admin-Token": "guess"}).status_code == 401


@pytest.mark.parametrize("method,path,body", ROUTES)
def test_without_an_admin_token_configured_benchmarks_stay_shut(monkeypatch, method, path, body):
    monkeypatch.setattr(settings, "ADMIN_TOKEN", "")
    assert _call(method, path, body, {**AUTH, "X-Admin-Token": ""}).status_code == 403


def test_the_admin_token_alone_opens_them(admin):
    assert client.get("/benchmark/status", headers=admin).status_code == 200
    assert client.get("/benchmark/presets", headers=admin).status_code == 200
    # Past the gate: an unknown preset is refused for what it is, before
    # anything is spawned.
    assert client.post("/benchmark/run", headers=admin, json={"preset": "no-such-preset"}).status_code == 422


def test_the_pages_send_the_admin_token_under_the_panels_key():
    """One unlock covers the panel and both benchmark pages."""
    for path in ("/benchmark", "/benchmarks/view"):
        body = client.get(path).text
        assert '"visor.admin"' in body, path
        assert "X-Admin-Token" in body, path
        assert "Bearer" not in body, path


def test_the_status_page_no_longer_asks_for_a_campaign():
    assert 'fetch("benchmarks"' not in client.get("/").text

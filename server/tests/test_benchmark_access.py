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


# --- where a battery writes, on a deployment that mounts nothing for it -------

def test_a_battery_writes_beside_the_cache_when_the_campaign_folder_is_not_writable(admin, monkeypatch, tmp_path):
    """The released compose mounts nothing at /benchmarks, and the server runs
    as a user who cannot create it: every battery failed to start with
    `Permission denied: '/benchmarks'`. It now lands under SCHEMA_CACHE_DIR,
    which the image gives the server and a volume keeps."""
    import benchmark_jobs
    monkeypatch.setattr(settings, "SADT_BENCHMARK_DIR", str(tmp_path / "not-mounted" / "benchmarks"))
    monkeypatch.setattr(settings, "SADT_BATTERY_DIR", "")
    monkeypatch.setattr(settings, "SCHEMA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(main, "_writable", lambda directory: False)
    seen = {}
    monkeypatch.setattr(benchmark_jobs, "start",
                        lambda plan, summary_dir, *rest: seen.setdefault("dir", summary_dir) and {"state": "running"})
    response = client.post("/benchmark/run", headers=admin, json={"preset": "smoke"})
    assert response.status_code == 200, response.text
    assert seen["dir"] == str(tmp_path / "cache" / "batteries")


def test_the_results_read_campaigns_and_batteries_together(admin, monkeypatch, tmp_path):
    campaigns, batteries = tmp_path / "campaigns", tmp_path / "cache" / "batteries"
    campaigns.mkdir()
    batteries.mkdir(parents=True)
    (campaigns / "b9-20261001T000000Z.json").write_text('{"generated_at": 1, "arms": []}')
    (batteries / "preset-custom-20261006T000000Z.json").write_text('{"generated_at": 2, "arms": [{}]}')
    monkeypatch.setattr(settings, "SADT_BENCHMARK_DIR", str(campaigns))
    monkeypatch.setattr(settings, "SADT_BATTERY_DIR", str(batteries))
    body = client.get("/benchmarks", headers=admin).json()
    assert [c["source"] for c in body["campaigns"]] == [
        "preset-custom-20261006T000000Z.json", "b9-20261001T000000Z.json"]
    assert body["campaign"]["source"] == "preset-custom-20261006T000000Z.json"
    older = client.get("/benchmarks?campaign=b9-20261001T000000Z.json", headers=admin)
    assert older.status_code == 200 and older.json()["campaign"]["generated_at"] == 1

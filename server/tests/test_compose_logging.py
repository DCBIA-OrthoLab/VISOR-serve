"""Every service that runs the server keeps a bounded log.

Docker's default `json-file` driver keeps everything a container ever wrote,
for as long as the container exists: a server that runs for months with a
uvicorn access line per poll fills the disk TEMP_DIR shares, and a run that
cannot write its upload fails for a reason nobody looks for in a log.

Skipped where docker-compose.yml is not beside this checkout: the docker
`test` service mounts server/ alone.
"""

import os

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPOSE = os.path.join(REPO_ROOT, "docker-compose.yml")


@pytest.mark.skipif(not os.path.isfile(COMPOSE), reason="docker-compose.yml is not beside server/")
def test_every_server_service_bounds_its_log():
    with open(COMPOSE, encoding="utf-8") as handle:
        services = yaml.safe_load(handle)["services"]
    servers = {
        name: service for name, service in services.items()
        # The test services run pytest, not a server, and are gone when it ends.
        if "serve.py" in str(service.get("command")) and "test" not in (service.get("profiles") or ())
    }
    assert {"inference", "inference-cpu", "inference-venvs"} <= set(servers)
    for name, service in servers.items():
        options = (service.get("logging") or {}).get("options") or {}
        assert options.get("max-size"), f"{name} keeps an unbounded log"
        assert options.get("max-file"), f"{name} keeps an unbounded log"

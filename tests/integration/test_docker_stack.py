"""Opt-in integration test against the real docker-compose stack.

Skipped by default -- `pytest` / `python -m pytest` (the main CI/unit test
command) never touches Docker, a network, or a cluster. Set
DRIFTGUARD_RUN_INTEGRATION=1 and have `docker compose up -d` already
running (see docker-compose.yml) before opting in, e.g.:

    docker compose up -d
    DRIFTGUARD_RUN_INTEGRATION=1 pytest tests/integration -m integration -q
"""
import os

import httpx
import pytest

pytestmark = pytest.mark.integration

RUN_INTEGRATION = os.getenv("DRIFTGUARD_RUN_INTEGRATION") == "1"
skip_reason = (
    "Integration tests are opt-in: set DRIFTGUARD_RUN_INTEGRATION=1 and run "
    "`docker compose up -d` first. Never required for the unit test suite."
)


@pytest.mark.skipif(not RUN_INTEGRATION, reason=skip_reason)
def test_app_healthz_reachable_through_docker_compose():
    resp = httpx.get("http://localhost:8000/healthz", timeout=5.0)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.skipif(not RUN_INTEGRATION, reason=skip_reason)
def test_prometheus_is_scraping_the_app():
    resp = httpx.get(
        "http://localhost:9090/api/v1/query",
        params={"query": "up{job=\"driftguard-llm\"}"},
        timeout=5.0,
    )
    assert resp.status_code == 200
    result = resp.json()["data"]["result"]
    assert result, "Prometheus has no 'up' series for the driftguard-llm job yet"
    assert result[0]["value"][1] == "1"


@pytest.mark.skipif(not RUN_INTEGRATION, reason=skip_reason)
def test_grafana_dashboard_is_provisioned():
    resp = httpx.get(
        "http://localhost:3000/api/dashboards/uid/driftguard-llm",
        timeout=5.0,
    )
    assert resp.status_code == 200

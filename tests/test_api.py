import pytest
from fastapi.testclient import TestClient

from app.api import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_endpoint(client):
    resp = client.post(
        "/ingest",
        json={"text": "driftguard-llm ingests documents and chunks them for retrieval.", "source": "test"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["num_chunks"] >= 1
    assert body["doc_id"]


def test_ingest_rejects_blank_text(client):
    resp = client.post("/ingest", json={"text": "   ", "source": "test"})
    assert resp.status_code == 422


def test_query_endpoint_returns_answer(client):
    client.post(
        "/ingest",
        json={
            "doc_id": "doc1",
            "text": "driftguard-llm detects embedding drift and scores answer quality using an LLM judge.",
            "source": "test",
        },
    )
    resp = client.post("/query", json={"query": "What does driftguard-llm detect?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["query_id"]
    assert isinstance(body["contexts"], list)


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/query", json={"query": "hello"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "driftguard_queries_total" in resp.text


def test_admin_drift_endpoint(client):
    resp = client.get("/admin/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "warn", "critical")


def test_admin_eval_endpoint(client):
    client.post(
        "/ingest",
        json={"doc_id": "doc1", "text": "Some content about driftguard evaluation harness.", "source": "test"},
    )
    client.post("/query", json={"query": "evaluation harness"})
    resp = client.get("/admin/eval")
    assert resp.status_code == 200
    body = resp.json()
    assert "num_samples" in body


def test_admin_alerts_endpoint(client):
    resp = client.get("/admin/alerts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] in ("healthy", "degraded", "firing")


def test_admin_reembed_endpoint_with_no_pending_docs(client):
    resp = client.post("/admin/reembed")
    assert resp.status_code == 200
    assert resp.json() == {"reembedded_doc_ids": []}


def test_full_flow_ingest_query_metrics(client):
    ingest_resp = client.post(
        "/ingest",
        json={"text": "The capital of France is Paris. It is known for the Eiffel Tower.", "source": "geo"},
    )
    assert ingest_resp.status_code == 200

    query_resp = client.post("/query", json={"query": "What is the capital of France?"})
    assert query_resp.status_code == 200
    assert "paris" in query_resp.json()["answer"].lower()

    metrics_resp = client.get("/metrics")
    assert "driftguard_query_latency_seconds" in metrics_resp.text

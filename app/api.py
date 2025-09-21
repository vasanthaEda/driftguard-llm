"""FastAPI application exposing ingestion, query, metrics, and admin
endpoints for driftguard-llm.

Use `create_app()` to get a fresh, isolated app instance (this is what the
test suite does, so tests never share vector-store/monitoring state). A
module-level `app` is also provided for `uvicorn app.api:app` / Docker.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app import metrics
from app.monitoring import MonitoringService
from app.rag import RAGService


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "manual-upload"
    doc_id: str | None = None


class IngestResponse(BaseModel):
    doc_id: str
    num_chunks: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = None


class QueryResponseModel(BaseModel):
    query_id: str
    answer: str
    contexts: list[str]
    sources: list[str]
    scores: list[float]
    latency_seconds: float


class DriftResponse(BaseModel):
    status: str
    centroid_cosine_similarity: float
    mean_psi: float
    max_psi: float
    ks_reject_fraction: float
    reference_size: int
    current_size: int
    reasons: list[str]


class EvalSummaryResponse(BaseModel):
    num_samples: int
    mean_faithfulness: float | None
    mean_relevance: float | None


class AlertResponse(BaseModel):
    state: str
    consecutive_breaches: int
    last_event_reasons: list[str] | None
    pending_reembed_doc_ids: list[str]


def create_app() -> FastAPI:
    rag = RAGService()
    monitor = MonitoringService(rag)

    app = FastAPI(
        title="driftguard-llm",
        description="RAG service instrumented with embedding drift + LLM-as-judge quality monitoring",
        version="0.1.0",
    )
    app.state.rag = rag
    app.state.monitor = monitor

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> IngestResponse:
        try:
            result = app.state.rag.ingest_document(req.doc_id, req.source, req.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestResponse(**result)

    @app.post("/query", response_model=QueryResponseModel)
    def query(req: QueryRequest) -> QueryResponseModel:
        with metrics.QUERY_LATENCY.time():
            result = app.state.rag.query(req.query, top_k=req.top_k)
        metrics.QUERY_COUNTER.inc()
        app.state.monitor.tick()
        return QueryResponseModel(
            query_id=result.query_id,
            answer=result.answer,
            contexts=result.contexts,
            sources=result.sources,
            scores=result.scores,
            latency_seconds=result.latency_seconds,
        )

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/admin/drift", response_model=DriftResponse)
    def admin_drift() -> DriftResponse:
        report = app.state.monitor.evaluate_drift()
        return DriftResponse(**report.as_dict())

    @app.get("/admin/eval", response_model=EvalSummaryResponse)
    def admin_eval() -> EvalSummaryResponse:
        sampler = app.state.monitor.eval_sampler
        mean_f, mean_r = sampler.rolling_means()
        return EvalSummaryResponse(
            num_samples=len(sampler), mean_faithfulness=mean_f, mean_relevance=mean_r
        )

    @app.get("/admin/alerts", response_model=AlertResponse)
    def admin_alerts() -> AlertResponse:
        state = app.state.monitor.tick()
        return AlertResponse(
            state=state.state.value,
            consecutive_breaches=state.consecutive_breaches,
            last_event_reasons=state.last_event.reasons if state.last_event else None,
            pending_reembed_doc_ids=list(app.state.monitor.reembed_registry.pending_doc_ids),
        )

    @app.post("/admin/reembed")
    def admin_reembed() -> dict:
        pending = app.state.monitor.reembed_registry.pop_pending()
        reembedded: list[str] = []
        for doc_id in pending:
            docs = [d for d in app.state.rag.vector_store.list_documents() if d.doc_id == doc_id]
            for doc in docs:
                rows = app.state.rag.vector_store._conn.execute(
                    "SELECT text FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
                ).fetchall()
                full_text = " ".join(r[0] for r in rows)
                if full_text.strip():
                    app.state.rag.ingest_document(doc_id, doc.source, full_text)
                    reembedded.append(doc_id)
        return {"reembedded_doc_ids": reembedded}

    return app


app = create_app()

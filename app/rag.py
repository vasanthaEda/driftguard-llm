"""Core RAG service: ingestion + retrieve-and-generate, instrumented for
drift and quality monitoring.

Every query embedding is pushed into the drift monitor's rolling windows,
and every (query, answer, contexts) triple is appended to the query log
that the eval sampler draws from -- this is the instrumentation the whole
"catch silent degradation" story depends on, wired directly into the hot
path rather than bolted on after the fact.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from app.chunking import chunk_text
from app.config import settings
from app.embeddings import Embedder, build_embedder
from app.llm import LLMGenerator, build_llm
from app.vector_store import SearchResult, VectorStore


@dataclass
class QueryLogEntry:
    query_id: str
    query: str
    answer: str
    contexts: list[str]
    chunk_ids: list[str]
    timestamp: float
    embedding: "object"  # np.ndarray, kept loosely typed to avoid import cost here


@dataclass
class QueryResponse:
    query_id: str
    answer: str
    contexts: list[str]
    sources: list[str]
    scores: list[float]
    latency_seconds: float


class RAGService:
    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: LLMGenerator | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.embedder = embedder or build_embedder()
        self.llm = llm or build_llm()
        self.vector_store = vector_store or VectorStore(
            db_path=settings.database_path, dim=self.embedder.dim
        )
        self.query_log: list[QueryLogEntry] = []

    def ingest_document(self, doc_id: str | None, source: str, text: str) -> dict:
        doc_id = doc_id or str(uuid.uuid4())
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("document text produced zero chunks after cleaning/splitting")

        self.vector_store.remove_document_chunks(doc_id)
        self.vector_store.upsert_document(doc_id, source=source)

        chunk_ids = [f"{doc_id}::{c.index}" for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.vector_store.add_chunks(doc_id, chunk_ids, texts, embeddings)

        return {"doc_id": doc_id, "num_chunks": len(chunks)}

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[list[SearchResult], "object"]:
        top_k = top_k or settings.top_k
        query_emb = self.embedder.embed([query])[0]
        results = self.vector_store.search(query_emb, top_k=top_k)
        return results, query_emb

    def query(self, query: str, top_k: int | None = None) -> QueryResponse:
        start = time.perf_counter()
        results, query_emb = self.retrieve(query, top_k=top_k)
        contexts = [r.text for r in results]
        answer = self.llm.generate(query, contexts)
        latency = time.perf_counter() - start

        query_id = str(uuid.uuid4())
        self.query_log.append(
            QueryLogEntry(
                query_id=query_id,
                query=query,
                answer=answer,
                contexts=contexts,
                chunk_ids=[r.chunk_id for r in results],
                timestamp=time.time(),
                embedding=query_emb,
            )
        )

        return QueryResponse(
            query_id=query_id,
            answer=answer,
            contexts=contexts,
            sources=[r.source for r in results],
            scores=[r.score for r in results],
            latency_seconds=latency,
        )

    def recent_log_entries(self, n: int) -> list[QueryLogEntry]:
        return self.query_log[-n:]

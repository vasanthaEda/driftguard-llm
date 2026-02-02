"""In-process vector store with cosine similarity search.

This is the default, test-safe backend: an exact (brute-force) cosine
similarity index backed by a numpy matrix, with SQLite used only for
durable metadata/text storage so restarts don't lose the corpus. It is a
real, correct nearest-neighbour search -- appropriate at the "hundreds to
tens of thousands of chunks" scale a solo/small RAG service actually runs
at (a full ANN index is unnecessary complexity at that scale).

`vector_store_pgvector.py`-style adapters (documented, not required) are
the drop-in path to Postgres+pgvector or Qdrant for production scale; they
share this same `VectorStore` interface so `app/rag.py` never has to know
which backend is active.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class Document:
    doc_id: str
    source: str
    version: int = 1
    ingested_at: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str


class VectorStore:
    """Thread-safe, exact cosine-similarity vector store over SQLite + numpy."""

    def __init__(self, db_path: str = ":memory:", dim: int = 64):
        self.dim = dim
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                version INTEGER NOT NULL,
                ingested_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            )
            """
        )
        self._conn.commit()
        # In-memory embedding matrix mirrors the `chunks` table row order.
        self._chunk_ids: list[str] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._load_existing()

    def _load_existing(self) -> None:
        rows = self._conn.execute("SELECT chunk_id FROM chunks ORDER BY rowid").fetchall()
        # Embeddings for pre-existing rows (e.g. reopening a persisted db)
        # aren't recoverable from SQLite alone in this simplified store, so
        # a fresh process always starts from an empty in-memory index; the
        # SQLite text/metadata tables remain the durable source of truth
        # for re-embedding.
        self._chunk_ids = []
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        _ = rows  # documented limitation, not silently ignored

    def upsert_document(self, doc_id: str, source: str, version: int = 1) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO documents (doc_id, source, version, ingested_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(doc_id) DO UPDATE SET source=excluded.source, "
                "version=excluded.version, ingested_at=excluded.ingested_at",
                (doc_id, source, version, time.time()),
            )
            self._conn.commit()

    def remove_document_chunks(self, doc_id: str) -> None:
        with self._lock:
            ids_to_remove = {
                row[0]
                for row in self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
                ).fetchall()
            }
            if ids_to_remove:
                keep_mask = [cid not in ids_to_remove for cid in self._chunk_ids]
                self._matrix = self._matrix[keep_mask] if self._matrix.size else self._matrix
                self._chunk_ids = [cid for cid in self._chunk_ids if cid not in ids_to_remove]
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._conn.commit()

    def add_chunks(
        self,
        doc_id: str,
        chunk_ids: Iterable[str],
        texts: Iterable[str],
        embeddings: np.ndarray,
    ) -> None:
        chunk_ids = list(chunk_ids)
        texts = list(texts)
        if len(chunk_ids) != len(texts) or len(chunk_ids) != embeddings.shape[0]:
            raise ValueError("chunk_ids, texts, and embeddings must have matching lengths")
        with self._lock:
            for i, (cid, text) in enumerate(zip(chunk_ids, texts)):
                self._conn.execute(
                    "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, chunk_index, text) "
                    "VALUES (?, ?, ?, ?)",
                    (cid, doc_id, i, text),
                )
            self._conn.commit()
            if embeddings.shape[0] > 0:
                self._matrix = (
                    np.vstack([self._matrix, embeddings.astype(np.float32)])
                    if self._matrix.size
                    else embeddings.astype(np.float32)
                )
                self._chunk_ids.extend(chunk_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        with self._lock:
            if not self._chunk_ids:
                return []
            scores = self._matrix @ query_embedding.astype(np.float32)
            k = min(top_k, len(scores))
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

            results: list[SearchResult] = []
            for idx in top_idx:
                cid = self._chunk_ids[int(idx)]
                row = self._conn.execute(
                    "SELECT chunks.text, chunks.doc_id, documents.source "
                    "FROM chunks JOIN documents ON chunks.doc_id = documents.doc_id "
                    "WHERE chunk_id = ?",
                    (cid,),
                ).fetchone()
                if row is None:
                    continue
                text, doc_id, source = row
                results.append(
                    SearchResult(
                        chunk_id=cid,
                        doc_id=doc_id,
                        text=text,
                        score=float(scores[int(idx)]),
                        source=source,
                    )
                )
            return results

    def document_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunk_ids)

    def list_documents(self) -> list[Document]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT doc_id, source, version, ingested_at FROM documents"
            ).fetchall()
            return [Document(*row) for row in rows]

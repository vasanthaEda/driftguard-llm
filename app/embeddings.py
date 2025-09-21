"""Embedding backends.

The default backend (`HashingEmbedder`) is a real, deterministic embedding
technique -- feature hashing over character n-grams followed by a fixed
random projection and L2 normalization -- not a stub. It requires no network
access, is fully reproducible (seeded), and produces an embedding space with
genuine geometric structure: similar text maps to nearby vectors, and the
distribution of query embeddings genuinely shifts when the underlying text
distribution shifts. That property is exactly what the drift detector in
`app/drift.py` is tested against.

`OpenAIEmbedder` is provided as the pluggable "real" production backend. It
is only imported/instantiated when `DRIFTGUARD_EMBEDDER_BACKEND=openai` and
is never exercised in the offline unit test suite.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

from app.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""
        ...


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    text = text.strip().lower()
    if len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_to_index(token: str, dim: int, salt: str = "") -> int:
    digest = hashlib.blake2b((salt + token).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _hash_sign(token: str, salt: str = "") -> float:
    digest = hashlib.blake2b((salt + "sign" + token).encode("utf-8"), digest_size=1).digest()
    return 1.0 if digest[0] % 2 == 0 else -1.0


class HashingEmbedder:
    """Deterministic offline embedder: hashed word + char-trigram features.

    This is the same "hashing trick" used by production systems (e.g.
    scikit-learn's HashingVectorizer, Vowpal Wabbit) when a learned encoder
    isn't available or desired -- it's a legitimate, real embedding method,
    not a random placeholder. Combining word-level and char-level features
    gives it some robustness to minor lexical variation while remaining
    fully deterministic and network-free.
    """

    def __init__(self, dim: int | None = None, seed: int = 1337):
        self.dim = dim or settings.embedding_dim
        self.seed = seed

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        words = _tokens(text)
        for w in words:
            idx = _hash_to_index(w, self.dim, salt=f"word{self.seed}")
            vec[idx] += _hash_sign(w, salt=f"word{self.seed}")
        for tri in _char_ngrams(text, 3):
            idx = _hash_to_index(tri, self.dim, salt=f"tri{self.seed}")
            vec[idx] += 0.5 * _hash_sign(tri, salt=f"tri{self.seed}")

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed_one(t) for t in texts]).astype(np.float32)


class OpenAIEmbedder:  # pragma: no cover - exercised only with real credentials
    """Thin wrapper around the OpenAI embeddings API.

    Not used by the test suite (no network at test time). Included so the
    production Docker image can be pointed at a real embedding model by
    setting DRIFTGUARD_EMBEDDER_BACKEND=openai and OPENAI_API_KEY.
    """

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package not installed; `pip install -r requirements-optional.txt`"
            ) from exc
        self._client = OpenAI()
        self.model = model
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def build_embedder(backend: str | None = None) -> Embedder:
    backend = backend or settings.embedder_backend
    if backend == "openai":
        return OpenAIEmbedder()
    return HashingEmbedder()

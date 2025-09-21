import numpy as np
import pytest

from app.embeddings import HashingEmbedder
from app.llm import ExtractiveLLM
from app.rag import RAGService
from app.vector_store import VectorStore


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64, seed=1337)


@pytest.fixture
def rag_service(embedder) -> RAGService:
    store = VectorStore(db_path=":memory:", dim=embedder.dim)
    return RAGService(embedder=embedder, llm=ExtractiveLLM(), vector_store=store)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)

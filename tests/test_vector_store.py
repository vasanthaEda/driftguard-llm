import numpy as np
import pytest

from app.vector_store import VectorStore


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(db_path=":memory:", dim=4)


def _unit(vec):
    v = np.array(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_add_and_search_returns_best_match(store):
    store.upsert_document("doc1", source="unit-test")
    embeddings = np.stack([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])])
    store.add_chunks("doc1", ["doc1::0", "doc1::1"], ["chunk about cats", "chunk about dogs"], embeddings)

    results = store.search(_unit([1, 0, 0, 0]), top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "doc1::0"
    assert results[0].text == "chunk about cats"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_on_empty_store_returns_empty_list(store):
    assert store.search(_unit([1, 0, 0, 0]), top_k=3) == []


def test_document_and_chunk_counts(store):
    store.upsert_document("doc1", source="s")
    store.add_chunks("doc1", ["doc1::0"], ["hello"], _unit([1, 0, 0, 0])[None, :])
    store.upsert_document("doc2", source="s")
    store.add_chunks("doc2", ["doc2::0", "doc2::1"], ["a", "b"], np.stack([_unit([0, 1, 0, 0]), _unit([0, 0, 1, 0])]))

    assert store.document_count() == 2
    assert store.chunk_count() == 3


def test_remove_document_chunks_removes_from_index_and_db(store):
    store.upsert_document("doc1", source="s")
    store.add_chunks("doc1", ["doc1::0"], ["hello"], _unit([1, 0, 0, 0])[None, :])
    store.upsert_document("doc2", source="s")
    store.add_chunks("doc2", ["doc2::0"], ["world"], _unit([0, 1, 0, 0])[None, :])

    store.remove_document_chunks("doc1")
    assert store.chunk_count() == 1
    results = store.search(_unit([1, 0, 0, 0]), top_k=5)
    assert all(r.doc_id != "doc1" for r in results)


def test_reingesting_a_document_replaces_its_chunks(store):
    store.upsert_document("doc1", source="s")
    store.add_chunks("doc1", ["doc1::0"], ["old content"], _unit([1, 0, 0, 0])[None, :])
    assert store.chunk_count() == 1

    store.remove_document_chunks("doc1")
    store.add_chunks("doc1", ["doc1::0", "doc1::1"], ["new content a", "new content b"],
                      np.stack([_unit([0, 1, 0, 0]), _unit([0, 0, 1, 0])]))
    assert store.chunk_count() == 2
    texts = {r.text for r in store.search(_unit([0, 1, 0, 0]), top_k=5)}
    assert "new content a" in texts
    assert "old content" not in texts


def test_mismatched_lengths_raise_value_error(store):
    with pytest.raises(ValueError):
        store.add_chunks("doc1", ["a", "b"], ["only one"], _unit([1, 0, 0, 0])[None, :])

import numpy as np

from app.embeddings import HashingEmbedder


def test_embeddings_are_deterministic():
    emb = HashingEmbedder(dim=32, seed=42)
    v1 = emb.embed(["the quick brown fox"])
    v2 = emb.embed(["the quick brown fox"])
    np.testing.assert_array_equal(v1, v2)


def test_embeddings_are_unit_normalized():
    emb = HashingEmbedder(dim=32, seed=42)
    vecs = emb.embed(["hello world", "completely different sentence about finance"])
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_similar_text_is_closer_than_dissimilar_text():
    emb = HashingEmbedder(dim=128, seed=42)
    base = "python programming language tutorial for beginners"
    similar = "python programming language guide for beginners"
    dissimilar = "the migratory patterns of arctic seabirds in winter"

    v_base, v_similar, v_dissimilar = emb.embed([base, similar, dissimilar])
    sim_close = float(np.dot(v_base, v_similar))
    sim_far = float(np.dot(v_base, v_dissimilar))
    assert sim_close > sim_far


def test_empty_input_returns_empty_array():
    emb = HashingEmbedder(dim=16)
    out = emb.embed([])
    assert out.shape == (0, 16)


def test_different_seeds_produce_different_spaces():
    text = "driftguard monitors embedding drift"
    e1 = HashingEmbedder(dim=32, seed=1)
    e2 = HashingEmbedder(dim=32, seed=2)
    v1 = e1.embed([text])[0]
    v2 = e2.embed([text])[0]
    assert not np.allclose(v1, v2)

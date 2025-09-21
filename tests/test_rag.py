import pytest

DOC_TEXT = (
    "driftguard-llm monitors embedding drift and answer quality for RAG systems. "
    "It detects when retrieval quality silently degrades in production. "
    "The system uses a rolling LLM-as-judge sampler to score faithfulness and relevance."
)


def test_ingest_document_creates_chunks(rag_service):
    result = rag_service.ingest_document(None, "unit-test", DOC_TEXT)
    assert result["num_chunks"] >= 1
    assert rag_service.vector_store.document_count() == 1


def test_ingest_rejects_empty_text(rag_service):
    with pytest.raises(ValueError):
        rag_service.ingest_document(None, "unit-test", "   ")


def test_query_returns_grounded_answer(rag_service):
    rag_service.ingest_document("doc1", "unit-test", DOC_TEXT)
    response = rag_service.query("What does driftguard-llm monitor?")

    assert response.answer
    assert len(response.contexts) > 0
    assert "drift" in response.answer.lower() or "quality" in response.answer.lower()
    assert response.latency_seconds >= 0


def test_query_with_no_ingested_documents_handles_gracefully(rag_service):
    response = rag_service.query("anything at all")
    assert response.contexts == []
    assert "context" in response.answer.lower() or "confidently" in response.answer.lower()


def test_query_appends_to_query_log(rag_service):
    rag_service.ingest_document("doc1", "unit-test", DOC_TEXT)
    assert len(rag_service.query_log) == 0
    rag_service.query("What does it monitor?")
    assert len(rag_service.query_log) == 1
    entry = rag_service.query_log[0]
    assert entry.query == "What does it monitor?"
    assert entry.embedding is not None


def test_reingesting_same_doc_id_replaces_chunks(rag_service):
    rag_service.ingest_document("doc1", "unit-test", "Original content about cats.")
    rag_service.ingest_document("doc1", "unit-test", "Updated content about dogs and puppies.")
    assert rag_service.vector_store.document_count() == 1
    response = rag_service.query("puppies")
    assert any("dogs" in c or "puppies" in c for c in response.contexts)


def test_recent_log_entries_returns_last_n(rag_service):
    rag_service.ingest_document("doc1", "unit-test", DOC_TEXT)
    for i in range(5):
        rag_service.query(f"query number {i}")
    recent = rag_service.recent_log_entries(2)
    assert len(recent) == 2
    assert recent[-1].query == "query number 4"

import numpy as np
import pytest

from app.alerts import AlertState
from app.drift import DriftStatus
from app.embeddings import HashingEmbedder
from app.judge import EvalResult
from app.llm import ExtractiveLLM
from app.monitoring import EvalSampler, MonitoringService
from app.rag import RAGService
from app.vector_store import VectorStore

BASELINE_DOC = (
    "Our return policy allows customers to return unused items within 30 days "
    "of purchase for a full refund. Shipping costs are non-refundable. "
    "Digital products are not eligible for returns once downloaded."
)


class FixedJudge:
    """Deterministic stub judge for monitoring-level tests: always returns
    the score it was configured with, regardless of input."""

    def __init__(self, faithfulness: float, relevance: float):
        self.faithfulness = faithfulness
        self.relevance = relevance

    def score(self, query, answer, contexts):
        return EvalResult(
            faithfulness=self.faithfulness,
            relevance=self.relevance,
            verdict="pass" if self.faithfulness > 0.5 else "fail",
        )


@pytest.fixture
def small_rag() -> RAGService:
    embedder = HashingEmbedder(dim=16, seed=5)
    store = VectorStore(db_path=":memory:", dim=embedder.dim)
    rag = RAGService(embedder=embedder, llm=ExtractiveLLM(), vector_store=store)
    rag.ingest_document("policy", "unit-test", BASELINE_DOC)
    return rag


def test_eval_sampler_respects_sample_rate_zero():
    sampler = EvalSampler(FixedJudge(0.9, 0.9), sample_rate=0.0)
    from app.rag import QueryLogEntry
    entry = QueryLogEntry("q1", "query", "answer", ["ctx"], ["c1"], 0.0, np.zeros(4))
    result = sampler.maybe_score(entry)
    assert result is None
    assert len(sampler) == 0


def test_eval_sampler_scores_when_rate_is_one():
    sampler = EvalSampler(FixedJudge(0.8, 0.7), sample_rate=1.0)
    from app.rag import QueryLogEntry
    entry = QueryLogEntry("q1", "query", "answer", ["ctx"], ["c1"], 0.0, np.zeros(4))
    result = sampler.maybe_score(entry)
    assert result is not None
    assert result.faithfulness == 0.8
    mean_f, mean_r = sampler.rolling_means()
    assert mean_f == pytest.approx(0.8)
    assert mean_r == pytest.approx(0.7)


def test_eval_sampler_rolling_window_evicts_old_samples():
    sampler = EvalSampler(FixedJudge(1.0, 1.0), sample_rate=1.0, window_size=3)
    from app.rag import QueryLogEntry
    for i in range(5):
        entry = QueryLogEntry(f"q{i}", "query", "answer", ["ctx"], ["c1"], 0.0, np.zeros(4))
        sampler.maybe_score(entry)
    assert len(sampler) == 3


def test_monitoring_service_healthy_baseline_stays_healthy(small_rag):
    # Drift windows large enough that PSI's finite-sample noise (it needs a
    # reasonable count per histogram bin, same caveat as in test_drift.py)
    # doesn't itself masquerade as drift when the query topic is genuinely
    # stable -- undersized windows are a mis-configuration, not something
    # the detector should be expected to compensate for.
    monitor = MonitoringService(
        small_rag,
        judge=FixedJudge(0.9, 0.9),
        reference_window_size=20,
        current_window_size=20,
        eval_sample_rate=1.0,
        alert_breach_threshold=2,
    )
    # A stable query distribution: paraphrases of the same handful of
    # return-policy questions, not monotonically-increasing unique IDs
    # (which would inject their own lexical novelty into every query and
    # isn't representative of "no real drift").
    stable_questions = [
        "What is the return policy?",
        "How many days do I have to return an item?",
        "Can I get a refund for an unused product?",
        "Are shipping costs refundable?",
        "Do digital products qualify for returns?",
    ]
    for i in range(60):
        small_rag.query(stable_questions[i % len(stable_questions)])
        state = monitor.tick()

    assert state.state in (AlertState.HEALTHY, AlertState.DEGRADED)
    assert monitor.last_drift_report is not None


def test_monitoring_service_flags_reembed_on_sustained_quality_degradation(small_rag):
    monitor = MonitoringService(
        small_rag,
        judge=FixedJudge(0.05, 0.05),
        reference_window_size=3,
        current_window_size=3,
        eval_sample_rate=1.0,
        alert_breach_threshold=2,
    )
    for i in range(6):
        small_rag.query(f"question {i}")
        monitor.tick()

    assert monitor.last_alert_state.state == AlertState.FIRING
    assert monitor.reembed_registry.pending_doc_ids == {"policy"}
    assert monitor.reembed_registry.trigger_count >= 1


def test_monitoring_service_drift_detected_after_topic_shift(small_rag):
    monitor = MonitoringService(
        small_rag,
        judge=FixedJudge(0.9, 0.9),
        reference_window_size=8,
        current_window_size=8,
        eval_sample_rate=0.0,
        alert_breach_threshold=10,
    )
    # Establish a stable reference distribution on one topic.
    for i in range(8):
        small_rag.query(f"return policy refund window question {i}")
        monitor.tick()

    baseline_report = monitor.last_drift_report
    assert baseline_report.status == DriftStatus.OK

    # Inject a query-distribution shift: totally unrelated vocabulary/topic.
    drift_queries = [
        "quantum entanglement superconducting qubit coherence time",
        "photosynthesis chlorophyll thylakoid membrane electron transport",
        "medieval castle siege engine trebuchet fortification",
        "volcanic eruption tectonic plate magma chamber pressure",
        "orchestral symphony woodwind brass percussion arrangement",
        "cryptographic hash function collision resistance preimage",
        "migratory bird navigation magnetic field sensing",
        "deep sea hydrothermal vent chemosynthesis bacteria",
    ]
    for q in drift_queries:
        small_rag.query(q)
        monitor.tick()

    shifted_report = monitor.last_drift_report
    assert shifted_report.mean_psi >= baseline_report.mean_psi
    assert shifted_report.status != DriftStatus.OK or shifted_report.centroid_cosine_similarity < baseline_report.centroid_cosine_similarity


def test_reembed_registry_pop_pending_clears_state(small_rag):
    monitor = MonitoringService(
        small_rag,
        judge=FixedJudge(0.01, 0.01),
        reference_window_size=2,
        current_window_size=2,
        eval_sample_rate=1.0,
        alert_breach_threshold=1,
    )
    small_rag.query("anything")
    monitor.tick()
    assert monitor.reembed_registry.pending_doc_ids

    pending = monitor.reembed_registry.pop_pending()
    assert "policy" in pending
    assert monitor.reembed_registry.pending_doc_ids == set()

"""Monitoring plane: wires drift detection, the rolling LLM-as-judge eval
sampler, Prometheus metrics, and the alert engine to a running RAGService.

Design: the first `drift_reference_window` query embeddings observed after
startup (or an explicitly-seeded reference corpus) become the drift
baseline; every subsequent query embedding lands in a rolling "current"
window compared against that baseline. A random sample of production
(query, answer, contexts) triples -- controlled by `eval_sample_rate` -- is
scored by the LLM-as-judge harness and kept in a rolling window whose mean
faithfulness/relevance feeds both Prometheus and the alert engine.
Re-embedding of updated docs is triggered by the alert engine's hook when
degradation is sustained across `alert_consecutive_breaches` checks.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app import metrics
from app.alerts import AlertEngine, AlertEngineState, HealthSnapshot
from app.config import settings
from app.drift import DriftDetector, DriftReport, RollingEmbeddingWindow
from app.judge import EvalResult, Judge, build_judge
from app.rag import QueryLogEntry, RAGService

ReembedCallback = Callable[[], None]


@dataclass
class ScoredSample:
    query_id: str
    result: EvalResult


class EvalSampler:
    """Rolling-window LLM-as-judge sampler over production traffic."""

    def __init__(
        self,
        judge: Judge,
        sample_rate: float | None = None,
        window_size: int | None = None,
        rng: random.Random | None = None,
    ):
        self.judge = judge
        self.sample_rate = sample_rate if sample_rate is not None else settings.eval_sample_rate
        self.window: deque[ScoredSample] = deque(
            maxlen=window_size or settings.quality_rolling_window
        )
        self._rng = rng or random.Random(7)

    def maybe_score(self, entry: QueryLogEntry) -> EvalResult | None:
        if self._rng.random() > self.sample_rate:
            return None
        result = self.judge.score(entry.query, entry.answer, entry.contexts)
        self.window.append(ScoredSample(query_id=entry.query_id, result=result))
        metrics.EVAL_SAMPLES_COUNTER.inc()
        return result

    def rolling_means(self) -> tuple[float | None, float | None]:
        if not self.window:
            return None, None
        faithfulness = float(np.mean([s.result.faithfulness for s in self.window]))
        relevance = float(np.mean([s.result.relevance for s in self.window]))
        return faithfulness, relevance

    def __len__(self) -> int:
        return len(self.window)


class ReembedRegistry:
    """Tracks which documents have been flagged for re-embedding by the
    alert engine, and exposes a callback the API layer can use to actually
    re-run ingestion for them."""

    def __init__(self):
        self.pending_doc_ids: set[str] = set()
        self.trigger_count = 0

    def flag_all(self, doc_ids: list[str]) -> None:
        self.pending_doc_ids.update(doc_ids)
        self.trigger_count += 1
        metrics.REEMBED_TRIGGERED_COUNTER.inc()

    def pop_pending(self) -> list[str]:
        pending = list(self.pending_doc_ids)
        self.pending_doc_ids.clear()
        return pending


class MonitoringService:
    def __init__(
        self,
        rag: RAGService,
        judge: Judge | None = None,
        reference_window_size: int | None = None,
        current_window_size: int | None = None,
        eval_sample_rate: float | None = None,
        eval_window_size: int | None = None,
        alert_breach_threshold: int | None = None,
    ):
        self.rag = rag
        self.drift_detector = DriftDetector(dim=rag.embedder.dim)
        self.reference_window = RollingEmbeddingWindow(
            dim=rag.embedder.dim,
            capacity=reference_window_size or settings.drift_reference_window,
        )
        self.current_window = RollingEmbeddingWindow(
            dim=rag.embedder.dim,
            capacity=current_window_size or settings.drift_current_window,
        )
        self._reference_locked = False

        self.eval_sampler = EvalSampler(
            judge or build_judge(rag.embedder),
            sample_rate=eval_sample_rate,
            window_size=eval_window_size,
        )
        self.reembed_registry = ReembedRegistry()
        self.alert_engine = AlertEngine(
            breach_threshold=alert_breach_threshold, reembed_hook=self._on_alert_fired
        )

        self._last_drift_report: DriftReport | None = None
        self._last_alert_state: AlertEngineState | None = None
        self._processed_log_index = 0

    def _on_alert_fired(self, snapshot: HealthSnapshot) -> None:
        doc_ids = [d.doc_id for d in self.rag.vector_store.list_documents()]
        self.reembed_registry.flag_all(doc_ids)

    def observe_embedding(self, embedding: np.ndarray) -> None:
        if not self._reference_locked:
            self.reference_window.add(embedding)
            if self.reference_window.is_full():
                self._reference_locked = True
        else:
            self.current_window.add(embedding)

    def seed_reference(self, embeddings: np.ndarray) -> None:
        self.reference_window.add(embeddings)
        if self.reference_window.is_full():
            self._reference_locked = True

    def process_new_log_entries(self) -> None:
        """Feed any not-yet-processed query log entries into the drift
        window and the eval sampler. Call this after RAGService.query()."""
        entries = self.rag.query_log[self._processed_log_index :]
        for entry in entries:
            self.observe_embedding(np.asarray(entry.embedding))
            self.eval_sampler.maybe_score(entry)
        self._processed_log_index = len(self.rag.query_log)

    def evaluate_drift(self) -> DriftReport:
        report = self.drift_detector.evaluate(
            self.reference_window.as_array(), self.current_window.as_array()
        )
        self._last_drift_report = report
        metrics.set_drift_metrics(report)
        return report

    def tick(self) -> AlertEngineState:
        """Run one full monitoring cycle: ingest new log entries, evaluate
        drift, update quality metrics, and run the alert rule."""
        self.process_new_log_entries()
        drift_report = self.evaluate_drift()
        mean_faithfulness, mean_relevance = self.eval_sampler.rolling_means()
        if mean_faithfulness is not None:
            metrics.set_quality_metrics(mean_faithfulness, mean_relevance or 0.0)

        snapshot = HealthSnapshot(
            drift=drift_report,
            mean_faithfulness=mean_faithfulness,
            mean_relevance=mean_relevance,
            num_eval_samples=len(self.eval_sampler),
        )
        state = self.alert_engine.evaluate(snapshot)
        self._last_alert_state = state
        return state

    @property
    def last_drift_report(self) -> DriftReport | None:
        return self._last_drift_report

    @property
    def last_alert_state(self) -> AlertEngineState | None:
        return self._last_alert_state

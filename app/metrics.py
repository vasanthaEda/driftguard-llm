"""Prometheus metric definitions, exposed at GET /metrics.

Grafana's bundled dashboard (grafana/dashboards/driftguard.json) graphs
exactly these series against a Prometheus datasource scraping this app.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

registry = CollectorRegistry()

QUERY_COUNTER = Counter(
    "driftguard_queries_total",
    "Total number of RAG queries served",
    registry=registry,
)

QUERY_LATENCY = Histogram(
    "driftguard_query_latency_seconds",
    "End-to-end query latency (retrieval + generation)",
    registry=registry,
)

DRIFT_MEAN_PSI = Gauge(
    "driftguard_drift_mean_psi",
    "Mean Population Stability Index across random embedding projections",
    registry=registry,
)

DRIFT_CENTROID_COSINE = Gauge(
    "driftguard_drift_centroid_cosine_similarity",
    "Cosine similarity between reference and current embedding centroids",
    registry=registry,
)

DRIFT_KS_REJECT_FRACTION = Gauge(
    "driftguard_drift_ks_reject_fraction",
    "Fraction of random projections where the KS test rejects the null hypothesis",
    registry=registry,
)

DRIFT_STATUS = Gauge(
    "driftguard_drift_status",
    "Current drift status (0=ok, 1=warn, 2=critical)",
    registry=registry,
)

QUALITY_FAITHFULNESS = Gauge(
    "driftguard_quality_faithfulness_avg",
    "Rolling average faithfulness score from the LLM-as-judge eval sampler",
    registry=registry,
)

QUALITY_RELEVANCE = Gauge(
    "driftguard_quality_relevance_avg",
    "Rolling average relevance score from the LLM-as-judge eval sampler",
    registry=registry,
)

EVAL_SAMPLES_COUNTER = Counter(
    "driftguard_eval_samples_total",
    "Total number of production responses scored by the eval harness",
    registry=registry,
)

ALERTS_FIRED_COUNTER = Counter(
    "driftguard_alerts_fired_total",
    "Total number of alerts fired by the alert engine",
    ["reason"],
    registry=registry,
)

REEMBED_TRIGGERED_COUNTER = Counter(
    "driftguard_reembed_triggered_total",
    "Total number of automated re-embedding jobs triggered by alerts",
    registry=registry,
)

_STATUS_TO_VALUE = {"ok": 0, "warn": 1, "critical": 2}


def set_drift_metrics(report) -> None:
    DRIFT_MEAN_PSI.set(report.mean_psi)
    DRIFT_CENTROID_COSINE.set(report.centroid_cosine_similarity)
    DRIFT_KS_REJECT_FRACTION.set(report.ks_reject_fraction)
    DRIFT_STATUS.set(_STATUS_TO_VALUE.get(report.status.value, 0))


def set_quality_metrics(mean_faithfulness: float, mean_relevance: float) -> None:
    QUALITY_FAITHFULNESS.set(mean_faithfulness)
    QUALITY_RELEVANCE.set(mean_relevance)

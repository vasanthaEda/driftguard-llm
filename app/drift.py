"""Embedding-space drift detection for query/retrieval traffic.

Detects distribution shift between a reference window of embeddings
(baseline "known good" traffic) and a current window (recent production
traffic) using three complementary, standard statistical techniques:

1. Centroid cosine similarity -- cheap, catches gross topic shift.
2. Population Stability Index (PSI) on random 1D projections -- the same
   metric tools like Evidently AI use for numeric/embedding drift, binned
   against reference-derived quantiles.
3. Kolmogorov-Smirnov two-sample test on the same projections -- a
   distribution-free significance test, aggregated across projections via
   the fraction of projections rejecting the null at `alpha`.

Projecting a high-dimensional embedding onto random directions and running
univariate tests on each projection is the standard "random projection"
strategy for making multivariate drift tests tractable without requiring a
learned density model; it's what underlies Evidently's default embeddings
drift detector (`DriftScoreDistance`/`Anderson-Darling on projections`) at a
description level, reimplemented here with no external dependency so it
runs fully offline/deterministically in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import stats

from app.config import settings


class DriftStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    CRITICAL = "critical"


def _status_rank(status: DriftStatus) -> int:
    return {DriftStatus.OK: 0, DriftStatus.WARN: 1, DriftStatus.CRITICAL: 2}[status]


def _worse(a: DriftStatus, b: DriftStatus) -> DriftStatus:
    return a if _status_rank(a) >= _status_rank(b) else b


@dataclass(frozen=True)
class DriftReport:
    status: DriftStatus
    centroid_cosine_similarity: float
    mean_psi: float
    max_psi: float
    ks_reject_fraction: float
    reference_size: int
    current_size: int
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "centroid_cosine_similarity": self.centroid_cosine_similarity,
            "mean_psi": self.mean_psi,
            "max_psi": self.max_psi,
            "ks_reject_fraction": self.ks_reject_fraction,
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "reasons": self.reasons,
        }


def _random_projection_matrix(dim: int, n_projections: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(dim, n_projections))
    mat /= np.linalg.norm(mat, axis=0, keepdims=True)
    return mat


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index between two 1D samples.

    PSI = sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct)) over quantile bins
    derived from the reference distribution. Standard interpretation:
    < 0.1 no significant shift, 0.1-0.25 moderate shift, > 0.25 major shift.
    """
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        # Reference has near-zero variance on this projection; fall back to
        # a simple mean/std based split so PSI stays well-defined.
        edges = np.linspace(reference.min() - eps, reference.max() + eps, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class DriftDetector:
    def __init__(
        self,
        dim: int,
        n_projections: int | None = None,
        psi_warn: float | None = None,
        psi_critical: float | None = None,
        ks_alpha: float | None = None,
        centroid_cosine_warn: float | None = None,
        seed: int = 42,
    ):
        self.dim = dim
        self.n_projections = n_projections or settings.drift_num_projections
        self.psi_warn = psi_warn if psi_warn is not None else settings.drift_psi_warn
        self.psi_critical = (
            psi_critical if psi_critical is not None else settings.drift_psi_critical
        )
        self.ks_alpha = ks_alpha if ks_alpha is not None else settings.drift_ks_alpha
        self.centroid_cosine_warn = (
            centroid_cosine_warn
            if centroid_cosine_warn is not None
            else settings.drift_centroid_cosine_warn
        )
        self._projections = _random_projection_matrix(dim, self.n_projections, seed=seed)

    def evaluate(self, reference: np.ndarray, current: np.ndarray) -> DriftReport:
        if reference.shape[0] < 5 or current.shape[0] < 5:
            return DriftReport(
                status=DriftStatus.OK,
                centroid_cosine_similarity=1.0,
                mean_psi=0.0,
                max_psi=0.0,
                ks_reject_fraction=0.0,
                reference_size=reference.shape[0],
                current_size=current.shape[0],
                reasons=["insufficient_samples"],
            )

        ref_centroid = reference.mean(axis=0)
        cur_centroid = current.mean(axis=0)
        denom = np.linalg.norm(ref_centroid) * np.linalg.norm(cur_centroid)
        centroid_cos = float(np.dot(ref_centroid, cur_centroid) / denom) if denom > 0 else 1.0

        ref_proj = reference @ self._projections
        cur_proj = current @ self._projections

        # PSI quantile bins need a handful of points per bin to be a
        # meaningful estimate of the reference distribution's shape; with
        # 10 fixed bins and a small reference window, each bin edge is
        # itself a noisy statistic and produces spuriously large PSI even
        # under zero real drift. Scale the bin count down for small
        # windows rather than pretending decile resolution is available.
        min_window = min(reference.shape[0], current.shape[0])
        bins = max(2, min(10, min_window // 10))

        psis = np.array(
            [_psi(ref_proj[:, j], cur_proj[:, j], bins=bins) for j in range(self.n_projections)]
        )
        ks_pvalues = np.array(
            [
                stats.ks_2samp(ref_proj[:, j], cur_proj[:, j]).pvalue
                for j in range(self.n_projections)
            ]
        )
        ks_reject_fraction = float(np.mean(ks_pvalues < self.ks_alpha))

        mean_psi = float(np.mean(psis))
        max_psi = float(np.max(psis))

        reasons: list[str] = []
        status = DriftStatus.OK

        if mean_psi >= self.psi_critical:
            status = _worse(status, DriftStatus.CRITICAL)
            reasons.append(f"mean_psi {mean_psi:.3f} >= critical threshold {self.psi_critical}")
        elif mean_psi >= self.psi_warn:
            status = _worse(status, DriftStatus.WARN)
            reasons.append(f"mean_psi {mean_psi:.3f} >= warn threshold {self.psi_warn}")

        if ks_reject_fraction >= 0.5:
            status = _worse(status, DriftStatus.CRITICAL)
            reasons.append(
                f"KS test rejected null on {ks_reject_fraction:.0%} of projections (alpha={self.ks_alpha})"
            )
        elif ks_reject_fraction >= 0.25:
            status = _worse(status, DriftStatus.WARN)
            reasons.append(
                f"KS test rejected null on {ks_reject_fraction:.0%} of projections (alpha={self.ks_alpha})"
            )

        if centroid_cos < self.centroid_cosine_warn:
            status = _worse(status, DriftStatus.WARN)
            reasons.append(
                f"centroid cosine similarity {centroid_cos:.3f} < warn threshold "
                f"{self.centroid_cosine_warn}"
            )

        if not reasons:
            reasons.append("no significant drift detected")

        return DriftReport(
            status=status,
            centroid_cosine_similarity=centroid_cos,
            mean_psi=mean_psi,
            max_psi=max_psi,
            ks_reject_fraction=ks_reject_fraction,
            reference_size=reference.shape[0],
            current_size=current.shape[0],
            reasons=reasons,
        )


class RollingEmbeddingWindow:
    """Fixed-capacity FIFO buffer of embeddings used as the drift reference
    or current window."""

    def __init__(self, dim: int, capacity: int):
        self.dim = dim
        self.capacity = capacity
        self._buf = np.zeros((0, dim), dtype=np.float32)

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        self._buf = np.vstack([self._buf, embeddings.astype(np.float32)])
        if len(self._buf) > self.capacity:
            self._buf = self._buf[-self.capacity :]

    def as_array(self) -> np.ndarray:
        return self._buf

    def __len__(self) -> int:
        return len(self._buf)

    def is_full(self) -> bool:
        return len(self._buf) >= self.capacity

"""Synthetic drift-injection suite.

Rather than relying on real production traffic, we synthesize embeddings
from controlled distributions and verify the DriftDetector correctly
distinguishes "same distribution" (no drift) from injected shifts of
increasing severity: a small mean shift, a large mean shift (topic
change), and a variance/scale change. This is the standard way to validate
a drift detector's sensitivity and specificity before trusting it on real
traffic.
"""
import numpy as np
import pytest

from app.drift import DriftDetector, DriftStatus, RollingEmbeddingWindow, _psi

DIM = 32


def _make_gaussian_embeddings(rng, n, mean, scale=1.0):
    raw = rng.normal(loc=mean, scale=scale, size=(n, DIM))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (raw / norms).astype(np.float32)


@pytest.fixture
def detector() -> DriftDetector:
    return DriftDetector(dim=DIM, seed=123)


def test_identical_distribution_reports_no_drift(detector):
    rng = np.random.default_rng(1)
    # A non-zero baseline mean keeps the centroid well away from the origin
    # so cosine similarity is a meaningful, numerically stable signal (an
    # isotropic zero-mean Gaussian has a centroid that is itself ~noise).
    # Sample sizes large enough that PSI's inherent finite-sample noise
    # (it needs a reasonable count per histogram bin) doesn't itself read
    # as drift -- the same reason real deployments should size their drift
    # windows in the hundreds, not dozens.
    reference = _make_gaussian_embeddings(rng, 2000, mean=3.0)
    current = _make_gaussian_embeddings(rng, 500, mean=3.0)

    report = detector.evaluate(reference, current)

    assert report.status == DriftStatus.OK
    assert report.mean_psi < detector.psi_warn
    assert report.centroid_cosine_similarity > 0.9


def test_large_mean_shift_is_flagged_critical(detector):
    rng = np.random.default_rng(2)
    reference = _make_gaussian_embeddings(rng, 300, mean=3.0)
    # Inject a severe topic shift: current traffic embeddings drawn from a
    # completely different region of the space (opposite side of the sphere).
    current = _make_gaussian_embeddings(rng, 100, mean=-3.0)

    report = detector.evaluate(reference, current)

    assert report.status == DriftStatus.CRITICAL
    assert report.mean_psi >= detector.psi_warn
    assert len(report.reasons) > 0
    assert report.reasons != ["no significant drift detected"]


def test_moderate_shift_is_flagged_warn_or_worse(detector):
    rng = np.random.default_rng(3)
    reference = _make_gaussian_embeddings(rng, 300, mean=3.0)
    current = _make_gaussian_embeddings(rng, 100, mean=3.6)

    report = detector.evaluate(reference, current)

    assert report.status in (DriftStatus.WARN, DriftStatus.CRITICAL)


def test_variance_only_shift_is_detected_by_psi_or_ks(detector):
    rng = np.random.default_rng(4)
    reference = _make_gaussian_embeddings(rng, 300, mean=3.0, scale=1.0)
    current = _make_gaussian_embeddings(rng, 100, mean=3.0, scale=4.0)

    report = detector.evaluate(reference, current)

    # A pure scale change without a mean shift should still show up in the
    # PSI/KS statistics even though centroid cosine similarity stays high.
    assert report.mean_psi > 0.0 or report.ks_reject_fraction > 0.0


def test_insufficient_samples_defaults_to_ok(detector):
    rng = np.random.default_rng(5)
    reference = _make_gaussian_embeddings(rng, 3, mean=3.0)
    current = _make_gaussian_embeddings(rng, 2, mean=10.0)

    report = detector.evaluate(reference, current)
    assert report.status == DriftStatus.OK
    assert "insufficient_samples" in report.reasons


def test_report_serializes_to_dict(detector):
    rng = np.random.default_rng(6)
    reference = _make_gaussian_embeddings(rng, 2000, mean=3.0)
    current = _make_gaussian_embeddings(rng, 500, mean=3.0)
    report = detector.evaluate(reference, current)
    d = report.as_dict()
    assert set(d.keys()) == {
        "status", "centroid_cosine_similarity", "mean_psi", "max_psi",
        "ks_reject_fraction", "reference_size", "current_size", "reasons",
    }
    assert d["status"] == "ok"


def test_psi_is_near_zero_for_identical_samples():
    rng = np.random.default_rng(7)
    sample = rng.normal(size=1000)
    assert _psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_increases_with_shift_magnitude():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=1000)
    small_shift = rng.normal(loc=0.2, size=1000)
    large_shift = rng.normal(loc=2.0, size=1000)

    psi_small = _psi(reference, small_shift)
    psi_large = _psi(reference, large_shift)

    assert psi_large > psi_small > 0


class TestRollingEmbeddingWindow:
    def test_window_respects_capacity(self):
        window = RollingEmbeddingWindow(dim=4, capacity=3)
        for i in range(5):
            window.add(np.array([float(i)] * 4, dtype=np.float32))
        assert len(window) == 3
        # oldest entries (0, 1) should have been evicted; last row should be 4.0s
        np.testing.assert_array_equal(window.as_array()[-1], np.array([4.0] * 4, dtype=np.float32))

    def test_is_full(self):
        window = RollingEmbeddingWindow(dim=2, capacity=2)
        assert not window.is_full()
        window.add(np.array([1.0, 1.0]))
        assert not window.is_full()
        window.add(np.array([2.0, 2.0]))
        assert window.is_full()

    def test_add_accepts_batch(self):
        window = RollingEmbeddingWindow(dim=2, capacity=10)
        batch = np.ones((4, 2), dtype=np.float32)
        window.add(batch)
        assert len(window) == 4

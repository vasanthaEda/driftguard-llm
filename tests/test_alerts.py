from app.alerts import AlertEngine, AlertState, HealthSnapshot
from app.drift import DriftReport, DriftStatus


def _drift_report(status: DriftStatus) -> DriftReport:
    return DriftReport(
        status=status,
        centroid_cosine_similarity=0.99,
        mean_psi=0.02,
        max_psi=0.03,
        ks_reject_fraction=0.0,
        reference_size=100,
        current_size=50,
        reasons=["synthetic"],
    )


def _healthy_snapshot() -> HealthSnapshot:
    return HealthSnapshot(
        drift=_drift_report(DriftStatus.OK),
        mean_faithfulness=0.9,
        mean_relevance=0.9,
        num_eval_samples=10,
    )


def _degraded_snapshot() -> HealthSnapshot:
    return HealthSnapshot(
        drift=_drift_report(DriftStatus.CRITICAL),
        mean_faithfulness=0.2,
        mean_relevance=0.2,
        num_eval_samples=10,
    )


def test_single_healthy_snapshot_stays_healthy():
    engine = AlertEngine(breach_threshold=3)
    state = engine.evaluate(_healthy_snapshot())
    assert state.state == AlertState.HEALTHY
    assert state.consecutive_breaches == 0


def test_single_bad_snapshot_is_degraded_not_firing():
    engine = AlertEngine(breach_threshold=3)
    state = engine.evaluate(_degraded_snapshot())
    assert state.state == AlertState.DEGRADED
    assert state.consecutive_breaches == 1


def test_sustained_degradation_fires_alert():
    engine = AlertEngine(breach_threshold=3)
    engine.evaluate(_degraded_snapshot())
    engine.evaluate(_degraded_snapshot())
    state = engine.evaluate(_degraded_snapshot())
    assert state.state == AlertState.FIRING
    assert state.last_event is not None
    assert len(state.history) == 1


def test_healthy_snapshot_resets_breach_counter():
    engine = AlertEngine(breach_threshold=3)
    engine.evaluate(_degraded_snapshot())
    engine.evaluate(_degraded_snapshot())
    state = engine.evaluate(_healthy_snapshot())
    assert state.consecutive_breaches == 0
    assert state.state == AlertState.HEALTHY

    # degradation must re-accumulate from zero after the reset
    engine.evaluate(_degraded_snapshot())
    engine.evaluate(_degraded_snapshot())
    state = engine.evaluate(_degraded_snapshot())
    assert state.state == AlertState.FIRING


def test_reembed_hook_invoked_only_once_alert_fires():
    calls = []

    def hook(snapshot):
        calls.append(snapshot)

    engine = AlertEngine(breach_threshold=2, reembed_hook=hook, auto_reembed=True)
    engine.evaluate(_degraded_snapshot())
    assert calls == []
    engine.evaluate(_degraded_snapshot())
    assert len(calls) == 1


def test_reembed_hook_not_invoked_when_auto_reembed_disabled():
    calls = []
    engine = AlertEngine(breach_threshold=1, reembed_hook=lambda s: calls.append(s), auto_reembed=False)
    engine.evaluate(_degraded_snapshot())
    assert calls == []


def test_quality_only_degradation_triggers_alert_even_with_ok_drift():
    engine = AlertEngine(breach_threshold=1)
    snapshot = HealthSnapshot(
        drift=_drift_report(DriftStatus.OK),
        mean_faithfulness=0.1,
        mean_relevance=0.9,
        num_eval_samples=5,
    )
    state = engine.evaluate(snapshot)
    assert state.state == AlertState.FIRING
    assert any("faithfulness" in r for r in state.last_event.reasons)

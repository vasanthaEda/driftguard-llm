"""Alert rule: combine drift + rolling quality signals, debounce with a
consecutive-breach counter, and trigger a re-embedding hook on sustained
degradation.

The debounce (require N consecutive unhealthy evaluations before firing) is
the standard way to avoid paging on a single noisy sample -- one bad PSI
reading or one low-scoring judge sample shouldn't page anyone, but three in
a row plausibly indicates real degradation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from app.config import settings
from app.drift import DriftReport, DriftStatus


class AlertState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FIRING = "firing"


@dataclass
class HealthSnapshot:
    drift: DriftReport | None
    mean_faithfulness: float | None
    mean_relevance: float | None
    num_eval_samples: int


@dataclass
class AlertEvent:
    fired_at: float
    reasons: list[str]
    snapshot: HealthSnapshot


@dataclass
class AlertEngineState:
    state: AlertState = AlertState.HEALTHY
    consecutive_breaches: int = 0
    last_event: AlertEvent | None = None
    history: list[AlertEvent] = field(default_factory=list)


ReembedHook = Callable[[HealthSnapshot], None]


def _is_unhealthy(snapshot: HealthSnapshot) -> list[str]:
    reasons: list[str] = []
    if snapshot.drift is not None and snapshot.drift.status != DriftStatus.OK:
        reasons.append(f"embedding drift status={snapshot.drift.status.value}")

    if snapshot.mean_faithfulness is not None and snapshot.num_eval_samples > 0:
        if snapshot.mean_faithfulness < settings.quality_faithfulness_warn:
            reasons.append(
                f"rolling faithfulness {snapshot.mean_faithfulness:.2f} < "
                f"{settings.quality_faithfulness_warn}"
            )
    if snapshot.mean_relevance is not None and snapshot.num_eval_samples > 0:
        if snapshot.mean_relevance < settings.quality_relevance_warn:
            reasons.append(
                f"rolling relevance {snapshot.mean_relevance:.2f} < "
                f"{settings.quality_relevance_warn}"
            )
    return reasons


class AlertEngine:
    """Stateful alert rule evaluated once per monitoring tick.

    Fires (transitions to FIRING and invokes the re-embed hook, if any)
    once `consecutive_breaches` unhealthy snapshots are observed in a row.
    A single healthy snapshot resets the breach counter, matching typical
    "N consecutive failures" alerting semantics.
    """

    def __init__(
        self,
        breach_threshold: int | None = None,
        reembed_hook: ReembedHook | None = None,
        auto_reembed: bool | None = None,
    ):
        self.breach_threshold = breach_threshold or settings.alert_consecutive_breaches
        self.reembed_hook = reembed_hook
        self.auto_reembed = (
            auto_reembed if auto_reembed is not None else settings.auto_reembed_on_alert
        )
        self.state = AlertEngineState()

    def evaluate(self, snapshot: HealthSnapshot) -> AlertEngineState:
        reasons = _is_unhealthy(snapshot)

        if reasons:
            self.state.consecutive_breaches += 1
        else:
            self.state.consecutive_breaches = 0
            self.state.state = AlertState.HEALTHY
            return self.state

        if self.state.consecutive_breaches >= self.breach_threshold:
            event = AlertEvent(fired_at=time.time(), reasons=reasons, snapshot=snapshot)
            self.state.state = AlertState.FIRING
            self.state.last_event = event
            self.state.history.append(event)
            if self.auto_reembed and self.reembed_hook is not None:
                self.reembed_hook(snapshot)
        else:
            self.state.state = AlertState.DEGRADED

        return self.state

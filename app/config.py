"""Central configuration for driftguard-llm.

All values are overridable via environment variables so the same code runs
in tests (offline, deterministic backends), locally, and in the Docker image
(optionally wired to real OpenAI / Qdrant / pgvector backends). Nothing here
requires a secret to be present -- missing secrets simply mean the
deterministic/offline backends stay in effect.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Embedding / chunking
    embedding_dim: int = field(default_factory=lambda: _env_int("DRIFTGUARD_EMBED_DIM", 64))
    chunk_size: int = field(default_factory=lambda: _env_int("DRIFTGUARD_CHUNK_SIZE", 400))
    chunk_overlap: int = field(default_factory=lambda: _env_int("DRIFTGUARD_CHUNK_OVERLAP", 60))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("DRIFTGUARD_TOP_K", 4))

    # Drift detection
    drift_reference_window: int = field(
        default_factory=lambda: _env_int("DRIFTGUARD_REF_WINDOW", 200)
    )
    drift_current_window: int = field(
        default_factory=lambda: _env_int("DRIFTGUARD_CUR_WINDOW", 50)
    )
    drift_psi_warn: float = field(default_factory=lambda: _env_float("DRIFTGUARD_PSI_WARN", 0.1))
    drift_psi_critical: float = field(
        default_factory=lambda: _env_float("DRIFTGUARD_PSI_CRITICAL", 0.25)
    )
    drift_ks_alpha: float = field(default_factory=lambda: _env_float("DRIFTGUARD_KS_ALPHA", 0.01))
    drift_centroid_cosine_warn: float = field(
        default_factory=lambda: _env_float("DRIFTGUARD_CENTROID_COS_WARN", 0.90)
    )
    drift_num_projections: int = field(
        default_factory=lambda: _env_int("DRIFTGUARD_NUM_PROJECTIONS", 8)
    )

    # LLM-as-judge / quality eval
    eval_sample_rate: float = field(
        default_factory=lambda: _env_float("DRIFTGUARD_EVAL_SAMPLE_RATE", 1.0)
    )
    quality_faithfulness_warn: float = field(
        default_factory=lambda: _env_float("DRIFTGUARD_FAITHFULNESS_WARN", 0.6)
    )
    quality_relevance_warn: float = field(
        default_factory=lambda: _env_float("DRIFTGUARD_RELEVANCE_WARN", 0.6)
    )
    quality_rolling_window: int = field(
        default_factory=lambda: _env_int("DRIFTGUARD_QUALITY_WINDOW", 30)
    )

    # Alerting
    alert_consecutive_breaches: int = field(
        default_factory=lambda: _env_int("DRIFTGUARD_ALERT_BREACHES", 3)
    )
    auto_reembed_on_alert: bool = field(
        default_factory=lambda: _env_bool("DRIFTGUARD_AUTO_REEMBED", True)
    )

    # Storage
    database_path: str = field(
        default_factory=lambda: os.getenv("DRIFTGUARD_DB_PATH", ":memory:")
    )

    # Backends: "fake" (offline/deterministic, default & test-safe) or "openai"
    embedder_backend: str = field(
        default_factory=lambda: os.getenv("DRIFTGUARD_EMBEDDER_BACKEND", "fake")
    )
    judge_backend: str = field(
        default_factory=lambda: os.getenv("DRIFTGUARD_JUDGE_BACKEND", "heuristic")
    )
    llm_backend: str = field(
        default_factory=lambda: os.getenv("DRIFTGUARD_LLM_BACKEND", "fake")
    )


settings = Settings()

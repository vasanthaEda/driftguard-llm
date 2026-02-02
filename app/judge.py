"""LLM-as-judge evaluation harness: faithfulness + relevance scoring.

`Judge` is the pluggable interface. Two backends are provided:

- `HeuristicJudge` (default, offline-safe): scores faithfulness as how well
  the answer is grounded in the retrieved context, and relevance as how
  well the answer addresses the query, using embedding cosine similarity
  (the same embedder that powers retrieval) blended with lexical overlap
  for robustness against embedding collisions. This is a real, deterministic
  scorer -- not a stub -- and is what the unit tests exercise, since it
  requires no network access and no API key.

- `OpenAIJudge` (production, opt-in): the actual "LLM-as-judge" pattern --
  prompts a chat model to rate faithfulness/relevance on a 0-1 scale with a
  short rationale, parsed from structured JSON output. Never invoked by the
  test suite; wired in only when DRIFTGUARD_JUDGE_BACKEND=openai.

Both backends implement the same `EvalResult`-returning interface so
`app/monitoring.py`'s rolling eval sampler doesn't care which is active.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from app.config import settings
from app.embeddings import Embedder

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to",
    "in", "on", "for", "and", "or", "with", "as", "at", "by", "it", "that",
    "this", "these", "those", "from", "into", "than", "then", "so", "but",
    "not", "no", "do", "does", "did", "has", "have", "had", "can", "will",
    "would", "could", "should", "its", "their", "his", "her", "he", "she",
    "they", "we", "you", "i",
    # interrogatives: content-free in isolation, but their presence in a
    # query shouldn't inflate the denominator of a content-word overlap
    # ratio the way "eiffel"/"tower"/"tall" should.
    "how", "what", "when", "where", "why", "which", "who", "whom",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def _lexical_overlap(a: str, b: str) -> float:
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return 0.0
    return float(np.dot(u, v) / denom)


@dataclass(frozen=True)
class EvalResult:
    faithfulness: float
    relevance: float
    verdict: str  # "pass" | "warn" | "fail"
    rationale: str = ""
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "details": self.details,
        }


class Judge(Protocol):
    def score(self, query: str, answer: str, contexts: Sequence[str]) -> EvalResult: ...


def _verdict_from_scores(faithfulness: float, relevance: float) -> str:
    f_warn = settings.quality_faithfulness_warn
    r_warn = settings.quality_relevance_warn
    if faithfulness < f_warn * 0.6 or relevance < r_warn * 0.6:
        return "fail"
    if faithfulness < f_warn or relevance < r_warn:
        return "warn"
    return "pass"


class HeuristicJudge:
    """Offline, deterministic faithfulness/relevance scorer.

    faithfulness = how much of the answer is supported by the retrieved
    context (blend of embedding similarity to the best-matching context
    chunk and lexical containment of answer tokens in the context).

    relevance = how well the answer addresses the query (blend of
    embedding similarity between answer and query, and lexical overlap).
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def score(self, query: str, answer: str, contexts: Sequence[str]) -> EvalResult:
        if not answer.strip():
            return EvalResult(
                faithfulness=0.0,
                relevance=0.0,
                verdict="fail",
                rationale="empty answer",
            )

        texts = [query, answer, *contexts]
        embs = self.embedder.embed(texts)
        q_emb, a_emb = embs[0], embs[1]
        ctx_embs = embs[2:] if contexts else np.zeros((0, embs.shape[1]))

        if len(ctx_embs):
            ctx_sims = [_cosine(a_emb, c) for c in ctx_embs]
            best_ctx_sim = max(ctx_sims)
            joined_ctx = " ".join(contexts)
            lexical_grounding = _lexical_overlap(answer, joined_ctx)
            faithfulness = float(np.clip(0.6 * max(best_ctx_sim, 0.0) + 0.4 * lexical_grounding, 0.0, 1.0))
        else:
            best_ctx_sim = 0.0
            lexical_grounding = 0.0
            faithfulness = 0.0

        answer_query_sim = _cosine(a_emb, q_emb)
        # Relevance lexical signal: what fraction of the *query's* content
        # terms actually show up in the answer (recall against the
        # question being asked) -- not the reverse, which would penalize
        # long, on-topic answers just for containing extra detail.
        lexical_relevance = _lexical_overlap(query, answer)
        # Lexical recall of the query's content words is weighted more
        # heavily than raw embedding similarity here: with a hashed,
        # non-learned embedding space, cross-sentence cosine similarity is
        # a noisier relevance signal than direct term coverage, but the
        # embedding term still matters for catching answers that reuse
        # query vocabulary while being semantically off-topic.
        relevance = float(np.clip(0.4 * max(answer_query_sim, 0.0) + 0.6 * lexical_relevance, 0.0, 1.0))

        verdict = _verdict_from_scores(faithfulness, relevance)
        rationale = (
            f"best_context_similarity={best_ctx_sim:.2f}, lexical_grounding={lexical_grounding:.2f}, "
            f"answer_query_similarity={answer_query_sim:.2f}, lexical_relevance={lexical_relevance:.2f}"
        )
        return EvalResult(
            faithfulness=faithfulness,
            relevance=relevance,
            verdict=verdict,
            rationale=rationale,
            details={
                "best_context_similarity": best_ctx_sim,
                "lexical_grounding": lexical_grounding,
                "answer_query_similarity": answer_query_sim,
                "lexical_relevance": lexical_relevance,
                "num_contexts": len(contexts),
            },
        )


_JUDGE_PROMPT = """You are an evaluation judge for a RAG system. Given a user \
query, retrieved context passages, and a generated answer, score the answer on:

- faithfulness (0.0-1.0): is every claim in the answer supported by the context?
- relevance (0.0-1.0): does the answer actually address the query?

Respond ONLY with JSON: {{"faithfulness": <float>, "relevance": <float>, "rationale": "<one sentence>"}}

Query: {query}
Context:
{context}
Answer: {answer}
"""


class OpenAIJudge:  # pragma: no cover - exercised only with real credentials
    """Real LLM-as-judge backend using an OpenAI chat model.

    Never called during the offline unit test suite (no network at test
    time). Requires OPENAI_API_KEY and DRIFTGUARD_JUDGE_BACKEND=openai.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package not installed; `pip install -r requirements-optional.txt`"
            ) from exc
        self._client = OpenAI()
        self.model = model

    def score(self, query: str, answer: str, contexts: Sequence[str]) -> EvalResult:
        prompt = _JUDGE_PROMPT.format(query=query, context="\n---\n".join(contexts), answer=answer)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = resp.choices[0].message.content or "{}"
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = {"faithfulness": 0.0, "relevance": 0.0, "rationale": "unparseable judge output"}
        faithfulness = float(payload.get("faithfulness", 0.0))
        relevance = float(payload.get("relevance", 0.0))
        return EvalResult(
            faithfulness=faithfulness,
            relevance=relevance,
            verdict=_verdict_from_scores(faithfulness, relevance),
            rationale=str(payload.get("rationale", "")),
        )


def build_judge(embedder: Embedder, backend: str | None = None) -> Judge:
    backend = backend or settings.judge_backend
    if backend == "openai":
        return OpenAIJudge()
    return HeuristicJudge(embedder=embedder)

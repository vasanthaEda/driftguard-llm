import pytest

from app.embeddings import HashingEmbedder
from app.judge import HeuristicJudge, _lexical_overlap


@pytest.fixture
def judge() -> HeuristicJudge:
    return HeuristicJudge(embedder=HashingEmbedder(dim=128, seed=99))


CONTEXT = [
    "The Eiffel Tower was completed in 1889 and stands 330 meters tall in Paris, France.",
    "It was designed by engineer Gustave Eiffel for the 1889 World's Fair.",
]


def test_grounded_relevant_answer_scores_high(judge):
    query = "How tall is the Eiffel Tower and when was it built?"
    answer = "The Eiffel Tower stands 330 meters tall and was completed in 1889 in Paris."

    result = judge.score(query, answer, CONTEXT)

    assert result.faithfulness > 0.5
    assert result.relevance > 0.5
    assert result.verdict == "pass"


def test_hallucinated_answer_scores_low_faithfulness(judge):
    query = "How tall is the Eiffel Tower and when was it built?"
    # Confidently wrong, unrelated claim not present anywhere in context.
    hallucinated_answer = "The Eiffel Tower was built by NASA in 1998 to study Martian soil samples."

    result = judge.score(query, hallucinated_answer, CONTEXT)

    grounded = judge.score(
        query,
        "The Eiffel Tower stands 330 meters tall and was completed in 1889 in Paris.",
        CONTEXT,
    )
    assert result.faithfulness < grounded.faithfulness


def test_off_topic_answer_scores_low_relevance(judge):
    query = "How tall is the Eiffel Tower and when was it built?"
    off_topic_answer = "Bananas are a great source of potassium and fiber for a balanced diet."

    result = judge.score(query, off_topic_answer, CONTEXT)

    assert result.relevance < 0.5


def test_empty_answer_fails(judge):
    result = judge.score("some query", "", CONTEXT)
    assert result.verdict == "fail"
    assert result.faithfulness == 0.0
    assert result.relevance == 0.0


def test_no_context_yields_zero_faithfulness(judge):
    result = judge.score("query", "some answer with no supporting context", [])
    assert result.faithfulness == 0.0


def test_scores_are_bounded_between_zero_and_one(judge):
    result = judge.score(
        "How tall is the Eiffel Tower?",
        "The Eiffel Tower stands 330 meters tall and was completed in 1889.",
        CONTEXT,
    )
    assert 0.0 <= result.faithfulness <= 1.0
    assert 0.0 <= result.relevance <= 1.0


def test_result_serializes_to_dict(judge):
    result = judge.score("q", "a", ["context"])
    d = result.as_dict()
    assert set(d.keys()) == {"faithfulness", "relevance", "verdict", "rationale", "details"}


def test_lexical_overlap_ignores_stopwords_and_case():
    overlap = _lexical_overlap("The Cat Sat On The Mat", "a mat with a cat")
    assert overlap > 0.0


def test_lexical_overlap_zero_for_disjoint_content():
    overlap = _lexical_overlap("quantum physics research", "banana bread recipe")
    assert overlap == 0.0


@pytest.mark.parametrize("verdict_input,expected", [
    ((0.9, 0.9), "pass"),
    ((0.05, 0.9), "fail"),
    ((0.9, 0.05), "fail"),
])
def test_verdict_thresholds(judge, verdict_input, expected):
    from app.judge import _verdict_from_scores
    faithfulness, relevance = verdict_input
    assert _verdict_from_scores(faithfulness, relevance) == expected

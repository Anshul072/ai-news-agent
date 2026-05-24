import json
from unittest.mock import MagicMock, patch

import pytest

from eval.judge import judge, DIMENSIONS

pytestmark = pytest.mark.usefixtures("_no_sleep")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("eval.judge.time.sleep", lambda _: None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NEWS_PARSE_INPUTS = {
    "article_title": "GPT-5 Released",
    "article_content": (
        "OpenAI announces GPT-5 with major improvements in reasoning and multi-step "
        "problem solving. The model scores state-of-the-art on several benchmarks."
    ),
}

NEWS_PARSE_OUTPUT = {
    "summary": "OpenAI released GPT-5 with major reasoning improvements.",
    "whats_new": "GPT-5 introduces chain-of-thought reasoning at scale.",
    "key_concepts": ["GPT-5", "reasoning", "OpenAI"],
    "use_cases": ["code generation", "research", "summarization"],
    "importance_score": 9,
    "importance_reasoning": "Major breakthrough in AI capabilities.",
}

MOCK_NEWS_PARSE_RESPONSE = {
    "summary_faithfulness": {
        "score": 4,
        "reasoning": "Summary accurately states 'OpenAI announces GPT-5' as in the article.",
    },
    "importance_score_calibration": {
        "score": 5,
        "reasoning": "Score 9/10 is appropriate for a major model release.",
    },
    "key_concepts_relevance": {
        "score": 5,
        "reasoning": "GPT-5 and reasoning are central to the article.",
    },
    "whats_new_specificity": {
        "score": 4,
        "reasoning": "Captures chain-of-thought reasoning as the novelty.",
    },
    "use_cases_plausibility": {
        "score": 3,
        "reasoning": "Use cases are somewhat generic but plausible.",
    },
}

MOCK_SENTIMENT_RESPONSE = {
    "label_score_consistency": {
        "score": 5,
        "reasoning": "Positive label aligns with 0.7 score.",
    },
    "concern_use_case_groundedness": {
        "score": 4,
        "reasoning": "Concerns traceable to HN comments.",
    },
    "quote_authenticity": {
        "score": 4,
        "reasoning": "Quotes read as genuine HN-style comments.",
    },
}

MOCK_RAG_RESPONSE = {
    "answer_relevance": {
        "score": 5,
        "reasoning": "Directly answers what GPT-5 is.",
    },
    "faithfulness": {
        "score": 5,
        "reasoning": "All claims supported by the retrieved context.",
    },
    "citation_accuracy": {
        "score": 5,
        "reasoning": "Cited article is the actual source.",
    },
}


def _mock_client(response_text: str):
    mock_response = MagicMock()
    mock_response.text = response_text
    mock = MagicMock()
    mock.models.generate_content.return_value = mock_response
    return mock


def _patch_client(response_text: str):
    """Return a context manager that patches _get_client to return a mock."""
    return patch("eval.judge._get_client", return_value=_mock_client(response_text))


# ---------------------------------------------------------------------------
# Behavior 1: news_parse_agent returns all expected dimension keys
# ---------------------------------------------------------------------------

def test_news_parse_returns_all_dimensions():
    with _patch_client(json.dumps(MOCK_NEWS_PARSE_RESPONSE)):
        result = judge("news_parse_agent", NEWS_PARSE_INPUTS, NEWS_PARSE_OUTPUT)

    assert set(result.keys()) == set(DIMENSIONS["news_parse_agent"])


# ---------------------------------------------------------------------------
# Behavior 2: sentiment_agent returns all expected dimension keys
# ---------------------------------------------------------------------------

def test_sentiment_agent_returns_all_dimensions():
    inputs = {
        "article_title": "GPT-5 Released",
        "hn_threads": [{"title": "GPT-5 is here", "top_comments": ["Amazing!", "Worried about jobs."]}],
    }
    output = {
        "sentiment_label": "Positive",
        "sentiment_score": 0.7,
        "top_concerns": ["Job displacement"],
        "top_use_cases": ["Coding assistant"],
        "notable_quotes": ["Amazing!"],
    }

    with _patch_client(json.dumps(MOCK_SENTIMENT_RESPONSE)):
        result = judge("sentiment_agent", inputs, output)

    assert set(result.keys()) == set(DIMENSIONS["sentiment_agent"])


# ---------------------------------------------------------------------------
# Behavior 3: rag_agent returns all expected dimension keys
# ---------------------------------------------------------------------------

def test_rag_agent_returns_all_dimensions():
    inputs = {
        "query": "What is GPT-5?",
        "context_chunks": "[Article 1] GPT-5 is OpenAI's latest model with improved reasoning.",
    }
    output = {
        "answer": "GPT-5 is OpenAI's latest language model with improved reasoning.",
        "citations": [{"article_id": 1, "title": "GPT-5 Released"}],
    }

    with _patch_client(json.dumps(MOCK_RAG_RESPONSE)):
        result = judge("rag_agent", inputs, output)

    assert set(result.keys()) == set(DIMENSIONS["rag_agent"])


# ---------------------------------------------------------------------------
# Behavior 4: all scores are integers in the 1-5 range
# ---------------------------------------------------------------------------

def test_all_scores_are_integers_in_valid_range():
    with _patch_client(json.dumps(MOCK_NEWS_PARSE_RESPONSE)):
        result = judge("news_parse_agent", NEWS_PARSE_INPUTS, NEWS_PARSE_OUTPUT)

    for dim, entry in result.items():
        assert isinstance(entry["score"], int), f"{dim}: score is not int"
        assert 1 <= entry["score"] <= 5, f"{dim}: score {entry['score']} out of range"


# ---------------------------------------------------------------------------
# Behavior 5: reasoning strings are non-empty for every dimension
# ---------------------------------------------------------------------------

def test_all_reasoning_strings_are_nonempty():
    with _patch_client(json.dumps(MOCK_NEWS_PARSE_RESPONSE)):
        result = judge("news_parse_agent", NEWS_PARSE_INPUTS, NEWS_PARSE_OUTPUT)

    for dim, entry in result.items():
        assert isinstance(entry["reasoning"], str), f"{dim}: reasoning is not str"
        assert entry["reasoning"], f"{dim}: reasoning is empty"


# ---------------------------------------------------------------------------
# Behavior 6: malformed JSON response does not crash; returns fallback structure
# ---------------------------------------------------------------------------

def test_malformed_response_returns_fallback_without_crash():
    with _patch_client("not valid json {{{"):
        result = judge("news_parse_agent", NEWS_PARSE_INPUTS, NEWS_PARSE_OUTPUT)

    assert set(result.keys()) == set(DIMENSIONS["news_parse_agent"])
    for dim, entry in result.items():
        assert 1 <= entry["score"] <= 5
        assert entry["reasoning"]


def test_empty_response_returns_fallback_without_crash():
    with _patch_client(""):
        result = judge("news_parse_agent", NEWS_PARSE_INPUTS, NEWS_PARSE_OUTPUT)

    assert set(result.keys()) == set(DIMENSIONS["news_parse_agent"])


# ---------------------------------------------------------------------------
# Behavior 7: rate-limiting sleep is applied on every call
# ---------------------------------------------------------------------------

def test_rate_limiting_sleep_is_applied():
    with _patch_client(json.dumps(MOCK_NEWS_PARSE_RESPONSE)), \
         patch("eval.judge.time.sleep") as mock_sleep:
        judge("news_parse_agent", NEWS_PARSE_INPUTS, NEWS_PARSE_OUTPUT)

    mock_sleep.assert_called_once_with(12)


# ---------------------------------------------------------------------------
# Behavior 8: out-of-range scores from judge are clamped to 1-5
# ---------------------------------------------------------------------------

def test_out_of_range_scores_are_clamped():
    bad_response = {
        dim: {"score": 99, "reasoning": "Extreme score."}
        for dim in DIMENSIONS["news_parse_agent"]
    }
    with _patch_client(json.dumps(bad_response)):
        result = judge("news_parse_agent", NEWS_PARSE_INPUTS, NEWS_PARSE_OUTPUT)

    for entry in result.values():
        assert 1 <= entry["score"] <= 5


# ---------------------------------------------------------------------------
# Behavior 9: unknown agent_name raises ValueError
# ---------------------------------------------------------------------------

def test_unknown_agent_name_raises():
    with pytest.raises(ValueError, match="Unknown agent_name"):
        judge("nonexistent_agent", {}, {})

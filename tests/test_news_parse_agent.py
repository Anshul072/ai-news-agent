import json
from unittest.mock import patch, MagicMock
import pytest

from agents.news_parse_agent import parse_articles


RAW_ARTICLE = {
    "id": 1,
    "url": "https://example.com/gpt5",
    "title": "GPT-5 Released",
    "content": "OpenAI announces GPT-5 with major improvements in reasoning.",
    "source_name": "AI News",
    "published_at": "2024-01-01T12:00:00",
}

GEMINI_RESPONSE = {
    "summary": "OpenAI released GPT-5.",
    "whats_new": "Chain-of-thought reasoning improved significantly.",
    "key_concepts": ["transformer", "RLHF", "scaling"],
    "concept_explanations": {
        "transformer": "Attention mechanism.",
        "RLHF": "Reward-based training.",
        "scaling": "More compute = better model.",
    },
    "who_made_it": "OpenAI",
    "use_cases": ["code generation", "summarization"],
    "importance_score": 9,
    "importance_reasoning": "Major capability jump.",
}


def _make_gemini_mock(response_dict: dict):
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_dict)
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    return mock_model


# ---------------------------------------------------------------------------
# Behavior 1: parse_articles returns enriched dicts for each article
# ---------------------------------------------------------------------------

def test_parse_articles_returns_enriched_for_each_article():
    mock_model = _make_gemini_mock(GEMINI_RESPONSE)
    with patch("agents.news_parse_agent.genai.GenerativeModel", return_value=mock_model):
        results = parse_articles([RAW_ARTICLE])

    assert len(results) == 1


# ---------------------------------------------------------------------------
# Behavior 2: enriched dict contains all required fields
# ---------------------------------------------------------------------------

def test_enriched_dict_has_all_required_fields():
    mock_model = _make_gemini_mock(GEMINI_RESPONSE)
    with patch("agents.news_parse_agent.genai.GenerativeModel", return_value=mock_model):
        results = parse_articles([RAW_ARTICLE])

    required = {
        "article_id", "summary", "whats_new", "key_concepts",
        "concept_explanations", "who_made_it", "use_cases",
        "importance_score", "importance_reasoning",
    }
    assert required.issubset(results[0].keys())


# ---------------------------------------------------------------------------
# Behavior 3: article_id is carried through from the raw article
# ---------------------------------------------------------------------------

def test_article_id_is_preserved():
    mock_model = _make_gemini_mock(GEMINI_RESPONSE)
    with patch("agents.news_parse_agent.genai.GenerativeModel", return_value=mock_model):
        results = parse_articles([RAW_ARTICLE])

    assert results[0]["article_id"] == RAW_ARTICLE["id"]


# ---------------------------------------------------------------------------
# Behavior 4: malformed JSON from Gemini skips the article (no crash)
# ---------------------------------------------------------------------------

def test_malformed_gemini_response_skipped():
    mock_response = MagicMock()
    mock_response.text = "not valid json {{{}"
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("agents.news_parse_agent.genai.GenerativeModel", return_value=mock_model):
        results = parse_articles([RAW_ARTICLE])

    assert results == []


# ---------------------------------------------------------------------------
# Behavior 5: multiple articles are all processed
# ---------------------------------------------------------------------------

def test_multiple_articles_all_processed():
    articles = [
        {**RAW_ARTICLE, "id": 1},
        {**RAW_ARTICLE, "id": 2, "url": "https://example.com/gemini"},
    ]
    mock_model = _make_gemini_mock(GEMINI_RESPONSE)
    with patch("agents.news_parse_agent.genai.GenerativeModel", return_value=mock_model):
        results = parse_articles(articles)

    assert len(results) == 2

import json
from unittest.mock import patch, MagicMock
import pytest

from agents.news_parse_agent import parse_articles

# Suppress the inter-request sleep for all tests in this module.
pytestmark = pytest.mark.usefixtures("_no_sleep")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("agents.news_parse_agent.time.sleep", lambda _: None)


RAW_ARTICLE = {
    "id": 1,
    "url": "https://example.com/gpt5",
    "title": "GPT-5 Released",
    "content": (
        "OpenAI announces GPT-5 with major improvements in reasoning and multi-step "
        "problem solving. The model scores state-of-the-art on several benchmarks including "
        "MMLU, HumanEval, and MATH. Researchers note a significant jump in chain-of-thought "
        "reasoning quality and a reduction in hallucination rate compared to GPT-4."
    ),
    "source_name": "AI News",
    "published_at": "2024-01-01T12:00:00",
}

GROQ_RESPONSE = {
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


def _make_mock_response(response_dict: dict):
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=json.dumps(response_dict)))]
    return mock


# ---------------------------------------------------------------------------
# Behavior 1: parse_articles returns enriched dicts for each article
# ---------------------------------------------------------------------------

def test_parse_articles_returns_enriched_for_each_article():
    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=_make_mock_response(GROQ_RESPONSE)):
        results = parse_articles([RAW_ARTICLE])

    assert len(results) == 1


# ---------------------------------------------------------------------------
# Behavior 2: enriched dict contains all required fields
# ---------------------------------------------------------------------------

def test_enriched_dict_has_all_required_fields():
    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=_make_mock_response(GROQ_RESPONSE)):
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
    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=_make_mock_response(GROQ_RESPONSE)):
        results = parse_articles([RAW_ARTICLE])

    assert results[0]["article_id"] == RAW_ARTICLE["id"]


# ---------------------------------------------------------------------------
# Behavior 4: malformed JSON from Groq skips the article (no crash)
# ---------------------------------------------------------------------------

def test_malformed_groq_response_skipped():
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content="not valid json {{{}"))]

    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=mock):
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
    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=_make_mock_response(GROQ_RESPONSE)):
        results = parse_articles(articles)

    assert len(results) == 2


# ---------------------------------------------------------------------------
# Behavior 6: truncated JSON is repaired and returned with partial data + defaults
# ---------------------------------------------------------------------------

def test_truncated_json_repaired_with_defaults():
    # json-repair closes unclosed strings/braces; missing fields get defaults
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content='{"summary": "partial...'))]

    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=mock):
        results = parse_articles([RAW_ARTICLE])

    assert len(results) == 1
    assert "partial" in results[0]["summary"]
    assert results[0]["key_concepts"] == []   # default for missing field


# ---------------------------------------------------------------------------
# Behavior 7: JSON array response — first element is used
# ---------------------------------------------------------------------------

def test_json_array_response_uses_first_element():
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=json.dumps([GROQ_RESPONSE])))]

    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=mock):
        results = parse_articles([RAW_ARTICLE])

    assert len(results) == 1
    assert results[0]["article_id"] == RAW_ARTICLE["id"]


# ---------------------------------------------------------------------------
# Behavior 8: missing required keys are filled with defaults
# ---------------------------------------------------------------------------

def test_missing_keys_filled_with_defaults():
    partial = {"summary": "Something happened.", "importance_score": 7}
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=json.dumps(partial)))]

    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=mock):
        results = parse_articles([RAW_ARTICLE])

    assert len(results) == 1
    result = results[0]
    assert result["key_concepts"] == []
    assert result["use_cases"] == []
    assert result["concept_explanations"] == {}
    assert result["whats_new"] == ""
    assert result["who_made_it"] == ""
    assert result["importance_reasoning"] == ""


# ---------------------------------------------------------------------------
# Behavior 9: wrong value types are coerced to the correct types
# ---------------------------------------------------------------------------

def test_wrong_types_are_coerced():
    bad_types = {
        **GROQ_RESPONSE,
        "key_concepts": "transformer, RLHF, scaling",   # str → list
        "use_cases": "code generation",                  # str → list
        "importance_score": "8",                         # str → int
        "concept_explanations": "not a dict",            # str → {}
    }
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=json.dumps(bad_types)))]

    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=mock):
        results = parse_articles([RAW_ARTICLE])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result["key_concepts"], list)
    assert isinstance(result["use_cases"], list)
    assert isinstance(result["importance_score"], int)
    assert result["importance_score"] == 8
    assert isinstance(result["concept_explanations"], dict)


# ---------------------------------------------------------------------------
# Behavior 10: importance_score is clamped to [1, 10]
# ---------------------------------------------------------------------------

def test_importance_score_clamped():
    for out_of_range, expected in [(-5, 1), (0, 1), (99, 10), (11, 10)]:
        response = {**GROQ_RESPONSE, "importance_score": out_of_range}
        mock = MagicMock()
        mock.choices = [MagicMock(message=MagicMock(content=json.dumps(response)))]

        with patch("agents.news_parse_agent._client.chat.completions.create",
                   return_value=mock):
            results = parse_articles([RAW_ARTICLE])

        assert results[0]["importance_score"] == expected, f"score {out_of_range} → expected {expected}"


# ---------------------------------------------------------------------------
# Behavior 11: prose wrapping a JSON object is handled via extraction fallback
# ---------------------------------------------------------------------------

def test_prose_wrapped_json_is_extracted():
    wrapped = "Sure, here is the JSON:\n" + json.dumps(GROQ_RESPONSE) + "\nHope that helps!"
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=wrapped))]

    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=mock):
        results = parse_articles([RAW_ARTICLE])

    assert len(results) == 1


# ---------------------------------------------------------------------------
# Behavior 12: unquoted string values are repaired and parsed successfully
# ---------------------------------------------------------------------------

def test_unquoted_string_values_are_repaired():
    # Simulate the real failure: LLM omits quotes around string values
    malformed = (
        '{\n'
        '  "summary": Elon Musk lost a lawsuit against OpenAI due to a unanimous verdict,\n'
        '  "whats_new": The court ruled against Musk on all claims,\n'
        '  "key_concepts": ["lawsuit", "OpenAI", "verdict"],\n'
        '  "concept_explanations": {"lawsuit": "Legal action."},\n'
        '  "who_made_it": OpenAI,\n'
        '  "use_cases": ["legal precedent"],\n'
        '  "importance_score": 8,\n'
        '  "importance_reasoning": Major ruling in AI governance\n'
        '}'
    )
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=malformed))]

    with patch("agents.news_parse_agent._client.chat.completions.create",
               return_value=mock):
        results = parse_articles([RAW_ARTICLE])

    assert len(results) == 1
    assert "Musk" in results[0]["summary"] or results[0]["summary"] != ""
    assert results[0]["importance_score"] == 8

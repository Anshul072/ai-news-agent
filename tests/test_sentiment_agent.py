import json
from unittest.mock import MagicMock, patch

import pytest

from storage.sqlite_store import SQLiteStore
from agents.sentiment_agent import run_sentiment


RAW_ARTICLE = {
    "url": "https://example.com/gpt5",
    "url_hash": "hash_gpt5",
    "title": "GPT-5 Released",
    "content": "OpenAI releases GPT-5 with major capability improvements.",
    "source_name": "AI News",
    "published_at": "2024-01-01T12:00:00",
    "fetched_at": "2024-01-01T13:00:00",
}

MOCK_THREADS = [
    {
        "subreddit": "artificial",
        "title": "GPT-5 is incredible",
        "score": 500,
        "num_comments": 200,
        "top_comments": ["This changes everything!", "Very impressive."],
    }
]

MOCK_SENTIMENT_RESPONSE = {
    "sentiment_label": "Positive",
    "sentiment_score": 0.85,
    "excitement_level": "Hyped",
    "top_concerns": ["Privacy", "Job displacement"],
    "top_use_cases": ["Code generation", "Research"],
    "notable_quotes": ["This changes everything!"],
    "subreddit_breakdown": {"artificial": "Very positive discussion"},
}


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    s.init_db()
    return s


@pytest.fixture
def article_id(store):
    store.insert_raw_article(RAW_ARTICLE)
    return store.get_raw_article_by_url_hash(RAW_ARTICLE["url_hash"])["id"]


def _make_model(*responses):
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = [
        MagicMock(text=r) for r in responses
    ]
    return mock_model


# ---------------------------------------------------------------------------
# Behavior 1: writes all sentiment fields to SQLite
# ---------------------------------------------------------------------------

def test_run_sentiment_writes_all_fields(store, article_id):
    mock_model = _make_model(
        '["GPT-5", "OpenAI"]',           # keyword extraction
        json.dumps(MOCK_SENTIMENT_RESPONSE),  # sentiment analysis
    )

    with patch("agents.sentiment_agent.genai.GenerativeModel", return_value=mock_model), \
         patch("agents.sentiment_agent.fetch_reddit_threads", return_value=MOCK_THREADS):
        run_sentiment(article_id, store)

    sentiment = store.get_sentiment(article_id)
    assert sentiment is not None
    assert sentiment["sentiment_label"] == "Positive"
    assert abs(sentiment["sentiment_score"] - 0.85) < 1e-9
    assert sentiment["excitement_level"] == "Hyped"
    assert isinstance(sentiment["top_concerns"], list)
    assert "Privacy" in sentiment["top_concerns"]
    assert isinstance(sentiment["notable_quotes"], list)
    assert sentiment["thread_count"] == 1
    assert sentiment["total_comments"] == 200


# ---------------------------------------------------------------------------
# Behavior 2: no threads found → neutral sentinel, no crash
# ---------------------------------------------------------------------------

def test_run_sentiment_no_threads_produces_neutral_record(store, article_id):
    mock_model = _make_model('["GPT-5"]')  # only keyword extraction needed

    with patch("agents.sentiment_agent.genai.GenerativeModel", return_value=mock_model), \
         patch("agents.sentiment_agent.fetch_reddit_threads", return_value=[]):
        run_sentiment(article_id, store)

    sentiment = store.get_sentiment(article_id)
    assert sentiment is not None
    assert sentiment["sentiment_label"] == "Neutral"
    assert sentiment["thread_count"] == 0
    assert sentiment["total_comments"] == 0


# ---------------------------------------------------------------------------
# Behavior 3: last_scanned_at updated on every scan
# ---------------------------------------------------------------------------

def test_run_sentiment_updates_last_scanned_at(store, article_id):
    with patch("agents.sentiment_agent.genai.GenerativeModel") as MockModel, \
         patch("agents.sentiment_agent.fetch_reddit_threads", return_value=MOCK_THREADS):
        MockModel.return_value = _make_model(
            '["GPT-5", "OpenAI"]', json.dumps(MOCK_SENTIMENT_RESPONSE),
            '["GPT-5", "OpenAI"]', json.dumps(MOCK_SENTIMENT_RESPONSE),
        )
        run_sentiment(article_id, store)
        first_ts = store.get_sentiment(article_id)["last_scanned_at"]

        MockModel.return_value = _make_model(
            '["GPT-5", "OpenAI"]', json.dumps(MOCK_SENTIMENT_RESPONSE),
        )
        run_sentiment(article_id, store)
        second_ts = store.get_sentiment(article_id)["last_scanned_at"]

    assert first_ts is not None
    assert second_ts is not None
    assert "T" in first_ts


# ---------------------------------------------------------------------------
# Behavior 4: subreddit_breakdown and top_use_cases are lists/dicts
# ---------------------------------------------------------------------------

def test_run_sentiment_json_fields_deserialize_correctly(store, article_id):
    mock_model = _make_model(
        '["GPT-5"]',
        json.dumps(MOCK_SENTIMENT_RESPONSE),
    )

    with patch("agents.sentiment_agent.genai.GenerativeModel", return_value=mock_model), \
         patch("agents.sentiment_agent.fetch_reddit_threads", return_value=MOCK_THREADS):
        run_sentiment(article_id, store)

    sentiment = store.get_sentiment(article_id)
    assert isinstance(sentiment["top_use_cases"], list)
    assert isinstance(sentiment["subreddit_breakdown"], dict)

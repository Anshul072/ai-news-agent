import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import chromadb
import pytest

from storage.sqlite_store import SQLiteStore
from storage.chroma_store import ChromaStore
from pipelines.sentiment_pipeline import run_sentiment_pipeline


def _now_minus(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


RECENT_ARTICLE_1 = {
    "url": "https://example.com/recent1",
    "url_hash": "hash_recent1",
    "title": "GPT-5 Released",
    "content": "OpenAI releases GPT-5.",
    "source_name": "AI News",
    "published_at": _now_minus(2),
    "fetched_at": _now_minus(2),
}

RECENT_ARTICLE_2 = {
    "url": "https://example.com/recent2",
    "url_hash": "hash_recent2",
    "title": "Claude 4 Announced",
    "content": "Anthropic announces Claude 4.",
    "source_name": "Tech News",
    "published_at": _now_minus(3),
    "fetched_at": _now_minus(3),
}

OLD_ARTICLE = {
    "url": "https://example.com/old",
    "url_hash": "hash_old",
    "title": "Old AI News",
    "content": "Some old news.",
    "source_name": "AI News",
    "published_at": _now_minus(10),
    "fetched_at": _now_minus(10),
}

ENRICHED = {
    "summary": "GPT-5 is OpenAI's latest model.",
    "whats_new": "New reasoning capabilities.",
    "key_concepts": ["transformer"],
    "concept_explanations": {"transformer": "Attention model."},
    "who_made_it": "OpenAI",
    "use_cases": ["coding"],
    "importance_score": 9,
    "importance_reasoning": "Major capability jump.",
}

MOCK_SENTIMENT = {
    "sentiment_label": "Positive",
    "sentiment_score": 0.85,
    "excitement_level": "Hyped",
    "top_concerns": ["Privacy"],
    "top_use_cases": ["Code generation"],
    "notable_quotes": ["This changes everything!"],
    "subreddit_breakdown": {"artificial": "Very positive"},
    "thread_count": 3,
    "total_comments": 150,
    "last_scanned_at": datetime.now(timezone.utc).isoformat(),
}


def _fake_embed(text: str) -> list[float]:
    return [abs(hash(text) % 1000) / 1000.0 + 0.001] * 768


def _run_sentiment_side_effect(article_id, store):
    """Simulates run_sentiment: writes to SQLite and returns sentiment dict."""
    store.upsert_sentiment(article_id, MOCK_SENTIMENT)
    return MOCK_SENTIMENT


@pytest.fixture
def sqlite_store():
    s = SQLiteStore(":memory:")
    s.init_db()
    return s


@pytest.fixture
def chroma_store():
    client = chromadb.EphemeralClient()
    return ChromaStore(client=client, collection_name=f"test_{uuid.uuid4().hex}")


def _seed_article(sqlite_store, article_dict, enriched_dict, story_group_id=None):
    sqlite_store.insert_raw_article(article_dict)
    db_article = sqlite_store.get_raw_article_by_url_hash(article_dict["url_hash"])
    article_id = db_article["id"]
    sqlite_store.insert_enriched_article(article_id, enriched_dict, story_group_id)
    return article_id


# ---------------------------------------------------------------------------
# Behavior 1: SQLite sentiment fields updated for articles in the window
# ---------------------------------------------------------------------------

def test_pipeline_updates_sqlite_sentiment_for_recent_articles(sqlite_store, chroma_store):
    article_id = _seed_article(sqlite_store, RECENT_ARTICLE_1, ENRICHED)

    with patch("pipelines.sentiment_pipeline.run_sentiment", side_effect=_run_sentiment_side_effect), \
         patch("pipelines.sentiment_pipeline.embed", side_effect=_fake_embed):
        run_sentiment_pipeline(sqlite_store, chroma_store)

    sentiment = sqlite_store.get_sentiment(article_id)
    assert sentiment is not None
    assert sentiment["sentiment_label"] == "Positive"
    assert abs(sentiment["sentiment_score"] - 0.85) < 1e-9
    assert sentiment["thread_count"] == 3
    assert sentiment["total_comments"] == 150


# ---------------------------------------------------------------------------
# Behavior 2: ChromaDB sentiment chunk upserted after pipeline run
# ---------------------------------------------------------------------------

def test_pipeline_upserts_chroma_sentiment_chunk(sqlite_store, chroma_store):
    article_id = _seed_article(sqlite_store, RECENT_ARTICLE_1, ENRICHED)

    with patch("pipelines.sentiment_pipeline.run_sentiment", side_effect=_run_sentiment_side_effect), \
         patch("pipelines.sentiment_pipeline.embed", side_effect=_fake_embed):
        run_sentiment_pipeline(sqlite_store, chroma_store)

    results = chroma_store._collection.get(
        where={"field": "sentiment"},
        include=["metadatas"],
    )
    assert len(results["ids"]) > 0
    assert any(m["article_id"] == str(article_id) for m in results["metadatas"])


# ---------------------------------------------------------------------------
# Behavior 3: articles outside the 7-day window are not re-scanned
# ---------------------------------------------------------------------------

def test_pipeline_skips_articles_outside_window(sqlite_store, chroma_store):
    old_id = _seed_article(sqlite_store, OLD_ARTICLE, ENRICHED)

    with patch("pipelines.sentiment_pipeline.run_sentiment") as mock_run:
        run_sentiment_pipeline(sqlite_store, chroma_store)

    mock_run.assert_not_called()
    assert sqlite_store.get_sentiment(old_id) is None


# ---------------------------------------------------------------------------
# Behavior 4: failed scan on one article does not abort the pipeline
# ---------------------------------------------------------------------------

def test_pipeline_continues_after_single_article_failure(sqlite_store, chroma_store):
    id1 = _seed_article(sqlite_store, RECENT_ARTICLE_1, ENRICHED)
    id2 = _seed_article(sqlite_store, RECENT_ARTICLE_2, ENRICHED)

    def _fail_first(article_id, store):
        if article_id == id1:
            raise RuntimeError("simulated failure")
        store.upsert_sentiment(article_id, MOCK_SENTIMENT)
        return MOCK_SENTIMENT

    with patch("pipelines.sentiment_pipeline.run_sentiment", side_effect=_fail_first), \
         patch("pipelines.sentiment_pipeline.embed", side_effect=_fake_embed):
        run_sentiment_pipeline(sqlite_store, chroma_store)

    assert sqlite_store.get_sentiment(id1) is None
    assert sqlite_store.get_sentiment(id2) is not None


# ---------------------------------------------------------------------------
# Behavior 5: second run updates (upserts) the existing ChromaDB sentiment chunk
# ---------------------------------------------------------------------------

def test_pipeline_reruns_upsert_chroma_no_duplicates(sqlite_store, chroma_store):
    article_id = _seed_article(sqlite_store, RECENT_ARTICLE_1, ENRICHED)

    with patch("pipelines.sentiment_pipeline.run_sentiment", side_effect=_run_sentiment_side_effect), \
         patch("pipelines.sentiment_pipeline.embed", side_effect=_fake_embed):
        run_sentiment_pipeline(sqlite_store, chroma_store)
        run_sentiment_pipeline(sqlite_store, chroma_store)

    results = chroma_store._collection.get(
        where={"field": "sentiment"},
        include=["metadatas"],
    )
    article_chunks = [m for m in results["metadatas"] if m["article_id"] == str(article_id)]
    assert len(article_chunks) == 1

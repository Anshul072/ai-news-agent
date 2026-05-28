import pytest
from storage.sqlite_store import SQLiteStore


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    s.init_db()
    return s


ARTICLE = {
    "url": "https://example.com/article1",
    "url_hash": "abc123",
    "title": "Test Article",
    "content": "Some content here.",
    "source_name": "Example Feed",
    "published_at": "2024-01-01T12:00:00",
    "fetched_at": "2024-01-01T13:00:00",
}


# ---------------------------------------------------------------------------
# Behavior 1: init_db creates the schema (no error)
# ---------------------------------------------------------------------------

def test_init_db_creates_schema():
    s = SQLiteStore(":memory:")
    s.init_db()  # must not raise


# ---------------------------------------------------------------------------
# Behavior 2: insert_raw_article stores the article
# ---------------------------------------------------------------------------

def test_insert_raw_article_stores_article(store):
    store.insert_raw_article(ARTICLE)
    result = store.get_raw_article_by_url_hash(ARTICLE["url_hash"])
    assert result is not None
    assert result["title"] == "Test Article"
    assert result["url"] == "https://example.com/article1"


# ---------------------------------------------------------------------------
# Behavior 3: url_hash_exists checks dedup table
# ---------------------------------------------------------------------------

def test_url_hash_exists_returns_false_before_insert(store):
    assert store.url_hash_exists(ARTICLE["url_hash"]) is False


def test_url_hash_exists_returns_true_after_insert(store):
    store.insert_raw_article(ARTICLE)
    assert store.url_hash_exists(ARTICLE["url_hash"]) is True


# ---------------------------------------------------------------------------
# Behavior 4: inserting same article twice is idempotent
# ---------------------------------------------------------------------------

def test_insert_is_idempotent(store):
    store.insert_raw_article(ARTICLE)
    store.insert_raw_article(ARTICLE)  # second insert ignored
    articles = store.get_all_raw_articles()
    assert len(articles) == 1


# ---------------------------------------------------------------------------
# Behavior 5: multiple distinct articles are all stored
# ---------------------------------------------------------------------------

def test_multiple_articles_stored(store):
    a2 = {**ARTICLE, "url": "https://example.com/article2", "url_hash": "def456", "title": "Article 2"}
    store.insert_raw_article(ARTICLE)
    store.insert_raw_article(a2)
    articles = store.get_all_raw_articles()
    assert len(articles) == 2


# ---------------------------------------------------------------------------
# Behavior 6: sentiment_history accumulates one row per article per day
# ---------------------------------------------------------------------------

def _insert_article_and_get_id(store) -> int:
    store.insert_raw_article(ARTICLE)
    return store.get_raw_article_by_url_hash(ARTICLE["url_hash"])["id"]


SENTIMENT = {
    "sentiment_label": "Positive",
    "sentiment_score": 0.75,
    "excitement_level": "Hyped",
    "top_concerns": [],
    "top_use_cases": [],
    "notable_quotes": [],
    "subreddit_breakdown": {},
    "thread_count": 3,
    "total_comments": 42,
    "last_scanned_at": "2024-06-01T08:00:00+00:00",
}


def test_sentiment_history_created_on_upsert(store):
    article_id = _insert_article_and_get_id(store)
    store.upsert_sentiment(article_id, SENTIMENT)
    history = store.get_all_sentiment_history()
    assert article_id in history
    rows = history[article_id]
    assert len(rows) == 1
    assert rows[0]["scan_date"] == "2024-06-01"
    assert rows[0]["sentiment_label"] == "Positive"
    assert rows[0]["sentiment_score"] == pytest.approx(0.75)
    assert rows[0]["thread_count"] == 3


def test_sentiment_history_deduplicates_same_day(store):
    article_id = _insert_article_and_get_id(store)
    store.upsert_sentiment(article_id, SENTIMENT)
    later_same_day = {**SENTIMENT, "sentiment_score": 0.5, "last_scanned_at": "2024-06-01T20:00:00+00:00"}
    store.upsert_sentiment(article_id, later_same_day)
    rows = store.get_all_sentiment_history()[article_id]
    assert len(rows) == 1
    assert rows[0]["sentiment_score"] == pytest.approx(0.5)


def test_sentiment_history_accumulates_across_days(store):
    article_id = _insert_article_and_get_id(store)
    for day, score in [("2024-06-01", 0.4), ("2024-06-02", 0.6), ("2024-06-03", 0.8)]:
        store.upsert_sentiment(article_id, {**SENTIMENT, "sentiment_score": score, "last_scanned_at": f"{day}T08:00:00+00:00"})
    rows = store.get_all_sentiment_history()[article_id]
    assert len(rows) == 3
    assert [r["scan_date"] for r in rows] == ["2024-06-01", "2024-06-02", "2024-06-03"]
    assert rows[2]["sentiment_score"] == pytest.approx(0.8)

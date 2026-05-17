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

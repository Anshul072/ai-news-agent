import pytest
from storage.sqlite_store import SQLiteStore


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    s.init_db()
    return s


RAW = {
    "url": "https://example.com/a1",
    "url_hash": "hash1",
    "title": "GPT-5 Released",
    "content": "OpenAI releases GPT-5.",
    "source_name": "AI News",
    "published_at": "2024-01-01T12:00:00",
    "fetched_at": "2024-01-01T13:00:00",
}

ENRICHED = {
    "summary": "GPT-5 is here.",
    "whats_new": "Improved reasoning via chain-of-thought.",
    "key_concepts": ["transformer", "RLHF"],
    "concept_explanations": {"transformer": "Attention-based neural net.", "RLHF": "Reward learning."},
    "who_made_it": "OpenAI",
    "use_cases": ["coding assistant", "summarization"],
    "importance_score": 9,
    "importance_reasoning": "Major capability jump.",
}


# ---------------------------------------------------------------------------
# Behavior 1: get_unenriched_articles returns only unprocessed raw articles
# ---------------------------------------------------------------------------

def test_get_unenriched_articles_returns_all_before_enrichment(store):
    store.insert_raw_article(RAW)
    unenriched = store.get_unenriched_articles()
    assert len(unenriched) == 1
    assert unenriched[0]["url"] == RAW["url"]


def test_get_unenriched_articles_excludes_already_enriched(store):
    store.insert_raw_article(RAW)
    article_id = store.get_raw_article_by_url_hash(RAW["url_hash"])["id"]
    store.insert_enriched_article(article_id, ENRICHED)
    unenriched = store.get_unenriched_articles()
    assert unenriched == []


# ---------------------------------------------------------------------------
# Behavior 2: insert_enriched_article stores all enriched fields
# ---------------------------------------------------------------------------

def test_insert_enriched_article_stores_fields(store):
    store.insert_raw_article(RAW)
    article_id = store.get_raw_article_by_url_hash(RAW["url_hash"])["id"]
    store.insert_enriched_article(article_id, ENRICHED)
    result = store.get_enriched_article(article_id)
    assert result is not None
    assert result["summary"] == "GPT-5 is here."
    assert result["importance_score"] == 9
    assert result["who_made_it"] == "OpenAI"


# ---------------------------------------------------------------------------
# Behavior 3: list fields (key_concepts, use_cases) round-trip as lists
# ---------------------------------------------------------------------------

def test_list_fields_round_trip(store):
    store.insert_raw_article(RAW)
    article_id = store.get_raw_article_by_url_hash(RAW["url_hash"])["id"]
    store.insert_enriched_article(article_id, ENRICHED)
    result = store.get_enriched_article(article_id)
    assert isinstance(result["key_concepts"], list)
    assert "transformer" in result["key_concepts"]
    assert isinstance(result["use_cases"], list)

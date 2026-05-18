import json
import logging
import uuid
from unittest.mock import MagicMock, patch

import chromadb
import pytest

import config
from storage.sqlite_store import SQLiteStore
from storage.chroma_store import ChromaStore
from pipelines.news_pipeline import run_news_pipeline


FIXTURE_ARTICLES = [
    {
        "url": "https://example.com/article1",
        "url_hash": "hash1",
        "title": "GPT-5 Released",
        "content": "OpenAI releases GPT-5.",
        "source_name": "AI News",
        "published_at": "2024-01-01T12:00:00",
        "fetched_at": "2024-01-01T13:00:00",
    },
    {
        "url": "https://example.com/article2",
        "url_hash": "hash2",
        "title": "Claude 4 Announced",
        "content": "Anthropic announces Claude 4.",
        "source_name": "Tech News",
        "published_at": "2024-01-02T12:00:00",
        "fetched_at": "2024-01-02T13:00:00",
    },
]

ENRICHED_TEMPLATE = [
    {
        "summary": "GPT-5 is OpenAI's latest model.",
        "whats_new": "New reasoning capabilities.",
        "key_concepts": ["transformer", "RLHF"],
        "concept_explanations": {"transformer": "Attention model.", "RLHF": "Reward learning."},
        "who_made_it": "OpenAI",
        "use_cases": ["coding", "summarization"],
        "importance_score": 9,
        "importance_reasoning": "Major capability jump.",
    },
    {
        "summary": "Claude 4 is Anthropic's latest.",
        "whats_new": "Constitutional AI improvements.",
        "key_concepts": ["constitutional AI"],
        "concept_explanations": {"constitutional AI": "Value-aligned training."},
        "who_made_it": "Anthropic",
        "use_cases": ["safety research"],
        "importance_score": 8,
        "importance_reasoning": "Safety improvements.",
    },
]


@pytest.fixture
def sqlite_store():
    s = SQLiteStore(":memory:")
    s.init_db()
    return s


@pytest.fixture
def chroma_store():
    client = chromadb.EphemeralClient()
    return ChromaStore(client=client, collection_name=f"test_{uuid.uuid4().hex}")


def _fake_embed(text: str) -> list[float]:
    # Deterministic non-zero vector so cosine similarity works
    return [abs(hash(text) % 1000) / 1000.0 + 0.001] * 768


def _make_parse_mock(templates):
    """Returns a side_effect function that injects article_id from the passed articles."""
    def _parse(articles):
        return [
            {**templates[i], "article_id": articles[i]["id"]}
            for i in range(min(len(articles), len(templates)))
        ]
    return _parse


# ---------------------------------------------------------------------------
# Behavior 1: pipeline stores enriched articles in SQLite
# ---------------------------------------------------------------------------

def test_pipeline_stores_enriched_articles_in_sqlite(sqlite_store, chroma_store):
    with patch("pipelines.news_pipeline.fetch_articles", return_value=FIXTURE_ARTICLES), \
         patch("pipelines.news_pipeline._parse_articles", side_effect=_make_parse_mock(ENRICHED_TEMPLATE)), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.9), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        run_news_pipeline(["https://fake.feed"], sqlite_store, chroma_store)

    articles = sqlite_store.get_all_raw_articles()
    assert len(articles) == 2
    for a in articles:
        enriched = sqlite_store.get_enriched_article(a["id"])
        assert enriched is not None
        assert enriched["summary"] is not None
        assert enriched["importance_score"] is not None


# ---------------------------------------------------------------------------
# Behavior 2: ChromaDB contains field chunks with correct article_id metadata
# ---------------------------------------------------------------------------

def test_pipeline_stores_field_chunks_in_chromadb(sqlite_store, chroma_store):
    with patch("pipelines.news_pipeline.fetch_articles", return_value=[FIXTURE_ARTICLES[0]]), \
         patch("pipelines.news_pipeline._parse_articles", side_effect=_make_parse_mock(ENRICHED_TEMPLATE)), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.9), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        run_news_pipeline(["https://fake.feed"], sqlite_store, chroma_store)

    raw = sqlite_store.get_all_raw_articles()
    article_id = str(raw[0]["id"])

    results = chroma_store._collection.get(include=["metadatas"])
    assert len(results["ids"]) > 0
    assert any(m["article_id"] == article_id for m in results["metadatas"])


# ---------------------------------------------------------------------------
# Behavior 3: failed parse on one article does not abort the pipeline
# ---------------------------------------------------------------------------

def test_pipeline_continues_on_parse_failure(sqlite_store, chroma_store):
    def _partial_parse(articles):
        # Succeed on first article, fail on second
        result = [{**ENRICHED_TEMPLATE[0], "article_id": articles[0]["id"]}]
        if len(articles) > 1:
            raise ValueError("simulated parse failure")
        return result

    with patch("pipelines.news_pipeline.fetch_articles", return_value=FIXTURE_ARTICLES), \
         patch("pipelines.news_pipeline._parse_articles", side_effect=_partial_parse), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.9), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        run_news_pipeline(["https://fake.feed"], sqlite_store, chroma_store)

    enriched_articles = [
        sqlite_store.get_enriched_article(a["id"])
        for a in sqlite_store.get_all_raw_articles()
    ]
    assert any(e is not None for e in enriched_articles)


# ---------------------------------------------------------------------------
# Behavior 4: running the pipeline twice does not duplicate records
# ---------------------------------------------------------------------------

def test_pipeline_dedup_prevents_duplicate_records(sqlite_store, chroma_store):
    common_kwargs = dict(
        feed_urls=["https://fake.feed"],
        sqlite_store=sqlite_store,
        chroma_store=chroma_store,
    )

    with patch("pipelines.news_pipeline.fetch_articles", return_value=[FIXTURE_ARTICLES[0]]), \
         patch("pipelines.news_pipeline._parse_articles", side_effect=_make_parse_mock(ENRICHED_TEMPLATE)), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.9), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        run_news_pipeline(**common_kwargs)

    with patch("pipelines.news_pipeline.fetch_articles", return_value=[FIXTURE_ARTICLES[0]]), \
         patch("pipelines.news_pipeline._parse_articles", side_effect=_make_parse_mock(ENRICHED_TEMPLATE)), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.9), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        run_news_pipeline(**common_kwargs)

    assert len(sqlite_store.get_all_raw_articles()) == 1


# ---------------------------------------------------------------------------
# Behavior 5: filtered article is saved to raw_articles but not enriched
# ---------------------------------------------------------------------------

def test_filtered_article_in_raw_not_in_enriched(sqlite_store, chroma_store):
    with patch("pipelines.news_pipeline.fetch_articles", return_value=[FIXTURE_ARTICLES[0]]), \
         patch("pipelines.news_pipeline._parse_articles", side_effect=_make_parse_mock(ENRICHED_TEMPLATE)), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.1), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        run_news_pipeline(["https://fake.feed"], sqlite_store, chroma_store)

    raw = sqlite_store.get_all_raw_articles()
    assert len(raw) == 1
    assert sqlite_store.get_enriched_article(raw[0]["id"]) is None


# ---------------------------------------------------------------------------
# Behavior 6: filtered article produces a WARNING log with title and score
# ---------------------------------------------------------------------------

def test_filtered_article_produces_warning_log(sqlite_store, chroma_store, caplog):
    with patch("pipelines.news_pipeline.fetch_articles", return_value=[FIXTURE_ARTICLES[0]]), \
         patch("pipelines.news_pipeline._parse_articles", side_effect=_make_parse_mock(ENRICHED_TEMPLATE)), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.1), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        with caplog.at_level(logging.WARNING, logger="pipelines.news_pipeline"):
            run_news_pipeline(["https://fake.feed"], sqlite_store, chroma_store)

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(FIXTURE_ARTICLES[0]["title"] in m for m in warning_messages)
    assert any("0.100" in m for m in warning_messages)


# ---------------------------------------------------------------------------
# Behavior 7: passing article produces an INFO log with title and score
# ---------------------------------------------------------------------------

def test_passing_article_produces_info_log(sqlite_store, chroma_store, caplog):
    with patch("pipelines.news_pipeline.fetch_articles", return_value=[FIXTURE_ARTICLES[0]]), \
         patch("pipelines.news_pipeline._parse_articles", side_effect=_make_parse_mock(ENRICHED_TEMPLATE)), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.9), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        with caplog.at_level(logging.INFO, logger="pipelines.news_pipeline"):
            run_news_pipeline(["https://fake.feed"], sqlite_store, chroma_store)

    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any(FIXTURE_ARTICLES[0]["title"] in m for m in info_messages)
    assert any("0.900" in m for m in info_messages)


# ---------------------------------------------------------------------------
# Behavior 8: Groq LLM (_parse_articles) never called for filtered article
# ---------------------------------------------------------------------------

def test_groq_never_called_for_filtered_article(sqlite_store, chroma_store):
    mock_parse = MagicMock(return_value=[])
    with patch("pipelines.news_pipeline.fetch_articles", return_value=[FIXTURE_ARTICLES[0]]), \
         patch("pipelines.news_pipeline._parse_articles", mock_parse), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.1), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        run_news_pipeline(["https://fake.feed"], sqlite_store, chroma_store)

    mock_parse.assert_not_called()


# ---------------------------------------------------------------------------
# Behavior 9: ChromaDB receives no chunks for a filtered article
# ---------------------------------------------------------------------------

def test_chroma_gets_no_chunks_for_filtered_article(sqlite_store, chroma_store):
    with patch("pipelines.news_pipeline.fetch_articles", return_value=[FIXTURE_ARTICLES[0]]), \
         patch("pipelines.news_pipeline._parse_articles", side_effect=_make_parse_mock(ENRICHED_TEMPLATE)), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.1), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        run_news_pipeline(["https://fake.feed"], sqlite_store, chroma_store)

    results = chroma_store._collection.get(include=["metadatas"])
    assert len(results["ids"]) == 0


# ---------------------------------------------------------------------------
# Behavior 10: ARTICLE_FILTER_THRESHOLD=0.0 causes all articles to pass
# ---------------------------------------------------------------------------

def test_threshold_zero_all_articles_pass(sqlite_store, chroma_store, monkeypatch):
    monkeypatch.setattr(config, "ARTICLE_FILTER_THRESHOLD", 0.0)
    with patch("pipelines.news_pipeline.fetch_articles", return_value=FIXTURE_ARTICLES), \
         patch("pipelines.news_pipeline._parse_articles", side_effect=_make_parse_mock(ENRICHED_TEMPLATE)), \
         patch("pipelines.news_pipeline._relevance_score", return_value=0.0), \
         patch("pipelines.news_pipeline.embed", side_effect=_fake_embed):
        run_news_pipeline(["https://fake.feed"], sqlite_store, chroma_store)

    raw = sqlite_store.get_all_raw_articles()
    assert len(raw) == 2
    for a in raw:
        assert sqlite_store.get_enriched_article(a["id"]) is not None

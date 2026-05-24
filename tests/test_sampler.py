import json
import os
import tempfile
from datetime import datetime, timezone, timedelta

import pytest

from storage.sqlite_store import SQLiteStore
from eval.sampler import sampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_minus(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _insert_article(store: SQLiteStore, n: int, importance: int) -> int:
    article = {
        "url": f"https://example.com/article{n}",
        "url_hash": f"hash_{n}",
        "title": f"Article {n}",
        "content": f"Detailed content for article {n} about AI developments.",
        "source_name": "Test Source",
        "published_at": _now_minus(5),
        "fetched_at": _now_minus(5),
    }
    store.insert_raw_article(article)
    raw = store.get_raw_article_by_url_hash(article["url_hash"])
    store.insert_enriched_article(
        raw["id"],
        {
            "summary": f"Summary for article {n}",
            "whats_new": "Something new",
            "key_concepts": ["concept"],
            "concept_explanations": {},
            "who_made_it": "Test Corp",
            "use_cases": ["testing"],
            "importance_score": importance,
            "importance_reasoning": "Test reasoning",
        },
    )
    return raw["id"]


def _write_fixture(directory: str, filename: str, data: dict) -> None:
    with open(os.path.join(directory, filename), "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    s.init_db()
    return s


@pytest.fixture
def store_with_articles(store):
    # 2 low, 4 medium, 4 high — enough for all stratification buckets
    importances = [2, 3, 4, 5, 6, 6, 7, 8, 9, 10]
    for i, imp in enumerate(importances):
        _insert_article(store, i, imp)
    return store


@pytest.fixture
def golden_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        for sub in ("news_parse", "sentiment", "rag"):
            os.makedirs(os.path.join(tmpdir, sub))

        # 3 news_parse fixtures
        for i in range(3):
            _write_fixture(
                os.path.join(tmpdir, "news_parse"),
                f"article_{i}.json",
                {
                    "agent_name": "news_parse_agent",
                    "article_id": None,
                    "inputs": {"article_title": f"Golden {i}", "article_content": f"Content {i}"},
                    "output": {"summary": f"Summary {i}", "importance_score": 7},
                },
            )

        # 2 rag fixtures
        for i in range(2):
            _write_fixture(
                os.path.join(tmpdir, "rag"),
                f"rag_{i}.json",
                {
                    "agent_name": "rag_agent",
                    "article_id": None,
                    "inputs": {"query": f"Query {i}", "context_chunks": f"Context {i}"},
                    "output": {"answer": f"Answer {i}", "citations": []},
                },
            )

        yield tmpdir


# ---------------------------------------------------------------------------
# Behavior 1: returns exactly n_golden + n_recent samples
# ---------------------------------------------------------------------------

def test_returns_exact_count(store_with_articles, golden_dir):
    samples = sampler(store_with_articles, golden_dir, n_golden=5, n_recent=5)
    assert len(samples) == 10


def test_returns_exact_count_with_custom_params(store_with_articles, golden_dir):
    samples = sampler(store_with_articles, golden_dir, n_golden=3, n_recent=2)
    assert len(samples) == 5


def test_returns_exact_count_n_recent_zero(store_with_articles, golden_dir):
    samples = sampler(store_with_articles, golden_dir, n_golden=5, n_recent=0)
    assert len(samples) == 5


# ---------------------------------------------------------------------------
# Behavior 2: golden fixtures are loaded and marked correctly
# ---------------------------------------------------------------------------

def test_golden_fixtures_are_loaded(store_with_articles, golden_dir):
    samples = sampler(store_with_articles, golden_dir, n_golden=5, n_recent=5)
    golden = [s for s in samples if s.get("_source") == "golden"]
    assert len(golden) == 5


def test_golden_fixture_fields_are_present(store_with_articles, golden_dir):
    samples = sampler(store_with_articles, golden_dir, n_golden=3, n_recent=0)
    for s in samples:
        assert "agent_name" in s
        assert "inputs" in s
        assert "output" in s


# ---------------------------------------------------------------------------
# Behavior 3: RAG golden samples load as query-answer pairs
# ---------------------------------------------------------------------------

def test_rag_golden_samples_have_query_and_answer(store_with_articles, golden_dir):
    samples = sampler(store_with_articles, golden_dir, n_golden=5, n_recent=0)
    rag_samples = [s for s in samples if s.get("agent_name") == "rag_agent"]
    assert len(rag_samples) >= 1
    for s in rag_samples:
        assert "query" in s["inputs"]
        assert "answer" in s["output"]


# ---------------------------------------------------------------------------
# Behavior 4: recent samples cover low, medium, and high importance buckets
# ---------------------------------------------------------------------------

def test_stratified_sample_covers_all_buckets(store_with_articles, golden_dir):
    samples = sampler(store_with_articles, golden_dir, n_golden=0, n_recent=5)
    recent = [s for s in samples if s.get("_source") == "recent"]

    scores = [s.get("importance_score") for s in recent]

    has_low = any(1 <= (sc or 0) <= 3 for sc in scores)
    has_medium = any(4 <= (sc or 0) <= 6 for sc in scores)
    has_high = any(7 <= (sc or 0) <= 10 for sc in scores)

    assert has_low, "No low-importance article sampled"
    assert has_medium, "No medium-importance article sampled"
    assert has_high, "No high-importance article sampled"


# ---------------------------------------------------------------------------
# Behavior 5: no article appears in both golden set and recent set
# ---------------------------------------------------------------------------

def test_no_overlap_between_golden_and_recent(store, golden_dir):
    # Insert an article and also reference it in a golden fixture
    article_id = _insert_article(store, 99, 8)
    _write_fixture(
        os.path.join(golden_dir, "news_parse"),
        "overlap.json",
        {
            "agent_name": "news_parse_agent",
            "article_id": article_id,
            "inputs": {"article_title": "Overlap Article", "article_content": "Content"},
            "output": {"summary": "Summary", "importance_score": 8},
        },
    )
    # Insert more articles so n_recent can be filled
    for i in range(10):
        _insert_article(store, i, 5 + (i % 5))

    samples = sampler(store, golden_dir, n_golden=5, n_recent=5)

    golden_ids = {s.get("article_id") for s in samples if s.get("_source") == "golden" and s.get("article_id") is not None}
    recent_ids = {s.get("article_id") for s in samples if s.get("_source") == "recent" and s.get("article_id") is not None}

    assert not (golden_ids & recent_ids), f"Overlap found: {golden_ids & recent_ids}"


# ---------------------------------------------------------------------------
# Behavior 6: empty SQLite store returns only golden samples
# ---------------------------------------------------------------------------

def test_empty_store_returns_only_golden(store, golden_dir):
    samples = sampler(store, golden_dir, n_golden=3, n_recent=5)
    # Store has no articles so n_recent will be 0
    assert all(s.get("_source") == "golden" for s in samples)
    assert len(samples) == 3


# ---------------------------------------------------------------------------
# Behavior 7: empty golden dir returns only recent samples
# ---------------------------------------------------------------------------

def test_empty_golden_dir_returns_only_recent(store_with_articles):
    with tempfile.TemporaryDirectory() as empty_dir:
        for sub in ("news_parse", "sentiment", "rag"):
            os.makedirs(os.path.join(empty_dir, sub))

        samples = sampler(store_with_articles, empty_dir, n_golden=5, n_recent=5)
        assert all(s.get("_source") == "recent" for s in samples)
        assert len(samples) == 5

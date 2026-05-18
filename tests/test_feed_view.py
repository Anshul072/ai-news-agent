import pytest
from storage.sqlite_store import SQLiteStore


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    s.init_db()
    return s


def _insert_article(store, url_hash, title, source_name, published_at, importance_score, story_group_id=None):
    article = {
        "url": f"https://example.com/{url_hash}",
        "url_hash": url_hash,
        "title": title,
        "content": "content",
        "source_name": source_name,
        "published_at": published_at,
        "fetched_at": published_at,
    }
    store.insert_raw_article(article)
    raw = store.get_raw_article_by_url_hash(url_hash)
    article_id = raw["id"]

    if story_group_id is None:
        story_group_id = store.create_story_group()
    else:
        store.increment_source_count(story_group_id)

    enriched = {
        "summary": f"Summary of {title}",
        "whats_new": f"What's new in {title}",
        "key_concepts": ["concept1"],
        "concept_explanations": {"concept1": "explanation"},
        "who_made_it": "Some Org",
        "use_cases": ["use case 1"],
        "importance_score": importance_score,
        "importance_reasoning": "Reasoning",
    }
    store.insert_enriched_article(article_id, enriched, story_group_id)
    return article_id, story_group_id


# ---------------------------------------------------------------------------
# Behavior 1: empty store returns empty list
# ---------------------------------------------------------------------------

def test_get_story_clusters_empty(store):
    assert store.get_story_clusters() == []


# ---------------------------------------------------------------------------
# Behavior 2: single article cluster has correct card fields
# ---------------------------------------------------------------------------

def test_get_story_clusters_single_cluster(store):
    _insert_article(store, "hash1", "Title 1", "Source A", "2024-01-01T12:00:00", 7)
    clusters = store.get_story_clusters()
    assert len(clusters) == 1
    c = clusters[0]
    assert c["title"] == "Title 1"
    assert c["importance_score"] == 7
    assert c["source_count"] == 1
    assert c["source_names"] == ["Source A"]
    assert c["published_at_min"] == "2024-01-01T12:00:00"
    assert c["published_at_max"] == "2024-01-01T12:00:00"


# ---------------------------------------------------------------------------
# Behavior 3: multiple clusters ordered by importance_score descending
# ---------------------------------------------------------------------------

def test_get_story_clusters_ordered_by_importance_descending(store):
    _insert_article(store, "hash1", "Low Priority", "Source A", "2024-01-01T12:00:00", 3)
    _insert_article(store, "hash2", "High Priority", "Source B", "2024-01-02T12:00:00", 9)
    _insert_article(store, "hash3", "Medium Priority", "Source C", "2024-01-03T12:00:00", 6)

    clusters = store.get_story_clusters()
    scores = [c["importance_score"] for c in clusters]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 9


# ---------------------------------------------------------------------------
# Behavior 4: multiple articles in the same cluster are grouped together
# ---------------------------------------------------------------------------

def test_get_story_clusters_groups_articles_by_story_group(store):
    article_id1, group_id = _insert_article(store, "hash1", "Title 1", "Source A", "2024-01-01T12:00:00", 7)
    _insert_article(store, "hash2", "Title 1 (Reuters)", "Source B", "2024-01-02T12:00:00", 6, group_id)

    clusters = store.get_story_clusters()
    assert len(clusters) == 1
    assert clusters[0]["source_count"] == 2
    assert set(clusters[0]["source_names"]) == {"Source A", "Source B"}


# ---------------------------------------------------------------------------
# Behavior 5: cluster title comes from highest-importance article
# ---------------------------------------------------------------------------

def test_get_story_clusters_title_from_highest_importance_article(store):
    article_id1, group_id = _insert_article(store, "hash1", "Low Importance Title", "Source A", "2024-01-01T12:00:00", 3)
    _insert_article(store, "hash2", "High Importance Title", "Source B", "2024-01-02T12:00:00", 9, group_id)

    clusters = store.get_story_clusters()
    assert clusters[0]["title"] == "High Importance Title"
    assert clusters[0]["importance_score"] == 9


# ---------------------------------------------------------------------------
# Behavior 6: sentiment present → cluster carries label and score
# ---------------------------------------------------------------------------

def test_get_story_clusters_includes_sentiment_when_present(store):
    article_id, _ = _insert_article(store, "hash1", "Title 1", "Source A", "2024-01-01T12:00:00", 7)
    store.upsert_sentiment(article_id, {
        "sentiment_label": "Positive",
        "sentiment_score": 0.8,
        "excitement_level": "Hyped",
        "top_concerns": [],
        "top_use_cases": [],
        "notable_quotes": [],
        "subreddit_breakdown": {},
        "thread_count": 5,
        "total_comments": 100,
        "last_scanned_at": "2024-01-01T14:00:00",
    })

    clusters = store.get_story_clusters()
    assert clusters[0]["sentiment_label"] == "Positive"
    assert clusters[0]["sentiment_score"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Behavior 7: sentiment absent → label and score are None
# ---------------------------------------------------------------------------

def test_get_story_clusters_sentiment_none_when_absent(store):
    _insert_article(store, "hash1", "Title 1", "Source A", "2024-01-01T12:00:00", 7)
    clusters = store.get_story_clusters()
    assert clusters[0]["sentiment_label"] is None
    assert clusters[0]["sentiment_score"] is None


# ---------------------------------------------------------------------------
# Behavior 8: date range spans all articles in the cluster
# ---------------------------------------------------------------------------

def test_get_story_clusters_published_date_range(store):
    article_id1, group_id = _insert_article(store, "hash1", "Title 1", "Source A", "2024-01-01T12:00:00", 7)
    _insert_article(store, "hash2", "Title 1 v2", "Source B", "2024-01-05T12:00:00", 6, group_id)

    clusters = store.get_story_clusters()
    assert clusters[0]["published_at_min"] == "2024-01-01T12:00:00"
    assert clusters[0]["published_at_max"] == "2024-01-05T12:00:00"


# ---------------------------------------------------------------------------
# Behavior 9: articles list contains enriched detail fields
# ---------------------------------------------------------------------------

def test_get_story_clusters_includes_article_details(store):
    _insert_article(store, "hash1", "Title 1", "Source A", "2024-01-01T12:00:00", 7)

    clusters = store.get_story_clusters()
    articles = clusters[0]["articles"]
    assert len(articles) == 1
    a = articles[0]
    assert a["title"] == "Title 1"
    assert a["summary"] == "Summary of Title 1"
    assert a["importance_score"] == 7
    assert a["key_concepts"] == ["concept1"]


# ---------------------------------------------------------------------------
# Behavior 10: articles in cluster include sentiment details when present
# ---------------------------------------------------------------------------

def test_get_story_clusters_articles_include_sentiment_details(store):
    article_id, _ = _insert_article(store, "hash1", "Title 1", "Source A", "2024-01-01T12:00:00", 7)
    store.upsert_sentiment(article_id, {
        "sentiment_label": "Mixed",
        "sentiment_score": -0.1,
        "excitement_level": "Skeptical",
        "top_concerns": ["privacy"],
        "top_use_cases": ["automation"],
        "notable_quotes": ["interesting quote"],
        "subreddit_breakdown": {"r/MachineLearning": "positive"},
        "thread_count": 3,
        "total_comments": 50,
        "last_scanned_at": "2024-01-01T14:00:00",
    })

    clusters = store.get_story_clusters()
    a = clusters[0]["articles"][0]
    assert a["sentiment_label"] == "Mixed"
    assert a["top_concerns"] == ["privacy"]
    assert a["notable_quotes"] == ["interesting quote"]
    assert a["subreddit_breakdown"] == {"r/MachineLearning": "positive"}

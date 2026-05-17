import chromadb
import pytest

from storage.chroma_store import ChromaStore


@pytest.fixture
def store():
    import uuid
    client = chromadb.EphemeralClient()
    return ChromaStore(client=client, collection_name=f"test_{uuid.uuid4().hex}")


FIELDS = {
    "summary": "GPT-5 is a powerful language model.",
    "whats_new": "Improved reasoning.",
    "use_cases": "Code generation and summarization.",
}
EMBEDDINGS = {
    "summary": [0.1] * 768,
    "whats_new": [0.2] * 768,
    "use_cases": [0.3] * 768,
}


# ---------------------------------------------------------------------------
# Behavior 1: insert_chunks creates one document per field
# ---------------------------------------------------------------------------

def test_insert_chunks_creates_one_doc_per_field(store):
    store.insert_chunks(
        article_id=1, story_group_id=10, source_name="AI News",
        published_at="2024-01-01T12:00:00", field_texts=FIELDS, field_embeddings=EMBEDDINGS,
    )
    results = store._collection.get()
    assert len(results["ids"]) == 3


# ---------------------------------------------------------------------------
# Behavior 2: all chunks carry correct metadata
# ---------------------------------------------------------------------------

def test_insert_chunks_sets_correct_metadata(store):
    store.insert_chunks(
        article_id=1, story_group_id=10, source_name="AI News",
        published_at="2024-01-01T12:00:00", field_texts=FIELDS, field_embeddings=EMBEDDINGS,
    )
    results = store._collection.get(include=["metadatas"])
    for meta in results["metadatas"]:
        assert meta["article_id"] == "1"
        assert meta["story_group_id"] == "10"
        assert meta["source_name"] == "AI News"
        assert meta["published_at"] == "2024-01-01T12:00:00"
        assert meta["field"] in {"summary", "whats_new", "use_cases"}


# ---------------------------------------------------------------------------
# Behavior 3: get_summaries_since returns only summary chunks since cutoff
# ---------------------------------------------------------------------------

def test_get_summaries_since_returns_recent_summaries(store):
    store.insert_chunks(
        article_id=1, story_group_id=10, source_name="AI News",
        published_at="2024-01-03T00:00:00", field_texts=FIELDS, field_embeddings=EMBEDDINGS,
    )
    store.insert_chunks(
        article_id=2, story_group_id=20, source_name="Tech News",
        published_at="2024-01-01T00:00:00",
        field_texts={"summary": "Old article."},
        field_embeddings={"summary": [0.9] * 768},
    )
    results = store.get_summaries_since("2024-01-02T00:00:00")
    assert len(results) == 1
    assert results[0]["article_id"] == 1


def test_get_summaries_since_excludes_non_summary_fields(store):
    store.insert_chunks(
        article_id=1, story_group_id=10, source_name="AI News",
        published_at="2024-01-03T00:00:00", field_texts=FIELDS, field_embeddings=EMBEDDINGS,
    )
    results = store.get_summaries_since("2024-01-01T00:00:00")
    assert len(results) == 1  # only summary chunk, not whats_new or use_cases
    assert results[0]["article_id"] == 1


def test_get_summaries_since_returns_embedding(store):
    expected = [0.1] * 768
    store.insert_chunks(
        article_id=1, story_group_id=10, source_name="AI News",
        published_at="2024-01-03T00:00:00",
        field_texts={"summary": "Text"},
        field_embeddings={"summary": expected},
    )
    results = store.get_summaries_since("2024-01-01T00:00:00")
    assert len(results) == 1
    assert len(results[0]["embedding"]) == 768


# ---------------------------------------------------------------------------
# Behavior 4: search returns results with article_id in metadata
# ---------------------------------------------------------------------------

def test_search_returns_chunks_with_metadata(store):
    store.insert_chunks(
        article_id=1, story_group_id=10, source_name="AI News",
        published_at="2024-01-01T12:00:00", field_texts=FIELDS, field_embeddings=EMBEDDINGS,
    )
    query_embedding = [0.1] * 768
    results = store.search(query_embedding, n_results=3)
    assert len(results) > 0
    for r in results:
        assert "article_id" in r
        assert r["article_id"] == 1


# ---------------------------------------------------------------------------
# Behavior 5: insert_chunks is idempotent (upsert semantics)
# ---------------------------------------------------------------------------

def test_insert_chunks_is_idempotent(store):
    store.insert_chunks(
        article_id=1, story_group_id=10, source_name="AI News",
        published_at="2024-01-01T12:00:00", field_texts=FIELDS, field_embeddings=EMBEDDINGS,
    )
    store.insert_chunks(
        article_id=1, story_group_id=10, source_name="AI News",
        published_at="2024-01-01T12:00:00", field_texts=FIELDS, field_embeddings=EMBEDDINGS,
    )
    results = store._collection.get()
    assert len(results["ids"]) == 3

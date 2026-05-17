import math
import pytest
import chromadb

from storage.sqlite_store import SQLiteStore
from storage.chroma_store import ChromaStore
from tools.story_clustering import assign_story_group


@pytest.fixture
def sqlite_store():
    s = SQLiteStore(":memory:")
    s.init_db()
    return s


@pytest.fixture
def chroma_store():
    import uuid
    client = chromadb.EphemeralClient()
    return ChromaStore(client=client, collection_name=f"test_{uuid.uuid4().hex}")


# Unit vectors: a is (1,0,...), b_near is (0.99, ~0.141, 0, ...) giving sim ~0.99
_A = [1.0] + [0.0] * 767
_B_NEAR = [0.99, math.sqrt(1 - 0.99**2)] + [0.0] * 766  # cosine sim = 0.99 vs _A
_B_FAR = [0.0, 1.0] + [0.0] * 766  # cosine sim = 0.0 vs _A

# Use a far-future date so the cutoff (now - 3 days) always includes these articles.
_FUTURE = "2099-01-01T00:00:00"
_CUTOFF = "2098-12-29T00:00:00"  # 3 days before _FUTURE


# ---------------------------------------------------------------------------
# Behavior 1: similar article (sim > 0.85) joins existing story group
# ---------------------------------------------------------------------------

def test_similar_article_joins_existing_story_group(sqlite_store, chroma_store):
    group_id = sqlite_store.create_story_group()
    chroma_store.insert_chunks(
        article_id=1, story_group_id=group_id, source_name="Test",
        published_at=_FUTURE, field_texts={"summary": "LLM paper"},
        field_embeddings={"summary": _A},
    )

    result = assign_story_group(
        article_id=2,
        summary_embedding=_B_NEAR,
        published_at=_FUTURE,
        sqlite_store=sqlite_store,
        chroma_store=chroma_store,
        cutoff=_CUTOFF,
    )

    assert result == group_id


# ---------------------------------------------------------------------------
# Behavior 2: unrelated article (sim < 0.85) gets a new story group
# ---------------------------------------------------------------------------

def test_unrelated_article_gets_new_story_group(sqlite_store, chroma_store):
    group_id = sqlite_store.create_story_group()
    chroma_store.insert_chunks(
        article_id=1, story_group_id=group_id, source_name="Test",
        published_at=_FUTURE, field_texts={"summary": "LLM paper"},
        field_embeddings={"summary": _A},
    )

    result = assign_story_group(
        article_id=2,
        summary_embedding=_B_FAR,
        published_at=_FUTURE,
        sqlite_store=sqlite_store,
        chroma_store=chroma_store,
        cutoff=_CUTOFF,
    )

    assert result != group_id


# ---------------------------------------------------------------------------
# Behavior 3: source count increments when article joins existing group
# ---------------------------------------------------------------------------

def test_source_count_increments_on_join(sqlite_store, chroma_store):
    group_id = sqlite_store.create_story_group()
    chroma_store.insert_chunks(
        article_id=1, story_group_id=group_id, source_name="Test",
        published_at=_FUTURE, field_texts={"summary": "LLM paper"},
        field_embeddings={"summary": _A},
    )

    assign_story_group(
        article_id=2,
        summary_embedding=_B_NEAR,
        published_at=_FUTURE,
        sqlite_store=sqlite_store,
        chroma_store=chroma_store,
        cutoff=_CUTOFF,
    )

    group = sqlite_store.get_story_group(group_id)
    assert group["source_count"] == 2


# ---------------------------------------------------------------------------
# Behavior 4: first article (empty ChromaDB) always gets a new story group
# ---------------------------------------------------------------------------

def test_first_article_creates_new_story_group(sqlite_store, chroma_store):
    result = assign_story_group(
        article_id=1,
        summary_embedding=_A,
        published_at=_FUTURE,
        sqlite_store=sqlite_store,
        chroma_store=chroma_store,
        cutoff=_CUTOFF,
    )
    group = sqlite_store.get_story_group(result)
    assert group is not None
    assert group["source_count"] == 1

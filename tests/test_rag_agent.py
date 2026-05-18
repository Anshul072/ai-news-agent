import json
import uuid
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from storage.sqlite_store import SQLiteStore
from storage.chroma_store import ChromaStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RAW_ARTICLE = {
    "url": "https://example.com/gpt5",
    "url_hash": "hash_gpt5",
    "title": "GPT-5 Released",
    "content": "OpenAI releases GPT-5 with major improvements.",
    "source_name": "AI News",
    "published_at": "2024-01-01T12:00:00",
    "fetched_at": "2024-01-01T13:00:00",
}

ENRICHED = {
    "summary": "GPT-5 is OpenAI's most capable model.",
    "whats_new": "Major reasoning improvements.",
    "key_concepts": ["transformer", "RLHF"],
    "concept_explanations": {"transformer": "Attention model."},
    "who_made_it": "OpenAI",
    "use_cases": ["coding", "summarization"],
    "importance_score": 9,
    "importance_reasoning": "Major capability jump.",
}


def _fake_embed(text: str) -> list[float]:
    return [abs(hash(text) % 1000) / 1000.0 + 0.001] * 768


def _make_mock_response(answer: str, cited_ids: list):
    mock_response = MagicMock()
    mock_response.text = json.dumps({"answer": answer, "cited_ids": cited_ids})
    return mock_response


@pytest.fixture
def sqlite_store():
    s = SQLiteStore(":memory:")
    s.init_db()
    return s


@pytest.fixture
def chroma_store():
    client = chromadb.EphemeralClient()
    return ChromaStore(client=client, collection_name=f"test_{uuid.uuid4().hex}")


def _seed(sqlite_store, chroma_store):
    """Insert one article into both stores and return its article_id."""
    sqlite_store.insert_raw_article(RAW_ARTICLE)
    raw = sqlite_store.get_raw_article_by_url_hash(RAW_ARTICLE["url_hash"])
    article_id = raw["id"]
    sqlite_store.insert_enriched_article(article_id, ENRICHED)
    embedding = _fake_embed("GPT-5 is OpenAI's most capable model.")
    chroma_store.insert_chunks(
        article_id=article_id,
        story_group_id=0,
        source_name="AI News",
        published_at="2024-01-01T12:00:00",
        field_texts={"summary": "GPT-5 is OpenAI's most capable model."},
        field_embeddings={"summary": embedding},
    )
    return article_id


# ---------------------------------------------------------------------------
# Behavior 1: answer_query returns dict with answer and citations keys
# ---------------------------------------------------------------------------

def test_answer_query_returns_answer_and_citations(sqlite_store, chroma_store):
    article_id = _seed(sqlite_store, chroma_store)

    with patch("agents.rag_agent.embed", side_effect=_fake_embed), \
         patch("agents.rag_agent._client.models.generate_content",
               return_value=_make_mock_response("GPT-5 is great.", [article_id])):
        from agents.rag_agent import answer_query
        result = answer_query("tell me about GPT-5", sqlite_store, chroma_store)

    assert "answer" in result
    assert "citations" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["citations"], list)


# ---------------------------------------------------------------------------
# Behavior 2: citations reference real article_ids present in SQLite
# ---------------------------------------------------------------------------

def test_answer_query_citations_reference_real_article_ids(sqlite_store, chroma_store):
    article_id = _seed(sqlite_store, chroma_store)

    with patch("agents.rag_agent.embed", side_effect=_fake_embed), \
         patch("agents.rag_agent._client.models.generate_content",
               return_value=_make_mock_response("GPT-5 is great.", [article_id])):
        from agents.rag_agent import answer_query
        result = answer_query("tell me about GPT-5", sqlite_store, chroma_store)

    assert len(result["citations"]) >= 1
    for citation in result["citations"]:
        assert sqlite_store.get_raw_article(citation["article_id"]) is not None


# ---------------------------------------------------------------------------
# Behavior 3: returns "not enough data" response when no chunks found
# ---------------------------------------------------------------------------

def test_answer_query_returns_no_data_message_when_empty(sqlite_store, chroma_store):
    with patch("agents.rag_agent.embed", side_effect=_fake_embed):
        from agents.rag_agent import answer_query
        result = answer_query("tell me about something unknown", sqlite_store, chroma_store)

    assert result["citations"] == []
    assert "not enough data" in result["answer"].lower()


# ---------------------------------------------------------------------------
# Behavior 4: the Gemini prompt contains article context from SQLite
# ---------------------------------------------------------------------------

def test_answer_query_passes_article_context_to_gemini(sqlite_store, chroma_store):
    article_id = _seed(sqlite_store, chroma_store)

    with patch("agents.rag_agent.embed", side_effect=_fake_embed), \
         patch("agents.rag_agent._client.models.generate_content",
               return_value=_make_mock_response("GPT-5 is great.", [article_id])) as mock_generate:
        from agents.rag_agent import answer_query
        answer_query("tell me about GPT-5", sqlite_store, chroma_store)

    call_kwargs = mock_generate.call_args.kwargs
    prompt_text = call_kwargs["contents"]
    assert "GPT-5 Released" in prompt_text
    assert "tell me about GPT-5" in prompt_text

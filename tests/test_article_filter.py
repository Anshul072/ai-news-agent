import importlib
import math
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixed test vectors (3-dimensional for simplicity)
# ---------------------------------------------------------------------------

_DIM = 3


def _unit_vec(hot: int = 0) -> list[float]:
    v = [0.0] * _DIM
    v[hot] = 1.0
    return v


def _at_threshold(threshold: float) -> list[float]:
    """Unit vector whose cosine similarity with _unit_vec(0) equals `threshold`."""
    v = [0.0] * _DIM
    v[0] = threshold
    v[1] = math.sqrt(1.0 - threshold ** 2)
    return v


_SEED_VECS = [_unit_vec(0)] * 10   # all seeds point in dimension 0
_RELEVANT   = _unit_vec(0)          # cos_sim with any seed = 1.0  ≥ 0.45
_IRRELEVANT = _unit_vec(2)          # cos_sim with any seed = 0.0  < 0.45


# ---------------------------------------------------------------------------
# Fixture: reload article_filter with embed mocked so seeds use fixed vectors
# ---------------------------------------------------------------------------

@pytest.fixture
def af():
    """Reload tools.article_filter with embed mocked to avoid real model calls."""
    with patch("tools.embedder.embed", return_value=_unit_vec(0)):
        import tools.article_filter as module
        importlib.reload(module)
        yield module


# ---------------------------------------------------------------------------
# Behavior 1: returns True when article scores >= threshold against a seed
# ---------------------------------------------------------------------------

def test_is_relevant_returns_true_for_high_similarity(af):
    with patch("tools.article_filter._SEED_EMBEDDINGS", _SEED_VECS), \
         patch("tools.embedder.embed", return_value=_RELEVANT):
        assert af.is_relevant({"title": "GPT-5 released", "content": "new model"}) is True


# ---------------------------------------------------------------------------
# Behavior 2: returns False when article scores < threshold against all seeds
# ---------------------------------------------------------------------------

def test_is_relevant_returns_false_for_low_similarity(af):
    with patch("tools.article_filter._SEED_EMBEDDINGS", _SEED_VECS), \
         patch("tools.embedder.embed", return_value=_IRRELEVANT):
        assert af.is_relevant({"title": "Sports news", "content": "football"}) is False


# ---------------------------------------------------------------------------
# Behavior 3: inclusive boundary — exactly at threshold returns True
# ---------------------------------------------------------------------------

def test_is_relevant_returns_true_at_exact_threshold(af):
    at_threshold = _at_threshold(0.45)
    with patch("tools.article_filter._SEED_EMBEDDINGS", _SEED_VECS), \
         patch("tools.embedder.embed", return_value=at_threshold):
        assert af.is_relevant({"title": "test", "content": "test"}) is True


# ---------------------------------------------------------------------------
# Behavior 4: seed embeddings computed exactly once at module load, not per call
# ---------------------------------------------------------------------------

def test_seed_embeddings_computed_once_at_module_load():
    with patch("tools.embedder.embed", return_value=_unit_vec(0)) as mock_embed:
        import tools.article_filter as module
        importlib.reload(module)
        seed_call_count = mock_embed.call_count
        assert seed_call_count == 10

        module.is_relevant({"title": "a", "content": "b"})
        module.is_relevant({"title": "c", "content": "d"})
        module.is_relevant({"title": "e", "content": "f"})

        assert mock_embed.call_count == seed_call_count + 3

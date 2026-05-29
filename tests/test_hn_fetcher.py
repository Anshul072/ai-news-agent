from unittest.mock import MagicMock, call, patch

import pytest

from tools.hn_fetcher import _cosine, _normalize_url, fetch_hn_threads

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _algolia_hit(object_id: str, title: str, points: int = 50) -> dict:
    return {"objectID": object_id, "title": title, "points": points, "num_comments": 10}


def _item_response(story_id: str) -> dict:
    return {"id": story_id, "children": []}


def _mock_requests_get(url_search_hits, keyword_hits=None):
    """Return a mock for requests.get.

    First call → Algolia search (URL or keyword).
    If keyword_hits is provided a second call returns those.
    Subsequent per-story item fetches return an empty item.
    """
    call_count = [0]

    def side_effect(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "algolia" in url:
            call_count[0] += 1
            if call_count[0] == 1:
                resp.json.return_value = {"hits": url_search_hits}
            else:
                resp.json.return_value = {"hits": keyword_hits or []}
        else:
            # Per-story item fetch
            story_id = url.rstrip("/").split("/")[-1]
            resp.json.return_value = _item_response(story_id)
        return resp

    return side_effect


# ---------------------------------------------------------------------------
# _normalize_url
# ---------------------------------------------------------------------------

def test_normalize_url_strips_scheme():
    assert _normalize_url("https://example.com/foo") == "example.com/foo"


def test_normalize_url_strips_trailing_slash():
    assert _normalize_url("https://example.com/foo/") == "example.com/foo"


def test_normalize_url_http_and_https_same():
    assert _normalize_url("http://example.com/bar") == _normalize_url("https://example.com/bar")


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_vectors():
    assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# URL-first path
# ---------------------------------------------------------------------------

def test_url_search_used_first_when_article_url_given():
    hit = _algolia_hit("1", "Claude Opus 4.8 Released", points=200)

    with patch("tools.hn_fetcher.requests.get",
               side_effect=_mock_requests_get([hit])) as mock_get:
        threads = fetch_hn_threads(["Claude"], article_url="https://anthropic.com/news/claude-4-8")

    # First Algolia call must use restrictSearchableAttributes=url
    first_algolia_call = mock_get.call_args_list[0]
    params = first_algolia_call[1].get("params") or first_algolia_call[0][1]
    assert params.get("restrictSearchableAttributes") == "url"
    assert len(threads) == 1
    assert threads[0]["title"] == "Claude Opus 4.8 Released"


def test_url_search_result_skips_similarity_filter():
    hit = _algolia_hit("1", "Completely Unrelated Title", points=200)

    with patch("tools.hn_fetcher.requests.get",
               side_effect=_mock_requests_get([hit])), \
         patch("tools.hn_fetcher.embed") as mock_embed:
        fetch_hn_threads(["Claude"], article_url="https://anthropic.com/news/claude-4-8")

    # embed() must not be called when URL search succeeds
    mock_embed.assert_not_called()


# ---------------------------------------------------------------------------
# Keyword fallback path
# ---------------------------------------------------------------------------

def test_keyword_fallback_fires_when_url_search_empty():
    keyword_hit = _algolia_hit("2", "Claude Opus Review", points=50)

    call_count = [0]

    def side_effect(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "algolia" in url and "search" in url:
            call_count[0] += 1
            if call_count[0] == 1:
                resp.json.return_value = {"hits": []}       # URL search → empty
            else:
                resp.json.return_value = {"hits": [keyword_hit]}  # keyword search → hit
        else:
            resp.json.return_value = _item_response(url.split("/")[-1])
        return resp

    with patch("tools.hn_fetcher.requests.get", side_effect=side_effect), \
         patch("tools.hn_fetcher.embed", return_value=[1.0, 0.0]):
        threads = fetch_hn_threads(["Claude"], article_url="https://anthropic.com/news/claude-4-8")

    assert call_count[0] == 2  # URL search + keyword search
    assert len(threads) == 1


def test_keyword_fallback_without_article_url():
    hit = _algolia_hit("3", "Claude AI Discussion", points=30)

    def side_effect(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "search" in url:
            resp.json.return_value = {"hits": [hit]}
        else:
            resp.json.return_value = _item_response(url.split("/")[-1])
        return resp

    with patch("tools.hn_fetcher.requests.get", side_effect=side_effect), \
         patch("tools.hn_fetcher.embed", return_value=[1.0, 0.0]):
        threads = fetch_hn_threads(["Claude", "AI"])

    assert len(threads) == 1


# ---------------------------------------------------------------------------
# Similarity post-filter
# ---------------------------------------------------------------------------

def test_similarity_filter_removes_low_scoring_hits():
    hits = [
        _algolia_hit("10", "Claude Opus 4.8 release", points=50),   # relevant
        _algolia_hit("11", "Dating App Built with Cursor", points=20),  # irrelevant
    ]

    def fake_embed(text: str) -> list[float]:
        if "Claude" in text or "Opus" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]  # orthogonal → cosine 0.0

    def side_effect(url, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "search" in url:
            resp.json.return_value = {"hits": hits}
        else:
            resp.json.return_value = _item_response(url.split("/")[-1])
        return resp

    with patch("tools.hn_fetcher.requests.get", side_effect=side_effect), \
         patch("tools.hn_fetcher.embed", side_effect=fake_embed):
        threads = fetch_hn_threads(["Claude", "Opus"])

    assert len(threads) == 1
    assert threads[0]["title"] == "Claude Opus 4.8 release"


# ---------------------------------------------------------------------------
# Result cap at limit
# ---------------------------------------------------------------------------

def test_results_capped_at_limit():
    hits = [_algolia_hit(str(i), f"Claude story {i}", points=50) for i in range(10)]

    with patch("tools.hn_fetcher.requests.get",
               side_effect=_mock_requests_get([], keyword_hits=hits)), \
         patch("tools.hn_fetcher.embed", return_value=[1.0, 0.0]):
        threads = fetch_hn_threads(["Claude"], limit=3)

    assert len(threads) <= 3

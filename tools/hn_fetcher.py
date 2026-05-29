import html
import math
import re
from urllib.parse import urlparse

import requests

from tools.embedder import embed

_ALGOLIA_SEARCH = "https://hn.algolia.com/api/v1/search"
_ALGOLIA_ITEMS = "https://hn.algolia.com/api/v1/items/{}"

_SIMILARITY_THRESHOLD = 0.55
_MIN_POINTS = 10


def _strip_html(text: str) -> str:
    text = html.unescape(text)
    return re.sub(r"<[^>]+>", " ", text).strip()


def _top_comments(item: dict, n: int) -> list[str]:
    comments = []
    for child in item.get("children") or []:
        if child.get("type") == "comment" and child.get("text"):
            comments.append(_strip_html(child["text"]))
            if len(comments) >= n:
                break
    return comments


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.netloc + parsed.path).rstrip("/")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _algolia_search(params: dict) -> list[dict]:
    try:
        resp = requests.get(_ALGOLIA_SEARCH, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("hits", [])
    except Exception:
        return []


def _fetch_items(hits: list[dict], top_comments_per_thread: int) -> list[dict]:
    threads = []
    for hit in hits:
        story_id = hit.get("objectID")
        if not story_id:
            continue
        try:
            item_resp = requests.get(_ALGOLIA_ITEMS.format(story_id), timeout=10)
            item_resp.raise_for_status()
            item = item_resp.json()
        except Exception:
            continue
        threads.append({
            "source": "Hacker News",
            "title": hit.get("title", ""),
            "url": f"https://news.ycombinator.com/item?id={story_id}",
            "score": hit.get("points") or 0,
            "num_comments": hit.get("num_comments") or 0,
            "top_comments": _top_comments(item, top_comments_per_thread),
        })
    return threads


def fetch_hn_threads(
    keywords: list[str],
    limit: int = 5,
    top_comments_per_thread: int = 3,
    article_url: str | None = None,
) -> list[dict]:
    # Primary: search by article URL — finds the exact HN submission if it exists
    if article_url:
        hits = _algolia_search({
            "query": _normalize_url(article_url),
            "tags": "story",
            "restrictSearchableAttributes": "url",
            "hitsPerPage": limit,
        })
        if hits:
            return _fetch_items(hits[:limit], top_comments_per_thread)

    # Fallback: keyword search with engagement filter + cosine similarity post-filter
    hits = _algolia_search({
        "query": " ".join(keywords),
        "tags": "story",
        "numericFilters": f"points>={_MIN_POINTS}",
        "hitsPerPage": limit * 2,
    })

    if hits and keywords:
        query_text = article_url or " ".join(keywords)
        query_vec = embed(query_text)
        hits = [
            h for h in hits
            if _cosine(embed(h.get("title", "")), query_vec) >= _SIMILARITY_THRESHOLD
        ]

    return _fetch_items(hits[:limit], top_comments_per_thread)

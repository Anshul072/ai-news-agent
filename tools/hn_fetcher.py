import html
import re

import requests

_ALGOLIA_SEARCH = "https://hn.algolia.com/api/v1/search"
_ALGOLIA_ITEMS = "https://hn.algolia.com/api/v1/items/{}"


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


def fetch_hn_threads(
    keywords: list[str],
    limit: int = 5,
    top_comments_per_thread: int = 3,
) -> list[dict]:
    query = " ".join(keywords)
    try:
        resp = requests.get(
            _ALGOLIA_SEARCH,
            params={"query": query, "tags": "story", "hitsPerPage": limit},
            timeout=10,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
    except Exception:
        return []

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
            "score": hit.get("points") or 0,
            "num_comments": hit.get("num_comments") or 0,
            "top_comments": _top_comments(item, top_comments_per_thread),
        })
    return threads

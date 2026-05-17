import hashlib
from datetime import datetime, timezone

import feedparser


def fetch_articles(feed_urls: list[str]) -> list[dict]:
    articles = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            source_name = feed.feed.get("title", url)
            for entry in feed.entries:
                link = entry.get("link", "")
                if not link:
                    continue
                articles.append({
                    "url": link,
                    "url_hash": hashlib.sha256(link.encode()).hexdigest(),
                    "title": entry.get("title", ""),
                    "content": entry.get("summary", entry.get("description", "")),
                    "source_name": source_name,
                    "published_at": _parse_date(entry),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception:
            continue
    return articles


def _parse_date(entry) -> str:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()

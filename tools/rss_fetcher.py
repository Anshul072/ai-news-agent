import hashlib
from datetime import datetime, timezone

import feedparser


def fetch_articles(feed_urls: list[str], since: datetime | None = None) -> list[dict]:
    articles = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            source_name = feed.feed.get("title", url)
            for entry in feed.entries:
                link = entry.get("link", "")
                if not link:
                    continue
                published_at = _parse_date(entry)
                if since is not None and _is_old(published_at, since):
                    continue
                articles.append({
                    "url": link,
                    "url_hash": hashlib.sha256(link.encode()).hexdigest(),
                    "title": entry.get("title", ""),
                    "content": entry.get("summary", entry.get("description", "")),
                    "source_name": source_name,
                    "published_at": published_at,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception:
            continue
    return articles


def _is_old(published_at: str, since: datetime) -> bool:
    """Return True if published_at is at or before since (should be skipped)."""
    try:
        pub_dt = datetime.fromisoformat(published_at)
        # Normalise to UTC for comparison
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        return pub_dt <= since
    except Exception:
        return False  # include entries whose date can't be parsed


def _parse_date(entry) -> str:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()

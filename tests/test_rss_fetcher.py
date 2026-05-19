import hashlib
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import feedparser
import pytest

from tools.rss_fetcher import fetch_articles


VALID_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>AI News</title>
    <item>
      <title>GPT-5 Released</title>
      <link>https://example.com/gpt5</link>
      <description>OpenAI releases GPT-5 with improved reasoning.</description>
      <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Gemini Update</title>
      <link>https://example.com/gemini</link>
      <description>Google updates Gemini model family.</description>
      <pubDate>Tue, 02 Jan 2024 09:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""

EMPTY_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Empty</title></channel></rss>"""

# Pre-parse once (before any patching) so mocks return real FeedParserDict objects
_PARSED_VALID = feedparser.parse(VALID_FEED_XML)
_PARSED_EMPTY = feedparser.parse(EMPTY_FEED_XML)


# ---------------------------------------------------------------------------
# Behavior 1: valid feed returns normalized articles with required fields
# ---------------------------------------------------------------------------

def test_valid_feed_returns_articles():
    with patch("tools.rss_fetcher.feedparser.parse", return_value=_PARSED_VALID):
        articles = fetch_articles(["https://fake-feed.example.com"])

    assert len(articles) == 2


def test_articles_have_required_fields():
    with patch("tools.rss_fetcher.feedparser.parse", return_value=_PARSED_VALID):
        articles = fetch_articles(["https://fake-feed.example.com"])

    required = {"url", "url_hash", "title", "content", "source_name", "published_at", "fetched_at"}
    for article in articles:
        assert required.issubset(article.keys()), f"Missing fields: {required - article.keys()}"


# ---------------------------------------------------------------------------
# Behavior 2: url_hash is SHA-256 of the URL
# ---------------------------------------------------------------------------

def test_url_hash_is_sha256_of_url():
    with patch("tools.rss_fetcher.feedparser.parse", return_value=_PARSED_VALID):
        articles = fetch_articles(["https://fake-feed.example.com"])

    for article in articles:
        expected = hashlib.sha256(article["url"].encode()).hexdigest()
        assert article["url_hash"] == expected


# ---------------------------------------------------------------------------
# Behavior 3: empty feed returns empty list (no crash)
# ---------------------------------------------------------------------------

def test_empty_feed_returns_empty_list():
    with patch("tools.rss_fetcher.feedparser.parse", return_value=_PARSED_EMPTY):
        articles = fetch_articles(["https://fake-feed.example.com"])

    assert articles == []


# ---------------------------------------------------------------------------
# Behavior 4: unreachable URL is skipped gracefully
# ---------------------------------------------------------------------------

def test_unreachable_url_skipped():
    boilerplate_error = MagicMock()
    boilerplate_error.entries = []
    boilerplate_error.bozo = True
    boilerplate_error.bozo_exception = ConnectionError("refused")

    with patch("tools.rss_fetcher.feedparser.parse", return_value=boilerplate_error):
        articles = fetch_articles(["https://unreachable.example.com"])

    assert articles == []


# ---------------------------------------------------------------------------
# Behavior 5: multiple feeds are aggregated
# ---------------------------------------------------------------------------

def test_multiple_feeds_aggregated():
    with patch("tools.rss_fetcher.feedparser.parse", return_value=_PARSED_VALID):
        articles = fetch_articles([
            "https://feed1.example.com",
            "https://feed2.example.com",
        ])

    assert len(articles) == 4


# ---------------------------------------------------------------------------
# Behavior 6: since=None fetches all entries (first run, no limit)
# ---------------------------------------------------------------------------

def test_since_none_fetches_all_articles():
    with patch("tools.rss_fetcher.feedparser.parse", return_value=_PARSED_VALID):
        articles = fetch_articles(["https://fake-feed.example.com"], since=None)

    assert len(articles) == 2


# ---------------------------------------------------------------------------
# Behavior 7: articles published after `since` are included
# ---------------------------------------------------------------------------

def test_since_includes_articles_published_after_cutoff():
    # Feed has: GPT-5 on Jan 1 12:00, Gemini on Jan 2 09:00
    # since = Jan 1 18:00 → only Gemini passes
    since = datetime(2024, 1, 1, 18, 0, 0, tzinfo=timezone.utc)
    with patch("tools.rss_fetcher.feedparser.parse", return_value=_PARSED_VALID):
        articles = fetch_articles(["https://fake-feed.example.com"], since=since)

    assert len(articles) == 1
    assert articles[0]["title"] == "Gemini Update"


# ---------------------------------------------------------------------------
# Behavior 8: articles published at or before `since` are excluded
# ---------------------------------------------------------------------------

def test_since_excludes_articles_published_before_cutoff():
    since = datetime(2024, 1, 3, 0, 0, 0, tzinfo=timezone.utc)
    with patch("tools.rss_fetcher.feedparser.parse", return_value=_PARSED_VALID):
        articles = fetch_articles(["https://fake-feed.example.com"], since=since)

    assert articles == []


# ---------------------------------------------------------------------------
# Behavior 9: entries with unparseable date are included (safe default)
# ---------------------------------------------------------------------------

NO_DATE_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>AI News</title>
    <item>
      <title>No Date Article</title>
      <link>https://example.com/no-date</link>
      <description>Article with no pubDate.</description>
    </item>
  </channel>
</rss>"""

_PARSED_NO_DATE = feedparser.parse(NO_DATE_FEED_XML)


def test_entries_without_date_are_included_when_since_set():
    since = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    with patch("tools.rss_fetcher.feedparser.parse", return_value=_PARSED_NO_DATE):
        articles = fetch_articles(["https://fake-feed.example.com"], since=since)

    assert len(articles) == 1

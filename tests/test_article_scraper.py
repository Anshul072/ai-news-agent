import logging
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Behavior 1: returns scraped text when extraction succeeds
# ---------------------------------------------------------------------------

def test_scrape_content_returns_scraped_text_on_success():
    scraped = "A" * 300

    with patch("tools.article_scraper.trafilatura.fetch_url", return_value="<html>...</html>"), \
         patch("tools.article_scraper.trafilatura.extract", return_value=scraped):
        from tools.article_scraper import scrape_content
        result = scrape_content("https://example.com/article", "fallback text")

    assert result == scraped


# ---------------------------------------------------------------------------
# Behavior 2: returns fallback when extract returns empty string
# ---------------------------------------------------------------------------

def test_scrape_content_returns_fallback_when_extract_returns_empty():
    with patch("tools.article_scraper.trafilatura.fetch_url", return_value="<html>...</html>"), \
         patch("tools.article_scraper.trafilatura.extract", return_value=""):
        from tools.article_scraper import scrape_content
        result = scrape_content("https://example.com/article", "fallback text")

    assert result == "fallback text"


# ---------------------------------------------------------------------------
# Behavior 3: returns fallback when extract returns None
# ---------------------------------------------------------------------------

def test_scrape_content_returns_fallback_when_extract_returns_none():
    with patch("tools.article_scraper.trafilatura.fetch_url", return_value="<html>...</html>"), \
         patch("tools.article_scraper.trafilatura.extract", return_value=None):
        from tools.article_scraper import scrape_content
        result = scrape_content("https://example.com/article", "fallback text")

    assert result == "fallback text"


# ---------------------------------------------------------------------------
# Behavior 4: returns fallback when an exception is raised
# ---------------------------------------------------------------------------

def test_scrape_content_returns_fallback_on_exception():
    with patch("tools.article_scraper.trafilatura.fetch_url", side_effect=Exception("network error")):
        from tools.article_scraper import scrape_content
        result = scrape_content("https://example.com/article", "fallback text")

    assert result == "fallback text"


# ---------------------------------------------------------------------------
# Behavior 5: returns fallback when scraped text is below minimum length
# ---------------------------------------------------------------------------

def test_scrape_content_returns_fallback_when_below_minimum_length():
    with patch("tools.article_scraper.trafilatura.fetch_url", return_value="<html>...</html>"), \
         patch("tools.article_scraper.trafilatura.extract", return_value="short"):
        from tools.article_scraper import scrape_content
        result = scrape_content("https://example.com/article", "fallback text")

    assert result == "fallback text"


# ---------------------------------------------------------------------------
# Behavior 6: failures are logged with URL and reason
# ---------------------------------------------------------------------------

def test_scrape_content_logs_failure_with_url(caplog):
    with patch("tools.article_scraper.trafilatura.fetch_url", side_effect=Exception("timeout")):
        with caplog.at_level(logging.WARNING, logger="tools.article_scraper"):
            from tools.article_scraper import scrape_content
            scrape_content("https://example.com/article", "fallback text")

    assert any("https://example.com/article" in r.message for r in caplog.records)

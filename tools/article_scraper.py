import logging

import trafilatura

logger = logging.getLogger(__name__)

_MIN_SCRAPED_LENGTH = 200


def scrape_content(url: str, fallback: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            logger.warning("Scrape returned empty download: %s", url)
            return fallback
        text = trafilatura.extract(downloaded)
        if not text:
            logger.warning("Scrape returned no text: %s", url)
            return fallback
        if len(text) < _MIN_SCRAPED_LENGTH:
            logger.warning("Scrape below minimum length (%d chars): %s", len(text), url)
            return fallback
        return text
    except Exception as exc:
        logger.warning("Scrape failed for %s: %s", url, exc)
        return fallback

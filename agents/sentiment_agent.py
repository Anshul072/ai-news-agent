import json
from datetime import datetime, timezone

from google import genai

import config
from tools.reddit_fetcher import fetch_reddit_threads

_client = genai.Client(api_key=config.GEMINI_API_KEY)

_KEYWORDS_PROMPT = """Given this article title, extract 3-5 concise search keywords suitable for Reddit search.
Return ONLY a JSON array of strings, no markdown fences.

Title: {title}"""

_SENTIMENT_PROMPT = """You are analyzing Reddit community sentiment about an AI news article.
Given the Reddit threads below, produce a sentiment analysis.
Return ONLY valid JSON with these exact keys:

- sentiment_label: "Positive" | "Negative" | "Mixed" | "Neutral"
- sentiment_score: float -1.0 to +1.0
- excitement_level: "Hyped" | "Skeptical" | "Indifferent"
- top_concerns: list of up to 3 strings
- top_use_cases: list of up to 3 strings (community-imagined use cases)
- notable_quotes: list of 2-3 verbatim comment excerpts
- subreddit_breakdown: object mapping subreddit name to a one-sentence sentiment summary

Reddit threads:
{threads}"""

_NEUTRAL_SENTINEL = {
    "sentiment_label": "Neutral",
    "sentiment_score": 0.0,
    "excitement_level": "Indifferent",
    "top_concerns": [],
    "top_use_cases": [],
    "notable_quotes": [],
    "subreddit_breakdown": {},
    "thread_count": 0,
    "total_comments": 0,
}


def _extract_keywords(title: str) -> list[str]:
    prompt = _KEYWORDS_PROMPT.format(title=title)
    response = _client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    return json.loads(response.text)


def _analyse_threads(threads: list[dict]) -> dict:
    threads_text = json.dumps(threads, indent=2)
    prompt = _SENTIMENT_PROMPT.format(threads=threads_text)
    response = _client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
    return json.loads(response.text)


def run_sentiment(article_id: int, sqlite_store) -> dict:
    article = sqlite_store.get_raw_article_by_url_hash(
        sqlite_store._get_conn().execute(
            "SELECT url_hash FROM raw_articles WHERE id = ?", (article_id,)
        ).fetchone()["url_hash"]
    )

    keywords = _extract_keywords(article["title"])
    threads = fetch_reddit_threads(keywords)

    if not threads:
        sentiment = dict(_NEUTRAL_SENTINEL)
    else:
        try:
            sentiment = _analyse_threads(threads)
        except (json.JSONDecodeError, Exception):
            sentiment = dict(_NEUTRAL_SENTINEL)
        sentiment["thread_count"] = len(threads)
        sentiment["total_comments"] = sum(t.get("num_comments", 0) for t in threads)

    sentiment["last_scanned_at"] = datetime.now(timezone.utc).isoformat()
    sqlite_store.upsert_sentiment(article_id, sentiment)
    return sentiment

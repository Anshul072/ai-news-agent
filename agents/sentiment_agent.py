import json
from datetime import datetime, timezone

from groq import Groq

import config
from tools.hn_fetcher import fetch_hn_threads

_client = Groq(api_key=config.GROQ_API_KEY)
_MODEL = "llama-3.3-70b-versatile"

_KEYWORDS_PROMPT = """Given this article title, extract 3-5 concise search keywords suitable for Hacker News search.
Return ONLY a JSON array of strings, no markdown fences.

Title: {title}"""

_SENTIMENT_PROMPT = """You are analyzing Hacker News community sentiment about an AI news article.
Given the Hacker News threads below, produce a sentiment analysis.
Return ONLY valid JSON with these exact keys:

- sentiment_label: "Positive" | "Negative" | "Mixed" | "Neutral"
- sentiment_score: float -1.0 to +1.0
- excitement_level: "Hyped" | "Skeptical" | "Indifferent"
- top_concerns: list of up to 3 strings
- top_use_cases: list of up to 3 strings (community-imagined use cases)
- notable_quotes: list of 2-3 verbatim comment excerpts
- subreddit_breakdown: object mapping source name to a one-sentence sentiment summary

Hacker News threads:
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


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _call_groq(prompt: str) -> str:
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _extract_keywords(title: str) -> list[str]:
    prompt = _KEYWORDS_PROMPT.format(title=title)
    return json.loads(_strip_fences(_call_groq(prompt)))


def _analyse_threads(threads: list[dict]) -> dict:
    threads_text = json.dumps(threads, indent=2)
    prompt = _SENTIMENT_PROMPT.format(threads=threads_text)
    return json.loads(_strip_fences(_call_groq(prompt)))


def run_sentiment(article_id: int, sqlite_store) -> dict:
    article = sqlite_store.get_raw_article_by_url_hash(
        sqlite_store._get_conn().execute(
            "SELECT url_hash FROM raw_articles WHERE id = ?", (article_id,)
        ).fetchone()["url_hash"]
    )

    keywords = _extract_keywords(article["title"])
    threads = fetch_hn_threads(keywords, article_url=article["url"])

    if not threads:
        sentiment = dict(_NEUTRAL_SENTINEL)
    else:
        try:
            sentiment = _analyse_threads(threads)
        except (json.JSONDecodeError, Exception):
            sentiment = dict(_NEUTRAL_SENTINEL)
        sentiment["thread_count"] = len(threads)
        sentiment["total_comments"] = sum(t.get("num_comments", 0) for t in threads)
        sentiment["hn_thread_urls"] = [t["url"] for t in threads if t.get("url")]

    sentiment["last_scanned_at"] = datetime.now(timezone.utc).isoformat()
    sqlite_store.upsert_sentiment(article_id, sentiment)
    return sentiment

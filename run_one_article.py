"""
Quick smoke-test: run the full news pipeline for a single hard-coded article.
Uses in-memory SQLite and ephemeral ChromaDB — no persistent state written.
Requires GROQ_API_KEY and GEMINI_API_KEY in .env (or environment).
"""
import hashlib
import logging
import sys
from datetime import datetime, timezone

import chromadb

import pipelines.news_pipeline as _pipeline_mod
from storage.sqlite_store import SQLiteStore
from storage.chroma_store import ChromaStore
from pipelines.news_pipeline import build_news_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stdout,
)

ARTICLE = {
    "url": "https://example.com/gemini-2-flash-release",
    "title": "Google releases Gemini 2 Flash with 1M context window",
    "content": (
        "Google today released Gemini 2 Flash, a lightweight multimodal model "
        "that supports a 1 million token context window. The model is available "
        "via the Gemini API and is optimised for low-latency tasks such as "
        "document summarisation, code generation, and real-time chat. "
        "Benchmarks show it outperforms GPT-4o-mini on MMLU while using half "
        "the inference cost."
    ),
    "source_name": "Example AI News",
    "published_at": datetime.now(timezone.utc).isoformat(),
    "fetched_at": datetime.now(timezone.utc).isoformat(),
}
ARTICLE["url_hash"] = hashlib.sha256(ARTICLE["url"].encode()).hexdigest()


def main():
    sqlite_store = SQLiteStore(db_path=":memory:")
    sqlite_store.init_db()

    chroma_client = chromadb.EphemeralClient()
    chroma_store = ChromaStore(client=chroma_client, collection_name="smoke_test")

    # Patch the name bound inside the pipeline module (imported by-name at module load time).
    _pipeline_mod.fetch_articles = lambda _urls, **_kw: [ARTICLE]

    pipeline = build_news_pipeline(sqlite_store, chroma_store)

    result = pipeline.invoke({
        "feed_urls": ["https://fake-feed-for-smoke-test"],
        "raw_articles": [],
        "new_articles": [],
        "parsed_articles": [],
        "stored_count": 0,
    })

    print("\n=== Pipeline result ===")
    print(f"stored_count : {result['stored_count']}")
    print(f"new_articles : {len(result.get('new_articles', []))}")
    print(f"parsed_articles: {len(result.get('parsed_articles', []))}")

    if result["stored_count"] > 0:
        # Pull back the enriched article from SQLite and display it.
        db_article = sqlite_store.get_raw_article_by_url_hash(ARTICLE["url_hash"])
        if db_article:
            enriched = sqlite_store.get_enriched_article(db_article["id"])
            if enriched:
                print("\n=== Enriched article ===")
                for k, v in enriched.items():
                    print(f"  {k}: {v}")
        print("\nSUCCESS — pipeline ran end-to-end without errors.")
    else:
        print("\nWARNING — stored_count is 0; check logs above for errors.")


if __name__ == "__main__":
    main()

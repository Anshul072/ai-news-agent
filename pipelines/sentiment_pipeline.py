from typing import TypedDict

from langgraph.graph import StateGraph, END

import config
from agents.sentiment_agent import run_sentiment
from tools.embedder import embed


class _State(TypedDict):
    articles: list[dict]
    sentiment_results: list[dict]
    processed_count: int


def _sentiment_to_text(sentiment: dict) -> str:
    parts = []
    label = sentiment.get("sentiment_label", "")
    excitement = sentiment.get("excitement_level", "")
    if label:
        parts.append(f"Sentiment: {label}.")
    if excitement:
        parts.append(f"Excitement: {excitement}.")
    concerns = sentiment.get("top_concerns", [])
    if concerns:
        parts.append(f"Concerns: {', '.join(concerns)}.")
    use_cases = sentiment.get("top_use_cases", [])
    if use_cases:
        parts.append(f"Community use cases: {', '.join(use_cases)}.")
    quotes = sentiment.get("notable_quotes", [])
    if quotes:
        parts.append(" ".join(f'"{q}"' for q in quotes))
    return " ".join(parts)


def build_sentiment_pipeline(sqlite_store, chroma_store):
    def load_recent_articles(state: _State) -> dict:
        return {"articles": sqlite_store.get_recent_enriched_articles(days=config.SENTIMENT_WINDOW_DAYS)}

    def fetch_and_analyse(state: _State) -> dict:
        results = []
        for article in state["articles"]:
            article_id = article["article_id"]
            try:
                sentiment = run_sentiment(article_id, sqlite_store)
                results.append({
                    "article_id": article_id,
                    "article": article,
                    "sentiment": sentiment,
                })
            except Exception:
                continue
        return {"sentiment_results": results}

    def update_storage(state: _State) -> dict:
        for item in state["sentiment_results"]:
            try:
                sentiment_text = _sentiment_to_text(item["sentiment"])
                if not sentiment_text:
                    continue
                sentiment_embedding = embed(sentiment_text)
                article = item["article"]
                chroma_store.insert_chunks(
                    article_id=item["article_id"],
                    story_group_id=article.get("story_group_id") or 0,
                    source_name=article.get("source_name", ""),
                    published_at=article.get("published_at", ""),
                    field_texts={"sentiment": sentiment_text},
                    field_embeddings={"sentiment": sentiment_embedding},
                )
            except Exception:
                continue
        return {"processed_count": len(state["sentiment_results"])}

    graph = StateGraph(_State)
    graph.add_node("load_recent_articles", load_recent_articles)
    graph.add_node("fetch_and_analyse", fetch_and_analyse)
    graph.add_node("update_storage", update_storage)

    graph.set_entry_point("load_recent_articles")
    graph.add_edge("load_recent_articles", "fetch_and_analyse")
    graph.add_edge("fetch_and_analyse", "update_storage")
    graph.add_edge("update_storage", END)

    return graph.compile()


def run_sentiment_pipeline(sqlite_store, chroma_store) -> dict:
    pipeline = build_sentiment_pipeline(sqlite_store, chroma_store)
    return pipeline.invoke({
        "articles": [],
        "sentiment_results": [],
        "processed_count": 0,
    })

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

import config
from agents.news_parse_agent import parse_articles as _parse_articles
from tools.article_filter import relevance_score as _relevance_score
from tools.rss_fetcher import fetch_articles
from tools.embedder import embed
from tools.story_clustering import assign_story_group

logger = logging.getLogger(__name__)


class _State(TypedDict):
    feed_urls: list[str]
    raw_articles: list[dict]
    new_articles: list[dict]
    parsed_articles: list[dict]
    stored_count: int


def _get_field_texts(enriched: dict) -> dict[str, str]:
    return {
        "summary": enriched.get("summary", ""),
        "whats_new": enriched.get("whats_new", ""),
        "concepts": ", ".join(enriched.get("key_concepts", [])),
        "use_cases": ", ".join(enriched.get("use_cases", [])),
    }


def build_news_pipeline(sqlite_store, chroma_store):
    def fetch_rss(state: _State) -> dict:
        articles = fetch_articles(state["feed_urls"])
        logger.info("RSS fetch: %d articles from %d feeds", len(articles), len(state["feed_urls"]))
        return {"raw_articles": articles}

    def dedup_check(state: _State) -> dict:
        new = [a for a in state["raw_articles"] if not sqlite_store.url_hash_exists(a["url_hash"])]
        logger.info("Dedup: %d new / %d total", len(new), len(state["raw_articles"]))
        return {"new_articles": new}

    def parse_articles_node(state: _State) -> dict:
        parsed = []
        for article in state["new_articles"]:
            sqlite_store.insert_raw_article(article)
            db_article = sqlite_store.get_raw_article_by_url_hash(article["url_hash"])
            if db_article is None:
                continue
            article_with_id = {**article, "id": db_article["id"]}
            score = _relevance_score(article_with_id)
            if score < config.ARTICLE_FILTER_THRESHOLD:
                logger.warning(
                    "Article filtered (score=%.3f): %s",
                    score, article.get("title", "?")[:80],
                )
                continue
            logger.info(
                "Article passed filter (score=%.3f): %s",
                score, article.get("title", "?")[:80],
            )
            try:
                results = _parse_articles([article_with_id])
                if results:
                    parsed.extend(results)
                    logger.info("Parsed: %s", article.get("title", "untitled")[:80])
                else:
                    logger.warning("Parse returned empty for: %s", article.get("title", "?")[:60])
            except Exception as exc:
                logger.warning("Parse failed for %s: %s", article.get("title", "?")[:60], exc)
                continue
        logger.info("Parsing complete: %d articles enriched", len(parsed))
        return {"parsed_articles": parsed}

    def cluster_and_store(state: _State) -> dict:
        count = 0
        for item in state["parsed_articles"]:
            try:
                article_id = item["article_id"]
                db_article = sqlite_store._get_conn().execute(
                    "SELECT * FROM raw_articles WHERE id = ?", (article_id,)
                ).fetchone()
                if db_article is None:
                    continue
                raw = dict(db_article)

                summary = item.get("summary", "")
                summary_embedding = embed(summary)

                story_group_id = assign_story_group(
                    article_id=article_id,
                    summary_embedding=summary_embedding,
                    published_at=raw.get("published_at", ""),
                    sqlite_store=sqlite_store,
                    chroma_store=chroma_store,
                )

                sqlite_store.insert_enriched_article(article_id, item, story_group_id)

                field_texts = _get_field_texts(item)
                field_embeddings = {
                    field: embed(text)
                    for field, text in field_texts.items()
                    if text
                }

                chroma_store.insert_chunks(
                    article_id=article_id,
                    story_group_id=story_group_id,
                    source_name=raw.get("source_name", ""),
                    published_at=raw.get("published_at", ""),
                    field_texts=field_texts,
                    field_embeddings=field_embeddings,
                )
                count += 1
                logger.info("Stored article_id=%d (story_group=%d)", article_id, story_group_id)
            except Exception as exc:
                logger.warning("Store failed for article_id=%s: %s", item.get("article_id"), exc)
                continue
        logger.info("Pipeline complete: %d articles stored", count)
        return {"stored_count": count}

    graph = StateGraph(_State)
    graph.add_node("fetch_rss", fetch_rss)
    graph.add_node("dedup_check", dedup_check)
    graph.add_node("parse_articles", parse_articles_node)
    graph.add_node("cluster_and_store", cluster_and_store)

    graph.set_entry_point("fetch_rss")
    graph.add_edge("fetch_rss", "dedup_check")
    graph.add_edge("dedup_check", "parse_articles")
    graph.add_edge("parse_articles", "cluster_and_store")
    graph.add_edge("cluster_and_store", END)

    return graph.compile()


def run_news_pipeline(feed_urls: list[str], sqlite_store, chroma_store) -> dict:
    pipeline = build_news_pipeline(sqlite_store, chroma_store)
    return pipeline.invoke({
        "feed_urls": feed_urls,
        "raw_articles": [],
        "new_articles": [],
        "parsed_articles": [],
        "stored_count": 0,
    })

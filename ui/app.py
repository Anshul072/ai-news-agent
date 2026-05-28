import sys
import os
import logging
import threading
import time
from datetime import datetime
from itertools import groupby

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import streamlit as st

from storage.sqlite_store import SQLiteStore
from storage.chroma_store import ChromaStore
from agents.rag_agent import answer_query
from scheduler import trigger_news_pipeline, trigger_sentiment_pipeline

st.set_page_config(page_title="AI News Agent", layout="wide")

@st.cache_resource
def _pipeline_result() -> dict:
    # Survives Streamlit hot-reloads; background threads write here, main thread reads on each rerun.
    return {"status": "", "error": ""}


@st.cache_resource
def get_stores():
    sqlite = SQLiteStore("storage/news.db")
    sqlite.init_db()
    chroma = ChromaStore()
    return sqlite, chroma


def _sentiment_badge(label: str | None, score: float | None) -> str:
    if label is None:
        return "⏳ Pending"
    emoji = {"Positive": "😊", "Negative": "😟", "Mixed": "😐", "Neutral": "😶"}.get(label, "")
    score_str = f" ({score:+.2f})" if score is not None else ""
    return f"{emoji} {label}{score_str}"


def _format_date(date_str: str | None) -> str:
    if not date_str:
        return "Unknown date"
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        return date_str[:10]


def _merge_cluster(articles: list[dict]) -> dict:
    primary = articles[0]  # already sorted by importance_score DESC

    seen_uc: set[str] = set()
    use_cases: list[str] = []
    for a in articles:
        for uc in (a.get("use_cases") or []):
            key = uc.lower().strip()
            if key not in seen_uc:
                use_cases.append(uc)
                seen_uc.add(key)

    seen_c: set[str] = set()
    concepts: list[str] = []
    explanations: dict[str, str] = {}
    for a in articles:
        for c in (a.get("key_concepts") or []):
            key = c.lower().strip()
            if key not in seen_c:
                concepts.append(c)
                seen_c.add(key)
            if c not in explanations:
                exp = (a.get("concept_explanations") or {}).get(c, "")
                if exp:
                    explanations[c] = exp

    primary_summary = (primary.get("summary") or "").strip()
    different_angles = [
        {"source": a.get("source_name", ""), "summary": (a.get("summary") or "").strip()}
        for a in articles[1:]
        if (a.get("summary") or "").strip() and (a.get("summary") or "").strip() != primary_summary
    ]

    sentiment_article = max(articles, key=lambda a: a.get("thread_count") or 0)

    return {
        "primary": primary,
        "use_cases": use_cases,
        "concepts": concepts,
        "explanations": explanations,
        "different_angles": different_angles,
        "sentiment_article": sentiment_article,
    }


def render_feed(sqlite_store: SQLiteStore) -> None:
    st.header("AI News Feed")
    clusters = sqlite_store.get_story_clusters()
    sentiment_history = sqlite_store.get_all_sentiment_history()

    if not clusters:
        st.info("No stories yet. Use the sidebar to fetch news.")
        return

    for date_key, date_clusters in groupby(clusters, key=lambda c: (c["published_at_max"] or "")[:10]):
        st.subheader(_format_date(date_key))
        for cluster in date_clusters:
            title = cluster["title"] or "Untitled Story"
            score = cluster["importance_score"] or 0
            sources = ", ".join(cluster["source_names"])
            sentiment = _sentiment_badge(cluster["sentiment_label"], cluster["sentiment_score"])
            date_min = (cluster["published_at_min"] or "")[:10]
            date_max = (cluster["published_at_max"] or "")[:10]
            date_range = date_min if date_min == date_max else f"{date_min} – {date_max}"

            header = f"**{title}** &nbsp;&nbsp; ⭐ {score}/10 &nbsp;&nbsp; {sentiment} &nbsp;&nbsp; 📰 {cluster['source_count']} sources"
            with st.expander(header):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.caption(f"Sources: {sources}")
                    st.caption(f"Published: {date_range}")

                merged = _merge_cluster(cluster["articles"])
                primary = merged["primary"]

                st.divider()
                if primary.get("summary"):
                    st.markdown(f"**Summary:** {primary['summary']}")
                if primary.get("whats_new"):
                    st.markdown(f"**What's new:** {primary['whats_new']}")
                if primary.get("who_made_it"):
                    st.markdown(f"**Who made it:** {primary['who_made_it']}")

                if merged["use_cases"]:
                    st.markdown("**Use cases:**")
                    for uc in merged["use_cases"]:
                        st.markdown(f"- {uc}")

                if merged["concepts"]:
                    st.markdown("**Key concepts:**")
                    for c in merged["concepts"]:
                        exp = merged["explanations"].get(c, "")
                        st.markdown(f"- **{c}**: {exp}" if exp else f"- {c}")

                if primary.get("importance_reasoning"):
                    st.markdown(f"**Importance:** {primary['importance_score']}/10 — {primary['importance_reasoning']}")

                if merged["different_angles"]:
                    st.markdown("**Different perspectives:**")
                    for angle in merged["different_angles"]:
                        st.markdown(f"> **{angle['source']}:** {angle['summary']}")

                article_id = merged["sentiment_article"].get("article_id")
                history = sentiment_history.get(article_id, [])
                _render_sentiment_section(merged["sentiment_article"], history)


def _render_sentiment_section(article: dict, history: list[dict]) -> None:
    if article.get("sentiment_label") is None:
        st.info("Sentiment: pending")
        return
    if article.get("thread_count") == 0:
        st.info("No Hacker News discussions found for this article.")
        return

    st.markdown(
        f"**Community sentiment:** {_sentiment_badge(article['sentiment_label'], article['sentiment_score'])} "
        f"· {article.get('excitement_level', '')}"
    )
    if article.get("top_concerns"):
        st.markdown("**Top concerns:**")
        for c in article["top_concerns"]:
            st.markdown(f"- {c}")
    if article.get("top_use_cases"):
        st.markdown("**Community use cases:**")
        for uc in article["top_use_cases"]:
            st.markdown(f"- {uc}")
    if article.get("notable_quotes"):
        st.markdown("**Notable quotes:**")
        for q in article["notable_quotes"]:
            st.markdown(f"> {q}")
    breakdown = article.get("subreddit_breakdown") or {}
    if breakdown:
        st.markdown("**Community breakdown:**")
        for sub, summary in breakdown.items():
            st.markdown(f"- **{sub}**: {summary}")
    thread_count = article.get("thread_count")
    total_comments = article.get("total_comments")
    if thread_count is not None:
        st.caption(f"Hacker News: {thread_count} threads, {total_comments} comments")
    if len(history) >= 2:
        st.markdown("**Sentiment trend:**")
        scores = [h["sentiment_score"] if h["sentiment_score"] is not None else 0.0
                  for h in history]
        st.line_chart({"score": scores}, height=120, y_label="Score")
        table_rows = [
            {
                "Date": h["scan_date"],
                "Label": h["sentiment_label"] or "—",
                "Score": f"{h['sentiment_score']:+.2f}" if h["sentiment_score"] is not None else "—",
            }
            for h in history
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)


def render_chat(sqlite_store: SQLiteStore, chroma_store: ChromaStore) -> None:
    st.header("Ask about AI News")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            _render_citations(entry["citations"])

    query = st.chat_input("Ask a question about AI news…")
    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                result = answer_query(query, sqlite_store, chroma_store, history=st.session_state.chat_history)
            answer = result.get("answer", "")
            citations = result.get("citations", [])

            if "Not enough data" in answer:
                st.warning(answer)
            else:
                st.write(answer)
            _render_citations(citations)

        st.session_state.chat_history.append({
            "question": query,
            "answer": answer,
            "citations": citations,
        })


def _render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.caption("Sources:")
    for c in citations:
        source = c.get("source_name", "Unknown")
        published = (c.get("published_at") or "")[:10]
        url = c.get("url", "")
        label = f"{source} · {published}" if published else source
        if url:
            st.caption(f"- [{label}]({url})")
        else:
            st.caption(f"- {label}")


def render_sidebar() -> None:
    with st.sidebar:
        st.title("AI News Agent")
        view = st.radio("View", ["Feed", "Chat"], label_visibility="collapsed")

        st.divider()
        st.subheader("Pipelines")

        # Sync cross-thread result into session_state (background threads can't touch session_state directly)
        _result = _pipeline_result()
        if _result["status"]:
            st.session_state.pipeline_status = _result["status"]
            st.session_state.pipeline_error = _result["error"]
            _result["status"] = ""
            _result["error"] = ""

        if "pipeline_status" not in st.session_state:
            st.session_state.pipeline_status = ""

        if st.button("Fetch news now", use_container_width=True):
            st.session_state.pipeline_status = "news_running"
            threading.Thread(target=_run_news, daemon=True).start()
            st.rerun()

        if st.button("Refresh sentiment now", use_container_width=True):
            st.session_state.pipeline_status = "sentiment_running"
            threading.Thread(target=_run_sentiment, daemon=True).start()
            st.rerun()

        status = st.session_state.get("pipeline_status", "")
        if status == "news_running":
            st.info("⏳ Fetching news…")
        elif status == "sentiment_running":
            st.info("⏳ Analysing sentiment…")
        elif status == "news_done":
            st.success("✅ News fetch complete")
        elif status == "sentiment_done":
            st.success("✅ Sentiment analysis complete")
        elif status == "news_error":
            st.error(f"❌ News pipeline failed: {st.session_state.get('pipeline_error', '')}")
        elif status == "sentiment_error":
            st.error(f"❌ Sentiment pipeline failed: {st.session_state.get('pipeline_error', '')}")

    return view


def _run_news():
    try:
        trigger_news_pipeline()
        _pipeline_result()["status"] = "news_done"
    except Exception as exc:
        result = _pipeline_result()
        result["status"] = "news_error"
        result["error"] = str(exc)


def _run_sentiment():
    try:
        trigger_sentiment_pipeline()
        _pipeline_result()["status"] = "sentiment_done"
    except Exception as exc:
        result = _pipeline_result()
        result["status"] = "sentiment_error"
        result["error"] = str(exc)


def main():
    sqlite_store, chroma_store = get_stores()
    view = render_sidebar()

    if view == "Feed":
        render_feed(sqlite_store)
    else:
        render_chat(sqlite_store, chroma_store)

    if st.session_state.get("pipeline_status") in ("news_running", "sentiment_running"):
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()

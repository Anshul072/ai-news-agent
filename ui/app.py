import sys
import os
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from storage.sqlite_store import SQLiteStore
from storage.chroma_store import ChromaStore
from agents.rag_agent import answer_query
from scheduler import trigger_news_pipeline, trigger_sentiment_pipeline

st.set_page_config(page_title="AI News Agent", layout="wide")


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


def render_feed(sqlite_store: SQLiteStore) -> None:
    st.header("AI News Feed")
    clusters = sqlite_store.get_story_clusters()

    if not clusters:
        st.info("No stories yet. Use the sidebar to fetch news.")
        return

    for cluster in clusters:
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

            for article in cluster["articles"]:
                st.divider()
                st.subheader(article.get("title", ""))
                st.caption(f"{article.get('source_name', '')} · {(article.get('published_at') or '')[:10]}")

                if article.get("summary"):
                    st.markdown(f"**Summary:** {article['summary']}")
                if article.get("whats_new"):
                    st.markdown(f"**What's new:** {article['whats_new']}")
                if article.get("who_made_it"):
                    st.markdown(f"**Who made it:** {article['who_made_it']}")
                if article.get("use_cases"):
                    st.markdown("**Use cases:**")
                    for uc in article["use_cases"]:
                        st.markdown(f"- {uc}")

                concepts = article.get("key_concepts", [])
                explanations = article.get("concept_explanations", {})
                if concepts:
                    st.markdown("**Key concepts:**")
                    for concept in concepts:
                        explanation = explanations.get(concept, "")
                        st.markdown(f"- **{concept}**: {explanation}" if explanation else f"- {concept}")

                if article.get("importance_reasoning"):
                    st.markdown(f"**Importance:** {article['importance_score']}/10 — {article['importance_reasoning']}")

                _render_sentiment_section(article)


def _render_sentiment_section(article: dict) -> None:
    if article.get("sentiment_label") is None:
        st.info("Sentiment: pending")
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
        st.markdown("**Per-subreddit breakdown:**")
        for sub, summary in breakdown.items():
            st.markdown(f"- **{sub}**: {summary}")
    thread_count = article.get("thread_count")
    total_comments = article.get("total_comments")
    if thread_count is not None:
        st.caption(f"Reddit: {thread_count} threads, {total_comments} comments")


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
                result = answer_query(query, sqlite_store, chroma_store)
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
            st.info("⏳ News pipeline running…")
        elif status == "sentiment_running":
            st.info("⏳ Sentiment pipeline running…")
        elif status == "news_done":
            st.success("✅ News pipeline complete")
        elif status == "sentiment_done":
            st.success("✅ Sentiment pipeline complete")

    return view


def _run_news():
    trigger_news_pipeline()
    st.session_state.pipeline_status = "news_done"


def _run_sentiment():
    trigger_sentiment_pipeline()
    st.session_state.pipeline_status = "sentiment_done"


def main():
    sqlite_store, chroma_store = get_stores()
    view = render_sidebar()

    if view == "Feed":
        render_feed(sqlite_store)
    else:
        render_chat(sqlite_store, chroma_store)


if __name__ == "__main__":
    main()

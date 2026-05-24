"""
One-time golden fixture generator.

Pulls a stratified sample from SQLite (1 low, 2 medium, 2 high importance with
distinct story_group_ids), calls Opus 4.7 via the Claude Code CLI to score each
rubric dimension for all three agent types, and saves JSON fixtures to
eval/golden/{news_parse,sentiment,rag}/.

Run from the project root:
    python eval/create_golden.py [--db storage/news.db]

Requires the `claude` CLI to be installed and authenticated (Claude Code).
No extra API key needed — reuses your existing Claude Code session.

After running, review fixtures in eval/golden/ and adjust any scores that look
wrong before committing them as locked ground truth for the eval system.
"""

import argparse
import json
import os
import random
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.sqlite_store import SQLiteStore
from storage.chroma_store import ChromaStore
from tools.embedder import embed

_MODEL = "claude-opus-4-7"

_STRATIFY_TARGETS = [
    ("low",    1,  3,  1),
    ("medium", 4,  6,  2),
    ("high",   7, 10,  2),
]

_RAG_QUERIES = [
    "What are the most significant recent AI breakthroughs?",
    "What concerns do developers and researchers have about recent AI developments?",
    "Which AI companies or labs have made major announcements recently?",
    "What are the most promising practical use cases for AI discussed recently?",
    "What new AI models or tools have been released recently?",
]


def _call_opus(prompt: str) -> str:
    """Call Opus 4.7 via the Claude Code CLI and return the text response."""
    result = subprocess.run(
        ["claude", "--model", _MODEL, "--print", "--output-format", "text"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {result.returncode}): {result.stderr[:300]}")
    return result.stdout.strip()


def _parse_scores(text: str, dimensions: list[str]) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {dim: {"score": 3, "reasoning": f"Parse failed: {text[:200]}"} for dim in dimensions}
    result = {}
    for dim in dimensions:
        entry = parsed.get(dim, {})
        if not isinstance(entry, dict):
            entry = {}
        try:
            score = max(1, min(5, int(entry.get("score", 3))))
        except (ValueError, TypeError):
            score = 3
        reasoning = entry.get("reasoning") or "No reasoning provided."
        result[dim] = {"score": score, "reasoning": str(reasoning)}
    return result


def _score_news_parse(article: dict) -> dict:
    output_obj = {
        "summary": article.get("summary"),
        "whats_new": article.get("whats_new"),
        "key_concepts": article.get("key_concepts", []),
        "use_cases": article.get("use_cases", []),
        "importance_score": article.get("importance_score"),
        "importance_reasoning": article.get("importance_reasoning"),
    }
    prompt = (
        "You are an expert LLM judge evaluating a news parse agent's output.\n"
        "Score each dimension 1-5 (integer). Provide a non-empty reasoning string for each.\n"
        "Return ONLY valid JSON — no markdown fences:\n\n"
        '{"summary_faithfulness": {"score": <int 1-5>, "reasoning": "<cite a specific sentence from the article>"},\n'
        ' "importance_score_calibration": {"score": <int 1-5>, "reasoning": "<string>"},\n'
        ' "key_concepts_relevance": {"score": <int 1-5>, "reasoning": "<string>"},\n'
        ' "whats_new_specificity": {"score": <int 1-5>, "reasoning": "<string>"},\n'
        ' "use_cases_plausibility": {"score": <int 1-5>, "reasoning": "<string>"}}\n\n'
        "Rubric:\n"
        "- summary_faithfulness: Does the summary accurately reflect the article without hallucinating? "
        "In reasoning, cite a specific sentence from the article that supports or contradicts the summary.\n"
        "- importance_score_calibration: Is the 1-10 importance_score appropriate? "
        "Breakthrough=high (8-10), minor update=low (1-3).\n"
        "- key_concepts_relevance: Are the key_concepts present and central in the article content?\n"
        "- whats_new_specificity: Does whats_new capture actual novelty, not just restate the summary?\n"
        "- use_cases_plausibility: Are use_cases grounded in the article, not generic AI boilerplate?\n\n"
        f"Article title: {article.get('title', '')}\n\n"
        f"Article content:\n{article.get('content', '')}\n\n"
        f"Agent output:\n{json.dumps(output_obj, indent=2)}"
    )
    text = _call_opus(prompt)
    return _parse_scores(text, [
        "summary_faithfulness",
        "importance_score_calibration",
        "key_concepts_relevance",
        "whats_new_specificity",
        "use_cases_plausibility",
    ])


def _score_sentiment(article: dict, sentiment: dict) -> dict:
    output_obj = {
        "sentiment_label": sentiment.get("sentiment_label"),
        "sentiment_score": sentiment.get("sentiment_score"),
        "excitement_level": sentiment.get("excitement_level"),
        "top_concerns": sentiment.get("top_concerns", []),
        "top_use_cases": sentiment.get("top_use_cases", []),
        "notable_quotes": sentiment.get("notable_quotes", []),
    }
    prompt = (
        "You are an expert LLM judge evaluating a sentiment analysis agent's output.\n"
        "Score each dimension 1-5 (integer). Provide a non-empty reasoning string for each.\n"
        "Return ONLY valid JSON — no markdown fences:\n\n"
        '{"label_score_consistency": {"score": <int 1-5>, "reasoning": "<string>"},\n'
        ' "concern_use_case_groundedness": {"score": <int 1-5>, "reasoning": "<string>"},\n'
        ' "quote_authenticity": {"score": <int 1-5>, "reasoning": "<string>"}}\n\n'
        "Rubric:\n"
        "- label_score_consistency: Does the sentiment_label (Positive/Negative/Mixed/Neutral) "
        "align with the numeric sentiment_score? (-1.0 to 1.0 maps to Negative..Positive)\n"
        "- concern_use_case_groundedness: Are top_concerns and top_use_cases traceable to "
        "the provided HN thread comments? If no HN threads are provided, score 3 (insufficient data).\n"
        "- quote_authenticity: Do notable_quotes read like real HN comments, not LLM-generated summaries? "
        "Look for informal tone, specific technical references, and varied sentence structure.\n\n"
        f"Article title: {article.get('title', '')}\n\n"
        "HN threads: []\n\n"
        f"Agent output:\n{json.dumps(output_obj, indent=2)}"
    )
    text = _call_opus(prompt)
    return _parse_scores(text, [
        "label_score_consistency",
        "concern_use_case_groundedness",
        "quote_authenticity",
    ])


def _score_rag(query: str, context_str: str, answer: str, citations: list) -> dict:
    output_obj = {"answer": answer, "citations": citations}
    prompt = (
        "You are an expert LLM judge evaluating a RAG agent's output.\n"
        "Score each dimension 1-5 (integer). Provide a non-empty reasoning string for each.\n"
        "Return ONLY valid JSON — no markdown fences:\n\n"
        '{"answer_relevance": {"score": <int 1-5>, "reasoning": "<string>"},\n'
        ' "faithfulness": {"score": <int 1-5>, "reasoning": "<string>"},\n'
        ' "citation_accuracy": {"score": <int 1-5>, "reasoning": "<string>"}}\n\n'
        "Rubric:\n"
        "- answer_relevance: Does the answer address what the query asked?\n"
        "- faithfulness: Is every claim in the answer supported by the retrieved context chunks? "
        "No external facts introduced?\n"
        "- citation_accuracy: Are the cited articles the ones that actually support the answer?\n\n"
        f"Query: {query}\n\n"
        f"Retrieved context chunks:\n{context_str}\n\n"
        f"Agent output:\n{json.dumps(output_obj, indent=2)}"
    )
    text = _call_opus(prompt)
    return _parse_scores(text, ["answer_relevance", "faithfulness", "citation_accuracy"])


def _stratified_sample(articles: list[dict]) -> list[dict]:
    buckets = {
        "low":    [a for a in articles if 1 <= (a.get("importance_score") or 0) <= 3],
        "medium": [a for a in articles if 4 <= (a.get("importance_score") or 0) <= 6],
        "high":   [a for a in articles if 7 <= (a.get("importance_score") or 0) <= 10],
    }
    for b in buckets.values():
        random.shuffle(b)

    selected: list[dict] = []
    seen_groups: set = set()

    for label, _lo, _hi, target in _STRATIFY_TARGETS:
        count = 0
        for article in buckets[label]:
            if count >= target:
                break
            gid = article.get("story_group_id")
            if gid is not None and gid in seen_groups:
                continue
            selected.append(article)
            if gid is not None:
                seen_groups.add(gid)
            count += 1

    return selected


def _run_rag_query(query: str, sqlite_store: SQLiteStore, chroma_store: ChromaStore) -> tuple[str, str, list]:
    """Returns (answer, context_str, citations). Returns empty strings if no results."""
    query_embedding = embed(query)
    chunks = chroma_store.search(query_embedding, n_results=5)
    if not chunks:
        return "", "", []

    article_ids = list(dict.fromkeys(c["article_id"] for c in chunks))
    context_parts = []
    citations = []
    for article_id in article_ids:
        raw = sqlite_store.get_raw_article(article_id)
        enriched = sqlite_store.get_enriched_article(article_id)
        if raw is None:
            continue
        summary = enriched.get("summary", "") if enriched else ""
        context_parts.append(
            f"[Article {article_id}] {raw.get('title', '')} ({raw.get('source_name', '')})\n"
            f"Summary: {summary}"
        )
        citations.append({
            "article_id": article_id,
            "title": raw.get("title", ""),
            "source_name": raw.get("source_name", ""),
            "url": raw.get("url", ""),
        })

    if not context_parts:
        return "", "", []

    context_str = "\n\n".join(context_parts)

    from agents.rag_agent import answer_query
    result = answer_query(query, sqlite_store, chroma_store)
    answer = result.get("answer", "")
    answer_citations = result.get("citations") or citations
    return answer, context_str, answer_citations


def _save_fixture(path: str, fixture: dict) -> None:
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval golden fixtures using Opus 4.7")
    parser.add_argument("--db", default="storage/news.db", help="Path to SQLite DB")
    args = parser.parse_args()

    sqlite_store = SQLiteStore(db_path=args.db)
    sqlite_store.init_db()

    all_articles = sqlite_store.get_all_enriched_articles()
    if not all_articles:
        print("No enriched articles in DB. Run the news pipeline first.")
        sys.exit(1)

    sample = _stratified_sample(all_articles)
    if not sample:
        print("Could not produce stratified sample — not enough articles in DB.")
        sys.exit(1)

    print(f"\nStratified sample ({len(sample)} articles):")
    for a in sample:
        print(f"  [{a.get('importance_score', '?'):>2}] {a.get('title', '')[:80]}")

    # --- news_parse fixtures ---
    print("\n=== news_parse fixtures ===")
    for article in sample:
        aid = article.get("article_id")
        print(f"  Scoring article {aid}: {article.get('title', '')[:60]}...")
        scores = _score_news_parse(article)
        fixture = {
            "agent_name": "news_parse_agent",
            "article_id": aid,
            "story_group_id": article.get("story_group_id"),
            "importance_score": article.get("importance_score"),
            "inputs": {
                "article_title": article.get("title", ""),
                "article_content": article.get("content", ""),
            },
            "output": {
                "summary": article.get("summary"),
                "whats_new": article.get("whats_new"),
                "key_concepts": article.get("key_concepts", []),
                "use_cases": article.get("use_cases", []),
                "importance_score": article.get("importance_score"),
                "importance_reasoning": article.get("importance_reasoning"),
            },
            "golden_scores": scores,
        }
        _save_fixture(f"eval/golden/news_parse/article_{aid}.json", fixture)

    # --- sentiment fixtures ---
    print("\n=== sentiment fixtures ===")
    for article in sample:
        aid = article.get("article_id")
        sentiment = sqlite_store.get_sentiment(aid)
        if sentiment is None:
            print(f"  Skipping article {aid} — no sentiment data")
            continue
        print(f"  Scoring sentiment for article {aid}: {article.get('title', '')[:60]}...")
        scores = _score_sentiment(article, sentiment)
        fixture = {
            "agent_name": "sentiment_agent",
            "article_id": aid,
            "story_group_id": article.get("story_group_id"),
            "inputs": {
                "article_title": article.get("title", ""),
                "hn_threads": [],
            },
            "output": {
                "sentiment_label": sentiment.get("sentiment_label"),
                "sentiment_score": sentiment.get("sentiment_score"),
                "excitement_level": sentiment.get("excitement_level"),
                "top_concerns": sentiment.get("top_concerns", []),
                "top_use_cases": sentiment.get("top_use_cases", []),
                "notable_quotes": sentiment.get("notable_quotes", []),
            },
            "golden_scores": scores,
        }
        _save_fixture(f"eval/golden/sentiment/article_{aid}.json", fixture)

    # --- RAG fixtures ---
    print("\n=== RAG fixtures ===")
    chroma_store = ChromaStore()
    saved = 0
    for i, query in enumerate(_RAG_QUERIES):
        print(f"  Query {i + 1}: {query[:60]}...")
        answer, context_str, citations = _run_rag_query(query, sqlite_store, chroma_store)
        if not answer:
            print(f"    Skipped — no RAG results")
            continue
        scores = _score_rag(query, context_str, answer, citations)
        fixture = {
            "agent_name": "rag_agent",
            "inputs": {
                "query": query,
                "context_chunks": context_str,
            },
            "output": {
                "answer": answer,
                "citations": citations,
            },
            "golden_scores": scores,
        }
        _save_fixture(f"eval/golden/rag/query_{i + 1}.json", fixture)
        saved += 1

    print(f"\nDone. {len(sample)} news_parse, up to {len(sample)} sentiment, {saved} RAG fixtures saved.")
    print("Review eval/golden/ and adjust any scores before committing.")


if __name__ == "__main__":
    main()

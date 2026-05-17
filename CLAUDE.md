# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent AI news workflow: fetches AI news from RSS feeds, enriches stories with structured insights via Gemini Flash, tracks evolving Reddit community sentiment, stores everything locally, and exposes a RAG-based Q&A interface via Streamlit.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
```

Required env vars: `GEMINI_API_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`.

## Running Things

```bash
# Start the Streamlit UI (feed + chat views)
streamlit run ui/app.py

# Start the scheduler (fires both pipelines on their configured schedules)
python scheduler.py

# Manually trigger pipelines
python -c "from scheduler import trigger_news_pipeline; trigger_news_pipeline()"
python -c "from scheduler import trigger_sentiment_pipeline; trigger_sentiment_pipeline()"

# Run tests
pytest
pytest tests/test_sqlite_store.py          # single test file
pytest -k "test_dedup"                     # single test by name
```

## Architecture

Two independent LangGraph graphs triggered by APScheduler:

**Graph 1 — News Pipeline** (Mon/Wed/Fri 23:00):
```
RSS Fetch → URL Dedup Check → News Parse Agent → Story Clustering → SQLite + ChromaDB
```

**Graph 2 — Sentiment Pipeline** (daily 08:00, rolling 7-day window):
```
Load recent articles (SQLite) → Reddit Search → Sentiment Agent → Update SQLite + ChromaDB
```

The graphs are intentionally separate: different schedules, independent failure domains.

## Storage Layer

Two stores, always used together:

- **SQLite** (`storage/sqlite_store.py`) — all structured data: raw articles, enriched fields, sentiment records, story groups, URL dedup hashes. Single source of truth for article data.
- **ChromaDB** (`storage/chroma_store.py`) — field-based vector chunks for RAG. Each article field (`summary`, `whats_new`, `concepts`, `sentiment`, `use_cases`) is a separate document. The `article_id` metadata field is the join key back to SQLite.

## RAG Pattern

Query → embed (text-embedding-004) → ChromaDB top-k field chunks → fetch full articles from SQLite by `article_id` → Gemini Flash generates answer with citations.

Field-based chunking means a question about sentiment retrieves sentiment chunks; a question about concepts retrieves concept chunks. Single retrieval pass handles multi-aspect queries.

## Story Clustering

After parsing, each article summary is embedded and compared (cosine similarity) against the last 3 days of ChromaDB entries. Threshold: 0.85. Articles above threshold share a `story_group_id` in SQLite; source count is incremented. All articles are kept — clustering enriches, never discards.

## Key Design Decisions

- **Single LLM**: Gemini Flash for all tasks (parsing, sentiment, RAG). Free tier: 1,500 req/day — well within budget at current schedule.
- **No hybrid search**: BM25 skipped — semantically rich structured fields make keyword search redundant.
- **No web search fallback**: RAG answers from the local DB only. User queries external tools for data not in the DB.
- **Dedup by URL hash**: SHA-256 of article URL stored in SQLite. Checked before any processing.
- **Sentiment re-scans**: The sentiment pipeline re-scans all articles within the 7-day window daily — community discussion matures over days, so a single snapshot is insufficient.

## Testing Conventions

- SQLite Store: integration tests with in-memory SQLite (`:memory:`) — no mocking, real behavior.
- ChromaDB Store: integration tests with ephemeral in-memory collection.
- RSS Fetcher, Reddit Fetcher, Gemini agents: mock at the API boundary with fixtures — no live network calls in tests.
- Pipeline integration tests: mock all external APIs, assert final SQLite + ChromaDB state from known input fixtures.

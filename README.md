# AI News Agent Workflow

A multi-agent pipeline that fetches AI news from RSS feeds, enriches each story with structured insights, tracks Hacker News community sentiment, and exposes a RAG-based Q&A interface via Streamlit.

## What it does

1. **Fetches** AI news from configurable RSS feeds on a schedule
2. **Filters** articles by semantic relevance to AI/ML topics before any LLM processing
3. **Enriches** each article: summary, what's new, key concepts, use cases, importance score (1–10)
4. **Clusters** related stories across sources using embedding similarity
5. **Tracks sentiment** from Hacker News threads — re-scanned daily as discussion matures
6. **Answers questions** via a RAG chat interface grounded in the local article database

## Architecture

Two independent LangGraph pipelines triggered by APScheduler:

```
News Pipeline (Mon/Wed/Fri 23:00)
  RSS Fetch → Relevance Filter → News Parse Agent → Story Clustering → SQLite + ChromaDB

Sentiment Pipeline (daily 08:00, rolling 7-day window)
  Load recent articles → HN Search → Sentiment Agent → Update SQLite + ChromaDB
```

The RAG Q&A layer sits on top of both stores: query → embed → ChromaDB top-k field chunks → fetch full articles from SQLite → Gemini Flash generates answer with citations.

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph + APScheduler |
| LLM (parse, sentiment, RAG) | Gemini Flash (Google genai SDK) |
| LLM (news parse) | Groq — Llama 3.3 70B |
| Embeddings | `sentence-transformers` (local, no API cost) |
| Vector store | ChromaDB (in-process) |
| Structured store | SQLite |
| UI | Streamlit |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

Required environment variables:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for news parse agent |
| `GEMINI_API_KEY` | Google Gemini API key for sentiment, RAG, and eval judge |

Optional:

| Variable | Default | Description |
|---|---|---|
| `ARTICLE_FILTER_THRESHOLD` | `0.5` | Cosine similarity cutoff for relevance pre-filter |
| `CLUSTERING_THRESHOLD` | `0.75` | Cosine similarity threshold for story grouping |
| `SENTIMENT_WINDOW_DAYS` | `7` | Rolling window for sentiment re-scans |
| `NEWS_SCHEDULE` | `0 23 * * 1,3,5` | Cron schedule for news pipeline |
| `SENTIMENT_SCHEDULE` | `0 8 * * *` | Cron schedule for sentiment pipeline |

## Running

```bash
# Streamlit UI (feed + chat)
streamlit run ui/app.py

# Scheduler (runs both pipelines on their configured schedules)
python scheduler.py

# Trigger pipelines manually
python -c "from scheduler import trigger_news_pipeline; trigger_news_pipeline()"
python -c "from scheduler import trigger_sentiment_pipeline; trigger_sentiment_pipeline()"
```

## Eval

Two independent evals:

```bash
# LLM-as-judge eval for news_parse, sentiment, and RAG agents
python eval/eval.py

# Binary classification eval for the article relevance filter
python eval/filter_eval.py
```

The **agent eval** (`eval/eval.py`) uses Gemini Flash as a judge against a golden fixture dataset, scoring dimensions like summary faithfulness, importance calibration, and citation accuracy. Results go to `eval/results/eval_<timestamp>.json`.

The **filter eval** (`eval/filter_eval.py`) runs a 20-article labeled golden dataset through the relevance filter and reports precision, recall, F1, and a threshold sensitivity table across five thresholds. No LLM calls — purely embedding-based. Results go to `eval/results/filter_eval_<timestamp>.json`.

## Tests

```bash
pytest                                      # all tests
pytest tests/test_sqlite_store.py           # single file
pytest -k "test_dedup"                      # single test
```

Tests use in-memory SQLite and ephemeral ChromaDB collections — no live database state required. External API calls (Groq, Gemini, HN) are mocked at the boundary.

## Storage layout

```
storage/news.db          SQLite — articles, enriched fields, sentiment, story groups
storage/chroma/          ChromaDB — field-based vector chunks for RAG
```

Each article field (`summary`, `whats_new`, `key_concepts`, `sentiment`, `use_cases`) is stored as a separate ChromaDB document. The `article_id` metadata field is the join key back to SQLite.

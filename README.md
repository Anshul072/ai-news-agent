# AI News Agent Workflow

**Keeping up with AI moves faster than anyone can read.** I built this to turn the daily firehose of AI news into something structured and queryable: a multi-agent pipeline that fetches stories from curated RSS feeds, enriches each one into structured insights, tracks how the Hacker News community actually reacts, and lets me ask questions in plain English and get grounded, cited answers back — instead of doom-scrolling ten newsletters.


### Real sample Q&A

Captured by running the RAG pipeline locally against the existing `storage/` database:

> **Q: What are the most important recent AI model releases and why do they matter?**
>
> Recent AI model releases that stand out include Anthropic's Claude Opus 4.8, which focuses on honesty and transparency, and Fable 5 and Mythos 5, which were taken offline due to US government orders citing national security concerns. The release of Claude Opus 4.8 matters because it signifies a notable advancement in AI model development, specifically targeting the issue of honesty and transparency. The incident with Fable 5 and Mythos 5 highlights the US government's control over AI technology and its potential impact on national security. Another significant development is Hark's $700M Series A funding for its universal AI interface... These releases matter because they showcase the ongoing efforts to advance AI technologies, address concerns around AI safety and security, and promote the development of more reliable and transparent AI systems.
>
> **Sources:**
> - [366] All the news about Anthropic's new AI fight with the White House — *The Verge*
> - [374] Import AI 461: "Alignment is not on track"; FrontierCode; and synthetic research interns — *Import AI*

The answer is synthesised across multiple retrieved articles and cites the specific source IDs it drew from — it does not paraphrase a single summary.

### Eval metrics (latest run — `eval/results/eval_20260626T194526Z.json`)

LLM-as-judge scores, 1–5 scale, averaged across sampled articles:

| Agent | Dimension | Score |
|---|---|---|
| **RAG** | answer relevance | **5.0** |
| **RAG** | faithfulness | **5.0** |
| **RAG** | citation accuracy | **5.0** |
| **Sentiment** | label/score consistency | 5.0 |
| **Sentiment** | concern/use-case groundedness | 4.5 |
| **Sentiment** | thread relevance | 5.0 |
| **Sentiment** | quote authenticity | 4.0 |
| **News parse** | key concepts relevance | 4.3 |
| **News parse** | summary faithfulness | 3.9 |
| **News parse** | importance calibration | 3.6 |
| **News parse** | use-cases plausibility | 3.3 |
| **News parse** | what's-new specificity | 2.6 |

The RAG and sentiment agents score strongly. The lower news-parse dimensions are honest and traceable: several fixture articles have **empty scraped bodies** (paywalled or JS-rendered sources the scraper couldn't reach), so the model was scored on title-only input — the eval correctly penalises the resulting generic output. This is a real, known limitation surfaced *by* the eval harness, not hidden from it.

## Architecture

Two independent LangGraph pipelines triggered by APScheduler:

```
News Pipeline (Mon/Wed/Fri 23:00)
  RSS Fetch → URL Dedup → Content Scrape → Relevance Filter → News Parse Agent → Story Clustering → SQLite + ChromaDB

Sentiment Pipeline (daily 08:00, rolling 7-day window)
  Load recent articles → HN Search (Algolia) → Sentiment Agent → Update SQLite + ChromaDB
```

The two graphs are intentionally separate — different schedules, independent failure domains. The RAG Q&A layer sits on top of both stores:

```
query → embed (local ONNX) → ChromaDB top-k field chunks → fetch full articles from SQLite → Groq Llama 3.3 70B generates answer with citations
```

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph + APScheduler |
| LLM — all 3 agents (parse, sentiment, RAG) | Groq — Llama 3.3 70B (`llama-3.3-70b-versatile`) |
| LLM — eval judge only | Gemini Flash (Google `genai` SDK) |
| Embeddings | `fastembed` — local `BAAI/bge-base-en-v1.5` via ONNX Runtime (no API cost) |
| Vector store | ChromaDB (in-process, persistent) |
| Structured store | SQLite |
| Community sentiment | Hacker News Algolia API (no credentials) |
| UI | Streamlit |

## Design decisions

A few choices worth explaining, each backed by the code or the eval data:

**1. One LLM (Groq Llama 3.3 70B) for all three agents — not a different model per task.**
News parsing, sentiment analysis, and RAG generation all run on the same Groq model (`agents/*.py`). A single model keeps prompt-engineering effort, rate-limit handling, and failure modes in one place. Because everything runs on Groq's free tier, the parse agent paces itself with a 2.1s inter-request delay to stay under the 30 RPM limit and backs off on `429`s (`news_parse_agent.py`).

**2. Local ONNX embeddings (`fastembed`) instead of `sentence-transformers`.**
This one was forced by a real bug. `sentence-transformers` pulls in PyTorch, whose *import alone* is ~20s and holds the GIL in long bursts — loading it in-process froze the Streamlit UI for the entire load window. `fastembed` imports in ~1s, loads the same `BAAI/bge-base-en-v1.5` model, and produces vectors that are cosine-identical to the old ones, so already-stored ChromaDB embeddings stayed valid across the switch. ONNX thread count is also capped so a big embedding batch can't starve the UI thread. (See `tools/embedder.py` and commit `7365d50`.)

**3. Field-based ChromaDB chunking — one document per article field, not per whole article.**
Each article is split into separate `summary` / `whats_new` / `concepts` / `use_cases` / `sentiment` chunks, each embedded independently (`pipelines/news_pipeline.py`, `storage/chroma_store.py`). A question about *community concerns* retrieves sentiment chunks; a question about *what's new* retrieves whats_new chunks. One retrieval pass handles multi-aspect queries without whole-article noise diluting the match.

**4. A different model family for the eval judge than for the system under test.**
The three agents run on Groq/Llama; the eval judge runs on Gemini Flash (`eval/judge.py`). Using a separate model family to grade the pipeline avoids a model rewarding its own stylistic habits (self-preference bias). The judge scores fixed 1–5 rubrics per agent (5 dimensions for news-parse, 4 for sentiment, 3 for RAG) and never raises on malformed output — it degrades to a low score with the raw response captured for debugging.

**5. The relevance filter threshold (0.5) is tuned against a labelled eval, not guessed.**
Before any LLM call, each article is scored by max cosine similarity against seed phrases and dropped below `ARTICLE_FILTER_THRESHOLD` (`tools/article_filter.py`). `eval/filter_eval.py` sweeps the threshold over a 20-article labelled golden set and reports precision/recall/F1 at each step. `0.5` is the sweet spot — F1 **0.909**, accuracy **0.9**, recall **1.0** (zero relevant articles dropped). Loosening to 0.45 lets in more false positives (F1 0.87); tightening to 0.55 starts dropping real articles (recall 0.9). The current false positives are adjacent-but-not-AI tech (React 19, PostgreSQL 17) — an acceptable trade to keep recall perfect.

> Note: the story-clustering threshold (`0.75`) is a chosen default, **not** eval-tuned — there's no clustering eval harness yet. It's a candidate for the same treatment.

**6. APScheduler, not Celery/cron.**
The whole system is a single-node local app. APScheduler runs the two cron-scheduled pipelines *in-process* (`scheduler.py`) with no external broker, Redis, or worker pool to stand up. For two jobs a day on one machine, a distributed task queue would be pure overhead.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
```

Required for the app:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — powers all three agents. The app won't start without it. |

Only needed if you run the eval harness under `eval/`:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key for the LLM-as-judge eval |

Hacker News sentiment uses the public Algolia API and needs no credentials.

Optional configuration (defaults shown):

| Variable | Default | Description |
|---|---|---|
| `RSS_FEEDS` | 12 curated sources | Comma-separated RSS URLs (Arxiv cs.AI/cs.LG, OpenAI, Anthropic, DeepMind, Meta AI, Mistral, Ars Technica, The Verge, The Batch, Import AI) |
| `ARTICLE_FILTER_THRESHOLD` | `0.5` | Cosine-similarity cutoff for the relevance pre-filter |
| `CLUSTERING_THRESHOLD` | `0.75` | Cosine-similarity threshold for story grouping |
| `SENTIMENT_WINDOW_DAYS` | `7` | Rolling window for sentiment re-scans |
| `NEWS_SCHEDULE` | `0 23 * * 1,3,5` | Cron schedule for the news pipeline |
| `SENTIMENT_SCHEDULE` | `0 8 * * *` | Cron schedule for the sentiment pipeline |

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

The **agent eval** (`eval/eval.py`) samples articles — by default 5 from a golden fixture set plus 5 recent live articles from the DB — runs each agent, and has **Gemini Flash judge** every output against per-agent rubrics (summary faithfulness, importance calibration, citation accuracy, and so on). It prints an averaged score table, warns on any dimension below 3.0/5.0, and writes full per-article results to `eval/results/eval_<timestamp>.json`.

The **filter eval** (`eval/filter_eval.py`) runs a 20-article labelled golden dataset through the relevance filter and reports precision, recall, F1, plus a threshold-sensitivity table across five thresholds. No LLM calls — purely embedding-based. Results go to `eval/results/filter_eval_<timestamp>.json`.

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
storage/chroma_db/       ChromaDB — field-based vector chunks for RAG
```

Each article field (`summary`, `whats_new`, `concepts`, `use_cases`, `sentiment`) is stored as a separate ChromaDB document. The `article_id` metadata field is the join key back to SQLite.
</content>
</invoke>

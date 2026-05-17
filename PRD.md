# PRD: AI News Agent Workflow

## Problem Statement

Keeping up with AI advancements is time-consuming and fragmented. News is spread across dozens of sources, community reactions are buried in Reddit threads, and there is no unified way to query across accumulated knowledge. Reading raw articles gives facts but not context; Reddit gives sentiment but not structure. There is no single place to ask "what did the community think about X, and what are the practical use cases people are imagining?"

## Solution

An agentic workflow that automatically fetches AI news from curated RSS feeds, enriches each story with structured insights (summary, key concepts explained with analogies, use cases, importance score), then separately tracks evolving community sentiment from Reddit across a rolling 7-day window. All data is stored locally and exposed through a Streamlit chat interface powered by a RAG system, allowing the user to ask natural language questions across the accumulated knowledge base.

## User Stories

1. As a user, I want the system to automatically fetch AI news from curated RSS feeds three times a week, so that I do not have to manually check multiple sources.
2. As a user, I want fetched articles to be deduplicated by URL hash, so that the same article is never processed or stored twice.
3. As a user, I want articles covering the same story from different sources to be grouped into a single story cluster, so that I see one unified story card rather than five near-identical entries.
4. As a user, I want each news story to include a plain English summary, so that I can quickly understand what happened without reading the full article.
5. As a user, I want each story to clearly state what is specifically new or novel about the advancement, so that I understand what changed from the prior state of the art.
6. As a user, I want key technical concepts extracted from each article and explained with analogies and real-world examples, so that I can understand unfamiliar ideas without prior domain knowledge.
7. As a user, I want to know who or which organization is behind each advancement, so that I can track which labs and researchers are leading in specific areas.
8. As a user, I want practical use cases identified for each advancement, so that I can understand how it might apply to my own work or interests.
9. As a user, I want each story to have an agent-generated importance score (1–10) with reasoning, so that I can prioritize which stories to explore further.
10. As a user, I want the sentiment pipeline to run daily against a rolling 7-day window of articles, so that I can see how community reactions evolve over time, not just at the moment of publication.
11. As a user, I want Reddit sentiment tracked across r/artificial, r/MachineLearning, r/LocalLLaMA, r/singularity, r/OpenAI, and r/technology, so that I get a broad cross-section of community opinion.
12. As a user, I want an overall sentiment classification (Positive / Negative / Mixed / Neutral) and a numeric score (-1.0 to +1.0) for each story, so that I can quickly gauge community mood.
13. As a user, I want to see the top concerns the community is raising about each advancement, so that I understand the risks and criticisms being discussed.
14. As a user, I want to see the top use cases the community is imagining for each advancement, so that I get crowdsourced application ideas beyond what the original article suggested.
15. As a user, I want 2–3 notable verbatim Reddit quotes captured per story, so that I can hear the community's voice directly.
16. As a user, I want sentiment broken down per subreddit, so that I can see how different communities (researchers vs. enthusiasts vs. builders) react differently to the same news.
17. As a user, I want the number of Reddit threads and total comments found per story tracked, so that high-engagement stories are surfaced as more important.
18. As a user, I want a Streamlit-based chat interface where I can ask natural language questions about stored news and sentiment, so that I can explore the knowledge base conversationally.
19. As a user, I want the RAG system to retrieve precise, field-specific chunks (e.g., sentiment chunks for sentiment questions, concept chunks for explanation questions), so that answers are targeted and relevant.
20. As a user, I want full article context passed to the LLM when generating answers, so that responses are rich and grounded rather than based on snippets alone.
21. As a user, I want the Streamlit feed to show stories grouped by cluster with all source outlets listed, so that I can see the breadth of coverage at a glance.
22. As a user, I want both pipelines to run automatically on their configured schedules without manual intervention, so that my knowledge base stays fresh without effort.
23. As a user, I want to be able to manually trigger either pipeline at any time, so that I can fetch fresh data outside the scheduled windows when I need it.
24. As a user, I want all data stored locally (SQLite + ChromaDB), so that I have full control over my data and incur no cloud storage costs.
25. As a user, I want the system to use only free-tier APIs (Gemini Flash, Google text-embedding-004, PRAW), so that running costs remain zero.

## Implementation Decisions

### Agent Framework
LangGraph is used as the orchestration framework. The workflow is modeled as two independent directed graphs, each triggered by APScheduler on separate schedules.

### Two Separate LangGraph Graphs

**Graph 1 — News Pipeline** (runs Mon/Wed/Fri nights):
```
RSS Fetch → URL Dedup Check → News Parse Agent → Story Clustering → Storage
```

**Graph 2 — Sentiment Pipeline** (runs daily, morning):
```
Load articles from last 7 days (SQLite) → Reddit Keyword Search → Sentiment Agent → Update Storage
```

The graphs are kept separate because they run on different schedules and have independent failure domains. A Reddit outage does not block news parsing; a news fetch failure does not block sentiment refresh.

### LLM
Gemini Flash (via Google AI Studio free tier) is used for all agent tasks: news parsing, concept explanation, sentiment analysis, and RAG answer generation. A single LLM is used across all tasks to minimize integration complexity and API key management. Rate limits (1,500 requests/day, 1M tokens/minute) are not a concern at the expected workflow frequency.

### Embeddings
Google `text-embedding-004` via the same Gemini API key is used for all embeddings. This covers both indexing (article fields → ChromaDB) and retrieval (user query → similarity search).

### RSS Fetcher Module
Fetches and parses a configurable list of 8 RSS feeds using `feedparser`. Returns normalized raw article objects. Responsible only for fetching — deduplication is handled downstream.

Configured feeds:
- ArXiv cs.AI, ArXiv cs.LG
- MIT Technology Review
- VentureBeat AI
- Hugging Face Blog
- The Verge AI
- Google DeepMind Blog
- OpenAI Blog

### Deduplication
URL-based: a SHA-256 hash of each article URL is stored in SQLite. Before parsing, the hash is checked. Articles with existing hashes are skipped. This prevents the same article from being processed across multiple pipeline runs.

### Story Clustering
After parsing, each article's summary is embedded and compared (cosine similarity) against articles stored in the last 3 days. If similarity exceeds 0.85, the article is assigned to an existing `story_group_id`. Otherwise, a new story group is created. Source count per story group is tracked and feeds into the importance score. All articles are retained — clustering enriches rather than discards.

### Article Schema (extracted by News Parse Agent)
Each article produces the following structured fields stored in SQLite:
- `summary` — 2–3 sentence plain English summary
- `whats_new` — the specific advancement or claim
- `key_concepts` — list of 3–5 concept names
- `concept_explanations` — each concept explained with analogies and examples
- `who_made_it` — organization or researchers behind the work
- `use_cases` — practical applications
- `importance_score` — integer 1–10 with agent reasoning
- `story_group_id` — cluster identifier
- `source_url`, `source_name`, `published_at`, `fetched_at`

### Sentiment Schema (extracted by Sentiment Agent)
Per article, after Reddit search:
- `sentiment_label` — Positive / Negative / Mixed / Neutral
- `sentiment_score` — float -1.0 to +1.0
- `excitement_level` — Hyped / Skeptical / Indifferent
- `top_concerns` — list of strings
- `top_use_cases` — list of strings (community-imagined, distinct from article use cases)
- `notable_quotes` — list of 2–3 verbatim Reddit comments
- `subreddit_breakdown` — per-subreddit sentiment summary
- `thread_count` — number of Reddit threads found
- `total_comments` — total comments across threads
- `last_scanned_at` — timestamp of most recent sentiment scan

### Reddit Fetcher Module
Given a list of keyword terms (extracted by Gemini from the article title), searches configured subreddits via PRAW. Returns top threads and top comments per thread. Handles PRAW authentication and rate limiting internally. Returns empty results gracefully if no threads are found — the sentiment agent handles the "no Reddit presence" case explicitly.

### SQLite Store Module
Single module managing all structured persistence: articles, sentiment records, story groups, URL dedup hashes. Exposes a clean CRUD interface. Schema migrations handled via versioned SQL scripts.

### ChromaDB Store Module
Manages vector storage. Each article field is stored as a separate document with metadata: `field`, `article_id`, `story_group_id`, `source_name`, `published_at`. Exposes insert and similarity-search interfaces. The `article_id` in metadata is the join key back to SQLite for full-document retrieval during RAG generation.

### RAG Implementation
Field-based chunking with parent fetch:
1. User query is embedded via `text-embedding-004`
2. ChromaDB returns top-k most similar field chunks with metadata
3. `article_id` values from results are used to fetch full enriched articles from SQLite
4. Full articles + user query are passed to Gemini Flash for answer generation

Multi-aspect queries (e.g., "what is the sentiment and use cases for X") are handled via single retrieval — the query embedding naturally surfaces multiple field types. Query decomposition is deferred unless retrieval quality proves insufficient.

No hybrid (BM25 + vector) search. The structured, semantically rich field content makes keyword search redundant for this dataset.

### Scheduler
APScheduler manages two independent jobs:
- News pipeline: Mon / Wed / Fri at 23:00 local time
- Sentiment pipeline: daily at 08:00 local time

Both pipelines also expose a manual trigger function callable from the CLI or Streamlit UI.

### Streamlit UI
Two views:
1. **Feed view** — stories grouped by `story_group_id`, showing importance score, sentiment label, source count, and list of outlets. Ordered by importance score descending.
2. **Chat view** — RAG Q&A interface. User types a natural language question, receives an answer with citations back to source articles. No automatic web search fallback — out of scope for v1.

## Testing Decisions

Good tests verify external behavior through the module's public interface without asserting on internal implementation details (query plans, intermediate state, private methods). Tests should be runnable in isolation with no network calls — external dependencies (RSS feeds, Reddit API, Gemini API, ChromaDB, SQLite) are replaced with in-memory fakes or recorded fixtures at the module boundary.

### Modules to test:

**RSS Fetcher** — feed with mock XML fixture; assert normalized article objects returned. Test degenerate cases: empty feed, malformed XML, unreachable URL.

**Reddit Fetcher** — mock PRAW client; assert correct subreddits searched, correct keyword terms used, graceful empty-result handling.

**Embedder** — mock embedding API; assert interface contract (text in, fixed-length vector out). Test batching behavior.

**SQLite Store** — integration tests with in-memory SQLite (`:memory:`). Test insert, dedup hash check, story group assignment, sentiment update, rolling window query. These are the most valuable tests — SQLite behavior is real, no mocking needed.

**ChromaDB Store** — integration tests with ephemeral in-memory ChromaDB collection. Test field-based insert, metadata filtering, similarity search returning correct `article_id` values.

**Story Clustering** — unit tests for the clustering decision logic: same-story detection above threshold, new-story creation below threshold, source count increment.

**News Pipeline (LangGraph Graph 1)** — integration test with all external calls mocked (RSS, Gemini, embedding API). Assert final SQLite + ChromaDB state after a run with known input fixtures.

**Sentiment Pipeline (LangGraph Graph 2)** — integration test with Reddit and Gemini mocked. Assert sentiment fields updated correctly in SQLite and ChromaDB for articles in the rolling window.

## Out of Scope

- Web search fallback in the RAG interface (user uses external tools for data not in the DB)
- Hybrid BM25 + vector search
- Cloud storage (all storage is local)
- Multi-user support or authentication
- Mobile or non-Streamlit frontend
- Support for non-AI news topics
- Email or push notifications when important stories are detected
- Fine-tuning or custom model training
- Support for paywalled articles
- Pushshift or any deprecated Reddit data source

## Further Notes

- The importance score produced by the News Parse Agent and the Reddit engagement metrics (thread count, comment count) from the Sentiment Pipeline together form a composite signal of story importance. Future work could combine these into a single ranked feed score.
- Sentiment data for a story evolves over time as the community discusses it further. The daily re-scan of the 7-day window captures this evolution. SQLite stores `last_scanned_at` per article to support this pattern.
- Story clustering using embedding cosine similarity (threshold 0.85) is a heuristic. The threshold may need tuning after observing real data — too low causes false merges, too high causes duplicate stories.
- The system runs entirely on free-tier APIs at the expected usage levels. The binding constraint is Gemini Flash's 1,500 requests/day limit. At the planned schedule (3x/week news + daily sentiment for ~20 articles), typical daily usage is well under 100 requests.
- PRAW requires a Reddit API application to be registered at reddit.com/prefs/apps. This is free and takes ~2 minutes.

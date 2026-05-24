import json
import os
import random

# PRD stratification targets: 1 low (1-3), 2 medium (4-6), 2 high (7-10)
_STRATIFY_TARGETS = [
    ("low",    1,  3,  1),
    ("medium", 4,  6,  2),
    ("high",   7, 10,  2),
]


def _load_golden_fixtures(golden_dir: str, subdir: str) -> list[dict]:
    path = os.path.join(golden_dir, subdir)
    if not os.path.isdir(path):
        return []
    fixtures = []
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".json"):
            with open(os.path.join(path, fname)) as f:
                fixtures.append(json.load(f))
    return fixtures


def _stratified_recent(articles: list[dict], n: int, exclude_ids: set) -> list[dict]:
    available = [a for a in articles if a.get("article_id") not in exclude_ids]

    buckets = {
        "low":    [a for a in available if 1 <= (a.get("importance_score") or 0) <= 3],
        "medium": [a for a in available if 4 <= (a.get("importance_score") or 0) <= 6],
        "high":   [a for a in available if 7 <= (a.get("importance_score") or 0) <= 10],
    }
    for b in buckets.values():
        random.shuffle(b)

    selected: list[dict] = []
    seen_groups: set = set()
    selected_ids: set = set()

    for label, _lo, _hi, target in _STRATIFY_TARGETS:
        count = 0
        for article in buckets[label]:
            if count >= target:
                break
            gid = article.get("story_group_id")
            if gid is not None and gid in seen_groups:
                continue
            selected.append(article)
            selected_ids.add(article.get("article_id"))
            if gid is not None:
                seen_groups.add(gid)
            count += 1

    if len(selected) < n:
        remaining = [a for a in available if a.get("article_id") not in selected_ids]
        random.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected[:n]


def _article_to_sample(article: dict) -> dict:
    return {
        "agent_name": "news_parse_agent",
        "article_id": article.get("article_id"),
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
        "_source": "recent",
    }


def sampler(sqlite_store, golden_dir: str, n_golden: int = 5, n_recent: int = 5) -> list[dict]:
    """Assemble eval samples from golden fixtures and recent SQLite articles.

    Returns exactly n_golden + n_recent samples (or fewer if insufficient data
    exists in the store or golden directory).

    Golden fixtures are loaded from eval/golden/{news_parse,sentiment,rag}/ subdirs.
    RAG golden samples are query-answer pairs; all others are article samples.
    Recent samples are stratified by importance score bucket to cover the full
    range of article types the pipeline processes.
    """
    news_parse_golden = _load_golden_fixtures(golden_dir, "news_parse")
    sentiment_golden = _load_golden_fixtures(golden_dir, "sentiment")
    rag_golden = _load_golden_fixtures(golden_dir, "rag")

    all_golden: list[dict] = []
    golden_article_ids: set = set()

    for fixture in news_parse_golden + sentiment_golden + rag_golden:
        sample = {**fixture, "_source": "golden"}
        all_golden.append(sample)
        if fixture.get("article_id") is not None:
            golden_article_ids.add(fixture["article_id"])

    golden_samples = all_golden[:n_golden]

    all_enriched = sqlite_store.get_all_enriched_articles()
    recent_raw = _stratified_recent(all_enriched, n_recent, golden_article_ids)
    recent_samples = [_article_to_sample(a) for a in recent_raw]

    return golden_samples + recent_samples

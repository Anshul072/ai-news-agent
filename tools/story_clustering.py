from datetime import datetime, timedelta, timezone

import numpy as np

import config


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def assign_story_group(
    article_id: int,
    summary_embedding: list[float],
    published_at: str,
    sqlite_store,
    chroma_store,
    cutoff: str | None = None,
    threshold: float | None = None,
) -> int:
    if cutoff is None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    if threshold is None:
        threshold = config.CLUSTERING_THRESHOLD

    recent = chroma_store.get_summaries_since(cutoff)

    best_group_id = None
    best_sim = -1.0
    for item in recent:
        if item["article_id"] == article_id:
            continue
        sim = _cosine_similarity(summary_embedding, item["embedding"])
        if sim >= threshold and sim > best_sim:
            best_sim = sim
            best_group_id = item["story_group_id"]

    if best_group_id is not None:
        sqlite_store.increment_source_count(best_group_id)
        return best_group_id

    return sqlite_store.create_story_group()

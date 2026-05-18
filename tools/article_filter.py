import numpy as np

import config
import tools.embedder

_SEED_PHRASES: list[str] = [
    "new AI model release announcement",
    "open source machine learning library launch",
    "machine learning research paper breakthrough",
    "developer tooling and programming tools",
    "cloud infrastructure and platform updates",
    "software engineering best practices",
    "AI benchmark evaluation results",
    "AI safety and alignment research",
    "natural language processing computer vision breakthrough",
    "large language model fine-tuning techniques",
]

_SEED_EMBEDDINGS: list[list[float]] = [tools.embedder.embed(p) for p in _SEED_PHRASES]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def relevance_score(article: dict) -> float:
    title = article.get("title", "")
    content = article.get("content", "")[:500]
    text = f"{title} {content}".strip()
    article_vec = tools.embedder.embed(text)
    return max(
        _cosine_similarity(article_vec, seed_vec)
        for seed_vec in _SEED_EMBEDDINGS
    )


def is_relevant(article: dict) -> bool:
    return relevance_score(article) >= config.ARTICLE_FILTER_THRESHOLD

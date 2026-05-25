import json
import os

import config
import tools.article_filter as af

_DEFAULT_GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "article_filter", "articles.json")
_BORDERLINE_MARGIN = 0.05


def run(golden_path: str = _DEFAULT_GOLDEN, threshold: float | None = None) -> list[dict]:
    """Run all golden fixtures through the article filter and return per-article results."""
    if threshold is None:
        threshold = config.ARTICLE_FILTER_THRESHOLD

    with open(golden_path) as f:
        fixtures = json.load(f)

    results = []
    for fixture in fixtures:
        score = af.relevance_score(fixture)
        actual_pass = score >= threshold
        expected_pass = fixture["expected_pass"]
        results.append({
            "title": fixture["title"],
            "category": fixture["category"],
            "expected_pass": expected_pass,
            "actual_pass": actual_pass,
            "score": round(score, 4),
            "correct": actual_pass == expected_pass,
            "is_borderline": abs(score - threshold) < _BORDERLINE_MARGIN,
            "rationale": fixture.get("rationale", ""),
        })
    return results

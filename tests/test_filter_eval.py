import json
import os
import tempfile
from unittest.mock import patch

import pytest

from eval.filter_runner import run
from eval.filter_reporter import _metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_golden(articles: list[dict], tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "articles.json")
    with open(path, "w") as f:
        json.dump(articles, f)
    return path


_RELEVANT = {"title": "GPT-5 released", "content": "new LLM", "expected_pass": True, "category": "ai_research", "rationale": ""}
_IRRELEVANT = {"title": "Football scores", "content": "sports", "expected_pass": False, "category": "irrelevant", "rationale": ""}


# ---------------------------------------------------------------------------
# Behavior 1: correct pass/fail decision recorded in results
# ---------------------------------------------------------------------------

def test_run_records_correct_pass(tmp_path):
    golden = _make_golden([_RELEVANT], str(tmp_path))
    with patch("tools.article_filter.relevance_score", return_value=0.9):
        results = run(golden_path=golden, threshold=0.45)
    assert len(results) == 1
    r = results[0]
    assert r["actual_pass"] is True
    assert r["expected_pass"] is True
    assert r["correct"] is True
    assert r["score"] == 0.9


def test_run_records_correct_block(tmp_path):
    golden = _make_golden([_IRRELEVANT], str(tmp_path))
    with patch("tools.article_filter.relevance_score", return_value=0.1):
        results = run(golden_path=golden, threshold=0.45)
    r = results[0]
    assert r["actual_pass"] is False
    assert r["correct"] is True


# ---------------------------------------------------------------------------
# Behavior 2: false positive and false negative detected
# ---------------------------------------------------------------------------

def test_run_detects_false_positive(tmp_path):
    golden = _make_golden([_IRRELEVANT], str(tmp_path))
    with patch("tools.article_filter.relevance_score", return_value=0.8):
        results = run(golden_path=golden, threshold=0.45)
    r = results[0]
    assert r["actual_pass"] is True
    assert r["expected_pass"] is False
    assert r["correct"] is False


def test_run_detects_false_negative(tmp_path):
    golden = _make_golden([_RELEVANT], str(tmp_path))
    with patch("tools.article_filter.relevance_score", return_value=0.2):
        results = run(golden_path=golden, threshold=0.45)
    r = results[0]
    assert r["actual_pass"] is False
    assert r["expected_pass"] is True
    assert r["correct"] is False


# ---------------------------------------------------------------------------
# Behavior 3: borderline flag set when score is within 0.05 of threshold
# ---------------------------------------------------------------------------

def test_run_sets_borderline_flag_near_threshold(tmp_path):
    golden = _make_golden([_RELEVANT], str(tmp_path))
    with patch("tools.article_filter.relevance_score", return_value=0.47):  # 0.02 above threshold
        results = run(golden_path=golden, threshold=0.45)
    assert results[0]["is_borderline"] is True


def test_run_clears_borderline_flag_away_from_threshold(tmp_path):
    golden = _make_golden([_RELEVANT], str(tmp_path))
    with patch("tools.article_filter.relevance_score", return_value=0.9):
        results = run(golden_path=golden, threshold=0.45)
    assert results[0]["is_borderline"] is False


# ---------------------------------------------------------------------------
# Behavior 4: threshold override respected
# ---------------------------------------------------------------------------

def test_run_respects_threshold_override(tmp_path):
    golden = _make_golden([_RELEVANT], str(tmp_path))
    with patch("tools.article_filter.relevance_score", return_value=0.5):
        results_low = run(golden_path=golden, threshold=0.40)
        results_high = run(golden_path=golden, threshold=0.55)
    assert results_low[0]["actual_pass"] is True
    assert results_high[0]["actual_pass"] is False


# ---------------------------------------------------------------------------
# Behavior 5: metrics computation is correct
# ---------------------------------------------------------------------------

def test_metrics_perfect_classifier():
    results = [
        {"expected_pass": True, "actual_pass": True},
        {"expected_pass": False, "actual_pass": False},
    ]
    m = _metrics(results)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["fp"] == 0
    assert m["fn"] == 0


def test_metrics_all_false_positives():
    results = [{"expected_pass": False, "actual_pass": True}] * 3
    m = _metrics(results)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0  # no true positives at all
    assert m["fp"] == 3
    assert m["fn"] == 0


def test_metrics_mixed_errors():
    results = [
        {"expected_pass": True, "actual_pass": True},   # TP
        {"expected_pass": True, "actual_pass": False},  # FN
        {"expected_pass": False, "actual_pass": True},  # FP
        {"expected_pass": False, "actual_pass": False}, # TN
    ]
    m = _metrics(results)
    assert m["tp"] == 1
    assert m["tn"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.5)

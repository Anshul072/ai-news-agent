import json
import os
from datetime import datetime, timezone

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
_RECALL_WARN_THRESHOLD = 0.85
_SENSITIVITY_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55]


def _metrics(results: list[dict]) -> dict:
    tp = sum(1 for r in results if r["expected_pass"] and r["actual_pass"])
    tn = sum(1 for r in results if not r["expected_pass"] and not r["actual_pass"])
    fp = sum(1 for r in results if not r["expected_pass"] and r["actual_pass"])
    fn = sum(1 for r in results if r["expected_pass"] and not r["actual_pass"])
    total = len(results)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "fpr": round(fpr, 3),
        "fnr": round(fnr, 3),
    }


def _print_metrics_table(m: dict, threshold: float) -> None:
    print(f"\n=== article_filter  (threshold={threshold}) ===")
    print(f"  {'precision':<20} {m['precision']:.3f}  ({m['tp']} TP, {m['fp']} FP)")
    print(f"  {'recall':<20} {m['recall']:.3f}  ({m['tp']} TP, {m['fn']} FN)")
    print(f"  {'f1':<20} {m['f1']:.3f}")
    print(f"  {'accuracy':<20} {m['accuracy']:.3f}  ({m['tp'] + m['tn']}/{m['tp'] + m['tn'] + m['fp'] + m['fn']} correct)")
    print(f"  {'false positive rate':<20} {m['fpr']:.3f}")
    print(f"  {'false negative rate':<20} {m['fnr']:.3f}")


def _print_failures(results: list[dict]) -> None:
    fps = [r for r in results if not r["expected_pass"] and r["actual_pass"]]
    fns = [r for r in results if r["expected_pass"] and not r["actual_pass"]]

    if fps:
        print("\nFalse positives (irrelevant articles passed through):")
        for r in fps:
            marker = " [borderline]" if r["is_borderline"] else ""
            print(f"  score={r['score']:.4f}{marker}  {r['title'][:70]}")
    if fns:
        print("\nFalse negatives (relevant articles dropped):")
        for r in fns:
            marker = " [borderline]" if r["is_borderline"] else ""
            print(f"  score={r['score']:.4f}{marker}  {r['title'][:70]}")
    if not fps and not fns:
        print("\nNo errors — all articles classified correctly.")


def _print_sensitivity_table(sensitivity: list[dict]) -> None:
    print("\n=== Threshold sensitivity ===")
    print(f"  {'threshold':<12} {'precision':<12} {'recall':<12} {'f1':<10}")
    for entry in sensitivity:
        m = entry["metrics"]
        marker = " <--" if entry["threshold"] == entry.get("active_threshold") else ""
        print(f"  {entry['threshold']:<12.2f} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:.3f}{marker}")


def _write_json(
    results: list[dict],
    metrics: dict,
    threshold: float,
    sensitivity: list[dict],
) -> str:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(_RESULTS_DIR, f"filter_eval_{timestamp}.json")
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "summary": metrics,
        "sensitivity": sensitivity,
        "per_article": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def report(
    results: list[dict],
    threshold: float,
    sensitivity_results: dict[float, list[dict]] | None = None,
) -> None:
    """Print classification metrics table, FP/FN cases, threshold sensitivity, and write JSON."""
    m = _metrics(results)
    _print_metrics_table(m, threshold)
    _print_failures(results)

    if m["recall"] < _RECALL_WARN_THRESHOLD:
        print(f"\nWARNING: recall {m['recall']:.3f} is below {_RECALL_WARN_THRESHOLD} — relevant articles are being dropped")

    sensitivity_payload: list[dict] = []
    if sensitivity_results:
        for t, res in sorted(sensitivity_results.items()):
            sm = _metrics(res)
            sensitivity_payload.append({
                "threshold": t,
                "active_threshold": threshold,
                "metrics": sm,
            })
        _print_sensitivity_table(sensitivity_payload)

    path = _write_json(results, m, threshold, sensitivity_payload)
    print(f"\nResults written to {path}")

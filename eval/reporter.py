import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from eval.judge import DIMENSIONS

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
_WARN_THRESHOLD = 3.0


def _compute_averages(results: list[dict]) -> dict[str, dict[str, float]]:
    sums: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        agent = r["agent_name"]
        for dim, entry in r["scores"].items():
            sums[agent][dim].append(entry["score"])
    return {
        agent: {dim: sum(scores) / len(scores) for dim, scores in dims.items()}
        for agent, dims in sums.items()
    }


def _print_table(averages: dict[str, dict[str, float]]) -> None:
    for agent in DIMENSIONS:
        if agent not in averages:
            continue
        print(f"\n=== {agent} ===")
        for dim in DIMENSIONS[agent]:
            avg = averages[agent].get(dim)
            if avg is None:
                continue
            print(f"  {dim:<38} {avg:.1f} / 5.0")


def _print_warnings(averages: dict[str, dict[str, float]]) -> None:
    for agent, dims in averages.items():
        for dim, avg in dims.items():
            if avg < _WARN_THRESHOLD:
                print(f"WARNING: {agent} {dim} avg score {avg:.1f}/5.0 — below threshold")


def _write_json(results: list[dict], averages: dict[str, dict[str, float]]) -> str:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(_RESULTS_DIR, f"eval_{timestamp}.json")
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "summary": averages,
        "per_article": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def report(results: list[dict]) -> None:
    """Print CLI summary table and write timestamped JSON to eval/results/."""
    averages = _compute_averages(results)
    _print_table(averages)
    print()
    _print_warnings(averages)
    path = _write_json(results, averages)
    print(f"\nResults written to {path}")

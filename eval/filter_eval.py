import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from eval.filter_reporter import _SENSITIVITY_THRESHOLDS, report
from eval.filter_runner import _DEFAULT_GOLDEN, run


def main() -> None:
    threshold = config.ARTICLE_FILTER_THRESHOLD
    results = run(_DEFAULT_GOLDEN, threshold=threshold)
    sensitivity_results = {t: run(_DEFAULT_GOLDEN, threshold=t) for t in _SENSITIVITY_THRESHOLDS}
    report(results, threshold, sensitivity_results)


if __name__ == "__main__":
    main()

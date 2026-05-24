import os
import sys

# ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.reporter import report
from eval.runner import run
from storage.sqlite_store import SQLiteStore

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def main() -> None:
    store = SQLiteStore()
    store.init_db()
    results = run(store, _GOLDEN_DIR)
    report(results)
    sys.exit(0)


if __name__ == "__main__":
    main()

"""
Targeted regenerator for sentiment golden fixtures.

Fetches real HN threads for each existing fixture and re-scores all four
sentiment dimensions (including the new thread_relevance) using the Gemini
judge. Run from the project root:

    python eval/regen_sentiment_fixtures.py [--db storage/news.db]

Review the printed scores before the fixtures are committed.
"""
import argparse
import glob
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.judge import judge
from storage.sqlite_store import SQLiteStore
from tools.hn_fetcher import fetch_hn_threads

_FIXTURE_GLOB = "eval/golden/sentiment/article_*.json"


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _save(path: str, fixture: dict) -> None:
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="storage/news.db")
    args = parser.parse_args()

    store = SQLiteStore(db_path=args.db)
    store.init_db()

    paths = sorted(glob.glob(_FIXTURE_GLOB))
    if not paths:
        print("No sentiment fixtures found — run create_golden.py first.")
        sys.exit(1)

    print(f"Found {len(paths)} sentiment fixtures to update.\n")

    for path in paths:
        fixture = _load(path)
        aid = fixture["article_id"]
        title = fixture["inputs"]["article_title"]
        print(f"Article {aid}: {title[:70]}")

        raw = store.get_raw_article(aid)
        enriched = store.get_enriched_article(aid)
        if raw is None or enriched is None:
            print(f"  SKIP — article {aid} not found in DB\n")
            continue

        keywords = enriched.get("key_concepts") or title.split()[:5]
        article_url = raw.get("url", "")

        print(f"  Fetching HN threads (keywords={keywords[:3]}, url={article_url[:60]})...")
        hn_threads = fetch_hn_threads(keywords, article_url=article_url)
        print(f"  Found {len(hn_threads)} thread(s)")

        inputs = {
            "article_title": title,
            "hn_threads": hn_threads,
        }
        scores = judge("sentiment_agent", inputs, fixture["output"])

        for dim, entry in scores.items():
            print(f"    {dim}: {entry['score']}/5 — {entry['reasoning'][:80]}")

        fixture["inputs"]["hn_threads"] = hn_threads
        fixture["golden_scores"] = scores
        _save(path, fixture)
        print()

    print("Done. Review fixtures in eval/golden/sentiment/ before committing.")


if __name__ == "__main__":
    main()

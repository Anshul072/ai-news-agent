import json
import sqlite3
from datetime import datetime, timezone


class SQLiteStore:
    def __init__(self, db_path: str = "storage/news.db"):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS url_hashes (
                url_hash TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS raw_articles (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT UNIQUE NOT NULL,
                url_hash    TEXT NOT NULL,
                title       TEXT,
                content     TEXT,
                source_name TEXT,
                published_at TEXT,
                fetched_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS story_groups (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source_count INTEGER NOT NULL DEFAULT 1,
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS article_sentiment (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id          INTEGER UNIQUE NOT NULL REFERENCES raw_articles(id),
                sentiment_label     TEXT,
                sentiment_score     REAL,
                excitement_level    TEXT,
                top_concerns        TEXT,
                top_use_cases       TEXT,
                notable_quotes      TEXT,
                subreddit_breakdown TEXT,
                thread_count        INTEGER,
                total_comments      INTEGER,
                last_scanned_at     TEXT
            );

            CREATE TABLE IF NOT EXISTS enriched_articles (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id           INTEGER UNIQUE NOT NULL REFERENCES raw_articles(id),
                story_group_id       INTEGER REFERENCES story_groups(id),
                summary              TEXT,
                whats_new            TEXT,
                key_concepts         TEXT,
                concept_explanations TEXT,
                who_made_it          TEXT,
                use_cases            TEXT,
                importance_score     INTEGER,
                importance_reasoning TEXT
            );
        """)
        conn.commit()

    def url_hash_exists(self, url_hash: str) -> bool:
        row = self._get_conn().execute(
            "SELECT 1 FROM url_hashes WHERE url_hash = ?", (url_hash,)
        ).fetchone()
        return row is not None

    def insert_raw_article(self, article: dict) -> None:
        if self.url_hash_exists(article["url_hash"]):
            return
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO url_hashes (url_hash) VALUES (?)", (article["url_hash"],)
        )
        conn.execute(
            """INSERT OR IGNORE INTO raw_articles
               (url, url_hash, title, content, source_name, published_at, fetched_at)
               VALUES (:url, :url_hash, :title, :content, :source_name, :published_at, :fetched_at)""",
            article,
        )
        conn.commit()

    def get_raw_article_by_url_hash(self, url_hash: str) -> dict | None:
        row = self._get_conn().execute(
            "SELECT * FROM raw_articles WHERE url_hash = ?", (url_hash,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_raw_articles(self) -> list[dict]:
        rows = self._get_conn().execute("SELECT * FROM raw_articles").fetchall()
        return [dict(r) for r in rows]

    def get_unenriched_articles(self) -> list[dict]:
        rows = self._get_conn().execute(
            """SELECT r.* FROM raw_articles r
               LEFT JOIN enriched_articles e ON e.article_id = r.id
               WHERE e.id IS NULL"""
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_enriched_article(self, article_id: int, enriched: dict, story_group_id: int | None = None) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR IGNORE INTO enriched_articles
               (article_id, story_group_id, summary, whats_new, key_concepts,
                concept_explanations, who_made_it, use_cases, importance_score, importance_reasoning)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                article_id,
                story_group_id,
                enriched.get("summary"),
                enriched.get("whats_new"),
                json.dumps(enriched.get("key_concepts", [])),
                json.dumps(enriched.get("concept_explanations", {})),
                enriched.get("who_made_it"),
                json.dumps(enriched.get("use_cases", [])),
                enriched.get("importance_score"),
                enriched.get("importance_reasoning"),
            ),
        )
        conn.commit()

    def get_enriched_article(self, article_id: int) -> dict | None:
        row = self._get_conn().execute(
            "SELECT * FROM enriched_articles WHERE article_id = ?", (article_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["key_concepts"] = json.loads(result["key_concepts"] or "[]")
        result["concept_explanations"] = json.loads(result["concept_explanations"] or "{}")
        result["use_cases"] = json.loads(result["use_cases"] or "[]")
        return result

    def create_story_group(self) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO story_groups (source_count, created_at) VALUES (1, ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()
        return cur.lastrowid

    def increment_source_count(self, story_group_id: int) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE story_groups SET source_count = source_count + 1 WHERE id = ?",
            (story_group_id,),
        )
        conn.commit()

    def get_story_group(self, story_group_id: int) -> dict | None:
        row = self._get_conn().execute(
            "SELECT * FROM story_groups WHERE id = ?", (story_group_id,)
        ).fetchone()
        return dict(row) if row else None

    def set_story_group(self, article_id: int, story_group_id: int) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE enriched_articles SET story_group_id = ? WHERE article_id = ?",
            (story_group_id, article_id),
        )
        conn.commit()

    def upsert_sentiment(self, article_id: int, sentiment: dict) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO article_sentiment
               (article_id, sentiment_label, sentiment_score, excitement_level,
                top_concerns, top_use_cases, notable_quotes, subreddit_breakdown,
                thread_count, total_comments, last_scanned_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(article_id) DO UPDATE SET
                sentiment_label     = excluded.sentiment_label,
                sentiment_score     = excluded.sentiment_score,
                excitement_level    = excluded.excitement_level,
                top_concerns        = excluded.top_concerns,
                top_use_cases       = excluded.top_use_cases,
                notable_quotes      = excluded.notable_quotes,
                subreddit_breakdown = excluded.subreddit_breakdown,
                thread_count        = excluded.thread_count,
                total_comments      = excluded.total_comments,
                last_scanned_at     = excluded.last_scanned_at""",
            (
                article_id,
                sentiment.get("sentiment_label"),
                sentiment.get("sentiment_score"),
                sentiment.get("excitement_level"),
                json.dumps(sentiment.get("top_concerns", [])),
                json.dumps(sentiment.get("top_use_cases", [])),
                json.dumps(sentiment.get("notable_quotes", [])),
                json.dumps(sentiment.get("subreddit_breakdown", {})),
                sentiment.get("thread_count", 0),
                sentiment.get("total_comments", 0),
                sentiment.get("last_scanned_at"),
            ),
        )
        conn.commit()

    def get_sentiment(self, article_id: int) -> dict | None:
        row = self._get_conn().execute(
            "SELECT * FROM article_sentiment WHERE article_id = ?", (article_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["top_concerns"] = json.loads(result["top_concerns"] or "[]")
        result["top_use_cases"] = json.loads(result["top_use_cases"] or "[]")
        result["notable_quotes"] = json.loads(result["notable_quotes"] or "[]")
        result["subreddit_breakdown"] = json.loads(result["subreddit_breakdown"] or "{}")
        return result

    def get_recent_enriched_articles(self, days: int = 7) -> list[dict]:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._get_conn().execute(
            """SELECT r.*, e.*
               FROM raw_articles r
               JOIN enriched_articles e ON e.article_id = r.id
               WHERE r.published_at >= ?
               ORDER BY r.published_at DESC""",
            (cutoff,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["key_concepts"] = json.loads(d["key_concepts"] or "[]")
            d["concept_explanations"] = json.loads(d["concept_explanations"] or "{}")
            d["use_cases"] = json.loads(d["use_cases"] or "[]")
            results.append(d)
        return results

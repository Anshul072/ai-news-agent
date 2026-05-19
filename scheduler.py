import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import config
from pipelines.news_pipeline import run_news_pipeline
from pipelines.sentiment_pipeline import run_sentiment_pipeline
from storage.sqlite_store import SQLiteStore
from storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

_sqlite_store = None
_chroma_store = None


def _get_stores():
    global _sqlite_store, _chroma_store
    if _sqlite_store is None:
        _sqlite_store = SQLiteStore("storage/news.db")
        _sqlite_store.init_db()
    if _chroma_store is None:
        _chroma_store = ChromaStore()
    return _sqlite_store, _chroma_store


def trigger_news_pipeline():
    try:
        sqlite_store, chroma_store = _get_stores()
        run_news_pipeline(config.RSS_FEEDS, sqlite_store, chroma_store)
        logger.info("News pipeline completed.")
    except Exception as exc:
        logger.error("News pipeline failed: %s", exc)


def trigger_sentiment_pipeline():
    try:
        sqlite_store, chroma_store = _get_stores()
        run_sentiment_pipeline(sqlite_store, chroma_store)
        logger.info("Sentiment pipeline completed.")
    except Exception as exc:
        logger.error("Sentiment pipeline failed: %s", exc)


def build_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler()
    sched.add_job(
        trigger_news_pipeline,
        CronTrigger.from_crontab(config.NEWS_SCHEDULE),
        id="news_pipeline",
        name="News Pipeline",
        misfire_grace_time=300,
    )
    sched.add_job(
        trigger_sentiment_pipeline,
        CronTrigger.from_crontab(config.SENTIMENT_SCHEDULE),
        id="sentiment_pipeline",
        name="Sentiment Pipeline",
        misfire_grace_time=300,
    )
    return sched


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sched = build_scheduler()
    sched.start()
    logger.info("Scheduler started. News: %s | Sentiment: %s", config.NEWS_SCHEDULE, config.SENTIMENT_SCHEDULE)
    for job in sched.get_jobs():
        logger.info("  Job '%s' next run: %s", job.id, job.next_run_time)
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()
        logger.info("Scheduler stopped.")

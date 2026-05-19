from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Behavior 1: trigger_news_pipeline calls run_news_pipeline with RSS_FEEDS
# ---------------------------------------------------------------------------

def test_trigger_news_pipeline_calls_run_news_pipeline():
    mock_sqlite = MagicMock()
    mock_chroma = MagicMock()

    with patch("scheduler._get_stores", return_value=(mock_sqlite, mock_chroma)), \
         patch("scheduler.run_news_pipeline") as mock_run:
        from scheduler import trigger_news_pipeline
        trigger_news_pipeline()

    mock_run.assert_called_once()
    args, _ = mock_run.call_args
    import config
    assert args[0] == config.RSS_FEEDS


# ---------------------------------------------------------------------------
# Behavior 2: trigger_sentiment_pipeline calls run_sentiment_pipeline
# ---------------------------------------------------------------------------

def test_trigger_sentiment_pipeline_calls_run_sentiment_pipeline():
    mock_sqlite = MagicMock()
    mock_chroma = MagicMock()

    with patch("scheduler._get_stores", return_value=(mock_sqlite, mock_chroma)), \
         patch("scheduler.run_sentiment_pipeline") as mock_run:
        from scheduler import trigger_sentiment_pipeline
        trigger_sentiment_pipeline()

    mock_run.assert_called_once_with(mock_sqlite, mock_chroma)


# ---------------------------------------------------------------------------
# Behavior 3: trigger_news_pipeline swallows pipeline exceptions
# ---------------------------------------------------------------------------

def test_trigger_news_pipeline_does_not_raise_on_error():
    mock_sqlite = MagicMock()
    mock_chroma = MagicMock()

    with patch("scheduler._get_stores", return_value=(mock_sqlite, mock_chroma)), \
         patch("scheduler.run_news_pipeline", side_effect=RuntimeError("boom")):
        from scheduler import trigger_news_pipeline
        trigger_news_pipeline()  # must not raise


# ---------------------------------------------------------------------------
# Behavior 4: trigger_sentiment_pipeline swallows pipeline exceptions
# ---------------------------------------------------------------------------

def test_trigger_sentiment_pipeline_does_not_raise_on_error():
    mock_sqlite = MagicMock()
    mock_chroma = MagicMock()

    with patch("scheduler._get_stores", return_value=(mock_sqlite, mock_chroma)), \
         patch("scheduler.run_sentiment_pipeline", side_effect=RuntimeError("boom")):
        from scheduler import trigger_sentiment_pipeline
        trigger_sentiment_pipeline()  # must not raise


# ---------------------------------------------------------------------------
# Behavior 5: build_scheduler registers news and sentiment jobs
# ---------------------------------------------------------------------------

def test_build_scheduler_registers_two_jobs():
    from scheduler import build_scheduler
    sched = build_scheduler()
    job_ids = {job.id for job in sched.get_jobs()}
    assert "news_pipeline" in job_ids
    assert "sentiment_pipeline" in job_ids

import sys
from unittest.mock import patch
import pytest


def _reload_config(monkeypatch, env: dict):
    """Load config with a clean env containing only the supplied vars."""
    for key in [
        "GROQ_API_KEY", "RSS_FEEDS", "NEWS_SCHEDULE", "SENTIMENT_SCHEDULE",
        "CLUSTERING_THRESHOLD", "SENTIMENT_WINDOW_DAYS", "ARTICLE_FILTER_THRESHOLD",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    if "config" in sys.modules:
        del sys.modules["config"]
    with patch("dotenv.load_dotenv"):
        import config
    return config


FULL_ENV = {"GROQ_API_KEY": "gr-key"}


# ---------------------------------------------------------------------------
# Behavior 1: missing GROQ_API_KEY raises
# ---------------------------------------------------------------------------

def test_missing_groq_key_raises(monkeypatch):
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        _reload_config(monkeypatch, {})


# ---------------------------------------------------------------------------
# Behavior 2: all vars present → config loads and exposes API keys
# ---------------------------------------------------------------------------

def test_api_keys_loaded(monkeypatch):
    cfg = _reload_config(monkeypatch, FULL_ENV)
    assert cfg.GROQ_API_KEY == "gr-key"


# ---------------------------------------------------------------------------
# Behavior 3: configurable values have sensible defaults
# ---------------------------------------------------------------------------

def test_defaults(monkeypatch):
    cfg = _reload_config(monkeypatch, FULL_ENV)
    assert isinstance(cfg.RSS_FEEDS, list) and len(cfg.RSS_FEEDS) > 0
    assert cfg.CLUSTERING_THRESHOLD == 0.75
    assert cfg.SENTIMENT_WINDOW_DAYS == 7


# ---------------------------------------------------------------------------
# Behavior 4: configurable values can be overridden via env vars
# ---------------------------------------------------------------------------

def test_overrides(monkeypatch):
    env = {**FULL_ENV, "CLUSTERING_THRESHOLD": "0.90", "SENTIMENT_WINDOW_DAYS": "14"}
    cfg = _reload_config(monkeypatch, env)
    assert cfg.CLUSTERING_THRESHOLD == 0.90
    assert cfg.SENTIMENT_WINDOW_DAYS == 14

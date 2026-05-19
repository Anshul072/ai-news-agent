import importlib
import os
import sys
from unittest.mock import patch
import pytest


def _reload_config(monkeypatch, env: dict):
    """Load config with a clean env containing only the supplied vars."""
    for key in ["GROQ_API_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]:
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    if "config" in sys.modules:
        del sys.modules["config"]
    with patch("dotenv.load_dotenv"):
        import config
    return config


FULL_ENV = {
    "GROQ_API_KEY": "gr-key",
    "REDDIT_CLIENT_ID": "r-id",
    "REDDIT_CLIENT_SECRET": "r-secret",
    "REDDIT_USER_AGENT": "r-agent",
}


# ---------------------------------------------------------------------------
# Behavior 1: missing GROQ_API_KEY raises; Reddit keys are optional
# ---------------------------------------------------------------------------

def test_missing_groq_key_raises(monkeypatch):
    env = {k: v for k, v in FULL_ENV.items() if k != "GROQ_API_KEY"}
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        _reload_config(monkeypatch, env)


def test_missing_reddit_keys_does_not_raise(monkeypatch):
    cfg = _reload_config(monkeypatch, {"GROQ_API_KEY": "gr-key"})
    assert cfg.REDDIT_CLIENT_ID == ""
    assert cfg.REDDIT_CLIENT_SECRET == ""


# ---------------------------------------------------------------------------
# Behavior 2: all vars present → config loads and exposes API keys
# ---------------------------------------------------------------------------

def test_api_keys_loaded(monkeypatch):
    cfg = _reload_config(monkeypatch, FULL_ENV)
    assert cfg.GROQ_API_KEY == "gr-key"
    assert cfg.REDDIT_CLIENT_ID == "r-id"
    assert cfg.REDDIT_CLIENT_SECRET == "r-secret"
    assert cfg.REDDIT_USER_AGENT == "r-agent"


# ---------------------------------------------------------------------------
# Behavior 3: configurable values have sensible defaults
# ---------------------------------------------------------------------------

def test_defaults(monkeypatch):
    cfg = _reload_config(monkeypatch, FULL_ENV)
    assert isinstance(cfg.RSS_FEEDS, list) and len(cfg.RSS_FEEDS) > 0
    assert isinstance(cfg.SUBREDDITS, list) and len(cfg.SUBREDDITS) > 0
    assert cfg.CLUSTERING_THRESHOLD == 0.85
    assert cfg.SENTIMENT_WINDOW_DAYS == 7


# ---------------------------------------------------------------------------
# Behavior 4: configurable values can be overridden via env vars
# ---------------------------------------------------------------------------

def test_overrides(monkeypatch):
    env = {**FULL_ENV, "CLUSTERING_THRESHOLD": "0.90", "SENTIMENT_WINDOW_DAYS": "14"}
    cfg = _reload_config(monkeypatch, env)
    assert cfg.CLUSTERING_THRESHOLD == 0.90
    assert cfg.SENTIMENT_WINDOW_DAYS == 14

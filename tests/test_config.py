import importlib
import os
import sys
import pytest


def _reload_config(monkeypatch, env: dict):
    """Load config with a clean env containing only the supplied vars."""
    for key in ["GEMINI_API_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]:
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    if "config" in sys.modules:
        del sys.modules["config"]
    import config
    return config


FULL_ENV = {
    "GEMINI_API_KEY": "g-key",
    "REDDIT_CLIENT_ID": "r-id",
    "REDDIT_CLIENT_SECRET": "r-secret",
    "REDDIT_USER_AGENT": "r-agent",
}


# ---------------------------------------------------------------------------
# Behavior 1: missing required env var → EnvironmentError with the var name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_key", list(FULL_ENV.keys()))
def test_missing_required_env_var_raises(monkeypatch, missing_key):
    env = {k: v for k, v in FULL_ENV.items() if k != missing_key}
    with pytest.raises(EnvironmentError, match=missing_key):
        _reload_config(monkeypatch, env)


# ---------------------------------------------------------------------------
# Behavior 2: all vars present → config loads and exposes API keys
# ---------------------------------------------------------------------------

def test_api_keys_loaded(monkeypatch):
    cfg = _reload_config(monkeypatch, FULL_ENV)
    assert cfg.GEMINI_API_KEY == "g-key"
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

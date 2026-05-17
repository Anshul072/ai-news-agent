import os
import sys

# Inject fake API keys before any module that calls `import config` at load time.
_FAKE_ENV = {
    "GEMINI_API_KEY": "fake-gemini-key",
    "REDDIT_CLIENT_ID": "fake-reddit-id",
    "REDDIT_CLIENT_SECRET": "fake-reddit-secret",
    "REDDIT_USER_AGENT": "fake-agent/1.0",
}
for _k, _v in _FAKE_ENV.items():
    os.environ.setdefault(_k, _v)

# Ensure project root is on sys.path so top-level packages are importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

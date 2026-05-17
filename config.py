import os

_REQUIRED = [
    "GEMINI_API_KEY",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
]

for _var in _REQUIRED:
    if not os.environ.get(_var):
        raise EnvironmentError(f"Required environment variable not set: {_var}")

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
REDDIT_CLIENT_ID = os.environ["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET = os.environ["REDDIT_CLIENT_SECRET"]
REDDIT_USER_AGENT = os.environ["REDDIT_USER_AGENT"]

RSS_FEEDS = os.environ.get("RSS_FEEDS", "https://feeds.feedburner.com/aisummary").split(",")
SUBREDDITS = os.environ.get("SUBREDDITS", "artificial,MachineLearning,LocalLLaMA").split(",")

NEWS_SCHEDULE = os.environ.get("NEWS_SCHEDULE", "0 23 * * 1,3,5")
SENTIMENT_SCHEDULE = os.environ.get("SENTIMENT_SCHEDULE", "0 8 * * *")

CLUSTERING_THRESHOLD = float(os.environ.get("CLUSTERING_THRESHOLD", "0.85"))
SENTIMENT_WINDOW_DAYS = int(os.environ.get("SENTIMENT_WINDOW_DAYS", "7"))

import os
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError("Required environment variable not set: GROQ_API_KEY")

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
RSS_FEEDS = os.environ.get("RSS_FEEDS", "https://feeds.feedburner.com/aisummary").split(",")

NEWS_SCHEDULE = os.environ.get("NEWS_SCHEDULE", "0 23 * * 1,3,5")
SENTIMENT_SCHEDULE = os.environ.get("SENTIMENT_SCHEDULE", "0 8 * * *")

CLUSTERING_THRESHOLD = float(os.environ.get("CLUSTERING_THRESHOLD", "0.75"))
SENTIMENT_WINDOW_DAYS = int(os.environ.get("SENTIMENT_WINDOW_DAYS", "7"))
ARTICLE_FILTER_THRESHOLD = float(os.environ.get("ARTICLE_FILTER_THRESHOLD", "0.5"))

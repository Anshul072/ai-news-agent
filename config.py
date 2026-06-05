import os
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError("Required environment variable not set: GROQ_API_KEY")

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
RSS_FEEDS = os.environ.get(
    "RSS_FEEDS",
    ",".join([
        # Research
        "https://export.arxiv.org/rss/cs.AI",
        "https://export.arxiv.org/rss/cs.LG",
        # Company research blogs
        "https://openai.com/news/rss.xml",
        "https://www.anthropic.com/rss.xml",
        "https://deepmind.google/blog/rss.xml",
        "https://ai.meta.com/blog/rss/",
        "https://mistral.ai/news/feed/",
        # Tech journalism
        "https://arstechnica.com/ai/feed/",
        "https://www.theverge.com/rss/tech/index.xml",
        # Curated newsletters
        "https://www.deeplearning.ai/the-batch/feed/",
        "https://importai.substack.com/feed",
    ]),
).split(",")

NEWS_SCHEDULE = os.environ.get("NEWS_SCHEDULE", "0 23 * * 1,3,5")
SENTIMENT_SCHEDULE = os.environ.get("SENTIMENT_SCHEDULE", "0 8 * * *")

CLUSTERING_THRESHOLD = float(os.environ.get("CLUSTERING_THRESHOLD", "0.75"))
SENTIMENT_WINDOW_DAYS = int(os.environ.get("SENTIMENT_WINDOW_DAYS", "7"))
ARTICLE_FILTER_THRESHOLD = float(os.environ.get("ARTICLE_FILTER_THRESHOLD", "0.5"))

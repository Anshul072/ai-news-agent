import praw

import config


def fetch_reddit_threads(
    keywords: list[str],
    subreddits: list[str] | None = None,
    limit: int = 5,
    time_filter: str = "month",
    top_comments_per_thread: int = 3,
) -> list[dict]:
    if subreddits is None:
        subreddits = config.SUBREDDITS

    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )

    query = " OR ".join(keywords)
    threads = []
    for sub_name in subreddits:
        sub = reddit.subreddit(sub_name)
        for submission in sub.search(query, limit=limit, time_filter=time_filter):
            submission.comments.replace_more(limit=0)
            top_comments = [
                c.body
                for c in list(submission.comments)[:top_comments_per_thread]
                if hasattr(c, "body")
            ]
            threads.append({
                "subreddit": sub_name,
                "title": submission.title,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "top_comments": top_comments,
            })
    return threads

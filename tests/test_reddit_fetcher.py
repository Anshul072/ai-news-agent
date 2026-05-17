from unittest.mock import MagicMock, patch

from tools.reddit_fetcher import fetch_reddit_threads


def _make_submission(title="Test post", score=100, num_comments=50, comments=()):
    sub = MagicMock()
    sub.title = title
    sub.score = score
    sub.num_comments = num_comments
    sub.comments.replace_more = MagicMock()
    sub.comments.__iter__ = MagicMock(return_value=iter(comments))
    return sub


# ---------------------------------------------------------------------------
# Behavior 1: searches all specified subreddits
# ---------------------------------------------------------------------------

def test_searches_all_specified_subreddits():
    mock_reddit = MagicMock()
    mock_sub = MagicMock()
    mock_sub.search.return_value = []
    mock_reddit.subreddit.return_value = mock_sub

    with patch("tools.reddit_fetcher.praw.Reddit", return_value=mock_reddit):
        fetch_reddit_threads(["GPT-5"], subreddits=["artificial", "MachineLearning"])

    assert mock_reddit.subreddit.call_count == 2
    called_subs = {call.args[0] for call in mock_reddit.subreddit.call_args_list}
    assert called_subs == {"artificial", "MachineLearning"}


# ---------------------------------------------------------------------------
# Behavior 2: returns threads with expected structure
# ---------------------------------------------------------------------------

def test_returns_threads_with_correct_structure():
    mock_comment = MagicMock()
    mock_comment.body = "Great advancement!"
    submission = _make_submission(
        title="GPT-5 released", score=200, num_comments=80, comments=[mock_comment]
    )

    mock_reddit = MagicMock()
    mock_sub = MagicMock()
    mock_sub.search.return_value = [submission]
    mock_reddit.subreddit.return_value = mock_sub

    with patch("tools.reddit_fetcher.praw.Reddit", return_value=mock_reddit):
        results = fetch_reddit_threads(["GPT-5"], subreddits=["artificial"])

    assert len(results) == 1
    t = results[0]
    assert t["subreddit"] == "artificial"
    assert t["title"] == "GPT-5 released"
    assert t["score"] == 200
    assert t["num_comments"] == 80
    assert "Great advancement!" in t["top_comments"]


# ---------------------------------------------------------------------------
# Behavior 3: returns empty list gracefully when no threads found
# ---------------------------------------------------------------------------

def test_returns_empty_list_when_no_threads():
    mock_reddit = MagicMock()
    mock_sub = MagicMock()
    mock_sub.search.return_value = []
    mock_reddit.subreddit.return_value = mock_sub

    with patch("tools.reddit_fetcher.praw.Reddit", return_value=mock_reddit):
        results = fetch_reddit_threads(["obscure niche topic xyz"], subreddits=["artificial"])

    assert results == []


# ---------------------------------------------------------------------------
# Behavior 4: keyword query is used in the subreddit search call
# ---------------------------------------------------------------------------

def test_keywords_used_in_search_query():
    mock_reddit = MagicMock()
    mock_sub = MagicMock()
    mock_sub.search.return_value = []
    mock_reddit.subreddit.return_value = mock_sub

    with patch("tools.reddit_fetcher.praw.Reddit", return_value=mock_reddit):
        fetch_reddit_threads(["GPT-5", "OpenAI"], subreddits=["artificial"])

    call_args = mock_sub.search.call_args
    query = call_args.args[0] if call_args.args else call_args.kwargs.get("query", "")
    assert "GPT-5" in query
    assert "OpenAI" in query

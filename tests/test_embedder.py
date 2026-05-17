from unittest.mock import patch, MagicMock
import pytest

from tools.embedder import embed


# ---------------------------------------------------------------------------
# Behavior 1: embed returns a list of floats
# ---------------------------------------------------------------------------

def test_embed_returns_list_of_floats():
    mock_values = [0.1] * 768
    mock_result = MagicMock()
    mock_result.embedding = mock_values

    with patch("tools.embedder.genai.embed_content", return_value=mock_result):
        result = embed("some text about AI")

    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Behavior 2: embed passes the text to the API with the correct model
# ---------------------------------------------------------------------------

def test_embed_calls_api_with_correct_model():
    mock_result = MagicMock()
    mock_result.embedding = [0.0] * 768

    with patch("tools.embedder.genai.embed_content", return_value=mock_result) as mock_api:
        embed("hello world")

    mock_api.assert_called_once()
    call_kwargs = mock_api.call_args
    assert "text-embedding-004" in str(call_kwargs)


# ---------------------------------------------------------------------------
# Behavior 3: embed returns the exact vector from the API
# ---------------------------------------------------------------------------

def test_embed_returns_api_vector():
    expected = [float(i) / 1000 for i in range(768)]
    mock_result = MagicMock()
    mock_result.embedding = expected

    with patch("tools.embedder.genai.embed_content", return_value=mock_result):
        result = embed("test")

    assert result == expected

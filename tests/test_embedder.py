from unittest.mock import patch, MagicMock
import pytest

from tools.embedder import embed, embed_many


def _mock_vector(values: list[float]):
    m = MagicMock()
    m.tolist.return_value = values
    return m


# ---------------------------------------------------------------------------
# Behavior 1: embed returns a list of floats
# ---------------------------------------------------------------------------

def test_embed_returns_list_of_floats():
    fake_model = MagicMock()
    fake_model.encode.return_value = _mock_vector([0.1] * 768)
    with patch("tools.embedder._get_model", return_value=fake_model):
        result = embed("some text about AI")

    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Behavior 2: embed passes the text to the model
# ---------------------------------------------------------------------------

def test_embed_passes_text_to_model():
    fake_model = MagicMock()
    fake_model.encode.return_value = _mock_vector([0.0] * 768)
    with patch("tools.embedder._get_model", return_value=fake_model):
        embed("hello world")

    fake_model.encode.assert_called_once_with("hello world")


# ---------------------------------------------------------------------------
# Behavior 3: embed returns the exact vector from the model
# ---------------------------------------------------------------------------

def test_embed_returns_model_vector():
    expected = [float(i) / 1000 for i in range(768)]
    fake_model = MagicMock()
    fake_model.encode.return_value = _mock_vector(expected)
    with patch("tools.embedder._get_model", return_value=fake_model):
        result = embed("test")

    assert result == expected


# ---------------------------------------------------------------------------
# Behavior 4: embed_many encodes the whole list in a single encode() call
# ---------------------------------------------------------------------------

def test_embed_many_encodes_list_in_one_call():
    expected = [[0.1] * 768, [0.2] * 768]
    fake_model = MagicMock()
    fake_model.encode.return_value = _mock_vector(expected)
    with patch("tools.embedder._get_model", return_value=fake_model):
        result = embed_many(["first text", "second text"])

    fake_model.encode.assert_called_once_with(["first text", "second text"])
    assert result == expected


# ---------------------------------------------------------------------------
# Behavior 5: embed_many short-circuits on an empty list (no encode call)
# ---------------------------------------------------------------------------

def test_embed_many_empty_list_skips_model():
    fake_model = MagicMock()
    with patch("tools.embedder._get_model", return_value=fake_model):
        result = embed_many([])

    assert result == []
    fake_model.encode.assert_not_called()

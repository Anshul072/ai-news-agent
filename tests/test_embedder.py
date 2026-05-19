from unittest.mock import patch, MagicMock
import pytest

from tools.embedder import embed


def _mock_vector(values: list[float]):
    m = MagicMock()
    m.tolist.return_value = values
    return m


# ---------------------------------------------------------------------------
# Behavior 1: embed returns a list of floats
# ---------------------------------------------------------------------------

def test_embed_returns_list_of_floats():
    with patch("tools.embedder._model.encode", return_value=_mock_vector([0.1] * 768)):
        result = embed("some text about AI")

    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Behavior 2: embed passes the text to the model
# ---------------------------------------------------------------------------

def test_embed_passes_text_to_model():
    with patch("tools.embedder._model.encode", return_value=_mock_vector([0.0] * 768)) as mock_encode:
        embed("hello world")

    mock_encode.assert_called_once_with("hello world")


# ---------------------------------------------------------------------------
# Behavior 3: embed returns the exact vector from the model
# ---------------------------------------------------------------------------

def test_embed_returns_model_vector():
    expected = [float(i) / 1000 for i in range(768)]
    with patch("tools.embedder._model.encode", return_value=_mock_vector(expected)):
        result = embed("test")

    assert result == expected

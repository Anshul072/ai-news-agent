import threading
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _model


def embed(text: str) -> list[float]:
    return _get_model().encode(text).tolist()


def embed_many(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in a single batched ``encode()`` forward pass."""
    if not texts:
        return []
    return _get_model().encode(texts).tolist()

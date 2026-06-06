import threading

import config

# We embed with fastembed (ONNX Runtime) rather than sentence-transformers.
# sentence-transformers pulls in torch, whose import alone is ~20s and holds the
# GIL in long bursts — loading it in-process froze the Streamlit UI for the whole
# load window. fastembed imports in ~1s, loads the same BAAI/bge-base-en-v1.5 in
# under a second, and produces embeddings that are cosine-identical to the
# sentence-transformers output (so already-stored ChromaDB vectors stay valid).
# The import is still deferred to first use to keep module import cheap.
_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed import TextEmbedding

                # Cap ONNX Runtime threads so a large batch can't grab every core
                # and starve the in-process Streamlit UI thread during a fetch.
                _model = TextEmbedding(
                    "BAAI/bge-base-en-v1.5",
                    threads=config.EMBED_TORCH_THREADS,
                )
    return _model


def embed(text: str) -> list[float]:
    return embed_many([text])[0]


def embed_many(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in a single batched pass, preserving input order."""
    if not texts:
        return []
    return [vec.tolist() for vec in _get_model().embed(list(texts))]

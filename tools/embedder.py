from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def embed(text: str) -> list[float]:
    return _model.encode(text).tolist()

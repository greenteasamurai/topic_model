from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def get_embedding_model() -> SentenceTransformer:
    return _get_model()

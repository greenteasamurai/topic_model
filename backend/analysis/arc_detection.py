from sentence_transformers import SentenceTransformer
import umap
import hdbscan

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def detect_arcs(chapters: list[str], min_cluster_size: int = 2) -> list[int]:
    if len(chapters) < 4:
        return [-1] * len(chapters)

    model = _get_model()
    embeddings = model.encode(chapters, show_progress_bar=False)

    n_neighbors = min(15, len(chapters) - 1)
    n_components = min(5, len(chapters) - 1)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    return [int(label) for label in labels]

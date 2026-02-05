"""Text embedding utilities for news features."""
from __future__ import annotations

from typing import Iterable, Sequence


def embed_texts(texts: Iterable[str], model_name: str | None) -> Sequence[Sequence[float]]:
    """Embed text using a sentence-transformers model."""
    if not model_name:
        raise ValueError("model_name is required for embeddings")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install with `pip install -e .[ai]`."
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(texts), convert_to_numpy=True)
    return embeddings.tolist()

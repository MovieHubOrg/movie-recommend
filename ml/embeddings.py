"""Embedding generation utilities."""
import numpy as np
from sklearn.preprocessing import normalize

from ml.model_loader import model
from utils.text import build_content_string


def generate_embeddings(texts: list[str], batch_size: int = 32, normalize_vecs: bool = True) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        batch_size: Batch size for encoding
        normalize_vecs: Whether to normalize the output vectors

    Returns:
        numpy array of embeddings
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
    )

    if normalize_vecs:
        embeddings = normalize(embeddings)

    return embeddings


def generate_movie_embeddings(movies: list[dict], batch_size: int = 64) -> tuple[np.ndarray, list]:
    """
    Generate embeddings for a list of movies.

    Args:
        movies: List of movie dictionaries
        batch_size: Batch size for encoding

    Returns:
        Tuple of (embeddings array, list of movie IDs)
    """
    contents = [build_content_string(m) for m in movies]
    ids = [m["id"] for m in movies]

    embeddings = model.encode(
        contents,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = normalize(embeddings)

    return embeddings, ids

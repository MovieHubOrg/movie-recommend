"""Catalog index and embedding management service."""
import os
import numpy as np
import faiss

from sklearn.preprocessing import normalize
from ml.embeddings import generate_movie_embeddings
from core.config import settings

CACHE_FILE = settings.embeddings_cache
INDEX_FILE = settings.index_file


def build_catalog_index(catalog: list):
    """
    Build FAISS index for movie catalog.

    Creates embeddings for all movies and builds a searchable index.
    Supports GPU acceleration if available.

    Args:
        catalog: List of movie dictionaries
    """
    print(f"Encoding {len(catalog)} movies...")

    embeddings, _ = generate_movie_embeddings(catalog, batch_size=64)

    embeddings = normalize(embeddings).astype("float32")

    ids = np.array([m["id"] for m in catalog])

    np.savez(CACHE_FILE, vecs=embeddings, ids=ids)

    dim = embeddings.shape[1]

    # CPU index
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(embeddings)

    # Try GPU
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("Using FAISS GPU")
    except Exception:
        index = cpu_index
        print("Using FAISS CPU")

    # save CPU version
    faiss.write_index(cpu_index, INDEX_FILE)

    print("Catalog index built")

    return index


def load_catalog_index(catalog=None):
    """
    Load FAISS index and cached embeddings.

    Builds the index if it doesn't exist.

    Args:
        catalog: Optional catalog to build index if not found

    Returns:
        Tuple of (faiss index, embeddings array, movie IDs list)

    Raises:
        Exception: If index not found and catalog not provided
    """
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CACHE_FILE):
        if catalog is None:
            raise Exception("Index not found. Need catalog to build.")
        index = build_catalog_index(catalog)
        cache = np.load(CACHE_FILE, allow_pickle=True)
        return index, cache["vecs"], cache["ids"].tolist()

    index = faiss.read_index(INDEX_FILE)

    cache = np.load(CACHE_FILE, allow_pickle=True)

    vecs = cache["vecs"]
    ids = cache["ids"].tolist()

    return index, vecs, ids
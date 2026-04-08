"""Catalog index and embedding management service."""
import os
import numpy as np
import faiss

from core.config import settings

CACHE_FILE = settings.embeddings_cache
INDEX_FILE = settings.index_file

# Cached index and data
_index_cache = {
    "index": None,
    "vecs": None,
    "ids": None
}


def load_catalog_index():
    """
    Load FAISS index and cached embeddings.

    Returns:
        Tuple of (faiss index, embeddings array, movie IDs list)

    Raises:
        Exception: If index not found (run sync first)
    """
    # Return cached if available
    if _index_cache["index"] is not None:
        return _index_cache["index"], _index_cache["vecs"], _index_cache["ids"]
    
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CACHE_FILE):
        raise Exception("Index not found. Run catalog sync first.")

    index = faiss.read_index(INDEX_FILE)
    cache = np.load(CACHE_FILE, allow_pickle=True)
    vecs = cache["vecs"]
    ids = cache["ids"].tolist()
    
    # Cache for subsequent calls
    _index_cache["index"] = index
    _index_cache["vecs"] = vecs
    _index_cache["ids"] = ids

    return index, vecs, ids


def reload_catalog_index():
    """Force reload index from disk (after sync)."""
    global _index_cache
    _index_cache = {"index": None, "vecs": None, "ids": None}
    return load_catalog_index()


def is_index_ready() -> bool:
    """Check if catalog index is available."""
    return os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE)
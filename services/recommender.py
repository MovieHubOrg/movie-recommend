"""Recommendation service."""
import numpy as np

from services.catalog_service import load_catalog_index


def recommend_from_history(user_vec, catalog, history_list, top_k=5):
    """
    Generate movie recommendations based on user profile and watch history.

    Args:
        user_vec: User profile vector
        catalog: List of movie catalogs
        history_list: List of watched movie history items
        top_k: Number of recommendations to return

    Returns:
        List of recommended movies with similarity scores
    """
    index, vecs, ids = load_catalog_index(catalog)

    watched_ids = {item["movie"]["id"] for item in history_list}

    search_k = top_k * 3
    scores, indices = index.search(
        user_vec.reshape(1, -1).astype("float32"),
        search_k
    )

    print(f"[recommend] catalog size: {len(catalog)}")
    print(f"[recommend] watched movies: {len(watched_ids)}")
    print(f"[recommend] FAISS searched top_k={search_k}, got {len(indices[0])} candidates")

    catalog_by_id = {m["id"]: m for m in catalog}

    results = []

    for i, idx in enumerate(indices[0]):
        movie_id = ids[idx]

        if movie_id in watched_ids:
            continue

        movie = catalog_by_id.get(movie_id)

        if not movie:
            continue

        movie = movie.copy()
        movie["similarity"] = float(scores[0][i])

        results.append(movie)

        if len(results) >= top_k:
            break

    return results

"""Recommendation service."""
from services.catalog_service import load_catalog_index, search_by_vector
import time

def recommend_from_history(user_vec, catalog, history_list, client, top_k=5):
    """
    Generate movie recommendations based on user profile and watch history.

    Args:
        user_vec: User profile vector
        catalog: List of movie catalogs
        history_list: List of watched movie history items
        client: QdrantClient instance
        top_k: Number of recommendations to return

    Returns:
        List of recommended movies with similarity scores
    """
    if user_vec is None:
        return []

    load_catalog_index(catalog, client)

    watched_ids = {item["movie"]["id"] for item in history_list}

    search_k = top_k * 3
    start_time = time.time()
    results = search_by_vector(user_vec, search_k, client)
    print(f"[recommend] Qdrant search time: {time.time() - start_time}s")

    print(f"[recommend] catalog size: {len(catalog)}")
    print(f"[recommend] watched movies: {len(watched_ids)}")
    print(f"[recommend] Qdrant searched top_k={search_k}, got {len(results)} candidates")

    recommendations = []

    for r in results:
        movie_id = r["id"]

        if movie_id in watched_ids:
            continue

        movie = r["payload"].copy()
        movie["similarity"] = r["score"]

        recommendations.append(movie)

        if len(recommendations) >= top_k:
            break

    return recommendations

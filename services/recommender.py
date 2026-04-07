"""Recommendation service."""
import uuid
import numpy as np
from services.catalog_service import search_by_vector
import time
from core.config import settings


def recommend_by_movie_ids(movie_ids: list, client, top_k=settings.default_top_k):
    """
    Generate movie recommendations based on a list of movie IDs.

    Args:
        movie_ids: List of movie ID strings
        client: QdrantClient instance
        top_k: Number of recommendations to return

    Returns:
        List of recommended movie IDs
    """
    if not movie_ids:
        return []

    qdrant_ids = [str(uuid.UUID(int=int(mid))) for mid in movie_ids]

    points = client.retrieve(collection_name="movies", ids=qdrant_ids, with_payload=True, with_vectors=True)

    if not points:
        return []

    vectors = np.array([p.vector for p in points])
    user_vec = np.mean(vectors, axis=0)

    exclude_ids = set(qdrant_ids)

    search_k = top_k * settings.search_multiplier
    start_time = time.time()
    results = search_by_vector(user_vec, search_k, client)
    print(f"[recommend-by-movies] Qdrant search time: {time.time() - start_time}s")

    print(f"[recommend-by-movies] input movies: {len(movie_ids)}")
    print(f"[recommend-by-movies] Qdrant searched top_k={search_k}, got {len(results)} candidates")

    recommendations = []

    for r in results:
        movie_id = r["id"]

        if movie_id in exclude_ids:
            continue

        recommendations.append(str(uuid.UUID(movie_id).int))

        if len(recommendations) >= top_k:
            break

    return recommendations

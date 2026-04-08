"""Recommendation service."""
from services.catalog_service import load_catalog_index


def recommend_from_history(user_vec, history_list, top_k=5):
    """
    Generate movie recommendations based on user profile and watch history.

    Args:
        user_vec: User profile vector
        history_list: List of watched movie history items
        top_k: Number of recommendations to return

    Returns:
        List of recommended movies with similarity scores
    """
    if user_vec is None:
        return []
    
    index, vecs, ids = load_catalog_index()
    
    # Handle empty index
    if len(ids) == 0:
        return []

    watched_ids = {item["movie"]["id"] for item in history_list}

    search_k = min(top_k * 3, len(ids))
    scores, indices = index.search(
        user_vec.reshape(1, -1).astype("float32"),
        search_k
    )

    print(f"[recommend] index size: {len(ids)}")
    print(f"[recommend] watched movies: {len(watched_ids)}")
    print(f"[recommend] FAISS searched top_k={search_k}, got {len(indices[0])} candidates")

    results = []

    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(ids):
            continue
            
        movie_id = ids[idx]

        if movie_id in watched_ids:
            continue

        movie = {"id": movie_id, "similarity": float(scores[0][i])}
        results.append(movie)

        if len(results) >= top_k:
            break

    return results

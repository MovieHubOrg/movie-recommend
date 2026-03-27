import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from util import build_content_string, compute_engagement_score
from get_catalog_embeddings import get_catalog_embeddings

model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")

def build_user_profile(history_list: list) -> np.ndarray:
    """
    Build user profile vector from watch history.
    Returns a vector representing user preferences.
    """
    contents = []
    weights = []

    for item in history_list:
        movie = item.get("movie", {})
        content = build_content_string(movie)
        score = compute_engagement_score(item)
        
        contents.append(content)
        weights.append(score)
        print(f"  {movie['title']:<35} score={score:.3f}")

    # Embed all movies in history
    embeddings = model.encode(contents)  # shape: (N, 384)
    weights = np.array(weights)
    
    # Weighted average
    weights = weights / weights.sum()  # normalize to sum = 1
    user_vec = np.average(embeddings, axis=0, weights=weights)  # shape: (384,)
    
    # Normalize to use cosine similarity
    user_vec = normalize(user_vec.reshape(1, -1))[0]
    return user_vec


def recommend_from_history(user_vec, catalog, history_list, top_k=5):
    watched_ids = {item["movie"]["id"] for item in history_list}
    candidates = [m for m in catalog if m["id"] not in watched_ids]

    if not candidates:
        return []

    scores = cosine_similarity([user_vec], get_catalog_embeddings(candidates))[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [{**candidates[idx], "similarity": round(float(scores[idx]), 4)} for idx in top_indices]


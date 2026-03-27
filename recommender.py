import numpy as np

from catalog_index import load_catalog_index


def recommend_from_history(user_vec, catalog, history_list, top_k=5):

    index, vecs, ids = load_catalog_index(catalog)

    watched_ids = {item["movie"]["id"] for item in history_list}

    scores, indices = index.search(
        user_vec.reshape(1, -1).astype("float32"),
        top_k * 3
    )

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
import numpy as np
from sklearn.preprocessing import normalize

from model_loader import model
from util import build_content_string, compute_engagement_score


def build_user_profile(history_list: list):
    contents = []
    weights = []

    for item in history_list:

        movie = item.get("movie", {})

        contents.append(build_content_string(movie))
        weights.append(compute_engagement_score(item))

    embeddings = model.encode(
        contents,
        batch_size=32,
        convert_to_numpy=True
    )

    weights = np.array(weights)
    weights = weights / weights.sum()

    user_vec = np.average(embeddings, axis=0, weights=weights)

    return normalize(user_vec.reshape(1, -1))[0]
"""User profile building service."""
import numpy as np
from typing import Union
from sklearn.preprocessing import normalize

from ml.model_loader import model
from utils.text import build_content_string
from utils.engagement import compute_engagement_score


def build_user_profile(history_list: list) -> Union[np.ndarray, None]:
    """
    Build a user profile vector from watch history.

    Creates a weighted average of movie embeddings based on engagement scores.

    Args:
        history_list: List of watch history items

    Returns:
        Normalized user profile vector
    """
    contents = []
    weights = []

    for item in history_list:
        movie = item.get("movie", {})
        contents.append(build_content_string(movie))
        weights.append(compute_engagement_score(item))

    if not contents:
        return None

    embeddings = model.encode(
        contents,
        batch_size=32,
        convert_to_numpy=True
    )

    weights = np.array(weights)
    weight_sum = weights.sum()
    if weight_sum == 0:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights = weights / weight_sum

    user_vec = np.average(embeddings, axis=0, weights=weights)

    return normalize(user_vec.reshape(1, -1))[0]

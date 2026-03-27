"""Recommendation endpoints."""
from fastapi import APIRouter

from services import build_user_profile, recommend_from_history
from mock_data import movie_list, movie_history_list

router = APIRouter()


@router.get("/recommend")
def get_movie_recommend(top_k: int = 5):
    """
    Get movie recommendations based on watch history.

    Args:
        top_k: Number of recommendations to return (default: 5)
    """
    history = movie_history_list["data"]
    catalog = movie_list["data"]["content"]

    user_vec = build_user_profile(history)
    recs = recommend_from_history(user_vec, catalog, history, top_k=top_k)

    return {"data": recs}

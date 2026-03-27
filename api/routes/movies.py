"""Movie and history endpoints."""
from fastapi import APIRouter

from mock_data import movie_list, movie_history_list

router = APIRouter()


@router.get("")
def get_movies():
    """Get all movies."""
    return movie_list


@router.get("/histories")
def get_movie_histories():
    """Get movie watch histories."""
    return movie_history_list

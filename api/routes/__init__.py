from fastapi import APIRouter
from .movie import router as movie_router

router = APIRouter()
router.include_router(movie_router, prefix="/movie", tags=["movies"])

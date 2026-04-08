from fastapi import APIRouter
from .movie import router as movie_router
from .admin import router as admin_router

router = APIRouter()
router.include_router(movie_router, prefix="/movie", tags=["movies"])
router.include_router(admin_router, prefix="/admin", tags=["admin"])

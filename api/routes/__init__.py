from fastapi import APIRouter
from .movies import router as movies_router
from .recommendations import router as recommendations_router

router = APIRouter()

router.include_router(movies_router, prefix="/movies", tags=["movies"])
router.include_router(recommendations_router, prefix="/movie", tags=["recommendations"])

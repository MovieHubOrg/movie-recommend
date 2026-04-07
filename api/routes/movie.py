"""Movie API endpoints."""
from fastapi import APIRouter, Request, Response

from core.config import settings
from services import recommend_by_movie_ids

router = APIRouter()


@router.get("/recommend-by-movies")
def get_recommend_by_movies(request: Request, movieIds: str, top_k: int = settings.default_top_k):
    """Get movie recommendations based on a list of movie IDs."""
    ids = [mid.strip() for mid in movieIds.split(",") if mid.strip()]
    if not ids:
        return Response(
            content='{"result": false, "message": "movieIds cannot be empty"}',
            status_code=400,
            media_type="application/json"
        )

    qdrant_client = request.app.state.qdrant_client
    recs = recommend_by_movie_ids(ids, qdrant_client, top_k=top_k)

    return {"result": True, "message": "Get recommend list successfully", "data": recs}
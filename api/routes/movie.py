"""Movie API endpoints."""
import json
import requests
import xmltodict
import jwt
from fastapi import APIRouter, Header, Response
from typing import Optional

from core.config import settings
from services import recommend_from_history, is_index_ready, sync_catalog, get_or_create_user_embedding

router = APIRouter()


def parse_response(response: requests.Response) -> Response:
    """Convert response content, transforming XML errors to JSON."""
    content_type = response.headers.get("content-type", "")
    content = response.content

    if "xml" in content_type or (content and content.lstrip().startswith(b"<")):
        try:
            parsed = xmltodict.parse(content)
            content = json.dumps({"result": False, **parsed}).encode()
        except Exception:
            pass
        return Response(content=content, status_code=response.status_code, media_type="application/json")

    return Response(content=content, status_code=response.status_code, media_type="application/json")


def get_user_id_from_token(authorization: str) -> Optional[str]:
    """Extract user ID from JWT token without verification."""
    try:
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            # Decode without verification to get user_id
            payload = jwt.decode(token, options={"verify_signature": False})
            return str(payload.get("sub") or payload.get("user_id") or payload.get("id", ""))
    except Exception:
        pass
    return None


def get_movie_client(authorization: Optional[str] = None):
    """Create movie API client with Bearer token from header."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = ""

    class MovieClient:
        def __init__(self):
            self.base_url = settings.movie_api
            self.token = token
            self.session = requests.Session()
            if self.token:
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            self.session.headers.update({"X-Client-Type": "WEB"})

        def get_movie_list(self, page=0, size=1000000):
            return self.session.get(f"{self.base_url}/movie/list?page={page}&size={size}")

        def get_movie_history(self):
            return self.session.get(f"{self.base_url}/movie/history")

    return MovieClient()


@router.get("/list")
def read_movie_list():
    """Get movie list from external API."""
    client = get_movie_client()
    return parse_response(client.get_movie_list(page=0, size=1000000))


@router.get("/history")
def read_movie_history(authorization: str = Header(None)):
    """Get movie history from external API."""
    if not authorization:
        return Response(
            content='{"result": false, "message": "Missing Authorization header"}',
            status_code=401,
            media_type="application/json"
        )

    client = get_movie_client(authorization)
    return parse_response(client.get_movie_history())


@router.get("/recommend")
def get_movie_recommend(top_k: int = 5, authorization: str = Header(None)):
    """Get movie recommendations based on watch history."""
    if not authorization:
        return Response(
            content='{"result": false, "message": "Missing Authorization header"}',
            status_code=401,
            media_type="application/json"
        )

    # Check if index is ready, trigger sync if not
    if not is_index_ready():
        print("[recommend] Index not ready, triggering sync...")
        result = sync_catalog()
        if not result["success"]:
            return {"result": False, "message": "Catalog sync failed. Please try again later."}

    client = get_movie_client(authorization)

    # Only fetch user history (catalog already indexed)
    history_response = client.get_movie_history()

    if history_response.status_code != 200:
        return parse_response(history_response)

    history_data = history_response.json()
    history = history_data.get("data", history_data.get("content", []))

    # Get user_id from token for caching
    user_id = get_user_id_from_token(authorization)
    
    if user_id:
        # Use cached embedding or create new one
        user_vec = get_or_create_user_embedding(user_id, history)
    else:
        # Fallback: build without caching
        from services import build_user_profile
        user_vec = build_user_profile(history)
    
    recs = recommend_from_history(user_vec, history, top_k=top_k)

    return {"result": True, "message": "Get recommend list successfully", "data": recs}
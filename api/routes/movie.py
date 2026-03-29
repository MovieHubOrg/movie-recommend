"""Movie API endpoints."""
import json
import requests
import xmltodict
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, Header, Response
from typing import Optional
from qdrant_client import QdrantClient

from core.config import settings
from services import build_user_profile, recommend_from_history

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)

qdrant_client = QdrantClient(path=settings.qdrant_path)


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


_session = requests.Session()
_session.headers.update({"X-Client-Type": "WEB"})


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
            self.session = _session
            if self.token:
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})

        def get_movie_list(self, page=0, size=1_000_000):
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

    client = get_movie_client(authorization)

    future_history = executor.submit(client.get_movie_history)
    future_list = executor.submit(client.get_movie_list, page=0, size=1000000)

    history_response = future_history.result()
    list_response = future_list.result()

    if history_response.status_code != 200:
        return parse_response(history_response)

    if list_response.status_code != 200:
        return parse_response(list_response)

    history_data = history_response.json()

    list_data = list_response.json()

    history_raw = history_data.get("data", history_data.get("content", []))
    if isinstance(history_raw, list):
        history = history_raw
    else:
        history = []
    
    list_raw = list_data.get("data", list_data.get("content", []))
    if isinstance(list_raw, dict):
        catalog = list_raw.get("content", [])
    elif isinstance(list_raw, list):
        catalog = list_raw
    else:
        catalog = []

    print(f"[recommend] history parsed: {len(history)} items")
    print(f"[recommend] catalog parsed: {len(catalog)} items")

    user_vec = build_user_profile(history)
    recs = recommend_from_history(user_vec, catalog, history, qdrant_client, top_k=top_k)

    return {"result": True, "message": "Get recommend list successfully", "data": recs}
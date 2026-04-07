"""Movie Recommendation API - FastAPI Application."""
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

import requests
from fastapi import FastAPI
from qdrant_client import QdrantClient

from api.routes import router as api_router
from core.config import settings
from services.catalog_service import load_catalog_index


def fetch_catalog() -> list:
    """Fetch movie catalog from external API."""
    session = requests.Session()
    session.headers.update({"X-Client-Type": "WEB"})
    response = session.get(f"{settings.movie_api}/movie/list?page=0&size=1000000")
    response.raise_for_status()
    data = response.json()
    list_raw = data.get("data", data.get("content", []))
    if isinstance(list_raw, dict):
        return list_raw.get("content", [])
    elif isinstance(list_raw, list):
        return list_raw
    return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading catalog index...")
    client = QdrantClient(path=settings.qdrant_path)
    catalog = fetch_catalog()
    print(f"Fetched {len(catalog)} movies from catalog API")
    load_catalog_index(catalog=catalog, client=client)
    print("Catalog index loaded successfully")
    app.state.qdrant_client = client
    yield


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

# Include API routers
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def hello():
    """Health check endpoint."""
    return {"message": "Hello World"}

print("ENV file path:", Path(__file__).parent / ".env")
print("File exists:", (Path(__file__).parent / ".env").exists())
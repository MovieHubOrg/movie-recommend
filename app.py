"""Movie Recommendation API - FastAPI Application."""
from fastapi import FastAPI

from api.routes import router as api_router
from core.config import settings

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
)

# Include API routers
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def hello():
    """Health check endpoint."""
    return {"message": "Hello World"}

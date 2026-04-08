"""Movie Recommendation API - FastAPI Application."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI

from api.routes import router as api_router
from core.config import settings
from core.scheduler import start_scheduler, stop_scheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup
    start_scheduler()
    yield
    # Shutdown
    stop_scheduler()


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan
)

# Include API routers
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def hello():
    """Health check endpoint."""
    return {"message": "Hello World"}


print("ENV file path:", Path(__file__).parent / ".env")
print("File exists:", (Path(__file__).parent / ".env").exists())
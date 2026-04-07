"""Application configuration."""
import os


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        self.app_name: str = os.getenv("APP_NAME")
        self.debug: bool = os.getenv("DEBUG").lower() == "true"
        self.model_name: str = os.getenv("MODEL_NAME")
        self.cache_folder: str = os.getenv("CACHE_FOLDER")
        self.default_top_k: int = int(os.getenv("DEFAULT_TOP_K"))
        self.search_multiplier: int = int(os.getenv("SEARCH_MULTIPLIER"))
        self.movie_api: str = os.getenv("MOVIE_API")
        self.qdrant_path: str = os.getenv("QDRANT_PATH")


settings = Settings()

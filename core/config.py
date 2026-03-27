"""Application configuration."""
import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # API Settings
        self.app_name: str = os.getenv("APP_NAME", "Movie Recommend API")
        self.debug: bool = os.getenv("DEBUG", "False").lower() == "true"

        # Model Settings
        self.model_name: str = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
        self.cache_folder: str = os.getenv("CACHE_FOLDER", "./models")

        # Index Settings
        self.index_file: str = os.getenv("INDEX_FILE", "./models/catalog_faiss.index")
        self.embeddings_cache: str = os.getenv("EMBEDDINGS_CACHE", "./models/catalog_embeddings.npz")

        # Recommendation Settings
        self.default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
        self.search_multiplier: int = int(os.getenv("SEARCH_MULTIPLIER", "3"))

    @property
    def use_gpu(self) -> bool:
        """Check if GPU should be used."""
        return os.getenv("USE_GPU", "True").lower() == "true"


settings = Settings()

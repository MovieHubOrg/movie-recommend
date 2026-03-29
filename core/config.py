"""Application configuration."""
import os


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # API Settings
        self.app_name: str = os.getenv("APP_NAME")
        self.debug: bool = os.getenv("DEBUG").lower() == "true"

        # Model Settings
        self.model_name: str = os.getenv("MODEL_NAME")
        self.cache_folder: str = os.getenv("CACHE_FOLDER")

        # Index Settings
        self.index_file: str = os.getenv("INDEX_FILE")
        self.embeddings_cache: str = os.getenv("EMBEDDINGS_CACHE")

        # Recommendation Settings
        self.default_top_k: int = int(os.getenv("DEFAULT_TOP_K"))
        self.search_multiplier: int = int(os.getenv("SEARCH_MULTIPLIER"))

        # Movie API
        self.movie_api: str = os.getenv("MOVIE_API")

        # Qdrant Settings
        self.qdrant_path: str = os.getenv("QDRANT_PATH")

    @property
    def use_gpu(self) -> bool:
        """Check if GPU should be used."""
        return os.getenv("USE_GPU", "True").lower() == "true"


settings = Settings()

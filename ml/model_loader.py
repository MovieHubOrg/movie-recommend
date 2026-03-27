from sentence_transformers import SentenceTransformer
import torch


def get_device() -> str:
    """Get available device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cache_folder: str = "./models", device: str = None) -> SentenceTransformer:
    """Load sentence transformer model."""
    if device is None:
        device = get_device()

    print(f"Loading model on device: {device}")
    return SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_folder, device=device)


# Global model instance
device = get_device()
print("Device:", device)
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models", device=device)

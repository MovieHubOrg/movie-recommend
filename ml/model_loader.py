from sentence_transformers import SentenceTransformer
import torch


def get_device() -> str:
    """Get available device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# Global model instance
device = get_device()
print("Device:", device)
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models", device=device)

from sentence_transformers import SentenceTransformer
import torch
from core.config import settings

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
model = SentenceTransformer(settings.model_name, cache_folder=settings.cache_folder, device=device)

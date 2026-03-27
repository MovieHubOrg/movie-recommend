from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models", device=device)
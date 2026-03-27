import numpy as np
import os
from util import build_content_string
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

CACHE_FILE = "./models/catalog_embeddings.npz"

model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")

def get_catalog_embeddings(candidates):
    cache_key = str(sorted([m["id"] for m in candidates]))
    
    if os.path.exists(CACHE_FILE):
        cache = np.load(CACHE_FILE, allow_pickle=True)
        if str(cache["key"]) == cache_key:
            print("  Cache hit — dùng embeddings đã lưu")
            return cache["vecs"]
    
    print("  Cache miss — đang encode...")
    contents = [build_content_string(m) for m in candidates]
    vecs = normalize(model.encode(contents))
    
    np.savez(CACHE_FILE, vecs=vecs, key=cache_key)
    return vecs
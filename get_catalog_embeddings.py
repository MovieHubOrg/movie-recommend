import numpy as np
import os
from util import build_content_string
from sklearn.preprocessing import normalize
from model_loader import model

CACHE_FILE = "./models/catalog_embeddings.npz"
def build_catalog_cache(catalog: list):
    print(f"  Encoding {len(catalog)} movies...")
    contents = [build_content_string(m) for m in catalog]
    vecs = normalize(model.encode(contents))
    ids = np.array([m["id"] for m in catalog])

    np.savez(CACHE_FILE, vecs=vecs, ids=ids)
    print(f"  Saved to {CACHE_FILE}")

def load_catalog_embeddings(catalog: list = None):
    if not os.path.exists(CACHE_FILE):
        if catalog is None:
            print("Cache chưa có, cần truyền catalog để build.")
        build_catalog_cache(catalog)

    cache = np.load(CACHE_FILE, allow_pickle=True)
    return cache["vecs"], cache["ids"].tolist()
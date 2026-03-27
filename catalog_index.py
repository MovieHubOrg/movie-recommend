import os
import numpy as np
import faiss

from sklearn.preprocessing import normalize
from model_loader import model
from util import build_content_string

CACHE_FILE = "./models/catalog_embeddings.npz"
INDEX_FILE = "./models/catalog_faiss.index"


def build_catalog_index(catalog: list):

    print(f"Encoding {len(catalog)} movies...")

    contents = [build_content_string(m) for m in catalog]

    embeddings = model.encode(
        contents,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = normalize(embeddings).astype("float32")

    ids = np.array([m["id"] for m in catalog])

    np.savez(CACHE_FILE, vecs=embeddings, ids=ids)

    dim = embeddings.shape[1]

    # CPU index
    cpu_index = faiss.IndexFlatIP(dim)

    # Try GPU
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("Using FAISS GPU")
    except:
        index = cpu_index
        print("Using FAISS CPU")

    index.add(embeddings)

    # save CPU version
    faiss.write_index(cpu_index, INDEX_FILE)

    print("Catalog index built")


def load_catalog_index(catalog=None):

    if not os.path.exists(INDEX_FILE):

        if catalog is None:
            raise Exception("Index not found. Need catalog to build.")

        build_catalog_index(catalog)

    index = faiss.read_index(INDEX_FILE)

    cache = np.load(CACHE_FILE, allow_pickle=True)

    vecs = cache["vecs"]
    ids = cache["ids"].tolist()

    return index, vecs, ids
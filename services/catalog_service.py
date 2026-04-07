"""Catalog index and embedding management service."""
import hashlib
import json
import uuid
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ml.embeddings import generate_movie_embeddings

COLLECTION_NAME = "movies"
CATALOG_HASH_FILE = Path(__file__).parent.parent / "qdrant_data" / "catalog_hash.json"


def _int_to_uuid(int_id: int) -> str:
    """Convert integer ID to UUID string."""
    return str(uuid.UUID(int=int_id))


def _compute_catalog_hash(catalog: list) -> str:
    """Compute a hash of the catalog based on sorted movie IDs."""
    ids = sorted(str(m["id"]) for m in catalog)
    return hashlib.md5(",".join(ids).encode()).hexdigest()


def _load_stored_hash() -> str:
    """Load previously stored catalog hash."""
    if CATALOG_HASH_FILE.exists():
        with open(CATALOG_HASH_FILE) as f:
            data = json.load(f)
            return data.get("hash", "")
    return ""


def _store_hash(catalog_hash: str):
    """Store catalog hash to disk."""
    CATALOG_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_HASH_FILE, "w") as f:
        json.dump({"hash": catalog_hash}, f)


def build_catalog_index(catalog: list, client: QdrantClient):
    """
    Build Qdrant index for movie catalog.

    Creates embeddings for all movies and builds a searchable index.

    Args:
        catalog: List of movie dictionaries
        client: QdrantClient instance
    """
    print(f"Encoding {len(catalog)} movies...")
    print(f"[build] catalog[0] keys: {list(catalog[0].keys()) if catalog else 'empty'}")
    print(f"[build] catalog[0] id: {catalog[0].get('id') if catalog else 'N/A'}")

    embeddings, movie_ids = generate_movie_embeddings(catalog, batch_size=64)
    print(f"[build] embeddings shape: {embeddings.shape}, movie_ids: {movie_ids[:3]}")

    embeddings = embeddings.astype("float32")

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    dim = embeddings.shape[1]
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=_int_to_uuid(int(m["id"])), vector=embeddings[i].tolist(), payload=m)
        for i, m in enumerate(catalog)
    ]
    print(f"[build] points to upsert: {len(points)}, first id: {points[0].id}")

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    _store_hash(_compute_catalog_hash(catalog))

    print("Catalog index built")


def load_catalog_index(catalog=None, client: QdrantClient = None):
    """
    Load Qdrant index.

    Builds the index if it doesn't exist or if the catalog has changed.

    Args:
        catalog: Optional catalog to build index if not found
        client: QdrantClient instance (required)

    Returns:
        QdrantClient instance

    Raises:
        Exception: If index not found and catalog not provided
    """
    if client is None:
        raise Exception("QdrantClient is required")

    should_rebuild = False
    
    if not client.collection_exists(COLLECTION_NAME):
        should_rebuild = True
    else:
        info = client.get_collection(collection_name=COLLECTION_NAME)
        if info.points_count == 0:
            should_rebuild = True
            client.delete_collection(collection_name=COLLECTION_NAME)
        elif catalog is not None:
            current_hash = _compute_catalog_hash(catalog)
            stored_hash = _load_stored_hash()
            if current_hash != stored_hash:
                print(f"[catalog] hash mismatch, rebuilding index (stored: {stored_hash}, current: {current_hash})")
                should_rebuild = True
                client.delete_collection(collection_name=COLLECTION_NAME)
    
    if should_rebuild:
        if catalog is None:
            raise Exception("Index not found. Need catalog to build.")
        build_catalog_index(catalog, client)

    return client


def search_by_vector(query_vector: np.ndarray, top_k: int, client: QdrantClient):
    """
    Search Qdrant index by query vector.

    Args:
        query_vector: Query embedding vector
        top_k: Number of results to return
        client: QdrantClient instance

    Returns:
        List of search results with id and score
    """
    print(f"[search] query_vector shape: {query_vector.shape}, norm: {np.linalg.norm(query_vector)}")
    print(f"[search] collection exists: {client.collection_exists(COLLECTION_NAME)}")
    
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"[search] points count: {info.points_count}")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True,
    ).points

    return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
"""Catalog synchronization service for scheduled updates."""
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import faiss
import requests
from sklearn.preprocessing import normalize

from core.config import settings
from ml.embeddings import generate_movie_embeddings

logger = logging.getLogger(__name__)

CACHE_FILE = settings.embeddings_cache
INDEX_FILE = settings.index_file
METADATA_FILE = settings.embeddings_cache.replace(".npz", "_metadata.json")


def _compute_movie_hash(movie: dict) -> str:
    """Compute hash of movie content for change detection."""
    content = json.dumps({
        "id": movie.get("id"),
        "title": movie.get("title", ""),
        "originalTitle": movie.get("originalTitle", ""),
        "description": movie.get("description", ""),
        "categories": [c.get("name", "") for c in movie.get("categories", [])],
        "country": movie.get("country", ""),
        "year": movie.get("year", ""),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


def _load_metadata() -> dict:
    """Load sync metadata from file."""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"hashes": {}, "last_sync": None}


def _save_metadata(metadata: dict):
    """Save sync metadata to file."""
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def fetch_catalog_from_api() -> Optional[list]:
    """Fetch movie catalog from external API."""
    try:
        session = requests.Session()
        session.headers.update({"X-Client-Type": "WEB"})
        response = session.get(
            f"{settings.movie_api}/movie/list",
            params={"page": 0, "size": 1000000},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        catalog = data.get("data", {}).get("content", data.get("data", []))
        logger.info(f"Fetched {len(catalog)} movies from API")
        return catalog
    except Exception as e:
        logger.error(f"Failed to fetch catalog: {e}")
        return None


def sync_catalog(force_full: bool = False) -> dict:
    """
    Synchronize catalog embeddings with external API.
    
    Performs incremental update by default - only embeds new/changed movies.
    
    Args:
        force_full: If True, re-embed entire catalog regardless of changes
        
    Returns:
        Dict with sync results: new_count, updated_count, total_count, success
    """
    result = {
        "success": False,
        "new_count": 0,
        "updated_count": 0,
        "removed_count": 0,
        "total_count": 0,
        "message": "",
        "timestamp": datetime.now().isoformat()
    }
    
    # Fetch catalog
    catalog = fetch_catalog_from_api()
    if catalog is None:
        result["message"] = "Failed to fetch catalog from API"
        return result
    
    if not catalog:
        result["message"] = "Empty catalog received"
        return result
    
    result["total_count"] = len(catalog)
    
    # Load existing metadata
    metadata = _load_metadata()
    old_hashes = metadata.get("hashes", {})
    
    # Compute new hashes and detect changes
    new_hashes = {}
    new_movies = []
    updated_movies = []
    unchanged_ids = []
    
    for movie in catalog:
        movie_id = str(movie["id"])
        movie_hash = _compute_movie_hash(movie)
        new_hashes[movie_id] = movie_hash
        
        if force_full:
            new_movies.append(movie)
        elif movie_id not in old_hashes:
            new_movies.append(movie)
        elif old_hashes[movie_id] != movie_hash:
            updated_movies.append(movie)
        else:
            unchanged_ids.append(movie_id)
    
    # Detect removed movies
    removed_ids = set(old_hashes.keys()) - set(new_hashes.keys())
    result["removed_count"] = len(removed_ids)
    
    movies_to_embed = new_movies + updated_movies
    result["new_count"] = len(new_movies)
    result["updated_count"] = len(updated_movies)
    
    logger.info(
        f"Sync: {len(new_movies)} new, {len(updated_movies)} updated, "
        f"{len(unchanged_ids)} unchanged, {len(removed_ids)} removed"
    )
    
    # If no changes and index exists, skip embedding
    if not movies_to_embed and not removed_ids and os.path.exists(INDEX_FILE):
        result["success"] = True
        result["message"] = "No changes detected"
        return result
    
    # Load existing embeddings if incremental update
    existing_vecs = {}
    if not force_full and os.path.exists(CACHE_FILE):
        try:
            cache = np.load(CACHE_FILE, allow_pickle=True)
            old_vecs = cache["vecs"]
            old_ids = cache["ids"].tolist()
            for i, mid in enumerate(old_ids):
                str_mid = str(mid)
                if str_mid in unchanged_ids:
                    existing_vecs[str_mid] = old_vecs[i]
        except Exception as e:
            logger.warning(f"Could not load existing cache: {e}")
    
    # Generate embeddings for new/updated movies
    new_vecs = {}
    if movies_to_embed:
        logger.info(f"Embedding {len(movies_to_embed)} movies...")
        embeddings, ids = generate_movie_embeddings(movies_to_embed, batch_size=64)
        embeddings = normalize(embeddings).astype("float32")
        for i, mid in enumerate(ids):
            new_vecs[str(mid)] = embeddings[i]
    
    # Merge embeddings (maintain catalog order)
    final_vecs = []
    final_ids = []
    for movie in catalog:
        movie_id = str(movie["id"])
        if movie_id in new_vecs:
            final_vecs.append(new_vecs[movie_id])
        elif movie_id in existing_vecs:
            final_vecs.append(existing_vecs[movie_id])
        else:
            # Should not happen, but fallback to embedding
            logger.warning(f"Missing embedding for movie {movie_id}, generating...")
            emb, _ = generate_movie_embeddings([movie], batch_size=1)
            final_vecs.append(normalize(emb).astype("float32")[0])
        final_ids.append(movie["id"])
    
    final_vecs = np.array(final_vecs, dtype="float32")
    final_ids = np.array(final_ids)
    
    # Save embeddings cache
    np.savez(CACHE_FILE, vecs=final_vecs, ids=final_ids)
    
    # Build and save FAISS index
    dim = final_vecs.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(final_vecs)
    faiss.write_index(cpu_index, INDEX_FILE)
    
    # Save metadata
    metadata["hashes"] = new_hashes
    metadata["last_sync"] = result["timestamp"]
    _save_metadata(metadata)
    
    result["success"] = True
    result["message"] = f"Synced successfully: {len(new_movies)} new, {len(updated_movies)} updated"
    logger.info(result["message"])
    
    return result


def get_last_sync_info() -> Optional[dict]:
    """Get information about last sync operation."""
    metadata = _load_metadata()
    if metadata.get("last_sync"):
        return {
            "last_sync": metadata["last_sync"],
            "movie_count": len(metadata.get("hashes", {}))
        }
    return None

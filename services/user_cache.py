"""User embedding cache service."""
import os
import json
import hashlib
import numpy as np
from typing import Optional, Tuple

from core.config import settings

USER_CACHE_DIR = os.path.join(os.path.dirname(settings.embeddings_cache), "user_embeddings")


def _ensure_cache_dir():
    """Ensure user cache directory exists."""
    if not os.path.exists(USER_CACHE_DIR):
        os.makedirs(USER_CACHE_DIR)


def _get_user_cache_path(user_id: str) -> str:
    """Get cache file path for a user."""
    return os.path.join(USER_CACHE_DIR, f"user_{user_id}.npz")


def _compute_history_hash(history_list: list) -> str:
    """Compute hash of user history for change detection."""
    movie_ids = sorted([item.get("movie", {}).get("id", 0) for item in history_list])
    content = json.dumps(movie_ids, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def get_cached_user_embedding(user_id: str, history_list: list) -> Optional[np.ndarray]:
    """
    Get cached user embedding if valid.
    
    Args:
        user_id: User identifier
        history_list: Current user history
        
    Returns:
        Cached embedding if valid, None otherwise
    """
    cache_path = _get_user_cache_path(user_id)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        cache = np.load(cache_path, allow_pickle=True)
        cached_hash = str(cache["history_hash"])
        current_hash = _compute_history_hash(history_list)
        
        if cached_hash == current_hash:
            print(f"[user_cache] HIT for user {user_id}")
            return cache["vector"]
        else:
            print(f"[user_cache] MISS for user {user_id} (history changed)")
            return None
    except Exception as e:
        print(f"[user_cache] Error loading cache for user {user_id}: {e}")
        return None


def save_user_embedding(user_id: str, history_list: list, vector: np.ndarray):
    """
    Save user embedding to cache.
    
    Args:
        user_id: User identifier
        history_list: User history used for embedding
        vector: User embedding vector
    """
    _ensure_cache_dir()
    cache_path = _get_user_cache_path(user_id)
    history_hash = _compute_history_hash(history_list)
    
    np.savez(
        cache_path,
        vector=vector,
        history_hash=history_hash,
        history_count=len(history_list)
    )
    print(f"[user_cache] Saved embedding for user {user_id}")


def get_or_create_user_embedding(user_id: str, history_list: list) -> Optional[np.ndarray]:
    """
    Get user embedding from cache or create new one.
    
    Args:
        user_id: User identifier
        history_list: User watch history
        
    Returns:
        User embedding vector
    """
    from services.user_profiling import build_user_profile
    
    # Try cache first
    cached = get_cached_user_embedding(user_id, history_list)
    if cached is not None:
        return cached
    
    # Build new embedding
    vector = build_user_profile(history_list)
    
    if vector is not None:
        save_user_embedding(user_id, history_list, vector)
    
    return vector


def clear_user_cache(user_id: Optional[str] = None):
    """
    Clear user embedding cache.
    
    Args:
        user_id: Specific user to clear, or None to clear all
    """
    if user_id:
        cache_path = _get_user_cache_path(user_id)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"[user_cache] Cleared cache for user {user_id}")
    else:
        if os.path.exists(USER_CACHE_DIR):
            for f in os.listdir(USER_CACHE_DIR):
                os.remove(os.path.join(USER_CACHE_DIR, f))
            print("[user_cache] Cleared all user caches")

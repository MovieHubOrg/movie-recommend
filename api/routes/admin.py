"""Admin API endpoints for system management."""
from fastapi import APIRouter

from services import sync_catalog, get_last_sync_info, reload_catalog_index

router = APIRouter()


@router.post("/sync")
def trigger_sync(force_full: bool = False):
    """
    Manually trigger catalog sync.
    
    Args:
        force_full: If True, re-embed entire catalog regardless of changes
    """
    result = sync_catalog(force_full=force_full)
    
    if result["success"]:
        # Reload index cache after sync
        try:
            reload_catalog_index()
        except Exception:
            pass
    
    return result


@router.get("/sync/status")
def get_sync_status():
    """Get information about last sync operation."""
    info = get_last_sync_info()
    if info:
        return {"result": True, "data": info}
    return {"result": False, "message": "No sync has been performed yet"}

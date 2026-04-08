"""Background scheduler for catalog sync."""
import logging
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from core.config import settings

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


def _sync_catalog_job():
    """Background job to sync catalog."""
    from services.sync_service import sync_catalog
    logger.info("Starting scheduled catalog sync...")
    result = sync_catalog()
    if result["success"]:
        logger.info(f"Scheduled sync completed: {result['message']}")
    else:
        logger.error(f"Scheduled sync failed: {result['message']}")


def _check_and_sync_on_startup():
    """Check if index exists, sync if needed on startup."""
    from services.sync_service import sync_catalog
    
    if not os.path.exists(settings.index_file) or not os.path.exists(settings.embeddings_cache):
        logger.info("Index not found, running initial sync...")
        result = sync_catalog()
        if result["success"]:
            logger.info(f"Initial sync completed: {result['message']}")
        else:
            logger.error(f"Initial sync failed: {result['message']}")
    else:
        logger.info("Index exists, skipping startup sync")


def start_scheduler():
    """Start the background scheduler."""
    # Add daily sync job
    scheduler.add_job(
        _sync_catalog_job,
        trigger=CronTrigger(hour=settings.sync_hour, minute=settings.sync_minute),
        id="catalog_sync",
        name="Daily catalog sync",
        replace_existing=True
    )
    
    scheduler.start()
    logger.info(f"Scheduler started. Catalog sync scheduled at {settings.sync_schedule}")
    
    # Check and sync on startup if enabled
    if settings.sync_on_startup:
        _check_and_sync_on_startup()


def stop_scheduler():
    """Stop the background scheduler gracefully."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

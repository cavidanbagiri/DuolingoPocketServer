# tasks/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit
import logging
import asyncio
import os

# Import your DATABASE_URL from environment
from app.database.setup import connection_string  # Assuming you export this

from app.tasks.context_cleanup import cleanup_old_inactive_contexts

logger = logging.getLogger(__name__)

# Create scheduler
scheduler = BackgroundScheduler(daemon=True)

# You need to define DATABASE_URL for SQLAlchemy jobstore
# Remove asyncpg from connection string for APScheduler
DATABASE_URL_FOR_SCHEDULER = connection_string.replace('+asyncpg', '')

# Initialize scheduler with jobstore (optional, but good for persistence)
try:
    scheduler.add_jobstore('sqlalchemy', url=DATABASE_URL_FOR_SCHEDULER)
except Exception as e:
    logger.warning(f"Could not setup SQLAlchemy jobstore: {e}")
    logger.info("Using memory jobstore instead")

def start_scheduler():
    """Start the periodic task scheduler"""

    # Avoid duplicate scheduler starts
    if scheduler.running:
        logger.warning("Scheduler is already running!")
        return

    try:
        # Schedule cleanup to run daily at 3 AM
        scheduler.add_job(
            func=lambda: asyncio.run(cleanup_old_inactive_contexts()),
            trigger=CronTrigger(hour=3, minute=0),
            id='context_cleanup',
            name='Clean up old inactive contexts',
            replace_existing=True,
            misfire_grace_time=300  # 5 minutes grace period
        )

        # Also run every 6 hours for testing (optional, remove in production)
        scheduler.add_job(
            func=lambda: asyncio.run(cleanup_old_inactive_contexts()),
            trigger='interval',
            hours=6,
            id='context_cleanup_test',
            name='Test cleanup every 6 hours',
            replace_existing=True
        )

        scheduler.start()
        logger.info("✅ Background scheduler started successfully")

        # Shut down scheduler when app exits
        atexit.register(lambda: scheduler.shutdown())

    except Exception as e:
        logger.error(f"❌ Failed to start scheduler: {str(e)}")
        raise


def stop_scheduler():
    """Stop the scheduler"""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("⏹️ Scheduler stopped")
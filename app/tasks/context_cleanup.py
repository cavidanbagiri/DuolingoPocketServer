# tasks/context_cleanup.py

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

# Import SessionLocal directly instead of get_db
from app.database.setup import SessionLocal
from app.repositories.word_repository import ConversationContextRepository
from app.logging_config import setup_logger

logger = setup_logger(__name__, "ai_context.log")

async def cleanup_old_inactive_contexts():
    """
    Task to clean up old inactive contexts.
    Run this periodically (e.g., every hour or daily).
    """
    # Create a new session instance
    db = SessionLocal()
    try:
        logger.info("🚀 Starting context cleanup task...")

        repo = ConversationContextRepository(db)

        # Delete inactive contexts older than 1 day
        deleted_count = await repo.cleanup_old_inactive_contexts(days_old=1)

        if deleted_count > 0:
            logger.info(f"✅ Cleanup completed. Deleted {deleted_count} old inactive contexts.")
        else:
            logger.info("✅ Cleanup completed. No old contexts to delete.")

        return deleted_count
    except Exception as e:
        logger.error(f"❌ Context cleanup task failed: {str(e)}")
        return 0
    finally:
        await db.close()
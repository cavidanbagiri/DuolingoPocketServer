# tasks/context_cleanup.py

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

# Import SessionLocal directly instead of get_db
from app.database.setup import SessionLocal
from app.repositories.word_repository import ConversationContextRepository, DirectChatContextRepository
from app.logging_config import setup_logger

logger = setup_logger(__name__, "ai_context.log")


# tasks/context_cleanup.py - Update cleanup function


# tasks/context_cleanup.py - Update cleanup_all_old_contexts

async def cleanup_all_old_contexts():
    """
    Clean up OLD INACTIVE CONTEXTS ONLY for word-specific conversations.
    Direct chat contexts are NOT deleted (each user has one permanent context).
    """
    db = SessionLocal()
    try:
        logger.info("🧹 Starting context cleanup (word-specific contexts only)...")

        # Only clean up word-specific contexts
        word_repo = ConversationContextRepository(db)
        word_deleted = await word_repo.cleanup_old_inactive_contexts(days_old=1)

        # DO NOT clean up direct chat contexts - they stay forever
        # direct_repo = DirectChatContextRepository(db)
        # direct_deleted = await direct_repo.cleanup_old_direct_contexts(days_old=1)
        direct_deleted = 0

        total_deleted = word_deleted

        if total_deleted > 0:
            logger.info(f"✅ Cleanup completed. Deleted {total_deleted} contexts:")
            logger.info(f"   - Word-specific contexts: {word_deleted}")
            logger.info(f"   - Direct chat contexts: {direct_deleted} (not deleted - permanent)")
        else:
            logger.info("✅ Cleanup completed. No old word contexts to delete.")

        return {
            "total_deleted": total_deleted,
            "word_contexts_deleted": word_deleted,
            "direct_contexts_deleted": direct_deleted,
            "note": "Direct chat contexts are permanent and not deleted"
        }

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {str(e)}")
        return {
            "total_deleted": 0,
            "error": str(e)
        }
    finally:
        await db.close()



# async def cleanup_all_old_contexts():
#     """
#     Clean up ALL old inactive contexts (both word-specific and direct chat).
#     """
#     db = SessionLocal()
#     try:
#         logger.info("🧹 Starting comprehensive context cleanup...")
#
#         # Clean up word-specific contexts
#         word_repo = ConversationContextRepository(db)
#         word_deleted = await word_repo.cleanup_old_inactive_contexts(days_old=1)
#
#         # Clean up direct chat contexts
#         direct_repo = DirectChatContextRepository(db)
#         direct_deleted = await direct_repo.cleanup_old_direct_contexts(days_old=1)
#
#         total_deleted = word_deleted + direct_deleted
#
#         if total_deleted > 0:
#             logger.info(f"✅ Cleanup completed. Deleted {total_deleted} contexts total:")
#             logger.info(f"   - Word-specific contexts: {word_deleted}")
#             logger.info(f"   - Direct chat contexts: {direct_deleted}")
#         else:
#             logger.info("✅ Cleanup completed. No old contexts to delete.")
#
#         return {
#             "total_deleted": total_deleted,
#             "word_contexts_deleted": word_deleted,
#             "direct_contexts_deleted": direct_deleted
#         }
#
#     except Exception as e:
#         logger.error(f"❌ Comprehensive cleanup failed: {str(e)}")
#         return {
#             "total_deleted": 0,
#             "error": str(e)
#         }
#     finally:
#         await db.close()







# async def cleanup_old_inactive_contexts():
#     """
#     Task to clean up old inactive contexts.
#     Run this periodically (e.g., every hour or daily).
#     """
#     # Create a new session instance
#     db = SessionLocal()
#     try:
#         logger.info("🚀 Starting context cleanup task...")
#
#         repo = ConversationContextRepository(db)
#
#         # Delete inactive contexts older than 1 day
#         deleted_count = await repo.cleanup_old_inactive_contexts(days_old=1)
#
#         if deleted_count > 0:
#             logger.info(f"✅ Cleanup completed. Deleted {deleted_count} old inactive contexts.")
#         else:
#             logger.info("✅ Cleanup completed. No old contexts to delete.")
#
#         return deleted_count
#     except Exception as e:
#         logger.error(f"❌ Context cleanup task failed: {str(e)}")
#         return 0
#     finally:
#         await db.close()
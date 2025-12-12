# tasks/background_worker.py

import asyncio
import logging
from datetime import datetime
import signal
import sys

from app.logging_config import setup_logger
from app.tasks.context_cleanup import cleanup_old_inactive_contexts

logger = setup_logger(__name__, "ai_context.log")


class BackgroundWorker:
    def __init__(self):
        self.cleanup_task = None
        self._shutdown = False

    async def periodic_cleanup(self):
        """Run cleanup every 24 hours"""
        while not self._shutdown:
            try:
                # Log when next cleanup will run
                next_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"🔄 Next cleanup scheduled for 24 hours from {next_run}")

                # Wait 24 hours
                await asyncio.sleep(24 * 60 * 60)
                # await asyncio.sleep(300)

                # Run cleanup
                logger.info("🧹 Starting scheduled context cleanup...")
                deleted = await cleanup_old_inactive_contexts()
                logger.info(f"✅ Cleanup completed. Deleted {deleted} contexts.")

            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in periodic cleanup: {e}")
                # Wait 1 hour before retrying on error
                await asyncio.sleep(3600)

    def start(self):
        """Start background worker"""
        if self.cleanup_task and not self.cleanup_task.done():
            logger.warning("Background worker already running")
            return

        self._shutdown = False
        self.cleanup_task = asyncio.create_task(self.periodic_cleanup())
        logger.info("✅ Background worker started")

        # Also run cleanup immediately on startup
        asyncio.create_task(self.run_cleanup_now())

    async def run_cleanup_now(self):
        """Run cleanup immediately (for testing)"""
        try:
            logger.info("🚀 Running immediate cleanup on startup...")
            deleted = await cleanup_old_inactive_contexts()
            logger.info(f"✅ Startup cleanup completed. Deleted {deleted} contexts.")
        except Exception as e:
            logger.error(f"❌ Startup cleanup failed: {e}")

    def stop(self):
        """Stop background worker"""
        self._shutdown = True
        if self.cleanup_task:
            self.cleanup_task.cancel()
        logger.info("⏹️ Background worker stopped")


# Global instance
worker = BackgroundWorker()
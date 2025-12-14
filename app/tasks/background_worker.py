
# tasks/background_worker.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
import traceback

from app.logging_config import setup_logger
from app.tasks.context_cleanup import cleanup_all_old_contexts

logger = setup_logger(__name__, "ai_context.log")

class BackgroundWorker:
    def __init__(self):
        self.cleanup_task = None
        self._shutdown = False
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.total_deleted: int = 0
        self.run_count: int = 0

    async def periodic_cleanup(self):
        """Run cleanup every 24 hours"""
        while not self._shutdown:
            try:
                # Calculate next run time (24 hours from now)
                self.next_run = datetime.now() + timedelta(hours=24)

                # Log schedule information
                last_run_str = self.last_run.strftime('%Y-%m-%d %H:%M:%S') if self.last_run else 'Never'
                next_run_str = self.next_run.strftime('%Y-%m-%d %H:%M:%S')

                logger.info("📅 Cleanup Schedule:")
                logger.info(f"   Last run: {last_run_str}")
                logger.info(f"   Next run: {next_run_str}")
                logger.info(f"   Total contexts deleted so far: {self.total_deleted}")
                logger.info(f"   Total cleanup runs: {self.run_count}")

                # Calculate wait time (24 hours = 86400 seconds)
                wait_seconds = 24 * 60 * 60  # 86400 seconds
                # wait_seconds = 10  # 86400 seconds
                wait_hours = wait_seconds / 3600  # 24 hours

                print(',,,,,,,,,,,,,,,,,,,,,,,,,, checking the removing context ')

                logger.info(f"⏰ Waiting {wait_hours:.1f} hours ({wait_seconds} seconds) until next cleanup...")

                # Wait for the specified time
                await asyncio.sleep(wait_seconds)

                # Run the cleanup
                logger.info("🧹 Starting scheduled context cleanup...")
                start_time = datetime.now()
                self.last_run = start_time

                # Execute the cleanup
                result = await cleanup_all_old_contexts()

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                # Extract the number of deleted contexts
                total_deleted_this_run = 0
                if isinstance(result, dict):
                    total_deleted_this_run = result.get('total_deleted', 0)

                    # Log detailed statistics if available
                    word_deleted = result.get('word_contexts_deleted', 0)
                    direct_deleted = result.get('direct_contexts_deleted', 0)

                    if word_deleted > 0 or direct_deleted > 0:
                        logger.info(f"   Breakdown:")
                        logger.info(f"     - Word-specific contexts: {word_deleted}")
                        logger.info(f"     - Direct chat contexts: {direct_deleted}")
                else:
                    total_deleted_this_run = result

                # Update statistics
                self.total_deleted += total_deleted_this_run
                self.run_count += 1

                # Log completion
                logger.info("✅ Cleanup completed successfully!")
                logger.info(f"   Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"   Duration: {duration:.2f} seconds")
                logger.info(f"   Contexts deleted this run: {total_deleted_this_run}")
                logger.info(f"   Total contexts deleted: {self.total_deleted}")
                logger.info(f"   Total cleanup runs: {self.run_count}")

                # Calculate next run time for the log
                self.next_run = datetime.now() + timedelta(hours=24)
                logger.info(f"   Next cleanup scheduled for: {self.next_run.strftime('%Y-%m-%d %H:%M:%S')}")

            except asyncio.CancelledError:
                # Task was cancelled (normal shutdown)
                logger.info("🛑 Cleanup task cancelled (normal shutdown)")
                break

            except Exception as e:
                # Log error and wait before retrying
                logger.error(f"❌ Error in periodic cleanup: {e}")
                logger.error("Stack trace:")
                logger.error(traceback.format_exc())

                # Wait 1 hour before retrying on error
                logger.info("⏳ Waiting 1 hour before retrying after error...")
                await asyncio.sleep(3600)  # 1 hour

    def start(self):
        """Start the background worker"""
        if self.cleanup_task and not self.cleanup_task.done():
            logger.warning("⚠️ Background worker is already running!")
            return

        self._shutdown = False

        # Start the periodic cleanup task
        self.cleanup_task = asyncio.create_task(self.periodic_cleanup())

        startup_time = datetime.now()
        logger.info(f"🚀 Background worker started at {startup_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Run immediate cleanup on startup (optional)
        asyncio.create_task(self.run_immediate_cleanup())

    async def run_immediate_cleanup(self):
        """Run cleanup immediately on startup"""
        try:
            logger.info("⚡ Running immediate cleanup on startup...")
            start_time = datetime.now()

            result = await cleanup_all_old_contexts()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Update statistics
            if isinstance(result, dict):
                total_deleted = result.get('total_deleted', 0)
            else:
                total_deleted = result

            self.total_deleted += total_deleted
            self.run_count += 1
            self.last_run = start_time

            logger.info(f"✅ Startup cleanup completed in {duration:.2f} seconds")
            logger.info(f"   Contexts deleted: {total_deleted}")
            logger.info(f"   Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            logger.error(f"❌ Startup cleanup failed: {e}")
            logger.error(traceback.format_exc())

    def stop(self):
        """Stop the background worker"""
        self._shutdown = True

        if self.cleanup_task:
            self.cleanup_task.cancel()

        stop_time = datetime.now()
        logger.info(f"⏹️ Background worker stopped at {stop_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def get_status(self) -> dict:
        """Get current status of the worker"""
        now = datetime.now()

        # Format times
        last_run_str = self.last_run.strftime('%Y-%m-%d %H:%M:%S') if self.last_run else "Never"
        next_run_str = self.next_run.strftime('%Y-%m-%d %H:%M:%S') if self.next_run else "Not scheduled yet"

        # Calculate time until next run
        hours_until_next = None
        next_run_in = "Not scheduled"

        if self.next_run:
            seconds_until_next = (self.next_run - now).total_seconds()
            if seconds_until_next > 0:
                hours_until_next = seconds_until_next / 3600
                hours = int(hours_until_next)
                minutes = int((hours_until_next - hours) * 60)
                next_run_in = f"{hours}h {minutes}m"
            else:
                hours_until_next = 0
                next_run_in = "Due now"

        return {
            "running": not self._shutdown,
            "last_run": last_run_str,
            "next_run": next_run_str,
            "hours_until_next": round(hours_until_next, 2) if hours_until_next is not None else None,
            "next_run_in": next_run_in,
            "total_deleted": self.total_deleted,
            "run_count": self.run_count,
            "task_active": self.cleanup_task is not None and not self.cleanup_task.done(),
            "shutdown_requested": self._shutdown
        }

# Global instance
worker = BackgroundWorker()
















#
# # tasks/background_worker.py
#
# import asyncio
# import logging
# from datetime import datetime, timedelta
# import signal
# import sys
# from typing import Optional
#
# from app.logging_config import setup_logger
# from app.tasks.context_cleanup import cleanup_all_old_contexts
#
# logger = setup_logger(__name__, "ai_context.log")
#
# class BackgroundWorker:
#
#     def __init__(self):
#         self.cleanup_task = None
#         self._shutdown = False
#         self.last_run: Optional[datetime] = None
#         self.next_run: Optional[datetime] = None
#         self.total_deleted: int = 0
#         self.run_count: int = 0
#
#
#
#
#
#
#     # async def periodic_cleanup(self):
#     #     """Run cleanup every 24 hours"""
#     #     while not self._shutdown:
#     #         try:
#     #             # Calculate next run time
#     #             self.next_run = datetime.now() + timedelta(hours=24)
#     #
#     #             # Log schedule info
#     #             logger.info(f"📅 Cleanup schedule:")
#     #             logger.info(f"   Last run: {self.last_run or 'Never'}")
#     #             logger.info(f"   Next run: {self.next_run.strftime('%Y-%m-%d %H:%M:%S')}")
#     #             logger.info(f"   Total deleted so far: {self.total_deleted} contexts")
#     #
#     #             wait_seconds = 24 * 60 * 60
#     #             logger.info(f"⏰ Waiting {wait_seconds} s /3600:.1f hours until next cleanup...")
#     #             await asyncio.sleep(wait_seconds)
#     #
#     #             # Run cleanup
#     #             logger.info("🧹 Starting scheduled context cleanup...")
#     #             start_time = datetime.now()
#     #             self.last_run = start_time
#     #
#     #             deleted = await cleanup_all_old_contexts()
#     #
#     #             end_time = datetime.now()
#     #             duration = (end_time - start_time).total_seconds()
#     #
#     #             # Update stats
#     #             self.total_deleted += deleted
#     #             self.run_count += 1
#     #
#     #             logger.info(f"✅ Cleanup completed!")
#     #             logger.info(f"   Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     #             logger.info(f"   Duration: {duration:.2f} seconds")
#     #             logger.info(f"   Deleted this run: {deleted} contexts")
#     #             logger.info(f"   Total deleted: {self.total_deleted} contexts")
#     #             logger.info(f"   Run count: {self.run_count}")
#     #
#     #         except asyncio.CancelledError:
#     #             logger.info("Cleanup task cancelled")
#     #             break
#     #         except Exception as e:
#     #             logger.error(f"❌ Error in periodic cleanup: {e}")
#     #             import traceback
#     #             logger.error(traceback.format_exc())
#     #             # Wait 1 hour before retrying on error
#     #             await asyncio.sleep(3600)
#
#     def start(self):
#         """Start background worker"""
#         if self.cleanup_task and not self.cleanup_task.done():
#             logger.warning("Background worker already running")
#             return
#
#         self._shutdown = False
#         self.cleanup_task = asyncio.create_task(self.periodic_cleanup())
#
#         startup_time = datetime.now()
#         logger.info(f"✅ Background worker started at {startup_time.strftime('%Y-%m-%d %H:%M:%S')}")
#
#         # Also run cleanup immediately on startup
#         asyncio.create_task(self.run_cleanup_now())
#
#     async def run_cleanup_now(self):
#         """Run cleanup immediately (for testing)"""
#         try:
#             logger.info("🚀 Running immediate cleanup on startup...")
#             start_time = datetime.now()
#
#             deleted = await cleanup_old_inactive_contexts()
#
#             end_time = datetime.now()
#             duration = (end_time - start_time).total_seconds()
#
#             self.total_deleted += deleted
#             self.run_count += 1
#             self.last_run = start_time
#
#             logger.info(f"✅ Startup cleanup completed in {duration:.2f} seconds")
#             logger.info(f"   Deleted: {deleted} contexts")
#             logger.info(f"   Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
#
#         except Exception as e:
#             logger.error(f"❌ Startup cleanup failed: {e}")
#
#     def stop(self):
#         """Stop background worker"""
#         self._shutdown = True
#         if self.cleanup_task:
#             self.cleanup_task.cancel()
#         stop_time = datetime.now()
#         logger.info(f"⏹️ Background worker stopped at {stop_time.strftime('%Y-%m-%d %H:%M:%S')}")
#
#     def get_status(self) -> dict:
#         """Get current status of the worker"""
#         now = datetime.now()
#         next_run_str = self.next_run.strftime('%Y-%m-%d %H:%M:%S') if self.next_run else "Not scheduled"
#         last_run_str = self.last_run.strftime('%Y-%m-%d %H:%M:%S') if self.last_run else "Never"
#
#         if self.next_run:
#             seconds_until_next = (self.next_run - now).total_seconds()
#             hours_until_next = seconds_until_next / 3600
#         else:
#             hours_until_next = None
#
#         return {
#             "running": not self._shutdown,
#             "last_run": last_run_str,
#             "next_run": next_run_str,
#             "hours_until_next": round(hours_until_next, 2) if hours_until_next else None,
#             "total_deleted": self.total_deleted,
#             "run_count": self.run_count,
#             "task_active": self.cleanup_task is not None and not self.cleanup_task.done()
#         }
#
# # Global instance
# worker = BackgroundWorker()
#
#
#
#
#
#
#
#
#
#
#
#
# # # tasks/background_worker.py
# #
# # import asyncio
# # import logging
# # from datetime import datetime
# # import signal
# # import sys
# #
# # from app.logging_config import setup_logger
# # from app.tasks.context_cleanup import cleanup_old_inactive_contexts
# #
# # logger = setup_logger(__name__, "ai_context.log")
# #
# #
# # class BackgroundWorker:
# #     def __init__(self):
# #         self.cleanup_task = None
# #         self._shutdown = False
# #
# #     async def periodic_cleanup(self):
# #         """Run cleanup every 24 hours"""
# #         while not self._shutdown:
# #             try:
# #                 # Log when next cleanup will run
# #                 next_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# #                 logger.info(f"🔄 Next cleanup scheduled for 24 hours from {next_run}")
# #
# #                 # Wait 24 hours
# #                 await asyncio.sleep(24 * 60 * 60)
# #                 # await asyncio.sleep(300)
# #
# #                 # Run cleanup
# #                 logger.info("🧹 Starting scheduled context cleanup...")
# #                 deleted = await cleanup_old_inactive_contexts()
# #                 logger.info(f"✅ Cleanup completed. Deleted {deleted} contexts.")
# #
# #             except asyncio.CancelledError:
# #                 logger.info("Cleanup task cancelled")
# #                 break
# #             except Exception as e:
# #                 logger.error(f"❌ Error in periodic cleanup: {e}")
# #                 # Wait 1 hour before retrying on error
# #                 await asyncio.sleep(3600)
# #
# #     def start(self):
# #         """Start background worker"""
# #         if self.cleanup_task and not self.cleanup_task.done():
# #             logger.warning("Background worker already running")
# #             return
# #
# #         self._shutdown = False
# #         self.cleanup_task = asyncio.create_task(self.periodic_cleanup())
# #         logger.info("✅ Background worker started")
# #
# #         # Also run cleanup immediately on startup
# #         asyncio.create_task(self.run_cleanup_now())
# #
# #     async def run_cleanup_now(self):
# #         """Run cleanup immediately (for testing)"""
# #         try:
# #             logger.info("🚀 Running immediate cleanup on startup...")
# #             deleted = await cleanup_old_inactive_contexts()
# #             logger.info(f"✅ Startup cleanup completed. Deleted {deleted} contexts.")
# #         except Exception as e:
# #             logger.error(f"❌ Startup cleanup failed: {e}")
# #
# #     def stop(self):
# #         """Stop background worker"""
# #         self._shutdown = True
# #         if self.cleanup_task:
# #             self.cleanup_task.cancel()
# #         logger.info("⏹️ Background worker stopped")
# #
# #
# # # Global instance
# # worker = BackgroundWorker()
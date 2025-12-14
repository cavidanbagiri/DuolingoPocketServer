# Add to your main.py or create a new admin_router.py

from datetime import datetime
from fastapi import APIRouter, Depends
from app.tasks.background_worker import worker

# Create admin router if you don't have one
router = APIRouter()


@router.get("/cleanup/status")
async def get_cleanup_status():
    """Get detailed status of the cleanup worker"""
    status = worker.get_status()

    # Add server time for reference
    status["server_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status["server_time_utc"] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

    if status["hours_until_next"]:
        hours = int(status["hours_until_next"])
        minutes = int((status["hours_until_next"] - hours) * 60)
        status["next_run_in"] = f"{hours}h {minutes}m"
    else:
        status["next_run_in"] = "Not scheduled"

    return status


@router.post("/cleanup/run-now")
async def run_cleanup_now():
    """Manually trigger cleanup now"""
    from app.tasks.context_cleanup import cleanup_old_inactive_contexts

    try:
        start_time = datetime.now()
        deleted = await cleanup_old_inactive_contexts()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Update worker stats
        worker.total_deleted += deleted
        worker.run_count += 1
        worker.last_run = start_time

        return {
            "success": True,
            "message": f"Cleanup completed in {duration:.2f} seconds",
            "deleted_count": deleted,
            "run_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "duration_seconds": round(duration, 2),
            "total_deleted": worker.total_deleted,
            "total_runs": worker.run_count
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Cleanup failed: {str(e)}"
        }


@router.get("/cleanup/stats")
async def get_cleanup_stats():
    """Get statistics about cleaned up contexts"""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import text
    from app.database.setup import get_db

    db: AsyncSession = get_db()

    try:
        # Get total contexts in database
        result = await db.execute(text("SELECT COUNT(*) as total FROM conversation_contexts"))
        total = result.scalar()

        # Get active contexts
        result = await db.execute(
            text("SELECT COUNT(*) as active FROM conversation_contexts WHERE is_active = true")
        )
        active = result.scalar()

        # Get inactive contexts
        result = await db.execute(
            text("SELECT COUNT(*) as inactive FROM conversation_contexts WHERE is_active = false")
        )
        inactive = result.scalar()

        # Get oldest inactive context
        result = await db.execute(
            text("""
                SELECT word, language, updated_at 
                FROM conversation_contexts 
                WHERE is_active = false 
                ORDER BY updated_at ASC 
                LIMIT 1
            """)
        )
        oldest_inactive = result.fetchone()

        # Get newest context
        result = await db.execute(
            text("""
                SELECT word, language, is_active, updated_at 
                FROM conversation_contexts 
                ORDER BY updated_at DESC 
                LIMIT 1
            """)
        )
        newest = result.fetchone()

        return {
            "database_stats": {
                "total_contexts": total,
                "active_contexts": active,
                "inactive_contexts": inactive,
                "oldest_inactive": {
                    "word": oldest_inactive[0] if oldest_inactive else None,
                    "language": oldest_inactive[1] if oldest_inactive else None,
                    "updated_at": oldest_inactive[2].isoformat() if oldest_inactive else None
                } if oldest_inactive else None,
                "newest_context": {
                    "word": newest[0] if newest else None,
                    "language": newest[1] if newest else None,
                    "is_active": newest[2] if newest else None,
                    "updated_at": newest[3].isoformat() if newest else None
                } if newest else None
            },
            "worker_stats": worker.get_status()
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        await db.aclose()

# Then include this router in your main.py
# app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])
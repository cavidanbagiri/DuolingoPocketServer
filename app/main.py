
# main.py

# ... existing imports ...
import os
import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.routers import user_router, word_router, structure_router, admin_router, status_router
from app.tasks.background_worker import worker


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        # Start the background worker
        worker.start()
        print("✅ Background worker started")

        # Log initial status
        status = worker.get_status()
        print(f"📊 Initial cleanup status:")
        print(f"   Last run: {status['last_run']}")
        print(f"   Next run: {status['next_run']}")
        if status['hours_until_next']:
            print(f"   Next run in: {status['hours_until_next']:.1f} hours")

    except Exception as e:
        print(f"⚠️ Could not start background worker: {e}")
        print("⚠️ Cleanup tasks will not run automatically")

    yield

    # Shutdown
    print(f"⏹️ Shutting down application at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    worker.stop()
    print("✅ Background worker stopped")

app = FastAPI(lifespan=lifespan)

# ... CORS middleware ...
origins = [
    "www.w9999.tech",
    "https://www.w9999.tech",
    "http://www.w9999.tech",
    "https://w9999-web.onrender.com/",
    # "http://localhost:5173",
    # "http://192.168.1.101:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Include Routers
app.include_router(router=user_router.router, prefix='/api/auth', tags=['User'])
app.include_router(router=word_router.router, prefix='/api/words', tags=['Word'])
app.include_router(router=structure_router.router, prefix='/api/structure', tags=['Structure'])

# Include Admin router (add this)
app.include_router(router=admin_router.router, prefix='/api/admin', tags=['Admin'])
app.include_router(router=status_router.router, prefix='/api/status', tags=['Status'])

# ... existing endpoints ...




















#
# # main.py
#
# import os
# import asyncio
# from fastapi import FastAPI
# from contextlib import asynccontextmanager
# from dotenv import load_dotenv
# from fastapi.middleware.cors import CORSMiddleware
#
# from app.routers import user_router, word_router, structure_router
# from app.tasks.background_worker import worker
#
# load_dotenv()
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Lifespan context manager for startup and shutdown events.
#     """
#     # Startup
#     # print("🚀 Starting application...")
#
#     try:
#         # Start the background worker
#         worker.start()
#         # print("✅ Background worker started")
#     except Exception as e:
#         print(f"⚠️ Could not start background worker: {e}")
#         print("⚠️ Cleanup tasks will not run automatically")
#
#     yield
#
#     # Shutdown
#     print("⏹️ Shutting down application...")
#     worker.stop()
#     print("✅ Background worker stopped")
#
# app = FastAPI(lifespan=lifespan)
#
# origins = [
#     "www.w9999.tech",
#     "https://www.w9999.tech",
#     "http://www.w9999.tech",
#     "https://w9999-web.onrender.com/",
#     "http://localhost:5173",
#     "http://192.168.1.101:5173"
# ]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )
#
# # Include Routers
# app.include_router(router=user_router.router, prefix='/api/auth', tags=['User'])
# app.include_router(router=word_router.router, prefix='/api/words', tags=['Word'])
# app.include_router(router=structure_router.router, prefix='/api/structure', tags=['Structure'])
#
# # Manual cleanup endpoint
# @app.post("/api/admin/cleanup")
# async def manual_cleanup():
#     """Manually trigger context cleanup (for testing)"""
#     from tasks.context_cleanup import cleanup_old_inactive_contexts
#     try:
#         deleted = await cleanup_old_inactive_contexts()
#         return {
#             "status": "success",
#             "message": f"Deleted {deleted} old inactive contexts",
#             "deleted_count": deleted
#         }
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
#
# @app.get("/api/admin/cleanup/status")
# async def cleanup_status():
#     """Check cleanup status"""
#     return {
#         "status": "running" if worker.cleanup_task and not worker.cleanup_task.done() else "stopped",
#         "worker_running": not worker._shutdown
#     }
#
# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy", "timestamp": datetime.now().isoformat()}
#
#










#
# import os
#
# from fastapi import FastAPI
# from contextlib import asynccontextmanager
#
# from dotenv import load_dotenv
#
# from fastapi.middleware.cors import CORSMiddleware
#
# from app.routers import user_router, word_router, structure_router
#
# load_dotenv()
#
# app = FastAPI()
#
# origins = [
#     # "http://localhost:5173",
#     # "http://192.168.1.101:5173",
#     # "http://localhost:5174",
#     # "http://192.168.1.76:8081",  # Expo's Metro bundler origin
#     # "http://localhost:8081",
#     # "http://192.168.1.76:19006",  # Expo Web Dev Tools
#     # "http://192.168.1.76:19000",  # Expo Go (Android)
#     # "http://192.168.1.101:8081",
#     "www.w9999.tech",
#     "https://www.w9999.tech",
#     "http://www.w9999.tech",
#     "https://w9999-web.onrender.com/"
#
#     # "*"
# ]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )
#
# # Include Routers
# app.include_router(router=user_router.router, prefix='/api/auth', tags=['User'])
# app.include_router(router=word_router.router, prefix='/api/words', tags=['Word'])
# app.include_router(router=structure_router.router, prefix='/api/structure', tags=['Structure'])
#
#
#
#
#


import os
import asyncio
import datetime
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from app.routers import (user_router, word_router, structure_router,
                         admin_router, status_router, note_router, chat_router, public_router)
from app.tasks.background_worker import worker


load_dotenv()


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
    "https://www.w9999.tech",
    "https://w9999.app",
    "https://www.w9999.app",
    "https://w9999-web.onrender.com/",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://192.168.1.101:5173",
    "http://api.w9999.app",
    "https://api.w9999.app"
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
app.include_router(router=note_router.router, prefix='/api/notes', tags=['Note'])
app.include_router(router=chat_router.router, prefix='/api/chat', tags=['Chat'])
app.include_router(router=structure_router.router, prefix='/api/structure', tags=['Structure'])
app.include_router(router=public_router.router, prefix='/api/public', tags=['Public'])

# Include Admin router (add this)
app.include_router(router=admin_router.router, prefix='/api/admin', tags=['Admin'])
app.include_router(router=status_router.router, prefix='/api/status', tags=['Status'])

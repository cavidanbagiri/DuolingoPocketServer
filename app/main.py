
import os

from fastapi import FastAPI
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

# from src.logging_config import setup_logger

# from app.routers import user_router, translate_router, word_router
from app.routers import user_router, word_router

# logger = setup_logger(__name__, "main.log")


load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://192.168.1.76:8081",  # Expo's Metro bundler origin
    "http://localhost:8081",
    "http://192.168.1.76:19006",  # Expo Web Dev Tools
    "http://192.168.1.76:19000",  # Expo Go (Android)
    "http://192.168.1.101:8081"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(router=user_router.router, prefix='/api/auth', tags=['User'])
app.include_router(router=word_router.router, prefix='/api/words', tags=['Word'])

# app.include_router(router=translate_router.router, prefix='/api/translate', tags=['Translate'])





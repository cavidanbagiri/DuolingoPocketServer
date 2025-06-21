
import os

from fastapi import FastAPI
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

# from src.logging_config import setup_logger

from app.routers import user_router, translate_router

# logger = setup_logger(__name__, "main.log")


load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:5174",
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
app.include_router(router=translate_router.router, prefix='/api/translate', tags=['Translate'])





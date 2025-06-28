import os

from fastapi.concurrency import run_in_threadpool
from functools import lru_cache
import spacy
from typing import Optional
import logging


import httpx
from fastapi import HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from app.models.word_model import WordModel, UserSavedWord
from app.schemas.translate_schema import TranslateSchema, WordSchema

from app.logging_config import setup_logger
logger = setup_logger(__name__, "translate.log")


class TranslateRepository:
    def __init__(self, data: TranslateSchema):
        self.data = data
        self.api_key = os.getenv("YANDEX_TRANSLATE_API_SECRET_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.translate_url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        self.detect_url = "https://translate.api.cloud.yandex.net/translate/v2/detect"

        # Pre-process the input text
        self.processed_text = self._preprocess_text(data.q)

    def _preprocess_text(self, text: str) -> str:
        """Clean and validate input text before sending to API"""
        if not text or not text.strip():
            raise ValueError("Empty text provided for translation")

        # Remove excessive whitespace but ensure at least one space if empty after trim
        text = ' '.join(text.strip().split())
        return text or " "  # Return single space if empty after processing

    async def translate(self):
        # Early return for empty text after preprocessing
        if not self.processed_text or len(self.processed_text) < 2:
            return {
                "translation": self.processed_text,
                "detected_lang": self.data.source if self.data.source != "auto" else None
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }

        payload = {
            "targetLanguageCode": self.data.target,
            "texts": [self.processed_text],
            "folderId": self.folder_id,
        }

        if self.data.source and self.data.source.strip().lower() != "auto":
            payload["sourceLanguageCode"] = self.data.source

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.translate_url,
                    json=payload,
                    headers=headers
                )

                # Add detailed error logging
                if response.status_code != 200:
                    error_detail = response.json().get('message', 'Unknown API error')
                    logger.error(f"Yandex API Error: {error_detail}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Translation API error: {error_detail}"
                    )

                result = response.json()["translations"][0]
                return {
                    "translation": result["text"],
                    "detected_lang": result.get("detectedLanguageCode")
                }

        except httpx.HTTPStatusError as e:
            logger.exception(f"Translation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Translation service unavailable"
            )
        except Exception as ex:
            logger.exception(f"Unexpected error during translation: {ex}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal Server Error"
            )



class SaveWordRepository:
    _nlp = None  # Class-level model instance for thread safety

    def __init__(self, data: WordSchema, user_id: int, db: AsyncSession):
        self.data = data
        self.user_id = int(user_id)
        self.db = db

    @classmethod
    async def get_nlp(cls):
        """Thread-safe model loader with optimized pipeline"""
        if cls._nlp is None:
            cls._nlp = await run_in_threadpool(
                lambda: spacy.load("en_core_web_sm",
                                   exclude=["parser", "ner", "lemmatizer"])
            )
            await run_in_threadpool(
                lambda: cls._nlp.enable_pipe("senter")
            )
        return cls._nlp

    @lru_cache(maxsize=1000)
    async def find_part_of_speech(self, selected_word: str) -> str:
        """Proper async POS tagging using FastAPI's threadpool"""
        if not selected_word.strip():
            return 'other'

        try:
            nlp = await self.get_nlp()

            # Process text in threadpool
            doc = await run_in_threadpool(
                nlp,
                selected_word.lower().strip()
            )

            pos_mapping = {
                'propn': 'noun',
                'noun': 'noun',
                'verb': 'verb',
                'adj': 'adjective',
                'adv': 'adverb',
                'pron': 'pronoun',
                'adp': 'preposition',
                'conj': 'conjunction'
            }

            for token in doc:
                if token.pos_.lower() in pos_mapping:
                    return pos_mapping[token.pos_.lower()]

            return 'other'

        except Exception as e:
            logger.error(f"POS tagging failed for '{selected_word}': {str(e)}")
            return 'other'

    async def save_word(self):
        normalized_word = self.data.word.lower().strip()

        # Get POS tag async
        self.data.part_of_speech = await self.find_part_of_speech(normalized_word)

        # Upsert word
        word = await self.db.execute(
            select(WordModel).where(
                func.lower(WordModel.word) == normalized_word,
                WordModel.from_lang == self.data.from_lang,
                WordModel.to_lang == self.data.to_lang,
                WordModel.translation == self.data.translation
            )
        )
        word = word.scalar()

        if not word:
            word = WordModel(**self.data.model_dump())
            word.word = normalized_word
            self.db.add(word)
            await self.db.flush()
        else:
            logger.info(f"Word '{word.word}' already exists")

        # Upsert user-word relationship
        await self.db.merge(
            UserSavedWord(user_id=self.user_id, word_id=word.id)
        )

        await self.db.commit()
        await self.db.refresh(word)
        return word


from sqlalchemy import select, func, text, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.word_model import WordModel, UserSavedWord
from app.schemas.translate_schema import WordSchema

from fastapi.concurrency import run_in_threadpool
from functools import lru_cache
import spacy
from typing import Optional
import logging


from app.logging_config import setup_logger
logger = setup_logger(__name__, "word.log")



class DashboardRepository:

    def __init__(self, user_id: int, db: AsyncSession):
        self.user_id = int(user_id)
        self.db = db


    async def get_language_pair_stats(self):

        query = """
                SELECT
                    w.from_lang,
                    w.to_lang,
                    COUNT(*) as word_length,
                    SUM(CASE WHEN usw.learned = TRUE THEN 1 ELSE 0 END) as learned,
                    SUM(CASE WHEN usw.starred = TRUE THEN 1 ELSE 0 END) as starred
                FROM words w
                JOIN user_saved_words usw ON w.id = usw.word_id
                WHERE usw.user_id = :user_id
                GROUP BY w.from_lang, w.to_lang
                ORDER BY word_length DESC
            """
        result = await self.db.execute(
            text(query),
            {"user_id": self.user_id}
        )
        return [
            {
                "user_id": self.user_id,
                "from_lang": row.from_lang,
                "to_lang": row.to_lang,
                "word_length": row.word_length,
                "learned": row.learned,
                "starred": row.starred
            }
            for row in result
        ]



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
                WordModel.translation == self.data.translation.strip().lower()
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




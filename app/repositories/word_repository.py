from fastapi import Query, HTTPException
from sqlalchemy import select, func, text, case, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.word_model import WordModel, UserSavedWord
from app.schemas.translate_schema import WordSchema

from fastapi.concurrency import run_in_threadpool
from functools import lru_cache
import spacy

from app.logging_config import setup_logger
from app.schemas.word_schema import ChangeWordStatusSchema

logger = setup_logger(__name__, "word.log")



class DashboardRepository:
    def __init__(self, user_id: int, db: AsyncSession):
        self.user_id = int(user_id)
        self.db = db

    async def get_language_pair_stats(self):
        # Main query for language pair stats
        query = """
            SELECT
                w.from_lang,
                w.to_lang,
                COUNT(*) as word_length,
                SUM(CASE WHEN usw.learned = TRUE THEN 1 ELSE 0 END) as learned,
                SUM(CASE WHEN usw.starred = TRUE THEN 1 ELSE 0 END) as starred,
                (
                    SELECT JSONB_OBJECT_AGG(pos, cnt)
                    FROM (
                        SELECT 
                            w2.part_of_speech as pos, 
                            COUNT(*) as cnt
                        FROM words w2
                        JOIN user_saved_words usw2 ON w2.id = usw2.word_id
                        WHERE usw2.user_id = :user_id
                        AND w2.from_lang = w.from_lang
                        AND w2.to_lang = w.to_lang
                        GROUP BY w2.part_of_speech
                    ) subq
                ) as pos_stats
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

        stats = []
        for row in result:
            # Convert SQLAlchemy row to dict
            stat = {
                "user_id": self.user_id,
                "from_lang": row.from_lang,
                "to_lang": row.to_lang,
                "total_word": row.word_length,
                "learned": row.learned,
                "starred": row.starred,
                "pos_stats": {}
            }

            # Parse POS stats if available
            if row.pos_stats:
                stat["pos_stats"] = dict(row.pos_stats.items())

            stats.append(stat)
        return stats



class DashboardRepositoryLang:
    def __init__(self, user_id: int, db: AsyncSession):
        self.user_id = int(user_id)
        self.db = db

    async def get_language_pair_stats_by_lang(self, from_lang: str, to_lang: str):
        # Get the words with their learned/starred status
        words_result = await self.db.execute(
            select(
                WordModel,
                UserSavedWord.learned,
                UserSavedWord.starred
            )
            .join(UserSavedWord)
            .where(
                WordModel.from_lang == from_lang,
                WordModel.to_lang == to_lang,
                UserSavedWord.user_id == self.user_id,
                UserSavedWord.learned == False
            )
        )

        # Convert results to a list of dictionaries with all fields
        words = []
        for word_model, learned, starred in words_result:
            word_dict = word_model.__dict__
            # word_dict['learned'] = learned
            word_dict['starred'] = starred
            words.append(word_dict)

        # Get POS statistics (unchanged)
        pos_stats = await self.db.execute(
            select(
                WordModel.part_of_speech,
                func.count().label("count")
            )
            .join(UserSavedWord)
            .where(
                WordModel.from_lang == from_lang,
                WordModel.to_lang == to_lang,
                UserSavedWord.user_id == self.user_id
            )
            .group_by(WordModel.part_of_speech)
        )
        pos_dict = {row.part_of_speech: row.count for row in pos_stats}

        # Get learned/starred counts (unchanged)
        status_counts = await self.db.execute(
            select(
                func.sum(case((UserSavedWord.learned == True, 1), else_=0)).label("learned"),
                func.sum(case((UserSavedWord.starred == True, 1), else_=0)).label("starred")
            )
            .join(WordModel)
            .where(
                WordModel.from_lang == from_lang,
                WordModel.to_lang == to_lang,
                UserSavedWord.user_id == self.user_id
            )
        )
        learned, starred = status_counts.first()

        return {
            "words": words,
            "stats": {
                "total_words": len(words),
                "learned": learned or 0,
                "starred": starred or 0,
                "pos_stats": pos_dict
            }
        }



class FilterRepository:

    def __init__(self, user_id: int, db: AsyncSession):
        self.user_id = int(user_id)
        self.db = db

    async def filter(self,  from_lang: str, to_lang: str, part_of_speech: str):

        words_result = await self.db.execute(
            select(WordModel)
            .join(UserSavedWord)
            .where(
                WordModel.from_lang == from_lang,
                WordModel.to_lang == to_lang,
                WordModel.part_of_speech == part_of_speech,
                UserSavedWord.user_id == self.user_id
            )
        )
        words = words_result.scalars().all()

        return words



class ChangeWordStatusRepository:

    def __init__(self, data: ChangeWordStatusSchema,  user_id: int, db: AsyncSession):
        self.data = data
        self.user_id = int(user_id)
        self.db = db

    async def change_word_status(self):

        word = await self.db.execute(select(UserSavedWord).where(UserSavedWord.word_id == self.data.word_id, UserSavedWord.user_id == int(self.user_id)))

        word = word.scalar()

        if not word:
            raise HTTPException(status_code=404, detail="Word not found")

        if self.data.w_status == 'starred':
            new_status = not word.starred
            await self.db.execute(
                update(UserSavedWord)
                .where(UserSavedWord.word_id == self.data.word_id, UserSavedWord.user_id == self.user_id)
                .values(starred=new_status)
            )
            await self.db.commit()

        elif self.data.w_status == 'learned':
            new_status = not word.learned
            await self.db.execute(
                update(UserSavedWord)
                .where(UserSavedWord.word_id == self.data.word_id, UserSavedWord.user_id == self.user_id)
                .values(learned=new_status)
            )
            await self.db.commit()

        return {
            'w_status': self.data.w_status,
            'detail': f'{self.data.w_status.title()} successfully changed',
            'word_id': word.word_id
        }


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




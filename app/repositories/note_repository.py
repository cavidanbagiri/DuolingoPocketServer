



import os
import re
from typing import List, Dict, Any, Optional, Dict
import httpx
import aiohttp
import asyncio
from datetime import datetime, timedelta, date
import hashlib
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import aiohttp

from fastapi import HTTPException, status

from sqlalchemy import select, func, and_, update, or_, case, delete, text, distinct
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging_config import setup_logger
from app.models.word_model import Word, Category
from app.models.user_model import Language, UserWord, NoteModel


logger = setup_logger(__name__, "note.log")

from app.schemas.word_schema import AIWordResponse


class CreateNoteRepository:
    def __init__(self, db: AsyncSession, user_id: int, note_data: dict):
        self.db = db
        self.user_id = user_id
        self.note_data = note_data

    async def create_note(self):
        try:
            # Create new note instance
            note = NoteModel(
                note_name=self.note_data.get('note_name'),
                user_id=self.user_id,
                target_lang=self.note_data.get('target_lang'),
                note_type=self.note_data.get('note_type', 'general'),
                content=self.note_data.get('content'),
                tags=self.note_data.get('tags', []),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Validate content length (50,000 chars max)
            if len(note.content) > 50000:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Note content exceeds maximum length of 50,000 characters"
                )

            # Validate note name length
            if len(note.note_name) > 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Note name exceeds maximum length of 200 characters"
                )

            # Validate note type
            valid_types = ['vocabulary', 'grammar', 'general']
            if note.note_type not in valid_types:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid note type. Must be one of: {', '.join(valid_types)}"
                )

            # Validate language if provided
            if note.target_lang and note.target_lang not in ['es', 'en', 'ru', 'tr']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid language code. Must be: es, en, ru, tr or null"
                )

            # Validate tags
            if note.tags and len(note.tags) > 20:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Maximum 20 tags allowed"
                )

            # Save to database
            self.db.add(note)
            await self.db.commit()
            await self.db.refresh(note)

            # Return the created note
            return {
                "id": note.id,
                "note_name": note.note_name,
                "user_id": note.user_id,
                "target_lang": note.target_lang,
                "note_type": note.note_type,
                "content": note.content,
                "tags": note.tags or [],
                "created_at": note.created_at.isoformat(),
                "updated_at": note.updated_at.isoformat()
            }

        except HTTPException:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error during note creation: {str(e)}")
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred"
            )
        except Exception as e:
            logger.error(f"Unexpected error during note creation: {str(e)}")
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )



class GetNotesRepository:
    def __init__(self, db: AsyncSession, user_id: int, **filters):
        self.db = db
        self.user_id = user_id
        self.filters = filters

    async def get_notes(self):
        try:
            query = select(NoteModel).where(NoteModel.user_id == self.user_id)

            # Apply filters
            if self.filters.get('target_lang'):
                if self.filters['target_lang'] == 'none':
                    query = query.where(NoteModel.target_lang.is_(None))
                else:
                    query = query.where(NoteModel.target_lang == self.filters['target_lang'])

            if self.filters.get('note_type'):
                query = query.where(NoteModel.note_type == self.filters['note_type'])

            # Apply search
            if self.filters.get('search'):
                search_term = f"%{self.filters['search']}%"
                query = query.where(
                    or_(
                        NoteModel.note_name.ilike(search_term),
                        NoteModel.content.ilike(search_term)
                    )
                )

            # Order by most recent first
            query = query.order_by(NoteModel.updated_at.desc())

            # Pagination (optional for future)
            if self.filters.get('skip'):
                query = query.offset(self.filters['skip'])
            if self.filters.get('limit'):
                query = query.limit(self.filters['limit'])

            result = await self.db.execute(query)
            notes = result.scalars().all()

            return [
                {
                    "id": note.id,
                    "note_name": note.note_name,
                    "user_id": note.user_id,
                    "target_lang": note.target_lang,
                    "note_type": note.note_type,
                    "content": note.content,
                    "tags": note.tags or [],
                    "created_at": note.created_at,
                    "updated_at": note.updated_at
                }
                for note in notes
            ]

        except Exception as e:
            logger.error(f"Error in GetNotesRepository: {str(e)}")
            raise



class GetNoteByIdRepository:
    def __init__(self, db: AsyncSession, user_id: int, note_id: int):
        self.db = db
        self.user_id = user_id
        self.note_id = note_id

    async def get_note(self):
        query = select(NoteModel).where(
            NoteModel.id == self.note_id,
            NoteModel.user_id == self.user_id
        )
        result = await self.db.execute(query)
        note = result.scalar_one_or_none()

        if note:
            return {
                "id": note.id,
                "note_name": note.note_name,
                "user_id": note.user_id,
                "target_lang": note.target_lang,
                "note_type": note.note_type,
                "content": note.content,
                "tags": note.tags or [],
                "created_at": note.created_at,
                "updated_at": note.updated_at
            }
        return None



class UpdateNoteRepository:
    def __init__(self, db: AsyncSession, user_id: int, note_id: int, note_data: dict):
        self.db = db
        self.user_id = user_id
        self.note_id = note_id
        self.note_data = note_data

    async def update_note(self):
        try:
            # First, get the note
            query = select(NoteModel).where(
                NoteModel.id == self.note_id,
                NoteModel.user_id == self.user_id
            )
            result = await self.db.execute(query)
            note = result.scalar_one_or_none()

            if not note:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Note not found"
                )

            # Update only the fields that are provided
            for key, value in self.note_data.items():
                if value is not None:
                    # Handle special cases
                    if key == 'tags' and value == []:
                        setattr(note, key, [])
                    elif key == 'target_lang' and value == '':
                        setattr(note, key, None)
                    else:
                        setattr(note, key, value)

            # Update timestamp
            note.updated_at = datetime.utcnow()

            # Validate
            if note.content and len(note.content) > 50000:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Note content exceeds maximum length of 50,000 characters"
                )

            if note.note_name and len(note.note_name) > 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Note name exceeds maximum length of 200 characters"
                )

            await self.db.commit()
            await self.db.refresh(note)

            return {
                "id": note.id,
                "note_name": note.note_name,
                "user_id": note.user_id,
                "target_lang": note.target_lang,
                "note_type": note.note_type,
                "content": note.content,
                "tags": note.tags or [],
                "created_at": note.created_at,
                "updated_at": note.updated_at
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in UpdateNoteRepository: {str(e)}")
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update note"
            )



class DeleteNoteRepository:
    def __init__(self, db: AsyncSession, user_id: int, note_id: int):
        self.db = db
        self.user_id = user_id
        self.note_id = note_id

    async def delete_note(self):
        query = select(NoteModel).where(
            NoteModel.id == self.note_id,
            NoteModel.user_id == self.user_id
        )
        result = await self.db.execute(query)
        note = result.scalar_one_or_none()

        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )

        await self.db.delete(note)
        await self.db.commit()



class GetDailyStreakRepository:
    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id
        self.DAILY_WORD_LIMIT = 20

    # async def fetch_daily_streak(self) -> int:
    #     """
    #     Returns the current daily streak count.
    #     Example: 3 means user learned 20+ words for 3 consecutive days
    #     Returns 0 if no streak or today's goal not met yet
    #     """
    #     # Get all dates where user learned at least 20 words
    #     qualified_dates = await self._get_qualified_dates()
    #
    #     if not qualified_dates:
    #         return 0
    #
    #     # Calculate current streak
    #     current_streak = self._calculate_current_streak(qualified_dates)
    #
    #     return current_streak
    #
    # async def _get_qualified_dates(self):
    #     """Get all dates where user learned at least DAILY_WORD_LIMIT words"""
    #     # Since we need to count words per date, we can:
    #     # 1. Get all learned words for this user
    #     # 2. Group them by date and count
    #     # 3. Filter dates with count >= DAILY_WORD_LIMIT
    #
    #     from app.models import UserWord
    #
    #     # Get all learned words for this user
    #     query = select(UserWord).where(
    #         and_(
    #             UserWord.user_id == self.user_id,
    #             UserWord.is_learned == True
    #         )
    #     ).order_by(UserWord.created_at)
    #
    #     result = await self.db.execute(query)
    #     user_words = result.scalars().all()
    #
    #     # Group by date and count
    #     words_by_date = {}
    #     for user_word in user_words:
    #         word_date = user_word.created_at.date()
    #         words_by_date[word_date] = words_by_date.get(word_date, 0) + 1
    #
    #     # Filter dates that meet the daily goal
    #     qualified_dates = []
    #     for word_date, count in words_by_date.items():
    #         if count >= self.DAILY_WORD_LIMIT:
    #             qualified_dates.append(word_date)
    #
    #     # Sort dates (oldest to newest)
    #     qualified_dates.sort()
    #
    #     return qualified_dates
    #
    # def _calculate_current_streak(self, qualified_dates):
    #     """
    #     Calculate current consecutive streak.
    #     Rules:
    #     1. Check if today is qualified (learned 20+ words today)
    #     2. If yes, count backwards consecutive days
    #     3. If no, check if yesterday was qualified and count backwards from there
    #     """
    #     today = date.today()
    #     yesterday = today - timedelta(days=1)
    #
    #     # If today is qualified, count from today backwards
    #     if today in qualified_dates:
    #         streak = 1
    #         check_date = yesterday
    #
    #         # Count consecutive days backwards
    #         while check_date in qualified_dates:
    #             streak += 1
    #             check_date -= timedelta(days=1)
    #
    #         return streak
    #
    #     # If yesterday is qualified (but not today), count from yesterday backwards
    #     elif yesterday in qualified_dates:
    #         streak = 1
    #         check_date = yesterday - timedelta(days=1)
    #
    #         # Count consecutive days backwards from yesterday
    #         while check_date in qualified_dates:
    #             streak += 1
    #             check_date -= timedelta(days=1)
    #
    #         return streak
    #
    #     # No current streak (no qualified days in the last 2 days)
    #     return 0

    # Alternative: More efficient database approach
    async def fetch_daily_streak_optimized(self) -> int:
        """
        More efficient version using database aggregation.
        This assumes your database supports DATE() function.
        """

        # First, let's get today's count to know if we're in a streak
        today = date.today()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today, datetime.max.time())

        # Count today's learned words
        today_count_query = select(func.count(UserWord.id)).where(
            and_(
                UserWord.user_id == self.user_id,
                UserWord.is_learned == True,
                UserWord.created_at >= today_start,
                UserWord.created_at <= today_end
            )
        )

        today_result = await self.db.execute(today_count_query)
        today_count = today_result.scalar() or 0

        # If today doesn't meet goal, check yesterday
        if today_count < self.DAILY_WORD_LIMIT:
            yesterday = today - timedelta(days=1)
            yesterday_start = datetime.combine(yesterday, datetime.min.time())
            yesterday_end = datetime.combine(yesterday, datetime.max.time())

            yesterday_count_query = select(func.count(UserWord.id)).where(
                and_(
                    UserWord.user_id == self.user_id,
                    UserWord.is_learned == True,
                    UserWord.created_at >= yesterday_start,
                    UserWord.created_at <= yesterday_end
                )
            )

            yesterday_result = await self.db.execute(yesterday_count_query)
            yesterday_count = yesterday_result.scalar() or 0

            # If yesterday also doesn't meet goal, streak is 0
            if yesterday_count < self.DAILY_WORD_LIMIT:
                return 0

            # Start counting from yesterday
            start_date = yesterday
        else:
            # Start counting from today
            start_date = today

        # Now count consecutive days backwards from start_date
        streak = 0
        current_date = start_date

        while True:
            date_start = datetime.combine(current_date, datetime.min.time())
            date_end = datetime.combine(current_date, datetime.max.time())

            count_query = select(func.count(UserWord.id)).where(
                and_(
                    UserWord.user_id == self.user_id,
                    UserWord.is_learned == True,
                    UserWord.created_at >= date_start,
                    UserWord.created_at <= date_end
                )
            )

            count_result = await self.db.execute(count_query)
            date_count = count_result.scalar() or 0

            if date_count >= self.DAILY_WORD_LIMIT:
                streak += 1
                current_date -= timedelta(days=1)
            else:
                break

        return streak



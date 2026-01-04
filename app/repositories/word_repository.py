
import os
from collections import defaultdict
import re
from typing import List, Dict, Any, Optional, Dict
import json
import httpx
import aiohttp
import asyncio
from functools import lru_cache
import random
from datetime import datetime, timezone, timedelta, date
import hashlib
import time
from pydantic import ValidationError
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# New Added For GOOGLE
import base64
from google.cloud import texttospeech
from google.oauth2 import service_account
# Use aiohttp for better streaming support
import aiohttp
#############################################



from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from sqlalchemy import select, func, and_, update, or_, case, delete, text, distinct
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, outerjoin, joinedload

from app.logging_config import setup_logger
from app.models.word_model import Word, Sentence, SentenceWord, WordMeaning, Translation, SentenceTranslation, \
    LearnedWord, word_category_association, Category
from app.models.user_model import Language, UserModel, UserLanguage, UserWord, ConversationContextModel, DirectChatContextModel, NoteModel
from app.schemas.word_schema import GenerateAIWordSchema, TranslateSchema
from app.schemas.conversation_contexts_schema import GenerateAIChatSchema
from app.schemas.favorite_schemas import FavoriteWordBase, FavoriteCategoryBase, FavoriteCategoryResponse, FavoriteFetchWordResponse
from app.schemas.note_schemas import NoteBase, NoteCreate, NoteUpdate, NoteResponse

from app.models.user_model import FavoriteCategory, FavoriteWord, DefaultCategory

logger = setup_logger(__name__, "word.log")

from app.schemas.word_schema import AIWordResponse



class RandomIconColor:

    def __init__(self):

        self.CATEGORY_ICONS = [
            {'icon': 'book', 'color': '#6366F1'},
            {'icon': 'play', 'color': '#10B981'},
            {'icon': 'cube', 'color': '#F59E0B'},
            {'icon': 'chatbubbles', 'color': '#EF4444'},
            {'icon': 'school', 'color': '#8B5CF6'},
            {'icon': 'heart', 'color': '#EC4899'},
            {'icon': 'earth', 'color': '#06B6D4'},
            {'icon': 'time', 'color': '#F97316'},
            {'icon': 'star', 'color': '#EAB308'},
            {'icon': 'list', 'color': '#64748B'}
        ]

    def default_icon(self):
        return self.CATEGORY_ICONS[0]

    def get_random_icon(self):
        icons = [
            {'icon': 'book', 'color': '#6366F1'},
            {'icon': 'play', 'color': '#10B981'},
            {'icon': 'cube', 'color': '#F59E0B'},
            {'icon': 'chatbubbles', 'color': '#EF4444'},
            {'icon': 'school', 'color': '#8B5CF6'},
            {'icon': 'heart', 'color': '#EC4899'},
            {'icon': 'earth', 'color': '#06B6D4'},
            {'icon': 'time', 'color': '#F97316'},
            {'icon': 'star', 'color': '#EAB308'},
            {'icon': 'list', 'color': '#64748B'}
        ]
        return random.choice(icons)



class GetStatisticsForDashboardRepository:
    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def get_statistics(self):
        # First, get the user's selected languages
        user_languages_query = select(UserLanguage.target_language_code).where(
            UserLanguage.user_id == self.user_id
        )
        user_languages_result = await self.db.execute(user_languages_query)
        user_language_codes = [row[0] for row in user_languages_result.all()]

        # If user hasn't selected any languages, return empty array
        if not user_language_codes:
            return []

        query = (
            select(
                Word.language_code,
                func.count(Word.id).label("total_words"),
                func.count(case((UserWord.is_learned == True, 1))).label("learned_words"),
                func.count(case((UserWord.is_starred == True, 1))).label("starred_words"),
            )
            .select_from(Word)
            .outerjoin(
                UserWord,
                (UserWord.word_id == Word.id) & (UserWord.user_id == self.user_id)
            )
            .where(Word.language_code.in_(user_language_codes))  # ✅ Filter by user's languages
            .group_by(Word.language_code)
            .order_by(Word.language_code)
        )

        result = await self.db.execute(query)
        rows = result.mappings().all()

        lang_code_map = {
            "ru": "Russian",
            "en": "English",
            "tr": "Turkish",
            "es": "Spanish",
        }

        return_data = [
            {
                "language_name": lang_code_map.get(row["language_code"], row["language_code"]),
                "language_code": row["language_code"],
                "total_words": row["total_words"] or 0,
                "learned_words": row["learned_words"] or 0,
                "starred_words": row["starred_words"] or 0
            }
            for row in rows
        ]

        return return_data

        # return [
        #     {
        #         "language_code": lang_code_map.get(row["language_code"], row["language_code"]),
        #         "total_words": row["total_words"] or 0,
        #         "learned_words": row["learned_words"] or 0,
        #         "starred_words": row["starred_words"] or 0
        #     }
        #     for row in rows
        # ]



class FetchWordRepository:

    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def get_available_languages(self):
        """Return just the available languages with counts"""
        # Get user's target languages
        lang_result = await self.db.execute(
            select(UserLanguage.target_language_code).where(UserLanguage.user_id == self.user_id)
        )
        target_lang_codes = [row[0] for row in lang_result.fetchall()]
        if not target_lang_codes:
            return []

        # Get word counts for each language
        lang_data = []
        for lang_code in target_lang_codes:
            # Total words count
            total_count_stmt = select(func.count(Word.id)).where(Word.language_code == lang_code)
            total_count_result = await self.db.execute(total_count_stmt)
            total_count = total_count_result.scalar_one()

            # Learned words count
            learned_count_stmt = (
                select(func.count(Word.id))
                .join(UserWord, and_(
                    UserWord.word_id == Word.id,
                    UserWord.user_id == self.user_id,
                    UserWord.is_learned == True
                ))
                .where(Word.language_code == lang_code)
            )
            learned_count_result = await self.db.execute(learned_count_stmt)
            learned_count = learned_count_result.scalar_one() or 0

            lang_data.append({
                "lang": lang_code,
                "language_code": lang_code,  # Add this for consistency
                "total_words": total_count,
                "learned_words": learned_count,  # Add learned count
                "language_name": self._get_language_name(lang_code)
            })

        return lang_data

    async def fetch_words_for_language(self, lang_code: str, only_starred: bool = False,
                                       only_learned: bool = False, skip: int = 0, limit: int = 50) -> Dict[str, Any]:
        """Fetch words for a specific language with pagination support"""

        # 1. Get user's native language (your existing code)
        user_result = await self.db.execute(
            select(UserModel).where(UserModel.id == self.user_id)
        )
        user = user_result.scalar_one_or_none()
        if not user:
            return {"words": [], "total_count": 0, "has_more": False}

        native_language = user.native
        lang_code_map = {"Russian": "ru", "English": "en", "Spanish": "es", "Turkish": "tr"}
        native_code = lang_code_map.get(native_language)

        if not native_code:
            raise ValueError("User's native language not supported")

        # 2. Build base query for counting total
        base_stmt = (
            select(Word.id)
            .outerjoin(
                UserWord,
                and_(
                    UserWord.word_id == Word.id,
                    UserWord.user_id == self.user_id,
                ),
            )
            .where(Word.language_code == lang_code)
        )

        # Apply the same filters for counting
        if only_starred:
            base_stmt = base_stmt.where(UserWord.is_starred == True)
        elif only_learned:
            base_stmt = base_stmt.where(UserWord.is_learned == True)
        else:
            learned_or_starred_subq = (
                select(UserWord.word_id)
                .where(
                    UserWord.user_id == self.user_id,
                    or_(UserWord.is_learned == True, UserWord.is_starred == True),
                )
                .subquery()
            )
            base_stmt = base_stmt.where(Word.id.notin_(select(learned_or_starred_subq.c.word_id)))

        # Get total count
        total_count_stmt = select(func.count()).select_from(base_stmt.subquery())
        total_count_result = await self.db.execute(total_count_stmt)
        total_count = total_count_result.scalar_one()

        # 3. Build main query for fetching words (your existing query)
        stmt = (
            select(Word, WordMeaning, Translation, UserWord.is_starred, UserWord.is_learned)
            .outerjoin(WordMeaning, WordMeaning.word_id == Word.id)
            .outerjoin(
                Translation,
                and_(
                    Translation.source_word_id == Word.id,
                    Translation.target_language_code == native_code,
                ),
            )
            .outerjoin(
                UserWord,
                and_(
                    UserWord.word_id == Word.id,
                    UserWord.user_id == self.user_id,
                ),
            )
            .where(Word.language_code == lang_code)
        )

        # Apply filters (same as above)
        if only_starred:
            stmt = stmt.where(UserWord.is_starred == True)
        elif only_learned:
            stmt = stmt.where(UserWord.is_learned == True)
        else:
            learned_or_starred_subq = (
                select(UserWord.word_id)
                .where(
                    UserWord.user_id == self.user_id,
                    or_(UserWord.is_learned == True, UserWord.is_starred == True),
                )
                .subquery()
            )
            stmt = stmt.where(Word.id.notin_(select(learned_or_starred_subq.c.word_id)))

        # If learned screen, fetch order desc
        if only_learned:
            stmt = stmt.order_by(UserWord.updated_at.desc())

        # 4. Apply pagination
        stmt = stmt.offset(skip).limit(limit)


        # 5. Execute and process (your existing code)
        result = await self.db.execute(stmt)
        rows = result.all()

        # 6. Group by Word.id (your existing code)
        word_map = defaultdict(lambda: {
            "id": None,
            "text": None,
            "frequency_rank": None,
            "level": None,
            "pos": set(),
            "translations": set(),
            "language_code": lang_code,
            "is_starred": False,
            "is_learned": False,
        })

        for word, meaning, translation, is_starred, is_learned in rows:
            word_id = word.id
            if word_id not in word_map:
                word_map[word_id].update({
                    "id": word.id,
                    "text": word.text,
                    "frequency_rank": word.frequency_rank,
                    "level": word.level,
                })

            if meaning and meaning.pos:
                word_map[word_id]["pos"].add(meaning.pos)

            if translation and translation.translated_text:
                word_map[word_id]["translations"].add(translation.translated_text)

            if is_starred:
                word_map[word_id]["is_starred"] = True
            if is_learned:
                word_map[word_id]["is_learned"] = True

        # 7. Convert to list
        words_list = []
        for data in word_map.values():
            words_list.append({
                "id": data["id"],
                "text": data["text"],
                "frequency_rank": data["frequency_rank"],
                "level": data["level"],
                "pos": sorted(list(data["pos"])) if data["pos"] else [],
                "translation_to_native": list(data["translations"])[0] if data["translations"] else None,
                "language_code": lang_code,
                "is_starred": data["is_starred"],
                "is_learned": data["is_learned"],
            })

        page_type: str = ''
        if only_learned:
            page_type = 'learned'
        else:
            page_type = 'unlearned'

        # 8. Return with pagination info
        return {
            "page_type": page_type,
            "words": words_list,
            "total_count": total_count,
            "has_more": (skip + len(words_list)) < total_count,
            "skip": skip,
            "limit": limit
        }

    def _get_language_name(self, code: str) -> str:
        lang_map = {
            "ru": "Russian",
            "en": "English",
            "es": "Spanish",
            "tr": "Turkish"
        }
        return lang_map.get(code, code)



class ChangeWordStatusRepository:
    def __init__(self, db: AsyncSession, word_id: int, action: str, user_id: int):
        self.word_id = word_id
        self.action = action
        self.db = db
        self.user_id = user_id

    async def set_word_status(self):

        if self.action not in {"star", "learned"}:
            raise HTTPException(status_code=400, detail="Invalid action")

        result = await self.db.execute(
            select(UserWord).where(UserWord.user_id == self.user_id, UserWord.word_id == self.word_id)
        )
        user_word = result.scalar_one_or_none()

        if not user_word:
            user_word = UserWord(user_id=self.user_id, word_id=self.word_id)
            self.db.add(user_word)

        if self.action == "star":
            user_word.is_starred = not user_word.is_starred
        elif self.action == "learned":
            user_word.is_learned = not user_word.is_learned

        await self.db.commit()
        return {
            "word_id": self.word_id,
            "is_starred": user_word.is_starred,
            "is_learned": user_word.is_learned,
        }



class VoiceHandleRepository:

    def __init__(self):
        self.client = self._create_google_client()

    def _create_google_client(self):
        """Create Google TTS client using environment variables directly"""
        try:
            # Build credentials dictionary from environment variables
            credentials_info = {
                "type": "service_account",
                "project_id": os.getenv("GOOGLE_PROJECT_ID"),
                "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
                "private_key": os.getenv("GOOGLE_PRIVATE_KEY", "").replace('\\n', '\n'),
                "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "auth_uri": os.getenv("GOOGLE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": os.getenv("GOOGLE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_X509_CERT_URL",
                                                         "https://www.googleapis.com/oauth2/v1/certs"),
            }

            # Create credentials from service account info
            credentials = service_account.Credentials.from_service_account_info(credentials_info)

            # Create and return the TTS client
            return texttospeech.TextToSpeechClient(credentials=credentials)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Google TTS service configuration error"
            )

    async def generate_speech(self, text: str, lang: str) -> bytes:
        """
        Generate speech using Google Text-to-Speech API
        """
        try:
            # Map language codes to Google TTS voice names
            mapped_lang, voice_name = self._get_voice_for_language(lang)

            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Build the voice request - use the MAPPED language code
            voice = texttospeech.VoiceSelectionParams(
                language_code=mapped_lang,  # Use the mapped language code, not the original
                name=voice_name
            )

            # Select the type of audio file you want returned
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # The response's audio_content is binary
            return response.audio_content

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Google TTS service error: {str(e)}"
            )

    def _get_voice_for_language(self, lang: str) -> tuple[str, str]:
        """
        Map simple language codes to Google TTS language codes and voice names
        Returns: (language_code, voice_name)
        """
        # Map of simple codes to Google language codes and specific voices
        language_mapping = {
            "en": ("en-US", "en-US-Neural2-F"),
            "ru": ("ru-RU", "ru-RU-Wavenet-D"),
            "tr": ("tr-TR", "tr-TR-Wavenet-B"),
            "de": ("de-DE", "de-DE-Neural2-F"),
            "fr": ("fr-FR", "fr-FR-Neural2-A"),
            "es": ("es-ES", "es-ES-Neural2-F"),
            "it": ("it-IT", "it-IT-Neural2-A"),
            "ja": ("ja-JP", "ja-JP-Neural2-B"),
            "ko": ("ko-KR", "ko-KR-Neural2-A"),
            "zh": ("zh-CN", "zh-CN-Neural2-A"),
            "pt": ("pt-BR", "pt-BR-Neural2-F"),  # Added Portuguese
            "ar": ("ar-XA", "ar-XA-Wavenet-B"),  # Added Arabic
        }

        # If the language is already in full format (like "es-ES"), use it directly
        if '-' in lang and lang in language_mapping:
            return language_mapping[lang]
        elif '-' in lang:
            # If it's already a full code but not in our mapping, use it as-is with default voice
            return (lang, f"{lang}-Wavenet-A")
        elif lang in language_mapping:
            # If it's a simple code, return the mapped full code and voice
            return language_mapping[lang]
        else:
            # Fallback: try to create a reasonable default
            # For 2-letter codes, try to map to the most common region
            default_mapping = {
                "en": "en-US",
                "es": "es-ES",
                "pt": "pt-BR",
                "zh": "zh-CN",
                "ar": "ar-XA"
            }
            default_lang = default_mapping.get(lang, f"{lang}-{lang.upper()}")
            return (default_lang, f"{default_lang}-Wavenet-A")



class GenerateAIWordRepository:

    def __init__(self):
        self.headers = {
            "Authorization": f"Api-Key {os.getenv('YANDEX_LANGMODEL_API_SECRET_KEY')}",
            "Content-Type": "application/json"}

        self.model = 'yandexgpt'
        self.folder_id = os.getenv('YANDEX_FOLDER_ID')

    async def _call_yandex_gpt(self, prompt: str) -> Optional[str]:
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.3,  # Lower temperature for more factual responses
                "maxTokens": 2000
            },
            "messages": [
                {
                    "role": "system",
                    "text": "You are a helpful language learning assistant. Provide accurate, educational responses about words and phrases."
                },
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                return result['result']['alternatives'][0]['message']['text']
            except httpx.HTTPStatusError as e:
                return None
            except (httpx.RequestError, KeyError, json.JSONDecodeError) as e:
                return None

    @lru_cache(maxsize=100)
    def _create_prompt(self, word: str, target_lang: str, native_lang: str) -> str:
        return f"""**ROLE**: You are an expert {target_lang} language teacher creating learning materials for {native_lang} speakers.

            **TASK**: Create comprehensive educational content for the word "{word}" in {target_lang}.

            **RESPONSE FORMAT**: You MUST return ONLY valid JSON with EXACTLY this structure:
            {{
                "word": "{word}",
                "target_language": "{target_lang}",
                "native_language": "{native_lang}",
                "definition": "clear definition in {native_lang}",
                "pronunciation": "phonetic pronunciation guide",
                "part_of_speech": "main part of speech in {native_lang}",
                "examples": [
                    "example 1 in {target_lang} - {native_lang} translation",
                    "example 2 in {target_lang} - {native_lang} translation",
                    "example 3 in {target_lang} - {native_lang} translation",
                    "example 4 in {target_lang} - {native_lang} translation",
                    "example 5 in {target_lang} - {native_lang} translation"
                ],
                "usage_contexts": [
                    "context 1 in {native_lang}",
                    "context 2 in {native_lang}",
                    "context 3 in {native_lang}"
                ],
                "common_phrases": [
                    "phrase 1 in {target_lang} - {native_lang} translation",
                    "phrase 2 in {target_lang} - {native_lang} translation",
                    "phrase 3 in {target_lang} - {native_lang} translation"
                ],
                "grammar_tips": [
                    "tip 1 in {native_lang}",
                    "tip 2 in {native_lang}",
                    "tip 3 in {native_lang}"
                ],
                "additional_insights": {{
                    "key": "value"
                }}
                }}

    **CRITICAL INSTRUCTIONS**:
    1. **NATIVE LANGUAGE FIRST**: All explanations, definitions, tips, and notes MUST be in {native_lang}
    2. **TRANSLATION FORMAT**: Examples and phrases must use "Target Text - Native Translation" format
    3. **DIVERSE EXAMPLES**: Provide 5 varied examples showing different:
       - Tenses (if verb)
       - Cases (if noun/adjective)
       - Sentence structures
       - Formality levels
    4. **ACCURATE CONTENT**: Ensure all translations and explanations are 100% accurate
    5. **CULTURAL CONTEXT**: Include relevant cultural usage notes
    6. **GRAMMAR FOCUS**: Provide specific grammar tips for this word
    7. **NO MARKDOWN**: Return ONLY raw JSON, no code blocks or explanations
    8. **WORD SPECIFIC**: Tailor all content specifically to the word "{word}"

    **WORD-SPECIFIC GUIDANCE**:
    - If "{word}" is a verb: include conjugation patterns, aspect pairs, government patterns
    - If "{word}" is a noun: include gender, declension patterns, plural forms
    - If "{word}" is short/functional: explain nuanced usage and common collocations
    - If "{word}" is ambiguous: clarify different meanings with examples

    **EXAMPLE FORMATTING**:
    - "Я читаю книгу. - I am reading a book."
    - "Он прочитал книгу вчера. - He read the book yesterday."
    - "Эта книга интересная. - This book is interesting."

    **REMEMBER**: You are helping a {native_lang} speaker master {target_lang}. Quality and accuracy are paramount.
    """

    def _get_native_language_name(self, language_code: str) -> str:
        template = fallback_templates.get(native_lang, fallback_templates['en'])
        return template[field_type]

    async def generate_ai_for_word(self, data):
        print(f'Received request for word: {data.text}, target: {data.language}, native: {data.native}')
        prompt = self._create_prompt(data.text, data.language, data.native)
        ai_response = await self._call_yandex_gpt(prompt)
        if not ai_response:
            print("AI returned empty response")
            raise HTTPException(status_code=503, detail="AI service unavailable")

        try:
            cleaned_response = ai_response.strip()
            cleaned_response = re.sub(r'^```(?:json)?\s*', '', cleaned_response)
            cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
            cleaned_response = cleaned_response.strip()
            parsed_response = json.loads(cleaned_response)
            print("✅ Successfully parsed JSON!")

            try:
                return AIWordResponse(**parsed_response)
            except ValidationError as e:
                print(f"❌ Pydantic validation failed: {e}")
                print(f"Parsed data: {parsed_response}")
                return self._create_fallback_response(data)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode failed! Error: {str(e)}")
            print(f"Failed content: {ai_response}")
            print(f"Cleaned content: {cleaned_response}")  # Add this
            return self._create_fallback_response(data)
        except Exception as e:
            print(f"❌ Other error: {str(e)}")
            return self._create_fallback_response(data)

    def _create_fallback_response(self, data) -> AIWordResponse:
        """Create a fallback response in the user's native language"""
        return AIWordResponse(
            word=data.text,
            target_language=data.language,
            native_language=data.native,
            definition=self._get_fallback_text('definition', data.native, data.language, data.text),
            pronunciation=None,
            part_of_speech=self._get_fallback_text('part_of_speech', data.native, data.language, data.text),
            examples=[
                self._get_fallback_text('example', data.native, data.language, data.text),
                self._get_fallback_text('example', data.native, data.language, data.text),
                self._get_fallback_text('example', data.native, data.language, data.text),
                self._get_fallback_text('example', data.native, data.language, data.text),
                self._get_fallback_text('example', data.native, data.language, data.text)
            ],
            usage_contexts=[
                               self._get_fallback_text('usage', data.native, data.language, data.text),
                               self._get_fallback_text('usage', data.native, data.language, data.text),

                                     "Pay attention to sentence structure",
                "Practice with different contexts"
            ],
            additional_insights = None
        )



class SearchRepository:

    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def search(self, native_language: str, target_language: str, query: str):

        if not query or len(query.strip()) < 1:
            return {"results": []}

        search_query = f"%{query.strip().lower()}%"

        # Base query - search in words and their translations
        stmt = (
            select(Word)
            .join(Translation, Translation.source_word_id == Word.id)
            .where(
                # Filter by target language if not 'all'
                (Translation.target_language_code == native_language) if target_language != 'all' else True,
                # Filter by source language if target is specific language
                (Word.language_code == target_language) if target_language != 'all' else True,
                # Search in both source word and translated text
                or_(
                    func.lower(Word.text).ilike(search_query),
                    func.lower(Translation.translated_text).ilike(search_query)
                )
            )
            .options(
                selectinload(Word.user_words),  # Load user progress
                selectinload(Word.translations)  # Load all translations
            )
            .distinct()  # Avoid duplicates
            .limit(50)  # Limit results for performance
        )

        try:
            result = await self.db.execute(stmt)
            words = result.scalars().all()

            formatted_results = []
            for word in words:
                # Check user's learning status
                is_learned = False
                is_starred = False

                # Find user-specific data if it exists
                for user_word in word.user_words:
                    if user_word.user_id == self.user_id:
                        is_learned = user_word.is_learned
                        is_starred = user_word.is_starred
                        break

                # Find the translation for the user's native language
                native_translation = None
                for translation in word.translations:
                    if translation.target_language_code == native_language:
                        native_translation = translation.translated_text
                        break

                # If no specific translation found, use any available translation
                if not native_translation and word.translations:
                    native_translation = word.translations[0].translated_text

                formatted_results.append({
                    "id": word.id,
                    "text": word.text,
                    "language_code": word.language_code,
                    "translation_to_native": native_translation,
                    "is_learned": is_learned,
                    "is_starred": is_starred
                })

            return {"results": formatted_results}

        except Exception as e:
            print(f"Database error during search: {str(e)}")
            # Return empty results instead of crashing
            return {"results": []}



class DetailWordRepository:

    def __init__(self, db: AsyncSession, word_id: int, user_id: int):
        self.db = db
        self.word_id = word_id
        self.user_id = user_id


    async def get_word_detail(self):
        # 1. Fetch the word with relationships
        result = await self.db.execute(
            select(Word)
            .where(Word.id == self.word_id)
            .options(
                selectinload(Word.meanings).selectinload(WordMeaning.sentences).selectinload(Sentence.translations),
                selectinload(Word.translations).selectinload(Translation.target_language),
                selectinload(Word.in_sentences).selectinload(SentenceWord.sentence).selectinload(Sentence.translations)
            )
        )
        word: Word = result.scalar_one_or_none()
        if not word:
            return None

        # 2. Check if word is starred or learned
        user_word_result = await self.db.execute(
            select(UserWord)
            .where(UserWord.user_id == self.user_id, UserWord.word_id == self.word_id)
        )
        user_word: UserWord = user_word_result.scalar_one_or_none()

        is_starred = user_word.is_starred if user_word else False
        is_learned = user_word.is_learned if user_word else False

        # 3. Check for learned word strength
        learned_result = await self.db.execute(
            select(LearnedWord)
            .where(LearnedWord.user_id == self.user_id, LearnedWord.word_id == self.word_id)
        )
        learned_word: LearnedWord = learned_result.scalar_one_or_none()
        strength = learned_word.strength if learned_word else 0

        # 4. Structure meanings
        meanings = []
        for meaning in word.meanings:
            meaning_sentences = []
            for sentence in meaning.sentences:
                meaning_sentences.append({
                    "id": sentence.id,
                    "text": sentence.text,
                    "translations": [
                        {
                            "language_code": t.language_code,
                            "translated_text": t.translated_text
                        } for t in sentence.translations
                    ]
                })
            meanings.append({
                "id": meaning.id,
                "pos": meaning.pos,
                # "example": meaning.example,
                "sentences": meaning_sentences
            })


        # 5. Get user native language and filter by language
        native_lang: str = await self.get_user_native_lang_code(self.user_id)
        print('found languages: {}'.format(native_lang))

        # 6. Structure general example sentences
        general_sentences = []
        for sentence_link in word.in_sentences:
            sentence = sentence_link.sentence
            translations = []
            for translation in sentence.translations:
                if translation.language_code == native_lang:
                    translations.append({
                        "language_code": translation.language_code,
                        "translated_text": translation.translated_text
                    })
            general_sentences.append({
                "id": sentence.id,
                "text": sentence.text,
                "translations": translations
            })

        # 6. Structure translations
        translations = []
        for t in word.translations:
            if t.target_language_code == native_lang:
                translations.append({
                    "language_code": t.target_language_code,
                    "translated_text": t.translated_text
                })

        # 7. Build response
        return {
            "id": word.id,
            "text": word.text,
            "language_code": word.language_code,
            "frequency_rank": word.frequency_rank,
            "level": word.level,
            "is_starred": is_starred,
            "is_learned": is_learned,
            "strength": strength,
            "meanings": meanings,
            "translations": translations,
            "example_sentences": general_sentences
        }

    async def get_user_native_lang_code(self, user_id: int):
        # Retrieve the entire user object
        langs_code = {
            "English":"en",
            "Spanish":"es",
            "Russian":"ru",
            "Turkish":"tr",
        }
        user_result = await self.db.execute(
            select(UserModel)
            .where(UserModel.id == user_id)
        )
        user = user_result.scalar_one_or_none()

        # If a user is found, access the 'native_lang' attribute
        if user:
            native_lang = user.native
            if native_lang:
                return langs_code.get(native_lang, None)
            else:
                return None
        else:
            print(f'User with ID {user_id} not found.')
            return None



class GetPosStatisticsRepository:
    def __init__(self, db: AsyncSession, user_id: int, lang_code: str):
        self.db = db
        self.user_id = user_id
        self.lang_code = lang_code

    async def get_pos_statistics(self):
        try:
            # 1. Get total POS counts for all words in the language
            total_pos_query = (
                select(WordMeaning.pos, func.count(WordMeaning.id))
                .join(Word, Word.id == WordMeaning.word_id)
                .where(Word.language_code == self.lang_code)
                .group_by(WordMeaning.pos)
            )

            total_result = await self.db.execute(total_pos_query)
            total_rows = total_result.all()

            # 2. Get user's learned POS counts
            learned_pos_query = (
                select(WordMeaning.pos, func.count(WordMeaning.id))
                .select_from(UserWord)
                .join(Word, UserWord.word_id == Word.id)
                .join(WordMeaning, Word.id == WordMeaning.word_id)
                .where(
                    UserWord.user_id == self.user_id,
                    Word.language_code == self.lang_code,
                    UserWord.is_learned == True
                )
                .group_by(WordMeaning.pos)
            )

            learned_result = await self.db.execute(learned_pos_query)
            learned_rows = learned_result.all()

            # Convert to dictionaries
            total_stats = {}
            for pos, count in total_rows:
                if pos:
                    total_stats[pos] = count

            learned_stats = {}
            for pos, count in learned_rows:
                if pos:
                    learned_stats[pos] = count

            # Combine both statistics
            combined_stats = {}
            all_pos = set(total_stats.keys()) | set(learned_stats.keys())

            for pos in all_pos:
                combined_stats[pos] = {
                    'total': total_stats.get(pos, 0),
                    'learned': learned_stats.get(pos, 0)
                }

            return combined_stats

        except Exception as e:
            print(f"Error in get_pos_statistics: {str(e)}")
            raise



class FetchWordByPosRepository:
    def __init__(self, db, user_id: int, pos_name: str, lang_code: str,
                 only_starred: bool = False, only_learned: bool = False,
                 skip: int = 0, limit: int = 50):
        self.db = db
        self.user_id = user_id
        self.pos_name = pos_name
        self.lang_code = lang_code
        self.only_starred = only_starred
        self.only_learned = only_learned
        self.skip = skip
        self.limit = limit

    async def fetch_words_by_pos(self) -> List[Dict[Any, Any]]:
        """Fetch words for a specific part of speech"""

        # 1. Get user's native language
        user_result = await self.db.execute(
            select(UserModel).where(UserModel.id == self.user_id)
        )
        user = user_result.scalar_one_or_none()
        if not user:
            return []

        native_language = user.native
        lang_code_map = {"Russian": "ru", "English": "en", "Spanish": "es", "Turkish": "tr"}
        native_code = lang_code_map.get(native_language)

        if not native_code:
            raise ValueError("User's native language not supported")

        # ✅ FIX: Build counting query with EXACT SAME filters as main query
        counting_stmt = (
            select(Word.id)
            .join(WordMeaning, WordMeaning.word_id == Word.id)
            .outerjoin(
                UserWord,
                and_(
                    UserWord.word_id == Word.id,
                    UserWord.user_id == self.user_id,
                ),
            )
            .where(
                Word.language_code == self.lang_code,
                WordMeaning.pos == self.pos_name
            )
        )

        # ✅ FIX: Apply the same filters to counting query
        if self.only_starred:
            counting_stmt = counting_stmt.where(UserWord.is_starred == True)
        elif self.only_learned:
            counting_stmt = counting_stmt.where(UserWord.is_learned == True)
        else:
            # For WordScreen (unlearned words) - exclude learned/starred words
            learned_subq = (
                select(UserWord.word_id)
                .where(
                    UserWord.user_id == self.user_id,
                    UserWord.is_learned == True,  # Only learned
                )
                .subquery()
            )
            counting_stmt = counting_stmt.where(Word.id.notin_(select(learned_subq.c.word_id)))

        # Get total count of FILTERED words
        total_count_stmt = select(func.count()).select_from(counting_stmt.subquery())
        total_count_result = await self.db.execute(total_count_stmt)
        total_count = total_count_result.scalar_one()

        # Build main query - filter by POS
        stmt = (
            select(Word, WordMeaning, Translation, UserWord.is_starred, UserWord.is_learned)
            .join(WordMeaning, WordMeaning.word_id == Word.id)
            .outerjoin(
                Translation,
                and_(
                    Translation.source_word_id == Word.id,
                    Translation.target_language_code == native_code,
                ),
            )
            .outerjoin(
                UserWord,
                and_(
                    UserWord.word_id == Word.id,
                    UserWord.user_id == self.user_id,
                ),
            )
            .where(
                Word.language_code == self.lang_code,
                WordMeaning.pos == self.pos_name
            )
        )

        if self.only_starred:
            stmt = stmt.where(UserWord.is_starred == True)
        elif self.only_learned:
            stmt = stmt.where(UserWord.is_learned == True)
        else:
            # ✅ FIX: For WordScreen - exclude ONLY learned words (keep starred words)
            learned_subq = (
                select(UserWord.word_id)
                .where(
                    UserWord.user_id == self.user_id,
                    UserWord.is_learned == True,  # Only learned
                )
                .subquery()
            )
            stmt = stmt.where(Word.id.notin_(select(learned_subq.c.word_id)))

        if self.only_learned:
            stmt = stmt.order_by(UserWord.updated_at.desc())

        # Apply pagination
        stmt = stmt.offset(self.skip).limit(self.limit)

        # Execute
        result = await self.db.execute(stmt)
        rows = result.all()

        # Group by Word.id
        word_map = defaultdict(lambda: {
            "id": None,
            "text": None,
            "frequency_rank": None,
            "level": None,
            "pos": set(),
            "translations": set(),
            "language_code": self.lang_code,
            "is_starred": False,
            "is_learned": False,
        })

        for word, meaning, translation, is_starred, is_learned in rows:
            word_id = word.id
            if word_id not in word_map:
                word_map[word_id].update({
                    "id": word.id,
                    "text": word.text,
                    "frequency_rank": word.frequency_rank,
                    "level": word.level,
                })

            # Merge POS
            if meaning and meaning.pos:
                word_map[word_id]["pos"].add(meaning.pos)

            # Merge translations
            if translation and translation.translated_text:
                word_map[word_id]["translations"].add(translation.translated_text)

            # Aggregate user flags
            if is_starred:
                word_map[word_id]["is_starred"] = True
            if is_learned:
                word_map[word_id]["is_learned"] = True

        # Convert to list
        words_list = []
        for data in word_map.values():
            words_list.append({
                "id": data["id"],
                "text": data["text"],
                "frequency_rank": data["frequency_rank"],
                "level": data["level"],
                "pos": sorted(list(data["pos"])) if data["pos"] else [],
                "translation_to_native": list(data["translations"])[0] if data["translations"] else None,
                "language_code": self.lang_code,
                "is_starred": data["is_starred"],
                "is_learned": data["is_learned"],
            })

        # Calculate has_more based on ACTUAL filtered total_count
        current_loaded = self.skip + len(words_list)
        has_more = current_loaded < total_count and len(words_list) > 0

        page_type: str = ''
        if self.only_learned:
            page_type = 'learned'
        else:
            page_type = 'unlearned'

        return_data = {
            "page_type": page_type,
            "words": words_list,
            "total_count": total_count,  # This will be the correct count now
            "has_more": has_more,
            "skip": self.skip,
            "limit": self.limit
        }

        return return_data



class TranslateRepository:
    def __init__(self):
        self.api_key = os.getenv("YANDEX_TRANSLATE_API_SECRET_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.translate_url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        self.detect_url = "https://translate.api.cloud.yandex.net/translate/v2/detect"

    def _preprocess_text(self, text: str) -> str:
        """Clean and validate input text before sending to API"""
        if not text or not text.strip():
            raise ValueError("Empty text provided for translation")

        # Remove excessive whitespace but ensure at least one space if empty after trim
        text = ' '.join(text.strip().split())
        return text or " "  # Return single space if empty after processing

    async def translate(self, data: TranslateSchema):
        processed_text = data.text
        # Early return for empty text after preprocessing
        if not processed_text or len(processed_text) < 1:
            return {
                "translation": processed_text,
                "detected_lang": data.from_lang if data.from_lang != "auto" else None
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }

        payload = {
            "targetLanguageCode": data.to_lang,
            "texts": [processed_text],
            "folderId": self.folder_id,
        }

        if data.from_lang and data.from_lang.strip().lower() != "auto":
            payload["sourceLanguageCode"] = data.from_lang

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

                print(f'1 after translation coming resuilt is {response.json()}')

                result = response.json()["translations"][0]
                print(f'after translation coming resuilt is {result}')
                return {
                    "translation": result["text"],
                    "detected_lang": result.get("detectedLanguageCode")
                }

        except httpx.RequestError as e:
            # This covers network errors, timeouts, etc.
            logger.error(f"Network error contacting Yandex API: {e}")
            raise HTTPException(
                status_code=503,
                detail="Translation service temporarily unavailable"
            )

        except httpx.HTTPStatusError as e:
            # Already handled well, but be more specific
            if e.response.status_code == 401:
                logger.error("Yandex API authentication failed")
                raise HTTPException(status_code=500, detail="Service configuration error")


        except Exception as ex:
            logger.exception(f"Unexpected error during translation: {ex}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal Server Error"
            )



class AddFavoritesRepository:

    def __init__(self, data: FavoriteWordBase, db: AsyncSession, user_id: int):
        self.data = data
        self.db = db
        self.user_id = user_id
        self.random_item_color = RandomIconColor()


    async def _get_or_create_default_category(self) -> int:
        """Get or create a default 'Uncategorized' category for the user"""
        try:
            # Check if default category already exists
            stmt = select(FavoriteCategory).where(
                FavoriteCategory.user_id == self.user_id,
                FavoriteCategory.name == "Default"
            )
            result = await self.db.execute(stmt)
            category = result.scalar_one_or_none()

            if category:
                return category.id

            icon_data = self.random_item_color.default_icon()

            # Create new default category
            new_category = FavoriteCategory(
                user_id=self.user_id,
                name="Default",
                icon=icon_data['icon'],
                color=icon_data['color']
            )
            self.db.add(new_category)
            await self.db.commit()
            await self.db.refresh(new_category)

            return new_category.id

        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Error creating default category: {str(e)}")
            print(f"2Error creating default category: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create default category"
            )

    async def _check_existing_favorite(self) -> bool:
        """Check if this word already exists in user's favorites"""
        try:
            stmt = select(FavoriteWord).where(
                FavoriteWord.user_id == self.user_id,
                FavoriteWord.from_lang == self.data.from_lang,
                FavoriteWord.to_lang == self.data.to_lang,
                FavoriteWord.original_text == self.data.original_text
            )
            result = await self.db.execute(stmt)
            existing_word = result.scalar_one_or_none()

            return existing_word is not None

        except SQLAlchemyError as e:
            logger.error(f"Error checking existing favorite: {str(e)}")
            print(f'error happen 1 - {e}')
            return False

    async def add_favorites(self) -> dict:
        try:

            # 1. Check if word already exists
            exists = await self._check_existing_favorite()
            if exists:
                return {
                    "status": "success",
                    "message": "Word already in favorites",
                    "action": "existing"
                }

            # 2. Handle category
            category_id = self.data.category_id

            # If no category provided, use default "Uncategorized"
            if not category_id:
                category_id = await self._get_or_create_default_category()

            # 3. Add new favorite word
            new_favorite = FavoriteWord(
                user_id=self.user_id,
                category_id=category_id,
                from_lang=self.data.from_lang,
                to_lang=self.data.to_lang,
                original_text=self.data.original_text,
                translated_text=self.data.translated_text,
                added_at=datetime.utcnow()
            )

            self.db.add(new_favorite)
            await self.db.commit()
            await self.db.refresh(new_favorite)

            return {
                "status": "success",
                "message": "Word added to favorites",
                "action": "created",
                "favorite_id": new_favorite.id
            }

        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Database error adding favorite: {str(e)}")
            print(f"Database error adding favorite: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while adding favorite"
            )
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Unexpected error adding favorite: {str(e)}")
            print(f"Unexpected error adding favorite: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected error occurred"
            )



class CreateNewFavoriteCategoryRepository:
    def __init__(self, db: AsyncSession, data: FavoriteCategoryBase, user_id: int):
        self.db = db
        self.data = data
        self.user_id = user_id
        self.random_item_color = RandomIconColor()

    async def _check_duplicate_category(self) -> bool:
        """Check if category with same name already exists for this user"""
        try:
            stmt = select(FavoriteCategory).where(
                FavoriteCategory.user_id == self.user_id,
                FavoriteCategory.name.ilike(self.data.name.strip())
            )
            result = await self.db.execute(stmt)
            existing_category = result.scalar_one_or_none()
            return existing_category is not None
        except SQLAlchemyError as e:
            logger.error(f"Error checking duplicate category: {str(e)}")
            return False

    async def create_new_category(self) -> dict:
        try:
            # Check for duplicate category name
            if await self._check_duplicate_category():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Category '{self.data.name}' already exists"
                )

            icon_data = self.random_item_color.get_random_icon()

            # Create new category
            new_category = FavoriteCategory(
                user_id=self.user_id,
                name=self.data.name.strip(),
                icon=icon_data['icon'],
                color=icon_data['color']
            )

            self.db.add(new_category)
            await self.db.commit()
            await self.db.refresh(new_category)

            return {
                "status": "success",
                "message": "Category created successfully",
                "category": {
                    "id": new_category.id,
                    "name": new_category.name,
                }
            }

        except IntegrityError as e:
            await self.db.rollback()
            logger.error(f"Integrity error creating category: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid category data"
            )
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Database error creating category: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while creating category"
            )
        except HTTPException:
            raise
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Unexpected error creating category: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected error occurred"
            )



class FavoriteCategoryRepository:
    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def get_user_categories(self) -> List[FavoriteCategoryResponse]:
        try:
            # Get categories with word counts using a join
            stmt = (select(
                FavoriteCategory,
                func.count(FavoriteWord.id).label('word_count')
            ).outerjoin(
                FavoriteWord, FavoriteWord.category_id == FavoriteCategory.id
            ).where(
                FavoriteCategory.user_id == self.user_id
            )
            .group_by(FavoriteCategory.id))

            result = await self.db.execute(stmt)
            categories_with_counts = result.all()

            # Convert to response format
            categories_response = []
            for category, word_count in categories_with_counts:
                categories_response.append({
                    "id": category.id,
                    "name": category.name,
                    "user_id": category.user_id,
                    "color": category.color,
                    "icon": category.icon,
                    "word_count": word_count,
                })

            return categories_response

        except SQLAlchemyError as e:
            logger.error(f"Database error fetching categories: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Database error while fetching categories"
            )



class CategoryWordsRepository:
    def __init__(self, db: AsyncSession, user_id: int, category_id: int):
        self.db = db
        self.user_id = user_id
        self.category_id = category_id

    async def get_category_words(self) -> dict:
        try:
            # First, verify the category belongs to the user and get its name
            category_stmt = select(FavoriteCategory).where(
                FavoriteCategory.id == self.category_id,
                FavoriteCategory.user_id == self.user_id
            )

            category_result = await self.db.execute(category_stmt)
            category = category_result.scalar_one_or_none()

            if not category:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Category not found"
                )

            # Get all words in this category
            words_stmt = select(FavoriteWord).where(
                FavoriteWord.category_id == self.category_id,
                FavoriteWord.user_id == self.user_id
            ).order_by(FavoriteWord.added_at.desc())

            words_result = await self.db.execute(words_stmt)
            words = words_result.scalars().all()

            # Convert to response format
            words_response = []
            for word in words:
                words_response.append({
                    "id": word.id,
                    "original_text": word.original_text,
                    "translated_text": word.translated_text,
                    "from_lang": word.from_lang,
                    "to_lang": word.to_lang,
                    "category_id": word.category_id,
                    "added_at": word.added_at,
                })

            return_data = {
                "category_id": category.id,
                "category_name": category.name,
                "word_count": len(words),
                "words": words_response
            }

            return return_data

        except SQLAlchemyError as e:
            logger.error(f"Database error fetching category words: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while fetching category words"
            )



class DeleteFavoriteWordRepository:
    def __init__(self, db: AsyncSession, user_id: int, word_id: int):
        self.db = db
        self.user_id = user_id
        self.word_id = word_id

    async def delete_word(self) -> dict:
        try:
            # First verify the word belongs to the user
            stmt = select(FavoriteWord).where(
                FavoriteWord.id == self.word_id,
                FavoriteWord.user_id == self.user_id
            )
            result = await self.db.execute(stmt)
            word = result.scalar_one_or_none()

            if not word:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Word not found in your favorites"
                )

            # Delete the word
            delete_stmt = delete(FavoriteWord).where(
                FavoriteWord.id == self.word_id,
                FavoriteWord.user_id == self.user_id
            )
            await self.db.execute(delete_stmt)
            await self.db.commit()

            return {
                "status": "success",
                "message": "Word removed from favorites",
                "deleted_word_id": self.word_id
            }

        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Database error deleting word: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while deleting word"
            )



class MoveFavoriteWordRepository:

    def __init__(self, db: AsyncSession, user_id: int, word_id: int):
        self.db = db
        self.user_id = user_id
        self.word_id = word_id

    # repository.py - Add to FavoriteWordRepository class
    async def move_word(self, target_category_id: int) -> dict:
        try:
            # Verify the word exists and belongs to the user
            stmt = select(FavoriteWord).where(
                FavoriteWord.id == self.word_id,
                FavoriteWord.user_id == self.user_id
            )
            result = await self.db.execute(stmt)
            word = result.scalar_one_or_none()

            if not word:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Word not found in your favorites"
                )
            if word.category_id == target_category_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Word is already in this category"
                )

            # Verify target category exists and belongs to the user
            category_stmt = select(FavoriteCategory).where(
                FavoriteCategory.id == target_category_id,
                FavoriteCategory.user_id == self.user_id
            )
            category_result = await self.db.execute(category_stmt)
            target_category = category_result.scalar_one_or_none()

            if not target_category:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Target category not found"
                )

            # Store old category ID for response
            old_category_id = word.category_id

            # Update the word's category
            word.category_id = target_category_id
            word.added_at = datetime.utcnow()  # Update timestamp

            self.db.add(word)
            await self.db.commit()
            await self.db.refresh(word)

            return {
                "status": "success",
                "message": "Word moved successfully",
                "word_id": self.word_id,
                "old_category_id": old_category_id,
                "new_category_id": target_category_id
            }

        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Database error moving word: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while moving word"
            )



class DeleteCategoryRepository:
    def __init__(self, db: AsyncSession, user_id: int, category_id: int):
        self.db = db
        self.user_id = user_id
        self.category_id = category_id

    async def _get_default_category_id(self) -> int:
        """Get or create default category for the user"""
        stmt = select(FavoriteCategory).where(
            FavoriteCategory.user_id == self.user_id,
            FavoriteCategory.name == "Default"
        )
        result = await self.db.execute(stmt)
        category = result.scalar_one_or_none()

        if category:
            return category.id

        # Create default category if it doesn't exist
        new_category = FavoriteCategory(
            user_id=self.user_id,
            name="Default",
            description="Default category for words"
        )
        self.db.add(new_category)
        await self.db.commit()
        await self.db.refresh(new_category)
        return new_category.id

    async def delete_category(self) -> dict:
        try:
            # Verify category exists and belongs to the user
            stmt = select(FavoriteCategory).where(
                FavoriteCategory.id == self.category_id,
                FavoriteCategory.user_id == self.user_id
            )
            result = await self.db.execute(stmt)
            category = result.scalar_one_or_none()

            if not category:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Category not found"
                )

            # Prevent deletion of default category
            if category.name == "Default":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete the default category"
                )

            # # Get default category for moving words
            # default_category_id = await self._get_default_category_id()
            #
            # # Move all words to default category
            # update_stmt = update(FavoriteWord).where(
            #     FavoriteWord.category_id == self.category_id,
            #     FavoriteWord.user_id == self.user_id
            # ).values(category_id=default_category_id)
            #
            # await self.db.execute(update_stmt)

            # Delete the category
            delete_stmt = delete(FavoriteCategory).where(
                FavoriteCategory.id == self.category_id,
                FavoriteCategory.user_id == self.user_id
            )
            await self.db.execute(delete_stmt)

            await self.db.commit()

            return {
                "status": "success",
                "message": "Category deleted successfully",
                "deleted_category_id": self.category_id,
            }

        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Database error deleting category: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while deleting category"
            )



class SearchFavoriteRepository:

    def __init__(self, db: AsyncSession, user_id: int, query: str, category_id: Optional[int] = None):
        self.db = db
        self.user_id = user_id
        self.query = query
        self.category_id = category_id

    async def search_words(self) -> List[dict]:
        try:
            # Base query with JOIN to get category name
            stmt = select(
                FavoriteWord,
                FavoriteCategory.name.label("category_name")  # Select category name
            ).join(
                FavoriteCategory,  # Join with categories table
                FavoriteWord.category_id == FavoriteCategory.id,
                isouter=True  # Use left outer join in case category is null
            ).where(
                FavoriteWord.user_id == self.user_id,
                or_(
                    FavoriteWord.original_text.ilike(f"%{self.query}%"),
                    FavoriteWord.translated_text.ilike(f"%{self.query}%")
                )
            ).order_by(FavoriteWord.added_at.desc())

            # Add category filter if provided
            if self.category_id:
                stmt = stmt.where(FavoriteWord.category_id == self.category_id)

            # Execute query
            result = await self.db.execute(stmt)
            rows = result.all()

            # Transform results to include category name
            words_with_category = []
            for word, category_name in rows:
                word_dict = {
                    "id": word.id,
                    "user_id": word.user_id,
                    "category_id": word.category_id,
                    "from_lang": word.from_lang,
                    "to_lang": word.to_lang,
                    "original_text": word.original_text,
                    "translated_text": word.translated_text,
                    "added_at": word.added_at,
                    "category_name": category_name  # Add category name here
                }
                words_with_category.append(word_dict)

            return words_with_category

        except SQLAlchemyError as e:
            logger.error(f"Database search error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error during search"
            )



class FetchStatisticsForProfileRepository:

    def __init__(self, db: AsyncSession, user_id: Optional[int] = None):
        self.db = db
        self.user_id = user_id

    async def fetch_statistics(self) -> Dict[str, Any]:
        """
        Fetch user statistics using a single query with join
        """
        if not self.user_id:
            raise HTTPException(status_code=400, detail="User ID is required")

        # Single query to get user info and count learned words
        query = select(
            UserModel.username,
            UserModel.email,
            UserModel.created_at,
            UserModel.is_premium,
            func.count(UserWord.id).label('total_learned_words')
        ).select_from(UserModel).outerjoin(
            UserWord,
            (UserWord.user_id == UserModel.id) & (UserWord.is_learned == True)
        ).where(
            UserModel.id == self.user_id
        ).group_by(
            UserModel.id,
            UserModel.username,
            UserModel.email,
            UserModel.created_at,
            UserModel.is_premium
        )

        result = await self.db.execute(query)
        user_data = result.first()

        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        # Calculate days since registration
        days_registered = await self._calculate_days_registered(user_data.created_at)

        # Prepare response data
        statistics = {
            "username": user_data.username,
            "email": user_data.email,
            "total_learned_words": user_data.total_learned_words or 0,
            "days_registered": days_registered,
            "join_date": user_data.created_at.isoformat() if user_data.created_at else None,
            "is_premium": user_data.is_premium
        }


        return statistics

    async def _calculate_days_registered(self, created_at: datetime) -> int:
        """Calculate how many days the user has been registered"""
        if not created_at:
            return 0

        # Ensure both datetimes are timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        current_time = datetime.now(timezone.utc)

        # Calculate difference in days
        time_difference = current_time - created_at
        days_registered = time_difference.days

        # Ensure at least 1 day for same-day registration
        return max(1, days_registered)



class DailyStreakRepository:
    def __init__(self, db: AsyncSession, user_id: Optional[int] = None):
        self.db = db
        self.user_id = user_id

    async def daily_streak(self) -> Dict[str, Any]:
        """
        Fetch daily streak statistics:
        - last_learned_language: The language_code of the most recently learned word
        - daily_learned_words: Number of words learned TODAY
        """
        if not self.user_id:
            raise HTTPException(status_code=400, detail="User ID is required")

        # Get today's date
        today = datetime.now(timezone.utc).date()
        start_of_today = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
        end_of_today = start_of_today + timedelta(days=1)

        # Get last learned language (using updated_at for accurate ordering)
        last_learned_language = await self._get_last_learned_language()

        # Get today's learned words count
        daily_learned_words = await self._get_today_learned_words(today)

        return_data = {
            "last_learned_language": last_learned_language,
            "daily_learned_words": daily_learned_words,
            "date": today.isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        return return_data

    async def _get_last_learned_language(self) -> Optional[str]:
        """Get the language_code of the most recently learned word using updated_at"""
        try:
            # Debug: Show recent learning activity
            debug_query = (
                select(UserWord.id, UserWord.created_at, UserWord.updated_at,
                       UserWord.is_learned, Word.language_code, Word.text)
                .select_from(UserWord)
                .join(Word, UserWord.word_id == Word.id)
                .where(UserWord.user_id == self.user_id)
                .order_by(UserWord.updated_at.desc())
                .limit(10)
            )

            debug_result = await self.db.execute(debug_query)
            recent_activity = debug_result.all()

            # print("=== DEBUG: Recent UserWord activity (ordered by updated_at) ===")
            for uw_id, created_at, updated_at, is_learned, lang_code, text in recent_activity:
                status = "LEARNED" if is_learned else "UNLEARNED"
                # print(f"  - ID: {uw_id}, Status: {status}, Lang: {lang_code}")
                # print(f"     Created: {created_at}, Updated: {updated_at}")
                # print(f"     Word: {text}")
            # print("=============================================================")

            # Main query using updated_at for most recent learning activity
            query = (
                select(Word.language_code)
                .select_from(UserWord)
                .join(Word, UserWord.word_id == Word.id)
                .where(
                    and_(
                        UserWord.user_id == self.user_id,
                        UserWord.is_learned == True
                    )
                )
                .order_by(UserWord.updated_at.desc())  # Use updated_at for accurate ordering
                .limit(1)
            )

            result = await self.db.execute(query)
            last_language = result.scalar_one_or_none()

            # print(f"Last learned word language_code: {last_language}")
            return last_language

        except Exception as e:
            logger.error(f"Error getting last learned language: {str(e)}")
            print(f"Error in _get_last_learned_language: {str(e)}")
            return None

    async def _get_today_learned_words(self, today: date) -> int:
        """Get count of words learned TODAY (based on when they were marked as learned)"""
        try:
            start_of_today = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
            end_of_today = start_of_today + timedelta(days=1)

            # Count words that were marked as learned today (using updated_at)
            query = (
                select(func.count(UserWord.id))
                .where(
                    and_(
                        UserWord.user_id == self.user_id,
                        UserWord.is_learned == True,
                        UserWord.updated_at >= start_of_today,  # Use updated_at instead of created_at
                        UserWord.updated_at < end_of_today
                    )
                )
            )

            result = await self.db.execute(query)
            count = result.scalar() or 0

            return count

        except Exception as e:
            logger.error(f"Error getting today's learned words: {str(e)}")
            return 0



class FetchWordCategoriesRepository:

    def __init__(self, db, user_id: int, lang_code: str):
        self.db = db
        self.user_id = user_id
        self.lang_code = lang_code

    async def fetch_words_categories(self):
        if not self.lang_code:
            raise HTTPException(status_code=400, detail="Language code is required")

        # Subquery for total words per category
        total_words_subquery = (
            select(
                word_category_association.c.category_id,
                func.count(Word.id).label('total_words')
            )
            .select_from(word_category_association)
            .join(Word, word_category_association.c.word_id == Word.id)
            .where(Word.language_code == self.lang_code)
            .group_by(word_category_association.c.category_id)
            .subquery()
        )

        # Subquery for learned words per category
        learned_words_subquery = (
            select(
                word_category_association.c.category_id,
                func.count(Word.id).label('learned_words')
            )
            .select_from(word_category_association)
            .join(Word, word_category_association.c.word_id == Word.id)
            .join(UserWord, and_(
                UserWord.word_id == Word.id,
                UserWord.user_id == self.user_id,
                UserWord.is_learned == True
            ))
            .where(Word.language_code == self.lang_code)
            .group_by(word_category_association.c.category_id)
            .subquery()
        )

        # Main query
        query = (
            select(
                Category.id,
                Category.name,
                func.coalesce(total_words_subquery.c.total_words, 0).label('total_words'),
                func.coalesce(learned_words_subquery.c.learned_words, 0).label('learned_words')
            )
            .select_from(Category)
            .outerjoin(
                total_words_subquery,
                Category.id == total_words_subquery.c.category_id
            )
            .outerjoin(
                learned_words_subquery,
                Category.id == learned_words_subquery.c.category_id
            )
            .where(total_words_subquery.c.total_words > 0)  # Only categories with words
            .order_by(Category.name)
        )

        result = await self.db.execute(query)
        categories_data = result.all()

        return_data = [
            {
                "id": category_id,
                "name": category_name,
                "total_words": total_words,
                "learned_words": learned_words,
                "progress_percentage": round((learned_words / total_words) * 100) if total_words > 0 else 0
            }
            for category_id, category_name, total_words, learned_words in categories_data
        ]

        return return_data



class FetchWordByCategoryIdRepository:
    def __init__(self, db, user_id: int, category_id: int, lang_code: str,
                 only_starred: bool = False, only_learned: bool = False,
                 skip: int = 0, limit: int = 50):
        self.db = db
        self.user_id = user_id
        self.category_id = category_id
        self.lang_code = lang_code
        self.only_starred = only_starred
        self.only_learned = only_learned
        self.skip = skip
        self.limit = limit

    async def fetch_words_by_category_id(self) -> List[Dict[Any, Any]]:
        """Fetch words for a specific category — following same standard as fetch_words_for_language"""

        # 1. Get user's native language
        user_result = await self.db.execute(
            select(UserModel).where(UserModel.id == self.user_id)
        )
        user = user_result.scalar_one_or_none()
        if not user:
            return []

        native_language = user.native
        lang_code_map = {"Russian": "ru", "English": "en", "Spanish": "es", "Turkish": "tr"}
        native_code = lang_code_map.get(native_language)

        if not native_code:
            raise ValueError("User's native language not supported")

        # ✅ FIX: Build counting query with EXACT SAME filters as main query
        counting_stmt = (
            select(Word.id)
            .join(word_category_association, Word.id == word_category_association.c.word_id)
            .outerjoin(
                UserWord,
                and_(
                    UserWord.word_id == Word.id,
                    UserWord.user_id == self.user_id,
                ),
            )
            .where(
                Word.language_code == self.lang_code,
                word_category_association.c.category_id == self.category_id
            )
        )

        # ✅ FIX: Apply the SAME filters to counting query
        if self.only_starred:
            counting_stmt = counting_stmt.where(UserWord.is_starred == True)
        elif self.only_learned:
            counting_stmt = counting_stmt.where(UserWord.is_learned == True)
        else:
            # For unlearned words (WordScreen) - exclude learned/starred words
            learned_or_starred_subq = (
                select(UserWord.word_id)
                .where(
                    UserWord.user_id == self.user_id,
                    or_(UserWord.is_learned == True, UserWord.is_starred == True),
                )
                .subquery()
            )
            counting_stmt = counting_stmt.where(Word.id.notin_(select(learned_or_starred_subq.c.word_id)))

        # Get total count of FILTERED words
        total_count_stmt = select(func.count()).select_from(counting_stmt.subquery())
        total_count_result = await self.db.execute(total_count_stmt)
        total_count = total_count_result.scalar_one()

        # Build main query (your existing query with filters)
        stmt = (
            select(Word, WordMeaning, Translation, UserWord.is_starred, UserWord.is_learned)
            .join(word_category_association, Word.id == word_category_association.c.word_id)
            .outerjoin(WordMeaning, WordMeaning.word_id == Word.id)
            .outerjoin(
                Translation,
                and_(
                    Translation.source_word_id == Word.id,
                    Translation.target_language_code == native_code,
                ),
            )
            .outerjoin(
                UserWord,
                and_(
                    UserWord.word_id == Word.id,
                    UserWord.user_id == self.user_id,
                ),
            )
            .where(
                Word.language_code == self.lang_code,
                word_category_association.c.category_id == self.category_id
            )
        )

        # Apply filters
        if self.only_starred:
            stmt = stmt.where(UserWord.is_starred == True)
        elif self.only_learned:
            stmt = stmt.where(UserWord.is_learned == True)
        else:
            learned_or_starred_subq = (
                select(UserWord.word_id)
                .where(
                    UserWord.user_id == self.user_id,
                    or_(UserWord.is_learned == True, UserWord.is_starred == True),
                )
                .subquery()
            )
            stmt = stmt.where(Word.id.notin_(select(learned_or_starred_subq.c.word_id)))

        if self.only_learned:
            stmt = stmt.order_by(UserWord.updated_at.desc())

        # Apply pagination
        stmt = stmt.offset(self.skip).limit(self.limit)

        # Execute
        result = await self.db.execute(stmt)
        rows = result.all()

        # Group by Word.id
        word_map = defaultdict(lambda: {
            "id": None,
            "text": None,
            "frequency_rank": None,
            "level": None,
            "pos": set(),
            "translations": set(),
            "language_code": self.lang_code,
            "is_starred": False,
            "is_learned": False,
        })

        for word, meaning, translation, is_starred, is_learned in rows:
            word_id = word.id
            if word_id not in word_map:
                word_map[word_id].update({
                    "id": word.id,
                    "text": word.text,
                    "frequency_rank": word.frequency_rank,
                    "level": word.level,
                })

            # Merge POS
            if meaning and meaning.pos:
                word_map[word_id]["pos"].add(meaning.pos)

            # Merge translations
            if translation and translation.translated_text:
                word_map[word_id]["translations"].add(translation.translated_text)

            # Aggregate user flags
            if is_starred:
                word_map[word_id]["is_starred"] = True
            if is_learned:
                word_map[word_id]["is_learned"] = True

        # Convert to list
        words_list = []
        for data in word_map.values():
            words_list.append({
                "id": data["id"],
                "text": data["text"],
                "frequency_rank": data["frequency_rank"],
                "level": data["level"],
                "pos": sorted(list(data["pos"])) if data["pos"] else [],
                "translation_to_native": list(data["translations"])[0] if data["translations"] else None,
                "language_code": self.lang_code,
                "is_starred": data["is_starred"],
                "is_learned": data["is_learned"],
            })

        # ✅ FIX: Calculate has_more based on ACTUAL filtered total_count
        current_loaded = self.skip + len(words_list)
        has_more = current_loaded < total_count and len(words_list) > 0

        page_type: str = ''
        if self.only_learned:
            page_type = 'learned'
        else:
            page_type = 'unlearned'

        result = {
            "page_type": page_type,
            "words": words_list,
            "total_count": total_count,  # This will now be 21, not 25
            "has_more": has_more,
            "skip": self.skip,
            "limit": self.limit
        }

        print(f'🔍 Category Debug: category_id={self.category_id}, total_words={total_count}, has_more={has_more}')

        return result



class ConversationContextRepository:
    """Repository for managing conversation context in database"""

    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def _generate_context_hash(user_id: int, word: str, language: str) -> str:
        """
        Generate a unique hash for the conversation context.
        This ensures we can quickly find context for a specific user-word-language combination.
        """
        # Create a consistent string representation
        context_string = f"{user_id}|{word.lower().strip()}|{language.lower().strip()}"

        # Generate SHA256 hash
        return hashlib.sha256(context_string.encode('utf-8')).hexdigest()

    async def deactivate_other_contexts(
            self,
            user_id: int,
            keep_word: str,
            keep_language: str
    ) -> int:
        """
        Deactivate all other contexts for this user except the current word.
        This ensures only one active context per user at a time.

        Returns: Number of contexts deactivated
        """
        try:
            from sqlalchemy import update

            # Generate hash for the context we want to keep
            keep_hash = self._generate_context_hash(user_id, keep_word, keep_language)

            # Deactivate all other ACTIVE contexts for this user
            stmt = (
                update(ConversationContextModel)
                .where(ConversationContextModel.user_id == user_id)
                .where(ConversationContextModel.context_hash != keep_hash)
                .where(ConversationContextModel.is_active == True)  # Only deactivate active ones
                .values(
                    is_active=False,
                    updated_at=datetime.now(timezone.utc)  # Update timestamp
                )
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            rows_affected = result.rowcount
            if rows_affected > 0:
                logger.info(f"Deactivated {rows_affected} old contexts for user {user_id}")

            return rows_affected

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deactivating old contexts: {str(e)}")
            return 0

    async def get_context(
            self,
            user_id: int,
            word: str,
            language: str,
            active_only: bool = True  # Default to only active contexts
    ) -> Optional[ConversationContextModel]:
        """
        Get conversation context for a user and word.

        Args:
            user_id: User ID
            word: The word
            language: The language
            active_only: If True, only return active contexts
        """
        try:
            # Generate hash for lookup
            context_hash = self._generate_context_hash(user_id, word, language)

            from sqlalchemy import select
            stmt = (
                select(ConversationContextModel)
                .where(ConversationContextModel.context_hash == context_hash)
            )

            # if active_only:
            #     stmt = stmt.where(ConversationContextModel.is_active == True)

            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if context:
                status = "active" if context.is_active else "inactive"
                logger.debug(f"Found {status} context for user {user_id}, word '{word}'")

            return context

        except Exception as e:
            logger.error(f"Error getting context for user {user_id}, word {word}: {str(e)}")
            return None


    # In ConversationContextRepository class - Update get_or_create_context method

    async def get_or_create_context(
            self,
            user_id: int,
            word: str,
            language: str,
            native_language: str,
            initial_message: Optional[Dict[str, Any]] = None
    ) -> ConversationContextModel:
        """
        Get existing context or create a new one if it doesn't exist.
        When creating/reactivating a context, deactivate all other contexts for this user.
        This ensures only one active context per user at a time.
        """
        try:
            # FIRST: Check if ANY context exists for this word (active or inactive)
            context_hash = self._generate_context_hash(user_id, word, language)

            from sqlalchemy import select
            stmt = (
                select(ConversationContextModel)
                .where(ConversationContextModel.context_hash == context_hash)
            )

            result = await self.db.execute(stmt)
            existing_context = result.scalar_one_or_none()

            if existing_context:
                # Context exists (could be active or inactive)
                logger.debug(
                    f"Found existing context for user {user_id}, word '{word}' (active={existing_context.is_active})")

                # Update native language if changed
                if existing_context.native_language != native_language:
                    existing_context.native_language = native_language

                # If context is inactive, we need to reactivate it and deactivate others
                if not existing_context.is_active:
                    logger.info(f"Reactivating inactive context for user {user_id}, word '{word}'")

                    # Deactivate all other contexts for this user
                    await self.deactivate_other_contexts(user_id, word, language)

                    # Reactivate this context
                    existing_context.is_active = True
                    existing_context.updated_at = datetime.now(timezone.utc)

                # Update timestamp
                existing_context.updated_at = datetime.now(timezone.utc)

                await self.db.commit()
                await self.db.refresh(existing_context)

                return existing_context

            # No context exists at all - create a new one

            # Deactivate all other contexts for this user
            await self.deactivate_other_contexts(user_id, word, language)

            # Prepare initial messages
            messages_list = []
            if initial_message:
                messages_list.append(initial_message)

            # Create new context (will be automatically active)
            new_context = ConversationContextModel(
                user_id=user_id,
                word=word,
                language=language,
                native_language=native_language,
                context_hash=context_hash,
                messages=json.dumps(messages_list, ensure_ascii=False),
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

            self.db.add(new_context)
            await self.db.commit()
            await self.db.refresh(new_context)

            logger.info(f"Created new ACTIVE context for user {user_id}, word '{word}' (deactivated old contexts)")
            return new_context

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error in get_or_create_context for user {user_id}: {str(e)}")
            raise

    async def update_context_messages(
            self,
            context_id: int,
            messages: List[Dict[str, Any]],
            max_messages: int = 20
    ) -> Optional[ConversationContextModel]:
        """
        Update the messages in a conversation context.
        Automatically limits the number of messages to prevent overflow.
        """
        try:
            from sqlalchemy import select

            # Limit messages to prevent token overflow
            if len(messages) > max_messages:
                messages = messages[-max_messages:]
                logger.debug(f"Trimmed messages to last {max_messages} entries")

            # Get the context
            stmt = select(ConversationContextModel).where(
                ConversationContextModel.id == context_id,
                ConversationContextModel.is_active == True  # Only update active contexts
            )

            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if not context:
                logger.warning(f"Context {context_id} not found or inactive")
                return None

            # Update messages and timestamp
            context.messages = json.dumps(messages, ensure_ascii=False)
            context.updated_at = datetime.now(timezone.utc)

            await self.db.commit()
            await self.db.refresh(context)

            logger.debug(f"Updated context {context_id} with {len(messages)} messages")
            return context

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating context {context_id}: {str(e)}")
            return None

    async def add_message_to_context(
            self,
            context_id: int,
            role: str,  # "user" or "assistant"
            content: str
    ) -> Optional[ConversationContextModel]:
        """
        Add a single message to an existing context.
        """
        try:
            from sqlalchemy import select

            # Get current context (only if active)
            stmt = select(ConversationContextModel).where(
                ConversationContextModel.id == context_id,
                ConversationContextModel.is_active == True
            )

            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if not context:
                logger.warning(f"Context {context_id} not found or inactive")
                return None

            # Parse existing messages
            try:
                messages = json.loads(context.messages)
            except json.JSONDecodeError:
                messages = []

            # Add new message
            new_message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            messages.append(new_message)

            # Limit messages (keep last 20)
            if len(messages) > 20:
                messages = messages[-20:]

            # Update context
            context.messages = json.dumps(messages, ensure_ascii=False)
            context.updated_at = datetime.now(timezone.utc)

            await self.db.commit()
            await self.db.refresh(context)

            logger.debug(f"Added {role} message to active context {context_id}")
            return context

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error adding message to context {context_id}: {str(e)}")
            return None

    async def clear_context(
            self,
            user_id: int,
            word: str,
            language: str
    ) -> bool:
        """
        Clear context for a specific word (mark as inactive).
        """
        try:
            from sqlalchemy import update

            context_hash = self._generate_context_hash(user_id, word, language)

            stmt = (
                update(ConversationContextModel)
                .where(ConversationContextModel.context_hash == context_hash)
                .values(
                    is_active=False,  # Mark as inactive instead of clearing messages
                    updated_at=datetime.now(timezone.utc)
                )
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            rows_affected = result.rowcount
            if rows_affected > 0:
                logger.info(f"Deactivated context for user {user_id}, word '{word}'")
                return True
            else:
                logger.debug(f"No active context to deactivate for user {user_id}, word '{word}'")
                return False

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error clearing context: {str(e)}")
            return False

    async def delete_context(
            self,
            user_id: int,
            word: str,
            language: str
    ) -> bool:
        """
        Permanently delete context (not just mark as inactive).
        Use this when you want to completely remove a context.
        """
        try:
            from sqlalchemy import delete

            context_hash = self._generate_context_hash(user_id, word, language)

            stmt = (
                delete(ConversationContextModel)
                .where(ConversationContextModel.context_hash == context_hash)
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            rows_affected = result.rowcount
            if rows_affected > 0:
                logger.info(f"Permanently deleted context for user {user_id}, word '{word}'")

            return rows_affected > 0

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting context: {str(e)}")
            return False

    async def get_user_active_context(self, user_id: int) -> Optional[ConversationContextModel]:
        """
        Get the current active context for a user.
        Returns None if no active context exists.
        """
        try:
            from sqlalchemy import select

            stmt = (
                select(ConversationContextModel)
                .where(ConversationContextModel.user_id == user_id)
                .where(ConversationContextModel.is_active == True)
                .order_by(ConversationContextModel.updated_at.desc())
            )

            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if context:
                logger.debug(f"Found active context for user {user_id}: word '{context.word}'")

            return context

        except Exception as e:
            logger.error(f"Error getting active context for user {user_id}: {str(e)}")
            return None

    async def get_user_contexts(
            self,
            user_id: int,
            active_only: bool = True,
            limit: int = 50
    ) -> List[ConversationContextModel]:
        """
        Get all contexts for a user.

        Args:
            user_id: User ID
            active_only: If True, only return active contexts
            limit: Maximum number of contexts to return
        """
        try:
            from sqlalchemy import select

            stmt = (
                select(ConversationContextModel)
                .where(ConversationContextModel.user_id == user_id)
            )

            if active_only:
                stmt = stmt.where(ConversationContextModel.is_active == True)

            stmt = stmt.order_by(ConversationContextModel.updated_at.desc()).limit(limit)

            result = await self.db.execute(stmt)
            contexts = result.scalars().all()

            logger.debug(f"Found {len(contexts)} contexts for user {user_id} (active_only={active_only})")
            return list(contexts)

        except Exception as e:
            logger.error(f"Error getting contexts for user {user_id}: {str(e)}")
            return []

    async def cleanup_old_inactive_contexts(
            self,
            days_old: int = 1  # Delete inactive contexts older than 1 day (adjust as needed)
    ) -> int:
        """
        Permanently delete inactive contexts older than specified days.
        This should be run as a periodic task.

        Returns: Number of contexts deleted
        """
        try:
            from sqlalchemy import delete
            from datetime import timedelta

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)

            stmt = (
                delete(ConversationContextModel)
                .where(ConversationContextModel.is_active == False)
                .where(ConversationContextModel.updated_at < cutoff_date)
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            rows_affected = result.rowcount
            if rows_affected > 0:
                logger.info(f"Cleaned up {rows_affected} old inactive contexts (older than {days_old} days)")

            return rows_affected

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error cleaning up inactive contexts: {str(e)}")
            return 0



class GenerateAIQuestionRepository:
    """
    Main repository for AI chat with conversation context support.
    Now includes memory of previous conversations per word.
    """

    def __init__(self, db: AsyncSession = None):
        """
        Initialize with optional database session for context persistence.
        If no db is provided, context will not be saved (backward compatibility).
        """
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.max_tokens = 6000
        self.max_context_messages = 10  # Keep last 10 messages for context
        self.db = db

        # Initialize context repository if db is provided
        if db:
            self.context_repo = ConversationContextRepository(db)
        else:
            self.context_repo = None

    @staticmethod
    def _generate_context_hash(user_id: int, word: str, language: str) -> str:
        """Generate hash for context lookup"""
        context_string = f"{user_id}|{word.lower().strip()}|{language.lower().strip()}"
        return hashlib.sha256(context_string.encode('utf-8')).hexdigest()

    def _get_system_prompt(self, word: str, language: str, native: str) -> str:
        """
        Generate system prompt with word context.
        This helps the AI remember it's discussing a specific word.
        """
        return (
            f"You are a helpful, precise, and enthusiastic language learning assistant. "
            f"The user is learning the {language} word '{word}'. "
            f"Their native language is {native}. Your answer must be in {native}."
            f"Answer the user's question specifically about this word. "
            f"Be concise, pedagogical, and provide clear examples. "
            f"Your answer must be in {native} to ensure the user understands. "
            f"Focus on explaining usage, grammar, nuances, or cultural context related to '{word}'. "
            f"\n\nIMPORTANT: Remember we're discussing the word '{word}' in {language}. "
            f"Keep all explanations focused on this word unless the user explicitly asks about something else."
        )

    async def _prepare_conversation_messages(
            self,
            user_id: int,
            word: str,
            language: str,
            native: str,
            message: str
    ) -> tuple[List[Dict[str, str]], Optional[ConversationContextModel]]:
        """
        Prepare messages for AI with context from database.
        Returns: (messages_list, context_object_or_none)
        """
        messages = []
        context = None

        # Add system prompt
        system_prompt = self._get_system_prompt(word, language, native)
        messages.append({"role": "system", "content": system_prompt})

        # Try to load existing context if db is available
        if self.context_repo and user_id:
            try:
                # Get or create context for this user/word combination
                context = await self.context_repo.get_or_create_context(
                    user_id=user_id,
                    word=word,
                    language=language,
                    native_language=native
                )

                # Load previous messages from context
                if context and context.messages:
                    try:
                        previous_messages = json.loads(context.messages)

                        # Add previous conversation (excluding any system messages)
                        for msg in previous_messages:
                            if msg.get('role') != 'system':  # Don't include system messages from history
                                messages.append({
                                    "role": msg.get('role', 'user'),
                                    "content": msg.get('content', '')
                                })

                        logger.debug(f"Loaded {len(previous_messages)} previous messages from context")

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse context messages: {str(e)}")
                        # Start fresh if messages are corrupted
                        context = None

            except Exception as e:
                logger.error(f"Error loading context: {str(e)}")
                # Continue without context if there's an error

        # Add current user message
        messages.append({
            "role": "user",
            "content": message
        })

        return messages, context

    async def _save_conversation_to_context(
            self,
            context: Optional[ConversationContextModel],
            user_id: int,
            word: str,
            language: str,
            native: str,
            user_message: str,
            ai_response: str
    ) -> None:
        """Save the conversation to database context"""
        if not self.context_repo or not context:
            return

        try:
            # Load existing messages
            existing_messages = []
            if context.messages:
                try:
                    existing_messages = json.loads(context.messages)
                except json.JSONDecodeError:
                    existing_messages = []

            # Add the new exchange (user message + AI response)
            exchange = [
                {
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]

            # Combine and limit messages
            all_messages = existing_messages + exchange
            if len(all_messages) > self.max_context_messages * 2:  # *2 because each exchange has 2 messages
                # Keep only the most recent exchanges
                all_messages = all_messages[-(self.max_context_messages * 2):]

            # Update context
            await self.context_repo.update_context_messages(
                context_id=context.id,
                messages=all_messages
            )

            logger.debug(f"Saved conversation to context {context.id}")

        except Exception as e:
            logger.error(f"Error saving conversation to context: {str(e)}")

    async def _call_deepseek_api(self, messages: list) -> str:
        """Generic method to call DeepSeek API."""
        logger.info('[_call_deepseek_api] Method called. Preparing payload...')

        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.deepseek_api_key}"
        }

        async with aiohttp.ClientSession() as session:
            try:
                logger.debug('[DEBUG] Making request to DeepSeek API...')
                async with session.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    # Get the response text for debugging
                    response_text = await response.text()
                    logger.debug(f'[DEBUG] DeepSeek API Response Status: {response.status}')

                    # Check for errors
                    response.raise_for_status()

                    # Parse JSON response
                    data = await response.json()
                    logger.debug(f'[DEBUG] Parsed JSON Response keys: {data.keys()}')

                    # Extract the response text from DeepSeek format
                    return data['choices'][0]['message']['content']

            except aiohttp.ClientResponseError as e:
                logger.error(f"DeepSeek API error: {e.status} - {e.message}")
                raise HTTPException(
                    status_code=502,
                    detail=f"AI service error: {e.status}. Please check the request parameters."
                )
            except aiohttp.ClientConnectorError as e:
                logger.error(f"Connection to DeepSeek API failed: {str(e)}")
                raise HTTPException(status_code=503, detail="Cannot connect to AI service. Check your network.")
            except asyncio.TimeoutError:
                logger.error("Request to DeepSeek API timed out.")
                raise HTTPException(status_code=504, detail="AI service request timed out.")
            except (aiohttp.ClientError, KeyError) as e:
                logger.error(f"Unexpected error during DeepSeek API call: {str(e)}")
                raise HTTPException(status_code=500, detail="An unexpected error occurred with the AI service.")

    async def generate_ai_chat_stream(self, user_id: int, data):
        """
        Streaming version of AI chat with context support.
        Now remembers previous conversations about the same word.

        Args:
            user_id: The authenticated user's ID
            data: GenerateAIChatSchema object (without user_id)
        """
        try:
            # Prepare messages with context
            messages, context = await self._prepare_conversation_messages(
                user_id=user_id,
                word=data.word,
                language=data.language,
                native=data.native,
                message=data.message
            )

            # Store messages for saving later
            full_ai_response = ""

            # Stream the response
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(
                        self.api_url,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.deepseek_api_key}"
                        },
                        json={
                            "model": "deepseek-chat",
                            "messages": messages,
                            "temperature": 0.2,
                            "max_tokens": self.max_tokens,
                            "stream": True
                        }
                ) as response:

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error: {response.status} - {error_text}")
                        yield f"data: {json.dumps({'error': f'AI service error: {response.status}'})}\n\n"
                        return

                    async for line in response.content:
                        line = line.decode('utf-8').strip()

                        if line.startswith('data: '):
                            data_line = line[6:]

                            if data_line.strip() == '[DONE]':
                                # Save conversation to context after streaming is complete
                                if full_ai_response and context:
                                    await self._save_conversation_to_context(
                                        context=context,
                                        user_id=user_id,
                                        word=data.word,
                                        language=data.language,
                                        native=data.native,
                                        user_message=data.message,
                                        ai_response=full_ai_response
                                    )

                                yield f"data: {json.dumps({'done': True})}\n\n"
                                return

                            try:
                                chunk_data = json.loads(data_line)
                                if 'choices' in chunk_data and chunk_data['choices']:
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')

                                    if content:
                                        full_ai_response += content
                                        yield f"data: {json.dumps({'content': content})}\n\n"
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': f'Streaming failed: {str(e)}'})}\n\n"

    async def clear_word_context(
            self,
            user_id: int,
            word: str,
            language: str
    ) -> bool:
        """
        Clear context for a specific word.
        Call this when user selects a new word or wants to start fresh.
        """
        if not self.context_repo:
            logger.warning("Cannot clear context: No database session")
            return False

        try:
            success = await self.context_repo.clear_context(
                user_id=user_id,
                word=word,
                language=language
            )

            if success:
                logger.info(f"Cleared context for user {user_id}, word '{word}'")
            else:
                logger.debug(f"No context to clear for user {user_id}, word '{word}'")

            return success

        except Exception as e:
            logger.error(f"Error clearing context: {str(e)}")
            return False

    async def get_conversation_history(
            self,
            user_id: int,
            word: str,
            language: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get conversation history for a specific word.
        Useful for displaying previous conversation in UI.
        """
        if not self.context_repo:
            return None

        try:
            context = await self.context_repo.get_context(
                user_id=user_id,
                word=word,
                language=language
            )

            if context and context.messages:
                return json.loads(context.messages)

            return []

        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return None



class DirectChatContextRepository:
    """Repository for managing direct chat context in database"""

    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def _generate_context_hash(user_id: int) -> str:
        """Generate hash for direct chat context - ONE PER USER"""
        context_string = f"direct_chat_{user_id}"
        return hashlib.sha256(context_string.encode('utf-8')).hexdigest()

    async def get_or_create_context(self, user_id: int) -> DirectChatContextModel:
        """
        Get existing direct chat context or create a new one.
        Each user has exactly ONE direct chat context.
        """
        try:
            from sqlalchemy import select

            # Generate hash - same for each user
            context_hash = self._generate_context_hash(user_id)

            # Try to get existing context
            stmt = select(DirectChatContextModel).where(
                DirectChatContextModel.context_hash == context_hash
            )

            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                # Always ensure it's active (should always be true)
                if not existing.is_active:
                    existing.is_active = True
                    await self.db.commit()
                    await self.db.refresh(existing)

                logger.debug(f"Found existing direct chat context for user {user_id}")
                return existing

            # Create new context (first time for this user)
            new_context = DirectChatContextModel(
                user_id=user_id,
                topic="language_learning",  # Default topic
                context_hash=context_hash,
                messages="[]",
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

            self.db.add(new_context)
            await self.db.commit()
            await self.db.refresh(new_context)

            logger.info(f"Created new direct chat context for user {user_id}")
            return new_context

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error getting/creating direct chat context for user {user_id}: {str(e)}")
            raise

    async def add_message_to_context(
            self,
            context_id: int,
            role: str,
            content: str,
            max_messages: int = 20  # Limit conversation history
    ) -> Optional[DirectChatContextModel]:
        """Add a message to direct chat context"""
        try:
            from sqlalchemy import select

            stmt = select(DirectChatContextModel).where(
                DirectChatContextModel.id == context_id
            )
            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if not context:
                return None

            # Parse existing messages
            try:
                messages = json.loads(context.messages)
            except json.JSONDecodeError:
                messages = []

            # Add new message
            messages.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Keep only last N messages (prevent infinite growth)
            if len(messages) > max_messages:
                # Remove oldest messages, keep system message if present
                if messages and messages[0].get('role') == 'system':
                    # Keep system message + recent messages
                    system_message = messages[0]
                    recent_messages = messages[-(max_messages - 1):] if max_messages > 1 else []
                    messages = [system_message] + recent_messages
                else:
                    messages = messages[-max_messages:]

                logger.debug(f"Trimmed direct chat messages to {len(messages)} messages")

            # Update context
            context.messages = json.dumps(messages, ensure_ascii=False)
            context.updated_at = datetime.now(timezone.utc)

            await self.db.commit()
            await self.db.refresh(context)

            logger.debug(f"Added message to direct chat context {context_id}")
            return context

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error adding message to direct chat context: {str(e)}")
            return None

    async def clear_context_messages(self, user_id: int) -> bool:
        """
        Clear all messages from user's direct chat context.
        This is used when user wants to start fresh.
        """
        try:
            from sqlalchemy import update

            context_hash = self._generate_context_hash(user_id)

            stmt = (
                update(DirectChatContextModel)
                .where(DirectChatContextModel.context_hash == context_hash)
                .values(
                    messages="[]",
                    updated_at=datetime.now(timezone.utc)
                )
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            rows_affected = result.rowcount
            if rows_affected > 0:
                logger.info(f"Cleared direct chat messages for user {user_id}")
                return True

            return False

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error clearing direct chat messages: {str(e)}")
            return False

    async def delete_context(self, user_id: int) -> bool:
        """
        Permanently delete user's direct chat context.
        Use this only when deleting user account.
        """
        try:
            from sqlalchemy import delete

            context_hash = self._generate_context_hash(user_id)

            stmt = (
                delete(DirectChatContextModel)
                .where(DirectChatContextModel.context_hash == context_hash)
            )

            result = await self.db.execute(stmt)
            await self.db.commit()

            rows_affected = result.rowcount
            if rows_affected > 0:
                logger.info(f"Permanently deleted direct chat context for user {user_id}")

            return rows_affected > 0

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting direct chat context: {str(e)}")
            return False

    # REMOVE the cleanup_old_direct_contexts method - we don't need it!
    # Each user has ONE context that stays forever

    async def get_user_stats(self, user_id: int) -> dict:
        """Get statistics about user's direct chat context"""
        try:
            from sqlalchemy import select, func
            import json

            context_hash = self._generate_context_hash(user_id)

            stmt = select(DirectChatContextModel).where(
                DirectChatContextModel.context_hash == context_hash
            )

            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if not context:
                return {
                    "has_context": False,
                    "message_count": 0,
                    "last_updated": None,
                    "created_at": None
                }

            # Count messages
            message_count = 0
            if context.messages:
                try:
                    messages = json.loads(context.messages)
                    message_count = len(messages)
                except:
                    pass

            return {
                "has_context": True,
                "context_id": context.id,
                "message_count": message_count,
                "last_updated": context.updated_at.isoformat() if context.updated_at else None,
                "created_at": context.created_at.isoformat() if context.created_at else None,
                "is_active": context.is_active
            }

        except Exception as e:
            logger.error(f"Error getting user direct chat stats: {str(e)}")
            return {"error": str(e)}



class GenerateDirectAIChat:
    """AI chat for general language learning conversations with context"""

    def __init__(self, db: AsyncSession = None):
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.db = db

        # Initialize context repository if db is provided
        if db:
            self.context_repo = DirectChatContextRepository(db)
        else:
            self.context_repo = None

        # System prompt
        self.system_prompt = """You are an AI language learning tutor. Your role is strictly limited to helping users learn languages.

CORE RESPONSIBILITIES:
- Answer questions about vocabulary, grammar, pronunciation, and language usage
- Provide language practice exercises and conversations
- Explain cultural aspects related to languages
- Help with translation and language comprehension
- Create learning activities and study plans

STRICT BOUNDARIES:
- ONLY discuss language learning topics
- If asked about unrelated topics (cars, sports, politics, etc.), politely redirect to language learning
- Do not provide information outside of language education
- Maintain a professional, educational tone

RESPONSE GUIDELINES:
- Be encouraging and supportive
- Provide clear, structured explanations
- Include practical examples when possible
- Adapt to the user's native language for better understanding
- Keep responses focused and educational"""

    # word_repository.py - Update GenerateDirectAIChat._prepare_messages_with_context

    async def _prepare_messages_with_context(
            self,
            user_id: int,
            native_language: str,
            user_message: str
    ) -> tuple[List[Dict[str, str]], Optional[DirectChatContextModel]]:
        """Prepare messages with conversation context"""
        messages = []
        context = None

        # Enhanced system prompt with native language
        enhanced_system_prompt = f"{self.system_prompt}\n\nThe user's native language is {native_language}. Please provide explanations in a way that's easy for them to understand."

        # Always include system message
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # Load conversation context if available
        if self.context_repo and user_id:
            try:
                # Get or create user's direct chat context (always returns one)
                context = await self.context_repo.get_or_create_context(user_id)

                if context and context.messages:
                    try:
                        previous_messages = json.loads(context.messages)

                        # Add previous conversation (excluding any system messages)
                        for msg in previous_messages:
                            if msg.get('role') != 'system':  # Don't duplicate system messages
                                messages.append({
                                    "role": msg.get('role', 'user'),
                                    "content": msg.get('content', '')
                                })

                        logger.debug(f"Loaded {len(previous_messages)} previous messages from direct chat")

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse context messages: {str(e)}")
                        # If messages are corrupted, clear them
                        if context:
                            context.messages = "[]"
                            await self.db.commit()

            except Exception as e:
                logger.error(f"Error loading direct chat context: {str(e)}")

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        return messages, context

    async def _save_conversation_to_context(
            self,
            context: Optional[DirectChatContextModel],
            user_id: int,
            user_message: str,
            ai_response: str
    ) -> None:
        """Save conversation to context"""
        if not self.context_repo or not context:
            return

        try:
            # Load existing messages
            existing_messages = []
            if context.messages:
                try:
                    existing_messages = json.loads(context.messages)
                except json.JSONDecodeError:
                    existing_messages = []

            # Add new exchange
            exchange = [
                {
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]

            # Combine and limit messages
            all_messages = existing_messages + exchange
            if len(all_messages) > 20:  # Keep last 10 exchanges
                all_messages = all_messages[-20:]


            # Update context
            context.messages = json.dumps(all_messages, ensure_ascii=False)
            context.updated_at = datetime.now(timezone.utc)

            await self.db.commit()
            logger.debug(f"Saved direct chat conversation to context {context.id}")

        except Exception as e:
            logger.error(f"Error saving direct chat context: {str(e)}")

    async def ai_direct_chat_stream(self, data):
        """Streaming version of AI chat with context support"""
        try:
            # Get user_id from data (assuming it's added by the endpoint)
            user_id = getattr(data, 'user_id', None)

            # Prepare messages with context
            messages, context = await self._prepare_messages_with_context(
                user_id=user_id,
                native_language=data.native_language,
                user_message=data.message
            )

            full_ai_response = ""

            # Track if we've sent anything
            has_sent_content = False

            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=300)) as session:  # Increased to 5 minutes
                async with session.post(
                        self.api_url,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.deepseek_api_key}"
                        },
                        json={
                            "model": "deepseek-chat",
                            "messages": messages,
                            "temperature": 0.7,
                            "max_tokens": 6000,  # Increase max tokens for longer responses
                            "stream": True
                        }
                ) as response:

                    if response.status != 200:
                        error_text = await response.text()
                        yield f"data: {json.dumps({'error': f'AI service error: {response.status}'})}\n\n"
                        return

                    async for line in response.content:
                        line = line.decode('utf-8').strip()

                        if line.startswith('data: '):
                            data_line = line[6:]

                            if data_line.strip() == '[DONE]':
                                # Save conversation to context
                                if full_ai_response and context:
                                    await self._save_conversation_to_context(
                                        context=context,
                                        user_id=user_id,
                                        user_message=data.message,
                                        ai_response=full_ai_response
                                    )

                                yield f"data: {json.dumps({'done': True})}\n\n"
                                return

                            try:
                                chunk_data = json.loads(data_line)
                                if 'choices' in chunk_data and chunk_data['choices']:
                                    delta = chunk_data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')

                                    if content:
                                        full_ai_response += content
                                        has_sent_content = True
                                        # Send content chunk
                                        yield f"data: {json.dumps({'content': content})}\n\n"

                                        # Flush the buffer periodically for long responses
                                        if len(full_ai_response) % 1000 == 0:
                                            await asyncio.sleep(0.001)  # Small yield
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logger.error(f"Error processing chunk: {str(e)}")
                                continue

                    # If we never got DONE but have content, send completion
                    if has_sent_content:
                        yield f"data: {json.dumps({'done': True})}\n\n"

        except asyncio.CancelledError:
            logger.info("Direct chat stream was cancelled by client")
            raise
        except Exception as e:
            logger.error(f"Direct chat streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': f'Streaming failed: {str(e)}'})}\n\n"



class FetchDirectAiChatContext:
    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def ai_direct_fetch_context(self):
        """Fetch the direct chat context for the user if it exists."""
        try:
            # Get the context for this user
            stmt = select(DirectChatContextModel).where(
                DirectChatContextModel.user_id == self.user_id
            )

            result = await self.db.execute(stmt)
            context = result.scalar_one_or_none()

            if context:
                # Parse messages from JSON string
                messages = json.loads(context.messages) if context.messages else []
                return {
                    "id": context.id,
                    "topic": context.topic,
                    "messages": messages,
                    "created_at": context.created_at.isoformat() if context.created_at else None,
                    "updated_at": context.updated_at.isoformat() if context.updated_at else None
                }

            # If no context exists, return null/empty response
            return None

        except Exception as e:
            logger.error(f"Error fetching direct chat context: {str(e)}")
            raise



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



class FetchActiveLangRepository:
    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def fetch_active_lang(self) -> int:
        """
        Count how many languages user has actively learned in the last 7 days.

        Returns: Number of languages with at least one learned word in the last week.
        Format: {'active': 1}
        """
        # Calculate date range (last 7 days)
        today = date.today()
        one_week_ago = today - timedelta(days=7)

        # Get all languages the user is learning
        user_langs_query = select(UserLanguage.target_language_code).where(
            UserLanguage.user_id == self.user_id
        )
        user_langs_result = await self.db.execute(user_langs_query)
        user_languages = [lang[0] for lang in user_langs_result.all()]

        if not user_languages:
            return 0

        # Count distinct languages where user learned words in the last week
        active_langs_query = select(
            func.count(distinct(Word.language_code))
        ).select_from(UserWord).join(
            Word, UserWord.word_id == Word.id
        ).where(
            and_(
                UserWord.user_id == self.user_id,
                UserWord.is_learned == True,
                UserWord.created_at >= one_week_ago,
                UserWord.created_at <= datetime.combine(today, datetime.max.time()),
                Word.language_code.in_(user_languages)
            )
        )

        active_langs_result = await self.db.execute(active_langs_query)
        active_count = active_langs_result.scalar() or 0

        return active_count


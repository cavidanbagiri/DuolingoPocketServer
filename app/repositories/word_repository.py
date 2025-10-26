
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

from fastapi import HTTPException, status
from sqlalchemy import select, func, and_, update, or_, case, delete
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, outerjoin


# New Added For GOOGLE
import base64
from google.cloud import texttospeech
from google.oauth2 import service_account
#############################################




from app.logging_config import setup_logger
from app.models.word_model import Word, Sentence, SentenceWord, WordMeaning, Translation, SentenceTranslation, \
    LearnedWord
from app.models.user_model import Language, UserModel, UserLanguage, UserWord
from app.schemas.word_schema import GenerateAIChatSchema, GenerateAIWordSchema, TranslateSchema
from app.schemas.favorite_schemas import FavoriteWordBase, FavoriteCategoryBase, FavoriteCategoryResponse, FavoriteFetchWordResponse

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
            count_stmt = select(func.count(Word.id)).where(Word.language_code == lang_code)
            count_result = await self.db.execute(count_stmt)
            total_count = count_result.scalar_one()
            lang_data.append({
                "lang": lang_code,
                "total_words": total_count,
                "language_name": self._get_language_name(lang_code)
            })

        return lang_data


    async def fetch_words_for_language(self, lang_code: str, only_starred: bool = False,
                                       only_learned: bool = False, skip: int = 0, limit: int = 50) -> List[Dict[Any, Any]]:
        """Fetch words for a specific language — deduplicated by word, with multiple POS merged"""

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

        # 2. Build query
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

        # 3. Apply filters
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

        # 4. Pagination
        stmt = stmt.offset(skip).limit(limit)

        # 5. Execute
        result = await self.db.execute(stmt)
        rows = result.all()

        # 6. Group by Word.id
        word_map = defaultdict(lambda: {
            "id": None,
            "text": None,
            "frequency_rank": None,
            "level": None,
            "pos": set(),
            "translations": set(),  # Use set to avoid dupes
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

            # Merge POS
            if meaning and meaning.pos:
                word_map[word_id]["pos"].add(meaning.pos)

            # Merge translations
            if translation and translation.translated_text:
                word_map[word_id]["translations"].add(translation.translated_text)

            # OR: Keep only one translation (first one)
            # → if you don't want a list

            # Aggregate user flags (if any row is starred/learned, mark it)
            if is_starred:
                word_map[word_id]["is_starred"] = True
            if is_learned:
                word_map[word_id]["is_learned"] = True

        # 7. Convert to list and clean up
        words_list = []
        for data in word_map.values():
            words_list.append({
                "id": data["id"],
                "text": data["text"],
                "frequency_rank": data["frequency_rank"],
                "level": data["level"],
                "pos": sorted(list(data["pos"])) if data["pos"] else [],  # sorted for consistency
                "translation_to_native": list(data["translations"])[0] if data["translations"] else None,
                # Or: "translations": list(data["translations"]) if you want multiple
                "language_code": lang_code,
                "is_starred": data["is_starred"],
                "is_learned": data["is_learned"],
            })


        return words_list

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
            print(f"Failed to create Google TTS client: {str(e)}")
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
            print(f"Google TTS error: {str(e)}")
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
                print(f"Yandex API HTTP error: {e.response.status_code} - {e.response.text}")
                return None
            except (httpx.RequestError, KeyError, json.JSONDecodeError) as e:
                print(f"Yandex API request error: {str(e)}")
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



class GenerateAIQuestionRepository:

    def __init__(self):
        self.headers = {
            "Authorization": f"Api-Key {os.getenv('YANDEX_LANGMODEL_API_SECRET_KEY')}", # Renamed for clarity
            "Content-Type": "application/json"
        }
        self.model = 'yandexgpt' # or 'yandexgpt', choose based on needs
        self.folder_id = os.getenv('YANDEX_FOLDER_ID')
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.max_tokens = 1500  # Prevent overly long responses


    async def _call_yandex_gpt(self, messages: list) -> str:
        """Generic method to call YandexGPT Completion API."""
        print('[_call_yandex_gpt] Method called. Preparing payload...')

        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.2,
                "maxTokens": self.max_tokens
            },
            "messages": messages
        }

        async with aiohttp.ClientSession() as session:
            try:
                print('[DEBUG] Making request to Yandex API...')
                async with session.post(self.api_url, json=payload, headers=self.headers,
                                        timeout=aiohttp.ClientTimeout(total=30)) as response:

                    # CRITICAL: Get the response text regardless of the status code
                    response_text = await response.text()
                    print(f'[DEBUG] Yandex API Response Status: {response.status}')
                    print(f'[DEBUG] Yandex API Response Body: {response_text}')

                    # Now check for errors
                    response.raise_for_status()

                    # If successful, try to parse JSON
                    data = await response.json()
                    print(f'[DEBUG] Parsed JSON Response: {data}')
                    return data['result']['alternatives'][0]['message']['text']

            except aiohttp.ClientResponseError as e:
                # This will now print the actual error message from Yandex
                logger.error(f"YandexGPT API error: {e.status} - {e.message}. Response: {response_text}")
                raise HTTPException(status_code=502,
                                    detail=f"AI service error: {e.status}. Please check the request parameters.")
            except aiohttp.ClientConnectorError as e:
                logger.error(f"Connection to YandexGPT failed: {str(e)}")
                raise HTTPException(status_code=503, detail="Cannot connect to AI service. Check your network.")
            except asyncio.TimeoutError:
                logger.error("Request to YandexGPT timed out.")
                raise HTTPException(status_code=504, detail="AI service request timed out.")
            except (aiohttp.ClientError, KeyError) as e:
                logger.error(f"Unexpected error during YandexGPT call: {str(e)}")
                raise HTTPException(status_code=500, detail="An unexpected error occurred with the AI service.")



    async def generate_ai_chat(self, data: GenerateAIChatSchema) -> dict:
        """
        Generates a conversational response about a specific word.
        Returns a dict with the AI's reply.
        """

        # 1. Construct a detailed system prompt to guide the AI's behavior.
        system_prompt = (
            f"You are a helpful, precise, and enthusiastic language learning assistant. "
            f"The user is learning the {data.language} word '{data.word}'. "
            f"Their native language is {data.native}. "
            f"Answer the user's question specifically about this word. "
            f"Be concise, pedagogical, and provide clear examples. "
            f"Your answer must be in {data.native} to ensure the user understands. "
            f"Focus on explaining usage, grammar, nuances, or cultural context related to '{data.word}'."
            f"If the user's question is not related to the word, politely steer the conversation back to language learning."
        )

        # 2. Structure the messages for the API
        messages = [
            {
                "role": "system",
                "text": system_prompt
            },
            {
                "role": "user",
                "text": data.message
            }
        ]

        # 3. Call the API
        ai_response_text = await self._call_yandex_gpt(messages)

        # 4. Return the response in a structured format for the frontend
        return {"reply": ai_response_text.strip()}



class SearchRepository:

    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def search(self, native_language: str, target_language: str, query: str):
        print(f'Searching for: "{query}" in target_lang: {target_language}, native_lang: {native_language}')

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
    def __init__(self, db: AsyncSession, user_id: int):
        self.db = db
        self.user_id = user_id

    async def get_pos_statistics(self):
        stmt = (
            select(
                Word.language_code.label("language_code"),
                WordMeaning.pos.label("pos"),
                func.count(Word.id).label("total_count"),
                func.count(case((UserWord.is_learned == True, 1))).label("learned_count"),
                func.count(case((UserWord.is_starred == True, 1))).label("starred_count"),
            )
            .join(WordMeaning, Word.id == WordMeaning.word_id)
            .outerjoin(
                UserWord,
                (UserWord.word_id == Word.id) & (UserWord.user_id == self.user_id)
            )
            .group_by(Word.language_code, WordMeaning.pos)
            .order_by(Word.language_code, WordMeaning.pos)
        )

        result = await self.db.execute(stmt)
        rows = result.all()

        # Shape into expected format
        stats = {}
        for lang_code, pos, total, learned, starred in rows:
            if lang_code not in stats:
                stats[lang_code] = {
                    "language_name": self.get_language_name(lang_code)
                }
            stats[lang_code][pos] = total
            stats[lang_code][f"{pos}_learned"] = learned
            stats[lang_code][f"{pos}_starred"] = starred

        return list(stats.values())

    def get_language_name(self, code: str) -> str:
        language_map = {
            "en": "English",
            "ru": "Russian",
            "tr": "Turkish"
        }
        return language_map.get(code, code)



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

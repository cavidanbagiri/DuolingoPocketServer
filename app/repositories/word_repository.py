
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

from fastapi import HTTPException, status
from sqlalchemy import select, func, and_, update, or_, case, delete
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, outerjoin

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



# Get Statistics For Dashabord
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
        lang_code_map = {"Russian": "ru", "English": "en", "Spanish": "es"}
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

    async def generate_speech(self, text: str, lang: str) -> bytes:

        url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"

        api_key = os.getenv("YANDEX_SPEECHKIT_API_KEY")
        folder_id = os.getenv("YANDEX_FOLDER_ID")

        data = {
            "text": text,
            "lang": lang,
            "format": "mp3",  # Or 'oggopus'
            "folderId": folder_id,
        }

        headers = {
            "Authorization": f"Api-Key {api_key}",
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, data=data, headers=headers, timeout=30.0)

                response.raise_for_status()

                return response.content

            except httpx.HTTPStatusError as e:
                error_detail = e.response.text
                print(f"Yandex API error: {e.response.status_code} - {error_detail}")
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Yandex TTS service error: {error_detail}"
                )
            except httpx.RequestError as e:
                print(f"Network error requesting Yandex TTS: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable. Please try again."
                )


class GenerateAIWordRepository:

    def __init__(self):
        # DeepSeek configuration only
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        self.client = httpx.AsyncClient(timeout=60.0)
        self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        self.deepseek_headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json",
        }

    async def _call_deepseek_gpt(self, prompt: str) -> Optional[str]:
        """
        Make authenticated request to DeepSeek GPT API
        """
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful language learning assistant. Provide accurate, educational responses about words and phrases. Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}  # Force JSON response
        }

        try:
            response = await self.client.post(
                self.deepseek_url,
                headers=self.deepseek_headers,
                json=payload
            )
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except httpx.HTTPStatusError as e:
            print(f"DeepSeek API HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except (httpx.RequestError, KeyError, json.JSONDecodeError) as e:
            print(f"DeepSeek API request error: {str(e)}")
            return None

    @lru_cache(maxsize=100)
    def _create_prompt(self, word: str, target_lang: str, native_lang: str) -> str:
        """
        Create a highly structured prompt for AI to get perfect JSON responses
        """
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
        """
        Map language codes to their native names for better fallback responses
        """
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'tr': 'Turkish',
            'zh': 'Chinese',
            'ru': 'Russian',
            'de': 'German',
            'fr': 'French',
            'it': 'Italian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'pt': 'Portuguese'
        }
        return language_map.get(language_code, language_code)

    def _get_fallback_text(self, field_type: str, native_lang: str, target_lang: str, word: str) -> str:
        """
        Generate fallback text in the user's native language
        """
        target_lang_name = self._get_native_language_name(target_lang)
        native_lang_name = self._get_native_language_name(native_lang)

        fallback_templates = {
            'en': {
                'definition': f"A word in {target_lang_name} that you're learning",
                'example': f"Example with {word} in {target_lang_name} - Translation in {native_lang_name}",
                'usage': f"In daily conversations in {target_lang_name}",
                'phrase': f"Common expression with {word} - Translation",
                'grammar': "Check a dictionary for complete conjugations",
                'cultural': f"Important word in {target_lang_name} culture",
                'part_of_speech': 'word'
            },
            'es': {
                'definition': f"Palabra en {target_lang_name} que estás aprendiendo",
                'example': f"Ejemplo con {word} en {target_lang_name} - Traducción en {native_lang_name}",
                'usage': f"En conversaciones diarias en {target_lang_name}",
                'phrase': f"Expresión común con {word} - Traducción",
                'grammar': "Consulta un diccionario para conjugaciones completas",
                'cultural': f"Palabra importante en la cultura {target_lang_name}",
                'part_of_speech': 'palabra'
            },
            'tr': {
                'definition': f"Öğrendiğiniz {target_lang_name} dilinde bir kelime",
                'example': f"{target_lang_name} dilinde {word} ile örnek - {native_lang_name} çevirisi",
                'usage': f"{target_lang_name} dilinde günlük konuşmalarda",
                'phrase': f"{word} ile ortak ifade - Çeviri",
                'grammar': "Tam çekimler için bir sözlüğe bakın",
                'cultural': f"{target_lang_name} kültüründe önemli bir kelime",
                'part_of_speech': 'kelime'
            },
            'zh': {
                'definition': f"您正在学习的{target_lang_name}单词",
                'example': f"{target_lang_name}中的{word}示例 - {native_lang_name}翻译",
                'usage': f"在{target_lang_name}的日常对话中",
                'phrase': f"常用表达与{word} - 翻译",
                'grammar': "查看字典以获取完整变位",
                'cultural': f"{target_lang_name}文化中的重要词汇",
                'part_of_speech': '单词'
            }
        }

        template = fallback_templates.get(native_lang, fallback_templates['en'])
        return template[field_type]

    async def generate_ai_for_word(self, data) -> AIWordResponse:
        """
        Generate AI content for a word using DeepSeek API
        """
        print(f'🎯 Received request for word: {data.text}, target: {data.language}, native: {data.native}')

        # Create the prompt
        prompt = self._create_prompt(data.text, data.language, data.native)
        print(f"📝 Prompt generated, length: {len(prompt)} chars")

        # Call DeepSeek API
        ai_response = await self._call_deepseek_gpt(prompt)

        if not ai_response:
            print("❌ DeepSeek API failed, using fallback response")
            return self._create_fallback_response(data)

        try:
            # Clean the response
            cleaned_response = ai_response.strip()
            cleaned_response = re.sub(r'^```(?:json)?\s*', '', cleaned_response)
            cleaned_response = re.sub(r'\s*```$', '', cleaned_response)
            cleaned_response = cleaned_response.strip()

            print(f"🧹 Cleaned response preview: {cleaned_response[:200]}...")

            # Parse the JSON
            parsed_response = json.loads(cleaned_response)
            print("✅ Successfully parsed JSON!")
            print(f"📊 Parsed keys: {list(parsed_response.keys())}")

            # Validate and return
            return AIWordResponse(**parsed_response)

        except json.JSONDecodeError as e:
            print(f"❌ JSON decode failed! Error: {str(e)}")
            print(f"❌ Failed content: {ai_response}")
            return self._create_fallback_response(data)
        except ValidationError as e:
            print(f"❌ Pydantic validation failed: {e}")
            print(f"❌ Parsed data: {parsed_response}")
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
                self._get_fallback_text('usage', data.native, data.language, data.text)
            ],
            common_phrases=[
                self._get_fallback_text('phrase', data.native, data.language, data.text),
                self._get_fallback_text('phrase', data.native, data.language, data.text),
                self._get_fallback_text('phrase', data.native, data.language, data.text)
            ],
            grammar_tips=[
                self._get_fallback_text('grammar', data.native, data.language, data.text),
                "Pay attention to sentence structure",
                "Practice with different contexts"
            ],
            additional_insights=None
        )

    async def generate_ai_for_word_with_fallback(self, data) -> AIWordResponse:
        """
        Enhanced version with retry logic and comprehensive fallback
        """
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return await self.generate_ai_for_word(data)
            except Exception as e:
                print(f"🔄 Attempt {attempt + 1} failed for word '{data.text}': {str(e)}")
                if attempt == max_retries - 1:
                    print(f"❌ All attempts failed for word '{data.text}': {str(e)}")
                    return self._create_fallback_response(data)
                await asyncio.sleep(1)  # Wait before retry

        raise Exception("Unexpected error in retry logic")

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class GenerateAIQuestionRepository:

    def __init__(self):
        # DeepSeek configuration only
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set in environment variables")

        self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        self.deepseek_headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json",
        }
        self.max_tokens = 1500

    async def _call_deepseek_gpt(self, messages: list) -> str:
        """Call DeepSeek GPT API for conversational responses."""
        print('[_call_deepseek_gpt] Method called. Preparing payload...')

        # Convert to DeepSeek format
        deepseek_messages = []
        for msg in messages:
            deepseek_messages.append({
                "role": msg["role"],
                "content": msg["text"]
            })

        payload = {
            "model": "deepseek-chat",
            "messages": deepseek_messages,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        async with aiohttp.ClientSession() as session:
            try:
                print('[DEBUG] Making request to DeepSeek API...')
                async with session.post(self.deepseek_url, json=payload, headers=self.deepseek_headers,
                                        timeout=aiohttp.ClientTimeout(total=30)) as response:

                    response_text = await response.text()
                    print(f'[DEBUG] DeepSeek API Response Status: {response.status}')
                    print(f'[DEBUG] DeepSeek API Response Body: {response_text}')

                    response.raise_for_status()

                    data = await response.json()
                    print(f'[DEBUG] Parsed JSON Response: {data}')
                    return data['choices'][0]['message']['content']

            except aiohttp.ClientResponseError as e:
                logger.error(f"DeepSeek API error: {e.status} - {e.message}. Response: {response_text}")
                raise HTTPException(status_code=502,
                                    detail=f"AI service error: {e.status}. Please check the request parameters.")
            except aiohttp.ClientConnectorError as e:
                logger.error(f"Connection to DeepSeek failed: {str(e)}")
                raise HTTPException(status_code=503, detail="Cannot connect to AI service. Check your network.")
            except asyncio.TimeoutError:
                logger.error("Request to DeepSeek timed out.")
                raise HTTPException(status_code=504, detail="AI service request timed out.")
            except (aiohttp.ClientError, KeyError) as e:
                logger.error(f"Unexpected error during DeepSeek call: {str(e)}")
                raise HTTPException(status_code=500, detail="An unexpected error occurred with the AI service.")

    def _create_system_prompt(self, data: GenerateAIChatSchema) -> str:
        """Create a detailed system prompt for language learning assistance."""
        return (
            f"You are a helpful, precise, and enthusiastic language learning assistant. "
            f"The user is learning the {data.language} word '{data.word}'. "
            f"Their native language is {data.native}. "
            f"Answer the user's question specifically about this word. "
            f"Be concise, pedagogical, and provide clear examples. "
            f"Your answer must be in {data.native} to ensure the user understands. "
            f"Focus on explaining usage, grammar, nuances, or cultural context related to '{data.word}'."
            f"If the user's question is not related to the word, politely steer the conversation back to language learning."
        )

    def _create_messages(self, data: GenerateAIChatSchema) -> list:
        """Structure messages for the API call."""
        system_prompt = self._create_system_prompt(data)

        return [
            {
                "role": "system",
                "text": system_prompt
            },
            {
                "role": "user",
                "text": data.message
            }
        ]

    async def generate_ai_chat(self, data: GenerateAIChatSchema) -> dict:
        """
        Generates a conversational response about a specific word using DeepSeek.
        Returns a dict with the AI's reply.
        """
        print(f"🎯 Generating AI chat response for word: '{data.word}'")
        print(f"🔧 Target language: {data.language}, Native language: {data.native}")
        print(f"💬 User message: {data.message}")

        try:
            # Create messages for the API
            messages = self._create_messages(data)

            # Call DeepSeek API
            ai_response_text = await self._call_deepseek_gpt(messages)

            # Return the response in a structured format
            response_data = {"reply": ai_response_text.strip()}
            print(f"✅ Successfully generated AI response: {response_data['reply'][:100]}...")
            return response_data

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_ai_chat: {str(e)}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your request.")

    async def generate_ai_chat_with_retry(self, data: GenerateAIChatSchema, max_retries: int = 2) -> dict:
        """
        Enhanced version with retry logic for DeepSeek API calls.
        """
        for attempt in range(max_retries):
            try:
                return await self.generate_ai_chat(data)
            except HTTPException as e:
                if e.status_code >= 500 and attempt < max_retries - 1:  # Retry on server errors
                    print(f"🔄 Attempt {attempt + 1} failed with server error: {e.detail}")
                    await asyncio.sleep(1)
                    continue
                raise  # Re-raise client errors immediately
            except Exception as e:
                print(f"🔄 Attempt {attempt + 1} failed with unexpected error: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise HTTPException(status_code=500, detail="Failed to generate AI response after multiple attempts.")

        raise HTTPException(status_code=503, detail="AI service is currently unavailable.")


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



# Get Detail Word with sentences
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
        print(f'................word is {word}')

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

        # 5. Structure general example sentences
        general_sentences = []
        for sentence_link in word.in_sentences:
            sentence = sentence_link.sentence
            general_sentences.append({
                "id": sentence.id,
                "text": sentence.text,
                "translations": [
                    {
                        "language_code": t.language_code,
                        "translated_text": t.translated_text
                    } for t in sentence.translations
                ]
            })

        # 6. Structure translations
        translations = []
        for t in word.translations:
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



# Get POS Statistics For Each lang
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


# repository.py
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


# repository.py
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






# from fastapi import Query, HTTPException
# from sqlalchemy import select, func, text, case, update, delete
# from sqlalchemy.ext.asyncio import AsyncSession
#
# from app.models.word_model import WordModel, UserSavedWord
# from app.schemas.translate_schema import WordSchema
#
# from fastapi.concurrency import run_in_threadpool
# from functools import lru_cache
# import spacy
#
# from app.logging_config import setup_logger
# from app.schemas.word_schema import ChangeWordStatusSchema
#
# logger = setup_logger(__name__, "word.log")
#
#
#
# class DashboardRepository:
#     def __init__(self, user_id: int, db: AsyncSession):
#         self.user_id = int(user_id)
#         self.db = db
#
#     async def get_language_pair_stats(self):
#         # Main query for language pair stats
#         query = """
#             SELECT
#                 w.from_lang,
#                 w.to_lang,
#                 COUNT(*) as word_length,
#                 SUM(CASE WHEN usw.learned = TRUE THEN 1 ELSE 0 END) as learned,
#                 SUM(CASE WHEN usw.starred = TRUE THEN 1 ELSE 0 END) as starred,
#                 (
#                     SELECT JSONB_OBJECT_AGG(pos, cnt)
#                     FROM (
#                         SELECT
#                             w2.part_of_speech as pos,
#                             COUNT(*) as cnt
#                         FROM words w2
#                         JOIN user_saved_words usw2 ON w2.id = usw2.word_id
#                         WHERE usw2.user_id = :user_id
#                         AND w2.from_lang = w.from_lang
#                         AND w2.to_lang = w.to_lang
#                         GROUP BY w2.part_of_speech
#                     ) subq
#                 ) as pos_stats
#             FROM words w
#             JOIN user_saved_words usw ON w.id = usw.word_id
#             WHERE usw.user_id = :user_id
#             GROUP BY w.from_lang, w.to_lang
#             ORDER BY word_length DESC
#         """
#
#         result = await self.db.execute(
#             text(query),
#             {"user_id": self.user_id}
#         )
#
#         stats = []
#         for row in result:
#             # Convert SQLAlchemy row to dict
#             stat = {
#                 "user_id": self.user_id,
#                 "from_lang": row.from_lang,
#                 "to_lang": row.to_lang,
#                 "total_word": row.word_length,
#                 "learned": row.learned,
#                 "starred": row.starred,
#                 "pos_stats": {}
#             }
#
#             # Parse POS stats if available
#             if row.pos_stats:
#                 stat["pos_stats"] = dict(row.pos_stats.items())
#
#             stats.append(stat)
#         return stats
#
#
#
# class DashboardRepositoryLang:
#     def __init__(self, user_id: int, db: AsyncSession):
#         self.user_id = int(user_id)
#         self.db = db
#
#     async def get_language_pair_stats_by_lang(self, from_lang: str, to_lang: str):
#         # Get the words with their learned/starred status
#         words_result = await self.db.execute(
#             select(
#                 WordModel,
#                 UserSavedWord.learned,
#                 UserSavedWord.starred
#             )
#             .join(UserSavedWord)
#             .where(
#                 WordModel.from_lang == from_lang,
#                 WordModel.to_lang == to_lang,
#                 UserSavedWord.user_id == self.user_id,
#                 UserSavedWord.learned == False
#             )
#         )
#
#         # Convert results to a list of dictionaries with all fields
#         words = []
#         for word_model, learned, starred in words_result:
#             word_dict = word_model.__dict__
#             # word_dict['learned'] = learned
#             word_dict['starred'] = starred
#             words.append(word_dict)
#
#         # Get POS statistics (unchanged)
#         pos_stats = await self.db.execute(
#             select(
#                 WordModel.part_of_speech,
#                 func.count().label("count")
#             )
#             .join(UserSavedWord)
#             .where(
#                 WordModel.from_lang == from_lang,
#                 WordModel.to_lang == to_lang,
#                 UserSavedWord.user_id == self.user_id
#             )
#             .group_by(WordModel.part_of_speech)
#         )
#         pos_dict = {row.part_of_speech: row.count for row in pos_stats}
#
#         # Get learned/starred counts (unchanged)
#         status_counts = await self.db.execute(
#             select(
#                 func.sum(case((UserSavedWord.learned == True, 1), else_=0)).label("learned"),
#                 func.sum(case((UserSavedWord.starred == True, 1), else_=0)).label("starred")
#             )
#             .join(WordModel)
#             .where(
#                 WordModel.from_lang == from_lang,
#                 WordModel.to_lang == to_lang,
#                 UserSavedWord.user_id == self.user_id
#             )
#         )
#         learned, starred = status_counts.first()
#
#         return {
#             "words": words,
#             "stats": {
#                 "total_words": len(words),
#                 "learned": learned or 0,
#                 "starred": starred or 0,
#                 "pos_stats": pos_dict
#             }
#         }
#
#
#
# class FilterRepository:
#
#     def __init__(self, user_id: int, db: AsyncSession):
#         self.user_id = int(user_id)
#         self.db = db
#
#     async def filter(self,  from_lang: str, to_lang: str, part_of_speech: str):
#
#         words_result = await self.db.execute(
#             select(WordModel)
#             .join(UserSavedWord)
#             .where(
#                 WordModel.from_lang == from_lang,
#                 WordModel.to_lang == to_lang,
#                 WordModel.part_of_speech == part_of_speech,
#                 UserSavedWord.user_id == self.user_id
#             )
#         )
#         words = words_result.scalars().all()
#
#         return words
#
#
#
# class ChangeWordStatusRepository:
#
#     def __init__(self, data: ChangeWordStatusSchema,  user_id: int, db: AsyncSession):
#         self.data = data
#         self.user_id = int(user_id)
#         self.db = db
#
#     async def change_word_status(self):
#
#         word = await self.db.execute(select(UserSavedWord).where(UserSavedWord.word_id == self.data.word_id, UserSavedWord.user_id == int(self.user_id)))
#
#         word = word.scalar()
#
#         if not word:
#             raise HTTPException(status_code=404, detail="Word not found")
#
#         if self.data.w_status == 'starred':
#             new_status = not word.starred
#             await self.db.execute(
#                 update(UserSavedWord)
#                 .where(UserSavedWord.word_id == self.data.word_id, UserSavedWord.user_id == self.user_id)
#                 .values(starred=new_status)
#             )
#             await self.db.commit()
#
#         elif self.data.w_status == 'learned':
#             new_status = not word.learned
#             await self.db.execute(
#                 update(UserSavedWord)
#                 .where(UserSavedWord.word_id == self.data.word_id, UserSavedWord.user_id == self.user_id)
#                 .values(learned=new_status)
#             )
#             await self.db.commit()
#
#         elif self.data.w_status == 'delete':
#             await self.db.execute(
#                 delete(UserSavedWord)
#                 .where(UserSavedWord.word_id == self.data.word_id, UserSavedWord.user_id == self.user_id)
#             )
#             await self.db.commit()
#
#         return {
#             'w_status': self.data.w_status,
#             'detail': f'{self.data.w_status.title()} successfully changed',
#             'word_id': word.word_id
#         }
#
#
# class SaveWordRepository:
#     _nlp = None  # Class-level model instance for thread safety
#
#     def __init__(self, data: WordSchema, user_id: int, db: AsyncSession):
#         self.data = data
#         self.user_id = int(user_id)
#         self.db = db
#
#     @classmethod
#     async def get_nlp(cls):
#         """Thread-safe model loader with optimized pipeline"""
#         if cls._nlp is None:
#             cls._nlp = await run_in_threadpool(
#                 lambda: spacy.load("en_core_web_sm",
#                                    exclude=["parser", "ner", "lemmatizer"])
#             )
#             await run_in_threadpool(
#                 lambda: cls._nlp.enable_pipe("senter")
#             )
#         return cls._nlp
#
#     @lru_cache(maxsize=1000)
#     async def find_part_of_speech(self, selected_word: str) -> str:
#         """Proper async POS tagging using FastAPI's threadpool"""
#         if not selected_word.strip():
#             return 'other'
#
#         try:
#             nlp = await self.get_nlp()
#
#             # Process text in threadpool
#             doc = await run_in_threadpool(
#                 nlp,
#                 selected_word.lower().strip()
#             )
#
#             pos_mapping = {
#                 'propn': 'noun',
#                 'noun': 'noun',
#                 'verb': 'verb',
#                 'adj': 'adjective',
#                 'adv': 'adverb',
#                 'pron': 'pronoun',
#                 'adp': 'preposition',
#                 'conj': 'conjunction'
#             }
#
#             for token in doc:
#                 if token.pos_.lower() in pos_mapping:
#                     return pos_mapping[token.pos_.lower()]
#
#             return 'other'
#
#         except Exception as e:
#             logger.error(f"POS tagging failed for '{selected_word}': {str(e)}")
#             return 'other'
#
#     async def save_word(self):
#         normalized_word = self.data.word.lower().strip()
#
#         # Get POS tag async
#         self.data.part_of_speech = await self.find_part_of_speech(normalized_word)
#
#         # Upsert word
#         word = await self.db.execute(
#             select(WordModel).where(
#                 func.lower(WordModel.word) == normalized_word,
#                 WordModel.from_lang == self.data.from_lang,
#                 WordModel.to_lang == self.data.to_lang,
#                 WordModel.translation == self.data.translation.strip().lower()
#             )
#         )
#         word = word.scalar()
#
#         if not word:
#             word = WordModel(**self.data.model_dump())
#             word.word = normalized_word
#             self.db.add(word)
#             await self.db.flush()
#         else:
#             logger.info(f"Word '{word.word}' already exists")
#
#         # Upsert user-word relationship
#         await self.db.merge(
#             UserSavedWord(user_id=self.user_id, word_id=word.id)
#         )
#
#         await self.db.commit()
#         await self.db.refresh(word)
#         return word
#
#
#

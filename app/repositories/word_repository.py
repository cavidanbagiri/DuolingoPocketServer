
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional, Dict
import json
import httpx
import aiohttp
import asyncio
from functools import lru_cache

from fastapi import HTTPException
from sqlalchemy import select, func, and_, update, or_, case
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, outerjoin

from app.logging_config import setup_logger
from app.models.word_model import Word, Sentence, SentenceWord, WordMeaning, Translation, SentenceTranslation, \
    LearnedWord
from app.models.user_model import Language, UserModel, UserLanguage, UserWord
from app.schemas.translate_schema import TranslateSchema

logger = setup_logger(__name__, "word.log")

from app.schemas.word_schema import AIWordResponse

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
        self.headers = {
            "Authorization": f"Api-Key {os.getenv('YANDEX_LANGMODEL_API_SECRET_KEY')}",
            "Content-Type": "application/json"
        }
        self.model = 'yandexgpt'
        self.folder_id = os.getenv('YANDEX_FOLDER_ID')

    async def _call_yandex_gpt(self, prompt: str) -> Optional[str]:
        """
        Make authenticated request to Yandex GPT API
        """
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
        """
        Create a detailed, structured prompt for comprehensive language learning content
        with native language translations
        """
        return f"""
        You are an expert language teacher. Create comprehensive learning material for the word "{word}" in {target_lang} for a {native_lang} speaker.

        Provide a JSON response with EXACTLY this structure:
        {{
            "word": "{word}",
            "target_language": "{target_lang}",
            "native_language": "{native_lang}",
            "definition": "clear definition in {native_lang}",
            "pronunciation": "phonetic pronunciation guide if helpful",
            "part_of_speech": "main part of speech in {native_lang}",
            "examples": [
                "example 1 in {target_lang} with {native_lang} translation",
                "example 2 in {target_lang} with {native_lang} translation",
                "example 3 in {target_lang} with {native_lang} translation",
                "example 4 in {target_lang} with {native_lang} translation",
                "example 5 in {target_lang} with {native_lang} translation"
            ],
            "usage_contexts": [
                "context 1 where this word is used (in {native_lang})",
                "context 2 where this word is used (in {native_lang})",
                "context 3 where this word is used (in {native_lang})"
            ],
            "common_phrases": [
                "common phrase 1 using this word with translation",
                "common phrase 2 using this word with translation",
                "common phrase 3 using this word with translation"
            ],
            "grammar_tips": [
                "grammar tip 1 (in {native_lang})",
                "grammar tip 2 (in {native_lang})",
                "grammar tip 3 (in {native_lang})"
            ],
            "cultural_notes": [
                "cultural insight 1 (in {native_lang})",
                "cultural insight 2 (in {native_lang})",
                "cultural insight 3 (in {native_lang})"
            ],
            "additional_insights": {{
                "optional_extra_category": "optional extra information"
            }}
        }}

        CRITICAL REQUIREMENTS:
        1. Provide EXACTLY 5 diverse examples with translations (format: "Target Language Sentence - Native Language Translation")
        2. ALL text (definition, tips, contexts, notes) must be in the user's native language ({native_lang})
        3. Only the example sentences and common phrases should be in the target language ({target_lang})
        4. Include practical usage contexts and common phrases with translations
        5. Add helpful grammar tips specific to this word
        6. Share cultural insights about how natives use this word
        7. Ensure all content is educational, accurate, and engaging
        8. Make definitions clear and beginner-friendly
        9. If the word is a verb, include conjugation details in additional_insights
        10. For nouns, include gender and plural forms if relevant

        Remember: You're helping a {native_lang} speaker learn {target_lang}! Provide all explanations in {native_lang}.
        """


    # async def generate_ai_for_word(self, data) :
    #     """
    #     Main method to generate AI content for a word
    #     """
    #     print(f'Received request for word: {data.text}, target: {data.language}, native: {data.native}')
    #
    #     # Create the prompt
    #     prompt = self._create_prompt(data.text, data.language, data.native)
    #
    #     # Call Yandex GPT
    #     ai_response = await self._call_yandex_gpt(prompt)
    #
    #     if not ai_response:
    #         raise HTTPException(
    #             status_code=503,
    #             detail="AI service is temporarily unavailable. Please try again later."
    #         )
    #
    #     try:
    #         # Parse the JSON response from GPT
    #         parsed_response = json.loads(ai_response)
    #
    #         # Validate and return structured response
    #         return AIWordResponse(**parsed_response)
    #
    #     except json.JSONDecodeError:
    #         # Fallback: GPT didn't return JSON, return the raw text
    #         return AIWordResponse(
    #             word=data.text,
    #             target_language=data.language,
    #             native_language=data.native,
    #             definition=f"Palabra en {data.language} que estás aprendiendo",
    #             pronunciation=None,
    #             part_of_speech="verbo" ,
    #             examples=[
    #                 f"Ejemplo básico con {data.text} en {data.language} - Ejemplo básico en {data.native}",
    #                 f"Otra oración usando {data.text} - Otra oración en {data.native}",
    #                 f"Uso común de {data.text} - Uso común en {data.native}",
    #                 f"Frase práctica con {data.text} - Frase práctica en {data.native}",
    #                 f"Ejemplo contextual con {data.text} - Ejemplo contextual en {data.native}"
    #             ],
    #             usage_contexts=[
    #                 f"En conversaciones diarias en {data.language}",
    #                 f"Al escribir en {data.language}",
    #                 f"En situaciones formales e informales"
    #             ],
    #             common_phrases=[
    #                 f"Expresión común con {data.text} - Traducción",
    #                 f"Frase idiomática con {data.text} - Traducción"
    #             ],
    #             grammar_tips=[
    #                 f"Consulta un diccionario para conjugaciones completas" if is_verb else f"Verifica el género y plural",
    #                 f"Presta atención a la estructura de la oración",
    #                 f"Practica con diferentes contextos"
    #             ],
    #             cultural_notes=[
    #                 f"Palabra importante en la cultura {data.language}",
    #                 f"Uso frecuente en la literatura {data.language}",
    #                 f"Común en conversaciones cotidianas"
    #             ],
    #             additional_insights=None
    #         )

    async def generate_ai_for_word(self, data):
        """
        Main method to generate AI content for a word
        """
        print(f'Received request for word: {data.text}, target: {data.language}, native: {data.native}')

        # Create the prompt
        prompt = self._create_prompt(data.text, data.language, data.native)

        # Call Yandex GPT
        ai_response = await self._call_yandex_gpt(prompt)

        if not ai_response:
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily unavailable. Please try again later."
            )

        try:
            # Parse the JSON response from GPT
            parsed_response = json.loads(ai_response)

            # Validate and return structured response
            return AIWordResponse(**parsed_response)

        except json.JSONDecodeError:
            # Fallback: GPT didn't return JSON, return the raw text
            return AIWordResponse(
                word=data.text,
                target_language=data.language,
                native_language=data.native,
                definition=f"Palabra en {data.language} que estás aprendiendo",
                pronunciation=None,
                part_of_speech="palabra",  # FIXED: Simple default value
                examples=[
                    f"Ejemplo básico con {data.text} en {data.language} - Ejemplo básico en {data.native}",
                    f"Otra oración usando {data.text} - Otra oración en {data.native}",
                    f"Uso común de {data.text} - Uso común en {data.native}",
                    f"Frase práctica con {data.text} - Frase práctica en {data.native}",
                    f"Ejemplo contextual con {data.text} - Ejemplo contextual en {data.native}"
                ],
                usage_contexts=[
                    f"En conversaciones diarias en {data.language}",
                    f"Al escribir en {data.language}",
                    f"En situaciones formales e informales"
                ],
                common_phrases=[
                    f"Expresión común con {data.text} - Traducción",
                    f"Frase idiomática con {data.text} - Traducción"
                ],
                grammar_tips=[
                    f"Consulta un diccionario para más información",  # FIXED: Removed is_verb condition
                    f"Presta atención a la estructura de la oración",
                    f"Practica con diferentes contextos"
                ],
                cultural_notes=[
                    f"Palabra importante en la cultura {data.language}",
                    f"Uso frecuente en la literatura {data.language}",
                    f"Común en conversaciones cotidianas"
                ],
                additional_insights=None
            )

    async def generate_ai_for_word(self, data):
        """
        Main method to generate AI content for a word
        """
        print(f'Received request for word: {data.text}, target: {data.language}, native: {data.native}')

        # Create the prompt
        prompt = self._create_prompt(data.text, data.language, data.native)

        # Call Yandex GPT
        ai_response = await self._call_yandex_gpt(prompt)

        if not ai_response:
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily unavailable. Please try again later."
            )

        try:
            # Parse the JSON response from GPT
            parsed_response = json.loads(ai_response)

            # Validate and return structured response
            return AIWordResponse(**parsed_response)

        except json.JSONDecodeError:
            # Fallback: GPT didn't return JSON, return the raw text
            return AIWordResponse(
                word=data.text,
                target_language=data.language,
                native_language=data.native,
                definition=f"Palabra en {data.language} que estás aprendiendo",
                pronunciation=None,
                part_of_speech="palabra",  # FIXED: Simple default value
                examples=[
                    f"Ejemplo básico con {data.text} en {data.language} - Ejemplo básico en {data.native}",
                    f"Otra oración usando {data.text} - Otra oración en {data.native}",
                    f"Uso común de {data.text} - Uso común en {data.native}",
                    f"Frase práctica con {data.text} - Frase práctica en {data.native}",
                    f"Ejemplo contextual con {data.text} - Ejemplo contextual en {data.native}"
                ],
                usage_contexts=[
                    f"En conversaciones diarias en {data.language}",
                    f"Al escribir en {data.language}",
                    f"En situaciones formales e informales"
                ],
                common_phrases=[
                    f"Expresión común con {data.text} - Traducción",
                    f"Frase idiomática con {data.text} - Traducción"
                ],
                grammar_tips=[
                    f"Consulta un diccionario para más información",  # FIXED: Removed is_verb condition
                    f"Presta atención a la estructura de la oración",
                    f"Practica con diferentes contextos"
                ],
                cultural_notes=[
                    f"Palabra importante en la cultura {data.language}",
                    f"Uso frecuente en la literatura {data.language}",
                    f"Común en conversaciones cotidianas"
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
                if attempt == max_retries - 1:
                    print(f"All attempts failed for word '{data.text}': {str(e)}")
                    return AIWordResponse(
                        word=data.text,
                        target_language=data.language,
                        native_language=data.native,
                        definition=f"Palabra en {data.language} que estás aprendiendo",
                        pronunciation=None,
                        part_of_speech="palabra",  # FIXED: Simple default value
                        examples=[
                            f"Ejemplo básico con {data.text} en {data.language} - Ejemplo básico en {data.native}",
                            f"Otra oración usando {data.text} - Otra oración en {data.native}",
                            f"Uso común de {data.text} - Uso común en {data.native}",
                            f"Frase práctica con {data.text} - Frase práctica en {data.native}",
                            f"Ejemplo contextual con {data.text} - Ejemplo contextual en {data.native}"
                        ],
                        usage_contexts=[
                            f"En conversaciones diarias en {data.language}",
                            f"Al escribir en {data.language}",
                            f"En situaciones formales e informales"
                        ],
                        common_phrases=[
                            f"Expresión común con {data.text} - Traducción",
                            f"Frase idiomática con {data.text} - Traducción"
                        ],
                        grammar_tips=[
                            f"Consulta un diccionario para más información",  # FIXED: Removed is_verb condition
                            f"Presta atención a la estructura de la oración",
                            f"Practica con diferentes contextos"
                        ],
                        cultural_notes=[
                            f"Palabra importante en la cultura {data.language}",
                            f"Uso frecuente en la literatura {data.language}",
                            f"Común en conversaciones cotidianas"
                        ],
                        additional_insights=None
                    )

        # This line should never be reached due to the loop logic, but included for safety
        raise Exception("Unexpected error in retry logic")

    # async def generate_ai_for_word_with_fallback(self, data) -> AIWordResponse:
    #     """
    #     Enhanced version with retry logic and comprehensive fallback
    #     """
    #     max_retries = 2
    #     for attempt in range(max_retries):
    #         try:
    #             return await self.generate_ai_for_word(data)
    #         except Exception as e:
    #             if attempt == max_retries - 1:
    #                 print(f"All attempts failed for word '{data.text}': {str(e)}")
    #                 return AIWordResponse(
    #                     word=data.text,
    #                     target_language=data.language,
    #                     native_language=data.native,
    #                     definition=f"Palabra en {data.language} que estás aprendiendo",
    #                     pronunciation=None,
    #                     part_of_speech="palabra",  # FIXED: Simple default value
    #                     examples=[
    #                         f"Ejemplo básico con {data.text} en {data.language} - Ejemplo básico en {data.native}",
    #                         f"Otra oración usando {data.text} - Otra oración en {data.native}",
    #                         f"Uso común de {data.text} - Uso común en {data.native}",
    #                         f"Frase práctica con {data.text} - Frase práctica en {data.native}",
    #                         f"Ejemplo contextual con {data.text} - Ejemplo contextual en {data.native}"
    #                     ],
    #                     usage_contexts=[
    #                         f"En conversaciones diarias en {data.language}",
    #                         f"Al escribir en {data.language}",
    #                         f"En situaciones formales e informales"
    #                     ],
    #                     common_phrases=[
    #                         f"Expresión común con {data.text} - Traducción",
    #                         f"Frase idiomática con {data.text} - Traducción"
    #                     ],
    #                     grammar_tips=[
    #                         f"Consulta un diccionario para más información",  # FIXED: Removed is_verb condition
    #                         f"Presta atención a la estructura de la oración",
    #                         f"Practica con diferentes contextos"
    #                     ],
    #                     cultural_notes=[
    #                         f"Palabra importante en la cultura {data.language}",
    #                         f"Uso frecuente en la literatura {data.language}",
    #                         f"Común en conversaciones cotidianas"
    #                     ],
    #                     additional_insights=None
    #                 )
    #
    #     # This line should never be reached due to the loop logic, but included for safety
    #     raise Exception("Unexpected error in retry logic")
    #
    #
    # async def generate_ai_for_word_with_fallback(self, data) -> AIWordResponse:
    #     """
    #     Enhanced version with retry logic and comprehensive fallback
    #     """
    #     max_retries = 2
    #     for attempt in range(max_retries):
    #         try:
    #             return await self.generate_ai_for_word(data)
    #         except Exception as e:
    #             if attempt == max_retries - 1:
    #                 print(f"All attempts failed for word '{data.text}': {str(e)}")
    #                 return AIWordResponse(
    #                     word=data.text,
    #                     target_language=data.language,
    #                     native_language=data.native,
    #                     definition=f"Palabra en {data.language} que estás aprendiendo",
    #                     pronunciation=None,
    #                     part_of_speech="verbo" ,
    #                     examples=[
    #                         f"Ejemplo básico con {data.text} en {data.language} - Ejemplo básico en {data.native}",
    #                         f"Otra oración usando {data.text} - Otra oración en {data.native}",
    #                         f"Uso común de {data.text} - Uso común en {data.native}",
    #                         f"Frase práctica con {data.text} - Frase práctica en {data.native}",
    #                         f"Ejemplo contextual con {data.text} - Ejemplo contextual en {data.native}"
    #                     ],
    #                     usage_contexts=[
    #                         f"En conversaciones diarias en {data.language}",
    #                         f"Al escribir en {data.language}",
    #                         f"En situaciones formales e informales"
    #                     ],
    #                     common_phrases=[
    #                         f"Expresión común con {data.text} - Traducción",
    #                         f"Frase idiomática con {data.text} - Traducción"
    #                     ],
    #                     grammar_tips=[
    #                         f"Consulta un diccionario para conjugaciones completas" if is_verb else f"Verifica el género y plural",
    #                         f"Presta atención a la estructura de la oración",
    #                         f"Practica con diferentes contextos"
    #                     ],
    #                     cultural_notes=[
    #                         f"Palabra importante en la cultura {data.language}",
    #                         f"Uso frecuente en la literatura {data.language}",
    #                         f"Común en conversaciones cotidianas"
    #                     ],
    #                     additional_insights=None
    #                 )
    #
    #
    #
    #     # This line should never be reached due to the loop logic, but included for safety
    #     raise Exception("Unexpected error in retry logic")
    #
    #




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
                "example": meaning.example,
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

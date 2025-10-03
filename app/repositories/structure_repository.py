


import os
import csv
from datetime import datetime
import aiohttp
import asyncio
import pandas as pd
import requests
from enum import Enum
from pathlib import Path
from collections import defaultdict

import httpx
import json
import re
from typing import List, Dict, Any, Optional

from app.models.ai_models import WordAnalysisResponse


from fastapi import HTTPException

from sqlalchemy import select, func, and_, update, or_, case, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, outerjoin
from sqlalchemy.future import select

from app.models.word_model import Word, Sentence, SentenceWord, WordMeaning, Translation, SentenceTranslation, \
    LearnedWord, Category
from app.models.user_model import Language, UserModel, UserLanguage, UserWord
from app.schemas.translate_schema import TranslateSchema

from app.services.ai_service import AIService

from app.logging_config import setup_logger
logger = setup_logger(__name__, "word.log")



################################################################## Call function for creating a structure

# Class For executing the functions
class HelperFunction:

    def __init__(self, db: AsyncSession):
        self.db = db

    # 1 - Create language model
    async def create_language_list(self):
        lang_repo = CreateMainStructureLanguagesListRepository(self.db)
        await lang_repo.create_main_languages()

    async def save_words_to_database(self) -> str:
        save_words_repo = SaveEnglishWordsToDatabaseRepository(self.db)
        success = await save_words_repo.save_words_to_database()
        return "Words saved successfully" if success else "Failed to save words"

    async def generate_english_words_with_ai(self) -> str:
        """Generate AI content for English words"""
        repo = FetchEnglishWordsGenerateWithAIToPOSDefCat(self.db)
        result = await repo.generate_words_with_ai(limit=1)

        if result["success"]:
            return f"Success: {result['message']}"
        else:
            return f"Failed: {result.get('error', 'Unknown error')}"


    async def test_batch_processing(self, start_id: int, end_id: int) -> Dict[str, Any]:
        """Test batch processing"""
        repo = CreateMainStructureRepository(self.db)
        return await repo.process_word_batch(start_id, end_id)



# Main Class Working
class CreateMainStructureRepository:
    def __init__(self, db):
        self.db = db
        self.helper_funcs = HelperFunction(db)

    async def create_main_structure(self):
        """Main method to orchestrate the word generation process"""
        logger.info("🏗️ Starting main structure creation process")

        # 1 - Create Language List [en, ru, es] - Uncomment if needed
        # await self.helper_funcs.create_language_list()

        # 2 - Create Word List For English lang - Uncomment if needed
        # await self.helper_funcs.save_words_to_database()

        # 3 - Generate English Words with AI (start with 1 word)
        # await self.helper_funcs.generate_english_words_with_ai()
        pass

    # This is for adding words to database For english
    async def process_word_batch(self, start_id: int, end_id: int, batch_size: int = 10):
        """Process words in specific ID range with controlled batch size"""
        print('////////////////here is workinf')
        logger.info(f"🔬 Testing batch processing: IDs {start_id} to {end_id}")

        repo = FetchEnglishWordsGenerateWithAIToPOSDefCat(self.db)

        # Process in smaller batches to avoid timeouts
        processed_count = 0
        errors = []

        for word_id in range(start_id, end_id + 1):
            try:
                # Check if word exists and needs processing
                word = await self._get_word_by_id(word_id)
                if not word:
                    logger.info(f"⏭️ Word ID {word_id} not found, skipping")
                    continue

                # Check if already processed
                if await self._word_already_processed(word_id):
                    logger.info(f"⏭️ Word '{word.text}' (ID: {word_id}) already processed, skipping")
                    continue

                logger.info(f"🔍 Processing word {processed_count + 1}: '{word.text}' (ID: {word_id})")

                # Process single word
                result = await repo.generate_single_word(word)

                if result["success"]:
                    processed_count += 1
                    logger.info(f"✅ Successfully processed: {word.text}")
                else:
                    errors.append(f"Word ID {word_id}: {result.get('error', 'Unknown error')}")

                # Small delay between words to avoid rate limiting
                await asyncio.sleep(1)

                # Log progress every 10 words
                if processed_count % 10 == 0:
                    logger.info(f"📊 Progress: {processed_count} words processed, {len(errors)} errors")

            except Exception as e:
                error_msg = f"Error processing word ID {word_id}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        logger.info(f"🎉 Batch processing complete: {processed_count} words processed, {len(errors)} errors")
        return {"processed": processed_count, "errors": errors}

    async def process_all_words(self, batch_size: int = 50):
        """Process all unprocessed words in batches"""
        logger.info("🚀 Starting full processing of all words")

        repo = FetchEnglishWordsGenerateWithAIToPOSDefCat(self.db)
        total_processed = 0
        total_errors = []

        while True:
            # Process words in batches
            result = await repo.generate_words_with_ai(limit=batch_size)

            if result["success"]:
                batch_processed = len(result["processed"])
                total_processed += batch_processed
                total_errors.extend(result["errors"])

                logger.info(f"📦 Processed batch: {batch_processed} words")
                logger.info(f"📈 Total progress: {total_processed} words")

                # If no words were processed in this batch, we're done
                if batch_processed == 0:
                    break

                # Longer delay between batches
                await asyncio.sleep(5)
            else:
                logger.error(f"❌ Batch processing failed: {result.get('error')}")
                total_errors.append(result.get('error', 'Batch processing failed'))
                break

        logger.info(f"🎉 Full processing complete: {total_processed} words processed, {len(total_errors)} errors")
        return {"total_processed": total_processed, "errors": total_errors}

    async def _get_word_by_id(self, word_id: int):
        """Get word by ID"""
        # from sqlalchemy import select
        # from models import Word

        result = await self.db.execute(select(Word).where(Word.id == word_id))
        return result.scalar_one_or_none()

    async def _word_already_processed(self, word_id: int) -> bool:
        """Check if word already has meanings"""
        # from sqlalchemy import select
        # from models import WordMeaning

        result = await self.db.execute(
            select(WordMeaning).where(WordMeaning.word_id == word_id)
        )
        return result.first() is not None





################################################################################################################ Production Updates
################################################################## Create Top languages List


class CEFRLevel(str, Enum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"




class CreateMainStructureLanguagesListRepository:

    def __init__(self, db):
        self.db = db

    async def create_main_languages(self):
        top_languages = [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "ru", "name": "Russian"},
            # {"code": "tr", "name": "Turkish"},
        ]

        for lang in top_languages:
            # Check if language already exists
            existing = await self.db.execute(
                select(Language).where(Language.code == lang["code"])
            )
            if not existing.scalar_one_or_none():
                self.db.add(Language(**lang))

        await self.db.commit()
        return "Top 10 languages created successfully"





############################################################################################################################################# For Spanish languages section

class CreateMainStructureForSpanishRepository:

    def __init__(self, db: AsyncSession):
        self.db = db


    async def insert_spanish_words_to_table(self):

        # Path to your Excel file
        excel_file_path = os.path.join(os.path.dirname(__file__), "spanish_top_10000.xlsx")

        try:
            # Read the Excel file
            df = pd.read_excel(excel_file_path, engine='openpyxl')

            # Check if the dataframe has the expected columns
            print("Excel file columns:", df.columns.tolist())
            print(f"Total words to process: {len(df)}")

            # Process each row and create Word objects
            words_to_add = []
            existing_words_count = 0

            for index, row in df.iterrows():
                # Adjust column names based on your Excel file structure
                # Assuming your Excel has columns: 'rank', 'word', 'frequency'
                word_text = str(row['word']).strip()
                frequency_rank = int(row['rank'])
                frequency = int(row['frequency'])

                # Check if word already exists in database
                existing_word = await self.db.execute(
                    select(Word).where(Word.text == word_text).where(Word.language_code == "es")
                )
                existing_word = existing_word.scalar_one_or_none()

                if existing_word:
                    existing_words_count += 1
                    continue  # Skip if word already exists

                # Determine level based on frequency rank (you can adjust this logic)
                level = self._determine_level(frequency_rank)

                # Create new Word object
                new_word = Word(
                    text=word_text,
                    language_code="es",  # Spanish
                    frequency_rank=frequency_rank,
                    level=level
                )

                words_to_add.append(new_word)

                # Batch insert every 100 words to avoid memory issues
                if len(words_to_add) >= 100:
                    self.db.add_all(words_to_add)
                    await self.db.commit()
                    words_to_add = []
                    print(f"Processed {index + 1} words...")

            # Insert any remaining words
            if words_to_add:
                self.db.add_all(words_to_add)
                await self.db.commit()

            print(f"Successfully inserted {len(df) - existing_words_count} Spanish words")
            print(f"Skipped {existing_words_count} existing words")

            return {
                'msg': 'Spanish words inserted successfully',
                'inserted_count': len(df) - existing_words_count,
                'skipped_count': existing_words_count
            }

        except FileNotFoundError:
            print(f"Error: Excel file not found at {excel_file_path}")
            return {'error': 'Excel file not found'}
        except Exception as e:
            await self.db.rollback()
            print(f"Error: {str(e)}")
            return {'error': str(e)}

    def _determine_level(self, frequency_rank: int) -> str:
        """Determine CEFR level based on frequency rank"""
        if frequency_rank <= 1000:
            return "A1"
        elif frequency_rank <= 2000:
            return "A2"
        elif frequency_rank <= 4000:
            return "B1"
        elif frequency_rank <= 6000:
            return "B2"
        elif frequency_rank <= 8000:
            return "C1"
        else:
            return "C2"


    def fetch_words_from_spanish(self):
        """Download and process top Spanish words from GitHub"""
        url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt"

        try:
            response = requests.get(url)
            lines = response.text.split('\n')

            data = []
            for i, line in enumerate(lines[:10000]):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        frequency = int(parts[1])
                        rank = i + 1
                        data.append([rank, word, frequency])

            df = pd.DataFrame(data, columns=['rank', 'word', 'frequency'])

            # Save to Excel
            excel_path = os.path.join(os.path.dirname(__file__), "spanish_top_10000.xlsx")
            df.to_excel(excel_path, index=False, engine='openpyxl')

            print(f"Created Excel with top 10,000 Spanish words at: {excel_path}")
            return df

        except Exception as e:
            print(f"Error: {str(e)}")
            return None



class GenerateSpanishSentences:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")

        self.client = httpx.AsyncClient(timeout=30.0)
        self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        self.auth_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_sentences(self, limit: int = 10) -> dict:
        """
        Generate 5 native Spanish example sentences for Spanish words
        that currently have fewer than 5 such sentences.
        """
        # Step 1: Get Spanish language from DB
        result = await self.db.execute(select(Language).where(Language.code == "es"))
        es_lang = result.scalars().first()
        if not es_lang:
            raise RuntimeError("Spanish language (es) not found in DB")

        # Step 2: Find Spanish words with < 5 example sentences
        # subquery = (
        #     select(SentenceWord.word_id)
        #     .join(Sentence)
        #     .where(Sentence.language_code == "es")
        #     .group_by(SentenceWord.word_id)
        #     .having(func.count(SentenceWord.sentence_id) < 5)
        # ).subquery()

        stmt = (
            select(Word)
            .where(Word.language_code == "es")
            .where(Word.id.between(19107, 19110))
            # .where(Word.id.in_(select(subquery.c.word_id)))
            # .limit(limit)
        )

        result = await self.db.execute(stmt)
        words = result.scalars().all()

        if not words:
            logger.info("✅ All Spanish words already have at least 5 example sentences.")
            return {"status": "complete", "processed": 0}

        logger.info(f"🎯 Starting generation for {len(words)} Spanish words: {[w.text for w in words]}")

        processed_count = 0
        async with httpx.AsyncClient() as client:
            for word in words:
                try:
                    logger.info(f"📌 Generating 5 Spanish sentences for '{word.text}' (ID: {word.id})")

                    # Generate exactly 5 good Spanish sentences
                    sentences = await self._generate_five_spanish_sentences(client, word.text)
                    if len(sentences) < 5:
                        logger.warning(f"⚠️ Only {len(sentences)} sentences generated for '{word.text}'")
                        continue

                    # Save all sentences and links
                    for sent_text in sentences:
                        await self._save_sentence_and_link(sent_text, word, es_lang)

                    await self.db.commit()
                    processed_count += 1
                    logger.info(f"✅ Saved 5 Spanish sentences for '{word.text}'")

                except Exception as e:
                    await self.db.rollback()
                    logger.error(f"💥 Failed processing word '{word.text}': {str(e)}", exc_info=True)
                    continue

        return {
            "status": "success",
            "words_processed": processed_count,
            "total_requested": len(words),
            "message": f"Generated native Spanish sentences for {processed_count}/{len(words)} words."
        }

    async def _generate_five_spanish_sentences(self, client: httpx.AsyncClient, word: str) -> List[str]:
        """
        Use DeepSeek to generate 5 diverse, natural Spanish sentences using the word.
        Must respond in valid JSON.
        """
        prompt = f"""
        Eres un profesor nativo de español.
        Genera exactamente 5 ejemplos naturales y gramaticalmente correctos usando la palabra "{word}" en español.

        Reglas:
        - Cada oración debe incluir "{word}" de forma natural
        - Usa diferentes contextos: pregunta, afirmación, emoción, vida diaria
        - Evita frases robóticas como "usar la palabra {word}"
        - No hables sobre la palabra en sí (ej. "la palabra hola")
        - Devuelve SOLO un objeto JSON:
          {{ "sentences": ["Oración 1", "Oración 2", ...] }}
        - Sin explicaciones ni formato adicional
        """

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "max_tokens": 600
        }

        try:
            response = await client.post(self.deepseek_url, headers=self.auth_headers, json=payload)
            if response.status_code != 200:
                logger.warning(f"📡 DeepSeek API error {response.status_code}: {response.text}")
                return self._fallback_sentences(word)

            content = response.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```json|```$", "", content).strip()
            parsed = json.loads(content)

            raw_sents = parsed.get("sentences")
            if not isinstance(raw_sents, list):
                raise ValueError("Invalid format")

            cleaned = []
            forbidden_patterns = [
                rf"\busar\s+{re.escape(word)}\b",
                rf"\bsobre\s+{re.escape(word)}\b",
                rf"\bpalabra\s+{re.escape(word)}\b"
            ]

            for s in raw_sents:
                if isinstance(s, str):
                    s_clean = s.strip('". ')
                    if (len(s_clean) > 5 and
                        word.lower() in s_clean.lower() and
                        not any(re.search(pat, s_clean, re.IGNORECASE) for pat in forbidden_patterns) and
                        s_clean not in cleaned):
                        cleaned.append(s_clean)

            # Fallback padding
            while len(cleaned) < 5:
                for fb in self._fallback_sentences(word):
                    if fb not in cleaned:
                        cleaned.append(fb)
                        break

            return cleaned[:5]

        except Exception as e:
            logger.error(f"🔥 Failed to parse AI response for '{word}': {str(e)}")
            return self._fallback_sentences(word)

    def _fallback_sentences(self, word: str) -> List[str]:
        """Safe fallback Spanish sentences."""
        return {
            "hola": [
                "¡Hola! ¿Cómo estás?",
                "Dile hola a tu hermano.",
                "Hoy dije hola a un amigo nuevo.",
                "Ella me saludó con un cálido 'hola'.",
                "¿Sabes cómo decir hola en francés?"
            ],
            "gracias": [
                "Muchas gracias por tu ayuda.",
                "Siempre digo gracias cuando alguien me ayuda.",
                "Él me dio un regalo y yo dije gracias.",
                "No olvides decir gracias después de comer.",
                "Las gracias son importantes en cualquier idioma."
            ]
        }.get(word.lower(), [
            f"Esta es una oración con la palabra {word}.",
            f"Me gusta usar {word} cuando hablo.",
            f"Ayer escuché a alguien decir {word}.",
            f"La palabra {word} suena muy bien.",
            f"Vamos a practicar la palabra {word}."
        ])

    async def _save_sentence_and_link(self, text: str, word: Word, lang: Language):
        """
        Save Spanish sentence and link to the word.
        Avoid duplicates.
        """
        # Check if sentence already exists
        result = await self.db.execute(
            select(Sentence).where(
                Sentence.text.icontains(text.strip()),
                Sentence.language_code == "es"
            )
        )
        sentence = result.scalars().first()

        if not sentence:
            sentence = Sentence(
                text=text.strip(),
                language_code="es",
                # created_at=datetime.utcnow()
            )
            self.db.add(sentence)
            await self.db.flush()
            logger.debug(f"<saved> New Spanish sentence ID {sentence.id}: '{text}'")
        else:
            logger.debug(f"<exists> Reusing sentence ID {sentence.id}")

        # Prevent duplicate link
        result = await self.db.execute(
            select(SentenceWord).where(
                SentenceWord.sentence_id == sentence.id,
                SentenceWord.word_id == word.id
            )
        )
        if not result.scalars().first():
            link = SentenceWord(sentence_id=sentence.id, word_id=word.id)
            self.db.add(link)
            logger.debug(f"<linked> Word '{word.text}' → Sentence '{text}'")
        else:
            logger.debug(f"<skip> Duplicate link for '{word.text}'")




############################################################################################################################################# For Russian languages section

############################################ Save Russian words to WordModel
class CreateMainStructureForRussianRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def insert_russian_words_to_table(self):
        # 1️⃣ Build the correct file path (same folder as this file)
        file_path = os.path.join(os.path.dirname(__file__), "russian_words.xlsx")

        # 2️⃣ Read Excel file
        df = pd.read_excel(file_path)

        # Expecting columns: russian, translation, lang_code, frequency_rank, level
        for _, row in df.iterrows():


            ru_word = str(row["russian"]).strip()
            translation_text = str(row["translation"]).strip()
            target_lang = "en"
            frequency_rank = int(row["rank"])
            level = None

            # 3️⃣ Check if Russian word already exists
            existing_word = await self.db.scalar(
                select(Word).where(Word.text == ru_word, Word.language_code == "ru")
            )
            if not existing_word:
                existing_word = Word(
                    text=ru_word,
                    language_code="ru",
                    frequency_rank=frequency_rank,
                    level=level
                )
                self.db.add(existing_word)
                await self.db.flush()  # gets ID without committing

            # 4️⃣ Check if translation exists
            existing_translation = await self.db.scalar(
                select(Translation).where(
                    Translation.source_word_id == existing_word.id,
                    Translation.target_language_code == target_lang,
                    Translation.translated_text == translation_text
                )
            )
            if not existing_translation:
                translation = Translation(
                    source_word_id=existing_word.id,
                    target_language_code=target_lang,
                    translated_text=translation_text
                )
                self.db.add(translation)

        # 5️⃣ Commit after processing all rows
        await self.db.commit()


class GenerateRussianSentences:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")

        self.client = httpx.AsyncClient(timeout=30.0)
        self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        self.auth_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_sentences(self, limit: int = 10) -> dict:
        """
        Generate 5 example sentences IN RUSSIAN for Russian words.
        Only for words that have < 5 Russian example sentences.
        """
        # Step 1: Get Russian language
        result = await self.db.execute(select(Language).where(Language.code == "ru"))
        ru_lang = result.scalars().first()
        if not ru_lang:
            raise RuntimeError("Russian language (ru) not found in DB")

        # Step 2: Find Russian words with fewer than 5 example sentences
        subquery = (
            select(SentenceWord.word_id)
            .join(Sentence)
            .where(Sentence.language_code == "ru")
            .group_by(SentenceWord.word_id)
            .having(func.count(SentenceWord.sentence_id) < 5)
        ).subquery()

        stmt = (
            select(Word)
            .where(Word.language_code == "ru")
            # .where(Word.id.in_(select(subquery.c.word_id)))
            .where(Word.id.between(9916, 9940))
            # .limit(limit)
        )

        result = await self.db.execute(stmt)
        words = result.scalars().all()

        if not words:
            logger.info("✅ All Russian words already have at least 5 example sentences.")
            return {"status": "complete", "processed": 0}

        logger.info(f"🎯 Starting generation for {len(words)} Russian words: {[w.text for w in words]}")

        processed_count = 0
        async with httpx.AsyncClient() as client:
            for word in words:
                try:
                    logger.info(f"📌 Generating 5 Russian sentences for '{word.text}' (ID: {word.id})")

                    # Generate exactly 5 good Russian sentences
                    sentences = await self._generate_five_russian_sentences(client, word.text)
                    if len(sentences) < 5:
                        logger.warning(f"⚠️ Only {len(sentences)} sentences generated for '{word.text}'")
                        continue

                    # Save all sentences and links
                    for sent_text in sentences:
                        await self._save_sentence_and_link(sent_text, word, ru_lang)

                    await self.db.commit()
                    processed_count += 1
                    logger.info(f"✅ Saved 5 Russian sentences for '{word.text}'")

                except Exception as e:
                    await self.db.rollback()
                    logger.error(f"💥 Failed processing word '{word.text}': {str(e)}", exc_info=True)
                    continue

        return {
            "status": "success",
            "words_processed": processed_count,
            "total_requested": len(words),
            "message": f"Generated native Russian sentences for {processed_count}/{len(words)} words."
        }

    async def _generate_five_russian_sentences(self, client: httpx.AsyncClient, word: str) -> List[str]:
        """
        Ask DeepSeek to generate 5 diverse, natural Russian sentences using the word.
        Must respond in valid JSON.
        """
        prompt = f"""
        Вы — профессиональный преподаватель русского языка.
        Сгенерируйте ровно 5 разнообразных, грамматически правильных примеров предложений на русском языке с использованием слова "{word}".

        Требования:
        - Каждое предложение должно естественным образом включать слово "{word}"
        - Используйте разные контексты: вопрос, утверждение, эмоция, повседневная жизнь
        - Избегайте бессмысленных или неестественных фраз
        - Никогда не говорите о самом слове (например, «слово привет»)
        - Верните ТОЛЬКО JSON объект:
          {{ "sentences": ["Предложение 1", "Предложение 2", ...] }}
        - Никаких объяснений, ничего лишнего
        """

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "max_tokens": 600
        }

        try:
            response = await client.post(self.deepseek_url, headers=self.auth_headers, json=payload)
            if response.status_code != 200:
                logger.warning(f"📡 DeepSeek API error {response.status_code}: {response.text}")
                return self._fallback_sentences(word)

            content = response.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```json|```$", "", content).strip()
            parsed = json.loads(content)

            raw_sents = parsed.get("sentences")
            if not isinstance(raw_sents, list):
                raise ValueError("Invalid format")

            cleaned = []
            forbidden_patterns = [
                rf"\bиспользовать\s+{re.escape(word)}\b",
                rf"\bпро\s+{re.escape(word)}\b",
                rf"\bслово\s+{re.escape(word)}\b"
            ]

            for s in raw_sents:
                if isinstance(s, str):
                    s_clean = s.strip('". ')
                    if (len(s_clean) > 5 and
                        word in s_clean and
                        not any(re.search(pat, s_clean, re.IGNORECASE) for pat in forbidden_patterns) and
                        s_clean not in cleaned):
                        cleaned.append(s_clean)

            # Pad if needed
            while len(cleaned) < 5:
                for fb in self._fallback_sentences(word):
                    if fb not in cleaned:
                        cleaned.append(fb)
                        break

            return cleaned[:5]

        except Exception as e:
            logger.error(f"🔥 Failed to parse AI response for '{word}': {str(e)}")
            return self._fallback_sentences(word)

    def _fallback_sentences(self, word: str) -> List[str]:
        """Safe fallback Russian sentences."""
        return {
            "привет": [
                "Привет! Как дела?",
                "Скажи привет маме!",
                "Привет, давно не виделись!",
                "Он сказал мне привет.",
                "Привет тебе от моего друга."
            ],
            "спасибо": [
                "Спасибо за помощь!",
                "Я очень благодарен тебе за спасибо.",
                "Спасибо, что пришёл вовремя.",
                "Она поблагодарила его за спасибо.",
                "Мы говорим спасибо, когда нам помогают."
            ]
        }.get(word.lower(), [
            f"Это хорошее слово — {word}.",
            f"Я люблю использовать слово {word}.",
            f"Вчера я услышал слово {word} в песне.",
            f"Мне нравится, как звучит {word}.",
            f"Давай включим слово {word} в диалог."
        ])

    async def _save_sentence_and_link(self, text: str, word: Word, lang: Language):
        """
        Save Russian sentence and link to the word.
        Avoid duplicates.
        """
        # Check if sentence already exists
        result = await self.db.execute(
            select(Sentence).where(
                Sentence.text.icontains(text.strip()),
                Sentence.language_code == "ru"
            )
        )
        sentence = result.scalars().first()

        if not sentence:
            sentence = Sentence(
                text=text.strip(),
                language_code="ru",
                # created_at=datetime.utcnow()
            )
            self.db.add(sentence)
            await self.db.flush()
            logger.debug(f"<saved> New Russian sentence ID {sentence.id}: '{text}'")
        else:
            logger.debug(f"<exists> Reusing sentence ID {sentence.id}")

        # Prevent duplicate SentenceWord
        result = await self.db.execute(
            select(SentenceWord).where(
                SentenceWord.sentence_id == sentence.id,
                SentenceWord.word_id == word.id
            )
        )
        if not result.scalars().first():
            link = SentenceWord(sentence_id=sentence.id, word_id=word.id)
            self.db.add(link)
            logger.debug(f"<linked> Word '{word.text}' → Sentence '{text}'")
        else:
            logger.debug(f"<skip> Duplicate link for '{word.text}'")


class TranslateRussianSentences:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.api_key = os.getenv("YANDEX_TRANSLATE_API_SECRET_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.translate_url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        if not self.api_key:
            raise RuntimeError("YANDEX_TRANSLATE_API_SECRET_KEY is not set")
        if not self.folder_id:
            raise RuntimeError("YANDEX_FOLDER_ID is not set")
        # Target languages to translate INTO

        # Target languages to translate INTO
        self.target_langs = ["en", "es", "tr"]
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }

    async def translate_sentence(self, min_id: int = None, max_id: int = None) -> Dict[str, any]:
        """
        Translate Russian sentences (language_code='ru') into en, es, tr.
        Only translates those missing at least one of the target translations.
        Processes all eligible sentences in [min_id, max_id], one by one.
        Commits after each sentence.
        """
        # Step 1: Ensure required languages exist in DB
        lang_map = await self._load_languages(["ru"] + self.target_langs)
        ru_lang = lang_map.get("ru")
        if not ru_lang:
            raise RuntimeError("Russian language (ru) not found in DB")

        # Step 2: Query Russian sentences in range
        query = select(Sentence).where(Sentence.language_code == "ru").where(Sentence.id.between(min_id, max_id))

        if min_id is not None:
            query = query.where(Sentence.id >= min_id)
        if max_id is not None:
            query = query.where(Sentence.id <= max_id)

        # Exclude already fully translated
        subquery = (
            select(SentenceTranslation.source_sentence_id)
            .where(SentenceTranslation.language_code.in_(self.target_langs))
            .group_by(SentenceTranslation.source_sentence_id)
            .having(func.count(SentenceTranslation.language_code) >= len(self.target_langs))
        ).subquery()

        # query = query.where(
        #     # Sentence.id.not_in(select(subquery.c.source_sentence_id))
        # .where(Word.id.between(9910, 9915))
        # )
        # query = query.order_by(Sentence.id)

        result = await self.db.execute(query)
        sentences = result.scalars().all()

        if not sentences:
            logger.info(f"✅ No untranslated Russian sentences found in range {min_id}–{max_id}")
            return {
                "status": "no_pending",
                "range": {"min_id": min_id, "max_id": max_id},
                "message": "No Russian sentences need translation."
            }

        logger.info(f"🔍 Found {len(sentences)} Russian sentences to translate in range {min_id}–{max_id}")

        processed_count = 0
        failed_count = 0

        # Process one sentence at a time
        for sentence in sentences:
            try:
                logger.info(f"📌 Translating Russian sentence ID {sentence.id}: '{sentence.text}'")

                # Get existing translations
                result_trans = await self.db.execute(
                    select(SentenceTranslation.language_code)
                    .where(SentenceTranslation.source_sentence_id == sentence.id)
                )
                existing_langs = {row[0] for row in result_trans.fetchall()}
                missing_langs = [lang for lang in self.target_langs if lang not in existing_langs]

                if not missing_langs:
                    logger.debug(f"⏭️ All translations exist for ID {sentence.id}")
                    continue

                logger.info(f"🌐 Translating to: {missing_langs}")

                # Translate all missing languages
                translations = await self._translate_text_batch(sentence.text, missing_langs)
                saved_count = 0

                for lang_code, translated_text in translations.items():
                    if not translated_text:
                        continue

                    trans_model = SentenceTranslation(
                        source_sentence_id=sentence.id,
                        language_code=lang_code,
                        translated_text=translated_text.strip(),
                    )
                    self.db.add(trans_model)
                    logger.debug(f"<saved> [{lang_code}] {translated_text}")
                    saved_count += 1

                # ✅ Commit after each sentence
                await self.db.commit()
                processed_count += 1
                logger.info(f"✅ Translated ID {sentence.id} → {saved_count} languages")

            except Exception as e:
                await self.db.rollback()
                logger.error(f"💥 Failed to process sentence ID {sentence.id}: {str(e)}", exc_info=True)
                failed_count += 1
                continue  # Keep going

        logger.info(f"🎉 Translation batch completed: {processed_count} done, {failed_count} failed")
        return {
            "status": "completed",
            "range": {"min_id": min_id, "max_id": max_id},
            "total_found": len(sentences),
            "translated": processed_count,
            "failed": failed_count,
            "message": f"Translated {processed_count}/{len(sentences)} Russian sentences"
        }

    async def _load_languages(self, codes: List[str]) -> Dict[str, Language]:
        """Load languages from DB, auto-create if missing."""
        result = await self.db.execute(select(Language).where(Language.code.in_(codes)))
        lang_map = {lang.code: lang for lang in result.scalars().all()}

        names = {
            "ru": "Russian",
            "en": "English",
            "es": "Spanish",
            "tr": "Turkish"
        }

        for code in codes:
            if code not in lang_map:
                new_lang = Language(code=code, name=names.get(code, code.capitalize()))
                self.db.add(new_lang)
                lang_map[code] = new_lang
                logger.info(f"🔧 Auto-added language: {code} ({new_lang.name})")

        if lang_map:
            await self.db.flush()

        return lang_map

    async def _translate_text_batch(self, text: str, targets: List[str]) -> Dict[str, str]:
        """
        Translate one Russian sentence into multiple languages.
        Makes separate request per language (Yandex v2 doesn't support plural targetLanguageCode).
        """
        results = {}

        async with aiohttp.ClientSession() as session:
            for lang_code in targets:
                try:
                    payload = {
                        "folderId": self.folder_id,
                        "texts": [text],
                        "targetLanguageCode": lang_code,  # ← SINGULAR!
                        "format": "PLAIN_TEXT"
                    }

                    async with session.post(self.translate_url, json=payload, headers=self.headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            translated = data["translations"][0]["text"].strip()
                            results[lang_code] = translated
                            logger.debug(f"<translation_OK> [{lang_code}] {translated}")
                        else:
                            error_text = await resp.text()
                            logger.warning(f"📡 Failed to translate to '{lang_code}': {error_text}")
                            results[lang_code] = None

                except Exception as e:
                    logger.error(f"🚨 Translation failed for '{lang_code}': {str(e)}", exc_info=True)
                    results[lang_code] = None

                # Optional: small delay to avoid rate limits
                await asyncio.sleep(0.1)

        return results





############################################################################################################################################# For English languages section

############################################ Save English words to WordModel
class SaveEnglishWordsToDatabaseRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.base_dir = Path(__file__).parent

    async def read_words_from_file(self, file_path: str = "english_words.txt") -> list[str]:
        """Read words from text file and return as list"""
        # try:
        #     with open(file_path, 'r', encoding='utf-8') as file:
        #         words = [line.strip() for line in file if line.strip()]
        #     logger.info(f"Read {len(words)} words from {file_path}")
        #     return words
        # except FileNotFoundError:
        #     logger.error(f"File {file_path} not found")
        #     return []
        # except Exception as e:
        #     logger.error(f"Error reading file: {e}")
        #     return []

        try:
            # Create absolute path to the file
            absolute_path = self.base_dir / file_path

            logger.info(f"Looking for file at: {absolute_path}")

            if not absolute_path.exists():
                logger.error(f"File not found at: {absolute_path}")
                # List files in directory for debugging
                files_in_dir = list(self.base_dir.glob("*.txt"))
                logger.info(f"Text files in directory: {[f.name for f in files_in_dir]}")
                return []

            with open(absolute_path, 'r', encoding='utf-8') as file:
                words = [line.strip() for line in file if line.strip()]

            logger.info(f"Successfully read {len(words)} words from {absolute_path}")
            return words

        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return []


    async def ensure_english_language_exists(self) -> Language:
        """Ensure English language exists in database"""
        # Check if English exists
        result = await self.db.execute(
            select(Language).where(Language.code == "en")
        )
        english_lang = result.scalar_one_or_none()

        if not english_lang:
            english_lang = Language(code="en", name="English")
            self.db.add(english_lang)
            await self.db.commit()
            await self.db.refresh(english_lang)
            logger.info("Created English language in database")

        return english_lang

    async def assign_frequency_ranks(self, words: list[str]) -> dict[str, int]:
        """Assign frequency ranks based on position in list (1-based index)"""
        return {word: rank + 1 for rank, word in enumerate(words)}

    async def estimate_cefr_level(self, frequency_rank: int, word: str) -> CEFRLevel:
        """Estimate CEFR level based on frequency rank and word characteristics"""
        # Simple heuristic - you can refine this later with AI
        if frequency_rank <= 1000:
            return CEFRLevel.A1
        elif frequency_rank <= 2500:
            return CEFRLevel.A2
        elif frequency_rank <= 5000:
            return CEFRLevel.B1
        elif frequency_rank <= 7500:
            return CEFRLevel.B2
        elif frequency_rank <= 9000:
            return CEFRLevel.C1
        else:
            return CEFRLevel.C2

    async def word_exists(self, word_text: str, language_code: str = "en") -> bool:
        """Check if word already exists in database"""
        result = await self.db.execute(
            select(Word).where(
                Word.text == word_text,
                Word.language_code == language_code
            )
        )
        return result.scalar_one_or_none() is not None

    async def save_words_to_database(self, batch_size: int = 100):
        """Main method to read words and save to database"""
        try:
            # Read words from file
            words = await self.read_words_from_file()
            if not words:
                logger.error("No words found to process")
                return False

            # Ensure English language exists
            english_lang = await self.ensure_english_language_exists()

            # Assign frequency ranks
            frequency_map = await self.assign_frequency_ranks(words)

            saved_count = 0
            skipped_count = 0

            # Process words in batches for better performance
            for i in range(0, len(words), batch_size):
                batch_words = words[i:i + batch_size]
                batch_saved = 0

                for word_text in batch_words:
                    # Skip if word already exists
                    if await self.word_exists(word_text):
                        skipped_count += 1
                        continue

                    # Estimate CEFR level
                    frequency_rank = frequency_map[word_text]
                    cefr_level = await self.estimate_cefr_level(frequency_rank, word_text)

                    # Create word object
                    word = Word(
                        text=word_text.lower().strip(),
                        language_code="en",
                        frequency_rank=frequency_rank,
                        level=cefr_level.value
                    )

                    self.db.add(word)
                    batch_saved += 1

                # Commit batch
                await self.db.commit()
                saved_count += batch_saved
                logger.info(f"Saved batch {i // batch_size + 1}: {batch_saved} words")

                # Small delay to avoid overwhelming the database
                await asyncio.sleep(0.1)

            logger.info(f"Successfully saved {saved_count} words to database")
            logger.info(f"Skipped {skipped_count} duplicate words")
            return True

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error saving words to database: {e}")
            return False

    async def get_word_count(self) -> int:
        """Get total word count in database"""
        result = await self.db.execute(select(Word))
        words = result.scalars().all()
        return len(words)


class GenerateEnglishSentence:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")

        self.client = httpx.AsyncClient(timeout=30.0)
        self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        self.auth_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_sentences_for_words(self, batch_limit: int = 10):
        """
                Hardcoded: Load words with id 1, 2, 3
                Generate 5 sentences for each
                Save to Sentence + SentenceWord
                """
        # Step 1: Get words by ID
        result = await self.db.execute(
            select(Word).where(Word.id.between(6499, 6500)).order_by(Word.id)
        )
        words = result.scalars().all()

        if len(words) == 0:
            raise RuntimeError("No words found with IDs 1, 2, or 3")

        # logger.info(f"🎯 Loaded {len(words)} words: {[w.text for w in words]}")

        processed_count = 0
        async with httpx.AsyncClient() as client:
            for word in words:
                try:
                    # logger.info(f"📌 Generating 5 sentences for word '{word.text}' (ID: {word.id})")

                    # Generate exactly 5 good sentences
                    sentences = await self._generate_five_natural_sentences(client, word.text)
                    # if len(sentences) < 5:
                    #     logger.warning(f"⚠️ Only generated {len(sentences)} sentences for '{word.text}'")
                    #     # Still proceed with what we have
                    # else:
                    #     logger.debug(f"📝 Sentences: {sentences}")

                    # Save all sentences and links
                    for sent_text in sentences:
                        await self._save_sentence_and_link(sent_text, word)

                    processed_count += 1
                    # logger.info(f"✅ Successfully processed '{word.text}'")

                    await self.db.commit()

                except Exception as e:
                    logger.error(f"💥 Failed processing word '{word.text}': {str(e)}", exc_info=True)
                    continue  # Don't stop on one failure

        await self.db.commit()
        # logger.info(f"🎉 Finished! Processed {processed_count}/{len(words)} words.")

        return {
            "status": "success",
            "words_processed": processed_count,
            "total_target": len(words),
            "target_ids": [1, 2, 3],
            "message": f"Attempted sentence generation for words ID 1, 2, 3."
        }

    async def _generate_five_natural_sentences(self, client: httpx.AsyncClient, word: str) -> List[str]:
        """
        Call DeepSeek to generate 5 diverse, realistic sentences.
        """
        prompt = f"""
                Generate exactly 5 diverse, grammatically correct example sentences in English using the word "{word}".

                Rules:
                - Each must naturally include "{word}"
                - Use different contexts: question, command, emotional, daily life
                - Avoid robotic phrases like "use you every day"
                - Do NOT say "the word {word}"
                - Return ONLY a JSON object:
                  {{ "sentences": ["Sentence 1", "Sentence 2", ...] }}
                - NO explanations, NO markdown
                """

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "max_tokens": 600
        }

        try:
            response = await client.post(self.deepseek_url, headers=self.auth_headers, json=payload)
            if response.status_code != 200:
                logger.warning(f"📡 DeepSeek error {response.status_code}: {response.text}")
                return self._fallback_sentences(word)

            content = response.json()["choices"][0]["message"]["content"].strip()
            # logger.debug(f"📥 Raw AI response for '{word}': {content[:200]}...")

            # Clean code block
            content = re.sub(r"^```json|```$", "", content).strip()
            parsed = json.loads(content)

            raw_sents = parsed.get("sentences")
            if not isinstance(raw_sents, list):
                raise ValueError("Not a list")

            cleaned = []
            forbidden = [
                rf"\buse\s+{re.escape(word)}\b",
                rf"\babout\s+{re.escape(word)}\b",
                rf"\bthe\s+word\s+{re.escape(word)}\b"
            ]

            for s in raw_sents:
                if isinstance(s, str):
                    s_clean = s.strip('". ')
                    if (len(s_clean) > 5 and
                            word.lower() in s_clean.lower() and
                            not any(re.search(pat, s_clean.lower()) for pat in forbidden) and
                            s_clean not in cleaned):
                        cleaned.append(s_clean)

            # Fallback padding
            while len(cleaned) < 5:
                for fb in self._fallback_sentences(word):
                    if fb not in cleaned:
                        cleaned.append(fb)
                        break

            return cleaned[:5]

        except Exception as e:
            logger.error(f"🔥 Parse failed for '{word}': {str(e)}")
            return self._fallback_sentences(word)

    def _fallback_sentences(self, word: str) -> List[str]:
        return {
            "you": [
                "Can you help me?",
                "I believe in you.",
                "Are you okay?",
                "You made my day!",
                "Could you please repeat that?"
            ],
            "the": [
                "The sun is shining.",
                "I left the book on the table.",
                "The movie was amazing.",
                "He is the best player.",
                "Don't forget the keys!"
            ],
            "be": [
                "I want to be a doctor.",
                "Let's be honest.",
                "They will be here soon.",
                "To be or not to be?",
                "We should be careful."
            ]
        }.get(word.lower(), [
            f"One day, {word} will change everything.",
            f"I’ve never seen such a {word}.",
            f"This isn’t just any {word}.",
            f"Do you have a {word}?",
            f"The story of {word} began long ago."
        ])

    async def _save_sentence_and_link(self, text: str, word: Word):
        """
        Save sentence and link to word (avoid duplicates)
        """
        # Check if sentence already exists (case-insensitive + trim)
        result = await self.db.execute(
            select(Sentence).where(
                Sentence.text.icontains(text.strip()),
                Sentence.language_code == "en"
            )
        )
        sentence = result.scalars().first()

        if not sentence:
            sentence = Sentence(
                text=text.strip(),
                language_code="en",
                # created_at=datetime.utcnow()
            )
            self.db.add(sentence)
            await self.db.flush()
            # logger.debug(f"<saved> New sentence ID {sentence.id}: '{text}'")
        else:
            logger.debug(f"<exists> Reusing sentence ID {sentence.id}")

        # Check if link already exists
        result = await self.db.execute(
            select(SentenceWord).where(
                SentenceWord.sentence_id == sentence.id,
                SentenceWord.word_id == word.id
            )
        )
        if not result.scalars().first():
            link = SentenceWord(sentence_id=sentence.id, word_id=word.id)
            self.db.add(link)
            # logger.debug(f"<linked> Word '{word.text}' → Sentence '{text}'")
        else:
            pass
            # logger.debug(f"<skip> Link already exists for '{word.text}' ↔ '{text}'")


class TranslateEnglishSentencesRepository:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.api_key = os.getenv("YANDEX_TRANSLATE_API_SECRET_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.translate_url = "https://translate.api.cloud.yandex.net/translate/v2/translate"

        if not self.api_key:
            raise RuntimeError("YANDEX_TRANSLATE_API_SECRET_KEY is not set")
        if not self.folder_id:
            raise RuntimeError("YANDEX_FOLDER_ID is not set")

        # Target languages
        self.target_langs = ["ru", "es", "tr"]
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }

    async def translate_english_sentences(self, min_id: int = None, max_id: int = None) -> Dict[str, any]:
        """
        Translate ALL English sentences in [min_id, max_id] that are missing translations.
        Processes up to 1000 sentences in a batch.
        Commits after each sentence (resilient).
        """
        # Step 1: Load required languages
        lang_map = await self._load_languages(["en"] + self.target_langs)
        if not lang_map.get("en"):
            raise RuntimeError("English language (en) not found in DB")

        # Step 2: Query all eligible sentences in range
        query = select(Sentence).where(Sentence.language_code == "en")

        if min_id is not None:
            query = query.where(Sentence.id >= min_id)
        if max_id is not None:
            query = query.where(Sentence.id <= max_id)

        # Exclude fully translated sentences
        subquery = (
            select(SentenceTranslation.source_sentence_id)
            .where(SentenceTranslation.language_code.in_(self.target_langs))
            .group_by(SentenceTranslation.source_sentence_id)
            .having(func.count(SentenceTranslation.language_code) >= len(self.target_langs))
        ).subquery()

        query = query.where(Sentence.id.not_in(select(subquery.c.source_sentence_id)))
        query = query.order_by(Sentence.id)  # Process from low to high

        result = await self.db.execute(query)
        sentences = result.scalars().all()

        if not sentences:
            # logger.info(f"✅ No untranslated sentences found in range {min_id}–{max_id}")
            return {
                "status": "complete",
                "total_found": 0,
                "translated": 0,
                "range": {"min_id": min_id, "max_id": max_id},
                "message": "No pending translations in range."
            }

        # logger.info(f"🎯 Found {len(sentences)} sentences to translate in range {min_id}–{max_id}")

        processed_count = 0
        failed_count = 0

        # Process each sentence one by one (safe & resilient)
        for sentence in sentences:
            try:
                # logger.info(f"📌 Processing sentence ID {sentence.id}: '{sentence.text}'")

                # Check existing translations
                result_trans = await self.db.execute(
                    select(SentenceTranslation.language_code)
                    .where(SentenceTranslation.source_sentence_id == sentence.id)
                )
                existing_langs = {row[0] for row in result_trans.fetchall()}
                missing_langs = [lang for lang in self.target_langs if lang not in existing_langs]

                if not missing_langs:
                    logger.debug(f"⏭️ All translations exist for ID {sentence.id}")
                    continue

                # logger.info(f"🌐 Translating to: {missing_langs}")

                # Call Yandex API (one request per target language)
                translations = await self._translate_text_batch(sentence.text, missing_langs)
                saved_count = 0

                for lang_code, translated_text in translations.items():
                    if not translated_text:
                        continue

                    trans_model = SentenceTranslation(
                        source_sentence_id=sentence.id,
                        language_code=lang_code,
                        translated_text=translated_text.strip(),
                    )
                    self.db.add(trans_model)
                    # logger.debug(f"<saved> [{lang_code}] {translated_text}")
                    saved_count += 1

                # ✅ Commit after each sentence
                await self.db.commit()
                processed_count += 1
                # logger.info(f"✅ Translated ID {sentence.id} → {saved_count} languages")

            except Exception as e:
                await self.db.rollback()
                logger.error(f"💥 Failed processing sentence ID {sentence.id}: {str(e)}", exc_info=True)
                failed_count += 1
                continue  # Keep going

        # logger.info(f"🎉 Batch translation completed: {processed_count} success, {failed_count} failed")
        return {
            "status": "completed",
            "range": {"min_id": min_id, "max_id": max_id},
            "total_in_range": len(sentences),
            "translated": processed_count,
            "failed": failed_count,
            "message": f"Batch translation finished for IDs {min_id} to {max_id}"
        }


    # async def translate_english_sentences(self, min_id: int = None, max_id: int = None) -> Dict[str, any]:
    #     """
    #     Translate ONE English sentence that is missing ru/es/tr translations.
    #     Optionally filter by Sentence.id BETWEEN min_id and max_id.
    #
    #     Ideal for step-by-step processing and testing.
    #     """
    #     # Step 1: Load required languages
    #     lang_map = await self._load_languages(["en"] + self.target_langs)
    #     en_lang = lang_map.get("en")
    #     if not en_lang:
    #         raise RuntimeError("English language (en) not found in DB")
    #
    #     # Step 2: Build query with optional ID bounds
    #     query = select(Sentence).where(Sentence.language_code == "en")
    #
    #     if min_id is not None:
    #         query = query.where(Sentence.id >= min_id)
    #     if max_id is not None:
    #         query = query.where(Sentence.id <= max_id)
    #
    #     # Order by ID and get only the first one
    #     query = query.order_by(Sentence.id).limit(5)
    #     print(f'................................query is {query}')
    #
    #     # result = await self.db.execute(query)
    #     # sentence = result.scalar_one_or_none()
    #     result = await self.db.execute(query.limit(1))  # ← Critical: ensure limit
    #     sentence = result.scalar_one_or_none()
    #
    #     if not sentence:
    #         logger.info(f"✅ No eligible English sentence found in range {min_id}–{max_id}.")
    #         return {"status": "not_found", "range": [min_id, max_id]}
    #
    #     logger.info(f"🔍 Processing single sentence ID {sentence.id}: '{sentence.text}'")
    #
    #     try:
    #         # Check existing translations
    #         existing_translations = await self._get_existing_translations(sentence.id)
    #         missing_langs = [lang for lang in self.target_langs if lang not in existing_translations]
    #
    #         if not missing_langs:
    #             logger.info(f"⏭️ All translations already exist for sentence ID {sentence.id}")
    #             return {
    #                 "status": "already_translated",
    #                 "sentence_id": sentence.id,
    #                 "text": sentence.text
    #             }
    #
    #         logger.info(f"🌐 Translating sentence ID {sentence.id} to: {missing_langs}")
    #
    #         # Call Yandex API
    #         translations = await self._translate_text_batch(sentence.text, missing_langs)
    #         if not translations:
    #             logger.error(f"❌ Translation failed for sentence ID {sentence.id}")
    #             return {
    #                 "status": "translation_failed",
    #                 "sentence_id": sentence.id,
    #                 "text": sentence.text
    #             }
    #
    #         # Save each translation
    #         for lang_code, translated_text in translations.items():
    #             if not translated_text.strip():
    #                 continue
    #
    #             trans_model = SentenceTranslation(
    #                 source_sentence_id=sentence.id,
    #                 language_code=lang_code,
    #                 translated_text=translated_text.strip(),
    #             )
    #             self.db.add(trans_model)
    #             logger.debug(f"<saved> [{lang_code}] {translated_text}")
    #
    #         # ✅ Commit immediately
    #         await self.db.commit()
    #         logger.info(f"✅ Successfully translated sentence ID {sentence.id}")
    #
    #         return {
    #             "status": "success",
    #             "sentence_id": sentence.id,
    #             "text": sentence.text,
    #             "translated_into": list(translations.keys())
    #         }
    #
    #     except Exception as e:
    #         await self.db.rollback()
    #         logger.error(f"💥 Failed to process sentence ID {sentence.id}: {str(e)}", exc_info=True)
    #         return {
    #             "status": "error",
    #             "sentence_id": sentence.id,
    #             "error": str(e)
    #         }


    async def _load_languages(self, codes: List[str]) -> Dict[str, Language]:
        """Load languages from DB, auto-create if missing."""
        result = await self.db.execute(select(Language).where(Language.code.in_(codes)))
        lang_map = {lang.code: lang for lang in result.scalars().all()}

        # Auto-create missing ones
        for code in codes:
            if code not in lang_map:
                name = {
                    "en": "English",
                    "ru": "Russian",
                    "es": "Spanish",
                    "tr": "Turkish"
                }.get(code, code.capitalize())
                new_lang = Language(code=code, name=name)
                self.db.add(new_lang)
                lang_map[code] = new_lang
                # logger.info(f"🔧 Auto-added language: {code} ({name})")

        if lang_map:
            await self.db.flush()

        return lang_map

    async def _get_existing_translations(self, sentence_id: int) -> Dict[str, str]:
        """Get already translated texts by lang_code."""
        result = await self.db.execute(
            select(SentenceTranslation)
            .where(SentenceTranslation.source_sentence_id == sentence_id)
        )
        translations = result.scalars().all()
        return {t.language_code: t.translated_text for t in translations}

    async def _translate_text_batch(self, text: str, targets: List[str]) -> Dict[str, str]:
        """
        Translate one sentence into multiple languages.
        Since Yandex v2 doesn't support multiple targetLanguageCode in one call,
        we make separate requests per language.
        """
        results = {}

        async with aiohttp.ClientSession() as session:
            for lang_code in targets:
                try:
                    payload = {
                        "folderId": self.folder_id,
                        "texts": [text],  # Must be list
                        "targetLanguageCode": lang_code,  # ← SINGULAR! Not 'Codes'
                        "format": "PLAIN_TEXT"
                    }

                    async with session.post(self.translate_url, json=payload, headers=self.headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            translated = data["translations"][0]["text"].strip()
                            results[lang_code] = translated
                            # logger.debug(f"<translation OK> [{lang_code}] {translated}")
                        else:
                            error_text = await resp.text()
                            # logger.warning(f"📡 Failed to translate to '{lang_code}' (HTTP {resp.status}): {error_text}")
                            results[lang_code] = None

                except Exception as e:
                    logger.error(f"🚨 Translation failed for '{lang_code}': {str(e)}", exc_info=True)
                    results[lang_code] = None

                # Optional: small delay to avoid rate limits
                await asyncio.sleep(0.1)

        return results



# Check and delete
class WordRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_unprocessed_word(self) -> Optional[Word]:
        """Get a word that hasn't been processed with AI yet (no meanings)"""
        try:
            # Fix: Use outerjoin to find words without meanings
            stmt = (
                select(Word)
                .outerjoin(WordMeaning, Word.id == WordMeaning.word_id)
                .where(WordMeaning.id.is_(None))
                .order_by(Word.id.asc())
                .limit(1)
            )

            logger.info(f"🔍 Executing query for unprocessed words")
            result = await self.db.execute(stmt)
            word = result.scalar_one_or_none()

            if word:
                logger.info(f"📝 Found unprocessed word: {word.text} (ID: {word.id})")
            else:
                logger.info("✅ No unprocessed words found")

            return word

        except Exception as e:
            logger.error(f"❌ Error fetching unprocessed word: {str(e)}", exc_info=True)
            return None

    async def get_or_create_category(self, category_name: str) -> Category:
        """Get existing category or create new one"""
        try:
            stmt = select(Category).where(Category.name == category_name)
            result = await self.db.execute(stmt)
            category = result.scalar_one_or_none()

            if not category:
                category = Category(name=category_name)
                self.db.add(category)
                await self.db.flush()
                # logger.info(f"📁 Created new category: {category_name}")
            else:
                pass
                # logger.info(f"📁 Using existing category: {category_name}")

            return category
        except Exception as e:
            logger.error(f"❌ Error in get_or_create_category: {str(e)}")
            raise

    async def save_word_analysis(self, word: Word, ai_response: WordAnalysisResponse) -> bool:
        """Save AI analysis results to database"""
        try:
            # logger.info(f"💾 Saving analysis for word: {word.text}")

            # Ensure we have the latest word object with relationships
            await self.db.refresh(word, ['categories'])

            # 1. Add categories to word
            for category_name in ai_response.categories:
                category = await self.get_or_create_category(category_name)
                if category not in word.categories:
                    word.categories.append(category)
                    logger.info(f"📂 Added category '{category_name}' to word '{word.text}'")

            # 2. Create word meanings with POS and definitions
            definitions_count = 0
            for pos_def in ai_response.pos_definitions:
                for definition in pos_def.definitions:
                    word_meaning = WordMeaning(
                        word_id=word.id,
                        pos=pos_def.pos,
                        definition=definition
                    )
                    self.db.add(word_meaning)
                    definitions_count += 1

            logger.info(f"📚 Added {definitions_count} definitions for {word.text}")

            # 3. Update word level based on frequency
            if word.frequency_rank <= 1000:
                word.level = "A1"
            elif word.frequency_rank <= 2500:
                word.level = "A2"
            elif word.frequency_rank <= 5000:
                word.level = "B1"
            elif word.frequency_rank <= 7500:
                word.level = "B2"
            else: word.level = "C1"


            logger.info(f"🎯 Set level '{word.level}' for word '{word.text}'")

            await self.db.commit()
            logger.info(f"✅ Successfully saved word analysis for: {word.text}")
            return True

        except Exception as e:
            await self.db.rollback()
            logger.error(f"❌ Error saving word analysis for {word.text}: {str(e)}", exc_info=True)
            return False

    async def word_needs_processing(self, word_id: int) -> bool:
        """Check if word already has meanings (already processed)"""
        try:
            stmt = (
                select(Word)
                .where(Word.id == word_id)
                .options(selectinload(Word.meanings))
            )
            result = await self.db.execute(stmt)
            word = result.scalar_one_or_none()
            return word is not None and len(word.meanings) == 0
        except Exception as e:
            logger.error(f"❌ Error checking word processing status: {str(e)}")
            return False

# Check and delete
class FetchEnglishWordsGenerateWithAIToPOSDefCat:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.word_repo = WordRepository(db)
        self.ai_service = AIService()

    async def generate_words_with_ai(self, limit: int = 1) -> Dict[str, Any]:
        """Generate AI content for words with better error handling"""
        try:
            processed_words = []
            errors = []

            for i in range(limit):
                word = await self.word_repo.get_unprocessed_word()
                if not word:
                    logger.info("✅ No more unprocessed words found")
                    return {
                        "success": True,
                        "message": "No unprocessed words found",
                        "processed": processed_words
                    }

                logger.info(f"🔍 [{i + 1}/{limit}] Processing word: {word.text} (ID: {word.id})")

                # Add delay to avoid rate limiting
                if i > 0:
                    await asyncio.sleep(1)  # 1 second delay between requests

                # Analyze word with AI
                ai_response = await self.ai_service.analyze_word(word.text)

                if not ai_response:
                    error_msg = f"AI analysis failed for: {word.text}"
                    logger.error(f"❌ {error_msg}")
                    errors.append(error_msg)
                    continue

                # Save analysis to database
                success = await self.word_repo.save_word_analysis(word, ai_response)

                if success:
                    processed_info = {
                        "word": word.text,
                        "word_id": word.id,
                        "categories": ai_response.categories,
                        "pos_count": len(ai_response.pos_definitions),
                        "definitions_count": sum(len(pos_def.definitions) for pos_def in ai_response.pos_definitions)
                    }
                    processed_words.append(processed_info)
                    logger.info(f"✅ Successfully processed: {word.text}")
                    logger.info(f"📊 Results: {processed_info}")
                else:
                    error_msg = f"Database save failed for: {word.text}"
                    logger.error(f"❌ {error_msg}")
                    errors.append(error_msg)

            summary = f"Processed {len(processed_words)} words, {len(errors)} errors"
            logger.info(f"📈 {summary}")

            return {
                "success": len(errors) == 0,
                "message": summary,
                "processed": processed_words,
                "errors": errors
            }

        except Exception as e:
            logger.error(f"💥 Critical error in generate_words_with_ai: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "processed": [],
                "errors": [str(e)]
            }

    async def close(self):
        await self.ai_service.close()

    # New added
    async def generate_single_word(self, word) -> Dict[str, Any]:
        """Generate AI content for a single word"""
        try:
            # logger.info(f"🔍 Processing single word: {word.text} (ID: {word.id})")

            # Analyze word with AI
            ai_response = await self.ai_service.analyze_word(word.text)

            if not ai_response:
                return {
                    "success": False,
                    "error": f"AI analysis failed for: {word.text}",
                    "word": word.text,
                    "word_id": word.id
                }

            # Save analysis to database
            success = await self.word_repo.save_word_analysis(word, ai_response)

            if success:
                result = {
                    "success": True,
                    "word": word.text,
                    "word_id": word.id,
                    "categories": ai_response.categories,
                    "pos_count": len(ai_response.pos_definitions),
                    "definitions_count": sum(len(pos_def.definitions) for pos_def in ai_response.pos_definitions)
                }
                # logger.info(f"✅ Successfully processed: {word.text}")
                return result
            else:
                return {
                    "success": False,
                    "error": f"Database save failed for: {word.text}",
                    "word": word.text,
                    "word_id": word.id
                }

        except Exception as e:
            logger.error(f"💥 Error processing single word {word.text}: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "word": word.text,
                "word_id": word.id
            }


























































################################################################## This is for spanish.txt

class CreateMainStructureForSpanishRepository:

    def __init__(self, db: AsyncSession):
        self.db = db


    async def insert_spanish_words_to_table(self):

        # Path to your Excel file
        excel_file_path = os.path.join(os.path.dirname(__file__), "spanish_top_10000.xlsx")

        try:
            # Read the Excel file
            df = pd.read_excel(excel_file_path, engine='openpyxl')

            # Check if the dataframe has the expected columns
            print("Excel file columns:", df.columns.tolist())
            print(f"Total words to process: {len(df)}")

            # Process each row and create Word objects
            words_to_add = []
            existing_words_count = 0

            for index, row in df.iterrows():
                # Adjust column names based on your Excel file structure
                # Assuming your Excel has columns: 'rank', 'word', 'frequency'
                word_text = str(row['word']).strip()
                frequency_rank = int(row['rank'])
                frequency = int(row['frequency'])

                # Check if word already exists in database
                existing_word = await self.db.execute(
                    select(Word).where(Word.text == word_text).where(Word.language_code == "es")
                )
                existing_word = existing_word.scalar_one_or_none()

                if existing_word:
                    existing_words_count += 1
                    continue  # Skip if word already exists

                # Determine level based on frequency rank (you can adjust this logic)
                level = self._determine_level(frequency_rank)

                # Create new Word object
                new_word = Word(
                    text=word_text,
                    language_code="es",  # Spanish
                    frequency_rank=frequency_rank,
                    level=level
                )

                words_to_add.append(new_word)

                # Batch insert every 100 words to avoid memory issues
                if len(words_to_add) >= 100:
                    self.db.add_all(words_to_add)
                    await self.db.commit()
                    words_to_add = []
                    print(f"Processed {index + 1} words...")

            # Insert any remaining words
            if words_to_add:
                self.db.add_all(words_to_add)
                await self.db.commit()

            print(f"Successfully inserted {len(df) - existing_words_count} Spanish words")
            print(f"Skipped {existing_words_count} existing words")

            return {
                'msg': 'Spanish words inserted successfully',
                'inserted_count': len(df) - existing_words_count,
                'skipped_count': existing_words_count
            }

        except FileNotFoundError:
            print(f"Error: Excel file not found at {excel_file_path}")
            return {'error': 'Excel file not found'}
        except Exception as e:
            await self.db.rollback()
            print(f"Error: {str(e)}")
            return {'error': str(e)}

    def _determine_level(self, frequency_rank: int) -> str:
        """Determine CEFR level based on frequency rank"""
        if frequency_rank <= 1000:
            return "A1"
        elif frequency_rank <= 2000:
            return "A2"
        elif frequency_rank <= 4000:
            return "B1"
        elif frequency_rank <= 6000:
            return "B2"
        elif frequency_rank <= 8000:
            return "C1"
        else:
            return "C2"


    def fetch_words_from_spanish(self):
        """Download and process top Spanish words from GitHub"""
        url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt"

        try:
            response = requests.get(url)
            lines = response.text.split('\n')

            data = []
            for i, line in enumerate(lines[:10000]):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        frequency = int(parts[1])
                        rank = i + 1
                        data.append([rank, word, frequency])

            df = pd.DataFrame(data, columns=['rank', 'word', 'frequency'])

            # Save to Excel
            excel_path = os.path.join(os.path.dirname(__file__), "spanish_top_10000.xlsx")
            df.to_excel(excel_path, index=False, engine='openpyxl')

            print(f"Created Excel with top 10,000 Spanish words at: {excel_path}")
            return df

        except Exception as e:
            print(f"Error: {str(e)}")
            return None


class TranslateSpanishWordsToEnglish:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.api_key = os.getenv("YANDEX_TRANSLATE_API_SECRET_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.translate_url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        self.batch_size = 100  # Process words in batches
        self.translate_batch_size = 10  # Yandex API batch limit

    async def translate_spanish_words_to_english(self):
        try:
            # Fetch Spanish words that don't have English translations yet
            spanish_words = await self._get_spanish_words_without_translations()

            if not spanish_words:
                return {'msg': 'No Spanish words need translation'}

            print(f"Found {len(spanish_words)} Spanish words to translate")

            # Translate in batches to avoid API rate limits
            translated_count = 0
            for i in range(0, len(spanish_words), self.translate_batch_size):
                batch = spanish_words[i:i + self.translate_batch_size]
                print(f'............................. batch is {batch}')
                translations = await self._translate_batch(batch)

                if translations:
                    await self._save_translations(translations)
                    translated_count += len(translations)

                print(f"Translated {min(i + self.translate_batch_size, len(spanish_words))}/{len(spanish_words)} words")

                # Add delay to avoid hitting rate limits
                await asyncio.sleep(0.1)

            return {
                'msg': 'Translation completed',
                'translated_count': translated_count,
                'total_words': len(spanish_words)
            }

        except Exception as e:
            print(f"Translation error: {str(e)}")
            return {'error': str(e)}

    async def _get_spanish_words_without_translations(self) -> List[Word]:
        """Fetch Spanish words that don't have English translations yet"""
        query = text("""
        SELECT w.* FROM words w
        WHERE w.language_code = 'es'
        AND NOT EXISTS (
            SELECT 1 FROM translations t 
            WHERE t.source_word_id = w.id 
            AND t.target_language_code = 'en'
        )
        ORDER BY w.frequency_rank
        LIMIT 7000
        """)

        result = await self.db.execute(query)
        words = [Word(id=row[0], text=row[1], language_code=row[2], frequency_rank=row[3], level=row[4])
                 for row in result.fetchall()]
        return words

    # Alternative method using SQLAlchemy ORM instead of raw SQL
    async def _get_spanish_words_without_translations_orm(self) -> List[Word]:
        """Alternative method using ORM queries instead of raw SQL"""
        # Get all Spanish words
        spanish_words_query = select(Word).where(Word.language_code == 'es')
        result = await self.db.execute(spanish_words_query)
        all_spanish_words = result.scalars().all()

        # Filter words that don't have English translations
        words_without_translations = []
        for word in all_spanish_words:
            # Check if translation exists using ORM
            translation_query = select(Translation).where(
                Translation.source_word_id == word.id,
                Translation.target_language_code == 'en'
            )
            translation_result = await self.db.execute(translation_query)
            existing_translation = translation_result.scalar_one_or_none()

            if not existing_translation:
                words_without_translations.append(word)

        return words_without_translations

    async def _translate_batch(self, words: List[Word]) -> List[dict]:
        """Translate a batch of words using Yandex API"""
        if not self.api_key or not self.folder_id:
            raise ValueError("Yandex API credentials not configured")

        texts = [word.text for word in words]

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "folderId": self.folder_id,
            "texts": texts,
            "targetLanguageCode": "en",
            "sourceLanguageCode": "es"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.translate_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        translations = data.get('translations', [])

                        # Map translations back to words
                        result = []
                        for word, translation in zip(words, translations):
                            result.append({
                                'source_word': word,
                                'translated_text': translation.get('text', ''),
                                'detected_language': translation.get('detectedLanguageCode', 'es')
                            })
                        return result
                    else:
                        error_text = await response.text()
                        print(f"Yandex API error {response.status}: {error_text}")
                        return []

        except Exception as e:
            print(f"API call error: {str(e)}")
            return []

    async def _save_translations(self, translations: List[dict]):
        """Save translations to database"""
        try:
            translation_objects = []

            for translation_data in translations:
                if translation_data['translated_text']:  # Only save valid translations
                    translation = Translation(
                        source_word_id=translation_data['source_word'].id,
                        target_language_code='en',
                        translated_text=translation_data['translated_text']
                    )
                    translation_objects.append(translation)

            if translation_objects:
                self.db.add_all(translation_objects)
                await self.db.commit()

        except Exception as e:
            await self.db.rollback()
            print(f"Error saving translations: {str(e)}")
            raise

    # More efficient ORM method using subquery
    async def _get_spanish_words_without_translations_efficient(self) -> List[Word]:
        """More efficient method using subquery"""
        # Create subquery for words that already have translations
        translated_words_subquery = select(Translation.source_word_id).where(
            Translation.target_language_code == 'en'
        ).subquery()

        # Main query to get Spanish words without translations
        query = select(Word).where(
            Word.language_code == 'es',
            ~Word.id.in_(select(translated_words_subquery))
        ).order_by(Word.frequency_rank).offset(3000)

        result = await self.db.execute(query)
        words = result.scalars().all()
        return words

    async def translate_all_spanish_words(self, limit: Optional[int] = None):
        """Main method to translate all Spanish words"""
        try:
            # Use the efficient ORM method
            spanish_words = await self._get_spanish_words_without_translations_efficient()

            if not spanish_words:
                return {'msg': 'No Spanish words need translation'}

            if limit:
                spanish_words = spanish_words[:limit]

            print(f"Translating {len(spanish_words)} Spanish words to English")

            translated_count = 0
            for i in range(0, len(spanish_words), self.translate_batch_size):
                batch = spanish_words[i:i + self.translate_batch_size]
                translations = await self._translate_batch(batch)

                if translations:
                    await self._save_translations(translations)
                    translated_count += len(translations)

                progress = min(i + self.translate_batch_size, len(spanish_words))
                print(f"Progress: {progress}/{len(spanish_words)} words translated")

                await asyncio.sleep(0.1)  # Rate limiting

            return {
                'msg': 'Translation completed successfully',
                'words_processed': len(spanish_words),
                'translations_created': translated_count
            }

        except Exception as e:
            print(f"Translation error: {str(e)}")
            await self.db.rollback()
            return {'error': str(e)}


class GenerateSentencesSpanishAndTranslate:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.api_key = os.getenv("YANDEX_LANGMODEL_API_SECRET_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls

    async def translate_sentence(self, text: str) -> str:
        """Translate a Spanish sentence into English"""
        headers = {
            "Authorization": f"Api-Key {os.getenv('YANDEX_TRANSLATE_API_SECRET_KEY')}"
        }
        json_data = {
            "folder_id": self.folder_id,
            "texts": [text],
            "sourceLanguageCode": "es",
            "targetLanguageCode": "en"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://translate.api.cloud.yandex.net/translate/v2/translate",
                    headers=headers,
                    json=json_data
            ) as response:
                if response.status != 200:
                    full_response = await response.text()
                    raise Exception(f"API error {response.status}: {full_response}")

                data = await response.json()
                return data['translations'][0]['text']

    async def _get_spanish_words_batch(self, limit: int = 30, offset: int = 0) -> List[Word]:
        """Fetch Spanish words with consistent ordering"""
        try:
            result = await self.db.execute(
                select(Word)
                .where(Word.language_code == "es")
                .order_by(Word.frequency_rank.asc())  # Get most common words first
                .limit(limit)
                .offset(offset)
            )

            words = result.scalars().all()
            print(f"Fetched {len(words)} Spanish words")
            return words
        except Exception as e:
            print(f"ERROR in _get_spanish_words_batch: {str(e)}")
            return []

    async def _process_word(self, word: Word) -> bool:
        """Process a single Spanish word"""
        try:
            print(f"Processing Spanish word ID {word.id}: {word.text}")

            # Generate sentences in Spanish
            sentences = await self.generate_spanish_sentences(word.text)
            if not sentences:
                print(f"No sentences generated for {word.text}")
                return False

            # Save to DB
            await self._save_sentences(word, sentences)
            return True

        except Exception as e:
            print(f"ERROR processing {word.text}: {str(e)}")
            return False

    async def main_func(self):
        """Main execution loop - generate sentences for first 30 Spanish words"""
        try:
            print("Starting processing of Spanish words...")

            # Fetch first 30 Spanish words
            words = await self._get_spanish_words_batch(limit=30, offset=0)
            if not words:
                print("No Spanish words found!")
                return {"status": "failed", "reason": "no words found"}

            success_count = 0
            processed_ids = []

            for word in words:
                success = await self._process_word(word)
                if success:
                    success_count += 1
                    processed_ids.append(word.id)
                await asyncio.sleep(1)  # avoid rate limit

            return {
                "status": "completed",
                "total_words": len(words),
                "processed": success_count,
                "failed": len(words) - success_count,
                "processed_ids": processed_ids
            }

        except Exception as e:
            print(f"FATAL ERROR: {str(e)}")
            return {"status": "failed", "error": str(e)}

    async def _save_sentences(self, word: Word, sentences: List[str]):
        """Save Spanish sentences, their English translations, and link to the word"""
        for sentence_text in sentences:
            # Save Spanish sentence
            sentence = Sentence(
                text=sentence_text,
                language_code="es"
            )
            self.db.add(sentence)
            await self.db.flush()

            # Link to word
            self.db.add(SentenceWord(
                sentence_id=sentence.id,
                word_id=word.id
            ))

            # Translate to English
            try:
                en_translation = await self.translate_sentence(sentence_text)
                self.db.add(SentenceTranslation(
                    source_sentence_id=sentence.id,
                    language_code="en",
                    translated_text=en_translation
                ))
            except Exception as e:
                print(f"ERROR translating '{sentence_text}': {e}")

        await self.db.commit()

    async def generate_spanish_sentences(self, word: str) -> List[str]:
        """Generate exactly 5 Spanish sentences using Yandex GPT"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }

            prompt = {
                "modelUri": f"gpt://{self.folder_id}/yandexgpt",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.6,
                    "maxTokens": "2000"
                },
                "messages": [
                    {
                        "role": "system",
                        "text": "Eres un asistente útil que genera oraciones de ejemplo para palabras en español."
                    },
                    {
                        "role": "user",
                        "text": f"Escribe exactamente 5 oraciones diferentes en español usando la palabra '{word}'. "
                                f"Cada oración debe mostrar un significado/contexto diferente. "
                                f"Separa las oraciones con el símbolo '|'. "
                                f"Devuelve solo las oraciones, sin texto adicional."
                    }
                ]
            }

            try:
                async with session.post(self.api_url, json=prompt, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()

                    sentences_text = result["result"]["alternatives"][0]["message"]["text"]
                    sentences = [s.strip() for s in sentences_text.split("|") if s.strip()]

                    # Ensure we get exactly 5 sentences
                    if len(sentences) > 5:
                        sentences = sentences[:5]
                    elif len(sentences) < 5:
                        # If we got fewer than 5, pad with empty strings
                        sentences.extend([""] * (5 - len(sentences)))

                    return sentences

            except Exception as e:
                print(f"API Error for Spanish word {word}: {str(e)}")
                return []

    async def generate_for_next_batch(self, batch_size: int = 30, offset: int = 0):
        """Generate sentences for a specific batch of words"""
        words = await self._get_spanish_words_batch(limit=batch_size, offset=offset)

        if not words:
            return {"status": "no_more_words", "offset": offset}

        success_count = 0
        for word in words:
            success = await self._process_word(word)
            if success:
                success_count += 1
            await asyncio.sleep(1)

        next_offset = offset + batch_size
        return {
            "status": "completed",
            "processed": success_count,
            "total": len(words),
            "next_offset": next_offset
        }


################################################################## This is for russian

class CreateMainStructureForRussianRepository:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def insert_russian_words_to_table(self):
        # 1️⃣ Build the correct file path (same folder as this file)
        file_path = os.path.join(os.path.dirname(__file__), "russian_words.xlsx")

        # 2️⃣ Read Excel file
        df = pd.read_excel(file_path)

        # Expecting columns: russian, translation, lang_code, frequency_rank, level
        for _, row in df.iterrows():


            ru_word = str(row["russian"]).strip()
            translation_text = str(row["translation"]).strip()
            target_lang = "en"
            frequency_rank = int(row["rank"])
            level = None

            # 3️⃣ Check if Russian word already exists
            existing_word = await self.db.scalar(
                select(Word).where(Word.text == ru_word, Word.language_code == "ru")
            )
            if not existing_word:
                existing_word = Word(
                    text=ru_word,
                    language_code="ru",
                    frequency_rank=frequency_rank,
                    level=level
                )
                self.db.add(existing_word)
                await self.db.flush()  # gets ID without committing

            # 4️⃣ Check if translation exists
            existing_translation = await self.db.scalar(
                select(Translation).where(
                    Translation.source_word_id == existing_word.id,
                    Translation.target_language_code == target_lang,
                    Translation.translated_text == translation_text
                )
            )
            if not existing_translation:
                translation = Translation(
                    source_word_id=existing_word.id,
                    target_language_code=target_lang,
                    translated_text=translation_text
                )
                self.db.add(translation)

        # 5️⃣ Commit after processing all rows
        await self.db.commit()


class FetchRussianWordsAndGenerateSentenceAndTranslateToEnglish:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.api_key = os.getenv("YANDEX_LANGMODEL_API_SECRET_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls

    async def translate_sentence(self, text: str) -> str:
        """Translate a Russian sentence into English"""
        headers = {
            "Authorization": f"Api-Key {os.getenv('YANDEX_TRANSLATE_API_SECRET_KEY')}"
        }
        json_data = {
            "folder_id": self.folder_id,
            "texts": [text],
            "sourceLanguageCode": "ru",
            "targetLanguageCode": "en"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://translate.api.cloud.yandex.net/translate/v2/translate",
                headers=headers,
                json=json_data
            ) as response:
                if response.status != 200:
                    full_response = await response.text()
                    raise Exception(f"API error {response.status}: {full_response}")

                data = await response.json()
                return data['translations'][0]['text']

    async def _get_words_batch(self, offset: int, limit: int) -> List[Word]:
        """Fetch Russian words with consistent ordering"""
        try:

            result = await self.db.execute(
                select(Word)
                .where(Word.id.between(18666, 19166)) # Just call the function.
                .order_by(Word.id.asc())
                .limit(limit)
                .offset(offset)
            )

            words = result.scalars().all()
            print(f"DEBUG: Fetched {len(words)} RU words (IDs: {[w.id for w in words]})")
            return words
        except Exception as e:
            print(f"ERROR in _get_words_batch: {str(e)}")
            return []

    async def _process_word(self, word: Word) -> bool:
        """Process a single word"""
        try:
            print(f"Processing RU word ID {word.id}: {word.text}")

            # Generate sentences in Russian
            sentences = await self.generate_sentences(word.text)
            if not sentences:
                print(f"No sentences generated for {word.text}")
                return False

            # Save to DB
            await self._save_sentences(word, sentences)
            return True

        except Exception as e:
            print(f"ERROR processing {word.text}: {str(e)}")
            return False

    async def main_func(self):
        """Main execution loop"""
        try:
            print("Starting processing of Russian words...")

            # Fetch first 2 Russian words
            words = await self._get_words_batch(0, 100)
            if not words:
                print("No Russian words found!")
                return {"status": "failed", "reason": "no words found"}

            success_count = 0
            processed_ids = []

            for word in words:
                success = await self._process_word(word)
                if success:
                    success_count += 1
                    processed_ids.append(word.id)
                await asyncio.sleep(1)  # avoid rate limit

            return {
                "status": "completed",
                "total_words": len(words),
                "processed": success_count,
                "failed": len(words) - success_count,
                "processed_ids": processed_ids
            }

        except Exception as e:
            print(f"FATAL ERROR: {str(e)}")
            return {"status": "failed", "error": str(e)}

    async def _save_sentences(self, word: Word, sentences: List[str]):
        """Save RU sentences, their EN translations, and link to the word"""
        for sentence_text in sentences:
            # Save RU sentence
            sentence = Sentence(
                text=sentence_text,
                language_code="ru"
            )
            self.db.add(sentence)
            await self.db.flush()

            # Link to word
            self.db.add(SentenceWord(
                sentence_id=sentence.id,
                word_id=word.id
            ))

            # Translate to English
            try:
                en_translation = await self.translate_sentence(sentence_text)
                self.db.add(SentenceTranslation(
                    source_sentence_id=sentence.id,
                    language_code="en",
                    translated_text=en_translation
                ))
            except Exception as e:
                print(f"ERROR translating '{sentence_text}': {e}")

        await self.db.commit()

    async def generate_sentences(self, word: str) -> List[str]:
        """Generate exactly 5 Russian sentences using Yandex GPT"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }

            prompt = {
                "modelUri": f"gpt://{self.folder_id}/yandexgpt",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.6,
                    "maxTokens": "2000"
                },
                "messages": [
                    {
                        "role": "system",
                        "text": "You are a helpful assistant that generates example sentences for Russian words."
                    },
                    {
                        "role": "user",
                        "text": f"Составь ровно 5 разных предложений на русском языке, используя слово '{word}'. "
                                f"Каждое предложение должно показывать разное значение/контекст. "
                                f"Раздели предложения символом '|'. "
                                f"Верни только предложения, без лишнего текста."
                    }
                ]
            }

            try:
                async with session.post(self.api_url, json=prompt, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()

                    sentences_text = result["result"]["alternatives"][0]["message"]["text"]
                    sentences = [s.strip() for s in sentences_text.split("|") if s.strip()]

                    if len(sentences) != 5:
                        raise ValueError(f"Expected 5 sentences, got {len(sentences)}")

                    return sentences[:5]

            except Exception as e:
                print(f"API Error for RU word {word}: {str(e)}")
                return []






################################################################## This is For english structure

# Generate and insert top 10.000 English words in Database
class CreateMainStructureRepositoryForEnglishLanguage:
    def __init__(self, db):
        self.db = db

    async def create_main_structure(self):

        # 2. Add English words
        await self.create_top_10000_words_in_english()

        return "Database initialized successfully"


    async def create_top_10000_words_in_english(self):

        # Step 1: Get the top 10,000 English words
        words = await self._fetch_english_wordlist()

        # Step 2: Add to database with frequency ranking
        for rank, word_data in enumerate(words, start=1):
            word = await self.db.scalar(
                select(Word)
                .where(
                    Word.text == word_data["word"],
                    Word.language_code == "en"
                )
            )

            if not word:
                self.db.add(Word(
                    text=word_data["word"],
                    language_code="en",
                    frequency_rank=rank
                ))

            # Commit every 500 words to avoid huge transactions
            if rank % 500 == 0:
                await self.db.commit()

        await self.db.commit()

    async def _fetch_english_wordlist(self):
        """Fetches top 10k English words with POS tags"""
        # url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
        #
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(url) as response:
        #         content = await response.text()

        # Basic processing - you might want to enhance this

        words = []



        # for word in content.splitlines()[:10000]:  # First 10k words
        #     words.append({
        #         "word": word.strip(),
        #     })

        try:
            with open("english_words.txt", 'r', encoding='utf-8') as file:
                for line in file:
                    word = line.strip()
                    if word:  # Skip empty lines
                        words.append({"word": word})

                        # Stop after 10,000 words if your file is larger
                        if len(words) >= 10000:
                            break
        except FileNotFoundError:
            raise Exception(f"Word list file not found at")

        return words



# Get the Rapid api and take the pos for each word
class UpdateMainStructureRepositoryForEnglishLanguage:
    def __init__(self, db_session):
        self.db = db_session  # Already an AsyncSession
        self.rapidapi_key = "814d977a29mshf4ff7ed76e201ffp19de4fjsn0c7133b7592b"
        self.rapidapi_host = "wordsapiv1.p.rapidapi.com"
        self.batch_size = 50

    async def update_top_10000_words_in_english_add_meanings(self):
        try:

            words = await self._fetch_unprocessed_words(limit=10000)
            if not words:
                return "No unprocessed words found"

            success_count = 0
            for i in range(0, len(words), self.batch_size):
                batch = words[i:i + self.batch_size]
                batch_success = await self._process_batch(batch)
                success_count += batch_success

                try:
                    await self.db.flush()  # Flush instead of commit to keep transaction open
                    print(f"Flushed batch {i // self.batch_size + 1}")
                except Exception as e:
                    await self.db.rollback()
                    print(f"Flush failed: {str(e)}")
                    continue

                await asyncio.sleep(1.5)  # Rate limiting

            await self.db.commit()  # Commit all at once at the end
            return f"Successfully processed {success_count} words"

        except Exception as e:
            await self.db.rollback()
            print(f"Critical error: {str(e)}")
            raise


    # async def import_words_from_file(self):
    #     try:
    #         # 1. Read words from file
    #         word_file = Path("english_words.txt")
    #         if not word_file.exists():
    #             raise HTTPException(status_code=404, detail="Word file not found")
    #
    #         words = word_file.read_text().splitlines()
    #         total_words = len(words)
    #
    #         # 2. Prepare batch insert
    #         inserted_count = 0
    #         batch_size = 100  # Adjust based on your DB performance
    #
    #         for i in range(0, total_words, batch_size):
    #             batch = words[i:i + batch_size]
    #
    #             # 3. Check existing words in batch
    #             existing = await self.db.execute(
    #                 select(Word.text).where(Word.text.in_(batch))
    #             )
    #             existing_words = {w[0] for w in existing.all()}
    #
    #             # 4. Prepare new words
    #             new_words = [
    #                 Word(
    #                     text=word,
    #                     language_code="en",
    #                     frequency_rank=i + 1,  # 1-based index
    #                     level=None,
    #                 )
    #                 for i, word in enumerate(batch, start=i)
    #                 if word not in existing_words
    #             ]
    #
    #             # 5. Bulk insert
    #             self.db.add_all(new_words)
    #             await self.db.commit()
    #             inserted_count += len(new_words)
    #             print(f"Inserted batch {i // batch_size + 1}: +{len(new_words)} words")
    #
    #         return {
    #             "status": "success",
    #             "inserted": inserted_count,
    #             "duplicates": total_words - inserted_count,
    #             "total": total_words
    #         }
    #
    #     except Exception as e:
    #         await self.db.rollback()
    #         raise HTTPException(
    #             status_code=500,
    #             detail=f"Import failed: {str(e)}"
    #         )


    async def _process_batch(self, batch):
        success_count = 0
        async with aiohttp.ClientSession() as session:
            for word in batch:
                try:
                    data = await self._fetch_word_details(session, word.text)
                    if not data:
                        print(f"No API data for {word.text}")
                        continue

                    await self._update_word_level(word, data)
                    await self._process_meanings(word, data)

                    success_count += 1
                    print(f"Processed {word.text}")

                except Exception as e:
                    print(f"Failed to process {word.text}: {str(e)}")
                    continue

        return success_count

    async def _fetch_word_details(self, session, word_text):
        url = f'https://{self.rapidapi_host}/words/{word_text}'
        headers = {
            'x-rapidapi-host': self.rapidapi_host,
            'x-rapidapi-key': self.rapidapi_key
        }

        try:
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"API success for {word_text}")
                    return data
                else:
                    print(f"API error for {word_text}: HTTP {response.status}")
                    return None
        except asyncio.TimeoutError:
            print(f"Timeout fetching {word_text}")
            return None
        except Exception as e:
            print(f"Network error for {word_text}: {str(e)}")
            return None

    async def _fetch_unprocessed_words(self, limit=10000):
        result = await self.db.execute(
            select(Word)
            .where(
                Word.language_code == "en",
                Word.level == None
            )
            .order_by(Word.frequency_rank)
            .limit(limit)
            .options(selectinload(Word.meanings))
        )
        return result.scalars().all()

    async def _update_word_level(self, word, api_data):
        frequency_data = api_data.get('frequency', {})
        zipf = 0
        if type(frequency_data) == {}:
            zipf = 0
        else:
            zipf = float(frequency_data)
        word.level = self._zipf_to_level(zipf)
        word.zipf_frequency = zipf
        word.updated_at = datetime.utcnow()
        self.db.add(word)
        print(f"Updated {word.text}: level={word.level}, zipf={zipf}")

    async def _process_meanings(self, word, api_data):
        meanings_data = api_data.get('results', [])
        for meaning_data in meanings_data:
            pos = self._normalize_pos(meaning_data.get('partOfSpeech', 'other'))
            # definition = meaning_data.get('definition', '')[:500]
            # definition = None
            example = self._clean_example(meaning_data.get('examples', [None])[0])

            existing = next((m for m in word.meanings if m.pos == pos), None)
            if not existing:
                meaning = WordMeaning(
                    word_id=word.id,
                    pos=pos,
                    # definition=definition,
                    example=example
                )
                self.db.add(meaning)
                print(f"Added meaning for {word.text} as {pos}")

    def _zipf_to_level(self, zipf_score):
        if zipf_score >= 6: return "A1"
        elif zipf_score >= 5.5: return "A2"
        elif zipf_score >= 5: return "B1"
        elif zipf_score >= 4: return "B2"
        elif zipf_score >= 3: return "C1"
        return "C2"

    @staticmethod
    def _normalize_pos(pos):
        pos = pos.lower()
        if pos in ['noun', 'n']: return 'noun'
        if pos in ['verb', 'v']: return 'verb'
        if pos in ['adjective', 'adj']: return 'adj'
        if pos in ['adverb', 'adv']: return 'adv'
        return 'other'

    @staticmethod
    def _clean_example(example):
        if not example:
            return None
        return example[:500]



# fetch english words and generate a sentences
class GenerateEnglishSentencesForEachWord:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.api_key = os.getenv("YANDEX_LANGMODEL_API_SECRET_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls


    async def translate_sentence(self, text: str) -> str:
        print('')
        """Translate an English sentence into Russian"""
        headers = {
            "Authorization": f"Api-Key {os.getenv('YANDEX_TRANSLATE_API_SECRET_KEY')}"
        }
        json_data = {
            "folder_id": self.folder_id,
            "texts": [text],
            "sourceLanguageCode": "en",
            "targetLanguageCode": "ru"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post("https://translate.api.cloud.yandex.net/translate/v2/translate", headers=headers,
                                    json=json_data) as response:
                if response.status != 200:
                    full_response = await response.text()
                    raise Exception(f"API error {response.status}: {full_response}")

                data = await response.json()
                return data['translations'][0]['text']

    async def _get_words_batch(self, offset: int, limit: int) -> List[Word]:
        """Fetch words with consistent ordering"""
        try:
            result = await self.db.execute(
                select(Word)
                .where(Word.id.between(101,1000)) # Need to start to generate a sentences from 184 to 200
                .order_by(Word.id.asc())

                .limit(limit)
                .offset(offset)
            )
            words = result.scalars().all()
            print(f"DEBUG: Fetched {len(words)} words (IDs: {[w.id for w in words]})")
            return words
        except Exception as e:
            print(f"ERROR in _get_words_batch: {str(e)}")
            return []

    async def _process_word(self, word: Word) -> bool:
        """Process a single word with proper error handling"""
        try:
            print(f"Processing word ID {word.id}: {word.text}")

            # Generate sentences
            sentences = await self.generate_sentences(word.text)
            if not sentences:
                print(f"No sentences generated for {word.text}")
                return False

            # Save to database
            await self._save_sentences(word, sentences)
            return True

        except Exception as e:
            print(f"ERROR processing {word.text}: {str(e)}")
            return False

    async def main_func(self):
        """Process words with robust error handling"""
        try:
            print("Starting processing...")

            # Get all target words first
            words = await self._get_words_batch(0, 1000)
            if not words:
                print("No words found!")
                return {"status": "failed", "reason": "no words found"}

            # Process with concurrency control
            success_count = 0
            processed_ids = []

            for word in words:
                success = await self._process_word(word)
                if success:
                    success_count += 1
                    processed_ids.append(word.id)
                # Small delay between words to avoid rate limiting
                await asyncio.sleep(1)

            return {
                "status": "completed",
                "total_words": len(words),
                "processed": success_count,
                "failed": len(words) - success_count,
                "processed_ids": processed_ids  # Now using the list we built
            }

        except Exception as e:
            print(f"FATAL ERROR: {str(e)}")
            return {"status": "failed", "error": str(e)}


    async def _get_word_with_meanings(self, word_text: str):
        """Get word with its meanings"""
        result = await self.db.execute(
            select(Word)
            .where(Word.text == word_text)
            .options(selectinload(Word.meanings))
        )
        return result.scalar()

    async def _save_sentences(self, word: Word, sentences: List[str]):
        """Save EN sentences and their RU translations, and link them to the word"""
        for sentence_text in sentences:
            # 1. Save EN sentence
            sentence = Sentence(
                text=sentence_text,
                language_code="en"
            )
            self.db.add(sentence)
            await self.db.flush()  # get sentence.id

            # 2. Link sentence to word
            sentence_word = SentenceWord(
                sentence_id=sentence.id,
                word_id=word.id
            )
            self.db.add(sentence_word)

            # 3. Translate to Russian
            try:
                ru_translation = await self.translate_sentence(sentence_text)

                sentence_translation = SentenceTranslation(
                    source_sentence_id=sentence.id,
                    language_code="ru",
                    translated_text=ru_translation
                )
                self.db.add(sentence_translation)

            except Exception as e:
                print(f"ERROR translating sentence '{sentence_text}': {e}")
                return

        await self.db.commit()


    async def generate_sentences(self, word: str) -> List[str]:
        """Generate sentences using Yandex GPT API"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }

            prompt = {
                "modelUri": f"gpt://{self.folder_id}/yandexgpt",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.6,
                    "maxTokens": "2000"
                },
                "messages": [
                    {
                        "role": "system",
                        "text": "You are a helpful assistant that generates example sentences for English words."
                    },
                    {
                        "role": "user",
                        "text": f"Generate exactly 5 different example sentences using the word '{word}'. "
                                f"Each sentence should demonstrate a different usage/meaning. "
                                f"Separate each sentence with a '|' character. "
                                f"Only return the sentences, nothing else."
                    }
                ]
            }

            try:
                async with session.post(self.api_url, json=prompt, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()

                    sentences_text = result["result"]["alternatives"][0]["message"]["text"]
                    sentences = [s.strip() for s in sentences_text.split("|") if s.strip()]

                    if len(sentences) != 5:
                        raise ValueError(f"Expected 5 sentences, got {len(sentences)}")

                    return sentences[:5]

            except Exception as e:
                print(f"API Error for word {word}: {str(e)}")
                return []



# Translate all the words english to russian
class TranslateAllENWordsToRU:

    def __init__(self, db: AsyncSession):
        self.db = db
        self.api_key = os.getenv("YANDEX_TRANSLATE_API_SECRET_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.translate_url = "https://translate.api.cloud.yandex.net/translate/v2/translate"

    async def fetch_english_words_without_ru_translation(self):

        # Query English words that do not have Russian translation yet
        subquery = select(Translation.source_word_id).where(Translation.target_language_code == "ru")

        result = await self.db.execute(
            select(Word)
            .where(
                Word.language_code == "en",
                Word.id > 1001,  # ⬅️ Start from ID 5
                Word.id <= 9909,  # ⬅️ End at ID 1000
                ~Word.id.in_(subquery)
            )
            .order_by(Word.id)
        )
        return result.scalars().all()

    async def translate_word(self, text: str) -> str:
        headers = {
            "Authorization": f"Api-Key {self.api_key}"
        }
        json_data = {
            "folder_id": self.folder_id,
            "texts": [text],
            "sourceLanguageCode": "en",
            "targetLanguageCode": "ru"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.translate_url, headers=headers, json=json_data) as response:
                data = await response.json()
                return data['translations'][0]['text']

    async def save_translation(self, word_id: int, translated_text: str):
        translation = Translation(
            source_word_id=word_id,
            target_language_code="ru",
            translated_text=translated_text
        )
        self.db.add(translation)

    async def main_func(self):
        words = await self.fetch_english_words_without_ru_translation()
        print(f"Translating {len(words)} words...")

        for word in words:
            try:
                translated = await self.translate_word(word.text)
                await self.save_translation(word.id, translated)
            except Exception as e:
                print(f"Error translating '{word.text}': {e}")

        await self.db.commit()
        return {'msg': f'Translated {len(words)} words to Russian'}



# Removing Duplicates from POS
class RemoveDuplicatePosFromEnglish:

    def __init__(self, db):
        self.db = db

    async def find_duplicate_meanings(self):
        # Query to find word_id and pos combinations with counts > 1
        duplicate_query = select(
            WordMeaning.word_id,
            WordMeaning.pos,
            func.count(WordMeaning.id).label('count')
        ).group_by(
            WordMeaning.word_id,
            WordMeaning.pos
        ).having(
            func.count(WordMeaning.id) > 1
        )

        result = await self.db.execute(duplicate_query)
        return result.all()

    async def remove_duplicate_meanings(self):
        # Find all duplicate groups
        duplicates = await self.find_duplicate_meanings()
        total_deleted = 0

        for word_id, pos, count in duplicates:
            print(f"Processing word_id {word_id}, pos '{pos}' ({count} duplicates)")

            # Get all meanings for this word+pos combination
            meanings_query = select(WordMeaning).where(
                WordMeaning.word_id == word_id,
                WordMeaning.pos == pos
            ).order_by(
                WordMeaning.id  # Keeps the oldest record (first created)
            )

            meanings_result = await self.db.execute(meanings_query)
            meanings = meanings_result.scalars().all()

            # Keep the first one, delete the rest
            for meaning in meanings[1:]:
                await self.db.delete(meaning)
                total_deleted += 1

            await self.db.commit()

        return {"message": f"Deleted {total_deleted} duplicate meanings", "total_duplicate_groups": len(duplicates)}



# Fix english pos others
class FixPosFunctionality:

    def __init__(self, db):
        self.db = db
        self.pos_mapping = {
            'you': 'pronoun',
            'to': 'preposition',
            'the': 'article',
            'and': 'conjunction',
            'that': 'pronoun',
            'of': 'preposition',
            'this': 'pronoun',
            'for': 'preposition',
            'with': 'preposition',
            'if': 'conjunction',
            'her': 'pronoun',
            'she': 'pronoun',
            'him': 'pronoun',
            'they': 'pronoun',
            'from': 'preposition',
            'because': 'conjunction',
            'our': 'determiner',
            'into': 'preposition',
            'than': 'conjunction',
            'yourself': 'pronoun',
            'myself': 'pronoun',
            'since': 'preposition',
            'until': 'preposition',
            'without': 'preposition',
            'against': 'preposition',
            'them': 'pronoun',
            'unless': 'conjunction',
            'himself': 'pronoun',
            'herself': 'pronoun',
            'during': 'preposition',
            'whose': 'pronoun',
            'itself': 'pronoun',
            'whoever': 'pronoun',
            'themselves': 'pronoun',
            'upon': 'preposition',
            'whom': 'pronoun',
            'it': 'pronoun',
            'beside': 'preposition',
            'via': 'preposition',
            'nor': 'conjunction',
            'per': 'preposition',
            'ourselves': 'pronoun',
            'versus': 'preposition',
            'oneself': 'pronoun',
            'onto': 'preposition',
            'toward': 'preposition',
            'yourselves': 'pronoun'
        }

    async def find_others_and_fix_it(self):
        updated_counts = {}
        total_updated = 0

        for word_text, new_pos in self.pos_mapping.items():
            stmt = (
                update(WordMeaning)
                .where(WordMeaning.pos == "other")
                .where(WordMeaning.word_id == select(Word.id).where(Word.text == word_text).scalar_subquery())
                .values(pos=new_pos)
                .execution_options(synchronize_session=False)
            )
            result = await self.db.execute(stmt)
            updated = result.rowcount or 0
            if updated:
                updated_counts[word_text] = updated
                total_updated += updated

        await self.db.commit()

        return {
            "message": "POS tags updated",
            "updated_counts": updated_counts,
            "total_updated": total_updated
        }



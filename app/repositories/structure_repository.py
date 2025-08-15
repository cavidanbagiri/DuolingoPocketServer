


import os
import csv
from datetime import datetime
import aiohttp
import asyncio
import pandas as pd

from collections import defaultdict
from typing import List



from fastapi import HTTPException

from sqlalchemy import select, func, and_, update, or_, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, outerjoin
from sqlalchemy.future import select

from app.models.word_model import Word, Sentence, SentenceWord, WordMeaning, Translation, SentenceTranslation, \
    LearnedWord
from app.models.user_model import Language, UserModel, UserLanguage, UserWord
from app.schemas.translate_schema import TranslateSchema


from app.logging_config import setup_logger
logger = setup_logger(__name__, "word.log")





# Class For executing the functions
class HelperFunction:

    def __init__(self, db):
        self.db = db

    # 1 - Create language model
    async def create_language_list(self):
        lang_repo = CreateMainStructureLanguagesListRepository(self.db)
        await lang_repo.create_main_languages()

    #2 - Generate Top 10.000 words in english and add to words to word model
    async def create_top_10000_words_english(self):
        eng_rep = CreateMainStructureRepositoryForEnglishLanguage(self.db)
        await eng_rep.create_top_10000_words_in_english()

    #3 - Update top 10.000 words and add meanings if work ))
    async def update_top_10000_words_english(self):
        eng_rep = UpdateMainStructureRepositoryForEnglishLanguage(self.db)
        await eng_rep.update_top_10000_words_in_english_add_meanings()

    #4 - Generate sentences about the 5 words
    async def generate_five_sentences_about_each_word(self):
        eng_rep = GenerateEnglishSentencesForEachWord(self.db)
        await eng_rep.main_func()

    #5 - Translate Words for eng to ru
    async def translateentoru(self):
        tr_repo = TranslateAllENWordsToRU(self.db)
        await tr_repo.main_func()

    #6 - Fix others pos to true pos
    async def fix_other_pos(self):
        repo = FixPosFunctionality(self.db)
        await repo.find_others_and_fix_it()


    #7 - Insert russian words and translation to words model and translations model
    async def insert_russian_words_to_table(self):
        repo = CreateMainStructureForRussianRepository(self.db)
        await repo.insert_russian_words_to_table()


    #7 - Fetch Russian Words, generate sentences and translate to english
    async def fetch_russian_words_generate_sentence_and_translate_english (self):
        repo = FetchRussianWordsAndGenerateSentenceAndTranslateToEnglish(self.db)
        await repo.main_func()

# Main Class Working
class CreateMainStructureRepository:

    def __init__(self, db):
        self.db = db
        self.helper_funcs = HelperFunction(db)

    async def create_main_structure(self):

        # 1 - Create Language List [en, ru, es, tr]
        # await self.helper_funcs.create_language_list()


        # 3 - Create Word List For English lang
        # await self.helper_funcs.create_top_10000_words_english()


        # 3 - Update Top 10.000 words in English
        # await self.helper_funcs.update_top_10000_words_english()

        # 4 - Create sentence with words in english
        # await self.helper_funcs.generate_five_sentences_about_each_word()

        # 5 - Translate words from eng to rus
        # await self.helper_funcs.translateentoru()

        # 6 - Fix Others pos
        # await self.helper_funcs.fix_other_pos()


        # 7 - Insert Russian words to table
        # await self.helper_funcs.insert_russian_words_to_table()

        # 8 - Fetch Russian Words and ganerate and translate to english
        await self.helper_funcs.fetch_russian_words_generate_sentence_and_translate_english()


        return 'Added'






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





################################################################## Create Top languages List
class CreateMainStructureLanguagesListRepository:

    def __init__(self, db):
        self.db = db

    async def create_main_languages(self):
        top_languages = [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "ru", "name": "Russian"},
            {"code": "tr", "name": "Turkish"},
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




import os
from typing import List


import aiohttp
import asyncio

from sqlalchemy import select, func
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload


from app.logging_config import setup_logger
from app.models.word_model import Language, Word, Sentence, SentenceWord, WordMeaning, Translation, SentenceTranslation
from app.schemas.translate_schema import TranslateSchema

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

        return 'Added'



# Create Top languages List
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


    # This code block is added by chat gpt
    async def translate_sentence(self, text: str) -> str:
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
                .where(Word.id.between(6, 100)) # Need to start to generate a sentences from 100 to 1000
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
            words = await self._get_words_batch(0, 84)
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

        await self.db.commit()

    # This is real code a and work, l add new function with this name as up
    # async def _save_sentences(self, word: Word, sentences: List[str]):
    #     """Save sentences and link them to word and meanings"""
    #     for sentence_text in sentences:
    #         # Create Sentence record
    #         sentence = Sentence(
    #             text=sentence_text,
    #             language_code="en"  # Assuming English sentences
    #         )
    #         self.db.add(sentence)
    #         await self.db.flush()  # Get the sentence ID
    #
    #         # Link sentence to word
    #         sentence_word = SentenceWord(
    #             sentence_id=sentence.id,
    #             word_id=word.id
    #         )
    #         self.db.add(sentence_word)
    #
    #     await self.db.commit()

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

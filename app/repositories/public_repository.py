# app/repositories/public_repository.py
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload
from app.models.word_model import (
    Word, Translation, WordMeaning, Language,
    Sentence, SentenceTranslation, SentenceWord,
    Category, word_category_association, MeaningSentenceLink
)
from app.schemas.public_seo import (
    WordRichPayload, TranslationOut, MeaningOut,
    SentenceOut, CategoryOut, RelatedWordOut, WordSEOPayload, SlugOut
)
from typing import List, Dict, Optional



class PublicSEORepo:

    def __init__(self, db: AsyncSession):
        self.db = db


    async def get_all_slugs(self):
        """
        Get all word slugs for static generation.
        Returns: [{lang: "en", word: "book"}, {lang: "es", word: "libro"}, ...]
        """
        try:
            stmt = (
                select(
                    Word.language_code.label("lang"),
                    Word.text.label("word"),
                )
                .distinct(Word.language_code, Word.text)
                .limit(10_000)  # ✅ Safety cap
            )
            rows = (await self.db.execute(stmt)).all()
            return [SlugOut(lang=r.lang, word=r.word) for r in rows]
        except Exception as e:
            print(f"Error in get_all_slugs: {str(e)}")
            return []


    async def get_word_seo(self, lang_from: str, lang_to: str, word: str):
        # 1. main word
        word_obj = await self.db.scalar(
            select(Word).where(
                and_(Word.text == word, Word.language_code == lang_from)
            ).limit(1)
        )
        if not word_obj:
            return None

        # 2. translation
        translation = await self.db.scalar(
            select(Translation.translated_text)
            .where(
                and_(
                    Translation.source_word_id == word_obj.id,
                    Translation.target_language_code == lang_to,
                )
            )
            .limit(1)
        )

        # 3. pull 3 example sentences that contain this word
        sentences = await self.db.scalars(
            select(Sentence.text)
            .join(SentenceWord, SentenceWord.sentence_id == Sentence.id)
            .where(
                and_(
                    SentenceWord.word_id == word_obj.id,
                    Sentence.language_code == lang_from,
                )
            )
            .limit(3)
        )
        examples = [s for s in sentences]

        # 4. pretty target language name
        target_lang_name = await self.db.scalar(
            select(Language.name).where(Language.code == lang_to).limit(1)
        )

        return WordSEOPayload(
            word=word_obj.text,
            translation=translation or "",
            targetLangName=target_lang_name or lang_to.upper(),
            audioUrl=f"https://api.w9999.app/tts/{lang_to}/{translation or word}",
            examples=examples,
            langFrom=lang_from,
            langTo=lang_to,
        )

    async def get_word_rich(self, lang_from: str, word: str) -> Optional[WordRichPayload]:
        """
        Fetch comprehensive word data for static page generation.
        Only source language - no translations to other languages.
        """
        try:
            # 1. Fetch word with categories (no translations)
            word_stmt = (
                select(Word)
                .where(and_(
                    Word.text == word,
                    Word.language_code == lang_from
                ))
                .options(
                    selectinload(Word.categories),
                )
                .limit(1)
            )

            word_obj = await self.db.scalar(word_stmt)
            if not word_obj:
                return None

            # 2. Fetch all meanings for this word
            meanings_stmt = (
                select(WordMeaning)
                .where(WordMeaning.word_id == word_obj.id)
            )
            meanings_result = await self.db.scalars(meanings_stmt)
            meanings_list = list(meanings_result)

            # 3. Process meanings
            meanings = []
            for meaning in meanings_list:
                if meaning:
                    meanings.append(MeaningOut(
                        pos=meaning.pos or "",
                        definition=meaning.definition or "",
                        example_sentences=[]
                    ))

            # 4. Fetch 7 example sentences in source language only
            example_sentences_stmt = (
                select(Sentence)
                .join(SentenceWord, SentenceWord.sentence_id == Sentence.id)
                .where(and_(
                    SentenceWord.word_id == word_obj.id,
                    Sentence.language_code == lang_from
                ))
                .order_by(func.random())
                .limit(7)
            )

            sentence_rows = await self.db.scalars(example_sentences_stmt)
            example_sentences = []
            for sentence in sentence_rows:
                if sentence and sentence.text:
                    example_sentences.append(SentenceOut(
                        text=sentence.text or "",
                        translation=None  # ✅ No translation
                    ))

            # 5. Get categories
            categories = []
            if word_obj.categories:
                for cat in word_obj.categories:
                    if cat:
                        categories.append(CategoryOut(
                            name=cat.name or "",
                            description=cat.description or None
                        ))

            # 6. Find related words (same categories, same language)
            related_words = []
            if categories and word_obj.categories:
                category_ids = [cat.id for cat in word_obj.categories if cat]
                if category_ids:
                    # related_stmt = (
                    #     select(Word)
                    #     .join(word_category_association, word_category_association.c.word_id == Word.id)
                    #     .where(and_(
                    #         word_category_association.c.category_id.in_(category_ids),
                    #         Word.id != word_obj.id,
                    #         Word.language_code == lang_from
                    #     ))
                    #     .order_by(Word.frequency_rank)
                    #     .limit(8)
                    #     .distinct(Word.id)
                    # )
                    related_stmt = (
                        select(Word)
                        .join(word_category_association, word_category_association.c.word_id == Word.id)
                        .where(and_(
                            word_category_association.c.category_id.in_(category_ids),
                            Word.id != word_obj.id,
                            Word.language_code == lang_from
                        ))
                        .distinct(Word.id)  # Move to BEFORE order_by
                        .order_by(Word.id, Word.frequency_rank)  # ✅ id FIRST, then frequency_rank
                        .limit(8)
                    )
                    related_word_objs = await self.db.scalars(related_stmt)

                    for rel_word in related_word_objs:
                        if rel_word:
                            related_words.append(RelatedWordOut(
                                text=rel_word.text or "",
                                level=rel_word.level or None,
                                frequency_rank=rel_word.frequency_rank
                            ))

            # 7. Get language name
            source_lang_name = lang_from.upper()
            try:
                source_lang_stmt = select(Language.name).where(Language.code == lang_from)
                result = await self.db.scalar(source_lang_stmt)
                if result:
                    source_lang_name = result
            except:
                pass

            # 8. Generate audio URL for source word only
            audio_urls = {
                lang_from: f"https://api.w9999.app/tts/{lang_from}/{word}"
            }

            # 9. Get ALL sentences for this word
            all_sentence_texts = []
            try:
                all_sentences_stmt = (
                    select(Sentence.text)
                    .join(SentenceWord, SentenceWord.sentence_id == Sentence.id)
                    .where(and_(
                        SentenceWord.word_id == word_obj.id,
                        Sentence.language_code == lang_from
                    ))
                    .limit(10)
                )
                result = await self.db.scalars(all_sentences_stmt)
                all_sentence_texts = [text for text in result if text]
            except:
                pass

            # 10. Return payload (no translations, no lang_to)
            return WordRichPayload(
                word=word_obj.text or "",
                ipa_pronunciation=None,
                level=word_obj.level or None,
                frequency_rank=word_obj.frequency_rank,
                translations=[],  # ✅ Empty
                meanings=meanings,
                example_sentences=example_sentences,
                categories=categories,
                related_words=related_words,
                audio_urls=audio_urls,
                source_language=lang_from,
                source_language_name=source_lang_name,
                target_languages=[],  # ✅ Empty
                translation="",  # ✅ Empty
                targetLangName="",  # ✅ Empty
                audioUrl=f"https://api.w9999.app/tts/{lang_from}/{word}",
                examples=all_sentence_texts[:3] if all_sentence_texts else [],
                langFrom=lang_from,
                langTo=lang_from  # ✅ Same as source
            )

        except Exception as e:
            print(f"Error in get_word_rich: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


class DeepSeekRepo:

    def __init__(self, db: AsyncSession = None):
        """
        After User select the word from Google SEO, the ai will give an information about the word.
        """
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.max_tokens = 6000
        self.max_context_messages = 10  # Keep last 10 messages for context
        self.db = db


    async def get_ai_answer(self, lang_from: str, lang_to: str | None, word: str):
        pass


class TopWordsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_mixed(self, language_code: str, limit: int = 1000):
        """
        Get top N words by frequency (Mixed categories).
        Query is faster because it does not require a JOIN.
        """
        stmt = (
            select(Word)
            .where(Word.language_code == language_code)
            .order_by(Word.frequency_rank.asc().nulls_last())
            .limit(limit)
        )
        result = await self.db.scalars(stmt)
        return self._format_list(result.all())

    async def get_by_pos(self, language_code: str, pos: str, limit: int = 1000):
        """
        Get top N words filtered by Part of Speech (JOINS WordMeaning).
        """
        stmt = (
            select(Word)
            .join(WordMeaning, WordMeaning.word_id == Word.id)
            .where(and_(
                Word.language_code == language_code,
                WordMeaning.pos.ilike(pos)  # Case insensitive (Verb, verb, VERB)
            ))
            .order_by(Word.frequency_rank.asc().nulls_last())
            .limit(limit)
            .distinct()  # vital: prevents seeing the same word twice if it has 2 verb meanings
        )

        result = await self.db.scalars(stmt)
        return self._format_list(result.all())

    def _format_list(self, words):
        """Helper to format output consistently"""
        return [
            {
                "id": w.id,
                "text": w.text,
                "frequency_rank": w.frequency_rank,
                "level": w.level,
                # We do not return strict POS here for mixed lists
                # because one word can be both noun and verb.
            }
            for w in words
        ]


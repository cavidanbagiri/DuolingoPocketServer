# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import select, and_
# from app.models.word_model import Word, Translation, WordMeaning, Language, Sentence, SentenceTranslation, SentenceWord
# from app.schemas.public_seo import WordSEOPayload, SlugOut



# app/repositories/public_repository.py
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
import asyncio




class PublicSEORepo:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_slugs(self):
        stmt = (
            select(
                Word.language_code.label("lf"),
                Translation.target_language_code.label("lt"),
                Word.text.label("word"),
            )
            .join(Translation, Translation.source_word_id == Word.id)
            .distinct(Word.id, Translation.target_language_code)  # ← only these cols
            .limit(10_000)  # ← safety cap
        )
        rows = (await self.db.execute(stmt)).all()
        return [SlugOut(lf=r.lf, lt=r.lt, word=r.word) for r in rows]

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

    async def get_word_rich(self, lang_from: str, lang_to: str, word: str) -> Optional[WordRichPayload]:
        """
        Fetch comprehensive word data for static page generation
        WITHOUT using MeaningSentenceLink
        """
        try:
            # 1. Fetch word with translations and categories
            word_stmt = (
                select(Word)
                .where(and_(
                    Word.text == word,
                    Word.language_code == lang_from
                ))
                .options(
                    selectinload(Word.translations).selectinload(Translation.target_language),
                    selectinload(Word.categories),
                )
                .limit(1)
            )

            word_obj = await self.db.scalar(word_stmt)
            if not word_obj:
                return None

            # 2. Fetch all meanings for this word (handle None values)
            meanings_stmt = (
                select(WordMeaning)
                .where(WordMeaning.word_id == word_obj.id)
            )
            meanings_result = await self.db.scalars(meanings_stmt)
            meanings_list = list(meanings_result)

            # 3. Fetch translations to all languages
            translations_stmt = (
                select(Translation, Language.name)
                .join(Language, Translation.target_language_code == Language.code)
                .where(Translation.source_word_id == word_obj.id)
            )
            translation_rows = await self.db.execute(translations_stmt)

            translations = []
            primary_translation = None
            for trans, lang_name in translation_rows:
                if trans and trans.translated_text:  # Check for valid translation
                    translation_out = TranslationOut(
                        language_code=trans.target_language_code,
                        language_name=lang_name,
                        translated_text=trans.translated_text or ""  # Ensure string
                    )
                    translations.append(translation_out)

                    if trans.target_language_code == lang_to:
                        primary_translation = trans.translated_text or ""

            # 4. Process meanings with None handling
            meanings = []
            for meaning in meanings_list:
                if meaning:  # Check if meaning exists
                    meanings.append(MeaningOut(
                        pos=meaning.pos or "",  # Convert None to empty string
                        definition=meaning.definition or "",  # Convert None to empty string
                        example_sentences=[]  # Empty since we don't have the link
                    ))

            # 5. Fetch 5-7 example sentences containing this word
            example_sentences_stmt = (
                select(Sentence, SentenceTranslation.translated_text)
                .join(SentenceWord, SentenceWord.sentence_id == Sentence.id)
                .outerjoin(SentenceTranslation, and_(
                    SentenceTranslation.source_sentence_id == Sentence.id,
                    SentenceTranslation.language_code == lang_to
                ))
                .where(and_(
                    SentenceWord.word_id == word_obj.id,
                    Sentence.language_code == lang_from
                ))
                .order_by(func.random())
                .limit(7)
            )

            sentence_rows = await self.db.execute(example_sentences_stmt)
            example_sentences = []
            for sentence, translation in sentence_rows:
                if sentence and sentence.text:  # Check for valid sentence
                    example_sentences.append(SentenceOut(
                        text=sentence.text or "",
                        translation=translation or None
                    ))

            # 6. Get categories
            categories = []
            if word_obj.categories:
                for cat in word_obj.categories:
                    if cat:  # Check if category exists
                        categories.append(CategoryOut(
                            name=cat.name or "",
                            description=cat.description or None
                        ))

            # 7. Find related words (same categories, same language)
            related_words = []
            if categories and word_obj.categories:
                category_ids = [cat.id for cat in word_obj.categories if cat]
                if category_ids:
                    related_stmt = (
                        select(Word)
                        .join(word_category_association, word_category_association.c.word_id == Word.id)
                        .where(and_(
                            word_category_association.c.category_id.in_(category_ids),
                            Word.id != word_obj.id,
                            Word.language_code == lang_from
                        ))
                        .order_by(Word.frequency_rank)
                        .limit(8)
                        .distinct(Word.id)
                    )
                    related_word_objs = await self.db.scalars(related_stmt)

                    for rel_word in related_word_objs:
                        if rel_word:  # Check if word exists
                            related_words.append(RelatedWordOut(
                                text=rel_word.text or "",
                                level=rel_word.level or None,
                                frequency_rank=rel_word.frequency_rank
                            ))

            # 8. Get language names
            source_lang_name = lang_from.upper()
            try:
                source_lang_stmt = select(Language.name).where(Language.code == lang_from)
                result = await self.db.scalar(source_lang_stmt)
                if result:
                    source_lang_name = result
            except:
                pass  # Use default if fails

            # 9. Generate audio URLs for all translations
            audio_urls = {}
            base_tts_url = "https://api.w9999.app/tts"
            for trans in translations:
                if trans.translated_text:
                    audio_urls[trans.language_code] = f"{base_tts_url}/{trans.language_code}/{trans.translated_text}"

            # 10. Get ALL sentences for this word
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

            # 11. Compile the payload with safe defaults
            return WordRichPayload(
                word=word_obj.text or "",
                ipa_pronunciation=None,
                level=word_obj.level or None,
                frequency_rank=word_obj.frequency_rank,
                translations=translations,
                meanings=meanings,
                example_sentences=example_sentences,
                categories=categories,
                related_words=related_words,
                audio_urls=audio_urls,
                source_language=lang_from,
                source_language_name=source_lang_name,
                target_languages=[t.language_code for t in translations if t.language_code],

                # Backward compatibility fields
                translation=primary_translation or "",
                targetLangName=next(
                    (t.language_name for t in translations if t.language_code == lang_to and t.language_name),
                    lang_to.upper()
                ),
                audioUrl=f"{base_tts_url}/{lang_to}/{primary_translation or word}" if primary_translation else "",
                examples=all_sentence_texts[:3] if all_sentence_texts else [],
                langFrom=lang_from,
                langTo=lang_to
            )

        except Exception as e:
            # Log the error for debugging
            print(f"Error in get_word_rich: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# This class must be used Public Seo Repo
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


class GetTopWordsRepository:
    def __init__(self, db: AsyncSession = None):
        self.db = db

    async def get_top_words_by_frequency(self, language_code: str, limit: int = 1000):
        """Get top N words by frequency rank - words only, no translations"""
        stmt = (
            select(Word)
            .where(Word.language_code == language_code)
            .order_by(Word.frequency_rank.asc().nulls_last())
            .limit(limit)
        )
        result = await self.db.scalars(stmt)
        words = result.all()

        return [
            {
                "id": w.id,
                "text": w.text,
                "frequency_rank": w.frequency_rank,
                "level": w.level,
            }
            for w in words
        ]
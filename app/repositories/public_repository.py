from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models.word_model import Word, Translation, WordMeaning, Language, Sentence, SentenceTranslation, SentenceWord
from app.schemas.public_seo import WordSEOPayload, SlugOut


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
            .select_from(Word)  # ← explicit FROM
            .join(Translation, Translation.source_word_id == Word.id)
            .distinct()
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
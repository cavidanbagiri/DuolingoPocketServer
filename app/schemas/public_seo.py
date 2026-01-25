# app/schemas/public_seo.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class MeaningOut(BaseModel):
    pos: Optional[str] = None
    definition: Optional[str] = None  # Make this optional
    example_sentences: List[str] = Field(default_factory=list)


class TranslationOut(BaseModel):
    language_code: str
    language_name: Optional[str] = None
    translated_text: str


class SentenceOut(BaseModel):
    text: str
    translation: Optional[str] = None


class CategoryOut(BaseModel):
    name: str
    description: Optional[str] = None


class RelatedWordOut(BaseModel):
    text: str
    level: Optional[str] = None
    frequency_rank: Optional[int] = None


class WordRichPayload(BaseModel):
    # Basic info
    word: str
    ipa_pronunciation: Optional[str] = None
    level: Optional[str] = None
    frequency_rank: Optional[int] = None

    # Translations
    translations: List[TranslationOut] = Field(default_factory=list)

    # Meanings with examples
    meanings: List[MeaningOut] = Field(default_factory=list)

    # Example sentences
    example_sentences: List[SentenceOut] = Field(default_factory=list)

    # Categories
    categories: List[CategoryOut] = Field(default_factory=list)

    # Related words
    related_words: List[RelatedWordOut] = Field(default_factory=list)

    # Audio
    audio_urls: dict = Field(default_factory=dict)

    # Language info
    source_language: str
    source_language_name: Optional[str] = None
    target_languages: List[str] = Field(default_factory=list)

    # For JSON-LD schema
    last_updated: datetime = Field(default_factory=datetime.now)

    # Backward compatibility
    translation: Optional[str] = None
    targetLangName: Optional[str] = None
    audioUrl: Optional[str] = None
    examples: Optional[List[str]] = None
    langFrom: str
    langTo: str

class SlugOut(BaseModel):
    lf: str
    lt: str
    word: str


class WordSEOPayload(BaseModel):
    word: str
    translation: str
    targetLangName: str
    audioUrl: str
    examples: List[str]
    langFrom: str
    langTo: str









#
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
#
#
# class MeaningOut(BaseModel):
#     pos: str
#     definition: str
#     example_sentences: List[str] = []
#
#
# class TranslationOut(BaseModel):
#     language_code: str
#     language_name: str
#     translated_text: str
#
#
# class SentenceOut(BaseModel):
#     text: str
#     translation: Optional[str] = None
#
#
# class CategoryOut(BaseModel):
#     name: str
#     description: Optional[str] = None
#
#
# class RelatedWordOut(BaseModel):
#     text: str
#     level: str
#     frequency_rank: int
#
#
# class WordRichPayload(BaseModel):
#     # Basic info
#     word: str
#     ipa_pronunciation: Optional[str] = None  # We'll need to add this field to Word model
#     level: str
#     frequency_rank: int
#
#     # Translations
#     translations: List[TranslationOut]
#
#     # Meanings with examples
#     meanings: List[MeaningOut]
#
#     # Example sentences (5-7 from database)
#     example_sentences: List[SentenceOut]
#
#     # Categories
#     categories: List[CategoryOut]
#
#     # Related words (same category, similar frequency)
#     related_words: List[RelatedWordOut]
#
#     # Audio
#     audio_urls: dict  # {"en": "url", "es": "url", etc.}
#
#     # Language info
#     source_language: str
#     source_language_name: str
#     target_languages: List[str]
#
#     # For JSON-LD schema
#     last_updated: datetime = datetime.now()
#
#     # Keep backward compatibility
#     translation: Optional[str] = None  # Primary translation for backward compat
#     targetLangName: Optional[str] = None
#     audioUrl: Optional[str] = None
#     examples: Optional[List[str]] = None
#     langFrom: str
#     langTo: str
#
#
# # Keep existing schemas for backward compatibility
# class SlugOut(BaseModel):
#     lf: str
#     lt: str
#     word: str
#
#
# class WordSEOPayload(BaseModel):
#     word: str
#     translation: str
#     targetLangName: str
#     audioUrl: str
#     examples: List[str]
#     langFrom: str
#     langTo: str
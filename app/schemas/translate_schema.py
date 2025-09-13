import os
from typing import Optional

from pydantic import BaseModel, field_validator


class TranslateSchema(BaseModel):
    text: str
    from_lang: str = "auto"
    to_lang: str
    alternatives: int = 3


class DetectLanguageSchema(BaseModel):
    text: str
    folder_id: str = os.getenv("YANDEX_FOLDER_ID")


class UserSavedWordCreateSchema(BaseModel):
    user_id: int
    word_id: int
    learned: bool = False
    starred: bool = False

    class Config:
        from_attributes = True


class WordSchema(BaseModel):
    from_lang: str
    to_lang: str
    word: str
    part_of_speech: Optional[str] = "other"
    translation: str

    @field_validator("word")
    def normalize_word(cls, value):
        return value.lower().strip()

    @field_validator("translation")
    def normalize_translation(cls, value):
        return value.lower().strip()

    class Config:
        from_attributes = True


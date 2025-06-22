import os

from pydantic import BaseModel


class TranslateSchema(BaseModel):
    q: str
    source: str = "auto"
    target: str
    alternatives: int = 3


class DetectLanguageSchema(BaseModel):
    text: str
    folder_id: str = os.getenv("YANDEX_FOLDER_ID")
from pydantic import BaseModel

class SlugOut(BaseModel):
    lf: str
    lt: str
    word: str

class WordSEOPayload(BaseModel):
    word: str
    translation: str
    targetLangName: str
    audioUrl: str
    examples: list[str]
    langFrom: str
    langTo: str
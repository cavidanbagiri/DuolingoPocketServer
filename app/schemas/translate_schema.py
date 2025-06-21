

from pydantic import BaseModel

class TranslateSchema(BaseModel):

    q: str | None
    source: str | None
    target: str | None
    format: str | None
    alternatives: int | None


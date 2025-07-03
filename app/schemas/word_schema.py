from typing import Optional, Literal

from pydantic import BaseModel, field_validator


class ChangeWordStatusSchema(BaseModel):
    word_id: int
    w_status: Literal['starred', 'learned', 'delete']

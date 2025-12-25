from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class NoteBase(BaseModel):
    note_name: str = Field(..., min_length=1, max_length=200)
    target_lang: Optional[str] = Field(None, max_length=10)
    note_type: str = Field(default="general")
    content: str = Field(..., min_length=1)
    tags: Optional[List[str]] = Field(default=[])

    @validator('note_type')
    def validate_note_type(cls, v):
        valid_types = ['vocabulary', 'grammar', 'general']
        if v not in valid_types:
            raise ValueError(f'Note type must be one of: {", ".join(valid_types)}')
        return v

    @validator('target_lang', pre=True)  # Add pre=True to process before validation
    def validate_target_lang(cls, v):
        # Convert empty string to None
        if v == "":
            return None

        if v is not None and v not in ['es', 'en', 'ru', 'tr']:
            raise ValueError('Language must be: es, en, ru, tr or null')
        return v

    @validator('content')
    def validate_content_length(cls, v):
        if len(v) > 50000:
            raise ValueError('Content cannot exceed 50,000 characters')
        return v

    @validator('tags', pre=True)  # Add pre=True
    def validate_tags(cls, v):
        # Ensure tags is always a list
        if v is None:
            return []

        if isinstance(v, str):
            # If it's a string, try to parse it as array
            try:
                import json
                v = json.loads(v)
            except:
                v = []

        if len(v) > 20:
            raise ValueError('Maximum 20 tags allowed')

        if v:
            for tag in v:
                if len(tag) > 50:
                    raise ValueError('Tag cannot exceed 50 characters')
        return v

    class Config:
        from_attributes = True


class NoteCreate(NoteBase):
    pass


class NoteUpdate(BaseModel):
    note_name: Optional[str] = Field(None, min_length=1, max_length=200)
    target_lang: Optional[str] = Field(None, max_length=10)
    note_type: Optional[str] = Field(None)
    content: Optional[str] = Field(None, min_length=1)
    tags: Optional[List[str]] = Field(None)

    @validator('note_type')
    def validate_note_type(cls, v):
        if v is not None:
            valid_types = ['vocabulary', 'grammar', 'general']
            if v not in valid_types:
                raise ValueError(f'Note type must be one of: {", ".join(valid_types)}')
        return v

    @validator('target_lang', pre=True)
    def validate_target_lang(cls, v):
        # Convert empty string to None
        if v == "":
            return None

        if v is not None and v not in ['es', 'en', 'ru', 'tr']:
            raise ValueError('Language must be: es, en, ru, tr or null')
        return v

    class Config:
        from_attributes = True


class NoteResponse(NoteBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
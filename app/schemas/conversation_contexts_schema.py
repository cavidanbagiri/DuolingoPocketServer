# schemas/conversation_context_schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class MessageSchema(BaseModel):
    """Schema for a single chat message"""
    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp")

    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError("Role must be 'system', 'user', or 'assistant'")
        return v


class ConversationContextBase(BaseModel):
    """Base schema for conversation context"""
    user_id: int = Field(..., description="User ID")
    word: str = Field(..., min_length=1, max_length=255, description="Word being discussed")
    language: str = Field(..., min_length=2, max_length=50, description="Target language")
    native_language: str = Field(..., min_length=2, max_length=50, description="User's native language")


class ConversationContextCreate(ConversationContextBase):
    """Schema for creating a new context"""
    messages: List[Dict[str, Any]] = Field(default=[], description="Initial messages")

    @validator('messages')
    def validate_messages(cls, v):
        # Ensure messages can be serialized to JSON
        try:
            json.dumps(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Messages must be JSON serializable: {str(e)}")
        return v


class ConversationContextResponse(ConversationContextBase):
    """Schema for context response"""
    id: int
    context_hash: str
    messages: List[Dict[str, Any]]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationContextUpdate(BaseModel):
    """Schema for updating context"""
    messages: Optional[List[Dict[str, Any]]] = None
    native_language: Optional[str] = None
    is_active: Optional[bool] = None


class GenerateAIChatSchema(BaseModel):
    # user_id: int = Field(..., description="User ID for context management")
    word: str = Field(..., min_length=1, max_length=100)
    language: str = Field(..., min_length=2, max_length=50)
    native: str = Field(..., min_length=2, max_length=50)
    message: str = Field(..., min_length=1, max_length=1000)

    class Config:
        from_attributes = True



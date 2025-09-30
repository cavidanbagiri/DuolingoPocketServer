# models/ai_models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class POSDefinition(BaseModel):
    pos: str  # Part of speech
    definitions: List[str]  # Multiple definitions

class WordAnalysisRequest(BaseModel):
    word: str
    language: str = "en"

class WordAnalysisResponse(BaseModel):
    word: str
    categories: List[str]
    pos_definitions: List[POSDefinition]
    examples: List[str]
    synonyms: Optional[List[str]] = None
    antonyms: Optional[List[str]] = None

class DeepSeekRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 2000
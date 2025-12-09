from typing import Optional, Literal, List, Dict, Any

from pydantic import BaseModel, field_validator, Field


class ChangeWordStatusSchema(BaseModel):
    word_id: int
    w_status: Literal['starred', 'learned', 'delete']

class VoiceSchema(BaseModel):
    text: str
    language: str

class GenerateAIWordSchema(BaseModel):
    text: str
    language: str
    native: str


# class GenerateAIChatSchema(BaseModel):
#     word: str  # The word being discussed
#     message: str  # The user's question/message
#     language: str  # Target language (e.g., 'en')
#     native: str    # User's native language (e.g., 'tr')


class TranslateSchema(BaseModel):
    text: str
    from_lang: str = "auto"
    to_lang: str
    alternatives: int = 3


# class AiDirectChatSchema(BaseModel):
#     message: str
#     native_language: str = "English"


class AiDirectChatSchema(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User's message to AI tutor")
    native_language: str = Field(default="English", description="User's native language for better explanations")

    class Config:
        schema_extra = {
            "example": {
                "message": "Can you explain the difference between ser and estar in Spanish?",
                "native_language": "English"
            }
        }




class AIWordResponse(BaseModel):
    word: str
    target_language: str  # Changed from 'language' for clarity
    native_language: str
    definition: str
    pronunciation: Optional[str] = None
    part_of_speech: str
    examples: List[str] = Field(..., min_items=5, description="At least 5 example sentences with translations")
    usage_contexts: List[str]
    common_phrases: List[str]
    grammar_tips: List[str]
    additional_insights: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "word": "gehen",
                "target_language": "de",
                "native_language": "es",
                "definition": "Verbo que significa 'ir' o 'caminar' en alemán",
                "pronunciation": "gué-jen",
                "part_of_speech": "verbo",
                "examples": [
                    "Ich gehe zur Schule. - Voy a la escuela.",
                    "Wir gehen ins Kino. - Vamos al cine.",
                    "Er geht jeden Tag spazieren. - Él va a pasear todos los días.",
                    "Gehst du mit uns? - ¿Vas con nosotros?",
                    "Sie sind gestern zum Markt gegangen. - Ellos fueron al mercado ayer."
                ],
                "usage_contexts": [
                    "Para describir movimiento a pie hacia un lugar",
                    "En conversaciones sobre actividades diarias",
                    "Para hacer planes e invitaciones",
                    "En contextos formales e informales"
                ],
                "common_phrases": [
                    "Wie geht es dir? - ¿Cómo estás? (Literalmente: ¿Cómo te va?)",
                    "Es geht mir gut. - Estoy bien. (Me va bien)",
                    "Das geht nicht. - Eso no funciona/no es posible."
                ],
                "grammar_tips": [
                    "Conjugación presente: ich gehe, du gehst, er/sie/es geht, wir gehen, ihr geht, sie/Sie gehen",
                    "Verbo irregular: pasado ging, participio perfecto gegangen",
                    "Usa preposiciones específicas: gehen zu, gehen in, gehen nach",
                    "Se usa con verbos modales: können gehen, müssen gehen"
                ],

                "additional_insights": {
                    "verb_conjugations": {
                        "presente": "ich gehe, du gehst, er/sie/es geht, wir gehen, ihr geht, sie/Sie gehen",
                        "pretérito": "ich ging, du gingst, er/sie/es ging, wir gingen, ihr gingt, sie/Sie gingen",
                        "perfecto": "ich bin gegangen"
                    },
                    "synonyms": ["laufen", "spazieren", "wandern"],
                    "antonyms": ["stehen", "bleiben", "ankommen"]
                }
            }
        }



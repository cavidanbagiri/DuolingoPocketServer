
from sqlalchemy import Integer, String, Boolean, DateTime, ForeignKey, func, Enum as SQLAlchemyEnum
from sqlalchemy.orm import mapped_column, Mapped, relationship
from enum import Enum

from base_model import Base


class WordType(str, Enum):
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    PREPOSITION = "Preposition"
    CONJUNCTION = "Conjunction"
    PHRASE = "phrase"
    OTHER = "other"


class SavedWordModel(Base):
    __tablename__ = "saved_words"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    word: Mapped[str] = mapped_column(String)
    translation: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)  # e.g., Google Translate, Yandex
    language_pair: Mapped[str] = mapped_column(String)  # e.g., ru-en, tr-ru
    learned: Mapped[str] = mapped_column(Boolean, default=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sentence: Mapped[str] = mapped_column(String, nullable=True)
    part_of_speech = mapped_column(SQLAlchemyEnum(WordType), default=WordType.OTHER)
    image_hint: Mapped[str] = mapped_column(String, nullable=True)  # URL or path to generated image

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))

    sentences = relationship("SentenceWordModel", back_populates="word")

class SentenceWordModel(Base):
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sentence: Mapped[str] = mapped_column(String)
    translation: Mapped[str] = mapped_column(String)

    word_id: Mapped[int] = mapped_column(Integer, ForeignKey('saved_words.id'))

    word = relationship("SavedWordModel", back_populates="sentences")
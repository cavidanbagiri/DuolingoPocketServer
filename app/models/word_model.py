
from sqlalchemy import Integer, String, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import mapped_column, Mapped, relationship
from enum import Enum

from app.models.base_model import Base



# SavedWordModel - Core word data
class WordModel(Base):
    __tablename__ = "words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_lang: Mapped[str] = mapped_column(String)
    to_lang: Mapped[str] = mapped_column(String)
    word: Mapped[str] = mapped_column(String, nullable=False)
    part_of_speech = mapped_column(String, default="other")
    translation: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user_saved_words = relationship("UserSavedWord", back_populates="word")
    sentences = relationship("SentenceWordModel", back_populates="word")

    def __repr__(self):
        return f'WordModel(from_lang={self.from_lang}, to_lang={self.to_lang}, word={self.word}, part_speech={self.part_of_speech}, translation={self.translation})'


# User <-> Word relationship
class UserSavedWord(Base):

    __tablename__ = "user_saved_words"

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), primary_key=True)
    word_id: Mapped[int] = mapped_column(Integer, ForeignKey("words.id"), primary_key=True)
    learned: Mapped[bool] = mapped_column(Boolean, default=False)
    starred: Mapped[bool] = mapped_column(Boolean, default=False)
    last_reviewed: Mapped[DateTime] = mapped_column(DateTime(timezone=True))

    # Relationships
    user = relationship("UserModel", foreign_keys=[user_id])
    word = relationship("WordModel", foreign_keys=[word_id])

    def __repr__(self):
        return f"UserSavedWord(user_id={self.user_id}, word_id={self.word_id})"


# Optional: Sentence hints
class SentenceWordModel(Base):
    __tablename__ = "sentence_hints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sentence: Mapped[str] = mapped_column(String)
    translation: Mapped[str] = mapped_column(String)

    word_id: Mapped[int] = mapped_column(Integer, ForeignKey("words.id"))
    word = relationship("WordModel", back_populates="sentences")

from datetime import datetime, timedelta
from typing import Optional
import secrets
import hashlib

from sqlalchemy import String, Boolean, DateTime, ForeignKey, func, Integer, Column, Text
from sqlalchemy.orm import mapped_column, Mapped, relationship
from sqlalchemy.sql import func

from app.models.language_model import Language
from app.models.base_model import Base



class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)

    username: Mapped[str] = mapped_column( nullable=True, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    # password: Mapped[str] = mapped_column(String(100), nullable=False) # need to change it
    password: Mapped[str] = mapped_column(String(100), nullable=True) # need to change it
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False)
    role: Mapped[str] = mapped_column(String, default="user")


    native: Mapped[str] = mapped_column(String, nullable=True)

    language_preferences = relationship("UserLanguage", back_populates="user", cascade="all, delete-orphan")

    favorite_categories = relationship("FavoriteCategory", back_populates="user")
    favorite_words = relationship("FavoriteWord", back_populates="user")


    password_reset_tokens = relationship("PasswordResetToken", back_populates="user", cascade="all, delete-orphan")

    conversation_contexts = relationship(
        "ConversationContextModel",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    direct_chat_contexts = relationship(
        "DirectChatContextModel",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f'UserModel(id:{self.id}, username:{self.username}, email: {self.email}, native: {self.native})'


class ConversationContextModel(Base):
    """
    Model to store AI conversation context per user per word.
    Each time a user selects a new word, a new context is created.
    """
    __tablename__ = "conversation_contexts"

    id: Mapped[int] = mapped_column(primary_key=True)

    # User who owns this conversation
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    # The word being discussed
    word: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # The target language (language being learned)
    language: Mapped[str] = mapped_column(String(50), nullable=False)

    # User's native language
    native_language: Mapped[str] = mapped_column(String(50), nullable=False)

    # Context hash for quick lookups (user_id + word + language)
    context_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True
    )

    # Store conversation history as JSON
    messages: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    # Active flag (useful for cleanup)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationship with User
    user: Mapped["UserModel"] = relationship(back_populates="conversation_contexts")

    def __repr__(self):
        return f"ConversationContextModel(id={self.id}, user_id={self.user_id}, word='{self.word}', language='{self.language}')"


class UserLanguage(Base):
    __tablename__ = "user_languages"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)  # ✅ new primary key
    user_id = mapped_column(Integer, ForeignKey("users.id"))
    target_language_code = mapped_column(String(2), ForeignKey("languages.code"))
    level = mapped_column(String(2), default="A1")
    updated_at = mapped_column(DateTime, default=datetime.utcnow)

    user = relationship("UserModel", back_populates="language_preferences")
    target_language = relationship("Language", foreign_keys=[target_language_code])


class TokenModel(Base):
    __tablename__ = 'tokens'

    id: Mapped[int] = mapped_column(primary_key=True)

    tokens: Mapped[str] = mapped_column(String)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))

    def __str__(self):
        return f"{self.id} {self.tokens} {self.user_id}"


class UserWord(Base):
    __tablename__ = "user_words"

    id = mapped_column(Integer, primary_key=True)
    user_id = mapped_column(Integer, ForeignKey("users.id"))
    word_id = mapped_column(Integer, ForeignKey("words.id"))

    is_starred = mapped_column(Boolean, default=False)
    is_learned = mapped_column(Boolean, default=False)

    created_at = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # Add this!

    # Relationships
    # user = relationship("UserModel", back_populates="user_words")
    word = relationship("Word", back_populates="user_words")


class FavoriteCategory(Base):
    __tablename__ = "favorite_categories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # Assuming you have users table
    name = Column(String, nullable=False)  # e.g., "Russian Nouns", "Russian Verbs"
    icon = Column(String, nullable=True)  # Book
    color = Column(String, nullable=True)  #F23ABF6

    # Relationships
    user = relationship("UserModel", back_populates="favorite_categories")
    favorite_words = relationship("FavoriteWord", back_populates="category")


class FavoriteWord(Base):
    __tablename__ = "favorite_words"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("favorite_categories.id"), nullable=True)  # Optional category

    # Translation data
    from_lang = Column(String(10), nullable=False)  # e.g., 'en'
    to_lang = Column(String(10), nullable=False)  # e.g., 'ru'
    original_text = Column(String, nullable=False)  # e.g., 'hello'
    translated_text = Column(String, nullable=False)  # e.g., 'привет'

    # Timestamp - MAKE SURE THIS EXISTS
    added_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("UserModel", back_populates="favorite_words")
    category = relationship("FavoriteCategory", back_populates="favorite_words")


class DefaultCategory(Base):
    __tablename__ = "default_categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)  # name will be -> Default
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    description = Column(String, nullable=True)

    # Users can copy these to their personal categories


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(64), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    used_at = Column(DateTime, nullable=True)

    # Relationship
    user = relationship("UserModel", back_populates="password_reset_tokens")

    @classmethod
    def create_reset_token(cls, user_id: int, expires_hours: int = 1):
        """Create a new reset token"""
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)

        return cls(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at
        ), token

    def is_valid(self):
        """Check if token is still valid"""
        return not self.used and datetime.utcnow() < self.expires_at


class DirectChatContextModel(Base):
    """
    Model to store direct AI chat conversations (not word-specific).
    This is for general language learning conversations.
    """
    __tablename__ = "direct_chat_contexts"

    id: Mapped[int] = mapped_column(primary_key=True)

    # User who owns this conversation
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    # Conversation topic or identifier (optional)
    topic: Mapped[str] = mapped_column(String(255), nullable=True, index=True)

    # Store conversation history as JSON
    messages: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    # Context hash for quick lookups
    context_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True
    )

    # Active flag
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationship with User
    user: Mapped["UserModel"] = relationship(back_populates="direct_chat_contexts")

    def __repr__(self):
        return f"DirectChatContextModel(id={self.id}, user_id={self.user_id}, topic='{self.topic}')"
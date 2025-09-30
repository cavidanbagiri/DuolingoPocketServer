from datetime import datetime

from sqlalchemy import String, Boolean, DateTime, ForeignKey, func, Integer, Column
from sqlalchemy.orm import mapped_column, Mapped, relationship


from app.models.language_model import Language

from app.models.base_model import Base

class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)

    username: Mapped[str] = mapped_column( nullable=True, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    password: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False)
    role: Mapped[str] = mapped_column(String, default="user")


    native: Mapped[str] = mapped_column(String, nullable=True)

    language_preferences = relationship("UserLanguage", back_populates="user", cascade="all, delete-orphan")

    favorite_categories = relationship("FavoriteCategory", back_populates="user")
    favorite_words = relationship("FavoriteWord", back_populates="user")

    def __repr__(self):
        return f'UserModel(id:{self.id}, username:{self.username}, email: {self.email}, native: {self.native})'



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

    # Relationships
    # user = relationship("UserModel", back_populates="user_words")
    word = relationship("Word", back_populates="user_words")


# Stores user-created categories
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


# Stores the actual favorite translations
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



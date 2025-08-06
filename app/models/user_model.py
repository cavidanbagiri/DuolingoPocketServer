from datetime import datetime

from sqlalchemy import String, Boolean, DateTime, ForeignKey, func, Integer
from sqlalchemy.orm import mapped_column, Mapped, relationship


from app.models.base_model import Base

# from app.models.word_model import Language

from app.models.language_model import Language

#
# class Language(Base):
#     __tablename__ = "languages"
#     code = mapped_column(String(2), primary_key=True)  # en, es, ru
#     name = mapped_column(String(50))  # English, Spanish
#


# class Language(Base):
#     __tablename__ = "languages"
#     code = mapped_column(String(2), primary_key=True)  # en, es, ru
#     name = mapped_column(String(50))  # English, Spanish



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



# Old code but works for one language
# class UserLanguage(Base):
#     __tablename__ = "user_languages"
#
#     user_id = mapped_column(Integer, ForeignKey("users.id"), primary_key=True)
#     target_language_code = mapped_column(String(2), ForeignKey("languages.code"))
#     level = mapped_column(String(2), default="A1")  # Optional: CEFR level
#     updated_at = mapped_column(DateTime, default=datetime.utcnow)
#
#     # user = relationship("UserModel", back_populates="language_preference")
#     # target_language = relationship("Language", foreign_keys=[target_language_code])
#
#     def __repr__(self):
#         return f"UserLanguage(user_id={self.user_id}, native='{self.native_language_code}', target='{self.target_language_code}', level='{self.level}')"



class TokenModel(Base):
    __tablename__ = 'tokens'

    id: Mapped[int] = mapped_column(primary_key=True)

    tokens: Mapped[str] = mapped_column(String)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))

    def __str__(self):
        return f"{self.id} {self.tokens} {self.user_id}"

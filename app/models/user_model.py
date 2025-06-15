from datetime import datetime

from sqlalchemy import String, Boolean, DateTime, ForeignKey, func
from sqlalchemy.orm import mapped_column, Mapped

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


    def __repr__(self):
        return f'UserModel(id:{self.id}, username:{self.username}, email: {self.email})'


class TokenModel(Base):
    __tablename__ = 'tokens'

    id: Mapped[int] = mapped_column(primary_key=True)

    tokens: Mapped[str] = mapped_column(String)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))

    def __str__(self):
        return f"{self.id} {self.tokens} {self.user_id}"

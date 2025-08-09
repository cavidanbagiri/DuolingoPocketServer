from sqlalchemy.orm import mapped_column
from app.models.base_model import Base
from sqlalchemy import String

class Language(Base):
    __tablename__ = "languages"
    code = mapped_column(String(2), primary_key=True)  # en, es, ru
    name = mapped_column(String(50))  # English, Spanish


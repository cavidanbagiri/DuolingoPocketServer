
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship, mapped_column
from enum import Enum
from datetime import datetime
from app.models.base_model import Base


# from app.models.user_model import Language

from app.models.language_model import Language

# Base = declarative_base()


class CEFRLevel(str, Enum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"
    C2 = "C2"

#
# class Language(Base):
#     __tablename__ = "languages"
#     code = mapped_column(String(2), primary_key=True)  # en, es, ru
#     name = mapped_column(String(50))  # English, Spanish



class Word(Base):
    __tablename__ = "words"
    id = Column(Integer, primary_key=True)
    text = Column(String(100), unique=True)  # "book" (shared spelling)
    language_code = Column(String(2))       # "en"
    frequency_rank = Column(Integer)
    level = mapped_column(String(2))  # A1-C2

    meanings = relationship("WordMeaning", back_populates="word")
    translations = relationship("Translation", back_populates="source_word")  # This was missing
    in_sentences = relationship("SentenceWord", back_populates="word")  # Also add this for comp

    def __repr__(self):
        return f'Word(id:({self.id}), text({self.text}, language_code({self.language_code}, frequency_rank:({self.frequency_rank}, level({self.level})))))'

class WordMeaning(Base):
    __tablename__ = "word_meanings"
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey("words.id"))
    pos = Column(String(50))     # "noun"/"verb"
    # definition = Column(String(500))        # "a written work"
    example = Column(String(500))           # "I'm reading a book"

    word = relationship("Word", back_populates="meanings")
    sentences = relationship("Sentence", secondary="meaning_sentence_links")


class MeaningSentenceLink(Base):
    __tablename__ = "meaning_sentence_links"
    meaning_id = Column(Integer, ForeignKey("word_meanings.id"), primary_key=True)
    sentence_id = Column(Integer, ForeignKey("sentences.id"), primary_key=True)


class Translation(Base):
    __tablename__ = "translations"
    id = mapped_column(Integer, primary_key=True)
    source_word_id = mapped_column(Integer, ForeignKey("words.id"))
    target_language_code = mapped_column(String(2), ForeignKey("languages.code"))
    translated_text = mapped_column(String(100))  # "hello" → "hola"

    source_word = relationship("Word", back_populates="translations")


    target_language = relationship("Language", lazy="joined")




# Define SentenceTranslation BEFORE Sentence
class SentenceTranslation(Base):
    __tablename__ = "sentence_translations"
    id = mapped_column(Integer, primary_key=True)
    source_sentence_id = mapped_column(Integer, ForeignKey("sentences.id"))
    language_code = mapped_column(String(2), ForeignKey("languages.code"))
    translated_text = mapped_column(String(500))

    source_sentence = relationship("Sentence", back_populates="translations")

    # This is old code, and change with below new code
    # language = relationship("Language")

    language = relationship("Language", lazy="joined")



class Sentence(Base):
    __tablename__ = "sentences"
    id = mapped_column(Integer, primary_key=True)
    text = mapped_column(String(500))  # "Hello world"
    language_code = mapped_column(String(2), ForeignKey("languages.code"))

    language = relationship("Language")
    contains_words = relationship("SentenceWord", back_populates="sentence")
    translations = relationship("SentenceTranslation", back_populates="source_sentence")


class SentenceWord(Base):
    __tablename__ = "sentence_words"
    sentence_id = mapped_column(Integer, ForeignKey("sentences.id"), primary_key=True)
    word_id = mapped_column(Integer, ForeignKey("words.id"), primary_key=True)

    sentence = relationship("Sentence", back_populates="contains_words")
    word = relationship("Word", back_populates="in_sentences")


class LearnedWord(Base):
    __tablename__ = "learned_words"
    user_id = mapped_column(Integer, ForeignKey("users.id"), primary_key=True)
    word_id = mapped_column(Integer, ForeignKey("words.id"), primary_key=True)
    target_language = mapped_column(String(2))  # Language being learned
    last_practiced = mapped_column(DateTime, default=datetime.utcnow)
    strength = mapped_column(Integer, default=0)  # 0-100 mastery level

    word = relationship("Word")




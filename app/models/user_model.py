
from datetime import datetime, timedelta
from typing import Optional, List
import secrets
import hashlib

from sqlalchemy import String, Boolean, DateTime, ForeignKey, func, Integer, Column, Text, ARRAY, Index, Date, CheckConstraint, Table
from sqlalchemy.orm import mapped_column, Mapped, relationship
from sqlalchemy.sql import func

from app.models.language_model import Language
from app.models.base_model import Base


from sqlalchemy.dialects.postgresql import ENUM as PGEnum



# Association table for friendships
friendship = Table(
    'friendships',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id', ondelete="CASCADE"), primary_key=True),
    Column('friend_id', Integer, ForeignKey('users.id', ondelete="CASCADE"), primary_key=True),
    Column('created_at', DateTime(timezone=True), server_default=func.now()),
    Column('status', String(20), default='pending')  # pending, accepted, blocked
)


class FriendshipRequest(Base):
    __tablename__ = "friendship_requests"

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    receiver_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))

    status = Column(String(20), default='pending')  # pending, accepted, rejected

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    # sender = relationship("UserModel", foreign_keys=[sender_id])
    # receiver = relationship("UserModel", foreign_keys=[receiver_id])

    sender = relationship("UserModel", foreign_keys=[sender_id], back_populates="sent_friend_requests")
    receiver = relationship("UserModel", foreign_keys=[receiver_id], back_populates="received_friend_requests")

    # Add check constraint for status
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'accepted', 'rejected')",
            name="check_friend_request_status"
        ),
    )

    def __repr__(self):
        return f"FriendshipRequest(id={self.id}, sender={self.sender_id}, receiver={self.receiver_id})"


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

    notes = relationship("NoteModel", back_populates="user", cascade="all, delete-orphan")

    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")

    # Chat relationships
    conversation_participations = relationship("ConversationParticipant", back_populates="user")
    sent_messages = relationship("Message", back_populates="sender")
    message_statuses = relationship("MessageStatus", back_populates="user")
    typing_indicators = relationship("TypingIndicator", back_populates="user")

    sent_friend_requests = relationship(
        "FriendshipRequest",
        foreign_keys="[FriendshipRequest.sender_id]",
        back_populates="sender"
    )

    received_friend_requests = relationship(
        "FriendshipRequest",
        foreign_keys="[FriendshipRequest.receiver_id]",
        back_populates="receiver"
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


class NoteModel(Base):
    __tablename__ = "notes"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True)

    # Core fields
    note_name: Mapped[str] = mapped_column(String(200), nullable=False)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Language and type
    target_lang: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    note_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="general"
    )

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Tags as PostgreSQL array
    tags: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(String(50)),
        nullable=True,
        default=[]
    )

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

    # Relationship
    user: Mapped["UserModel"] = relationship(back_populates="notes")

    def __repr__(self):
        return f"NoteModel(id={self.id}, name='{self.note_name}', type='{self.note_type}', user_id={self.user_id})"


######################### For chatting

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)

    first_name = Column(String(100))
    middle_name = Column(String(100))
    last_name = Column(String(100))

    date_of_birth = Column(Date)
    age = Column(Integer)  # Can be calculated from date_of_birth or stored

    country = Column(String(100))
    city = Column(String(100))

    gender = Column(String(20))  # male, female, other, prefer_not_to_say
    bio = Column(Text)

    profile_image_url = Column(String(500))
    cover_image_url = Column(String(500))

    phone_number = Column(String(20))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    user = relationship("UserModel", back_populates="profile", uselist=False)

    # Add check constraint for gender
    __table_args__ = (
        CheckConstraint(
            "gender IN ('male', 'female', 'other', 'prefer_not_to_say')",
            name="check_gender_values"
        ),
    )

    def __repr__(self):
        return f"UserProfile(id={self.id}, user_id={self.user_id})"



class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    is_group = Column(Boolean, default=False)
    group_name = Column(String(200), nullable=True)
    group_image_url = Column(String(500), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    participants = relationship("ConversationParticipant", back_populates="conversation", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"Conversation(id={self.id}, is_group={self.is_group})"


class ConversationParticipant(Base):
    __tablename__ = "conversation_participants"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))

    # Additional participant info
    joined_at = Column(DateTime(timezone=True), server_default=func.now())
    left_at = Column(DateTime(timezone=True), nullable=True)

    # For groups: admin status, mute settings
    is_admin = Column(Boolean, default=False)
    is_muted = Column(Boolean, default=False)

    # Last read position
    last_read_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="participants")
    user = relationship("UserModel")
    last_read_message = relationship("Message", foreign_keys=[last_read_message_id])

    __table_args__ = (CheckConstraint(
        "conversation_id IS NOT NULL AND user_id IS NOT NULL",
        name="uq_conversation_participant"
    ),)

    def __repr__(self):
        return f"Participant(id={self.id}, conversation={self.conversation_id}, user={self.user_id})"


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"))
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))

    content = Column(Text, nullable=True)  # Nullable for media-only messages
    message_type = Column(String(20), default='text')  # text, image, video, audio, file

    # For media messages
    media_url = Column(String(500), nullable=True)
    media_thumbnail_url = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)

    # Message status
    is_edited = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)

    # For replies
    reply_to_message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    sender = relationship("UserModel")
    reply_to = relationship("Message", remote_side=[id], backref="replies")

    # Add check constraint for message_type
    __table_args__ = (
        CheckConstraint(
            "message_type IN ('text', 'image', 'video', 'audio', 'file')",
            name="check_message_type"
        ),
    )

    def __repr__(self):
        return f"Message(id={self.id}, conversation={self.conversation_id}, sender={self.sender_id})"


class MessageStatus(Base):
    __tablename__ = "message_status"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))

    status = Column(String(20), default='sent')  # sent, delivered, read

    read_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    message = relationship("Message", backref="statuses")
    user = relationship("UserModel")

    # Add unique constraint and check constraint
    __table_args__ = (
        CheckConstraint(
            "status IN ('sent', 'delivered', 'read')",
            name="check_delivery_status"
        ),
    )

    def __repr__(self):
        return f"MessageStatus(id={self.id}, message={self.message_id}, user={self.user_id}, status={self.status})"


class TypingIndicator(Base):
    __tablename__ = "typing_indicators"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))

    is_typing = Column(Boolean, default=False)
    last_updated = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    conversation = relationship("Conversation")
    user = relationship("UserModel")

    def __repr__(self):
        return f"TypingIndicator(id={self.id}, conversation={self.conversation_id}, user={self.user_id})"
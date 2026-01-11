

from pydantic import BaseModel, EmailStr, field_validator, Field
from typing import Optional, Dict, Any
from datetime import date
import re


class UserRegisterSchema(BaseModel):

    username: str | None
    email: EmailStr()
    password: str
    native: str

    @field_validator('password')
    @classmethod
    def validate_password(cls, value: str) -> str:

        if len(value.strip()) < 8:
            raise ValueError("Password too short (min 8 chars)")
        if value.lower() in ["password", "12345678"]:
            raise ValueError("Password too common")
        if ' ' in value:
            raise ValueError("Password can't contain space")

        return value


class UserTokenSchema(BaseModel):
    sub: str
    email: EmailStr()
    project_id: int


class UserLoginSchema(BaseModel):

    email: EmailStr()
    password: str

    @field_validator('password')
    @classmethod
    def validate_password(cls, value: str) -> str:

        if len(value.strip()) < 8:
            raise ValueError("Password too short (min 8 chars)")
        if value.lower() in ["password", "12345678"]:
            raise ValueError("Password too common")
        if ' ' in value:
            raise ValueError("Password can't contain space")

        return value


class NativeLangSchema(BaseModel):

    native: str


class ChooseLangSchema(BaseModel):

    target_language_code: str


class ChangeWordStatusSchema(BaseModel):
    word_id: int
    action: str


from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
import re

class EditUserProfileSchema(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    middle_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    date_of_birth: Optional[str] = Field(None)
    age: Optional[str] = Field(None, max_length=3)  # Changed to string
    country: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=100)
    gender: Optional[str] = Field(None)
    bio: Optional[str] = Field(None, max_length=2000)
    phone_number: Optional[str] = Field(None, max_length=20)
    # profile_image_url: Optional[str] = Field(None, max_length=500)
    # cover_image_url: Optional[str] = Field(None, max_length=500)

    @field_validator('date_of_birth')
    @classmethod
    def validate_date_of_birth(cls, v):
        if v is None or v == '':
            return None
        try:
            date.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Date of birth must be in YYYY-MM-DD format")

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v is None or v == '':
            return None
        valid_genders = ['male', 'female', 'other', 'prefer_not_to_say']
        if v not in valid_genders:
            raise ValueError(f"Gender must be one of: {', '.join(valid_genders)}")
        return v

    class Config:
        from_attributes = True


from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class UserResponseSchema(BaseModel):
    """Standardized user response format"""
    sub: str
    email: Optional[str] = None
    username: Optional[str] = None
    native: Optional[str] = None
    learning_targets: Optional[List[str]] = None
    profile: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class AuthResponseSchema(BaseModel):
    """Standardized authentication response format"""
    user: UserResponseSchema
    access_token: str
    refresh_token: Optional[str] = None


class ProfileUpdateResponseSchema(BaseModel):
    """Profile update response format"""
    status: str
    message: str
    data: Dict[str, Any]
    user: Optional[UserResponseSchema] = None
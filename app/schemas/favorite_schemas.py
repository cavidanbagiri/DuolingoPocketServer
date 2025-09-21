# schemas/favorite.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


# Category Schemas
class FavoriteCategoryBase(BaseModel):
    name: str


class FavoriteCategoryCreate(FavoriteCategoryBase):
    pass


class FavoriteCategoryUpdate(FavoriteCategoryBase):
    pass


class FavoriteCategoryResponse(FavoriteCategoryBase):
    id: int
    user_id: int
    word_count: Optional[int] = 0  # Will be computed
    name: str | None
    color: str | None
    icon: str | None

    class Config:
        from_attributes = True


# Favorite Word Schemas
class FavoriteWordBase(BaseModel):
    from_lang: str = Field(..., min_length=2, max_length=50, example="en")
    to_lang: str = Field(..., min_length=2, max_length=50, example="ru")
    original_text: str = Field(..., min_length=1, max_length=500, example="hello")
    translated_text: str = Field(..., min_length=1, max_length=500, example="привет")
    category_id: Optional[int] = Field(None, ge=1, description="Category ID for organization")


class FavoriteWordCreate(FavoriteWordBase):
    pass


class FavoriteWordUpdate(BaseModel):
    category_id: Optional[int] = None


class FavoriteWordResponse(BaseModel):
    status: str
    message: str
    action: str
    favorite_id: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Word added to favorites",
                "action": "created",
                "favorite_id": 123
            }
        }


# Default Category Schemas
class DefaultCategoryBase(BaseModel):
    name: str
    description: Optional[str] = None


class DefaultCategoryCreate(DefaultCategoryBase):
    pass


class DefaultCategoryResponse(DefaultCategoryBase):
    id: int
    user_id: int

    class Config:
        from_attributes = True


# Response schemas for relationships
class FavoriteCategoryWithWords(FavoriteCategoryResponse):
    favorite_words: List[FavoriteWordResponse] = []


class UserFavoritesResponse(BaseModel):
    categories: List[FavoriteCategoryResponse]
    uncategorized_words: List[FavoriteWordResponse]
    default_categories: List[DefaultCategoryResponse]



# schemas.py
class FavoriteFetchWordResponse(BaseModel):
    id: int
    original_text: str
    translated_text: str
    from_lang: str
    to_lang: str
    category_id: Optional[int]

    class Config:
        from_attributes = True

class CategoryWordsResponse(BaseModel):
    category_id: int
    category_name: str
    word_count: int
    words: List[FavoriteWordResponse]

    class Config:
        from_attributes = True



class MoveWordRequest(BaseModel):
    target_category_id: int = Field(..., gt=0, description="ID of the target category")

class MoveWordResponse(BaseModel):
    status: str
    message: str
    word_id: int
    old_category_id: int
    new_category_id: int

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Word moved successfully",
                "word_id": 123,
                "old_category_id": 1,
                "new_category_id": 2
            }
        }
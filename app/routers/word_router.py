import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import Response
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.token_handler import TokenHandler
from app.database.setup import get_db
from app.repositories.word_repository import FetchWordRepository, \
    ChangeWordStatusRepository, DetailWordRepository, GetStatisticsForDashboardRepository, GetPosStatisticsRepository, \
    VoiceHandleRepository, GenerateAIWordRepository, GenerateAIQuestionRepository, SearchRepository, \
    TranslateRepository, AddFavoritesRepository, CreateNewFavoriteCategoryRepository, FavoriteCategoryRepository, \
    CategoryWordsRepository, DeleteFavoriteWordRepository, MoveFavoriteWordRepository, DeleteCategoryRepository, \
    SearchFavoriteRepository
from app.schemas.user_schema import ChangeWordStatusSchema
from app.schemas.word_schema import VoiceSchema, GenerateAIWordSchema, GenerateAIChatSchema, TranslateSchema
from app.schemas.favorite_schemas import (FavoriteWordBase, FavoriteWordResponse, FavoriteCategoryBase, FavoriteCategoryResponse,
                                          FavoriteFetchWordResponse, CategoryWordsResponse, MoveWordResponse, MoveWordRequest)

from app.repositories.structure_repository import CreateMainStructureRepository


router = APIRouter()

from app.logging_config import setup_logger
logger = setup_logger(__name__, "word.log")





# Router for executing the creation update and generate the english words
@router.post('/create', status_code=200)
async def update_words(db: AsyncSession = Depends(get_db)):
    try:
        repo = CreateMainStructureRepository(db)
        result = await repo.create_main_structure()
        await db.commit()
        return {"message": result}
    except Exception as ex:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(ex))




@router.get('/get_statistics', status_code=200)
async def get_statistics(db: AsyncSession = Depends(get_db),
                         user_info = Depends(TokenHandler.verify_access_token)):
    try:
        repo = GetStatisticsForDashboardRepository(db, user_id=int(user_info.get('sub')))
        result = await repo.get_statistics()
        return result
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))



# endpoints.py
@router.get("/user/languages")
async def get_user_languages(
    user_info: int = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db)
):
    repo = FetchWordRepository(db, user_id=int(user_info.get('sub')))
    languages = await repo.get_available_languages()
    return languages



@router.get('/search-test', status_code=201)
async def search_word(
    native_language: str,
    target_language: str,
    query: str,
    db: AsyncSession = Depends(get_db),
    user_info = Depends(TokenHandler.verify_access_token)
):
    try:
        repo = SearchRepository(db=db, user_id=int(user_info.get('sub')))
        result = await repo.search(native_language=native_language, target_language=target_language, query=query,)  # Fixed: pass actual query and language
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in search for query '{query}': {str(e)}")  # Fixed error message
        logger.error(f"Unexpected error in search for query '{query}': {str(e)}")  # Fixed error message
        raise HTTPException(
            status_code=500,
            detail="We're having trouble processing your search. Please try again in a moment."
        )


@router.post('/translate', status_code=201)
async def translate(data: TranslateSchema,
                    repo: TranslateRepository = Depends(TranslateRepository),):

    try:
        # repo = TranslateRepository()
        result = await repo.translate(data = data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in translate : {str(e)}")  # Fixed error message
        logger.error(f"Unexpected error in translate for query : {str(e)}")  # Fixed error message
        raise HTTPException(
            status_code=500,
            detail="We're having trouble processing your translate. Please try again in a moment."
        )





@router.get("/{language_code}")
async def get_words_for_language(
    language_code: str,
    only_starred: bool = False,
    only_learned: bool = False,
    skip: int = 0,
    limit: int = 50,
    user_info: int = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db)
):
    repo = FetchWordRepository(db, user_id=int(user_info.get('sub')))
    words = await repo.fetch_words_for_language(
        language_code, only_starred, only_learned, skip, limit
    )
    return words



@router.post('/setstatus', status_code=200)
async def set_word_status(data: ChangeWordStatusSchema, db:
                            AsyncSession = Depends(get_db),
                            user_info = Depends(TokenHandler.verify_access_token)):
    try:
        repo = ChangeWordStatusRepository(db=db,
                                          word_id=data.word_id,
                                          action=data.action,
                                          user_id=int(user_info.get('sub')))
        result = await repo.set_word_status()
        return result
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@router.post('/voice', response_class=Response, status_code=200)
async def handle_voice(
        data: VoiceSchema,
        voice_repo: VoiceHandleRepository = Depends()
):
    """
    Generate speech for given text and return an MP3 audio file.
    """
    try:
        # Get audio bytes from Yandex SpeechKit
        audio_bytes = await voice_repo.generate_speech(data.text, data.language)

        # Return the audio file directly in the response
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg"  # Use "audio/ogg" for oggopus format
        )

    except HTTPException:
        # Re-raise HTTPExceptions from the repository
        raise
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Unexpected error in /voice: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")



@router.post('/generateaiword',  status_code=200)
async def generate_ai_for_word(
        data: GenerateAIWordSchema,
        repo: GenerateAIWordRepository = Depends()
):
    """
    Generate comprehensive AI-powered language learning content.

    Returns detailed information including:
    - Definition and pronunciation
    - 5+ example sentences with translations
    - Usage contexts and common phrases
    - Grammar tips and cultural notes
    - Motivational message
    - Difficulty level assessment
    """
    try:
        result = await repo.generate_ai_for_word_with_fallback(data)

        # Log successful generation
        logger.info(f"Successfully generated AI content for word: {data.text}")

        print(f'the coming result is {result}')

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing word '{data.text}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble generating content right now. Please try again in a moment."
        )





@router.post('/aichat', status_code=200)
async def generate_ai_chat(data: GenerateAIChatSchema, repo: GenerateAIQuestionRepository = Depends()):
    try:
        # The repo method now needs to handle a conversational prompt
        result = await repo.generate_ai_chat(data)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in AI chat for word '{data.word}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble processing your question. Please try again in a moment."
        )



@router.get('/get_detail_word/{word_id}', status_code=200)
async def get_detail_word(word_id: int,
                          db: AsyncSession = Depends(get_db),
                          user_info = Depends(TokenHandler.verify_access_token)):

    try:
        repo = DetailWordRepository(db=db, word_id=word_id, user_id=int(user_info.get('sub')))
        result = await repo.get_word_detail()
        return result
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))




@router.get('/get_pos_statistics', status_code=200)
async def get_pos_statistics(db: AsyncSession = Depends(get_db),
                             user_info = Depends(TokenHandler.verify_access_token)):

    try:
        repo = GetPosStatisticsRepository(db=db, user_id=int(user_info.get('sub')))
        result = await repo.get_pos_statistics()
        return result

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))





@router.post('/add_favorites', status_code=200, response_model=FavoriteWordResponse)
async def add_favorites(
    data: FavoriteWordBase,
    db: AsyncSession = Depends(get_db),
    user_info = Depends(TokenHandler.verify_access_token)
):
    try:
        repo = AddFavoritesRepository(data=data, db=db, user_id=int(user_info.get('sub')))
        result = await repo.add_favorites()
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error adding word to favorites: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble adding words right now"
        )


@router.post('/favorites/categories', status_code=201, response_model=dict)
async def create_new_category(
        data: FavoriteCategoryBase,
        db: AsyncSession = Depends(get_db),
        user_info: dict = Depends(TokenHandler.verify_access_token)
):
    """
    Create a new favorite category for the authenticated user.

    - **name**: Category name (required, unique per user)
    """
    try:
        repo = CreateNewFavoriteCategoryRepository(
            data=data,
            db=db,
            user_id=int(user_info.get('sub'))
        )
        result = await repo.create_new_category()
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating new category: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble creating new categories"
        )



@router.get('/favorites/categories', response_model=List[FavoriteCategoryResponse])
async def get_user_categories(
    db: AsyncSession = Depends(get_db),
    user_info: dict = Depends(TokenHandler.verify_access_token)
):
    """
    Get all favorite categories for the authenticated user.
    Returns categories with word counts.
    """
    print('here work')
    try:
        repo = FavoriteCategoryRepository(db=db, user_id=int(user_info.get('sub')))
        categories = await repo.get_user_categories()
        return categories

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching categories: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble fetching your categories"
        )


# router.py
@router.get('/favorites/categories/{category_id}/words')
async def get_category_words(
        category_id: int,
        db: AsyncSession = Depends(get_db),
        user_info: dict = Depends(TokenHandler.verify_access_token)
):
    """
    Get all words in a specific category for the authenticated user.

    - **category_id**: ID of the category to fetch words from
    """
    try:
        repo = CategoryWordsRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            category_id=category_id
        )
        result = await repo.get_category_words()
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching category words: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble fetching category words"
        )





@router.delete('/favorites/words/{word_id}', status_code=200)
async def delete_favorite_word(
    word_id: int,
    db: AsyncSession = Depends(get_db),
    user_info: dict = Depends(TokenHandler.verify_access_token)
):
    """
    Delete a word from user's favorites.
    """
    try:
        repo = DeleteFavoriteWordRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            word_id=word_id
        )
        result = await repo.delete_word()
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting word: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble deleting the word"
        )




@router.put('/favorites/words/{word_id}/move', response_model=MoveWordResponse)
async def move_word_to_category(
    word_id: int,
    move_data: MoveWordRequest,
    db: AsyncSession = Depends(get_db),
    user_info: dict = Depends(TokenHandler.verify_access_token)
):
    """
    Move a word to a different category.
    """
    try:
        repo = MoveFavoriteWordRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            word_id=word_id
        )
        result = await repo.move_word(move_data.target_category_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error moving word: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble moving the word"
        )


@router.delete('/favorites/categories/delete/{category_id}', status_code=200)
async def delete_category(
    category_id: int,
    db: AsyncSession = Depends(get_db),
    user_info: dict = Depends(TokenHandler.verify_access_token)
):
    """
    Delete a category and handle its words (move to default or delete).
    """
    try:
        repo = DeleteCategoryRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            category_id=category_id
        )
        result = await repo.delete_category()
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting category: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble deleting the category"
        )



# router.py
# @router.get('/favorites/search', response_model=List[FavoriteWordResponse])
@router.get('/favorites/search', status_code=200)
async def search_favorites(
    query: str = Query(..., min_length=1, description="Search query"),
    category_id: Optional[int] = Query(None, description="Filter by category ID"),
    db: AsyncSession = Depends(get_db),
    user_info: dict = Depends(TokenHandler.verify_access_token)
):
    """
    Search across user's favorite words.
    Optionally filter by specific category.
    """
    try:
        repo = SearchFavoriteRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            query=query,
            category_id=category_id
        )
        results = await repo.search_words()
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble with the search"
        )



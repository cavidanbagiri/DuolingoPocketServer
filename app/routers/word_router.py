import os
import asyncio
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import Response
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse

from app.auth.token_handler import TokenHandler
from app.database.setup import get_db
from app.repositories.word_repository import (FetchWordRepository, \
                                              ChangeWordStatusRepository, DetailWordRepository,
                                              GetStatisticsForDashboardRepository, GetPosStatisticsRepository, \
                                              VoiceHandleRepository, GenerateAIWordRepository,
                                              GenerateAIQuestionRepository, SearchRepository, \
                                              TranslateRepository, AddFavoritesRepository,
                                              CreateNewFavoriteCategoryRepository, FavoriteCategoryRepository, \
                                              CategoryWordsRepository, DeleteFavoriteWordRepository,
                                              MoveFavoriteWordRepository, DeleteCategoryRepository, \
                                              SearchFavoriteRepository, FetchStatisticsForProfileRepository,
                                              DailyStreakRepository, GenerateDirectAIChat, DirectChatContextRepository,
                                              FetchWordCategoriesRepository,
                                              FetchWordByCategoryIdRepository, FetchWordByPosRepository,
                                              FetchDirectAiChatContext, GetDailyStreakRepository,
                                              FetchActiveLangRepository
                                              )
from app.schemas.user_schema import ChangeWordStatusSchema
from app.schemas.word_schema import VoiceSchema, GenerateAIWordSchema, TranslateSchema, AiDirectChatSchema, STTRequest
from app.schemas.conversation_contexts_schema import GenerateAIChatSchema

from app.schemas.favorite_schemas import (FavoriteWordBase, FavoriteWordResponse, FavoriteCategoryBase, FavoriteCategoryResponse,
                                          FavoriteFetchWordResponse, CategoryWordsResponse, MoveWordResponse, MoveWordRequest)

from app.repositories.structure_repository import CreateMainStructureRepository, GenerateEnglishSentence, TranslateEnglishSentencesRepository

from app.services.ai_service import AIService

from app.schemas.note_schemas import NoteBase, NoteCreate, NoteUpdate, NoteResponse



# from google.cloud import speech_v1 as speech
from google.cloud import speech
from google.oauth2 import service_account
import base64
import json


router = APIRouter()

from app.logging_config import setup_logger
logger = setup_logger(__name__, "word.log")


@router.get('/get_statistics', status_code=200)
async def get_statistics(db: AsyncSession = Depends(get_db),
                         user_info = Depends(TokenHandler.verify_access_token)):
    try:
        repo = GetStatisticsForDashboardRepository(db, user_id=int(user_info.get('sub')))
        result = await repo.get_statistics()
        return result
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))



@router.get('/user/statistics/daily_streak', status_code=200)
async def fetch_daily_streak(
        db: AsyncSession = Depends(get_db),
        user_info=Depends(TokenHandler.verify_access_token)
):
    """
    Get user's current daily streak count.

    Returns: Integer representing consecutive days of learning 20+ words.
    Example: 3 means user learned 20+ words for 3 consecutive days.
    Returns 0 if no current streak.
    """
    try:
        repo = GetDailyStreakRepository(db, user_id=int(user_info.get('sub')))

        # Choose which implementation to use
        # result = await repo.fetch_daily_streak()  # Simple approach
        result = await repo.fetch_daily_streak_optimized()  # More efficient

        return {"daily_streak": result}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))




@router.get('/user/statistics/active_lang', status_code=200)
async def fetch_active_lang(
        db: AsyncSession = Depends(get_db),
        user_info=Depends(TokenHandler.verify_access_token)
):
    """
    Count languages user has actively learned in the last week.

    Returns: Number of active languages (with at least one learned word in last 7 days)
    """
    try:
        repo = FetchActiveLangRepository(db, user_id=int(user_info.get('sub')))

        # Basic count only
        active_count = await repo.fetch_active_lang()

        return {"active": active_count}

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@router.get("/user/languages")
async def get_user_languages(
    user_info: int = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db)
):
    repo = FetchWordRepository(db, user_id=int(user_info.get('sub')))
    languages = await repo.get_available_languages()
    return languages

@router.get("/main/words")
async def get_words(
    language_code: str = Query(..., description="Language code"),
    only_starred: bool = Query(False, description="Filter only starred words"),
    only_learned: bool = Query(False, description="Filter only learned words"),
    skip: int = Query(0, description="Pagination offset"),
    limit: int = Query(20, description="Pagination limit"),
    user_info: dict = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db)
):
    try:
        repo = FetchWordRepository(db, user_id=int(user_info.get('sub')))
        result = await repo.fetch_words_for_language(
            lang_code=language_code,
            only_starred=only_starred,
            only_learned=only_learned,
            skip=skip,
            limit=limit
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during words fetching: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble with the words fetching"
        )


@router.get('/main/fetch_words_by_categories')
async def fetch_words_by_category_id(
    category_id: int = Query(..., description="Filter by category ID"),
    lang_code: str = Query(..., description="Language code for the words"),
    only_starred: bool = Query(False, description="Filter only starred words"),
    only_learned: bool = Query(False, description="Filter only learned words"),
    skip: int = Query(0, description="Pagination offset"),
    limit: int = Query(20, description="Pagination limit"),
    db: AsyncSession = Depends(get_db),
    user_info: dict = Depends(TokenHandler.verify_access_token)
):
    try:
        repo = FetchWordByCategoryIdRepository(
            db,
            user_id=int(user_info.get('sub')),
            category_id=category_id,
            lang_code=lang_code,
            only_starred=only_starred,
            only_learned=only_learned,
            skip=skip,
            limit=limit
        )
        data = await repo.fetch_words_by_category_id()
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during fetching words by category id: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble with the fetching words by category id"
        )




@router.get('/main/fetch_words_by_posname')
async def fetch_words_by_pos_name(
    pos_name: str = Query(..., description="Filter by Pos name"),
    lang_code: str = Query(..., description="Language code for the words"),
    only_starred: bool = Query(False, description="Filter only starred words"),
    only_learned: bool = Query(False, description="Filter only learned words"),
    skip: int = Query(0, description="Pagination offset"),
    limit: int = Query(20, description="Pagination limit"),
    db: AsyncSession = Depends(get_db),
    user_info: dict = Depends(TokenHandler.verify_access_token)
):
    try:
        repo = FetchWordByPosRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            pos_name=pos_name,
            lang_code=lang_code,
            only_starred=only_starred,
            only_learned=only_learned,
            skip=skip,
            limit=limit
        )
        data = await repo.fetch_words_by_pos()
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during fetching words by pos name: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble with the fetching words by pos name"
        )



@router.get('/search-test', status_code=201)
async def search_word(
    target_language: str,
    query: str,
    native_language: str | None = None,
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
        result = await repo.generate_ai_for_word(data)

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



@router.post('/aichat_stream')
async def generate_ai_chat_stream(
        data: GenerateAIChatSchema,
        db: AsyncSession = Depends(get_db),
        user_info=Depends(TokenHandler.verify_access_token)
):
    """
    Stream AI chat response with context management.
    The AI will remember previous conversations about the same word.
    """
    try:
        # Get user_id from token
        user_id = int(user_info.get('sub'))

        # Initialize repository with database session
        repo = GenerateAIQuestionRepository(db)

        print(f'User ID from token: {user_id}')
        print(f'Processing word: {data.word}, language: {data.language}')

        return StreamingResponse(
            repo.generate_ai_chat_stream(user_id, data),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Important for streaming
            }
        )
    except Exception as e:
        logger.error(f"Streaming chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Streaming service error")


@router.post('/clear_context')
async def clear_context(
        word: str,
        language: str,
        db: AsyncSession = Depends(get_db),
        user_info=Depends(TokenHandler.verify_access_token)  # Add auth here too
):
    """
    Clear context for a specific word.
    Call this when user changes words or wants to start fresh.
    """
    try:
        # Get user_id from token
        user_id = int(user_info.get('sub'))

        repo = GenerateAIQuestionRepository(db)
        success = await repo.clear_word_context(user_id, word, language)

        return {
            "success": success,
            "message": "Context cleared" if success else "No context to clear"
        }

    except Exception as e:
        logger.error(f"Error clearing context: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear context")


@router.get('/wordai/conversation_history')
async def get_conversation_history(
        word: str,
        language: str,
        db: AsyncSession = Depends(get_db),
        user_info=Depends(TokenHandler.verify_access_token)
):
    """
    Get conversation history for a specific word.
    Useful for displaying previous conversation in UI.
    """
    # print('here is working................................')
    try:
        # Get user_id from token
        user_id = int(user_info.get('sub'))

        repo = GenerateAIQuestionRepository(db)
        history = await repo.get_conversation_history(user_id, word, language)

        return {
            "word": word,
            "language": language,
            "history": history or []
        }

    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get conversation history")




@router.get('/active_context')
async def get_active_context(
        db: AsyncSession = Depends(get_db),
        user_info=Depends(TokenHandler.verify_access_token)
):
    """
    Get the current active context for the authenticated user.
    Useful for frontend to know which word the user is currently discussing.
    """
    try:
        # Get user_id from token
        user_id = int(user_info.get('sub'))

        repo = GenerateAIQuestionRepository(db)

        # Get the current active context
        active_context = None
        if repo.context_repo:
            active_context = await repo.context_repo.get_user_active_context(user_id)

        if active_context:
            return {
                "has_active_context": True,
                "word": active_context.word,
                "language": active_context.language,
                "native_language": active_context.native_language,
                "last_updated": active_context.updated_at.isoformat(),
                "message_count": len(json.loads(active_context.messages)) if active_context.messages else 0
            }
        else:
            return {
                "has_active_context": False,
                "message": "No active conversation context"
            }

    except Exception as e:
        logger.error(f"Error getting active context: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get active context")



# router.py
@router.post('/ai_direct/fetch')
async def ai_direct_fetch_context(
        db: AsyncSession = Depends(get_db),
        user_info=Depends(TokenHandler.verify_access_token)
):
    try:
        # Get user_id from token
        user_id = int(user_info.get('sub'))

        repo = FetchDirectAiChatContext(db, user_id)
        result = await repo.ai_direct_fetch_context()


        return result

    except Exception as e:
        logger.error(f"Error getting direct chat context: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get chat context")



@router.post('/ai_direct_chat_stream')
async def ai_direct_chat_stream(
        data: AiDirectChatSchema,
        db: AsyncSession = Depends(get_db),  # Add database dependency
        user_info=Depends(TokenHandler.verify_access_token)  # Add authentication
):
    """
    Streaming AI Direct Chat Endpoint with context management
    Returns responses as they're generated, remembers conversation history
    """
    try:
        # Get user_id from token
        user_id = int(user_info.get('sub'))

        # Add user_id to data
        data.user_id = user_id

        logger.info(f"Streaming AI chat request from user {user_id}: {data.message[:50]}...")

        # Initialize repository with database
        repo = GenerateDirectAIChat(db)

        return StreamingResponse(
            repo.ai_direct_chat_stream(data),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as ex:
        logger.exception(f"Streaming chat error: {str(ex)}")
        raise HTTPException(status_code=500, detail="Streaming service error")


@router.post("/direct-chat/clear-history")
async def clear_direct_chat_history(
        db: AsyncSession = Depends(get_db),
        user_info=Depends(TokenHandler.verify_access_token)
):
    """Clear user's direct chat conversation history (start fresh)"""
    try:
        user_id = int(user_info.get('sub'))

        repo = DirectChatContextRepository(db)
        success = await repo.clear_context_messages(user_id)

        if success:
            return {
                "success": True,
                "message": "Direct chat history cleared successfully"
            }
        else:
            return {
                "success": False,
                "message": "No direct chat context found to clear"
            }

    except Exception as e:
        logger.error(f"Error clearing direct chat history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear chat history")


@router.get("/direct-chat/stats")
async def get_direct_chat_stats(
        db: AsyncSession = Depends(get_db),
        user_info=Depends(TokenHandler.verify_access_token)
):
    """Get statistics about user's direct chat"""
    try:
        user_id = int(user_info.get('sub'))

        repo = DirectChatContextRepository(db)
        stats = await repo.get_user_stats(user_id)

        return {
            "success": True,
            "user_id": user_id,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error getting direct chat stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get chat stats")




# Add this code block for Google Stt
@router.post("/google/stt")
async def speech_to_text(request: STTRequest):
    """
    Convert speech to text using Google Cloud Speech-to-Text
    """
    try:
        # Get Google credentials from environment
        credentials_info = {
            "type": "service_account",
            "project_id": os.getenv("GOOGLE_PROJECT_ID"),
            "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("GOOGLE_PRIVATE_KEY", "").replace("\\n", "\n"),
            "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "auth_uri": os.getenv("GOOGLE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": os.getenv("GOOGLE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
            "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_X509_CERT_URL",
                                                     "https://www.googleapis.com/oauth2/v1/certs"),
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('GOOGLE_CLIENT_EMAIL', '').replace('@', '%40')}",
            "universe_domain": os.getenv("GOOGLE_UNIVERSE_DOMAIN", "googleapis.com")
        }

        # Validate credentials
        required_fields = ["project_id", "private_key", "client_email"]
        for field in required_fields:
            if not credentials_info.get(field):
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing Google credential: {field}"
                )

        # Create credentials
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        client = speech.SpeechClient(credentials=credentials)

        # Decode base64 audio
        audio_content = base64.b64decode(request.audio)

        # Configure recognition for language learning
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code=request.language_code,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,  # Useful for pronunciation analysis
            model="latest_short",  # Best for short phrases
            use_enhanced=True,  # Enhanced model for better accuracy
            speech_contexts=[{
                "phrases": [
                    # Add common language learning phrases for better recognition
                    "hello", "goodbye", "thank you", "please",
                    "excuse me", "how are you", "I don't understand"
                ],
                "boost": 10.0  # Boost these phrases
            }]
        )

        audio = speech.RecognitionAudio(content=audio_content)
        response = client.recognize(config=config, audio=audio)

        # Extract results
        transcripts = []
        confidence_scores = []
        word_details = []

        for result in response.results:
            alternative = result.alternatives[0]
            transcripts.append(alternative.transcript)
            confidence_scores.append(alternative.confidence)

            # Extract word-level details for pronunciation analysis
            if alternative.words:
                for word_info in alternative.words:
                    word_details.append({
                        "word": word_info.word,
                        "start_time": word_info.start_time.total_seconds(),
                        "end_time": word_info.end_time.total_seconds(),
                        "confidence": word_info.confidence
                    })

        full_transcript = " ".join(transcripts)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        # Return only JSON-serializable data
        return {
            "success": True,
            "transcript": full_transcript,
            "confidence": avg_confidence,
            "language": request.language_code,
            "word_details": word_details,
        }

    except Exception as e:
        print(f"Google STT error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Speech recognition failed: {str(e)}"
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



@router.get('/statistics/get_pos_statistics', status_code=200)
async def get_pos_statistics(
                            lang_code: str,
                            db: AsyncSession = Depends(get_db),
                            user_info = Depends(TokenHandler.verify_access_token)):

    try:
        repo = GetPosStatisticsRepository(db=db, user_id=int(user_info.get('sub')), lang_code=lang_code)
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



@router.get('/profile/fetch_statistics', status_code=200, response_model=Dict[str, Any])
async def search_statistics_for_profile(
        db: AsyncSession = Depends(get_db),
        user_info: dict = Depends(TokenHandler.verify_access_token)
):
    """
    Fetch user statistics: username, email, total learned words, and days registered
    """
    try:
        repository = FetchStatisticsForProfileRepository(db, user_id=int(user_info.get('sub')))
        data = await repository.fetch_statistics()
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during profile fetch statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble with the profile fetch statistics"
        )


@router.get('/user/daily_statistics', status_code=200, response_model=Dict[str, Any])
async def search_statistics_for_profile(
        db: AsyncSession = Depends(get_db),
        user_info: dict = Depends(TokenHandler.verify_access_token)
):
    """
    Fetch daily streak statistics: last learned language, daily learned words
    """
    try:
        repository = DailyStreakRepository(db, user_id=int(user_info.get('sub')))
        data = await repository.daily_streak()
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during header fetch daily streak: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble with the header fetch daily streak"
        )


@router.get('/main/words_categories', status_code=200)
async def fetch_words_categories(
    lang_code: Optional[str] = Query(),
    user_info: dict = Depends(TokenHandler.verify_access_token),
    db: AsyncSession = Depends(get_db),
):

    try:
        repo = FetchWordCategoriesRepository(db, user_id=int(user_info.get('sub')), lang_code= lang_code)
        data = await repo.fetch_words_categories()
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during word categories fetching: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="We're having trouble with the word categories fetching"
        )


import asyncio

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import Response
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.token_handler import TokenHandler
from app.database.setup import get_db
from app.repositories.word_repository import FetchWordRepository, \
    ChangeWordStatusRepository, DetailWordRepository, GetStatisticsForDashboardRepository, GetPosStatisticsRepository, \
    VoiceHandleRepository, GenerateAIWordRepository, GenerateAIQuestionRepository
from app.schemas.user_schema import ChangeWordStatusSchema
from app.schemas.word_schema import VoiceSchema, GenerateAIWordSchema, GenerateAIChatSchema

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

        print('coming result is {}........................'.format(result))

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

















# @router.post('/create', status_code=201)
# async def create_main_structure(db: AsyncSession = Depends(get_db)):
#
#     try:
#         repository = CreateMainStructureRepository(db)
#         data = await repository.create_main_structure()
#         return data
#     except HTTPException as ex:
#         raise ex
#     except Exception as ex:
#         logger.error(f'Fetch Dashboard {ex}')
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f'Internal Server Error {ex}'
#         )


# @router.post('/create', status_code=201)
# async def create_main_structure(db: AsyncSession = Depends(get_db)):
#     try:
#         # Create repository with the session
#         repository = CreateMainStructureRepository(db)
#         data = await repository.create_main_structure()
#
#         # Explicitly commit the transaction
#         await db.commit()
#
#         return data
#     except HTTPException as ex:
#         await db.rollback()
#         raise ex
#     except Exception as ex:
#         await db.rollback()
#         logger.error(f'Error creating main structure: {ex}')
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f'Internal Server Error: {str(ex)}'
#         )
#     finally:
#         await db.close()




# import asyncio
# from typing import Annotated
#
# from fastapi import APIRouter, HTTPException, status, Query
# from fastapi.params import Depends
# from sqlalchemy.ext.asyncio import AsyncSession
#
# from app.auth.token_handler import TokenHandler
# from app.database.setup import get_db
# from app.repositories.word_repository import SaveWordRepository, DashboardRepository, DashboardRepositoryLang, \
#     FilterRepository, ChangeWordStatusRepository
# from app.schemas.translate_schema import WordSchema
# from app.schemas.word_schema import ChangeWordStatusSchema
#
# router = APIRouter()
#
#
# from app.logging_config import setup_logger
# logger = setup_logger(__name__, "word.log")
#
#
# @router.get('/dashboard/stats', status_code=200)
# async def get_language_pair_stats(db:Annotated[AsyncSession, Depends(get_db)],
#                     user_info = Depends(TokenHandler.verify_access_token)
#                     ):
#     try:
#         repository = DashboardRepository(user_info.get('sub'), db)
#         return_data = await repository.get_language_pair_stats()
#         return return_data
#
#     except HTTPException as ex:
#         raise ex
#     except Exception as ex:
#         logger.error(f'Fetch Dashboard {ex}')
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f'Internal Server Error {ex}'
#         )
#
#
# @router.get('/dashboard/stats/lang', status_code=200)
# async def get_language_pair_stats_by_lang(from_lang: str = Query(...), to_lang: str = Query(...),
#                                           db: AsyncSession = Depends(get_db),
#                                           user_info = Depends(TokenHandler.verify_access_token)):
#     try:
#         repository = DashboardRepositoryLang(user_info.get('sub'), db)
#         return_data = await repository.get_language_pair_stats_by_lang(from_lang, to_lang)
#         return return_data
#
#     except HTTPException as ex:
#         raise ex
#     except Exception as ex:
#         logger.error(f'Fetch Dashboard {ex}')
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f'Internal Server Error {ex}'
#         )
#
#
# @router.get('/filter/pos', status_code=201)
# async def filter(from_lang: str = Query(...), to_lang: str = Query(...), part_of_speech: str = Query(...),
#                     db: AsyncSession = Depends(get_db),
#                     user_info = Depends(TokenHandler.verify_access_token)
#                     ):
#     try:
#         repository = FilterRepository(user_info.get('sub'), db)
#         return_data = await repository.filter(from_lang, to_lang, part_of_speech)
#         return return_data
#     except HTTPException as ex:
#         raise ex
#     except Exception as ex:
#         logger.error(f'Pos Filter Error {ex}')
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail='Internal Server Error')
#
#
#
# @router.post('/set_status', status_code=201)
# async def change_word_status(data: ChangeWordStatusSchema,
#                       db: AsyncSession = Depends(get_db),
#                       user_info=Depends(TokenHandler.verify_access_token)
#                       ):
#     try:
#         repository = ChangeWordStatusRepository(data, user_info.get('sub'), db)
#         return_data = await repository.change_word_status()
#         return return_data
#     except HTTPException as ex:
#         raise ex
#     except Exception as ex:
#         logger.error(f'Change word Error {ex}')
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail='Internal Server Error'
#         )
#
#
# @router.post('/save', status_code=201)
# async def save_word(data: WordSchema,
#                     db:Annotated[AsyncSession, Depends(get_db)],
#                     user_info = Depends(TokenHandler.verify_access_token)
#                     ):
#     try:
#         repository = SaveWordRepository(data, user_info.get('sub'), db)
#         return_data = await repository.save_word()
#         return {"msg":"created","data":return_data}
#     except HTTPException as ex:
#         raise ex
#     except Exception as ex:
#         logger.error(f'Translate Error {ex}')
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail='Internal Server Error'
#         )
#
#

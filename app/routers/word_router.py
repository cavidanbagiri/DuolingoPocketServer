import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.token_handler import TokenHandler
from app.database.setup import get_db
from app.repositories.word_repository import SaveWordRepository, DashboardRepository, DashboardRepositoryLang, \
    FilterRepository, ChangeWordStatusRepository
from app.schemas.translate_schema import WordSchema
from app.schemas.word_schema import ChangeWordStatusSchema

router = APIRouter()


from app.logging_config import setup_logger
logger = setup_logger(__name__, "word.log")


@router.get('/dashboard/stats', status_code=200)
async def get_language_pair_stats(db:Annotated[AsyncSession, Depends(get_db)],
                    user_info = Depends(TokenHandler.verify_access_token)
                    ):
    try:
        repository = DashboardRepository(user_info.get('sub'), db)
        return_data = await repository.get_language_pair_stats()
        return return_data

    except HTTPException as ex:
        raise ex
    except Exception as ex:
        logger.error(f'Fetch Dashboard {ex}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Internal Server Error {ex}'
        )


@router.get('/dashboard/stats/lang', status_code=200)
async def get_language_pair_stats_by_lang(from_lang: str = Query(...), to_lang: str = Query(...),
                                          db: AsyncSession = Depends(get_db),
                                          user_info = Depends(TokenHandler.verify_access_token)):
    try:
        repository = DashboardRepositoryLang(user_info.get('sub'), db)
        return_data = await repository.get_language_pair_stats_by_lang(from_lang, to_lang)
        return return_data

    except HTTPException as ex:
        raise ex
    except Exception as ex:
        logger.error(f'Fetch Dashboard {ex}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Internal Server Error {ex}'
        )


@router.get('/filter/pos', status_code=201)
async def filter(from_lang: str = Query(...), to_lang: str = Query(...), part_of_speech: str = Query(...),
                    db: AsyncSession = Depends(get_db),
                    user_info = Depends(TokenHandler.verify_access_token)
                    ):
    try:
        repository = FilterRepository(user_info.get('sub'), db)
        return_data = await repository.filter(from_lang, to_lang, part_of_speech)
        return return_data
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        logger.error(f'Pos Filter Error {ex}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Internal Server Error')



@router.post('/set_status', status_code=201)
async def change_word_status(data: ChangeWordStatusSchema,
                      db: AsyncSession = Depends(get_db),
                      user_info=Depends(TokenHandler.verify_access_token)
                      ):
    try:
        repository = ChangeWordStatusRepository(data, user_info.get('sub'), db)
        return_data = await repository.change_word_status()
        return return_data
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        logger.error(f'Change word Error {ex}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Internal Server Error'
        )


@router.post('/save', status_code=201)
async def save_word(data: WordSchema,
                    db:Annotated[AsyncSession, Depends(get_db)],
                    user_info = Depends(TokenHandler.verify_access_token)
                    ):
    try:
        repository = SaveWordRepository(data, user_info.get('sub'), db)
        return_data = await repository.save_word()
        return {"msg":"created","data":return_data}
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        logger.error(f'Translate Error {ex}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Internal Server Error'
        )



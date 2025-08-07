

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.token_handler import TokenHandler
from app.database.setup import get_db
from app.repositories.word_repository import CreateMainStructureRepository, FetchWordRepository, \
    ChangeWordStatusRepository
from app.schemas.user_schema import ChangeWordStatusSchema

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



@router.get('/fetch_words', status_code=200)
async def fetch_words(only_starred: bool = False,
                      db: AsyncSession = Depends(get_db),
                      user_info = Depends(TokenHandler.verify_access_token)):
    try:
        repo = FetchWordRepository(db, user_id=int(user_info.get('sub')), only_starred=only_starred)
        result = await repo.fetch_words()
        return result
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))




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

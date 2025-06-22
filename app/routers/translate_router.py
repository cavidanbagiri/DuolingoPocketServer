
import os

from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, status
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.setup import get_db
from app.repositories.translate_repository import TranslateRepository
from app.schemas.translate_schema import TranslateSchema


router = APIRouter()


from app.logging_config import setup_logger
logger = setup_logger(__name__, "translate.log")


@router.post("/")
async def translate(data: TranslateSchema, db: Annotated[AsyncSession, Depends(get_db)]):

    try:
        repository = TranslateRepository(data)
        data = await repository.translate()
        return data
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        logger.error(f'Create Warehouse Error {ex}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Internal Server Error')

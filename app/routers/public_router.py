
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.setup import get_db
from app.repositories.public_repository import PublicSEORepo
from app.schemas.public_seo import WordSEOPayload, SlugOut
from typing import List
from urllib.parse import unquote

router = APIRouter()

@router.get("/slugs", response_model=List[SlugOut])
async def list_slugs(db: AsyncSession = Depends(get_db)):
    print('The slugs is working inside of the public router cavidan ..............')
    repo = PublicSEORepo(db)
    result = await repo.get_all_slugs()
    print(result)
    return result

@router.get("/word", response_model=WordSEOPayload)
async def get_seo_word(
    from_: str = Query(alias="from", description="source language code"),
    to: str = Query(description="target language code"),
    word: str = Query(description="word text"),
    db: AsyncSession = Depends(get_db),
):
    print('get word is working ................')
    repo = PublicSEORepo(db)
    word = unquote(word)
    payload = await repo.get_word_seo(from_, to, word)
    if not payload:
        raise HTTPException(404, "Word not found")
    print('the payload is {}'.format(payload))
    return payload
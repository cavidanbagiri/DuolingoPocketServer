
# from fastapi import APIRouter, Depends, Query
# from sqlalchemy.ext.asyncio import AsyncSession
# from app.database.setup import get_db
# from app.repositories.public_repository import PublicSEORepo
# from app.schemas.public_seo import WordSEOPayload, SlugOut
# from typing import List
# from urllib.parse import unquote


from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.setup import get_db
from app.repositories.public_repository import PublicSEORepo, GetTopWordsRepository
from app.schemas.public_seo import  WordRichPayload, SlugOut, WordSEOPayload
from typing import List
from urllib.parse import unquote


router = APIRouter()

@router.get("/slugs", response_model=List[SlugOut])
async def list_slugs(db: AsyncSession = Depends(get_db)):
    repo = PublicSEORepo(db)
    result = await repo.get_all_slugs()
    return result

@router.get("/word", response_model=WordSEOPayload)
async def get_seo_word(
    from_: str = Query(alias="from", description="source language code"),
    to: str = Query(description="target language code"),
    word: str = Query(description="word text"),
    db: AsyncSession = Depends(get_db),
):
    repo = PublicSEORepo(db)
    word = unquote(word)
    payload = await repo.get_word_seo(from_, to, word)
    if not payload:
        raise HTTPException(404, "Word not found")
    return payload


@router.get("/word-rich", response_model=WordRichPayload)
async def get_rich_word(
        from_: str = Query(alias="from", description="source language code"),
        to: str = Query(description="target language code"),
        word: str = Query(description="word text"),
        db: AsyncSession = Depends(get_db),
):
    """
    Get comprehensive word data for static page generation.
    Includes all meanings, translations, examples, categories, and related words.
    """
    repo = PublicSEORepo(db)
    word_decoded = unquote(word)
    payload = await repo.get_word_rich(from_, to, word_decoded)

    if not payload:
        raise HTTPException(status_code=404, detail="Word not found")

    return payload





@router.get("/top-words/{language_code}")
async def get_top_words(
    language_code: str,
    limit: int = Query(default=1000, le=10000),
    db: AsyncSession = Depends(get_db)
):
    """Get top words by frequency for cluster pages"""

    try:
        repo = GetTopWordsRepository(db)
        words = await repo.get_top_words_by_frequency(language_code, limit)
        return {"words": words, "count": len(words), "language": language_code}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


















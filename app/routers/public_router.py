
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.setup import get_db
from app.repositories.public_repository import PublicSEORepo, TopWordsRepository
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


# ✅ NEW: Helper function to get default target language
@router.get("/word-rich", response_model=WordRichPayload)
async def get_rich_word(
        from_: str = Query(alias="from", description="source language code"),
        word: str = Query(description="word text"),
        db: AsyncSession = Depends(get_db),
):
    """
    Get word data in source language only.
    URL: /word-rich?from=en&word=book
    """
    repo = PublicSEORepo(db)
    word_decoded = unquote(word)
    payload = await repo.get_word_rich(from_, word_decoded)

    if not payload:
        raise HTTPException(status_code=404, detail="Word not found")

    return payload


@router.get("/top-words/{language_code}")
async def get_top_words(
        language_code: str,
        limit: int = Query(default=1000, le=10000),
        db: AsyncSession = Depends(get_db)
):
    """
    Get top words (Mixed POS) for dynamic pages like /top/english-words/1000
    """
    try:
        repo = TopWordsRepository(db)
        words = await repo.get_mixed(language_code, limit)

        return {
            "words": words,
            "count": len(words),
            "language": language_code,
            "type": "mixed"
        }

    except Exception as e:
        # Log the error internally here if you have a logger
        # print(f"Error fetching top words: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/top-words/{language_code}/{pos}')
async def get_top_words_pos(
        language_code: str,
        pos: str,
        limit: int = Query(default=1000, le=10000),
        db: AsyncSession = Depends(get_db)
):
    """
    Get top words filtered by POS for dynamic pages like /top/english-verbs/1000
    """
    try:
        repo = TopWordsRepository(db)
        # Validation: Ensure pos is not something crazy
        if len(pos) > 20:
            raise HTTPException(status_code=400, detail="Invalid POS category")

        words = await repo.get_by_pos(language_code, pos, limit)

        return {
            "words": words,
            "count": len(words),
            "language": language_code,
            "type": pos
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
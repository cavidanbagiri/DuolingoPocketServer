
import asyncio
from typing import List, Optional, Dict, Any
import os

from fastapi import APIRouter, HTTPException, status, Query, BackgroundTasks
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

from app.repositories.structure_repository import (CreateMainStructureRepository,
                                                   GenerateEnglishSentence, TranslateEnglishSentencesRepository,
                                                   CreateMainStructureForRussianRepository, GenerateRussianSentences, TranslateRussianSentences,
                                                    CreateMainStructureForSpanishRepository, GenerateSpanishSentences, TranslateSpanishSentences
                                                   )

from app.services.ai_service import AIService




router = APIRouter()






############################################################################################ Spanish
@router.post('/spanish/create', status_code=201)
async def create_spanish(db: AsyncSession = Depends(get_db)):
    try:
        repo = CreateMainStructureForSpanishRepository(db)
        result = await repo.insert_spanish_words_to_table()
        return {"message": result}
    except Exception as ex:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(ex))



@router.post("/spanish/generate_sentence", status_code=200)
async def generate_spanish_sentences(
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate 5 native Spanish example sentences for Spanish words
    that currently have fewer than 5 such sentences.
    """
    try:
        generator = GenerateSpanishSentences(db)
        result = await generator.generate_sentences(limit=limit)
        return {"success": True, "data": result}
    except Exception as ex:
        await db.rollback()
        logger.error(f"💥 Spanish sentence generation failed: {str(ex)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(ex)}")



@router.post('/spanish/translate_sentence', status_code=200)
async def translate_spanish_sentences(
    min_id: int = None,
    max_id: int = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Translate Spanish sentences into en, ru, tr.
    Processes ALL eligible sentences in the given ID range.
    Commits after each sentence (resilient).
    """
    try:
        translator = TranslateSpanishSentences(db)
        result = await translator.translate_sentence(min_id=min_id, max_id=max_id)
        return {"success": True, "data": result}
    except Exception as ex:
        await db.rollback()
        logger.error(f"💥 Spanish sentence translation failed: {str(ex)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(ex)}")





############################################################################################ Russian
@router.post('/russian/create', status_code=201)
async def create(db: AsyncSession = Depends(get_db)):
    try:
        repo = CreateMainStructureForRussianRepository(db)
        result = await repo.insert_russian_words_to_table()
        return {"message": result}
    except Exception as ex:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(ex))


@router.post("/russian/generate_sentence", status_code=200)
async def generate_russian_sentences(
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate 5 native Russian example sentences for Russian words
    that currently have fewer than 5 such sentences.
    """
    try:
        generator = GenerateRussianSentences(db)
        result = await generator.generate_sentences(limit=limit)
        return {"success": True, "data": result}
    except Exception as ex:
        await db.rollback()
        logger.error(f"💥 Russian sentence generation failed: {str(ex)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(ex)}")



@router.post('/russian/translate_sentence', status_code=200)
async def translate_russian_sentences(
    min_id: int = None,
    max_id: int = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Translate Russian sentences into en, es, tr.
    Processes ALL eligible sentences in the given ID range.
    Commits after each sentence (resilient).
    """
    try:
        translator = TranslateRussianSentences(db)
        result = await translator.translate_sentence(min_id=min_id, max_id=max_id)
        return {"success": True, "data": result}
    except Exception as ex:
        await db.rollback()
        logger.error(f"💥 Sentence translation failed: {str(ex)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(ex)}")



############################################################################################ English


@router.post("/generate_english_sentences", status_code=200)
async def generate_example_sentences(
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate 5 example sentences for up to `limit` English words
    that currently have fewer than 5 sentences.
    """
    try:
        generator = GenerateEnglishSentence(db)
        result = await generator.generate_sentences_for_words(batch_limit=limit)
        return {"success": True, "data": result}
    except Exception as ex:
        await db.rollback()
        logger.error(f"💥 Sentence generation failed: {str(ex)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(ex)}")


@router.post('/translate_english_sentences', status_code=200)
async def translate_english_sentences(
    min_id: int = None,
    max_id: int = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Translate ONE English sentence (lowest ID) that needs translation.
    Optionally restrict to ID range: ?min_id=10&max_id=20
    """
    try:
        translator = TranslateEnglishSentencesRepository(db)
        result = await translator.translate_english_sentences(min_id=min_id, max_id=max_id)
        return {"success": True, "data": result}
    except Exception as ex:
        await db.rollback()
        logger.error(f"💥 Sentence translation failed: {str(ex)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(ex)}")





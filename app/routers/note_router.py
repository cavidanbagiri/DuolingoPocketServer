

import os
import asyncio
from typing import List, Optional, Dict

from fastapi import APIRouter, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import Response
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse

from app.auth.token_handler import TokenHandler
from app.database.setup import get_db
from app.repositories.note_repository import CreateNoteRepository,GetNotesRepository, GetNoteByIdRepository,UpdateNoteRepository, DeleteNoteRepository

from app.schemas.note_schemas import NoteCreate, NoteUpdate, NoteResponse

router = APIRouter()

from app.logging_config import setup_logger

logger = setup_logger(__name__, "note.log")

@router.post('/notes/create', response_model=NoteResponse, status_code=status.HTTP_201_CREATED)
async def create_note(
    note_data: NoteCreate,
    db: AsyncSession = Depends(get_db),
    user_info: dict = Depends(TokenHandler.verify_access_token),
):
    """
    Create a new note for the authenticated user
    """
    try:
        repo = CreateNoteRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            note_data=note_data.model_dump()  # Use model_dump() instead of dict()
        )
        created_note = await repo.create_note()
        return created_note
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during note creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


# router.py
@router.get('/notes/fetch', response_model=List[NoteResponse])
async def get_user_notes(
        db: AsyncSession = Depends(get_db),
        user_info: dict = Depends(TokenHandler.verify_access_token),
        target_lang: Optional[str] = Query(None,
                                           description="Filter by language (es, en, ru, tr, 'none' for no language)"),
        note_type: Optional[str] = Query(None, description="Filter by note type (vocabulary, grammar, general)"),
        search: Optional[str] = Query(None, description="Search in note_name and content"),
        skip: Optional[int] = Query(0, description="Number of records to skip"),
        limit: Optional[int] = Query(100, description="Maximum number of records to return")
):
    """
    Get all notes for the authenticated user with optional filters
    """
    try:
        user_id = int(user_info.get('sub'))
        logger.info(f"Fetching notes for user {user_id} with filters: "
                    f"target_lang={target_lang}, note_type={note_type}, search={search}")

        # Create repository instance
        repo = GetNotesRepository(
            db=db,
            user_id=user_id,
            target_lang=target_lang,
            note_type=note_type,
            search=search,
            skip=skip,
            limit=limit
        )

        # Get notes using repository
        notes = await repo.get_notes()

        logger.info(f"Found {len(notes)} notes for user {user_id}")
        return notes

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching notes: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch notes"
        )




# @router.get('/words/notes/{note_id}', response_model=NoteResponse)
@router.get('/notes/{note_id}', response_model=NoteResponse)
async def get_note_by_id(
        note_id: int,
        db: AsyncSession = Depends(get_db),
        user_info: dict = Depends(TokenHandler.verify_access_token),
):
    """
    Get a specific note by ID
    """
    try:
        user_id = int(user_info.get('sub'))

        repo = GetNoteByIdRepository(
            db=db,
            user_id=user_id,
            note_id=note_id
        )
        note = await repo.get_note()

        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note not found"
            )

        return note

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching note: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch note"
        )






@router.put('/notes/{note_id}', response_model=NoteResponse)
async def update_note(
        note_id: int,
        note_data: NoteUpdate,
        db: AsyncSession = Depends(get_db),
        user_info: dict = Depends(TokenHandler.verify_access_token),
):
    """
    Update a note
    """
    try:
        user_id = int(user_info.get('sub'))

        repo = UpdateNoteRepository(
            db=db,
            user_id=user_id,
            note_id=note_id,
            note_data=note_data.dict(exclude_unset=True)  # or note_data.model_dump(exclude_unset=True) for Pydantic v2
        )
        updated_note = await repo.update_note()
        return updated_note
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating note: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update note"
        )






@router.delete('/notes/{note_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_note(
    note_id: int,
    db: AsyncSession = Depends(get_db),
    user_info: dict = Depends(TokenHandler.verify_access_token),
):
    """
    Delete a note
    """
    try:
        repo = DeleteNoteRepository(
            db=db,
            user_id=int(user_info.get('sub')),
            note_id=note_id
        )
        await repo.delete_note()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting note: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete note"
        )
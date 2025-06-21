
import os

from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.params import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.setup import get_db
# from app.schemas.translate_schema import TranslateSchema

router = APIRouter()


from pydantic import BaseModel

class TranslateSchema(BaseModel):
    q: str
    source: str = "auto"
    target: str
    format: str = "text"
    alternatives: int = 3


@router.post("/")
async def translate(data: TranslateSchema, db: Annotated[AsyncSession, Depends(get_db)]):

    url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
    api_key = os.getenv("YANDEX_TRANSLATE_API_SECRET_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {api_key}"
    }

    payload = {
        "sourceLanguageCode": data.source,
        "targetLanguageCode": data.target,
        "texts": [data.q],
        "folderId": folder_id,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            translated_text = response.json()["translations"][0]["text"]

        return {"translation": translated_text}

    except httpx.HTTPStatusError as e:
        print(f"Yandex API error: {e.response.text}")
        raise HTTPException(status_code=500, detail="Translation service failed")
    except Exception as e:
        print("Unexpected error during translation %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")







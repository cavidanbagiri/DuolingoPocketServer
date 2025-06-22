import os

import httpx
from fastapi import HTTPException

from app.schemas.translate_schema import TranslateSchema, DetectLanguageSchema



from app.logging_config import setup_logger
logger = setup_logger(__name__, "translate.log")


class TranslateRepository:
    def __init__(self, data: TranslateSchema):
        self.data = data
        self.api_key = os.getenv("YANDEX_TRANSLATE_API_SECRET_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.translate_url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        self.detect_url = "https://translate.api.cloud.yandex.net/translate/v2/detect"

    async def translate(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }

        payload = {
            "targetLanguageCode": self.data.target,
            "texts": [self.data.q],
            "folderId": self.folder_id,
        }

        # Only add source if not 'auto'
        if self.data.source and self.data.source.strip().lower() != "auto":
            payload["sourceLanguageCode"] = self.data.source
        else:
            # Let Yandex auto-detect
            logger.info("Auto-detecting source language...")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.translate_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()["translations"][0]

            return {"translation": result["text"], "detected_lang": result.get("detectedLanguageCode")}

        except httpx.HTTPStatusError as e:
            logger.exception(f"Translation failed: {e}")
            raise HTTPException(status_code=500, detail="Translation service unavailable")
        except Exception as ex:
            logger.exception("Unexpected error during translation %s", ex)
            raise HTTPException(status_code=500, detail="Internal Server Error")

    async def detect_language(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }

        payload = {
            "text": self.data.q,
            "folderId": self.folder_id,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.detect_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

            return {"translation": result["languageCode"]}

        except httpx.HTTPStatusError as e:
            logger.exception(f"Language detection failed: {e}")
            raise HTTPException(status_code=500, detail="Language detection failed")
        except Exception as ex:
            logger.exception("Unexpected error during detection %s", ex)
            raise HTTPException(status_code=500, detail="Internal Server Error")

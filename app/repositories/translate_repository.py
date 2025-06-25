import os

import httpx
from fastapi import HTTPException
from starlette import status

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

        # Pre-process the input text
        self.processed_text = self._preprocess_text(data.q)

    def _preprocess_text(self, text: str) -> str:
        """Clean and validate input text before sending to API"""
        if not text or not text.strip():
            raise ValueError("Empty text provided for translation")

        # Remove excessive whitespace but ensure at least one space if empty after trim
        text = ' '.join(text.strip().split())
        return text or " "  # Return single space if empty after processing

    async def translate(self):
        # Early return for empty text after preprocessing
        if not self.processed_text or len(self.processed_text) < 2:
            return {
                "translation": self.processed_text,
                "detected_lang": self.data.source if self.data.source != "auto" else None
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }

        payload = {
            "targetLanguageCode": self.data.target,
            "texts": [self.processed_text],
            "folderId": self.folder_id,
        }

        if self.data.source and self.data.source.strip().lower() != "auto":
            payload["sourceLanguageCode"] = self.data.source

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.translate_url,
                    json=payload,
                    headers=headers
                )

                # Add detailed error logging
                if response.status_code != 200:
                    error_detail = response.json().get('message', 'Unknown API error')
                    logger.error(f"Yandex API Error: {error_detail}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Translation API error: {error_detail}"
                    )

                result = response.json()["translations"][0]
                return {
                    "translation": result["text"],
                    "detected_lang": result.get("detectedLanguageCode")
                }

        except httpx.HTTPStatusError as e:
            logger.exception(f"Translation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Translation service unavailable"
            )
        except Exception as ex:
            logger.exception(f"Unexpected error during translation: {ex}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal Server Error"
            )



# class TranslateRepository:
#     def __init__(self, data: TranslateSchema):
#         self.data = data
#         self.api_key = os.getenv("YANDEX_TRANSLATE_API_SECRET_KEY")
#         self.folder_id = os.getenv("YANDEX_FOLDER_ID")
#         self.translate_url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
#         self.detect_url = "https://translate.api.cloud.yandex.net/translate/v2/detect"
#
#     async def translate(self):
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Api-Key {self.api_key}"
#         }
#
#         payload = {
#             "targetLanguageCode": self.data.target,
#             "texts": [self.data.q],
#             "folderId": self.folder_id,
#         }
#
#         # Only add source if not 'auto'
#         if self.data.source and self.data.source.strip().lower() != "auto":
#             payload["sourceLanguageCode"] = self.data.source
#         else:
#             # Let Yandex auto-detect
#             logger.info("Auto-detecting source language...")
#
#         try:
#             print(f'payload is {payload}')
#             print(f'headers is {headers}')
#             async with httpx.AsyncClient() as client:
#                 response = await client.post(self.translate_url, json=payload, headers=headers)
#                 response.raise_for_status()
#                 result = response.json()["translations"][0]
#
#             return {"translation": result["text"], "detected_lang": result.get("detectedLanguageCode")}
#
#         except httpx.HTTPStatusError as e:
#             logger.exception(f"Translation failed: {e}")
#             print(f"Translation failed: {e}")
#             raise HTTPException(status_code=500, detail="Translation service unavailable")
#         except Exception as ex:
#             print(f"Unexpected error during translation - {ex}")
#             logger.exception(f"Unexpected error during translation - {ex}")
#             raise HTTPException(status_code=500, detail="Internal Server Error")
#
#     async def detect_language(self):
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Api-Key {self.api_key}"
#         }
#
#         payload = {
#             "text": self.data.q,
#             "folderId": self.folder_id,
#         }
#
#         try:
#             async with httpx.AsyncClient() as client:
#                 response = await client.post(self.detect_url, json=payload, headers=headers)
#                 response.raise_for_status()
#                 result = response.json()
#
#             return {"translation": result["languageCode"]}
#
#         except httpx.HTTPStatusError as e:
#             logger.exception(f"Language detection failed: {e}")
#             raise HTTPException(status_code=500, detail="Language detection failed")
#         except Exception as ex:
#             logger.exception("Unexpected error during detection %s", ex)
#             raise HTTPException(status_code=500, detail="Internal Server Error")

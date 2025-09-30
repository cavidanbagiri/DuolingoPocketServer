import os
import httpx
import json
import logging
from typing import Dict, Any, Optional
from app.models.ai_models import WordAnalysisRequest, WordAnalysisResponse, DeepSeekRequest

# logger = logging.getLogger(__name__)


from app.logging_config import setup_logger
logger = setup_logger(__name__, "word.log")

class AIService:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.client = httpx.AsyncClient(timeout=60.0)  # Increased timeout

    async def analyze_word(self, word: str) -> Optional[WordAnalysisResponse]:
        """Analyze a word using DeepSeek API to get POS, definitions, categories"""
        try:
            # logger.info(f"🔍 Analyzing word: {word}")
            prompt = self._build_word_analysis_prompt(word)
            raw_response = await self._make_deepseek_request(prompt)

            if raw_response:
                # Safe logging of raw_response
                response_str = str(raw_response) if raw_response else "None"
                # logger.info(f"📥 Raw AI response for {word}: {response_str[:200] if response_str else 'Empty'}...")
                return self._parse_ai_response(word, raw_response)
            return None

        except Exception as e:
            logger.error(f"❌ Error analyzing word {word}: {str(e)}", exc_info=True)
            return None


    def _build_word_analysis_prompt(self, word: str) -> str:
        return f"""
        Analyze the English word "{word}" and provide a JSON response with accurate linguistic information.

        For the word "{word}", determine:
        1. What categories best describe it (e.g., for "horse": ["animal", "mammal", "equine", "transportation", "sports"])
        2. Its part(s) of speech with appropriate definitions
        3. Example sentences showing usage
        4. Relevant synonyms and antonyms

        Return ONLY this JSON format:

        {{
            "categories": ["category1", "category2", "category3"],
            "pos_definitions": [
                {{
                    "pos": "accurate-part-of-speech",
                    "definitions": [
                        "clear definition 1",
                        "clear definition 2"
                    ]
                }}
            ],
            "examples": [
                "natural example sentence 1",
                "natural example sentence 2"
            ],
            "synonyms": ["relevant-synonym1", "relevant-synonym2"],
            "antonyms": ["relevant-antonym1", "relevant-antonym2"]
        }}

        Be specific and accurate for the word "{word}".
        """

    async def _make_deepseek_request(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make request to DeepSeek API with better error handling"""
        try:
            if not self.api_key:
                logger.error("❌ DeepSeek API key not configured")
                return None

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }

            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000,
                "stream": False
            }

            # logger.info(f"🌐 Calling DeepSeek API: {self.api_url}")
            # logger.info(f"📤 Request data: {json.dumps(data, indent=2)}")

            response = await self.client.post(self.api_url, json=data, headers=headers)

            # logger.info(f"📊 API Response Status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"❌ API Error {response.status_code}: {response.text}")
                return None

            # Parse the response
            result = response.json()
            # logger.info(f"🔍 Full API Response: {json.dumps(result, indent=2)}")

            # Extract content from the response structure
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                # logger.info(f"📝 AI Response Content: {content}")

                # Try to parse JSON from the content
                try:
                    parsed_content = json.loads(content)
                    # logger.info(f"✅ Successfully parsed JSON response")
                    return parsed_content
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                    # logger.error(f"📝 Problematic content: {content}")

                    # Try to extract JSON from text
                    extracted_json = self._extract_json_from_text(content)
                    if extracted_json:
                        # logger.info(f"✅ Successfully extracted JSON from text")
                        return extracted_json
                    else:
                        logger.error("❌ Failed to extract JSON from response")
                        return None
            else:
                logger.error("❌ No choices in API response")
                return None

        except httpx.HTTPError as e:
            logger.error(f"❌ HTTP error calling DeepSeek API: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error in API response: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error in API call: {str(e)}", exc_info=True)
            return None

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text response if AI returns text with JSON"""
        try:
            # Try to find JSON object in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = text[start_idx:end_idx]
                # logger.info(f"🔍 Extracted JSON string: {json_str}")
                return json.loads(json_str)
            else:
                logger.error("❌ No JSON object found in response")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to extract JSON: {str(e)}")
            logger.error(f"📝 Problematic text: {text}")
            return None

    def _parse_ai_response(self, word: str, ai_data: Dict[str, Any]) -> WordAnalysisResponse:
        """Parse AI response into structured data with validation"""
        try:
            # Validate and set default values
            categories = ai_data.get("categories", [])
            pos_definitions = ai_data.get("pos_definitions", [])
            examples = ai_data.get("examples", [])

            # Ensure we have at least one category and POS definition
            if not categories:
                categories = ["common", "basic"]

            if not pos_definitions:
                pos_definitions = [{
                    "pos": "unknown",
                    "definitions": [f"Definition for {word}"]
                }]

            if not examples:
                examples = [f"Example sentence with {word}."]

            response = WordAnalysisResponse(
                word=word,
                categories=categories,
                pos_definitions=pos_definitions,
                examples=examples,
                synonyms=ai_data.get("synonyms"),
                antonyms=ai_data.get("antonyms")
            )

            # logger.info(f"✅ Successfully parsed response for {word}: {response}")
            return response

        except Exception as e:
            logger.error(f"❌ Error parsing AI response for {word}: {str(e)}", exc_info=True)
            # Return a fallback response
            return WordAnalysisResponse(
                word=word,
                categories=["common"],
                pos_definitions=[{
                    "pos": "noun",
                    "definitions": [f"A basic definition of {word}"]
                }],
                examples=[f"This is an example using {word}."]
            )

    async def close(self):
        await self.client.aclose()
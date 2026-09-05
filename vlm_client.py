"""
Async client for the Vision-Language Model used by the Cognitive Loop.

Supports two interchangeable providers behind one interface:
  - OpenAI GPT-4o-mini  (VLM_PROVIDER=openai)
  - Google Gemini 1.5 Flash (VLM_PROVIDER=gemini)

Both are called over plain HTTPS with httpx.AsyncClient so a slow VLM
response never blocks the asyncio event loop running the Reflex Loop or
audio output.
"""

import base64
import logging
from typing import Optional

import httpx

from config import Config

logger = logging.getLogger(__name__)

# Requirement #2: the fixed Orientation & Mobility (O&M) system prompt.
SYSTEM_PROMPT = (
    "You are an expert Orientation and Mobility guide assisting a blind user. "
    "Based on this image, describe the immediate walking environment in "
    "maximum two short sentences. Identify the clear path (the affordance) "
    "and any notable landmarks. Use clock-face directions for objects (e.g., "
    "'at 2 o'clock'). Do not describe the sky, background aesthetics, or "
    "colors."
)

_REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)


class VLMClient:
    """Thin, provider-agnostic wrapper the Cognitive Loop calls every cycle."""

    def __init__(self, config: Config):
        self.config = config
        self._client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)

    async def close(self) -> None:
        await self._client.aclose()

    async def describe_scene(self, jpeg_bytes: bytes, history_context: str) -> str:
        """Returns a short O&M description of `jpeg_bytes`, or "" on failure
        (failures are logged, never raised, so one bad VLM call can't take
        down the whole pipeline)."""
        b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
        user_prompt = "Describe the current scene now, prioritizing anything new or changed."
        if history_context:
            user_prompt = f"{history_context}\n\n{user_prompt}"

        try:
            if self.config.vlm_provider == "openai":
                return await self._call_openai(b64_image, user_prompt)
            elif self.config.vlm_provider == "gemini":
                return await self._call_gemini(b64_image, user_prompt)
            else:
                logger.error("Unknown VLM_PROVIDER: %s", self.config.vlm_provider)
                return ""
        except httpx.HTTPStatusError as exc:
            logger.error("VLM HTTP error %s: %s", exc.response.status_code, exc.response.text[:200])
        except Exception:
            logger.exception("VLM call failed; skipping this cognitive cycle.")
        return ""

    # ------------------------------------------------------------------
    # OpenAI GPT-4o-mini
    # ------------------------------------------------------------------
    async def _call_openai(self, b64_image: str, user_prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                },
            ],
            "max_tokens": 120,
            "temperature": 0.2,
        }
        resp = await self._client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Google Gemini 1.5 Flash
    # ------------------------------------------------------------------
    async def _call_gemini(self, b64_image: str, user_prompt: str) -> str:
        model = self.config.gemini_model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.config.gemini_api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
                    ],
                }
            ],
            "generationConfig": {"maxOutputTokens": 120, "temperature": 0.2},
        }
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

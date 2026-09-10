"""
Async client for the Vision-Language Model used by the Cognitive Loop and
the Query Loop.

Supports two interchangeable providers behind one interface:
  - OpenAI GPT-4o-mini  (VLM_PROVIDER=openai)
  - Google Gemini (model configurable, VLM_PROVIDER=gemini)

Both are called over plain HTTPS with httpx.AsyncClient so a slow VLM
response never blocks the asyncio event loop running the Reflex Loop or
audio output.

Two call shapes share the same underlying _call_openai/_call_gemini
plumbing, differing only in system prompt:
  - describe_scene(): the fixed O&M ambient-narration prompt (Cognitive Loop).
  - answer_query(): the user's own spoken question, answered against the
    current frame (Query Loop) -- same image-understanding call, different
    instructions and no forced two-sentence O&M format.
"""

import base64
import logging
from typing import Optional

import httpx

from config import Config

logger = logging.getLogger(__name__)

# The fixed Orientation & Mobility (O&M) system prompt, used
# for periodic AMBIENT scene narration.

AMBIENT_SYSTEM_PROMPT = (
    "You are an expert Orientation and Mobility guide assisting a blind user. "
    "Based on this image, describe the immediate walking environment in "
    "maximum two short sentences. Identify the clear path (the affordance) "
    "and any notable STATIC landmarks (walls, signs, parked vehicles, curbs, "
    "doorways) using clock-face directions (e.g., 'at 2 o'clock'). "
    "For PEOPLE and other moving objects (pedestrians, cyclists, moving "
    "vehicles): only mention them if something has changed since you last "
    "described the scene -- they newly appeared, they left, they are now at "
    "a clearly different clock position, or one is now noticeably closer. "
    "If people are present but nothing about them has changed, do not "
    "mention them again; describe the static environment instead. "
    "Do not describe the sky, background aesthetics, or colors."
)

# Used for user-initiated spoken queries

QUERY_SYSTEM_PROMPT = (
    "You are an expert Orientation and Mobility guide assisting a blind "
    "user who just asked you a spoken question. Based on this image, answer "
    "their question directly and concisely -- normally one or two short "
    "sentences, longer only if the question truly requires it. Use "
    "clock-face directions for objects (e.g., 'at 2 o'clock') where "
    "relevant. Do not describe the sky, background aesthetics, or colors "
    "unless directly asked."
)

_REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)


class VLMClient:
    """Thin, provider-agnostic wrapper the Cognitive Loop and Query Loop call."""

    def __init__(self, config: Config):
        self.config = config
        self._client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)

    async def close(self) -> None:
        await self._client.aclose()

    async def describe_scene(self, jpeg_bytes: bytes, history_context: str) -> str:
        """Returns a short O&M description of `jpeg_bytes`, or "" on failure
        (failures are logged, never raised, so one bad VLM call can't take
        down the whole pipeline)."""
        user_prompt = "Describe the current scene now, prioritizing anything new or changed."
        if history_context:
            user_prompt = f"{history_context}\n\n{user_prompt}"
        return await self._call(jpeg_bytes, AMBIENT_SYSTEM_PROMPT, user_prompt, max_tokens=120)

    async def answer_query(self, jpeg_bytes: bytes, question: str) -> str:
        """Answers a user's spoken question grounded in the current frame,
        or "" on failure. `question` is the raw transcription (e.g. "where
        am I?", "briefly describe my surroundings") -- passed through as
        the user prompt rather than folded into a fixed template, so the
        VLM responds to what was actually asked."""
        question = question.strip()
        if not question:
            return ""
        return await self._call(jpeg_bytes, QUERY_SYSTEM_PROMPT, question, max_tokens=150)

    async def _call(self, jpeg_bytes: bytes, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
        try:
            if self.config.vlm_provider == "openai":
                return await self._call_openai(b64_image, system_prompt, user_prompt, max_tokens)
            elif self.config.vlm_provider == "gemini":
                return await self._call_gemini(b64_image, system_prompt, user_prompt, max_tokens)
            else:
                logger.error("Unknown VLM_PROVIDER: %s", self.config.vlm_provider)
                return ""
        except httpx.HTTPStatusError as exc:
            logger.error("VLM HTTP error %s: %s", exc.response.status_code, exc.response.text[:200])
        except Exception:
            logger.exception("VLM call failed.")
        return ""

    # ------------------------------------------------------------------
    # OpenAI GPT-4o-mini
    # ------------------------------------------------------------------
    async def _call_openai(self, b64_image: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
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
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        resp = await self._client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------
    async def _call_gemini(self, b64_image: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        model = self.config.gemini_model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.config.gemini_api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
                    ],
                }
            ],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
        }
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

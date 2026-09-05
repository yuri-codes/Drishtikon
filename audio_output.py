"""
Audio Output: a persistent WebSocket connection to Rime AI's flagship JSON
endpoint (/ws3) plus a real-time PCM playback stream via PyAudio.

Two asyncio.Queue objects feed this task:
  - hazard_queue  : CRITICAL_HAZARD alerts from the Reflex Loop. Always
                    drained first. Speaking one immediately sends
                    {"operation": "clear"} to Rime AI to wipe its buffer
                    (cutting off whatever it's currently synthesizing/
                    speaking), then speaks the warning.
  - ambient_queue : AMBIENT descriptions from the Cognitive Loop, spoken
                    whenever no hazard is pending.

Interrupt handling detail: Rime's /ws3 protocol supports an optional
`contextId` on every text message, echoed back on the audio chunks it
generates. We use that as a client-side "is this audio still current"
filter: every outgoing utterance gets a fresh contextId, and the receiver
only ever plays chunks whose contextId matches the most recently dispatched
one. Combined with the explicit `clear` operation, this means even if a
few stale audio chunks from a just-interrupted ambient message are already
in flight when a hazard fires, they get silently dropped instead of
bleeding into the hazard warning.

Trade-off worth knowing: because every new utterance (ambient or hazard)
claims the "active" context, if a previous ambient description is *still*
playing when the next 3-second cognitive cycle produces a new one, the
tail of the old sentence will be cut short in favor of the fresher
description. In practice ambient descriptions are capped at two short
sentences, so this rarely matters -- but it's a deliberate simplification,
not an oversight.
"""

import asyncio
import base64
import json
import logging
import uuid
from typing import Optional

import pyaudio
import websockets

from config import Config

logger = logging.getLogger(__name__)

RIME_WS_BASE = "wss://users-ws.rime.ai/ws3"


class RimeAudioOutput:
    def __init__(
        self,
        config: Config,
        hazard_queue: "asyncio.Queue",
        ambient_queue: "asyncio.Queue",
    ):
        self.config = config
        self.hazard_queue = hazard_queue
        self.ambient_queue = ambient_queue

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,  
            channels=2,               # Mac speakers require stereo (2)
            rate=config.rime_sample_rate,
            output=True,
        )
        self._active_context_id: Optional[str] = None
        self._pcm_leftover: bytes = b""
        self._stopped = asyncio.Event()

    def _build_url(self) -> str:
        return (
            f"{RIME_WS_BASE}"
            f"?speaker={self.config.rime_speaker}"
            f"&modelId=mistv3"
            f"&audioFormat=pcm"
            f"&samplingRate={self.config.rime_sample_rate}"
        )

    async def run(self) -> None:
        """Maintains the persistent connection, reconnecting with backoff on
        any drop, until `close()` is called."""
        backoff = 1.0
        headers = {"Authorization": f"Bearer {self.config.rime_api_key}"}

        while not self._stopped.is_set():
            try:
                async with websockets.connect(self._build_url(), additional_headers=headers) as ws:
                    logger.info("Connected to Rime AI /ws3 (speaker=%s)", self.config.rime_speaker)
                    backoff = 1.0
                    await asyncio.gather(self._sender(ws), self._receiver(ws))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # connection dropped, refused, DNS hiccup, etc.
                logger.warning("Rime AI connection issue (%s); reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _sender(self, ws) -> None:
        while True:
            # Hazards always win: check (non-blocking) before ever waiting
            # on the ambient queue.
            try:
                item = self.hazard_queue.get_nowait()
                await self._speak(ws, item["text"], is_hazard=True)
                continue
            except asyncio.QueueEmpty:
                pass

            try:
                item = await asyncio.wait_for(self.ambient_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            await self._speak(ws, item["text"], is_hazard=False)

    async def _speak(self, ws, text: str, is_hazard: bool) -> None:
        new_context_id = str(uuid.uuid4())
        if is_hazard:
            # Preempt immediately: stop honoring audio tagged with the old
            # context, then tell Rime to drop whatever it was buffering.
            self._active_context_id = new_context_id
            await ws.send(json.dumps({"operation": "clear"}))
        else:
            self._active_context_id = new_context_id

        # A new context means a brand-new PCM stream; any dangling odd byte
        # left over from a preempted/finished utterance is meaningless now
        # and must not be prepended to this one.
        self._pcm_leftover = b""

        await ws.send(json.dumps({"text": text, "contextId": new_context_id}))
        await ws.send(json.dumps({"operation": "flush"}))
        logger.info("[%s] -> Rime AI: %s", "HAZARD" if is_hazard else "AMBIENT", text)

    async def _receiver(self, ws) -> None:
        import numpy as np
        loop = asyncio.get_running_loop()
        async for raw_message in ws:
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                logger.warning("Non-JSON message from Rime AI, ignoring.")
                continue

            msg_type = data.get("type")

            if msg_type == "chunk":
                if data.get("contextId") != self._active_context_id:
                    logger.debug("Dropped stale chunk")
                    continue

                try:
                    pcm_bytes = base64.b64decode(data["data"])

                    # PCM chunks can split a 16-bit sample across a chunk
                    # boundary, leaving an odd trailing byte that
                    # np.frombuffer(..., dtype=np.int16) can't parse on its
                    # own. Carry that byte over and prepend it to the next
                    # chunk instead of crashing (which used to kill the
                    # whole websocket connection and drop the rest of the
                    # utterance -- exactly the "only a single word came"
                    # symptom).
                    if self._pcm_leftover:
                        pcm_bytes = self._pcm_leftover + pcm_bytes
                        self._pcm_leftover = b""

                    if len(pcm_bytes) % 2 != 0:
                        self._pcm_leftover = pcm_bytes[-1:]
                        pcm_bytes = pcm_bytes[:-1]

                    if not pcm_bytes:
                        continue

                    # Upmix 16-bit Mono to Stereo for macOS CoreAudio
                    mono_array = np.frombuffer(pcm_bytes, dtype=np.int16)
                    stereo_bytes = np.repeat(mono_array, 2).tobytes()

                    await loop.run_in_executor(None, self._stream.write, stereo_bytes)
                except Exception:
                    # A single malformed/unplayable chunk should never take
                    # down the persistent connection -- log it and keep
                    # going so the rest of the utterance still plays.
                    logger.exception("Failed to play an audio chunk; skipping it.")

            elif msg_type == "error":
                logger.error("Rime AI error: %s", data.get("message"))
            # "timestamps" and "done" events are informational only here.

    def close(self) -> None:
        self._stopped.set()
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()

"""
Audio Output: a persistent WebSocket connection to Rime AI's flagship JSON
endpoint (/ws3) plus a real-time PCM playback stream via PyAudio.

Three asyncio.Queue objects feed this task, in strict priority order:
  - hazard_queue  : CRITICAL_HAZARD alerts from the Reflex Loop, and
                    SYSTEM_STATUS degraded-state announcements. Always
                    drained first -- checked before query_queue or
                    ambient_queue every iteration. Speaking one immediately
                    sends {"operation": "clear"} to Rime AI to wipe its
                    buffer (cutting off whatever it's currently
                    synthesizing/speaking, hazard or otherwise), then
                    speaks the warning.
  - query_queue   : QUERY_RESPONSE answers from the Query Loop (user-
                    initiated spoken questions, answered against the
                    current camera frame). Checked after hazard_queue but
                    before ambient_queue -- a query answer interrupts
                    ambient narration, but is itself interrupted by a
                    hazard exactly like ambient speech is (see below).
  - ambient_queue : AMBIENT descriptions from the Cognitive Loop, spoken
                    whenever no hazard or query response is pending.

Interrupt handling detail: Rime's /ws3 protocol supports an optional
`contextId` on every text message, echoed back on the audio chunks it
generates. We use that as a client-side "is this audio still current"
filter: every outgoing utterance gets a fresh contextId, and the receiver
only ever plays chunks whose contextId matches the most recently dispatched
one. Combined with the explicit `clear` operation, this means even if a
few stale audio chunks from a just-interrupted ambient (or query) message
are already in flight when a hazard fires, they get silently dropped
instead of bleeding into the hazard warning. Because _speak() doesn't wait
for playback to finish before returning, this same claim-a-fresh-context
mechanism is what lets a hazard cut off an in-progress query answer with
no extra cancellation logic needed -- the _sender loop simply checks
hazard_queue first on every pass, so the moment a hazard is queued it wins
the very next iteration, regardless of what's currently sounding.

Trade-off worth knowing: because every new utterance (ambient, query, or
hazard) claims the "active" context, if a previous ambient description is
*still* playing when the next 3-second cognitive cycle produces a new one,
the tail of the old sentence will be cut short in favor of the fresher
description. In practice ambient descriptions are capped at two short
sentences, so this rarely matters -- but it's a deliberate simplification,
not an oversight. The same trade-off now also applies between ambient and
query speech.
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
from platform_utils import audio_troubleshooting_hint

logger = logging.getLogger(__name__)

RIME_WS_BASE = "wss://users-ws.rime.ai/ws3"


class RimeAudioOutput:
    def __init__(
        self,
        config: Config,
        hazard_queue: "asyncio.Queue",
        ambient_queue: "asyncio.Queue",
        query_queue: "asyncio.Queue",
        speaking_event: "asyncio.Event",
    ):
        self.config = config
        self.hazard_queue = hazard_queue
        self.ambient_queue = ambient_queue
        self.query_queue = query_queue

        # Shared with QueryLoop: set for the duration of every utterance
        # (ambient, query, or hazard) being sent to Rime and played back,
        # so the mic can pause wake-word listening instead of picking up
        # the system's OWN voice through speaker bleed. There's no
        # acoustic echo cancellation in this prototype (single mic, single
        # speaker, no AEC library) -- without this gate, Rime's own
        # narration gets captured by the mic, false-triggers the wake
        # word, and gets transcribed back in as if the user asked it -- a
        # genuine feedback loop (confirmed in testing: a "query" that was
        # a near-verbatim echo of the ambient sentence spoken moments
        # earlier).
        self.speaking_event = speaking_event
        self._speaking_watchdog_task: Optional["asyncio.Task"] = None

        self._pa = pyaudio.PyAudio()
        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=2,
                rate=config.rime_sample_rate,
                output=True,
            )
        except Exception as exc:
            self._pa.terminate()
            raise RuntimeError(
                f"Could not open an audio output device: {exc}\n{audio_troubleshooting_hint()}"
            ) from exc
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
                    await self._run_connection(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # connection dropped, refused, DNS hiccup, etc.
                logger.warning("Rime AI connection issue (%s); reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _run_connection(self, ws) -> None:
        """Runs _sender and _receiver for one connection's lifetime as a
        pair that lives and dies together.

        Plain `asyncio.gather(a, b)` does NOT do this: if one of the two
        raises, gather() re-raises immediately but leaves the other
        coroutine running in the background. In practice that meant: the
        connection drops, _receiver raises and we start reconnecting, but
        the old _sender is still alive, still polling the same shared
        hazard_queue/query_queue/ambient_queue as the *new* sender created
        on the fresh connection. Whichever one wins the race to
        `get_nowait()` a hazard is luck of the draw -- if it's the orphaned
        sender pointed at a dead socket, `ws.send()` raises, the item is
        already dequeued, and that hazard is gone for good with no log
        trail pointing at the real cause. For a system whose whole job is
        "never let a hazard go unspoken," that's not acceptable.

        Using asyncio.wait(..., FIRST_COMPLETED) plus an explicit cancel of
        whichever task didn't finish closes that gap: the moment either
        side ends (cleanly or with an exception), the other is torn down
        before we ever return to the reconnect loop.
        """
        sender_task = asyncio.create_task(self._sender(ws))
        receiver_task = asyncio.create_task(self._receiver(ws))
        tasks = {sender_task, receiver_task}
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc

    async def _sender(self, ws) -> None:
        while True:
            try:
                item = self.hazard_queue.get_nowait()
                await self._speak(ws, item["text"], priority="hazard")
                continue
            except asyncio.QueueEmpty:
                pass

            try:
                item = self.query_queue.get_nowait()
                await self._speak(ws, item["text"], priority="query")
                continue
            except asyncio.QueueEmpty:
                pass

            try:
                item = await asyncio.wait_for(self.ambient_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            await self._speak(ws, item["text"], priority="ambient")

    async def _speak(self, ws, text: str, priority: str) -> None:
        new_context_id = str(uuid.uuid4())
        if priority == "hazard":
            self._active_context_id = new_context_id
            await ws.send(json.dumps({"operation": "clear"}))
        else:
            self._active_context_id = new_context_id

        self._pcm_leftover = b""

        self.speaking_event.set()
        self._arm_speaking_watchdog(new_context_id)

        await ws.send(json.dumps({"text": text, "contextId": new_context_id}))
        await ws.send(json.dumps({"operation": "flush"}))
        logger.info("[%s] -> Rime AI: %s", priority.upper(), text)

    def _arm_speaking_watchdog(self, context_id: str, timeout: float = 15.0) -> None:
        """Safety net: if Rime never sends an explicit `done` for this
        utterance (dropped message, protocol quirk), speaking_event must
        still get cleared eventually -- otherwise the mic would be
        permanently gated and the wake word would silently stop working
        forever, which is exactly the kind of silent failure this whole
        project tries to avoid."""
        if self._speaking_watchdog_task is not None:
            self._speaking_watchdog_task.cancel()

        async def _watchdog():
            await asyncio.sleep(timeout)
            if self._active_context_id == context_id:
                logger.warning(
                    "No 'done' received from Rime AI after %.0fs; "
                    "clearing speaking flag as a safety fallback.", timeout,
                )
                self.speaking_event.clear()

        self._speaking_watchdog_task = asyncio.create_task(_watchdog())

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
            elif msg_type == "done":
                # Only clear on a "done" for the CURRENTLY active context --
                # if a hazard interrupted an older utterance, that older
                # one's belated "done" must not clear the flag while the
                # hazard's own audio is still playing.
                if data.get("contextId") == self._active_context_id:
                    self.speaking_event.clear()
                    if self._speaking_watchdog_task is not None:
                        self._speaking_watchdog_task.cancel()
                        self._speaking_watchdog_task = None
            # "timestamps" events are informational only here.

    def close(self) -> None:
        self._stopped.set()
        if self._speaking_watchdog_task is not None:
            self._speaking_watchdog_task.cancel()
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()

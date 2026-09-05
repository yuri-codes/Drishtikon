"""
The Cognitive Loop: low-frequency (every `COGNITIVE_INTERVAL_SEC`, default
3s) environmental description via a VLM.

Grabs the latest frame from SharedState, JPEG-encodes + base64s it, sends it
to the VLM with the fixed O&M system prompt, and queues the resulting
sentence(s) as an AMBIENT message for the Audio Output task to speak.

State management (requirement #3): a short rolling history of recent
descriptions is kept and folded back into the next prompt, explicitly
instructing the VLM not to repeat static objects (e.g. the same parked car)
unless something has actually changed. This is a prompt-level nudge rather
than a strict dedup filter -- cheap, robust to phrasing variation, and good
enough for a prototype; a production version might additionally diff
CLIP/embedding similarity between consecutive descriptions.
"""

import asyncio
import collections
import logging

import cv2

from config import Config
from shared_state import SharedState
from vlm_client import VLMClient

logger = logging.getLogger(__name__)


class CognitiveLoop:
    def __init__(
        self,
        config: Config,
        shared_state: SharedState,
        ambient_queue: "asyncio.Queue",
        vlm_client: VLMClient,
        hazard_queue: "asyncio.Queue",
    ):
        self.config = config
        self.shared_state = shared_state
        self.ambient_queue = ambient_queue
        self.vlm_client = vlm_client
        self.hazard_queue = hazard_queue
        self._history = collections.deque(maxlen=config.description_history_size)
        self._consecutive_failures = 0
        self._degraded_announced = False

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.config.cognitive_interval_sec)
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cognitive Loop cycle failed; will retry next interval.")
                self._note_failure()

    async def _run_once(self) -> None:
        frame, frame_idx, _ts = await self.shared_state.get_latest()
        if frame is None:
            return

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            logger.warning("Failed to JPEG-encode frame %s for VLM.", frame_idx)
            self._note_failure()
            return

        history_context = self._build_history_context()
        description = await self.vlm_client.describe_scene(buf.tobytes(), history_context)
        if not description:
            self._note_failure()
            return

        self._note_success()
        self._history.append(description)
        self._enqueue_ambient(description)

    def _note_failure(self) -> None:
        """Silence must never mean 'all clear'. If the VLM has failed
        repeatedly (bad key, deprecated model, network outage, etc.), speak
        a one-time degraded-state warning via the hazard queue so it
        actually interrupts and gets heard, rather than waiting behind
        ambient descriptions that were never going to arrive anyway."""
        self._consecutive_failures += 1
        threshold = self.config.cognitive_failure_announce_threshold
        if self._consecutive_failures == threshold and not self._degraded_announced:
            self._degraded_announced = True
            logger.error(
                "Cognitive Loop has failed %d consecutive cycles; "
                "announcing degraded state.", self._consecutive_failures,
            )
            self._enqueue_hazard(
                "Scene description is unavailable right now. "
                "Please rely on your other senses and mobility aid."
            )

    def _note_success(self) -> None:
        if self._degraded_announced:
            logger.info("Cognitive Loop recovered after %d failed cycles.", self._consecutive_failures)
            self._enqueue_hazard("Scene description has recovered.")
        self._consecutive_failures = 0
        self._degraded_announced = False

    def _enqueue_hazard(self, text: str) -> None:
        try:
            self.hazard_queue.put_nowait({"type": "SYSTEM_STATUS", "text": text})
        except asyncio.QueueFull:
            logger.error("Hazard queue full -- dropping system-status message: %s", text)

    def _build_history_context(self) -> str:
        if not self._history:
            return ""
        recent = " | ".join(self._history)
        return (
            "You already told the user this about the recent scene: "
            f"{recent}. Do not repeat these unless something has clearly "
            "moved, changed, or a new hazard/landmark has appeared."
        )

    def _enqueue_ambient(self, description: str) -> None:
        message = {"type": "AMBIENT", "text": description}
        try:
            self.ambient_queue.put_nowait(message)
        except asyncio.QueueFull:
            # Prefer the freshest read of the environment over a stale one
            # that never got spoken.
            try:
                self.ambient_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.ambient_queue.put_nowait(message)

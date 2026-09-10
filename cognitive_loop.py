"""
The Cognitive Loop: adaptive-cadence environmental description via a VLM.

Grabs the latest frame from SharedState, JPEG-encodes + base64s it, sends it
to the VLM with the fixed O&M system prompt, and queues the resulting
sentence(s) as an AMBIENT message for the Audio Output task to speak.

Adaptive cadence (not a fixed timer): a genuinely static scene doesn't need
fresh narration every few seconds -- repeating "several pedestrians ahead"
every cycle while nothing has actually changed is accurate but unhelpful,
and testing against real footage showed exactly this pattern even with
history-based and similarity-based dedup in place (see vlm_client.py and
the near-duplicate suppression below): the *content* was being deduped
correctly, but the *cadence* of trying was still fixed and too frequent for
a scene that wasn't changing.

This mirrors the design used in recent assistive-navigation research (e.g.
AMAVA, Klein/Rahman/Ghose 2026 -- ICPRAM), which throttles descriptive
narration much more loosely in low-motion scenes (15s+) than in active
ones, and reported this reduced user-perceived overload without hurting
safety in a blindfolded-navigation user study. Rather than a separate
motion classifier (their approach), this reuses signals the Reflex Loop
already computes every frame -- each tracked object's smoothed position and
size (see shared_state.py's TrackedObjectSnapshot) -- to score how much the
scene has changed since the last VLM call:
  - a tracked object appearing or disappearing counts fully
  - a persisting object's lateral position or size shifting counts
    proportionally to how much it moved
The loop polls frequently (cognitive_poll_interval_sec) but only actually
calls the VLM once the accumulated change score crosses
cognitive_change_threshold, OR cognitive_max_interval_sec has elapsed
regardless of change (a ceiling, so a truly static scene still gets an
occasional description rather than going silent forever -- silence must
never be mistaken for "nothing to describe", mirroring the same principle
already applied to VLM-failure handling below).

State management (requirement #3): a short rolling history of recent
descriptions is kept and folded back into the next prompt, explicitly
instructing the VLM not to repeat static objects (e.g. the same parked car)
unless something has actually changed. This is a prompt-level nudge rather
than a strict dedup filter -- cheap, robust to phrasing variation, and good
enough for a prototype; a production version might additionally diff
CLIP/embedding similarity between consecutive descriptions.

Code-level backstop: prompt instructions aren't a hard guarantee -- the VLM
can and does sometimes restate something nearly verbatim even when told not
to. As a backstop (not a replacement for the prompt-level instruction),
each new description is compared against the immediately-previous SPOKEN
one using difflib's SequenceMatcher; if they're too similar
(ambient_similarity_threshold), the new one is suppressed from actually
being queued/spoken, though it still updates _history so future prompts
still see it as "already told the user this." This only catches near-
verbatim repeats (measured against real examples: an exact repeat scores
1.0, a legitimately-evolved description scores ~0.7), so it won't
over-suppress genuinely new information that happens to share some phrasing
with the last utterance.
"""

import asyncio
import collections
import logging
from difflib import SequenceMatcher
from typing import Dict

import cv2

from config import Config
from shared_state import SharedState, TrackedObjectSnapshot
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
        self._last_spoken_description: str = ""
        self._consecutive_failures = 0
        self._degraded_announced = False
        # Scene-change tracking: the tracked-object snapshot as of the last
        # VLM call, and how long it's been since that call. Compared
        # against the CURRENT snapshot each poll to decide whether enough
        # has changed to justify calling the VLM again.
        self._last_snapshot: Dict[int, TrackedObjectSnapshot] = {}
        self._time_since_last_call: float = 0.0

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.config.cognitive_poll_interval_sec)
            self._time_since_last_call += self.config.cognitive_poll_interval_sec
            try:
                await self._maybe_run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cognitive Loop cycle failed; will retry next poll.")
                self._note_failure()

    async def _maybe_run_once(self) -> None:
        """Decides whether this poll should actually trigger a VLM call,
        based on the scene-change score and the max-interval ceiling, then
        does so if warranted."""
        current_snapshot = await self.shared_state.get_tracked_objects()
        change_score = self._compute_change_score(self._last_snapshot, current_snapshot)

        past_max_interval = self._time_since_last_call >= self.config.cognitive_max_interval_sec
        past_min_interval = self._time_since_last_call >= self.config.cognitive_min_interval_sec
        enough_changed = change_score >= self.config.cognitive_change_threshold

        if not past_max_interval and not (past_min_interval and enough_changed):
            return

        self._last_snapshot = current_snapshot
        self._time_since_last_call = 0.0
        await self._run_once()

    def _compute_change_score(
        self,
        old_snapshot: Dict[int, TrackedObjectSnapshot],
        new_snapshot: Dict[int, TrackedObjectSnapshot],
    ) -> float:
        """A cheap scalar summarizing how much the tracked scene has
        changed: appeared/disappeared objects count fully; persisting
        objects contribute proportionally to how far their smoothed
        position/size have shifted. No new inference -- reuses signals the
        Reflex Loop already computes every frame."""
        old_ids = set(old_snapshot)
        new_ids = set(new_snapshot)

        appeared = len(new_ids - old_ids)
        disappeared = len(old_ids - new_ids)
        score = float(appeared + disappeared)

        for track_id in old_ids & new_ids:
            old_obj = old_snapshot[track_id]
            new_obj = new_snapshot[track_id]
            position_shift = abs(new_obj.cx_ratio - old_obj.cx_ratio)
            size_shift = abs(new_obj.area_ratio - old_obj.area_ratio)
            # Position shift (0-1 scale, fraction of frame width) and size
            # shift (0-1 scale, fraction of frame area) are already
            # comparable ranges, so summing them directly is reasonable
            # without extra normalization.
            score += position_shift + size_shift

        return score

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

        if self._is_near_duplicate(description, self._last_spoken_description):
            logger.info("Ambient description suppressed as a near-duplicate of the last spoken one.")
            return

        self._last_spoken_description = description
        self._enqueue_ambient(description)

    def _is_near_duplicate(self, new_description: str, last_spoken: str) -> bool:
        if not last_spoken:
            return False
        ratio = SequenceMatcher(None, new_description, last_spoken).ratio()
        return ratio >= self.config.ambient_similarity_threshold

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
            f"{recent}. Static landmarks (walls, signs, parked vehicles) can "
            "be mentioned again briefly if still relevant to the path. "
            "People and moving objects should NOT be re-mentioned unless "
            "something about them has clearly changed -- moved to a "
            "different clock position, gotten notably closer, appeared, or "
            "left -- since simply still being present is not a change."
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

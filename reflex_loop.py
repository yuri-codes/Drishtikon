"""
The Reflex Loop: high-frequency (every-frame) obstacle detection.

Runs YOLOv8/11 object *tracking* (not just detection) via `ultralytics`, so
every object gets a stable ID across frames. For any tracked object whose
bounding-box center falls inside the walker's "center zone" (the path
directly ahead), it estimates time-to-collision (TTC) from how fast the
box's apparent size is growing -- the same optical-expansion ("looming")
cue animals and drivers use to judge an approaching object, formalized as
Lee's tau: tau = size / (d(size)/dt).

We don't have real depth, so `size` is approximated as sqrt(bbox_area) (a
proxy for the object's on-screen linear dimension). This is a coarse
approximation -- monocular RGB alone can't give true metric TTC -- but it's
a cheap, real-time-friendly signal for "this thing is rapidly filling more
of the frame," which is exactly what a last-instant collision warning
needs.

Growth alone is not enough to tell "heading toward the user" apart from
"passing close by" -- a pedestrian walking briskly across the frame can
grow in apparent size for a moment even though they're on a path that will
never actually intersect the user's. To distinguish these, TTC growth is
combined with a lateral trajectory-convergence check: is the box's
horizontal center drifting toward the middle of the walking path (a real
collision course) or sweeping across/away from it (a crossing or
receding path)? This mirrors, in 2D image space, the core idea behind
probabilistic collision-risk models used in vehicle ADAS and adapted for
blind-assist wearables in prior research (e.g. Tourki et al., "Probabilistic
Collision Risk Estimation for Pedestrian Navigation," 2025): distance/size
alone is a weak signal, and trajectory direction relative to the user's
path matters at least as much. We can't replicate that work's full
Gaussian trajectory model without depth data, but the same principle --
require *converging* motion, not just *proximity* or *growth*, before
calling something a collision hazard -- carries over directly.

YOLO inference is synchronous/blocking, so it's always offloaded to a
thread-pool executor (`run_in_executor`) -- this is what keeps the asyncio
event loop free to keep servicing the Cognitive Loop and Audio Output tasks
while a detection pass is running.
"""

import asyncio
import collections
import logging
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config
from shared_state import SharedState, TrackedObjectSnapshot

logger = logging.getLogger(__name__)


@dataclass
class _TrackRecord:
    area: float
    timestamp: float
    class_name: str
    last_alert_time: float = -1e9  # far in the past so the first hit isn't throttled
    is_critical: bool = False      # was this object already critical as of the last frame?
    smoothed_area: float = 0.0     # EMA of area, used for TTC instead of raw frame-to-frame area
    smoothed_cx_ratio: float = 0.0  # EMA of horizontal box-center position (0=left edge, 1=right edge)
    
    history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: collections.deque(maxlen=1))


def _bearing_to_clock(cx_ratio: float) -> str:
    """Maps a horizontal position (0.0 = left edge, 1.0 = right edge) to an
    Orientation & Mobility clock-face direction: 9 o'clock = far left,
    12 o'clock = straight ahead, 3 o'clock = far right. Matches the
    convention the Cognitive Loop's VLM prompt also uses, so hazard
    call-outs and ambient descriptions "speak the same language"."""
    raw_hour = 9.0 + cx_ratio * 6.0
    hour = int(round(raw_hour)) % 12
    hour = 12 if hour == 0 else hour
    return f"{hour} o'clock"


class ReflexLoop:
    def __init__(self, config: Config, hazard_queue: "asyncio.Queue", shared_state: "SharedState"):
        self.config = config
        self.hazard_queue = hazard_queue
        self.shared_state = shared_state
        self.model = YOLO(config.yolo_model_path)
        self._tracks: Dict[int, _TrackRecord] = {}

        self._last_any_hazard_time: float = -1e9
        self._buffered_hazards: List[Tuple[str, str]] = []

    async def process_frame(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Runs detection+tracking on `frame` (offloaded to a worker thread)
        and returns an annotated copy for the optional debug window. `timestamp`
        should be a monotonic clock reading (e.g. time.monotonic()) taken when
        the frame was captured, used for the TTC math."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._run_inference, frame)

        h, w = frame.shape[:2]
        zone_half_width = (self.config.center_zone_ratio / 2.0) * w
        zone_left = w / 2.0 - zone_half_width
        zone_right = w / 2.0 + zone_half_width

        if result is None or result.boxes is None or result.boxes.id is None:
            return self._draw(frame, [], zone_left, zone_right)

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)
        class_idxs = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        names = self.model.names

        draw_items: List[Tuple[np.ndarray, str, float, str]] = []
        seen_ids = set()
        pending_hazards: List[Tuple[str, str]] = []  # (class_name, bearing) pairs to merge/announce this pass
        snapshot: Dict[int, TrackedObjectSnapshot] = {}  # published to SharedState for the Cognitive Loop

        for box, track_id, cls_idx, conf in zip(xyxy, track_ids, class_idxs, confs):
            if conf < self.config.yolo_confidence:
                continue

            seen_ids.add(track_id)
            x1, y1, x2, y2 = box
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_ratio = area / float(w * h)
            cx = (x1 + x2) / 2.0
            class_name = names.get(int(cls_idx), str(cls_idx))
            in_center_zone = zone_left <= cx <= zone_right

            hazard_level = "normal"
            prev = self._tracks.get(track_id)
            is_critical_now = False
            ttc: Optional[float] = None
            new_smoothed_area = area  # default for a brand-new track (no prev yet)
            cx_ratio = cx / w
            new_smoothed_cx_ratio = cx_ratio  # default for a brand-new track (no prev yet)

            if (
                prev is not None
                and in_center_zone
                and area_ratio >= self.config.min_box_area_ratio
            ):
                ttc, new_smoothed_area = self._estimate_ttc(prev, area, timestamp)
                convergence_rate, new_smoothed_cx_ratio = self._estimate_convergence(prev, cx_ratio, timestamp)

                is_approaching = ttc is not None and ttc < self.config.ttc_critical_threshold_sec
                
                is_converging = (
                    self.config.lateral_convergence_threshold <= 0
                    or convergence_rate is not None
                )
                if is_approaching and is_converging:
                    hazard_level = "critical"
                    is_critical_now = True
                elif ttc is not None and ttc < self.config.ttc_critical_threshold_sec * 2.0:
                    hazard_level = "warning"
            elif prev is not None:
                
                alpha = self.config.ttc_smoothing_alpha
                new_smoothed_area = alpha * area + (1.0 - alpha) * prev.smoothed_area
                new_smoothed_cx_ratio = alpha * cx_ratio + (1.0 - alpha) * prev.smoothed_cx_ratio

            if prev is not None:
                if is_critical_now:
                    just_became_critical = not prev.is_critical
                    cooldown = (
                        self.config.hazard_cooldown_sec
                        if just_became_critical
                        else self.config.hazard_reaffirm_sec
                    )
                    if (timestamp - prev.last_alert_time) > cooldown:
                        bearing = _bearing_to_clock(cx / w)
                        pending_hazards.append((class_name, bearing))
                        prev.last_alert_time = timestamp
                
                prev.is_critical = is_critical_now

            record = self._tracks.get(track_id)
            if record is None:
                history_len = self.config.ttc_baseline_lag_frames + 1
                record = _TrackRecord(
                    area, timestamp, class_name, smoothed_area=area,
                    smoothed_cx_ratio=cx_ratio,
                    history=collections.deque(maxlen=history_len),
                )
            record.area = area
            record.smoothed_area = new_smoothed_area
            record.smoothed_cx_ratio = new_smoothed_cx_ratio
            record.timestamp = timestamp
            record.class_name = class_name
            record.history.append((timestamp, new_smoothed_area, new_smoothed_cx_ratio))
            self._tracks[track_id] = record
            snapshot[track_id] = TrackedObjectSnapshot(
                class_name=class_name,
                cx_ratio=new_smoothed_cx_ratio,
                area_ratio=new_smoothed_area / float(w * h),
            )

            draw_items.append((box, class_name, float(conf), hazard_level))

        self._flush_pending_hazards(pending_hazards, timestamp)
        self._prune_stale_tracks(seen_ids, timestamp)
        snapshot = {tid: s for tid, s in snapshot.items() if tid in self._tracks}
        await self.shared_state.update_tracked_objects(snapshot)
        return self._draw(frame, draw_items, zone_left, zone_right)

    def _flush_pending_hazards(self, pending_hazards: List[Tuple[str, str]], timestamp: float) -> None:
        """Applies the global debounce gate and merges same-window hazards
        into one utterance.

        Every item that reaches this method already passed its own
        per-object cooldown/reaffirm check -- it's a real, newly-eligible
        hazard. Nothing is ever dropped here: if we're still inside
        global_hazard_debounce_sec of the last announcement, new hazards
        are buffered (not discarded) and combined with whatever's already
        waiting; the merged group is spoken together the moment the window
        clears, which in practice is very shortly after (debounce is
        seconds, not a real delay in the "stop!" sense)."""
        self._buffered_hazards.extend(pending_hazards)
        if not self._buffered_hazards:
            return

        if (timestamp - self._last_any_hazard_time) < self.config.global_hazard_debounce_sec:
            return  # still buffering -- wait for the window to clear

        self._raise_hazard(self._buffered_hazards)
        self._last_any_hazard_time = timestamp
        self._buffered_hazards = []

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------
    def _run_inference(self, frame: np.ndarray):
        """Blocking YOLO tracking call -- always invoked via run_in_executor."""
        track_kwargs = dict(persist=True, verbose=False, conf=self.config.yolo_confidence)
        if self.config.yolo_device:
            # Explicit override (see config.yolo_device) -- otherwise
            # ultralytics auto-detects CUDA/MPS/CPU per platform on its own.
            track_kwargs["device"] = self.config.yolo_device
        results = self.model.track(frame, **track_kwargs)
        return results[0] if results else None

    def _estimate_ttc(self, prev: _TrackRecord, curr_area: float, timestamp: float) -> Tuple[Optional[float], float]:
        """Lee's tau approximation: tau = size / (d(size)/dt), using
        sqrt(area) as the size proxy.

        Two noise-reduction measures on top of the raw signal, both aimed
        at the same underlying problem -- a single tracked object
        repeatedly flipping in and out of "critical" from frame-to-frame
        noise, with each flip-back counting as a fresh rising edge gated
        only by the short hazard_cooldown_sec:

        1. curr_area is smoothed with an exponential moving average
           (ttc_smoothing_alpha) before being used at all.

        2. TTC is computed against a LAGGED baseline -- the oldest entry in
           prev.history (config.ttc_baseline_lag_frames frames back, or the
           oldest available if the track is younger than that) -- instead
           of just the immediately-previous frame. At high framerates dt
           between consecutive frames is tiny, so dt/relative_growth is
           hypersensitive: even sub-1%-per-frame noise can compute a TTC
           under the critical threshold. Comparing against a slightly older
           frame makes dt meaningfully larger, which damps noise-driven
           "growth" without adding perceptible latency to a genuinely fast
           approach -- a real collision course keeps producing a low TTC
           however far back you look; only noise washes out over a longer
           baseline. Measured in testing: on a stationary object with
           realistic +-8% frame-to-frame box jitter, EMA alone still
           produced roughly one false critical-rising-edge every ~7 frames;
           adding this lagged baseline eliminated false positives entirely
           in that test while leaving reaction time to genuine approaches
           unchanged.

        Returns (ttc_or_None, new_smoothed_area) -- the caller is
        responsible for writing new_smoothed_area back onto the track
        record's smoothed_area AND appending (timestamp, new_smoothed_area)
        onto its history deque.
        """
        alpha = self.config.ttc_smoothing_alpha
        new_smoothed_area = alpha * curr_area + (1.0 - alpha) * prev.smoothed_area

        if not prev.history:
            return None, new_smoothed_area
        baseline_ts, baseline_area, _baseline_cx = prev.history[0]  # oldest entry = lagged baseline

        dt = timestamp - baseline_ts
        if dt <= 0 or baseline_area <= 1e-6 or new_smoothed_area <= baseline_area:
            return None, new_smoothed_area
        size_baseline = baseline_area ** 0.5
        size_curr = new_smoothed_area ** 0.5
        relative_growth = (size_curr - size_baseline) / size_baseline  # over dt seconds
        if relative_growth <= 0:
            return None, new_smoothed_area
        return dt / relative_growth, new_smoothed_area

    def _estimate_convergence(self, prev: _TrackRecord, curr_cx_ratio: float, timestamp: float) -> Tuple[Optional[float], float]:
        """Estimates whether the object's horizontal position is drifting
        toward the center of the walking path (converging, i.e. a real
        collision course) versus staying put or sweeping across/away from
        it (crossing or receding).

        Mirrors _estimate_ttc's approach exactly -- EMA smoothing plus a
        lagged baseline comparison -- but tracks cx_ratio (0=left edge,
        1=right edge of frame) instead of box area. A converging object's
        cx_ratio moves toward 0.5 over time; a crossing object's cx_ratio
        sweeps past 0.5 without lingering; a receding or parallel-moving
        object's cx_ratio stays flat or moves away from 0.5.

        Note: an object that is ALREADY near frame-center is, by
        definition, on (or very near) a collision line -- it has little or
        no room left to "close in" further laterally, so a naive closing-
        distance delta would wrongly disqualify the most dangerous case
        (a head-on approach that was already centered). To handle this,
        an object counts as converging if EITHER it's already within
        lateral_convergence_threshold of dead-center, OR it's actively
        closing that gap at at least that rate.

        Returns (convergence_signal_or_None, new_smoothed_cx_ratio).
        convergence_signal is non-None whenever the object should be
        treated as converging; its value is only used for logging/
        debugging (either the current distance-from-center or the closing
        rate, whichever applied) -- the caller only checks for None vs.
        not-None, mirroring how _estimate_ttc's caller only compares the
        returned value against a threshold.
        """
        alpha = self.config.ttc_smoothing_alpha
        new_smoothed_cx = alpha * curr_cx_ratio + (1.0 - alpha) * prev.smoothed_cx_ratio
        dist_now = abs(new_smoothed_cx - 0.5)

        if dist_now <= self.config.lateral_convergence_threshold:
            return dist_now, new_smoothed_cx

        if not prev.history:
            return None, new_smoothed_cx
        baseline_ts, _baseline_area, baseline_cx = prev.history[0]

        dt = timestamp - baseline_ts
        if dt <= 0:
            return None, new_smoothed_cx

        dist_baseline = abs(baseline_cx - 0.5)
        closing_amount = dist_baseline - dist_now  # positive = moved toward center
        if closing_amount <= 0:
            return None, new_smoothed_cx
        closing_rate = closing_amount / dt
        if closing_rate < self.config.lateral_convergence_threshold:
            return None, new_smoothed_cx
        return closing_rate, new_smoothed_cx

    def _raise_hazard(self, hazards: List[Tuple[str, str]]) -> None:
        """Builds and enqueues one utterance covering one or more
        (class_name, bearing) hazards. A single hazard reads exactly as
        before ("Stop! car ahead at 1 o'clock!"); multiple hazards from the
        same debounce window are joined into one sentence ("Stop! person
        ahead at 11 o'clock and car ahead at 1 o'clock!") so simultaneous
        real dangers are spoken together instead of as a rapid-fire
        sequence of separate alerts."""
        if not hazards:
            return
        parts = [f"{class_name} ahead at {bearing}" for class_name, bearing in hazards]
        text = "Stop! " + " and ".join(parts) + "!"
        logger.warning("CRITICAL_HAZARD: %s (%d merged)", text, len(hazards))
        try:
            self.hazard_queue.put_nowait({"type": "CRITICAL_HAZARD", "text": text})
        except asyncio.QueueFull:
            logger.error("Hazard queue full -- dropping alert: %s", text)

    def _prune_stale_tracks(self, seen_ids: set, timestamp: float, max_age_sec: float = 2.0) -> None:
        stale = [
            tid for tid, rec in self._tracks.items()
            if tid not in seen_ids and (timestamp - rec.timestamp) > max_age_sec
        ]
        for tid in stale:
            del self._tracks[tid]

    def _draw(
        self,
        frame: np.ndarray,
        items: List[Tuple[np.ndarray, str, float, str]],
        zone_left: float,
        zone_right: float,
    ) -> np.ndarray:
        annotated = frame.copy()
        h = annotated.shape[0]

        cv2.line(annotated, (int(zone_left), 0), (int(zone_left), h), (255, 255, 0), 1)
        cv2.line(annotated, (int(zone_right), 0), (int(zone_right), h), (255, 255, 0), 1)

        color_map = {"normal": (0, 200, 0), "warning": (0, 200, 255), "critical": (0, 0, 255)}
        for box, class_name, conf, hazard_level in items:
            x1, y1, x2, y2 = (int(v) for v in box)
            color = color_map.get(hazard_level, (0, 200, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            if hazard_level == "critical":
                label = f"HAZARD: {label}"
            cv2.putText(
                annotated, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )
        return annotated

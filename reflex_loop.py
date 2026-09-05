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

YOLO inference is synchronous/blocking, so it's always offloaded to a
thread-pool executor (`run_in_executor`) -- this is what keeps the asyncio
event loop free to keep servicing the Cognitive Loop and Audio Output tasks
while a detection pass is running.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class _TrackRecord:
    area: float
    timestamp: float
    class_name: str
    last_alert_time: float = -1e9  # far in the past so the first hit isn't throttled


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
    def __init__(self, config: Config, hazard_queue: "asyncio.Queue"):
        self.config = config
        self.hazard_queue = hazard_queue
        self.model = YOLO(config.yolo_model_path)
        self._tracks: Dict[int, _TrackRecord] = {}

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

            if (
                prev is not None
                and in_center_zone
                and area_ratio >= self.config.min_box_area_ratio
            ):
                ttc = self._estimate_ttc(prev, area, timestamp)
                if ttc is not None:
                    if ttc < self.config.ttc_critical_threshold_sec:
                        hazard_level = "critical"
                        if (timestamp - prev.last_alert_time) > self.config.hazard_cooldown_sec:
                            bearing = _bearing_to_clock(cx / w)
                            self._raise_hazard(class_name, bearing, ttc)
                            prev.last_alert_time = timestamp
                    elif ttc < self.config.ttc_critical_threshold_sec * 2.0:
                        hazard_level = "warning"

            record = self._tracks.get(track_id) or _TrackRecord(area, timestamp, class_name)
            record.area = area
            record.timestamp = timestamp
            record.class_name = class_name
            self._tracks[track_id] = record

            draw_items.append((box, class_name, float(conf), hazard_level))

        self._prune_stale_tracks(seen_ids, timestamp)
        return self._draw(frame, draw_items, zone_left, zone_right)

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------
    def _run_inference(self, frame: np.ndarray):
        """Blocking YOLO tracking call -- always invoked via run_in_executor."""
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.config.yolo_confidence,
        )
        return results[0] if results else None

    def _estimate_ttc(self, prev: _TrackRecord, curr_area: float, timestamp: float) -> Optional[float]:
        """Lee's tau approximation: tau = size / (d(size)/dt), using
        sqrt(area) as the size proxy. Returns None if the object isn't
        clearly growing (i.e. not approaching)."""
        dt = timestamp - prev.timestamp
        if dt <= 0 or prev.area <= 1e-6 or curr_area <= prev.area:
            return None
        size_prev = prev.area ** 0.5
        size_curr = curr_area ** 0.5
        relative_growth = (size_curr - size_prev) / size_prev  # over dt seconds
        if relative_growth <= 0:
            return None
        return dt / relative_growth

    def _raise_hazard(self, class_name: str, bearing: str, ttc: float) -> None:
        text = f"Stop! {class_name} ahead at {bearing}!"
        logger.warning("CRITICAL_HAZARD: %s (TTC=%.2fs)", text, ttc)
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

"""
Thread/task-safe holder for state the Reflex Loop and Cognitive Loop both
need. The video reader task writes the frame to it every frame; the Reflex
Loop writes a lightweight snapshot of what it's currently tracking after
each frame; the Cognitive Loop reads both every few seconds. An
asyncio.Lock keeps each read/copy atomic.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrackedObjectSnapshot:
    """A minimal, Reflex-Loop-internals-free view of one tracked object, for
    the Cognitive Loop to compute a scene-change signal from -- deliberately
    NOT the full _TrackRecord (keeps the two loops decoupled; the Cognitive
    Loop shouldn't need to know about TTC/convergence internals, just
    "what's out there and roughly where")."""
    class_name: str
    cx_ratio: float  # smoothed horizontal position, 0=left edge, 1=right edge
    area_ratio: float  # smoothed box area as a fraction of frame area


@dataclass
class SharedState:
    frame: Optional[np.ndarray] = None
    frame_idx: int = -1
    timestamp: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    tracked_objects: Dict[int, TrackedObjectSnapshot] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def update(self, frame: np.ndarray, frame_idx: int) -> None:
        async with self._lock:
            self.frame = frame
            self.frame_idx = frame_idx
            self.timestamp = time.monotonic()
            self.frame_height, self.frame_width = frame.shape[:2]

    async def update_tracked_objects(self, tracked_objects: Dict[int, TrackedObjectSnapshot]) -> None:
        """Called by the Reflex Loop after each frame's tracking pass.
        Replaces the whole dict (not merged) -- objects the Reflex Loop no
        longer sees should disappear from here too, since "an object left"
        is itself part of the change signal the Cognitive Loop cares about."""
        async with self._lock:
            self.tracked_objects = tracked_objects

    async def get_latest(self) -> Tuple[Optional[np.ndarray], int, float]:
        """Returns a *copy* of the latest frame so the caller can safely hand
        it off to a JPEG encoder / API call without racing the next write."""
        async with self._lock:
            if self.frame is None:
                return None, -1, 0.0
            return self.frame.copy(), self.frame_idx, self.timestamp

    async def get_tracked_objects(self) -> Dict[int, TrackedObjectSnapshot]:
        """Returns a shallow copy of the current tracked-object snapshot.
        Shallow is fine: TrackedObjectSnapshot instances are replaced
        wholesale on each update, never mutated in place."""
        async with self._lock:
            return dict(self.tracked_objects)

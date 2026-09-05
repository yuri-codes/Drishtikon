"""
Thread/task-safe holder for the single piece of state the Reflex Loop and
Cognitive Loop both need: "what does the camera see right now". The video
reader task writes to it every frame; the Cognitive Loop reads from it every
few seconds. An asyncio.Lock keeps the read/copy atomic.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class SharedState:
    frame: Optional[np.ndarray] = None
    frame_idx: int = -1
    timestamp: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def update(self, frame: np.ndarray, frame_idx: int) -> None:
        async with self._lock:
            self.frame = frame
            self.frame_idx = frame_idx
            self.timestamp = time.monotonic()
            self.frame_height, self.frame_width = frame.shape[:2]

    async def get_latest(self) -> Tuple[Optional[np.ndarray], int, float]:
        """Returns a *copy* of the latest frame so the caller can safely hand
        it off to a JPEG encoder / API call without racing the next write."""
        async with self._lock:
            if self.frame is None:
                return None, -1, 0.0
            return self.frame.copy(), self.frame_idx, self.timestamp

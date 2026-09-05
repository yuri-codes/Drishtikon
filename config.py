"""
Centralized configuration for the Dual-Loop AI Sighted Guide prototype.

All tunables are read from environment variables (with sane defaults) so the
same code can move from a desktop test rig to a wearable device without
edits. Copy `.env.example` to `.env` and fill in real API keys, then load it
with `python-dotenv` or export the variables in your shell before running
`main.py`.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# MUST be called before Config() reads os.environ.
load_dotenv()

def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # All fields default to None/init=False and are populated in
    # __post_init__ instead of as field-default expressions. Field defaults
    # on a dataclass are evaluated once, when the class body executes (i.e.
    # at import time) -- so os.environ.get(...) as a default would freeze
    # every value at whatever the environment was at import time, silently
    # ignoring any env vars set (or changed) afterward. __post_init__ runs
    # on every Config() call, so it always reflects the current environment.

    # ------------------------------------------------------------------
    # Video input (simulates the live camera feed for this prototype)
    # ------------------------------------------------------------------
    video_path: str = field(init=False)
    show_debug_window: bool = field(init=False)

    # ------------------------------------------------------------------
    # Reflex Loop (YOLOv8/11 obstacle detection + TTC)
    # ------------------------------------------------------------------
    yolo_model_path: str = field(init=False)
    yolo_confidence: float = field(init=False)

    # Fraction of the frame width (centered) considered the walker's direct
    # path. Only objects whose bbox center falls inside this zone are
    # evaluated for collision hazards.
    center_zone_ratio: float = field(init=False)

    # An object is a CRITICAL_HAZARD if its estimated time-to-collision
    # drops below this many seconds.
    ttc_critical_threshold_sec: float = field(init=False)

    # Ignore TTC math for objects still too small/far away to matter yet
    # (as a fraction of total frame area). Cuts down on noisy far-field
    # jitter triggering false alarms.
    min_box_area_ratio: float = field(init=False)

    # Minimum seconds between repeated hazard alerts for the *same* tracked
    # object, so a single approaching obstacle doesn't spam the audio queue.
    hazard_cooldown_sec: float = field(init=False)

    # ------------------------------------------------------------------
    # Cognitive Loop (VLM environmental description)
    # ------------------------------------------------------------------
    cognitive_interval_sec: float = field(init=False)

    # "openai" (GPT-4o-mini) or "gemini"
    vlm_provider: str = field(init=False)
    openai_api_key: str = field(init=False)
    gemini_api_key: str = field(init=False)
    # gemini-1.5-flash was decommissioned outright (404 on generateContent).
    # gemini-2.5-flash-lite is also now closed to new API keys/projects
    # (Google's 404 for that model explicitly points to 3.5-flash-lite as
    # the replacement). Override via GEMINI_MODEL if this drifts again.
    gemini_model: str = field(init=False)

    # How many recent descriptions to remind the VLM about, so it doesn't
    # re-describe the same parked car every single cycle.
    description_history_size: int = field(init=False)

    # After this many consecutive Cognitive Loop failures (VLM errors, bad
    # frames, etc.), speak a one-time "scene description unavailable"
    # warning so silence is never mistaken for "all clear". At the default
    # 3s interval this is ~30s of no ambient description before it fires.
    cognitive_failure_announce_threshold: int = field(init=False)

    # ------------------------------------------------------------------
    # Audio Output (Rime AI /ws3 + PyAudio playback)
    # ------------------------------------------------------------------
    rime_api_key: str = field(init=False)
    rime_speaker: str = field(init=False)
    rime_sample_rate: int = field(init=False)

    def __post_init__(self) -> None:
        self.video_path = os.environ.get("VIDEO_PATH", "7.mov")
        self.show_debug_window = _env_bool("SHOW_DEBUG_WINDOW", True)

        self.yolo_model_path = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
        self.yolo_confidence = float(os.environ.get("YOLO_CONFIDENCE", "0.45"))
        self.center_zone_ratio = float(os.environ.get("CENTER_ZONE_RATIO", "0.4"))
        self.ttc_critical_threshold_sec = float(os.environ.get("TTC_THRESHOLD_SEC", "2.0"))
        self.min_box_area_ratio = float(os.environ.get("MIN_BOX_AREA_RATIO", "0.02"))
        self.hazard_cooldown_sec = float(os.environ.get("HAZARD_COOLDOWN_SEC", "2.5"))

        self.cognitive_interval_sec = float(os.environ.get("COGNITIVE_INTERVAL_SEC", "3.0"))
        self.vlm_provider = os.environ.get("VLM_PROVIDER", "openai")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder-openai-key")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "placeholder-gemini-key")
        self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.description_history_size = int(os.environ.get("DESCRIPTION_HISTORY_SIZE", "3"))
        self.cognitive_failure_announce_threshold = int(
            os.environ.get("COGNITIVE_FAILURE_ANNOUNCE_THRESHOLD", "10")
        )

        self.rime_api_key = os.environ.get("RIME_API_KEY", "placeholder-rime-key")
        self.rime_speaker = os.environ.get("RIME_SPEAKER", "cove")
        self.rime_sample_rate = int(os.environ.get("RIME_SAMPLE_RATE", "24000"))

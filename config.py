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
    video_path: str = field(init=False)
    show_debug_window: bool = field(init=False)

    yolo_model_path: str = field(init=False)
    yolo_confidence: float = field(init=False)

    yolo_device: str = field(init=False)

    center_zone_ratio: float = field(init=False)

    ttc_critical_threshold_sec: float = field(init=False)

    min_box_area_ratio: float = field(init=False)

    ttc_smoothing_alpha: float = field(init=False)

    ttc_baseline_lag_frames: int = field(init=False)

    lateral_convergence_threshold: float = field(init=False)

    hazard_cooldown_sec: float = field(init=False)

    hazard_reaffirm_sec: float = field(init=False)

    global_hazard_debounce_sec: float = field(init=False)

    # ------------------------------------------------------------------
    # Cognitive Loop (VLM environmental description, adaptive cadence)
    # ------------------------------------------------------------------

    cognitive_poll_interval_sec: float = field(init=False)

    cognitive_min_interval_sec: float = field(init=False)

    cognitive_max_interval_sec: float = field(init=False)

    cognitive_change_threshold: float = field(init=False)

    # "openai" (GPT-4o-mini) or "gemini"
    vlm_provider: str = field(init=False)
    openai_api_key: str = field(init=False)
    gemini_api_key: str = field(init=False)
   
    gemini_model: str = field(init=False)

    
    description_history_size: int = field(init=False)

    # Code-level backstop against the VLM restating the last ambient
    ambient_similarity_threshold: float = field(init=False)

    cognitive_failure_announce_threshold: int = field(init=False)

    # ------------------------------------------------------------------
    # Audio Output (Rime AI /ws3 + PyAudio playback)
    # ------------------------------------------------------------------
    rime_api_key: str = field(init=False)
    rime_speaker: str = field(init=False)
    rime_sample_rate: int = field(init=False)

    # ------------------------------------------------------------------
    # Query Loop (wake-word-gated voice queries, grounded in current frame)
    # ------------------------------------------------------------------
    groq_api_key: str = field(init=False)

    # openWakeWord ships no built-in "hey guide" model; "hey_jarvis_v0.1" is used as a placeholder until a custom model is trained. 
    wake_word_model: str = field(init=False)
    wake_word_name: str = field(init=False)
    wake_word_threshold: float = field(init=False)

    post_speech_grace_sec: float = field(init=False)

    vad_aggressiveness: int = field(init=False)
   
    query_silence_ms: int = field(init=False)
   
    query_max_utterance_sec: float = field(init=False)
    
    query_min_utterance_sec: float = field(init=False)

    def __post_init__(self) -> None:
        self.video_path = os.environ.get("VIDEO_PATH", "Rainfall.mov")
        self.show_debug_window = _env_bool("SHOW_DEBUG_WINDOW", True)

        self.yolo_model_path = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
        self.yolo_confidence = float(os.environ.get("YOLO_CONFIDENCE", "0.45"))
        self.yolo_device = os.environ.get("YOLO_DEVICE", "")
        self.center_zone_ratio = float(os.environ.get("CENTER_ZONE_RATIO", "0.4"))
        self.ttc_critical_threshold_sec = float(os.environ.get("TTC_THRESHOLD_SEC", "2.0"))
        self.min_box_area_ratio = float(os.environ.get("MIN_BOX_AREA_RATIO", "0.02"))
        self.ttc_smoothing_alpha = float(os.environ.get("TTC_SMOOTHING_ALPHA", "0.5"))
        self.ttc_baseline_lag_frames = int(os.environ.get("TTC_BASELINE_LAG_FRAMES", "5"))
        self.lateral_convergence_threshold = float(os.environ.get("LATERAL_CONVERGENCE_THRESHOLD", "0.03"))
        self.hazard_cooldown_sec = float(os.environ.get("HAZARD_COOLDOWN_SEC", "2.5"))
        self.hazard_reaffirm_sec = float(os.environ.get("HAZARD_REAFFIRM_SEC", "8.0"))
        self.global_hazard_debounce_sec = float(os.environ.get("GLOBAL_HAZARD_DEBOUNCE_SEC", "1.5"))

        self.cognitive_poll_interval_sec = float(os.environ.get("COGNITIVE_POLL_INTERVAL_SEC", "1.0"))
        self.cognitive_min_interval_sec = float(os.environ.get("COGNITIVE_MIN_INTERVAL_SEC", "3.0"))
        self.cognitive_max_interval_sec = float(os.environ.get("COGNITIVE_MAX_INTERVAL_SEC", "15.0"))
        self.cognitive_change_threshold = float(os.environ.get("COGNITIVE_CHANGE_THRESHOLD", "0.35"))
        self.vlm_provider = os.environ.get("VLM_PROVIDER", "openai")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder-openai-key")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "placeholder-gemini-key")
        self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.description_history_size = int(os.environ.get("DESCRIPTION_HISTORY_SIZE", "3"))
        self.ambient_similarity_threshold = float(os.environ.get("AMBIENT_SIMILARITY_THRESHOLD", "0.85"))
        self.cognitive_failure_announce_threshold = int(
            os.environ.get("COGNITIVE_FAILURE_ANNOUNCE_THRESHOLD", "10")
        )

        self.rime_api_key = os.environ.get("RIME_API_KEY", "placeholder-rime-key")
        self.rime_speaker = os.environ.get("RIME_SPEAKER", "cove")
        self.rime_sample_rate = int(os.environ.get("RIME_SAMPLE_RATE", "24000"))

        self.groq_api_key = os.environ.get("GROQ_API_KEY", "placeholder-groq-key")
        self.wake_word_model = os.environ.get("WAKE_WORD_MODEL", "hey_jarvis_v0.1")
        self.wake_word_name = os.environ.get("WAKE_WORD_NAME", "hey_jarvis_v0.1")
        self.wake_word_threshold = float(os.environ.get("WAKE_WORD_THRESHOLD", "0.5"))
        self.post_speech_grace_sec = float(os.environ.get("POST_SPEECH_GRACE_SEC", "0.5"))
        self.vad_aggressiveness = int(os.environ.get("VAD_AGGRESSIVENESS", "2"))
        self.query_silence_ms = int(os.environ.get("QUERY_SILENCE_MS", "700"))
        self.query_max_utterance_sec = float(os.environ.get("QUERY_MAX_UTTERANCE_SEC", "12.0"))
        self.query_min_utterance_sec = float(os.environ.get("QUERY_MIN_UTTERANCE_SEC", "0.3"))

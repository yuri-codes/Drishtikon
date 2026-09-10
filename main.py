"""
Dual-Loop AI Sighted Guide -- prototype entry point.

Reads a pre-recorded MP4 file (standing in for a live camera feed) and
strictly throttles playback to the video's native FPS with
`await asyncio.sleep(...)`, so downstream timing (especially TTC math)
behaves the same way it would against a real-time camera stream.

Four tasks run concurrently under one asyncio event loop:
  1. video_reader_task -- reads frames, updates SharedState, drives the
     Reflex Loop on every single frame.
  2. CognitiveLoop.run  -- VLM environmental description every N seconds.
  3. QueryLoop.run      -- wake-word-gated voice queries, answered against
     the current camera frame.
  4. RimeAudioOutput.run -- persistent Rime AI websocket + PCM playback,
     with hazard > query > ambient priority.

Usage:
    export RIME_API_KEY=...
    export OPENAI_API_KEY=...            # or GEMINI_API_KEY + VLM_PROVIDER=gemini
    export GROQ_API_KEY=...              # Whisper transcription for voice queries
    export VIDEO_PATH=/path/to/walk.mp4
    python main.py
"""

import asyncio
import logging
import time

import cv2

from audio_output import RimeAudioOutput
from cognitive_loop import CognitiveLoop
from config import Config
from platform_utils import ensure_utf8_console
from query_loop import QueryLoop
from reflex_loop import ReflexLoop
from shared_state import SharedState
from vlm_client import VLMClient

# Must happen before logging is configured, so every log line (including
# ones written during import-time errors) benefits from it -- see
# platform_utils.ensure_utf8_console for why this matters on Windows.
ensure_utf8_console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sighted_guide")


async def video_reader_task(config: Config, shared_state: SharedState, reflex_loop: ReflexLoop) -> None:
    # Parse '0' as an integer for webcams, leave as string for video files
    source = int(config.video_path) if config.video_path.isdigit() else config.video_path
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.video_path}")

    frame_idx = 0
    logger.info("Reading from live video source: %s", source)
    
    debug_window_active = config.show_debug_window
    loop = asyncio.get_running_loop()

    try:
        while True:
            # Offload the blocking hardware read to a thread to keep the event loop free
            ok, frame = await loop.run_in_executor(None, cap.read)
            if not ok:
                logger.info("Camera disconnected or stream ended.")
                break

            capture_ts = time.monotonic()
            await shared_state.update(frame, frame_idx)
            annotated = await reflex_loop.process_frame(frame, capture_ts)

            if debug_window_active:
                try:
                    cv2.imshow("Dual-Loop AI Sighted Guide", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Quit requested from debug window.")
                        break
                except cv2.error:
                    logger.warning("Debug window unavailable. Continuing without it.")
                    debug_window_active = False

            frame_idx += 1
            
            # Yield control back to the event loop (the camera naturally throttles the pace)
            await asyncio.sleep(0)
    finally:
        cap.release()
        if debug_window_active:
            cv2.destroyAllWindows()

def _validate_config(config: Config) -> None:
    """Fail fast and loud on obviously-broken config, instead of discovering
    it only after several silent, log-buried VLM/Rime failures."""
    problems = []

    if config.rime_api_key in ("", "placeholder-rime-key"):
        problems.append("RIME_API_KEY is not set (audio output will never connect).")

    if config.vlm_provider == "openai" and config.openai_api_key in ("", "sk-placeholder-openai-key"):
        problems.append("OPENAI_API_KEY is not set but VLM_PROVIDER=openai.")
    elif config.vlm_provider == "gemini" and config.gemini_api_key in ("", "placeholder-gemini-key"):
        problems.append("GEMINI_API_KEY is not set but VLM_PROVIDER=gemini.")
    elif config.vlm_provider not in ("openai", "gemini"):
        problems.append(f"VLM_PROVIDER={config.vlm_provider!r} is not 'openai' or 'gemini'.")

    if config.groq_api_key in ("", "placeholder-groq-key"):
        problems.append("GROQ_API_KEY is not set (voice queries will never transcribe).")

    if problems:
        for p in problems:
            logger.error("Config problem: %s", p)
        raise SystemExit(
            "Refusing to start with invalid config -- see errors above. "
            "Set the required environment variables and try again."
        )


async def main() -> None:
    config = Config()
    _validate_config(config)

    shared_state = SharedState()
    hazard_queue: asyncio.Queue = asyncio.Queue()
    ambient_queue: asyncio.Queue = asyncio.Queue(maxsize=3)
    query_queue: asyncio.Queue = asyncio.Queue(maxsize=3)
    speaking_event: asyncio.Event = asyncio.Event()

    reflex_loop = ReflexLoop(config, hazard_queue, shared_state)
    vlm_client = VLMClient(config)
    cognitive_loop = CognitiveLoop(config, shared_state, ambient_queue, vlm_client, hazard_queue)
    query_loop = QueryLoop(config, shared_state, query_queue, vlm_client, hazard_queue, speaking_event)
    audio_output = RimeAudioOutput(config, hazard_queue, ambient_queue, query_queue, speaking_event)

    video_task = asyncio.create_task(video_reader_task(config, shared_state, reflex_loop), name="video_reader")
    background_tasks = [
        asyncio.create_task(cognitive_loop.run(), name="cognitive_loop"),
        asyncio.create_task(query_loop.run(), name="query_loop"),
        asyncio.create_task(audio_output.run(), name="audio_output"),
    ]

    try:
        # The prototype's "session" ends when the video file runs out.
        # The background loops run forever otherwise, so cancel them
        # once the video is done.
        await video_task
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        for task in background_tasks:
            task.cancel()
        await asyncio.gather(*background_tasks, return_exceptions=True)
        audio_output.close()
        query_loop.close()
        await vlm_client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

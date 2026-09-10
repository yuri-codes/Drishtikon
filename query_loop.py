"""
The Query Loop: continuous local microphone listening for user-initiated
spoken questions ("where am I?", "briefly describe my surroundings"),
answered against the CURRENT camera frame via the VLM -- not a canned
prompt, not a generic text-only chat completion.

Pipeline, per audio frame read from the mic:
  1. Wake-word detection (openWakeWord). Nothing else happens until the
     wake word fires -- this is what stops the system from treating every
     nearby conversation as a query. NOTE: openWakeWord ships pretrained
     models for phrases like "hey jarvis" / "alexa" / "hey mycroft"; there
     is no built-in "hey guide" model. This uses "hey jarvis" as a
     placeholder wake word (config.wake_word_model) until a custom model
     is trained -- see openWakeWord's docs for the training path.
  2. Once woken, VAD (webrtcvad) buffers audio while the user is actively
     speaking and detects when they've stopped (utterance boundary), with
     a hard max-utterance-length safety cap.
  3. The buffered utterance is transcribed via Groq Whisper.
  4. The transcribed question + the CURRENT frame from SharedState are
     sent to the VLM (VLMClient.answer_query) -- this is what grounds the
     answer in what the camera sees right now, not a generic LLM response.
  5. The answer is queued as a QUERY_RESPONSE message on query_queue for
     RimeAudioOutput to speak. Per the existing priority design, a
     CRITICAL_HAZARD arriving from the Reflex Loop always preempts a
     query answer, exactly like it preempts AMBIENT speech.

Audio capture uses PyAudio (input direction) -- the same library
audio_output.py already uses for output -- so no new heavyweight audio
dependency is introduced.
"""

import asyncio
import logging
import time
from typing import List, Optional

import numpy as np
import pyaudio
import webrtcvad
from groq import AsyncGroq

from config import Config
from platform_utils import audio_troubleshooting_hint
from shared_state import SharedState
from vlm_client import VLMClient

logger = logging.getLogger(__name__)

# webrtcvad requires 16-bit mono PCM at 8/16/32/48kHz, in exact 10/20/30ms
# frames. 16kHz/30ms is a good balance: fine for VAD and plenty for Whisper.
_VAD_SAMPLE_RATE = 16000
_VAD_FRAME_MS = 30
_VAD_FRAME_BYTES = int(_VAD_SAMPLE_RATE * (_VAD_FRAME_MS / 1000.0)) * 2  # int16 = 2 bytes/sample


class QueryLoop:
    def __init__(
        self,
        config: Config,
        shared_state: SharedState,
        query_queue: "asyncio.Queue",
        vlm_client: VLMClient,
        hazard_queue: "asyncio.Queue",
        speaking_event: "asyncio.Event",
    ):
        self.config = config
        self.shared_state = shared_state
        self.query_queue = query_queue
        self.vlm_client = vlm_client
        self.hazard_queue = hazard_queue
        self.speaking_event = speaking_event

        self._groq_client = AsyncGroq(api_key=config.groq_api_key)

        self._pa = pyaudio.PyAudio()
        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=_VAD_SAMPLE_RATE,
                input=True,
                frames_per_buffer=int(_VAD_SAMPLE_RATE * (_VAD_FRAME_MS / 1000.0)),
            )
        except Exception as exc:
            self._pa.terminate()
            raise RuntimeError(
                f"Could not open a microphone input device: {exc}\n{audio_troubleshooting_hint()}"
            ) from exc

        self._vad = webrtcvad.Vad(config.vad_aggressiveness)
        self._wake_model = self._load_wake_word_model()
        self._logged_prediction_keys = False
        self._wake_score_log_counter = 0
        self._was_speaking = False
        self._speech_cleared_ts = -1e9  # far in the past so no grace period applies at startup

        self._stopped = asyncio.Event()
        self._consecutive_failures = 0
        self._degraded_announced = False

    def _load_wake_word_model(self):
        # Imported lazily so the rest of the module still imports cleanly in environments/tests without openwakeword installed.
        
        from openwakeword.model import Model

        model_path = self.config.wake_word_model
        try:
            return Model(wakeword_models=[model_path])
        except TypeError:
            return Model(wakeword_model_paths=[model_path])

    async def run(self) -> None:
        """Reads mic frames continuously, offloading blocking PyAudio reads
        to a thread so the event loop stays free for the other loops."""
        loop = asyncio.get_running_loop()
        listening_for_wake_word = True
        utterance_frames: List[bytes] = []
        silence_frame_count = 0
        utterance_start_ts = 0.0

       
        silence_frames_to_end = int(self.config.query_silence_ms / _VAD_FRAME_MS)
        max_utterance_frames = int(self.config.query_max_utterance_sec * 1000 / _VAD_FRAME_MS)

        while not self._stopped.is_set():
            try:
                frame_bytes = await loop.run_in_executor(
                    None, self._stream.read, int(_VAD_SAMPLE_RATE * (_VAD_FRAME_MS / 1000.0)), False
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Query Loop mic read failed; retrying shortly.")
                self._note_failure()
                await asyncio.sleep(0.5)
                continue

            is_speaking_now = self.speaking_event.is_set()
            if self._was_speaking and not is_speaking_now:
                self._speech_cleared_ts = time.monotonic()
            self._was_speaking = is_speaking_now

            in_grace_period = (
                time.monotonic() - self._speech_cleared_ts
            ) < self.config.post_speech_grace_sec

            if is_speaking_now or in_grace_period:
               
                if not listening_for_wake_word:
                    listening_for_wake_word = True
                    utterance_frames = []
                    silence_frame_count = 0
                continue

            if listening_for_wake_word:
                audio_array = np.frombuffer(frame_bytes, dtype=np.int16)
                try:
                    prediction = self._wake_model.predict(audio_array)
                except Exception:
                    logger.exception("Wake-word prediction failed; skipping this frame.")
                    continue

                if not self._logged_prediction_keys:
                   
                    logger.info(
                        "Wake-word model loaded; available prediction keys: %s",
                        list(prediction.keys()),
                    )
                    self._logged_prediction_keys = True

                score = prediction.get(self.config.wake_word_name, 0.0)
                self._wake_score_log_counter += 1
                if self._wake_score_log_counter % 30 == 0:  # roughly every ~1s at 30ms frames
                    
                    logger.info("Wake-word score for '%s': %.3f", self.config.wake_word_name, score)
                if score >= self.config.wake_word_threshold:
                    logger.info("Wake word detected (score=%.2f); listening for a query.", score)
                    listening_for_wake_word = False
                    utterance_frames = []
                    silence_frame_count = 0
                    utterance_start_ts = time.monotonic()
                continue

            # Actively capturing an utterance after the wake word fired.
            utterance_frames.append(frame_bytes)
            is_speech = self._safe_is_speech(frame_bytes)

            if is_speech:
                silence_frame_count = 0
            else:
                silence_frame_count += 1

            utterance_too_long = len(utterance_frames) >= max_utterance_frames
            utterance_finished = silence_frame_count >= silence_frames_to_end

            if utterance_finished or utterance_too_long:
                if utterance_too_long:
                    logger.warning(
                        "Query utterance hit the %.1fs max length cap; processing what we have.",
                        self.config.query_max_utterance_sec,
                    )
                pcm_bytes = b"".join(utterance_frames)
                listening_for_wake_word = True
                utterance_frames = []
                silence_frame_count = 0

                
                min_bytes = int(self.config.query_min_utterance_sec * _VAD_SAMPLE_RATE) * 2
                if len(pcm_bytes) >= min_bytes:
                    asyncio.create_task(self._handle_utterance(pcm_bytes))
                else:
                    logger.info("Query utterance too short after wake word; ignoring.")

    def _safe_is_speech(self, frame_bytes: bytes) -> bool:
        
        if len(frame_bytes) != _VAD_FRAME_BYTES:
            return False
        try:
            return self._vad.is_speech(frame_bytes, _VAD_SAMPLE_RATE)
        except Exception:
            return False

    async def _handle_utterance(self, pcm_bytes: bytes) -> None:
        try:
            question = await self._transcribe(pcm_bytes)
            if not question:
                return
            logger.info("[QUERY] User asked: %s", question)

            frame, frame_idx, _ts = await self.shared_state.get_latest()
            if frame is None:
                self._enqueue_query_response(
                    "I can't see anything right now, so I can't answer that."
                )
                return

            import cv2  # local import: keeps this module's import cost low when unused
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                logger.warning("Failed to JPEG-encode frame %s for query VLM call.", frame_idx)
                self._note_failure()
                return

            answer = await self.vlm_client.answer_query(buf.tobytes(), question)
            if not answer:
                self._note_failure()
                self._enqueue_query_response(
                    "Sorry, I couldn't process that just now."
                )
                return

            self._note_success()
            self._enqueue_query_response(answer)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Query Loop failed to handle an utterance.")
            self._note_failure()

    async def _transcribe(self, pcm_bytes: bytes) -> str:
        """Wraps raw 16kHz mono PCM in a minimal WAV header (Whisper needs a
        real audio container, not headerless PCM) and sends it to Groq."""
        wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate=_VAD_SAMPLE_RATE)
        try:
            transcription = await self._groq_client.audio.transcriptions.create(
                file=("query.wav", wav_bytes),
                model="whisper-large-v3-turbo",
                response_format="json",
            )
            text = (transcription.text or "").strip()
            if not text or len(text) > 300:
                logger.info("Query transcription ignored (empty or too long).")
                return ""
            return text
        except Exception:
            logger.exception("Query transcription failed.")
            self._note_failure()
            return ""

    def _enqueue_query_response(self, text: str) -> None:
        try:
            self.query_queue.put_nowait({"type": "QUERY_RESPONSE", "text": text})
        except asyncio.QueueFull:
            logger.error("Query queue full -- dropping response: %s", text)

    # ------------------------------------------------------------------
    # Failure/recovery tracking, mirroring CognitiveLoop's watchdog so a
    # broken mic/Whisper/VLM path is announced rather than silently mute.
    # ------------------------------------------------------------------
    def _note_failure(self) -> None:
        self._consecutive_failures += 1
        threshold = self.config.cognitive_failure_announce_threshold
        if self._consecutive_failures == threshold and not self._degraded_announced:
            self._degraded_announced = True
            logger.error(
                "Query Loop has failed %d consecutive times; announcing degraded state.",
                self._consecutive_failures,
            )
            try:
                self.hazard_queue.put_nowait({
                    "type": "SYSTEM_STATUS",
                    "text": "Voice queries are unavailable right now.",
                })
            except asyncio.QueueFull:
                pass

    def _note_success(self) -> None:
        self._consecutive_failures = 0
        self._degraded_announced = False

    def close(self) -> None:
        self._stopped.set()
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wraps raw PCM bytes in a minimal WAV header. Avoids a hard dependency
    on the `wave` module's file-based API for an in-memory buffer."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

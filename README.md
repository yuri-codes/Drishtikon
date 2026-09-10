# Drishtikon: Dual-Loop AI Sighted Guide (Prototype)

Drishtikon is a voice-native, cross-platform accessibility prototype designed to act as an Orientation & Mobility (O&M) guide for blind or low-vision users. It processes a live-camera feed and uses a dual-loop AI architecture to provide real-time spatial awareness and contextual scene descriptions, spoken entirely through Rime AI.
 It features a fast **Reflex Loop** for immediate obstacle avoidance, a slower **Cognitive Loop** for richer environmental description, and a **Query Loop** for wake-word-gated spoken questions. Rime AI provides the primary spoken output, ensuring voice is essential to the navigation and situational awareness experience.

## Architecture

```text
                    ┌─────────────────────┐
                    │  video_reader_task  │
                    └──────────┬──────────┘
                               │ frame + timestamp
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
      ┌────────────────────┐      ┌────────────────────┐
      │    Reflex Loop     │      │    SharedState     │
      └──────────┬─────────┘      └──────┬──────┬──────┘
                 │ CRITICAL_HAZARD       │      │
                 ▼              every 3s ▼      ▼ on wake word
         hazard_queue◄────────┐   ┌─────────┐ ┌─────────────┐
                 │            │   │Cognitive│ │ Query Loop  │
                 │            │   └────┬────┘ └──────┬──────┘
                 │            │        │AMBIENT      │QUERY_RESPONSE
                 ▼            └────────┘             ▼
        ┌─────────────────────────────────────────────────────┐
        │        RimeAudioOutput (audio_output.py)            │
        │  priority: hazard > query > ambient                 │
        └─────────────────────────────────────────────────────┘
```


### Architecture
The system runs four concurrent `asyncio` tasks:
1. **Reflex Loop (High-Frequency):** Runs YOLO object tracking on every frame to estimate Time-To-Collision (TTC).
2. **Cognitive Loop (Adaptive Cadence):** Throttles Vision Language Model (VLM) calls based on scene-change metrics, providing environmental descriptions only when necessary to prevent redundant narration.
3. **Query Loop (Continuous Listening):** A wake-word-gated microphone stream that captures user queries, transcribes them, and grounds the VLM's answer in the current camera frame.
4. **Audio Output:** A persistent Rime AI WebSocket connection that prioritizes hazards over user queries, and queries over ambient descriptions.

### Third-Party Services
* **Rime AI:** Primary Text-to-Speech (TTS) engine.
* **Ultralytics (YOLOv8/11):** Local object tracking and bounding box generation.
* **OpenAI (GPT-4o-mini) / Google (Gemini):** Vision Language Models for scene description.
* **Groq (Whisper-large-v3-turbo):** Fast audio transcription.
* **openWakeWord & webrtcvad:** Wake-word detection and Voice Activity Detection (VAD).

### Rime Integration Details
* **Model ID:** `mistv3`[cite: 14]
* **Speaker:** `cove`[cite: 14]
* **Language:** English[cite: 14]
* **Endpoint:** `wss://users-ws.rime.ai/ws3`[cite: 14]
* **Audio Format:** `pcm` (16-bit upmixed to stereo via PyAudio)[cite: 14]
* **Transport:** WebSocket[cite: 14]

### Setup Instructions
1. `pip install -r requirements.txt` (Ensure PortAudio is installed at the OS level for PyAudio).
2. Copy `.env.example` to `.env` and insert your API keys (Rime, Groq, OpenAI/Gemini).
3. Run `python main.py` with a valid `VIDEO_PATH` environment variable.

### Known Limitations
* **Monocular TTC:** Time-To-Collision relies on optical expansion of bounding boxes rather than true depth sensors, making it an approximation.
* **No Acoustic Echo Cancellation (AEC):** The system relies on a software gate to pause listening while speaking, meaning true "barge-in" interruption via voice is not supported.

### Failure Behavior
The system defaults to visible fallbacks[cite: 14]. If the VLM or Query loops fail repeatedly (e.g., network drop), the system injects a `SYSTEM_STATUS` message into the hazard queue (e.g., "Scene description is unavailable right now"). Silence is never allowed to be mistakenly interpreted as an "all clear" state.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real keys, ensuring no secrets are committed
```
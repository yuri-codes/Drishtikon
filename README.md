# Dual-Loop AI Sighted Guide (Prototype)

A desktop prototype that plays back a pre-recorded MP4 as if it were a live
camera feed, and runs two concurrent AI "loops" over it to assist a blind or
low-vision user: a fast **Reflex Loop** for immediate obstacle avoidance and
a slower **Cognitive Loop** for richer environmental description, both
spoken aloud through Rime AI.

## Architecture

```
                    ┌─────────────────────┐
                    │   video_reader_task  │  cv2.VideoCapture, throttled to
                    │   (main.py)          │  native FPS via asyncio.sleep
                    └──────────┬───────────┘
                               │ frame + timestamp
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
      ┌────────────────────┐      ┌────────────────────┐
      │   Reflex Loop       │      │   SharedState       │
      │  (reflex_loop.py)   │      │  (shared_state.py)  │
      │  YOLO track() every │      │  latest frame, for  │
      │  frame -> TTC/hazard│      │  the Cognitive Loop │
      └──────────┬──────────┘      └──────────┬──────────┘
                 │ CRITICAL_HAZARD             │ every 3s
                 ▼                             ▼
         hazard_queue                ┌────────────────────┐
                 │                   │  Cognitive Loop     │
                 │                   │ (cognitive_loop.py) │
                 │                   │  VLM O&M prompt      │
                 │                   └──────────┬──────────┘
                 │                              │ AMBIENT
                 │                              ▼
                 │                       ambient_queue
                 ▼                              │
        ┌─────────────────────────────────────┴─┐
        │        RimeAudioOutput (audio_output.py)│
        │  hazards preempt ambient speech via      │
        │  {"operation": "clear"} + fresh contextId│
        │  persistent wss://users-ws.rime.ai/ws3   │
        │  -> PyAudio real-time PCM playback        │
        └───────────────────────────────────────┘
```

- **Reflex Loop** (every frame): `ultralytics` YOLO tracking gives each
  object a stable ID. For objects inside the center "walking path" zone,
  it estimates time-to-collision from how fast the box is growing
  (Lee's tau / optical-expansion heuristic) and fires `CRITICAL_HAZARD`
  when TTC drops below `TTC_THRESHOLD_SEC`.
- **Cognitive Loop** (every `COGNITIVE_INTERVAL_SEC`): sends the current
  frame to a VLM (GPT-4o-mini or Gemini 1.5 Flash) with a fixed O&M system
  prompt, and reminds the model what it already described recently so it
  doesn't re-narrate the same parked car every cycle.
- **Audio Output**: one persistent Rime AI `/ws3` connection. Hazards always
  preempt ambient speech: they send `{"operation": "clear"}` and claim a
  fresh `contextId`; the receiver only plays audio chunks tagged with the
  currently active `contextId`, so stale ambient audio can't bleed into a
  hazard warning.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real keys, or export the vars directly
```

PyAudio needs PortAudio installed at the OS level:
- macOS: `brew install portaudio`
- Debian/Ubuntu: `sudo apt install portaudio19-dev`
- Windows: PyAudio wheels bundle PortAudio, usually no extra step needed.

The first run of `main.py` will download the YOLO weights (`yolov8n.pt` by
default) via `ultralytics` if not already present locally.

## Run

```bash
export RIME_API_KEY=...
export OPENAI_API_KEY=...        # or set VLM_PROVIDER=gemini and GEMINI_API_KEY
export VIDEO_PATH=/path/to/your/walk.mp4
python main.py
```

A debug window (bounding boxes, color-coded by hazard level, plus the
center "walking zone" lines) shows by default; set `SHOW_DEBUG_WINDOW=false`
to run headless. Press `q` in the debug window to quit early.

## Known approximations / where this needs more work before real-world use

- **TTC is monocular and unitless in real distance.** Without stereo/depth,
  "time to collision" here is really "rate the object is filling more of
  the frame," which correlates with approach speed but isn't a calibrated
  physical measurement. Tune `TTC_THRESHOLD_SEC`, `MIN_BOX_AREA_RATIO`, and
  `CENTER_ZONE_RATIO` against real footage before trusting it.
- **Cognitive Loop dedup is a prompt nudge, not a hard filter.** It works
  well in practice but an embedding-similarity check between consecutive
  descriptions would be more robust for a production build.
- **Rime `/ws3` keeps only one active context at a time**, per Rime's own
  docs -- our context-ID filtering is a client-side safety net on top of
  that, not a replacement for testing real interruption behavior end to
  end with your Rime account/voice.
- **This is a research prototype, not a certified mobility aid.** It has
  not been validated against the reliability, latency, and failure-mode
  standards required to replace a white cane, guide dog, or O&M
  professional's training. Treat it as a supplementary information
  channel during testing, with a human safety net, until it has been
  rigorously evaluated by O&M specialists and real users.

## Files

| File | Purpose |
|---|---|
| `main.py` | Orchestrates the video reader + both async loops |
| `config.py` | All environment-variable-driven settings |
| `shared_state.py` | Thread/task-safe "latest frame" holder |
| `reflex_loop.py` | YOLO tracking, center-zone TTC math, hazard triggering |
| `cognitive_loop.py` | Periodic VLM call + description history |
| `vlm_client.py` | OpenAI / Gemini VLM API wrapper |
| `audio_output.py` | Rime AI `/ws3` websocket + PyAudio playback |

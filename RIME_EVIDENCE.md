# Hard Voice Problem: Interruption and Recovery

### The Claim
Drishtikon successfully solves **Interruption and Recovery** by stopping queued TTS input and local playback promptly when a higher-priority event occurs, fencing obsolete results so they do not re-enter the conversation[cite: 14]. In our use case, real-time physical safety (`CRITICAL_HAZARD`) must instantly preempt both ambient scene descriptions and active user queries.

### Acceptance Test
When an ambient description or query response is actively being synthesized and spoken by Rime AI, the detection of a physical hazard in the Reflex Loop must instantly halt the current audio playback and immediately speak the hazard warning without any audio bleed from the previous utterance[cite: 14].

### Procedure
1. Initialize the system with a video feed containing a static environment followed immediately by a rapidly approaching object.
2. Trigger an ambient description of the static scene.
3. While the ambient description is playing, allow the Reflex Loop to detect the approaching object crossing the Time-To-Collision (TTC) threshold.
4. Observe the system's WebSocket payload and audio output.

### Result
The system successfully handles the interruption. The moment a hazard is queued, the audio output loop preempts the active stream by sending an `{"operation": "clear"}` command to the Rime `/ws3` endpoint, flushing the server-side buffer. It simultaneously claims a new `contextId` (UUID) for the hazard message. The local PyAudio receiver drops any incoming PCM chunks tagged with the old `contextId`, ensuring the hazard warning ("Stop! person ahead!") plays instantly with zero audio bleed from the interrupted sentence[cite: 14]. 

### Limitations
Because every new utterance claims the "active" context, if an ambient description is still playing when a legitimately fresh scene change triggers a *new* ambient description, the tail of the old sentence is cut short. Additionally, due to the lack of hardware Acoustic Echo Cancellation, the user cannot interrupt the system using their own voice (barge-in); interruption is strictly driven by the system's internal priority hierarchy (Hazard > Query > Ambient)[cite: 14].

### Reproducibility
To test this behavior, run `python main.py` using the provided `stress_test.mp4` fixture.
1. Say the wake word to trigger a long query response.
2. The video will introduce a fast-moving object at timestamp 00:04.
3. Listen as the long query response is cleanly severed and replaced by the hazard alert.
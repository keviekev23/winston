# Phase B: Cerebellum & Expression Validation

> **Parent:** `STEERING.md` Section 4
> **Subsystem docs:** `docs/architecture/brain.md`, `docs/architecture/expression.md`

## Goal

Validate whether an edge LLM can act as a "local controller" that can understand higher-level cloud model directive for state change in the scene and work with local VLM to detect the desirable changes. 

## Key Assumptions Being Tested

- edge VLM selected by phase A is performant in latency and accuracy for answering targeted object(s) states in yes/no fashion
- edge LLM can translate high-level situational change requests from cloud LLM into yes/no questions for edge VLM to process
- edge LLM can denoise and summarize high-level event changes with good accuracy and reasonable latency and trigger event to cloud LLM as appropriate
- Filler behavior (fillers when cloud is slow) feels natural, not annoying
- The overall interaction loop (speak → agent acknowledges → agent responds) feels good
- Cloud TTS produces acceptable quality at <500ms additional latency for typical utterances

## Deliverables

- [ ] SmolLM2-1.7B loaded alongside perception models (validate memory budget holds)
- [ ] Cerebellum receives perception signals → emits expression state on MQTT
- [ ] Eyes UI rendering all 8 expression states with smooth transitions
- [ ] Cerebellum generates filler text/sounds when simulated cloud delay > 3s
- [ ] End-to-end latency measurement: speech end → expression state change (target: < 500ms)
- [ ] Qualitative evaluation: 3+ test interactions, self-report on naturalness
- [ ] **Cloud TTS adapter** (`src/expression/tts/`) — evaluate providers at Phase B start, pick one:
  - ElevenLabs: highest quality, higher cost (original plan)
  - Google TTS: low cost ($4/1M chars standard), solid quality
  - Kokoro: open-source, runs locally, surprisingly good — zero cost, low latency
- [ ] **VLM MQTT prompt interface** — cerebellum wires `SCENE_DETECT_REQUEST` / `SCENE_EVENT` topics so cerebrum can send detection prompts and receive confirmed events

## Success Criteria

- Speech → expression reaction in < 500ms consistently
- All 8 eye states visually distinguishable and contextually appropriate
- Memory budget holds with 3 models loaded simultaneously (STT + cerebellum + VLM)
- Subjective: interactions feel responsive, not robotic or laggy
- Cloud TTS produces acceptable voice quality at <500ms additional latency

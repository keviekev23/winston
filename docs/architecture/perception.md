# Perception Understanding (Edge Compute)

> **Parent:** `STEERING.md` Section 2
> **Related:** `docs/architecture/mqtt_topics.md`, `docs/phases/phase_a_perception.md`

## Purpose

Transform raw sensor input into semantic signals the brain can act on.

## Models

| Component | Model | Size | Inference RAM | Role |
|-----------|-------|------|---------------|------|
| Speech-to-Text | Whisper Small.en | 244M | ~2 GB | Transcribe user speech |
| Voice Activity Detection | Silero VAD | ~2M | negligible | Detect speech start/stop |
| Speaker Diarization | pyannote/embedding or resemblyzer | ~50-100 MB | ~100-200 MB | Identify who is speaking |
| Scene Understanding | SmolVLM2-500M-Video | 500M | ~1.2-1.8 GB | Periodic scene snapshots, change detection |

## Responsibilities

1. **Speech transcription + sentiment signal:** Transcribe what the user says. Attach a confidence score per utterance. Flag low-confidence transcriptions for later clarification by the brain. Detect basic sentiment (positive/negative/neutral) from prosody cues and word choice.

2. **Speaker identification (diarization):** Identify which enrolled family member is speaking via speaker embedding matching. Each family member completes a voice enrollment during setup (a few recorded sentences → speaker embedding profile). At runtime, each transcribed utterance is matched against enrolled embeddings and tagged with `speaker_id` + confidence. Overlapping speech is flagged as low-confidence diarization — the cerebrum can resolve ambiguity through natural clarification. Per-speaker audio data feeds into the fine-tuning flywheel for speaker-specific STT improvement over time.

3. **Scene understanding (snapshot mode):** Capture scene snapshots every N seconds (configurable, default 10s). Emit general scene characteristics: location estimate, visible objects, activity state (cooking, idle, eating). Detect significant scene changes and emit alerts. Defer complex visual reasoning to cloud brain — edge VLM answers "what changed?" not "what should we do?"

4. **Sound event detection (future):** Extensible to non-speech audio events (timer beeps, alarms, doorbell). Topic structure supports this via `perception/sound/{event_type}`.

## Deferred to Later Phases

- Deictic gesture resolution ("what do you think of *this*?")
- Touch/haptic input modalities
- Continuous video understanding (vs. snapshot)

## Pseudo-Label & Confidence Pipeline

All perception outputs include a confidence score (0.0-1.0). Low-confidence outputs (threshold configurable, default <0.7) are:
1. Logged with raw input data (audio WAV, image frame) for cloud fine-tuning pipeline
2. Flagged to the brain as uncertain — brain can choose to: (a) ask user for clarification naturally in conversation, (b) use cloud model for immediate re-evaluation if critical, (c) defer and batch for offline labeling

## Data Collection for Fine-Tuning Flywheel

- All inference inputs/outputs logged locally with timestamps and confidence scores
- Raw audio stored as WAV, scene frames as JPEG
- Low-confidence samples prioritized as training candidates
- Batch upload to cloud on configurable schedule (default: daily)
- Cloud labeling pipeline: Whisper Large-v3 (STT) + GPT-4o/Claude (VLM) + multi-model consensus
- QLoRA fine-tuning on cloud GPU → adapter download → local hot-swap
- Target: meaningful personalization improvement within 2-4 flywheel cycles

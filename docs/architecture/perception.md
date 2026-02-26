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

### Population Separation (critical)

| Population | Location | Purpose |
|-----------|----------|---------|
| **Benchmark** | `data/benchmark/` | Fixed, deliberately recorded, verified GT. Measures model quality. **Never used for training.** |
| **Collection** | `data/collection/` | Real usage. Pseudo-labeled by cloud models. **Never used for evaluation.** |

Mixing these produces circular evaluation: the model learns the test set, benchmark WER drops,
but real-world quality is unchanged.

### STT Collection & Training
- All utterances saved as WAV + JSON sidecar (`save_all: true` until confidence calibration improves)
- Sidecar includes: transcript, confidence, speaker_id, low_confidence flag
- Phase C addition: `corrected_text` + `correction_source: "kevin_realtime"` for in-conversation corrections (see `brain.md`)
- Upload via `scripts/upload_training_data.py` → rclone → Google Drive
- Cloud Colab: Whisper Large-v3 pseudo-labels → QLoRA fine-tune → adapter export
- Adapter download via `scripts/download_adapter.py`, hot-swap in config

### VLM Collection & Scene Schema Tiers

Scene output is tiered by empirically measured reliability (calibrated per cycle via `label_scene_data.py --report`):

| Tier | Criteria | Brain behavior |
|------|----------|---------------|
| **1 — Trusted** | Activity accuracy > 80%; object recall > 60%, precision > 70% | Brain acts on it directly |
| **2 — Flagged** | Below thresholds or newly added | Emitted with `low_confidence: true`; brain treats as weak signal |
| **3 — On-demand** | Complex reasoning (food state, exact object, person count) | Not emitted; cerebrum loads `image_path` and queries Claude with a targeted question |

Initial Tier 1 hypotheses (validate from data): broad activity (cooking vs idle), person presence
(0 vs ≥1), change detection (pixel MAD — not VLM-dependent), large static objects (stove/fridge/counter/sink).

### VLM Annotation Strategy

Kevin has privileged information Claude lacks — he was present during collection.

| Signal | Who annotates | Rationale |
|--------|-------------|-----------|
| Activity label | Kevin via `scripts/review_scene_labels.py` | He knows what he was doing; ground truth is unambiguous |
| Objects + description | Claude via `scripts/label_scene_data.py` | Better than memory for JPEG content |

Ground truth resolution: `kevin_activity` > `claude_annotation.activity` > "unknown".
Cycle 2+: Claude drafts → Kevin spot-checks systematic failures only.

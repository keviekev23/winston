# Phase A: Perception & Fine-Tuning Validation

> **Parent:** `STEERING.md` Section 4
> **Subsystem docs:** `docs/architecture/perception.md`, `docs/architecture/mqtt_topics.md`

## Goal

Validate that edge models can perceive the kitchen usefully, and that the cloud fine-tuning flywheel works end-to-end.

## Key Assumptions Being Tested

- Whisper Small.en can transcribe kitchen speech (noisy environment, distance from mic) at acceptable accuracy
- Speaker embedding matching can reliably identify enrolled family members in normal turn-taking
- SmolVLM2-500M can produce useful scene descriptions from a MacBook camera angle
- The confidence flagging pipeline correctly identifies low-quality outputs
- Cloud labeling (Whisper Large-v3, GPT-4o/Claude) produces reliable pseudo-labels
- QLoRA fine-tuning measurably improves edge model accuracy on user-specific data

## Deliverables

**Complete:**
- [x] MQTT backbone + system health + memory monitoring operational
- [x] VAD + Whisper STT pipeline running, transcribing live speech with confidence scores
- [x] SmolVLM2-500M scene snapshot pipeline running, producing structured scene descriptions
- [x] Data collection daemon: logging all inference I/O with confidence scores
- [x] Memory profiling: SmolVLM2 ~1.2 GB + Whisper ~0.5 GB active; ~3.9 GB headroom on 16 GB
- [x] STT benchmark set + baseline evaluation (`evaluate_whisper.py`)
- [x] Cloud upload pipeline (`upload_training_data.py` via rclone → Google Drive)

**In progress / next:**
- [ ] Voice enrollment + speaker diarization (`diarization.enabled: false` until enrollment complete)
- [ ] First real-data cooking collection session — audio + visual
- [ ] First QLoRA fine-tuning cycle for Whisper (record before/after WER per speaker)
- [ ] Scene schema tier list calibrated from ≥1 collection session (see Flywheel section below)
- [ ] Per-speaker WER tracking in `evaluate_whisper.py` *(blocked until June enrolled)*
- [ ] Debug overlay: real-time MQTT message viewer alongside eyes UI

## Success Criteria

- STT latency < 2s from end of utterance to transcript available
- Speaker identification accuracy > 80% in normal turn-taking (non-overlapping speech)
- Scene snapshot processing < 3s per frame (met: ~32s per frame — see LESSONS_LEARNED 2026-02-26)
- At least one fine-tuning cycle shows measurable improvement on user-specific test set
- Memory usage stable (no leaks) over 30+ minute sustained operation
- Clear understanding of what the VLM can/cannot detect → informs scene schema design

---

## Flywheel Iteration Design

### Core Principle: Benchmark and Collection Are Strictly Separate

| Population | Location | Purpose | Used for |
|-----------|----------|---------|---------|
| **Benchmark** | `data/benchmark/` | Fixed, deliberately recorded, verified GT | WER/accuracy measurement only — **never training** |
| **Collection** | `data/collection/` | Real kitchen usage, pseudo-labeled | Training only — **never evaluation** |

If benchmark failures ever feed into training, you're optimizing to the test set. Benchmark WER drops
but real-world quality doesn't improve.

---

### STT Flywheel

#### Per-cycle procedure
```
1. Collect real kitchen audio across 1-4 cooking sessions (perception service running)
2. python scripts/upload_training_data.py --all    → sync WAV+JSON to Google Drive
3. Colab: Whisper Large-v3 pseudo-labels → QLoRA fine-tune → export adapter
4. python scripts/download_adapter.py
5. python scripts/evaluate_whisper.py --label cycle-N   → record WER per speaker/difficulty
6. python scripts/evaluate_whisper.py --compare before.json after.json
7. If WER improved across all speaker groups → deploy adapter (update config)
   If any speaker regressed → collect more from that speaker, re-run cycle
```

#### Per-speaker WER tracking

Current benchmark is Kevin-only. When June is enrolled:
1. Record ~15 benchmark utterances in June's voice (easy/medium/hard tiers)
2. Update `evaluate_whisper.py` to report WER stratified by `speaker_id`
3. After each cycle: check Kevin WER, June WER, overall separately
4. If a cycle improves Kevin but degrades June → training data is imbalanced → collect more June utterances

Until diarization is enabled, all collection is `speaker_id: null` and fine-tuning is speaker-agnostic.

#### Benchmark maintenance triggers

1. **New speaker enrolled** → record ~15 benchmark utterances for that speaker; add to benchmark
2. **Benchmark WER < 5%** (model saturated on current items) → add harder items
3. **Real-world WER diverges from benchmark WER by >5%** → audit benchmark for staleness

Benchmark expansion does NOT create training data — it only extends coverage for measurement.

---

### VLM Flywheel

#### Per-cycle procedure
```
1. Collect scene images (scene service running during cooking sessions)
2. python scripts/label_scene_data.py           → Claude annotates objects/description
3. python scripts/review_scene_labels.py        → Kevin reviews/corrects activity labels (~2 min for 60 images)
4. python scripts/label_scene_data.py --report  → check per-category accuracy
5. Update schema tier list based on precision/recall results (see Schema Tiers below)
6. Note new scenarios not yet in benchmark (multi-person, new equipment) → add to backlog
7. When volume > 200 corrected images → consider QLoRA fine-tune cycle for SmolVLM2
```

#### Annotation strategy (hybrid: Kevin + Claude)

Kevin has privileged information Claude lacks — he was physically present during cooking.

| Signal | Who annotates | Why |
|--------|-------------|-----|
| **Activity label** | Kevin (via `review_scene_labels.py`) | He knows what he was doing at minute 23. 2 min for 60 images. |
| **Objects** | Claude (via `label_scene_data.py`) | Better than memory for specific visible items in a JPEG. |
| **Description** | Claude | Used as context, not GT. |

Ground truth resolution: `kevin_activity` wins over `claude_annotation.activity` when set.

For cycle 2+ (hundreds of images): Claude annotates → Kevin spot-checks `--report` output →
corrects only systematic errors.

**Validating Claude accuracy:** `--report` surfaces patterns (e.g., Claude consistently labels
active cooking as "idle" because the person is out of frame). Spot-checking 10-15 images against
memory confirms. Where Claude is systematically wrong → Kevin corrections are the signal.

#### Scene schema tiers

Tiers are empirically calibrated from collection data — not assumed. After each cycle,
`label_scene_data.py --report` gives per-object recall and precision.

| Tier | Criteria | Brain behavior |
|------|----------|---------------|
| **1 — Trusted** | Activity accuracy > 80%; object recall > 60% and precision > 70% | Emit directly, brain acts on it |
| **2 — Flagged** | Below Tier 1 thresholds, or new/untested | Emit with `low_confidence: true`; brain treats as weak signal |
| **3 — On-demand** | Complex visual reasoning (what kind of vegetable, is stove on, how many people) | Not emitted; cerebrum loads `image_path` and queries Claude directly |

Initial hypotheses for Tier 1 (validate from data):
- Broad activity state (cooking vs idle) when scene is unambiguous
- Person presence (0 vs ≥1)
- Change detection (pixel-level MAD — not VLM-dependent, always reliable)
- Large static objects: stove, refrigerator, counter, sink

#### Benchmark maintenance triggers

1. **New person in kitchen regularly** → add multi-person scenes to `data/scene_benchmark/`, measure accuracy degradation
2. **New equipment / objects** → add to `_KITCHEN_OBJECTS` in `scene.py` + `label_scene_data.py`; new objects start Tier 2 until calibrated
3. **Activity accuracy drops on `evaluate_scene.py`** → audit new scenarios, expand benchmark

#### Fine-tuning SmolVLM2

Fine-tuning is a later milestone (200+ corrected images needed). Phase A goal is schema calibration,
not model improvement. The schema tier list IS the Phase A VLM deliverable.

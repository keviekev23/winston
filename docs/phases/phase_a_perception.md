# Phase A: Perception & Fine-Tuning Validation

> **Parent:** `STEERING.md` Section 4
> **Subsystem docs:** `docs/architecture/perception.md`, `docs/architecture/mqtt_topics.md`

## Goal

Was: Validate that edge models can perceive the kitchen usefully, and that the cloud fine-tuning flywheel works end-to-end. 
New (2026-02-26): Validate edge VLM is performant enough (latency, accuracy) to detect targeted object state changes, tracking multiple at a time (see Evaluated Assumptions).

## Key Assumptions Being Tested

- Whisper Small.en can transcribe kitchen speech (noisy environment, distance from mic) at acceptable accuracy
- Speaker embedding matching can reliably identify enrolled family members in normal turn-taking
- An on-device VLM with targeted prompts can detect specific kitchen events reliably at ≥1 fps
- The confidence flagging pipeline correctly identifies low-quality outputs
- Cloud labeling (Whisper Large-v3) produces reliable STT pseudo-labels
- QLoRA fine-tuning measurably improves Whisper accuracy on user-specific data

## Deliverables

**Complete:**
- [x] MQTT backbone + system health + memory monitoring operational
- [x] VAD + Whisper STT pipeline running, transcribing live speech with confidence scores
- [x] Data collection daemon: logging all inference I/O with confidence scores
- [x] Memory profiling: ~0.5 GB Whisper active; headroom validated
- [x] STT benchmark set + baseline evaluation (`evaluate_whisper.py`)
- [x] VLM adapter package (`src/perception/vlm/`) — abstract interface + Moondream2 implementation
- [x] Event detection script (`scripts/detect_event.py`) — scenario-driven, temporal confirmation
- [x] Seed scenario YAML (`prompts/event_detection/cooking_prep.yaml`) + prompts folder
- [x] Manual verification workflow (`scripts/label_scene_data.py --manual`)
- [x] VLM evaluation script (`scripts/evaluate_vlm.py`) — accuracy + latency per event

**Punted:**
- [ ] Voice enrollment + speaker diarization (`diarization.enabled: false` until enrollment complete)
- [ ] First real-data cooking collection session — audio + visual
- [ ] Cloud upload pipeline (`upload_training_data.py` via rclone → Google Drive)
- [ ] First QLoRA fine-tuning cycle for Whisper (record before/after WER per speaker)
- [ ] Per-speaker WER tracking in `evaluate_whisper.py` *(blocked until June enrolled)*

**In progress / next:**
- [x] Label token optimization — short labels (CUT/WASH/IDLE/NONE) + prompt "label ONLY" (v2 YAML)
- [x] `--benchmark-latency N` mode in `detect_event.py` — explicit clean-env latency measurement
- [ ] **Run `--benchmark-latency 20` with all apps closed** — record definitive latency baseline
- [ ] **Lock on VLM model** — run detection sessions, verify, evaluate, confirm ≥1 fps + accuracy

## Success Criteria

- Moondream2 detection latency < 1000ms/frame (≥1 fps) — measured via `evaluate_vlm.py`
- Manual verification shows reliable event detection for primary events in `cooking_prep.yaml`
- VLM model locked for Phase B (or InternVL2-1B activated if Moondream2 fails)

**Punted:**
- STT latency < 2s from end of utterance to transcript available
- Speaker identification accuracy > 80% in normal turn-taking (non-overlapping speech)
- At least one fine-tuning cycle shows measurable improvement on user-specific test set
- Memory usage stable (no leaks) over 30+ minute sustained operation

---

## STT Flywheel

### Core Principle: Benchmark and Collection Are Strictly Separate

| Population | Location | Purpose | Used for |
|-----------|----------|---------|---------|
| **Benchmark** | `data/benchmark/` | Fixed, deliberately recorded, verified GT | WER/accuracy measurement only — **never training** |
| **Collection** | `data/collection/` | Real kitchen usage, pseudo-labeled | Training only — **never evaluation** |

### Per-cycle procedure
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

### Per-speaker WER tracking

Current benchmark is Kevin-only. When June is enrolled:
1. Record ~15 benchmark utterances in June's voice (easy/medium/hard tiers)
2. Update `evaluate_whisper.py` to report WER stratified by `speaker_id`
3. After each cycle: check Kevin WER, June WER, overall separately
4. If a cycle improves Kevin but degrades June → training data is imbalanced → collect more June utterances

### Benchmark maintenance triggers

1. **New speaker enrolled** → record ~15 benchmark utterances for that speaker; add to benchmark
2. **Benchmark WER < 5%** → add harder items
3. **Real-world WER diverges from benchmark WER by >5%** → audit benchmark for staleness

---

## VLM Evaluation: Targeted Event Detection

### Paradigm

VLM is **cerebrum-directed**: given a targeted classification prompt from a scenario YAML,
it classifies each frame and confirms when an event accumulates N consecutive detections.
See `STEERING.md` for the full integration design.

### Tools

| Script | Purpose |
|--------|---------|
| `scripts/detect_event.py` | Run event detection with a scenario YAML, beep + save on trigger |
| `scripts/label_scene_data.py --manual` | Manually verify captured images (y/n) |
| `scripts/evaluate_vlm.py` | Compute accuracy + latency from verified images |
| `prompts/event_detection/` | Scenario YAMLs — versioned evaluation artifacts, never delete |

### VLM Evaluation Cheatsheet

> **IMPORTANT:** If the evaluation process flow changes (new scripts, new output directories,
> new steps), update this cheatsheet BEFORE the change is merged. A new session should be
> able to follow this cold without reading any other file.

#### 0. Prerequisites (one-time setup)
```bash
# Verify InternVL2.5-1B loads correctly (active model)
python -c "from src.perception.vlm.internvl2 import InternVL2Adapter; a = InternVL2Adapter(); a.load(); print('OK'); a.unload()"

# Verify Moondream2 loads correctly (secondary candidate — needs transformers<5.0)
# NOTE: Moondream2 and mlx-vlm (InternVL2 backend) cannot coexist in the same venv.
#       Moondream2 requires transformers<5.0; mlx-vlm requires transformers>=5.0.
#       Run Moondream2 tests in a separate environment if needed.
python -c "from src.perception.vlm.moondream import MoondreamAdapter; a = MoondreamAdapter(); a.load(); print('OK'); a.unload()"

# Verify camera is accessible
python -c "from src.perception.camera import Camera; c = Camera(); c.open(); f = c.capture(); print(f.size); c.close()"
```

#### 0.5. Latency benchmark (REQUIRED before locking VLM model)
> **⚠️ Close all other applications first.** Concurrent load invalidates measurements.
> This is the only valid source of Phase A exit gate latency numbers.
```bash
# InternVL2.5-1B (active candidate)
python scripts/detect_event.py \
  --scenario prompts/event_detection/cooking_prep.yaml \
  --adapter internvl2_1b \
  --benchmark-latency 20

# Moondream2 (if evaluating as alternative — separate venv required)
python scripts/detect_event.py \
  --scenario prompts/event_detection/cooking_prep.yaml \
  --adapter moondream2 \
  --benchmark-latency 20
```
- Prints a clean-environment confirmation prompt
- Runs 1 JIT warmup frame (excluded from stats)
- Reports mean/min/max/p50/p90 latency and fps estimate
- Record results in `docs/LESSONS_LEARNED.md` before locking the model

#### 1. Run an event detection session
```bash
python scripts/detect_event.py \
  --scenario prompts/event_detection/cooking_prep.yaml \
  [--interval 1.0]           # seconds between frames (default: 1.0)
  [--adapter internvl2_1b]   # model adapter (default: internvl2_1b); also: moondream2
  [--confirm-frames N]       # override per-event confirm_frames from YAML
```
- Logs per-frame: detected label, confidence, latency_ms
- Beeps + prints "EVENT TRIGGERED: {event_id}" when confirm_frames accumulated
- Saves JPEG + `_detection.json` to `data/detection/`
- Exits after first confirmed event — re-run to capture more

#### 2. Manually inspect saved images
```bash
python scripts/label_scene_data.py --manual [--dir data/detection/]
```
- Opens each JPEG in Preview (macOS)
- Prompts: `Event '{event_id}' detected correctly? [y/n/skip]:`
- Saves `*_verified.json` alongside each image

#### 3. Run evaluation report
```bash
python scripts/evaluate_vlm.py [--label phase_a_baseline] [--adapter internvl2_1b]
# also valid: --adapter moondream2
```
- Reads all `*_verified.json` files from `data/detection/`
- Prints per-scenario, per-event accuracy + avg latency_ms + estimated fps
- Saves to `data/scene_benchmark/evals/{timestamp}_{adapter}_{label}.json`
- Updates `evaluations[]` in each scenario YAML
- Note: latency figures in the report are from detection sessions (not clean-environment
  benchmarks) — use `--benchmark-latency` output for exit gate decisions

#### 4. Review results
- **Console:** Accuracy table + fps estimate + Phase A exit gate assessment
- **Eval JSON:** `data/scene_benchmark/evals/` — compare with `--compare before.json after.json`
- **Scenario YAMLs:** `evaluations[]` grows longitudinally — full history of what was tried

#### 5. Iterate a prompt
1. Edit event `description` in scenario YAML, bump `version`, add a note explaining why
2. Re-run detect → label → evaluate
3. Compare: `python scripts/evaluate_vlm.py --compare evals/before.json evals/after.json`

#### Exit Gate
Lock VLM model when:
- **`--benchmark-latency` confirms mean < 1000ms** in a clean-environment run (≥1 fps)
  — do NOT use latency from regular detection sessions (concurrent load skews numbers)
- Manual verification accuracy acceptable for primary events (set threshold from first data)
- At least one scenario YAML has `evaluations[]` populated

---

## Evaluated Assumptions

### SmolVLM2-500M General Scene Understanding — 2026-02

**Original assumption:** On-device VLM (SmolVLM2-500M) may be accurate enough for semantic scene understanding.

**What we did:** Collected kitchen scene images during cooking sessions. Ran SmolVLM2 scene descriptions on collected images. Manually reviewed outputs for quality.

**Outcome:** Description quality was insufficient for reliable activity/object classification. Critically, inference runs at ~32s/frame on M2 Pro MPS — incompatible with the ≥1 fps target for real-time event detection. SmolVLM2 fails both on quality and latency.

**Impact:**
1. Shifted to cerebrum-directed targeted event detection paradigm
2. SmolVLM2 disqualified — Moondream2 (1.86B, ~1-3 fps on MPS) is Phase A primary candidate
3. SmolVLM2 adapter not implemented (disqualified before adapter work began)
4. VLM adapter abstraction (`src/perception/vlm/`) built to enable fast model swapping

**Relevant data:** `data/collection/images/` (collected samples), `scripts/label_scene_data.py --report`

### Local TTS — 2026-02

**Original assumption:** Local TTS would be used for Phase A/B voice output.

**What we did:** Phase A was perception-focused; TTS was deferred from Phase A scope.

**Outcome:** No local TTS code was built. Local TTS adds memory pressure (~500MB-1GB) and requires significant quality-to-latency tuning — not worth the effort given cloud TTS availability.

**Impact:** TTS moved to Phase B as a cloud adapter (provider to be selected at Phase B start — evaluate ElevenLabs, Google TTS, and Kokoro). No local TTS in Phase A or B.

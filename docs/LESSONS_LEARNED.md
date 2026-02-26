# Lessons Learned — Winston Project

Operational findings, surprises, and hard-won knowledge. Both Kevin and Claude contribute here. Entries are chronological with date and context.

---

### 2026-02-24 — Phase A Layer 1-2 Implementation (Flywheel Gap Retrospective)

Three gaps surfaced *during implementation* that should have been caught *during planning*. Pattern and root cause documented here so future phases catch them earlier.

**Gap 1: Upload trigger had no mechanism.** The Phase A deliverable said "batch upload to cloud." Neither the doc nor the plan named the tool, credentials, or command that would execute this. Result: we had to design rclone + Drive sync after the fact. Root cause: deliverable written at *what* level, not *how*.

**Gap 2: Confidence threshold was an uncalibrated assumption.** The value `0.7` in `config/default.yaml` was chosen arbitrarily. No methodology existed to validate it, which means the flywheel could be uploading the wrong samples (wrong confidence boundary) and we'd have no way to know. Root cause: any parameter in a plan that controls selection/routing needs an empirical validation method before it has meaning.

**Gap 3: "Measure WER before/after" implied a benchmark that didn't exist.** A before/after comparison requires a fixed benchmark set recorded before fine-tuning begins. Without it, you have no baseline to compare against. Root cause: "measure X" in a deliverable list is incomplete without "using what instrument on what test set."

**What catches these gaps:** Three questions to ask for every phase deliverable before writing code — now in `CLAUDE.md` point 6 under "When Making Decisions": (a) what concrete tool/command makes this happen? (b) what benchmark/baseline measures success? (c) which parameters need empirical calibration?

---

### 2026-02-26 — Phase A Layer 4: SmolVLM2 on MPS

**transformers 5.x renamed `AutoModelForVision2Seq` to `AutoModelForImageTextToText`.** If you get `ImportError: cannot import name 'AutoModelForVision2Seq'`, it's this. Update all VLM loading code accordingly.

**SmolVLM2-500M-Video-Instruct on MPS requires `torchvision` and `num2words`.** Neither is pulled in transitively. Both must be explicit dependencies. `torchvision` is needed for the video processor's transform pipeline; `num2words` is required by `SmolVLMProcessor.__init__`.

**`sdpa` attention is 2.4x faster than `eager` on MPS for SmolVLM2.** Measured on M2 Pro 16 GB:
- `eager`: ~75s inference, 40-370s load (highly variable, likely memory pressure sensitive)
- `sdpa`: ~32s inference, ~22s load
- `sdpa` works correctly on MPS with PyTorch 2.10. The old advice to use `eager` for MPS compatibility was for older PyTorch versions. Always try `sdpa` first.

**SmolVLM2-500M inference is ~32s per frame on M2 Pro MPS.** This sets a hard lower bound on the snapshot interval — must be > 32s. Default config set to 60s (32s inference + ~28s sleep). For a kitchen scene monitor this cadence is fine; meaningful activity changes take tens of seconds to minutes.

**Memory footprint: ~1.2 GB unified memory** for SmolVLM2-500M in bfloat16 on MPS. This leaves ~3.9 GB headroom when running alongside Whisper Small (~0.5 GB) with ~1.1 GB system overhead on an otherwise-idle 16 GB Mac.

---

### 2026-02-23 — Design Session (Pre-Implementation)

- **Memory budget is tighter than it looks on paper.** Estimated 3-5 GB headroom with all models loaded, but real-world overhead (Python runtime, MQTT broker, web UI, macOS background processes) may reduce this to 1-2 GB. Phase A memory profiling is critical before committing to model sizes. Have mitigation levers ready: reduce VLM resolution, more aggressive quantization, lazy-loading.

- **SmolVLM2-500M is the sweet spot for edge VLM.** The 256M→500M jump yields ~8% average benchmark improvement for only 0.4 GB more RAM — best value-per-parameter in the SmolVLM family. The 500M video variant has capabilities close to the 2.2B model. However, it's optimized for structured visual tasks (OCR, documents, charts) more than open-ended scene understanding. Expect it to be the weakest perception link for kitchen scenes; design around it with cloud fallback.

- **Speaker diarization belongs in the perception phase, not later.** The `speaker_id` field propagates through the entire system (brain reactions, memory attribution, personalization). Adding it retroactively would require refactoring every downstream consumer. Enrollment-based embedding matching is lightweight enough (~100-200 MB) to include from the start.

- **Defer empathy, not patience.** AI emotional attunement risks the uncanny valley and sets expectations the system can't meet. But practical patience (soft eyes, waiting state, offering help without pressure) is both achievable and valuable. The distinction matters for expression engine design.

- **The cerebrum should be an orchestrator, not a responder.** Framing the cloud LLM as a plan lifecycle manager (create → execute → replan → complete) with delegation capability future-proofs for when planning logic moves to local models. A Q&A framing would need to be rewritten later.

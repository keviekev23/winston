# Lessons Learned — Winston Project

Operational findings, surprises, and hard-won knowledge. Both Kevin and Claude contribute here. Entries are chronological with date and context.

---

### 2026-02-27 — Phase A Latency Measurements Are Invalid (Concurrent Load)

**Context:** All InternVL2.5-1B latency measurements taken to date were collected with other
applications running in the background (browser, IDE, other processes). This invalidates them
as Phase A exit gate evidence.

**What this means:** The ~800ms (max_tokens=5) and ~2s (max_tokens=10) warm latency figures
in LESSONS_LEARNED are directionally informative but not definitive. The clean-environment
numbers could be meaningfully better or worse.

**Action required:** Before locking the VLM model for Phase B, run an explicit latency
benchmark with all other applications closed:
```bash
python scripts/detect_event.py \
  --scenario prompts/event_detection/cooking_prep.yaml \
  --adapter internvl2_1b \
  --benchmark-latency 20
```
The `--benchmark-latency N` flag is the only valid source of Phase A exit gate latency numbers.
It prints a clean-environment warning, runs a JIT warmup frame, then N timed frames with stats.

---

### 2026-02-27 — Phase A VLM Latency: Full Benchmark on M2 Pro

**Context:** Attempted to run Moondream2 and then InternVL2.5-1B on real 1280x720 camera frames.

**Moondream2 via transformers/PyTorch (vikhyatk/moondream2 rev "2025-01-09"):**
- Load: ~40s from cache. Inference: ~20s/frame warm on MPS (token-by-token generation + MPS sync overhead per `.item()` call). Unusable.
- transformers 5.x breaks the `trust_remote_code` path (`all_tied_weights_keys` API change). Downgrade to `transformers<5.0` is required. But even then, latency is unacceptable.

**mlx-vlm is the right backend for VLMs on Apple Silicon.** transformers/PyTorch for generation is crippled by per-token MPS sync overhead. mlx-vlm uses Metal natively with no sync overhead.

**InternVL2.5-1B-4bit via mlx-vlm (`mlx-community/InternVL2_5-1B-4bit`):**
- Load: ~20-40s from cache (~2 GB weights)
- First frame (JIT cold): ~5-9s
- Warm frames: 1.6-3.0s with `max_tokens=10`; ~800ms with `max_tokens=5`
- **max_tokens is the dominant latency lever.** Generation at ~3-5 tok/s means 10 tokens = 2-3s, 5 tokens = ~800ms.
- **Image resolution doesn't affect token count.** Full 1280x720 produces the same `prompt_tokens=36` as 224x224 or 448x448 — mlx-vlm's InternVL2 processor uses a fixed compact visual representation regardless of input size. Do NOT resize camera frames before inference; it doesn't help latency and may reduce accuracy.
- **Label truncation at max_tokens=5:** "CUTTING_VEGETABLES" gets cut to "CUTTING_VEGET" (7 tokens). max_tokens=10 captures all labels reliably.

**Phase A exit gate (<1000ms, ≥1fps) is NOT met on real camera input with safe max_tokens=10.**
Best achievable warm latency: ~800ms with max_tokens=5 (but truncation risk).
Discussion needed: whether 0.4fps (max_tokens=10) is acceptable for kitchen activity detection, or whether to pursue label token optimization.

**Dependency note:** mlx-vlm requires transformers>=5.0 as a transitive dep (conflicts with the moondream2 path). Since moondream2 is disqualified for latency anyway, this isn't a problem — but be aware: installing mlx-vlm will upgrade transformers to 5.x. Running moondream2 in the same environment as mlx-vlm will fail at `load()` time (trust_remote_code breaks under transformers 5.x). The adapters import lazily inside load() so the script won't crash at startup — just switch adapters with `--adapter`.

**Label token optimization (2026-02-27):** Shortened scenario labels to single-token words (CUT/WASH/IDLE/NONE → ~1 token each) so max_tokens=5 safely captures all labels. v1 compound labels (CUTTING_VEGETABLES = ~6 tokens, WASHING_PRODUCE = ~5 tokens) were truncated at max_tokens=5. Also removed "followed by explanation" from prompt instruction — the explanation sentence cost 5-15 output tokens per call at ~3-5 tok/s with no benefit (parse logic only reads the label token). These changes allow max_tokens=5 to be used reliably, keeping target warm latency ~800ms rather than ~2s.

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

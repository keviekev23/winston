# Lessons Learned — Winston Project

Operational findings, surprises, and hard-won knowledge. Both Kevin and Claude contribute here. Entries are chronological with date and context.

---

### 2026-02-23 — Design Session (Pre-Implementation)

- **Memory budget is tighter than it looks on paper.** Estimated 3-5 GB headroom with all models loaded, but real-world overhead (Python runtime, MQTT broker, web UI, macOS background processes) may reduce this to 1-2 GB. Phase A memory profiling is critical before committing to model sizes. Have mitigation levers ready: reduce VLM resolution, more aggressive quantization, lazy-loading.

- **SmolVLM2-500M is the sweet spot for edge VLM.** The 256M→500M jump yields ~8% average benchmark improvement for only 0.4 GB more RAM — best value-per-parameter in the SmolVLM family. The 500M video variant has capabilities close to the 2.2B model. However, it's optimized for structured visual tasks (OCR, documents, charts) more than open-ended scene understanding. Expect it to be the weakest perception link for kitchen scenes; design around it with cloud fallback.

- **Speaker diarization belongs in the perception phase, not later.** The `speaker_id` field propagates through the entire system (brain reactions, memory attribution, personalization). Adding it retroactively would require refactoring every downstream consumer. Enrollment-based embedding matching is lightweight enough (~100-200 MB) to include from the start.

- **Defer empathy, not patience.** AI emotional attunement risks the uncanny valley and sets expectations the system can't meet. But practical patience (soft eyes, waiting state, offering help without pressure) is both achievable and valuable. The distinction matters for expression engine design.

- **The cerebrum should be an orchestrator, not a responder.** Framing the cloud LLM as a plan lifecycle manager (create → execute → replan → complete) with delegation capability future-proofs for when planning logic moves to local models. A Q&A framing would need to be rewritten later.

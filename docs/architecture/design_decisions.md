# Design Decisions & Rationale

> **Parent:** `STEERING.md`
> **Convention:** New decisions are appended with the next number. Decisions are never deleted — if superseded, mark as `[SUPERSEDED by #N]`.

## 1. Why 2-LLM instead of 1?

Latency. A single cloud LLM means 1-3s of dead air on every turn. The cerebellum provides immediate conversational presence while the cerebrum thinks.

## 2. Why MQTT instead of direct function calls?

Decoupling. Each subsystem can be developed, tested, and restarted independently. MQTT also gives us a natural logging/debugging layer.

## 3. Why ChromaDB for episodic memory?

Simplicity. The research question is whether cross-session memory helps at all, not whether the retrieval is optimal. ChromaDB is zero-config and in-process.

## 4. Why snapshot VLM instead of continuous video?

Memory and compute. Continuous video on 16 GB alongside three other models is infeasible. Snapshots every 10s are sufficient — cooking state changes slowly.

## 5. Why defer empathy/emotional attunement?

Risk of uncanny valley. The agent handles negative moments with patience and practical help, not emotional mirroring.

## 6. Why ElevenLabs over local TTS?

Quality. On-device TTS that sounds natural is unsolved at this model size. The cerebellum covers the latency gap with immediate non-verbal reactions.

## 7. Why cerebrum as orchestrator?

Future-proofing. Planning logic may eventually move to local models or be distributed. An orchestration model is cleaner than treating the cloud model as a Q&A endpoint.

## 8. Why separate session types (cooking episode vs conversation)?

Different interaction patterns need different summaries. Cooking sessions are structured (steps, timing, outcome). Conversations are freeform (topics, preferences, mood).

## 9. Why "dreaming" as async post-session?

Real-time memory updates add complexity and latency. Batch consolidation is simpler, mirrors human cognition, and lets us evaluate impact cleanly.

## 10. Why speaker diarization in Phase A (not later)?

The `speaker_id` field propagates through the entire system. Adding it retroactively would require refactoring every downstream consumer. Enrollment-based embedding matching is lightweight enough to include from the start.

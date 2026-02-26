# Phase C: First End-to-End (Cooking Prototype)

> **Parent:** `STEERING.md` Section 4
> **Builds on:** Phase A (perception works) + Phase B (interaction feels responsive)

## Goal

Assemble all subsystems into a working cooking assistant. Validate the core value proposition: does this agent meaningfully help with cooking?

## Additional Components Integrated

- Cerebrum (Sonnet 4.6) for planning, reasoning, and conversation
- TTS (ElevenLabs) for spoken responses
- Recipe parsing + step-by-step guidance skill
- Timer management skill
- Structured profiles + episodic memory (ChromaDB)
- Session lifecycle (cooking episodes + conversation sessions with 5-min timeout)
- Persona, mental model, and sous_chef role prompts

## Deliverables

- [ ] Full system operational: perception → brain (cerebellum + cerebrum) → expression
- [ ] Agent guides user through a complete cooking recipe via voice
- [ ] Agent handles cooking questions (substitutions, timing, techniques)
- [ ] Timer management: multiple concurrent timers with proactive alerts
- [ ] Session summaries generated for both cooking episodes and conversation sessions
- [ ] Conversation session auto-ends after 5 min inactivity (when not awaiting user response)
- [ ] Episodic memory: session summaries stored and retrievable
- [ ] Structured profiles: basic family member data, dietary restrictions, preferences
- [ ] Spaces: kitchen objects tracked, states updated from VLM snapshots

## Success Criteria (measured over 5+ cooking sessions)

1. **Task performance:** Self-reported cooking difficulty (1-5), time-to-complete, error count
2. **Interaction quality:** Unprompted engagements/session, suggestion acceptance rate, clarification frequency
3. **Trust/stickiness:** "Would you use again?" (1-5), naturalness of interaction (1-5)
4. **Mental model quality:** After N sessions, accuracy of preference predictions (spot-check)
5. **Personalization delta:** Compare session 1 vs session 5 metrics

## Additional Deliverable: Real-Time Flywheel Loop

When the cerebrum receives `low_confidence: true` from perception:
- [ ] Generate a natural clarification request when the ambiguity affects the current response
  (e.g., "did you say X?", "I can see something changed — what are you working on?")
- [ ] Store the user's correction in the collection sidecar with `correction_source: "kevin_realtime"`
- [ ] Corrections are prioritized by the upload/training pipeline over pseudo-labeled samples

Design constraints (defer full design to Phase C implementation):
- Rate-limit: ask only when ambiguity affects current response AND asking is conversationally natural
- Threshold requires Phase A failure data to calibrate — do not hard-code
- STT corrections (`corrected_text`) and VLM corrections (`corrected_activity`) are separate queues

See `docs/architecture/brain.md` — Perception Confidence Integration for storage schema.

## Research Questions

- Does a stationary multimodal agent provide meaningful value during cooking?
- Does cross-session memory improve task performance and/or conversational quality?
- What is the right balance of proactive help vs. waiting to be asked?
- How quickly does personalization (via flywheel) produce noticeable improvements?
- What are the failure modes that break user trust?
- **Does real-time active learning (conversational corrections) produce higher-quality flywheel signal than batch post-session labeling?** Hypothesis: yes — in-the-moment correction has fresher context and unambiguous ground truth. Measure by comparing improvement rate on benchmark between realtime-corrected vs pseudo-labeled samples across flywheel cycles.

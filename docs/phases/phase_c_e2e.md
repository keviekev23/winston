# Phase C: First End-to-End (Cooking Prototype)

> **Parent:** `STEERING.md` Section 4
> **Builds on:** Phase A (perception + VLM model locked) + Phase B (interaction feels responsive, cloud TTS working)

## Goal

Assemble all subsystems into a working cooking assistant. Validate the core value proposition: does this agent meaningfully help with cooking?

## Components Integrated

- Cerebrum (Sonnet 4.6) for planning, reasoning, and conversation
- Cloud TTS (provider selected in Phase B)
- Recipe parsing + step-by-step guidance skill
- Timer management skill
- Structured profiles + episodic memory (ChromaDB)
- Session lifecycle (cooking episodes + conversation sessions with 5-min timeout)
- Persona, mental model, and sous_chef role prompts

**VLM integration (cerebrum-directed):**
1. Cerebrum determines when it needs scene context (based on recipe step, user question, etc.)
2. Cerebrum sends a targeted detection prompt via `SCENE_DETECT_REQUEST` MQTT
3. VLM confirms event via temporal window, fires on `SCENE_EVENT` with `image_path`
4. Cerebrum uses image directly (Anthropic API vision) for complex visual reasoning

Cerebrum owns the "when" and "what to look for" — edge VLM owns the "is it happening now?"

## Deliverables

- [ ] Full system operational: perception → brain (cerebellum + cerebrum) → expression
- [ ] Agent guides user through a complete cooking recipe via voice
- [ ] Agent handles cooking questions (substitutions, timing, techniques)
- [ ] Timer management: multiple concurrent timers with proactive alerts
- [ ] Session summaries generated for both cooking episodes and conversation sessions
- [ ] Conversation session auto-ends after 5 min inactivity
- [ ] Episodic memory: session summaries stored and retrievable
- [ ] Structured profiles: basic family member data, dietary restrictions, preferences
- [ ] Cerebrum selects scenario YAML based on recipe step type (`step_type → scenario_yaml` mapping)
- [ ] Cerebrum uses `SCENE_EVENT` image_path for visual context in Anthropic API calls

## Success Criteria (measured over 5+ cooking sessions)

1. **Task performance:** Self-reported cooking difficulty (1-5), time-to-complete, error count
2. **Interaction quality:** Unprompted engagements/session, suggestion acceptance rate, clarification frequency
3. **Trust/stickiness:** "Would you use again?" (1-5), naturalness of interaction (1-5)
4. **Mental model quality:** After N sessions, accuracy of preference predictions (spot-check)
5. **Personalization delta:** Compare session 1 vs session 5 metrics

## Real-Time Flywheel Loop

When cerebrum receives `low_confidence: true` from perception:
- [ ] Generate a natural clarification request when ambiguity affects the current response
- [ ] Store user's correction in collection sidecar with `correction_source: "kevin_realtime"`
- [ ] Corrections prioritized by upload/training pipeline over pseudo-labeled samples

Design constraints: rate-limit to conversationally natural moments; thresholds calibrated from Phase A failure data.

## Research Questions

- Does a stationary multimodal agent provide meaningful value during cooking?
- Does cross-session memory improve task performance and/or conversational quality?
- What is the right balance of proactive help vs. waiting to be asked?
- How quickly does personalization (via flywheel) produce noticeable improvements?
- What are the failure modes that break user trust?
- **Does real-time active learning (conversational corrections) produce higher-quality flywheel signal than batch post-session labeling?** Hypothesis: yes — fresher context, unambiguous GT. Measure by comparing improvement rate across flywheel cycles.
- **Should cerebrum emit `confirm_frames` per event dynamically, or are thresholds baked into scenario YAMLs?** Dynamic allows execution-context calibration; static is simpler and avoids round-trip latency. Evaluate once cerebrum-VLM integration is running.
- **What is the right `step_type → scenario_yaml` mapping?** Options: hardcoded table, recipe-level metadata, or cerebrum-generated selection. Evaluate once we have ≥3 recipes in the system.

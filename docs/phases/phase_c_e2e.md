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

## Research Questions

- Does a stationary multimodal agent provide meaningful value during cooking?
- Does cross-session memory improve task performance and/or conversational quality?
- What is the right balance of proactive help vs. waiting to be asked?
- How quickly does personalization (via flywheel) produce noticeable improvements?
- What are the failure modes that break user trust?

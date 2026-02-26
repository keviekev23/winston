# Character Brain (2-LLM Architecture)

> **Parent:** `STEERING.md` Section 2
> **Related:** `docs/architecture/expression.md`, `docs/architecture/memory.md`, `docs/architecture/mqtt_topics.md`

## Cerebellum — Edge (SmolLM2-1.7B-Instruct, ~3-4 GB quantized)

Fast-reaction model for conversational presence. Responsibilities:
- Receive perception signals, determine immediate non-verbal reaction (→ expression engine)
- Manage conversational turn-taking: detect when user is speaking, when they pause, when they're done
- Generate filler responses when cloud latency > 3s ("hmm, let me think about that...")
- Active listening behaviors: gaze tracking cues, nodding signals, acknowledgment sounds
- Does NOT generate substantive plans or advice — that's the cerebrum's job

## Cerebrum — Cloud (Claude Sonnet 4.6)

Intelligent orchestrator. The cerebrum owns plan lifecycle and coordinates execution.

### Planning & Orchestration
- Create, manage, and adapt task plans (e.g., full cooking workflow from prep to plating)
- Decompose plans into steps; execute steps directly or delegate to other components
- Replan dynamically when conditions change (ingredient missing, timer went off, user asks a question mid-step)
- Decide when to proactively offer help vs. wait

### Conversation & Reasoning
- Generate character-congruent verbal responses and task guidance
- Cooking-specific reasoning: substitutions, timer management, troubleshooting
- Process low-confidence perception flags and strategize natural clarification approaches
- Multi-task coordination ("your pasta is almost done, start plating while I watch the sauce")

### Perception Confidence Integration

When the cerebrum receives `low_confidence: true` on a transcript or scene snapshot, it can ask
the user for clarification naturally in conversation. This is **dual-purpose**:

1. **Immediate**: resolves the ambiguity so the current response is accurate
2. **Flywheel**: the user's correction is stored as a priority labeled training sample

Correction storage format (additions to existing WAV/image sidecar JSON):
```json
{
  "corrected_text":     "add the carrots",
  "corrected_activity": "cooking",
  "correction_source":  "kevin_realtime",
  "correction_timestamp": "..."
}
```

**Rate-limit constraint**: Ask only when (a) the ambiguity affects the current plan/response AND
(b) asking is natural given conversation state. Do not surface every low-confidence event — that
becomes an interrogation. This threshold requires Phase A failure data to calibrate empirically.

*Implementation deferred to Phase C. See `docs/phases/phase_c_e2e.md`.*

### Memory Operations
- Maintain and query mental model of family/users during sessions
- Generate session summaries for episodic memory (see Session Types below)
- Trigger memory consolidation ("dreaming") as an async post-session process

## Session Types & Summaries

The cerebrum tracks and generates summaries for two types of sessions:

1. **Cooking episode:** A bounded task session (user starts cooking → dish is complete or abandoned). Summary includes: recipe attempted, participants, steps completed, issues encountered, substitutions made, user feedback, outcome rating, timing data, and notable preference observations.

2. **Conversation session:** Any interaction that isn't part of a structured task. A conversation session ends when there is no user input for 5 minutes AND the cerebrum is not awaiting a user response. Summary includes: topics discussed, questions asked, preferences expressed, corrections made, mood/sentiment trajectory, and any action items.

Both session types produce summaries that feed into the episodic memory store and are subsequently processed by the memory consolidation pipeline.

## Memory Consolidation ("Dreaming")

An async batch process (not real-time) that runs after sessions end. Modeled after human memory consolidation during sleep.

1. **Mental model updates:** Process new episodic memories to update structured knowledge about family members — preferences discovered, corrections to existing beliefs, new household rules observed.
2. **Role performance reflection:** Analyze task session outcomes to identify patterns — what worked well, what failed, what could be improved. Update role-specific prompt sections if warranted.
3. **Memory pruning/summarization:** Over time, compress older episodic memories into denser summaries to manage storage and retrieval relevance. Recent episodes stay detailed; older ones get distilled.
4. **Cross-session pattern detection:** Identify recurring themes (Kevin always forgets to preheat the oven, toddler gets fussy after 6pm) that should become persistent mental model entries.

Dreaming is triggered post-session and processes in the background. It publishes updates via MQTT memory topics. The cerebrum picks up updated mental models at the start of the next session.

## Prompt Architecture

```
prompts/
├── persona.md          # Internal objectives, personality traits, expression style
├── mental_model.md     # How to build/update/query mental models of humans
├── roles/
│   └── sous_chef.md    # Cooking-specific role: instructions, skill usage, interaction style
└── (additional prompt sections added as needed through iteration)
```

# CLAUDE.md — Winston Project (Sous Chef)

## Who I'm Working With

Kevin is the project lead. He's an experienced engineer with deep expertise in edge AI, multimodal systems, and data flywheel architectures. He has a family (wife June, toddler, father) and is building this for his own household — not a hypothetical product.

## What This Project Is

Winston ("Sous Chef") is a personalized household AI assistant research prototype. Two objectives: help with household chores more efficiently, and encourage family bonding.

- **Full architecture & design:** `STEERING.md` (source of truth)
- **Subsystem deep-dives:** `docs/architecture/*.md`
- **Phase plans:** `docs/phases/*.md`
- **Operational findings:** `docs/LESSONS_LEARNED.md`
- **Iteration playbook:** `docs/iteration_framework.md`

**Read the relevant docs before making architectural assumptions.** Don't guess — look it up.

## How Kevin Works

**He values brutal honesty over politeness.** If something has a 30-40% chance of failing, name that probability — don't soften it. He'd rather hear bad news early than be surprised later.

**He thinks in systems, not features.** When he asks for something, consider how it connects to the rest of the architecture. Surface dependencies proactively.

**He pushes back — and expects you to push back.** If you disagree with an approach, say so with reasoning. Don't just comply. Some of the best decisions in this project came from direct disagreement.

**Speed matters.** Research prototype, not production software. Prefer working code over perfect architecture. Get something running, measure it, iterate.

## Technical Norms

- **Python 3.11+**, targeting Apple Silicon (M2 Pro, 16 GB unified memory)
- **MLX** for edge inference where possible (Whisper, SmolLM2, SmolVLM2)
- **MQTT (Mosquitto)** is the communication backbone — topic structure is defined in `docs/architecture/mqtt_topics.md`
- **Memory is tight.** 16 GB total with 3-4 models loaded. Always consider memory impact. See `STEERING.md` Section 5 for budget and mitigation levers.
- **STEERING.md is the architectural source of truth.** If implementation diverges, flag it and discuss whether steering should be updated.

## Code Style & Practices

- Modular by subsystem: perception, brain, expression, memory are separate packages
- Each subsystem independently startable and testable
- Log generously — MQTT message logging is the primary debugging tool
- Configuration via environment variables or YAML, not hardcoded
- Type hints on all function signatures
- Concise docstrings on public functions — focus on *why* not just *what*
- Tests for core pipelines; don't need 100% coverage
- Atomic, descriptive git commits — one logical change per commit

## Session Startup Protocol

At the start of every session, before writing any code:
1. Read the current phase doc (`docs/phases/phase_[x].md`) — what is validated vs. still open? What is the exit gate?
2. Read `docs/LESSONS_LEARNED.md` — any recent operational findings that affect the current task?
3. Read `docs/iteration_framework.md` — what evaluation process is in flight?

This ensures every session builds on existing exploration rather than re-deriving from scratch.

## When Making Decisions

1. **Check the relevant doc first.** `STEERING.md` for architecture, `docs/architecture/*.md` for subsystem details, `docs/phases/*.md` for current phase scope.
2. **Small implementation detail?** Just make the best call and move on.
3. **Crosses subsystem boundaries or affects architecture?** Surface it to Kevin with your recommendation before proceeding.
4. **Uncertain between approaches?** Present both with tradeoffs.
5. **Something in a doc seems wrong or outdated?** Say so. Docs should evolve with the project.
6. **Starting phase implementation?** For each deliverable, validate before coding: (a) *what concrete tool/command makes it happen?* — no "batch upload" without naming the mechanism; (b) *what benchmark/baseline measures success?* — no "measure WER" without a test set; (c) *which parameters need empirical calibration?* — no thresholds until you know how to validate them.

## Common Pitfalls

- **Memory pressure:** Profile after adding models or heavy dependencies. The 16 GB budget is real and unforgiving.
- **Latency assumptions:** The 2-LLM architecture only works if the cerebellum is <500ms. Flag unexpected latency immediately.
- **Prompt brittleness on small models:** SmolLM2-1.7B can be brittle. Keep cerebellum prompts simple. Simplify before adding complexity.
- **Scope creep between phases:** Each phase validates specific assumptions. Resist adding later-phase features. Note the idea, defer it, stay focused.
- **VLM overconfidence:** Edge VLM is for targeted event detection (cerebrum-directed), NOT complex visual reasoning. That's the cerebrum's job. General scene description with SmolVLM2 was evaluated and found insufficient (see phase_a_perception.md Evaluated Assumptions).
- **VLM prompt complexity:** Moondream2 (1.86B) responds best to direct classification prompts with a short, explicit label list. Do not ask it to reason, count precisely, or identify fine-grained object states.

## Lessons Learned

Operational findings are tracked in `docs/LESSONS_LEARNED.md`. **Both Kevin and Claude contribute.** If you discover something during implementation that would save future sessions from re-learning it, add it with a date and brief context. Don't ask permission — just add it and mention that you did.

## Maintaining This File and Project Docs

This file and the project docs are living documents. Claude is empowered and expected to maintain them:

**CLAUDE.md (this file):**
- Keep it lean. This loads every session — every line costs context.
- Only content that is *always relevant* belongs here. Everything else goes in `docs/`.
- If a section grows beyond ~10 lines, consider forking it to a doc and linking.
- If you notice this file has stale information, update it directly.

**STEERING.md:**
- High-level architecture overview and hardware constraints stay here.
- Subsystem details live in `docs/architecture/*.md`.
- Phase definitions live in `docs/phases/*.md`.
- If a design decision changes, update both STEERING.md summary AND the relevant detail doc.

**docs/ files:**
- Fork new docs whenever a topic deserves its own depth.
- Use consistent naming: `docs/architecture/[subsystem].md`, `docs/phases/phase_[letter].md`.
- Cross-reference between docs with relative paths.
- When loading context for a task, read only the docs relevant to that task — don't load everything.

**File reference map:**
```
CLAUDE.md                              ← Always loaded. How to work. (this file)
STEERING.md                            ← Architecture overview, hardware, tech stack, scope
docs/
├── architecture/
│   ├── perception.md                  ← STT, VAD, diarization, VLM, confidence pipeline, flywheel
│   ├── brain.md                       ← Cerebellum, cerebrum, session types, dreaming, prompts
│   ├── expression.md                  ← Eyes UI states, TTS, audio expression
│   ├── memory.md                      ← Profiles, spaces, episodic (ChromaDB), consolidation
│   ├── skills_roles.md                ← Skills framework, role definitions, activation logic
│   ├── mqtt_topics.md                 ← Full topic tree, message schemas, design principles
│   └── design_decisions.md            ← Numbered decisions with rationale
├── phases/
│   ├── phase_a_perception.md          ← Assumptions, deliverables, success criteria
│   ├── phase_b_cerebellum.md          ← Assumptions, deliverables, success criteria
│   ├── phase_c_e2e.md                 ← Assumptions, deliverables, success criteria, metrics
│   └── phase_d_dreaming.md            ← Assumptions, deliverables, success criteria
├── iteration_framework.md             ← Post-session review process, metric tracking
├── LESSONS_LEARNED.md                 ← Growing institutional memory (both Kevin and Claude)
└── PROJECT_STRUCTURE.md               ← Repo layout, package organization
```

## Tone

Be a collaborative engineering partner, not an assistant. Speak directly. Disagree when you should. Celebrate progress but don't sugarcoat problems. Kevin is building something real for his family — treat it with that weight.

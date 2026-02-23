# STEERING.md — Sous Chef: Personalized Household AI Assistant

> **Version:** 0.4.0
> **Last updated:** 2026-02-23
> **Authors:** Kevin (project lead), Claude (HRI research collaborator)
> **Repo:** /Users/Kevin/Documents/code/winston

---

## 1. Project Vision

Build a personalized household AI assistant ("Sous Chef") with two clear objectives:

1. **Help complete household chores more efficiently** — reduce time, reduce errors, handle unforeseen problems
2. **Encourage family connection** — create moments of bonding through shared activity, not just task completion

The agent becomes more effective over time by learning household norms, personal preferences, and engagement styles of each family member.

### Design Philosophy

- **Speed of iteration over perfection.** Prototype fast, measure, learn, improve.
- **Trust is earned through competence and character.** The agent must be both useful and pleasant to interact with.
- **Latency kills trust.** Immediate acknowledgment of user input is non-negotiable, even if the full response takes time.
- **Honest about limitations.** The agent should never fake understanding or empathy it doesn't have.
- **The interaction IS the product.** Hardware and models are means — the quality of human-agent collaboration is what we're evaluating.

---

## 2. System Architecture Overview

Three subsystems working together, connected by MQTT:

```
┌─────────────────────────────────────────────────────────────────┐
│                        SOUS CHEF AGENT                          │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   PERCEPTION     │  │  CHARACTER BRAIN  │  │  EXPRESSION  │  │
│  │   UNDERSTANDING  │──│                   │──│  ENGINE      │  │
│  │                  │  │  cerebellum(edge) │  │              │  │
│  │  Whisper Small   │  │  SmolLM2-1.7B    │  │  Eyes (LCD)  │  │
│  │  SmolVLM2-500M   │  │                   │  │  TTS (cloud) │  │
│  │  Silero VAD      │  │  cerebrum (cloud)  │  │  Speaker     │  │
│  │  Speaker Embed.  │  │  Sonnet 4.6       │  │              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    MQTT MESSAGE BUS                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    MEMORY SYSTEM                          │   │
│  │  Structured (profiles, spaces) + Episodic (ChromaDB)      │   │
│  │  + Memory Consolidation ("Dreaming")                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    SKILLS / ROLES                         │   │
│  │  skills/*.md (capabilities) + roles/*.md (behaviors)      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Subsystem detail docs:**
- Perception: `docs/architecture/perception.md`
- Character Brain: `docs/architecture/brain.md`
- Expression Engine: `docs/architecture/expression.md`
- Memory System: `docs/architecture/memory.md`
- Skills & Roles: `docs/architecture/skills_roles.md`
- MQTT Topics (API contract): `docs/architecture/mqtt_topics.md`
- Design Decisions & Rationale: `docs/architecture/design_decisions.md`

---

## 3. Hardware Constraints

| Resource | Spec |
|----------|------|
| Device | MacBook Pro (M2 Pro, 16 GB unified memory) |
| Microphone | Built-in MacBook microphone |
| Camera | Built-in MacBook camera (720p/1080p) |
| Speaker | Built-in MacBook speakers |
| Display | MacBook screen (eyes UI + debug overlay) |

**Memory Budget:**
| Component | Estimated RAM |
|-----------|---------------|
| Whisper Small.en (inference) | ~2 GB |
| Speaker embedding model | ~100-200 MB |
| SmolVLM2-500M (inference) | ~1.2-1.8 GB |
| SmolLM2-1.7B (inference, INT4) | ~3-4 GB |
| Silero VAD | negligible |
| ChromaDB + app overhead | ~1-2 GB |
| OS + system processes | ~3-4 GB |
| **Total estimated** | **~11-13 GB** |
| **Headroom** | **~3-5 GB** |

> **⚠️ MEMORY IS TIGHT. MONITOR ACTIVELY.**
>
> **Monitoring:** `system/debug/memory_monitor` publishes per-model usage every 30s. Alert at <2 GB headroom. Integration testing must include 30+ min sustained load.
>
> **Mitigation levers (priority order):**
> 1. Reduce SmolVLM2 image resolution (N=2 instead of N=4)
> 2. Quantize SmolLM2 more aggressively (INT4 → GGUF Q4_K_M)
> 3. Lazy-load VLM (load on demand, unload between snapshots)
> 4. Reduce Whisper size (Small → Base.en, trades ~1% WER for ~1 GB)

---

## 4. Development Phases

Focused phases, each validating key assumptions. Sequential — later phases depend on earlier learnings.

| Phase | Goal | Detail Doc |
|-------|------|------------|
| **A: Perception & Fine-Tuning** | Validate edge models + cloud flywheel | `docs/phases/phase_a_perception.md` |
| **B: Cerebellum & Expression** | Validate 2-LLM responsiveness + eyes UI | `docs/phases/phase_b_cerebellum.md` |
| **C: First E2E (Cooking)** | Full cooking assistant prototype | `docs/phases/phase_c_e2e.md` |
| **D: Dreaming Deep Dive** | Validate memory consolidation with real data | `docs/phases/phase_d_dreaming.md` |

---

## 5. Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.11+ | Ecosystem compatibility (HF, MLX, MQTT) |
| Edge inference | MLX (Apple Silicon) | Native M2 Pro optimization |
| STT | whisper.cpp via MLX or HF Transformers | Fast local inference |
| VLM | SmolVLM2-500M via HF Transformers | Periodic scene snapshots |
| Small LLM | SmolLM2-1.7B-Instruct via MLX | Fast cerebellum reactions |
| Large LLM | Claude Sonnet 4.6 (Anthropic API) | Cerebrum reasoning |
| TTS | ElevenLabs API | High-quality voice synthesis |
| VAD | Silero VAD (PyTorch) | Lightweight, accurate |
| Speaker ID | pyannote/embedding or resemblyzer | Speaker diarization |
| Message bus | Mosquitto (MQTT broker) + paho-mqtt | Lightweight pub/sub |
| Episodic memory | ChromaDB (local) | Simple vector store, no server |
| Structured data | JSON files (local) | Human-readable profiles, spaces |
| Eyes UI | Web-based (HTML/CSS/JS) or PyGame | Animated eye display |
| Audio I/O | PyAudio or sounddevice | Mic capture, speaker output |

---

## 6. What This Project Is NOT (Current Scope)

- Not a mobile robot — stationary MacBook only
- Not a general-purpose assistant — cooking role only (initially)
- Not emotionally intelligent — practical patience, not empathy
- Not privacy-hardened — local data storage, cloud API calls, no encryption
- Not production-ready — research prototype with manual setup
- Not multi-room — single room (kitchen) only
- Not handling deictic gestures — no "this" / "that" pointing resolution

---

## 7. Future Roadmap (Informational Only)

Future work may explore: deictic gesture resolution, additional roles (homework, cleanup, game night), emotional attunement, mobile embodiment, multi-room awareness, deep personality modeling, privacy hardening, multi-user simultaneous interaction, touch/haptic input, smart home integration, sound event detection.

# Project Structure

> **Parent:** `CLAUDE.md`

```
winston/
├── CLAUDE.md                              ← Permanent context for Claude Code
├── STEERING.md                            ← Architecture overview (source of truth)
├── README.md                              ← Project overview and setup instructions
├── pyproject.toml                         ← Project dependencies
├── config/                                ← Configuration files (YAML)
├── prompts/                               ← LLM prompt files
│   ├── persona.md
│   ├── mental_model.md
│   └── roles/
│       └── sous_chef.md
├── skills/                                ← Agent capability definitions
├── data/                                  ← Local data storage
│   ├── profiles/                          ← Family member profiles (JSON)
│   ├── spaces/                            ← Spatial data (JSON)
│   ├── episodes/                          ← Episodic memory (ChromaDB)
│   └── collection/                        ← Raw data for fine-tuning flywheel
├── src/
│   ├── perception/                        ← STT, VAD, diarization, VLM scene
│   ├── brain/                             ← Cerebellum (edge) + Cerebrum (cloud)
│   ├── expression/                        ← Eyes UI + TTS
│   ├── memory/                            ← Profiles, spaces, episodic, dreaming
│   ├── skills/                            ← Skill execution runtime
│   ├── transport/                         ← MQTT client wrappers, topic definitions
│   └── debug/                             ← Debug overlay, memory monitor, logging
├── tests/
├── scripts/                               ← Setup, enrollment, fine-tuning utilities
└── docs/                                  ← Architecture docs, phase plans, lessons
    ├── architecture/
    │   ├── perception.md
    │   ├── brain.md
    │   ├── expression.md
    │   ├── memory.md
    │   ├── skills_roles.md
    │   ├── mqtt_topics.md
    │   └── design_decisions.md
    ├── phases/
    │   ├── phase_a_perception.md
    │   ├── phase_b_cerebellum.md
    │   ├── phase_c_e2e.md
    │   └── phase_d_dreaming.md
    ├── iteration_framework.md
    ├── LESSONS_LEARNED.md
    └── PROJECT_STRUCTURE.md (this file)
```

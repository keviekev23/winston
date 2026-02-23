# MQTT Communication Backbone

> **Parent:** `STEERING.md` Section 2
> **Broker:** Mosquitto (local, lightweight)

## Topic Structure

```
souschef/
├── perception/
│   ├── speech/
│   │   ├── transcript         # {text, confidence, speaker_id, speaker_confidence, timestamp}
│   │   ├── sentiment          # {sentiment, confidence, speaker_id, timestamp}
│   │   ├── vad                # {is_speaking, timestamp}
│   │   └── diarization        # {speaker_id, confidence, enrollment_match, timestamp}
│   ├── scene/
│   │   ├── snapshot           # {description, objects[], activity, confidence, timestamp}
│   │   └── change             # {change_type, description, affected_objects[], timestamp}
│   └── sound/
│       ├── timer              # {source, confidence, timestamp} (future)
│       ├── alarm              # {source, confidence, timestamp} (future)
│       └── event              # {event_type, confidence, timestamp} (future)
│
├── brain/
│   ├── cerebellum/
│   │   └── reaction           # {expression_state, filler_text?, timestamp}
│   ├── cerebrum/
│   │   ├── response           # {text, tts_request?, timestamp}
│   │   ├── plan/
│   │   │   ├── created        # {plan_id, goal, steps[], timestamp}
│   │   │   ├── step_update    # {plan_id, step_id, status, detail, timestamp}
│   │   │   ├── replan         # {plan_id, reason, new_steps[], timestamp}
│   │   │   └── completed      # {plan_id, outcome, summary, timestamp}
│   │   ├── clarification      # {query, target_perception_id, timestamp}
│   │   └── delegation         # {target, task, context, timestamp}
│   └── session/
│       ├── start              # {session_type, metadata, timestamp}
│       ├── end                # {session_type, trigger_reason, timestamp}
│       └── summary            # {session_type, summary_text, metadata, timestamp}
│
├── expression/
│   ├── eyes/
│   │   └── state              # {state, transition_speed, timestamp}
│   ├── tts/
│   │   ├── speak              # {text, voice_id, priority, timestamp}
│   │   └── status             # {is_speaking, timestamp}
│   └── sound/
│       └── effect             # {sound_id, timestamp} (chirps, alerts)
│
├── memory/
│   ├── profile/
│   │   ├── update             # {member, field, value, source_session, timestamp}
│   │   └── query              # {member?, field?, timestamp}
│   ├── spaces/
│   │   ├── object_update      # {space_id, object_id, field, value, timestamp}
│   │   └── query              # {space_id?, object_id?, timestamp}
│   ├── episode/
│   │   ├── save               # {session_summary, metadata, timestamp}
│   │   ├── query              # {query_text, filters?, timestamp}
│   │   └── result             # {results[], timestamp}
│   └── dreaming/
│       ├── trigger            # {session_ids[], timestamp}
│       ├── status             # {phase, progress, timestamp}
│       └── updates            # {update_type, changes[], timestamp}
│
└── system/
    ├── health                 # {subsystem, status, memory_usage_mb, timestamp}
    ├── config                 # Runtime configuration changes
    └── debug/
        └── memory_monitor     # {total_mb, per_model_mb{}, headroom_mb, timestamp}
```

## Design Principles

- `perception/{modality}/{event_type}` — extensible to new modalities and event types
- `brain/cerebrum/plan/*` — cerebrum is the orchestrator; plan lifecycle is explicit (created → step_update → replan? → completed)
- `brain/cerebrum/delegation` — cerebrum can delegate subtasks to other components (future: to local models)
- `brain/session/*` — session lifecycle is explicit, decoupled from plan lifecycle
- `memory/dreaming/*` — consolidation is observable; other subsystems can subscribe to updates

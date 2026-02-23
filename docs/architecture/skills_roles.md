# Skills & Roles Framework

> **Parent:** `STEERING.md` Section 2
> **Related:** `docs/architecture/brain.md`

## Skills (Capabilities)

```
skills/
├── web_search.md        # Search recipes, substitutions, cooking techniques
├── timer_management.md  # Set, track, alert on multiple concurrent timers
├── recipe_parsing.md    # Parse recipe from URL or text into structured steps
└── smart_home.md        # (Stub for later: oven preheat, etc.)
```

## Roles (Behavioral Modes)

```
roles/
├── sous_chef.md         # Cooking assistant role — step guidance, proactive help
└── (additional roles added in later phases)
```

## Role Activation

Role activation is explicit: the brain enters a role when context triggers it (user starts cooking → sous_chef role activates). Roles modify:
- Tone
- Skill usage priorities
- Interaction patterns
- Proactivity level

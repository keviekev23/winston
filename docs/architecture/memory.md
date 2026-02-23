# Memory System

> **Parent:** `STEERING.md` Section 2
> **Related:** `docs/architecture/brain.md` (dreaming spec), `docs/architecture/mqtt_topics.md`

## Structured Profiles (JSON, local)

```json
{
  "family_id": "chen_household",
  "members": [
    {
      "name": "Kevin",
      "role": "parent",
      "dietary_restrictions": [],
      "known_preferences": {"garlic": "loves it", "spice_level": "medium"},
      "engagement_style_notes": "sarcastic humor, direct communicator"
    }
  ],
  "household_rules": []
}
```

## Spaces (JSON, local)

Spaces represent areas of the house and the spatial relationships of objects within them. Schema will be iterated during Phase A based on what the VLM can actually detect.

```json
{
  "spaces": [
    {
      "id": "kitchen",
      "label": "Kitchen",
      "objects": [
        {
          "id": "stove",
          "label": "Gas Stove (4 burner)",
          "location": "counter_south",
          "state": "off",
          "children": [
            {"id": "burner_front_left", "label": "Front Left Burner", "state": "off"},
            {"id": "burner_front_right", "label": "Front Right Burner", "state": "off"},
            {"id": "burner_back_left", "label": "Back Left Burner", "state": "off"},
            {"id": "burner_back_right", "label": "Back Right Burner", "state": "off"}
          ]
        },
        {
          "id": "oven",
          "label": "Oven",
          "location": "under_stove",
          "state": "off",
          "properties": {"current_temp": null, "target_temp": null}
        },
        {
          "id": "cutting_board_area",
          "label": "Cutting Board Area",
          "location": "counter_west",
          "state": "clear",
          "children": []
        },
        {
          "id": "sink",
          "label": "Kitchen Sink",
          "location": "counter_north",
          "state": "empty"
        },
        {
          "id": "fridge",
          "label": "Refrigerator",
          "location": "wall_east",
          "inventory": []
        }
      ]
    }
  ]
}
```

The VLM scene snapshots update object states. The cerebrum reasons about spatial relationships when planning. Spaces schema will be iterated during Phase A.

## Episodic Memory (ChromaDB, local)

- Each cooking episode and each conversation session → one episode document
- Cloud brain (Sonnet) generates session summaries (see Session Types in `docs/architecture/brain.md`)
- Stored with metadata: date, session_type (cooking|conversation), participants, outcome, notable events
- Semantic retrieval via ChromaDB for cross-session context

## Memory Consolidation ("Dreaming")

Full spec in `docs/architecture/brain.md`. Summary: async post-session batch process that updates structured profiles, spaces, and role performance knowledge from episodic memories.

## Deferred to Later Phases

- Deep personality modeling and inference chains
- Knowledge graph / structured relationship reasoning
- Conflict resolution patterns
- Long-term preference drift tracking

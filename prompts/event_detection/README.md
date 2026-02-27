# Event Detection Prompts

Scenario YAML files for Phase A VLM evaluation and Phase C runtime event detection.

These are **valuable evaluation artifacts** — never delete. The `evaluations` list in each YAML is a longitudinal record of what worked and what didn't.

---

## Concept: Scenarios vs. Steps

Each YAML represents a **monitoring scenario** (an attention mode), not a cooking step.
A scenario groups events that co-occur during a phase of cooking:

| Scenario | Events it watches for |
|----------|----------------------|
| `cooking_prep.yaml` | Cutting, washing, counter idle |
| `active_cooking.yaml` *(future)* | Stove events, stirring, boiling |
| `kitchen_safety.yaml` *(future)* | Unattended stove, spills |

In Phase C, cerebrum selects the appropriate scenario based on current recipe step context.
The `step_type → scenario_yaml` mapping is a Phase C deliverable — not assumed here.

---

## File Format

```yaml
scenario: unique_scenario_id          # snake_case, matches filename stem
version: 1                            # bump when event descriptions change
description: "What this scenario watches for"
notes: "Iteration notes, design decisions"

events:
  - id: snake_case_event_id           # used in filenames and verified.json
    label: UPPER_CASE_LABEL           # what VLM is asked to output (all caps)
    description: "Concrete, visually observable description for VLM prompt"
    confirm_frames: 3                 # consecutive detections required to trigger
    notes: "Why this confirm_frames value? What false positive patterns?"

# Auto-generated prompt (constructed at runtime by detect_event.py):
# Never stored in this file — it is built from the events list above.

evaluations: []                       # DO NOT edit manually — populated by evaluate_vlm.py
```

---

## Guidelines for Writing Event Descriptions

**Do:**
- Describe what is *visually observable* in a single frame
- Use concrete, specific language: "cutting on a cutting board" not "preparing food"
- Keep descriptions to one sentence

**Don't:**
- Describe abstract states that require reasoning ("person looks tired")
- Use time-dependent descriptions ("has been cooking for 5 minutes")
- Make descriptions so specific that minor variations fail ("dicing carrots into 1cm cubes")

**Prompt iteration:**
1. Bump `version` when changing a description; add a comment with the old text
2. Add a note explaining *why* you changed it ("v1 caused false positives on stirring")
3. Re-run `detect_event.py` + `label_scene_data.py --manual` + `evaluate_vlm.py --label after_v2`
4. The `evaluations` list will show the before/after delta

---

## Confirm Frames Calibration

`confirm_frames` is set empirically — not assumed. Starting points:

| Event type | Suggested starting value | Reasoning |
|-----------|-------------------------|-----------|
| Instantaneous (stove ignition) | 1-2 | Happens in a single frame |
| Short action (picking up knife) | 2-3 | 2-3s at 1fps |
| Sustained action (chopping) | 3-5 | Needs a few seconds to "settle" visually |
| State (person standing idle) | 2-3 | Not instantaneous but not long |

Phase C research question: should cerebrum emit recommended `confirm_frames` per event
dynamically, or are thresholds baked into scenario YAMLs? (See phase_c_e2e.md)

---

## Adding a New Scenario

1. Copy `cooking_prep.yaml` as a template
2. Update `scenario`, `version`, `description`, `events`
3. Set `evaluations: []`
4. Run `detect_event.py --scenario your_new_scenario.yaml` to collect samples
5. Verify with `label_scene_data.py --manual`
6. Evaluate with `evaluate_vlm.py --label baseline`

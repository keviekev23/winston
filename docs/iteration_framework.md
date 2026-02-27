# Iteration & Learning Framework

> **Parent:** `STEERING.md`

## After Each Cooking Session

1. **Debrief:** Quick self-report from user (5 min questionnaire)
2. **Log review:** Examine MQTT logs for failure points, latency spikes, misunderstandings
3. **Memory check:** Verify episodic memory storage and retrieval accuracy
4. **Prompt iteration:** Update persona.md, mental_model.md, or sous_chef.md based on findings
5. **Model check:** Review low-confidence perception samples, assess if fine-tuning cycle is warranted

## After Every 3-5 Sessions

1. **Metric review:** Plot trends in outcome metrics
2. **Fine-tuning cycle:** If sufficient data, run cloud QLoRA fine-tuning for STT/VLM
3. **Architecture review:** Any subsystem changes needed?
4. **Dreaming review (Phase D+):** Evaluate mental model accuracy and consolidation quality

---

## Documenting Evaluated Assumptions

When an assumption in a phase doc is tested and the outcome changes the plan, document it.
**Claude is empowered to write these entries without asking permission** — record findings proactively.

### Format (add to the relevant phase doc's "Evaluated Assumptions" section):

```markdown
### [Assumption Name] — [DATE]

**Original assumption:** What was believed going in.
**What we did:** Brief description of the evaluation (scripts used, sample size, approach).
**Outcome:** What was found. Be specific — numbers where available.
**Impact:** How this changed the plan (paradigm shifts, deliverable changes, deferred items).
**Relevant data:** Links to eval files, scripts, or data directories.
```

### Rules:
- After every evaluation cycle, add an entry to the relevant phase doc
- document build progress when each implementation cycle completes in the current phase.  if the entire phase is completed, write a quick summarized assessment based on success criteria before moving onto the next phase.  
- The goal: a new agent or session reading the phase doc cold understands what has already been tried, either when phase just started or picking up in progress. 

### Where to add entries:
- Phase docs (`docs/phases/phase_[x].md`) — for phase-level assumption changes
- Architecture docs (`docs/architecture/*.md`) — for subsystem-level findings
- `docs/LESSONS_LEARNED.md` — for operational findings (latency surprises, model behaviors, etc.)

---

## VLM Prompt Iteration Cycle (Phase A)

For iterating event detection prompts during Phase A evaluation:

```
1. Edit event description in scenario YAML — bump version, add note
2. python scripts/detect_event.py --scenario prompts/event_detection/{scenario}.yaml
3. python scripts/label_scene_data.py --manual
4. python scripts/evaluate_vlm.py --label after_v{N}
5. python scripts/evaluate_vlm.py --compare evals/before.json evals/after.json
6. Update evaluations[] in YAML (done automatically by evaluate_vlm.py)
```

If evaluation process steps change, update the cheatsheet in `docs/phases/phase_a_perception.md` first.

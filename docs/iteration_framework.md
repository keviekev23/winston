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

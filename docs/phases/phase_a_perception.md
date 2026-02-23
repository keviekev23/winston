# Phase A: Perception & Fine-Tuning Validation

> **Parent:** `STEERING.md` Section 4
> **Subsystem docs:** `docs/architecture/perception.md`, `docs/architecture/mqtt_topics.md`

## Goal

Validate that edge models can perceive the kitchen usefully, and that the cloud fine-tuning flywheel works end-to-end.

## Key Assumptions Being Tested

- Whisper Small.en can transcribe kitchen speech (noisy environment, distance from mic) at acceptable accuracy
- Speaker embedding matching can reliably identify enrolled family members in normal turn-taking
- SmolVLM2-500M can produce useful scene descriptions from a MacBook camera angle
- The confidence flagging pipeline correctly identifies low-quality outputs
- Cloud labeling (Whisper Large-v3, GPT-4o/Claude) produces reliable pseudo-labels
- QLoRA fine-tuning measurably improves edge model accuracy on user-specific data

## Deliverables

- [ ] MQTT backbone + system health + memory monitoring operational
- [ ] VAD + Whisper STT pipeline running, transcribing live speech with confidence scores
- [ ] Voice enrollment flow: each family member records sample sentences → speaker embedding stored
- [ ] Speaker diarization: post-transcription embedding match → `speaker_id` + confidence on each utterance
- [ ] SmolVLM2-500M scene snapshot pipeline running, producing structured scene descriptions
- [ ] Scene schema (spaces/objects JSON) drafted and iterated based on what VLM actually detects
- [ ] Data collection daemon: logging all inference I/O with confidence scores (per-speaker for audio)
- [ ] Cloud labeling pipeline: batch upload → multi-model consensus labeling → filtered dataset
- [ ] One complete QLoRA fine-tuning cycle for Whisper (record before/after WER)
- [ ] One complete QLoRA fine-tuning cycle for SmolVLM2 (record before/after scene accuracy)
- [ ] Memory profiling report: actual RAM usage per model under sustained load
- [ ] Debug overlay: real-time MQTT message viewer alongside eyes UI

## Success Criteria

- STT latency < 2s from end of utterance to transcript available
- Speaker identification accuracy > 80% in normal turn-taking (non-overlapping speech)
- Scene snapshot processing < 3s per frame
- At least one fine-tuning cycle shows measurable improvement on user-specific test set
- Memory usage stable (no leaks) over 30+ minute sustained operation
- Clear understanding of what the VLM can/cannot detect → informs scene schema design

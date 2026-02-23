# Phase B: Cerebellum & Expression Validation

> **Parent:** `STEERING.md` Section 4
> **Subsystem docs:** `docs/architecture/brain.md`, `docs/architecture/expression.md`

## Goal

Validate that the 2-LLM architecture produces responsive, natural-feeling interactions. Test that the cerebellum + eyes create the sensation of an attentive agent.

## Key Assumptions Being Tested

- SmolLM2-1.7B can map perception signals to expression states in < 500ms
- The eyes UI creates a convincing sense of agent "presence"
- Filler behavior (fillers when cloud is slow) feels natural, not annoying
- The overall interaction loop (speak → agent acknowledges → agent responds) feels good

## Deliverables

- [ ] SmolLM2-1.7B loaded alongside perception models (validate memory budget holds)
- [ ] Cerebellum receives perception signals → emits expression state on MQTT
- [ ] Eyes UI rendering all 8 expression states with smooth transitions
- [ ] Cerebellum generates filler text/sounds when simulated cloud delay > 3s
- [ ] End-to-end latency measurement: speech end → expression state change (target: < 500ms)
- [ ] Qualitative evaluation: 3+ test interactions, self-report on naturalness

## Success Criteria

- Speech → expression reaction in < 500ms consistently
- All 8 eye states visually distinguishable and contextually appropriate
- Memory budget holds with 3 models loaded simultaneously
- Subjective: interactions feel responsive, not robotic or laggy

# Phase D: Memory Consolidation ("Dreaming") Deep Dive

> **Parent:** `STEERING.md` Section 4
> **Subsystem docs:** `docs/architecture/brain.md`, `docs/architecture/memory.md`
> **Requires:** Real interaction data from Phase C

## Goal

Validate that async memory consolidation meaningfully improves the agent's mental model and task performance over time.

## Key Assumptions Being Tested

- Batch processing of episodic summaries can extract useful structured knowledge
- Mental model updates from dreaming improve the agent's predictions and suggestions
- Role performance reflection leads to measurable task improvement
- Memory pruning/summarization maintains retrieval quality as episodes accumulate

## Deliverables

- [ ] Dreaming pipeline operational: triggered post-session, processes episodic memory
- [ ] Mental model updates: preferences, corrections, household rules extracted and persisted
- [ ] Role performance analysis: patterns identified from cooking session outcomes
- [ ] Memory pruning: older episodes compressed, retrieval quality maintained
- [ ] Cross-session pattern detection: recurring themes surfaced as persistent entries
- [ ] Synthetic data exploration: simulated session histories to stress-test consolidation at scale
- [ ] A/B comparison: 5+ sessions with dreaming vs. Phase C baseline (without dreaming)

## Success Criteria

- Mental model accuracy improves after dreaming (measured by preference prediction spot-checks)
- Agent behavior observably changes based on dreaming outputs
- Retrieval quality doesn't degrade as episode count grows (test at 10, 25, 50 episodes)
- Dreaming completes within reasonable time (< 5 min per session)

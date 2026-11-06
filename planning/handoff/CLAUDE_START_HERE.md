# SNDL Starting Context for Claude

We are preparing a research showcase in a few days: approximately ten minutes for a technical audience unfamiliar with our specific topic and possibly AI. Please help us evaluate the current research state and choose the next experiments. We need a critical technical collaborator, not a marketing rewrite.

Read `SNDL_RESEARCH_HANDOFF_2026-11-04.md` first, then `CONTEXT.json`. These names refer to files in the supplied packet. If you cannot access a file, say so and distinguish the report's description from evidence you have actually inspected.

Our current position is that a constrained integrated Engine prototype works, but effective real-data continual learning, the benefit of growth, and recursive self-improvement have not been demonstrated.

Two evidence streams must remain separate:

- Fixed-width SHD audio component: joint macro accuracy 41.16%; sequential 6.22%; repeat 7.21%; replay 12.68%, over four already-exposed development seeds. The official test set was untouched. Replay improved final accuracy over matched repeat, but the registered forgetting metric was worse. Joint training was only two passes.
- Full persisted Brain: a task-conditioned growth/retention/restart cycle passed on four unique inputs per task, repeated during evaluation. Success required corrective feedback and a frozen old core while grown outputs learned. The all-plastic control learned B but forgot A completely. This was not SHD, unseen-input generalization, or a capacity-matched growth comparison.

Please produce the following, in this order:

1. Audit the main conclusions against the supplied records. Identify unsupported claims, methodological confounds, measurement errors, and important missing evidence. Separate findings from hypotheses.
2. Give a short independent assessment of what contribution is defensible now and whether the work should be framed primarily as an engineering prototype, a continual-learning method, or an architecture proposal.
3. Rank the plausible causes of weak SHD performance as hypotheses, with the cheapest discriminating test for each. Inspect the existing learner/preprocessing/analysis code if available. Do not infer a cause from low accuracy alone.
4. Propose the smallest useful next experiment, with fixed sample IDs/splits, controls, metrics, a justified prospective decision criterion, an initial profiling step, and a bounded compute budget suitable for an M2 Mac with 8 GB RAM. Start by testing basic learnability rather than assuming a larger Engine campaign is the right next move.
5. Define a later fair growth comparison: growth-on/off, equal-capacity pre-sized model, protected versus plastic core, task-context access, replay access, feedback, search cost, training exposures, and genuinely held-out examples.
6. Recommend an honest ten-minute showcase story and explain what additional evidence would change that recommendation.

Treat a dataset/model/runtime change as a new development configuration; retain the original results. Do not cherry-pick seeds/checkpoints or tune on the official test set. A repeated exposed seed is not independent confirmation. Passing a tiny engineering fixture is not a benchmark result. Do not invent variables, an RSI score, calibrated probabilities, or measurements.

The source commit is local, `26501ed3` on `engine-cycle-integration`, based on `c4c10b285f1741aa798d5ee9c5296ae371fabecf`. The packet includes the source snapshot, incremental Git bundle, patch, and experiment records; a GitHub clone alone may not contain these commits or local evidence. Consult `PACKET_README.md` for access and recovery details.

Before executing or changing Brain code, read the repository instructions and environment notes. Use a new isolated experimental home and new run IDs; preserve the original lake and final-consumption records. Branch names must not use a `codex` prefix. Do not edit the presentation, publish, merge, or push merely because you have reviewed this packet; first return the assessment and proposed next experiment for the team to choose its direction.

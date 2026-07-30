# Coverage evidence -- Memory

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| episodic_retrieval_to_reply | 40 | src/snn_data_lake/cortex/brain.py:3016; src/snn_data_lake/cortex/brain.py:3031 | - | memory-auditor (opus-5) / live respond() instrumentation | 2026-07-30 |
| swr_value_consolidation | 40 | src/snn_data_lake/cortex/brain.py:2322; src/snn_data_lake/cortex/brain.py:1778 | - | memory-auditor (opus-5) / untracked buffer-neuter A/B | 2026-07-30 |
| affective_somatic_retrieval | 40 | src/snn_data_lake/cortex/brain.py:3183; src/snn_data_lake/cortex/brain.py:2972 | - | memory-auditor (opus-5) / live respond() instrumentation | 2026-07-30 |
| episodic_encode | 30 | src/snn_data_lake/cortex/brain.py:1688 | - | memory-auditor (opus-5) / live-tick instrumentation | 2026-07-30 |
| episodic_retrieval_to_action | 30 | src/snn_data_lake/cortex/brain.py:1448; src/snn_data_lake/cortex/brain.py:1464 | recall_300.txt | memory-auditor (opus-5) / tools/charge_probe.py recall 300 | 2026-07-30 |
| working_memory_store | 30 | src/snn_data_lake/brainstem/autonomous_loop.py:1042; src/snn_data_lake/cortex/live_brain.py:238 | - | memory-auditor (opus-5) / live-tick instrumentation | 2026-07-30 |
| attractor_sleep_replay | 30 | src/snn_data_lake/cortex/brain.py:2313; src/snn_data_lake/cortex/brain.py:2318 | - | memory-auditor (opus-5) / live-tick instrumentation, 6 nights | 2026-07-30 |
| durable_episode_consolidation | 25 | src/snn_data_lake/cortex/brain.py:2293; src/snn_data_lake/hippocampus/episode_consolidation.py:332 | - | memory-auditor (opus-5) / live-tick instrumentation, 6 nights | 2026-07-30 |
| semantic_store_durable_lake | 25 | src/snn_data_lake/substrate/home.py:82 | - | memory-auditor (opus-5) / static read + read-only record count | 2026-07-30 |
| chat_memory_store_and_recall | 25 | src/snn_data_lake/substrate/chat_memory.py:208; src/snn_data_lake/reasoning/spike_chat_lake.py:1011 | - | memory-auditor (opus-5) / static read (call chain traced) | 2026-07-30 |
| synaptic_consolidation_benna_fusi | 25 | src/snn_data_lake/cortex/brain.py:2304; src/snn_data_lake/plasticity/plasticity_rule_stack.py:217 | - | memory-auditor (opus-5) / live-tick instrumentation, taught vs untaught | 2026-07-30 |
| use_it_or_lose_it_forgetting | 20 | src/snn_data_lake/reasoning/spike_chat_lake.py:1029; src/snn_data_lake/cortex/brain.py:2473 | - | memory-auditor (opus-5) / static read + live-tick instrumentation | 2026-07-30 |
| semantic_schema_induction | 15 | src/snn_data_lake/cortex/brain.py:2430 | - | memory-auditor (opus-5) / live-tick instrumentation, 6 nights | 2026-07-30 |
| affective_content_retrieval | 10 | src/snn_data_lake/hippocampus/affective_memory.py:151 | - | memory-auditor (opus-5) / whole-tree caller grep | 2026-07-30 |
| contrastive_episode_analysis | 10 | src/snn_data_lake/brainstem/autonomous_loop.py:1563; src/snn_data_lake/cortex/live_brain.py:79 | - | memory-auditor (opus-5) / live-tick instrumentation | 2026-07-30 |
| chat_memory_flush_noop | 0 | src/snn_data_lake/substrate/chat_memory.py:239 | - | memory-auditor (opus-5) / static read | 2026-07-30 |

**Mean over the 16 rows: 24.7.** No faculty is above 40. Sixteen rows, zero UNSCORED.

## What the number is and is not

The withdrawn first-pass file scored Memory **20.0** on **three** rows. 24.7 is **not a capability
gain and is not comparable to it** -- the two roster are different sizes, which is the exact mistake
`docs/coverage/README.md:59-62` warns about. Of the prior three rows, one (`recall_drive_source`)
survives as `episodic_retrieval_to_action`, one (`chat_memory_write_only`) is **refuted** below, and
one (`chat_memory_uncalled_method`) was a backward-compat no-op counted as a faculty and is now
scored **0** rather than 15. The movement is a wider roster measured properly, in both directions:
three faculties the prior roster never enumerated are runtime-verified at the instrument ceiling,
and four new precondition-never-fires findings score 10-25.

## The ceiling, and why nothing here is above 40

`tools/charge_probe.py` is the only tracked neuter-A/B in the repository and it can only neuter a
key of `Brain._drive_w`. Memory has exactly one such key, `recall`. Every other memory faculty --
a store that reaches a reply, a consolidation that rewrites a value table -- has **no tracked
prober**, and the scorer refuses band > 40 without a machine-recorded artifact (F6/F8).

I could have recorded an artifact whose `command:` pointed at a scratchpad script. I did not.
That artifact survives `score` and dies on `--rerun`, which `tools/coverage_score.py:37-41` names
as the forger's move, and untracked local probes are the specific defect `tools/charge_probe.py`
was created to fix. So three faculties whose evidence supports band 61-80 on the rubric's own
definitions are recorded at **40**, deliberately under-scored, with the measurement written down so
a later auditor with a tracked prober can raise them on evidence instead of re-litigating.

**The single highest-value fix to the instrument** is a non-drive-source neuter mode in
`tools/charge_probe.py` (neuter an arbitrary named subsystem, observe `arm.q` / the reply text as
well as `arm_stopped` / `map_position`). I was scoped out of `tools/**` and did not write it.

## Harness control

Before scoring anything I re-ran both existing artifacts through the sanctioned path:

- `--rerun drive_60` -> recorded exit 0 / now 0, observables `(('0','60'), ('22','60'))` identical.
  22/60 trajectory changes, the figure `COVERAGE_BASELINE.md:90` cites. The observable is not degenerate.
- `--rerun recall_60` -> recorded exit 1 / now 1, `(('0','60'), ('0','60'))` identical.

## The one recorded A/B: episodic recall does not move the arm

`artifacts/recall_300.txt`, exit 1. Pre-committed before running: 300 ticks, one shot, and I score
21-40 unless the flip rate is materially comparable to the `drive` control's 0.367.

Result at 5x the previous horizon: **33** learned `(recall, action)` pairs (so the source is
ELIGIBLE -- it fired, repeatedly), and still **0/300** act-hold flips and **0/300** trajectory
changes. The source weight *fell*, 0.1 -> 0.0837, which is the competitive `-mean_e` term steadily
down-weighting a source that never earns dopamine. `recall_60` was not too short a run; this signal
is consumed and inert. It is scored 30 rather than 20 because `elig["recall"]`
(`brain.py:1464`) enters the real drive arithmetic and its value table genuinely learns -- more
than telemetry -- but the exit code caps it at 40 and the measurement puts it nowhere near that.

## The strongest memory faculty in this brain was never audited

`swr_value_consolidation` (`brain.py:1778` add, `brain.py:2322` replay). The prior sweep produced
one Memory finding across eight lenses and this was not it.

Measured, same seed, 60 wake + 40 night + 60 wake, A wired and B with a buffer that records nothing
so the sleep-side `replay()` runs over an empty ring:

- `buffer.replay` is called **40 times** in one night and applies **1280** value updates
- the grown-unit action-value table `arm.q` **differs** between A and B after the night
- on the following day the arm trajectory diverges on **55/60** ticks (act/hold 0/60, matching
  `drive_60`'s 0.000 HOLD rate in both arms)

The neuter is clean: `grep` finds exactly three touch points for `self.buffer` in `brain.py`
(construction, `add`, `replay`), no `len()` read, and the buffer owns a private `random.Random`, so
there is no telemetry or RNG leak between arms. Memory consolidation demonstrably changes a
learning update and, one day later, the action. That is band 61-80 language. It is filed at 40.

## Four preconditions that do not fire (rubric rule 4)

1. **Synaptic (Benna-Fusi) consolidation relaxes nothing on a brain nobody talks to.**
   `brain.py:2304` loops all 7 `_ixn` neurons every sleep tick; `_plastic_relaxed` was **0 on all
   six nights**. Cause traced: `_BennaFusiConsolidation.relax` returns `None` when the neuron's
   cascade was never created (`plasticity_rule_stack.py:217`), and cascades are created only inside
   `consolidate()`, which runs only on an R-STDP deposit. Decisive test: 0/7 cascades after 20 wake
   ticks; **3/7 cascades and `_plastic_relaxed == 3`** after two `LiveBrain.teach()` calls. So this
   is live on the chat-carrying daemon and a rule-4 zero on the arm-only loop -- including inside
   `charge_probe`'s own brain. `brain.py:231` claims the stack composes "Benna-Fusi consolidation
   ... onto the live learner"; that is true only after the brain has been taught.

2. **"Sleep replay consolidates the real episodes (no forgetting)" (`brain.py:2291`) is
   contradicted by its own metric.** Over six nights the durable store grew 8, 16, 24, 32, 40, 48
   assemblies and `retained()` read 1.000, 0.500, 0.333, 0.250, 0.200, 0.167 -- the recoverable
   **count is pinned at exactly 8** while the store grows, against a nominal `capacity=64`. The
   fraction is not a stuck constant, it is `8/n`. Consistent with this, the `(durable: ...)`
   annotation at `brain.py:3040` fired on **0 of 2** live recall queries in my probe.

3. **Contrastive episode analysis never constructs.** `contrastive` defaults False
   (`live_brain.py:79`), `Brain.__init__` does not pass it, and it additionally needs
   `reward_fn is not None` (`autonomous_loop.py:1563`). Measured on a canonical brain:
   `arm._contrast_on=False`, `arm._contrast=None`. The only `contrastive=True` in the tree is
   `live_brain.py`'s own self-check.

4. **`AffectiveEpisodicStore.recall_similar` has no caller outside its own `__main__`.** Whole-tree
   grep: definition at `affective_memory.py:151`, then only lines 227 and 261 of the same file's
   self-check. Content-addressed affective retrieval is built and not live. Note the *sibling*
   method is a different story -- see the refutations.

Two lesser observations, not scored separately: `AttractorNet.downscale_prune` reported
`pruned: 0 / 552` on every night of every run, and `_consolidation_acc` sat at 0.3333 (one dip to
0.2778) across six nights -- roughly 1 of its 3 stored assemblies completing. It is folded into the
self-model at `brain.py:2360`, which the band table calls an internal read; I did not measure
whether that fold changes anything, so `attractor_sleep_replay` stays at 30.

## Refutations of prior claims -- both directions

- **REFUTED: "chat memory is write-only."** `DurableChatMemory.recall` is called at
  `spike_chat_lake.py:1011` and its hit is appended as a competing reply candidate above score
  0.34, and a hit also bumps the `RecallTrace` that decides what survives nightly pruning
  (`spike_chat_lake.py:1029`). Both are gated on `persist=True`, which `live_brain.py:452` passes
  through and the daemon supplies. The row is *still* only 25, because the evidence is static: I
  traced the call chain but did not execute it (see below).
- **CONFIRMED, narrowly:** the lead I was given said `affective_memory.recall_similar` has zero
  external callers. True. But it does not generalise to affective memory: the sibling
  `recall_by_emotion` (`affective_memory.py:161`) is called every `respond()` via `_somatic_recall`
  (`brain.py:3183`), the store held 31 episodes after 60 ticks, and the somatic-marker note
  appeared in **2 of 2** live replies. One dormant method, one charged method, same file.
- **CONFIRMED: SpikeLake is gone.** No runtime constructor; nothing in this roster depends on it.

## What I could not measure, and why

`semantic_store_durable_lake` and `chat_memory_store_and_recall` are **static-only** and scored
low for exactly that reason (rule 1). Both need `persist=True`, and a `persist=True` brain runs
schema induction on night 4 (`brain.py:2430`, which **appends** to the lake) and lake consolidation
on night 8 (`brain.py:2473`, which **rewrites and prunes** it). The production lake at
`~/.snn_data_lake/brain_lake.bin` holds **19,829** `_T_FACT` records of the user's real learned
knowledge. An auditor is not entitled to prune it to score a row, so I read the count read-only and
stopped. I considered `UNSCORED` for both; a low scored band with the evidence type named is more
informative and counts the same against the mean as an honest 25 should.

`semantic_schema_induction` and `use_it_or_lose_it_forgetting` are gated on the same `persist` flag
plus `nights % 4` / `nights % 8`; on my runs `_schemas` stayed 0 and `_lake_kept`/`_lake_pruned`
stayed `None`, i.e. they never executed. They are scored 15 and 20 as built-and-conditionally-live,
not as verified.

## Capability inventory (Policy 2: what provides what)

- `hippocampus/episodic_recall.py` -- reconstructive episodic store; `encode` + cue-addressed
  `complete`. Sole provider of pattern-completed episodic recall on the tick. 48 episodes / 6 days.
- `hippocampus/episode_consolidation.py` -- the *durable* sparse-attractor episode store with
  covariance Hebbian storage, nightly `replay`, `retained`, and `recall_episode` (readable episode
  dict). Overlaps `episodic_recall` in role but not in mechanism; both are live and consumed.
- `hippocampus/experience_replay.py` -- SWR forward+reverse value consolidation into `arm.q`. Sole
  provider of memory-to-learning-update. Also imported by `plasticity/lifelong_consolidation.py`,
  `substrate/lake_memory.py`, `cortex/quality_diversity.py`, `cortex/reasoning_curriculum.py`.
- `hippocampus/affective_memory.py` -- affect-tagged episodic retention; `recall_by_emotion`
  (charged) and `recall_similar` (dormant). Sole provider of somatic-marker retrieval.
- `hippocampus/lake_consolidation.py` -- `RecallTrace` use-decayed utility + prune policy for the
  symbolic lake. Sole provider of retrieval-driven forgetting.
- `hippocampus/sleep_replay.py` -- `AttractorNet` noise-driven assembly consolidation + SHY pruning.
- `substrate/home.py` -- `KnowledgeLog`, the ONE durable semantic store (`brain_lake.bin`, `_T_FACT`).
- `substrate/chat_memory.py` -- noise-filtered content-addressable view over that same lake.
- `substrate/lake_working_memory.py` -- the working-memory ring in the spike lake (`arm.lake_wm`).
- `substrate/lake_memory.py` -- a documented drop-in superset of `LakeWorkingMemory` adding prune +
  consolidate. **EXEMPT "permanent"** in `dormancy_gate` as a reasoning-tool primitive, while the
  module it was built to replace is the one on the tick. Flagged as a Policy-2 question for whoever
  owns that decision, not scored here.
- `hippocampus/sleep_consolidation.py` -- `SleepConsolidator` (wake -> quiescent consolidation).
  Imported by `cortex/self_improvement_suite.py`, `reasoning/lake_skill_library.py` and
  `reasoning/brain_adapters/hippocampal_memory.py`, **not** by `cortex/brain.py`, which reimplements
  the wake/sleep consolidation inline. Also a Policy-2 question; not a separate roster row, because
  the *faculty* it provides is already counted by the three consolidation rows.
- `hippocampus/contrastive_episode_analysis.py` -- success-vs-failure context pooling; not enabled.

## Gaps ranked by distance_to_100 x feasibility

1. **Give `swr_value_consolidation` a tracked prober.** It already measures as a 61-80 faculty; the
   only thing between it and that band is a probe file in `tools/`. Highest value per unit of work
   in this whole category, and it needs no new capability.
2. **Fix the pinned recoverable-8 in `episode_consolidation`.** Retention decaying as `8/n` while
   the store grows to 48 is the difference between a durable episodic memory and a one-day buffer,
   and it is why the `(durable: ...)` reply annotation never fires. Diagnosis first: 8 recoverable
   out of a nominal capacity of 64 points at the k-WTA/threshold pair or assembly-overlap
   interference, not at replay coverage.
3. **Make episodic recall decisive, or admit it is not a drive.** 33 learned pairs and 0/300 flips
   over 300 ticks says the recall eligibility never wins the competition. Either the gate
   (`_recall_conf`, max 0.438) is too weak to tip a three-way choice, or the source belongs
   elsewhere. Wiring more sources into `_route_percept_to_action` will not fix a signal that is
   already there and already loses.
4. **Initialise the Benna-Fusi cascades from the tick, not only from a teach deposit.** A one-line
   equilibrium `_ensure` on the first sleep tick would make synaptic consolidation live on the
   arm-only brain, where it is currently a measured zero.
5. **Runtime-verify the semantic and chat lakes against a throwaway home.** `NORD_HOME`-style
   redirection (or a temp-dir home) would let an auditor exercise the persist-gated half of Memory
   without touching 19,829 real records. Four of my sixteen rows are static-only for lack of it.

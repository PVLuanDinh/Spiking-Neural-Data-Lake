# Coverage evidence -- Imagination

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| latent_world_model | 61 | src/snn_data_lake/cortex/brain.py:200;src/snn_data_lake/brainstem/autonomous_loop.py:1314;src/snn_data_lake/brainstem/autonomous_loop.py:1619 | imag_jepa_action_60.txt | opus-5 REFUTER / docs/coverage/artifacts/imagination_refute_ab.py 60 | 2026-07-30 |
| counterfactual_do_operator | 40 | src/snn_data_lake/cortex/brain.py:1956;src/snn_data_lake/cortex/brain.py:1972;src/snn_data_lake/cortex/brain.py:3117 | imag_cf_60.txt | opus-5 / docs/coverage/artifacts/imagination_ab.py counterfactual 60 | 2026-07-30 |
| model_based_plan_rollout | 35 | src/snn_data_lake/brainstem/autonomous_loop.py:981;src/snn_data_lake/brainstem/autonomous_loop.py:1018 | imag_plan_60.txt | opus-5 / docs/coverage/artifacts/imagination_ab.py model_based_plan 60 | 2026-07-30 |
| mental_imagery | 20 | src/snn_data_lake/cortex/brain.py:3477;src/snn_data_lake/cortex/brain.py:2885 | - | opus-5 REFUTER / band-definition correction on the auditor's own live cue-sensitivity probe | 2026-07-30 |
| dream_replay | 25 | src/snn_data_lake/cortex/brain.py:2313;src/snn_data_lake/hippocampus/sleep_replay.py:294 | imag_dream_60.txt | opus-5 / docs/coverage/artifacts/imagination_ab.py dream_replay 60 | 2026-07-30 |
| spiking_world_model | 10 | src/snn_data_lake/substrate/dormancy_gate.py:146;src/snn_data_lake/cerebellum/spiking_jepa.py:101 | - | dormancy_gate EXEMPT permanent + its recorded encode() negative (static) | 2026-07-30 |
| efference_copy_forward_model | 10 | src/snn_data_lake/cortex/live_brain.py:329;src/snn_data_lake/cerebellum/forward_model.py:81 | - | opus-5 / live_brain._planner is None after 20 ticks, seeds 0 and 3 | 2026-07-30 |
| hierarchical_abstract_world_model | 0 | src/snn_data_lake/brainstem/autonomous_loop.py:118;src/snn_data_lake/brainstem/autonomous_loop.py:945;src/snn_data_lake/brainstem/autonomous_loop.py:2163 | - | opus-5 / arm.hjepa_serve is None on make_brain(seed=0) | 2026-07-30 |
| generative_recombination | 0 | src/snn_data_lake/cortex/self_improvement_suite.py:213;src/snn_data_lake/substrate/dormancy_gate.py:107 | - | opus-5 / caller search over src: no live-path generator | 2026-07-30 |

**Mean over the nine faculties: 201/9 = 22.3.** Nine rows, none UNSCORED, one above 40.

**This file has been through the rubric's adversarial pass (`COVERAGE_RUBRIC.md:93-95`) and the score
came DOWN, from 23.9 to 22.3.** Two cells moved: `latent_world_model` 65 -> **61** and
`mental_imagery` 30 -> **20**. The single band-41+ claim was attacked on every line the rubric asks
for and it **held as a band and fell as a number**: the 61-80 grade is now supported by better evidence
than the auditor had, and the four points above the band floor were not. The full attack, including the
one finding that argues for taking it below 41 and the reason that reading was not taken, is in "The
adversarial pass" below. The sections after it are the original auditor's, edited only where a number
changed or where a claim of theirs was measured, because a refutation that rewrites the case it is
attacking is not checkable.

This is a **downgrade** from the withdrawn `39` at `COVERAGE_BASELINE.md:30`, and rubric rule 5 says
downgrades are results. The 39 was recorded "not verified (<=40)" with no per-faculty table and no
artifact, while `COVERAGE_BASELINE.md:48` reports the auditor had claimed **two** faculties at 68 on a
single A/B each. Neither of those A/Bs exists in the tree. What follows is the first Imagination
evidence in this repository that a reader can re-execute.

## The adversarial pass: what was attacked, what fell, what held

New evidence: `docs/coverage/artifacts/imagination_refute_ab.py`, recorded as
`imag_jepa_action_60.txt` (exit 0). It re-runs the auditor's neuter and adds the observable their probe
does not have.

**Forgery check: passed, twice.** `--rerun imag_jepa_60` reproduces at HEAD 8b6aff77 (exit 0,
`(0/60, 47/60, 56/60)`), and the control `--rerun drive_60` reproduces its documented
`(0/60, 22/60)` exit 0. The refuter's own arm A independently reproduces **47/60 and 56/60 exactly**,
which is the check that its recording wrapper is behaviour-preserving. Nothing here is hand-written.

**What fell: the band position, on the meaning of the two large numbers.** `imagination_ab.py` records
three observables. One is per-tick -- `arm_stopped` -- and it took **one distinct value** on the wired
run, so it is degenerate and its `0/60` says nothing. The other two, `map_position` and the
value-table checksum, are **cumulative state**: once two runs differ they keep differing. Measured
first-divergence ticks are 14 for the trajectory and 5 for the value table, and `60 - 13 = 47`,
`60 - 4 = 56`. **Both headline numbers are the divergence-onset count, which is the trap
`Executive.md:110-116` records for its own `22/60`.** So band 65 was awarded without a single working
per-tick decision observable.

**And the numbers are not even faculty-specific.** `exec_conflict_60.txt` neuters
`ExecutiveControl.conflict` -- the ACC conflict scalar, nothing to do with a world model -- and reports
`act/hold 0/60, trajectory 47/60, selected action 23/60, first divergence tick 13`. The refuter's
jepa neuter reports `act/hold 0/60, trajectory 47/60, selected action 23/60, first divergence tick 14`
(same tick; the two probes index from 0 and from 1). **Two unrelated signals, one byte-identical
triple.** That triple is a property of when this arm loop at seed 0 can first fork, not a measure of
how much either faculty matters -- and had these two rows sat in the same category file,
`coverage_score.py`'s W3 check would have flagged them as one serial chain scored twice.

**What held: the band itself, 61-80, on a real decision observable.** Pinning
`JepaWorldModel.prediction_error` to 0.0 changes the **selected action on 23/60 ticks** (first at tick
14, arm A ranging over 7 distinct commands) and the **greedy policy of the action-value table -- the
argmax per grown unit -- on 53/60 ticks**. That is a decision and a learning update, measured on
per-tick, non-cumulative observables. The rubric's 61-80 test is met, and a refuter who pushed this row
below 41 would be ignoring a measured action change, which is the same error in the other direction.
So the row is **61, the floor of its range** -- the position `Executive.md:27` already assigns to this
exact grade of evidence ("a recorded neuter-A/B whose exit code says a decision changed"), and the four
points above the floor were being carried by a harness constant.

**The auditor's own untested caveat, now measured, and it goes against them.** They wrote that a
stricter neuter "would inject a shuffled error of the same magnitude" and predicted the verdict would
hold. Arm C does exactly that -- arm A's own prediction errors replayed in a deterministically
shuffled order, so the multiset of magnitudes is identical and only the pairing between error and
transition is destroyed. Result: **selected action 0/60, trajectory 0/60**, value table 56/60, greedy
policy 45/60. So the world model's *predictive content* -- which error belongs to which transition --
is **behaviourally inert at n=60**. What charges the arm's actions is that the intrinsic-reward channel
has magnitude at all, not that the magnitude is a correct forward-model error. The band survives
because the rubric's test is mechanical and the signal is the model's output; but "the brain's actions
depend on its world model's predictions" is **not** what this row measures, and a reader who requires a
faculty's content rather than its scale to be decisive should score this 41-60. That reading was not
taken, and it is the strongest surviving case against the row.

A second thing arm C exposes, worth more than the band: the greedy policy diverged on 45/60 ticks while
the executed action diverged on **0/60**. The learned value table's preferred action is therefore not
what selects the action in this regime, so "the value table changed" does not license "behaviour
changed" here. The 23/60 in arm B is doing all the work; the 56/60 value-table figure is not
independent evidence.

**Shared credit: checked across the whole ledger, and DISCHARGED.** The auditor flagged that if Affect
or Perception also scored curiosity, one mechanism would be funding two rows -- the defect that cost
Executive 9 points. A search of all eleven category files for `curiosity|intrinsic|jepa|world_model|
prediction_error|empower` returns no faculty row in any other file; the only other hit is a prose
mention in `WholeBrain.md:68`. `prediction_error` also has exactly **one** call site in `src/`
(`autonomous_loop.py:1314`), so the neuter is genuinely surgical and the empowerment drive at `:1316`
is untouched by it. No double count.

**Roster padding: checked, and it runs the other way.** The rubric names four things Imagination
scores; this file splits "forward models" into six rows. That is a real granularity choice, but it is
not the inflating kind: the four rows an auditor could have omitted (`spiking_world_model` 10,
`efference_copy_forward_model` 10, `hierarchical_abstract_world_model` 0, and this correction's
`mental_imagery` 20) sit far below the category mean, so the expanded roster **costs** about 9 points
rather than buying them. The `W3` warning on the three all-zero negatives is confirmed a false
positive: the three pins are independent (`arm._planned_seq`, `Brain._cfsim`, `attractor.sleep`) and
each artifact shows its own signal was live on the wired arm (56/60, 57/60, 29/60).

**What was left standing, deliberately.** `model_based_plan_rollout` at 35 was checked against
Executive's `multi_step_plan_sequence` at 30 -- the same beam search, the same call site
(`autonomous_loop.py:981`), the same 0/60 result, pinned one step apart on one serial chain. Lowering
it to 30 for cross-file consistency was considered and **rejected as point-shaving**: the plan's output
does reach the bias vector (`:1018`, and `self.jepa.predict` really is its transition function at
`:986`, so it is a world-model rollout and not merely Executive's planner re-rowed), and Executive
measured that neutering it changes `_control_gain` on 33/60 ticks, which is an internal read. The upper
end of 21-40 is defensible. `counterfactual_do_operator` at 40 was also checked and stands: its reply
change is measured, and the F8 cap the auditor describes is real, not a judgment they made.

**`mental_imagery` 30 -> 20, a band-definition correction.** The auditor scored this 30 while writing
that "a refuter could reasonably argue for 20", so the claim was flagged as not compelled. Band 21-40
is defined as "**on the wake/sleep tick**, but its output reaches only telemetry", and `Brain.imagine()`
is provably never on the tick -- it fires only when a chat line begins `imagine`/`picture`/`visualize`
(`brain.py:2883-2885`). This ledger's own precedent already treats that path as outside the live loop:
`Executive.md:160-162` downgrades a faculty because "the abstain gate at `brain.py:2949` is inside
`respond()`, which is not on the tick at all". With no tick-side evidence of any kind, the honest cell
is the top of 1-20: built, reachable through a real entrypoint, and not part of the brain's own
cognition. Stated against myself: the cockpit chat *is* a live path, so a reader who counts it scores
30, and neither band definition covers "live only when a human types a prefix" cleanly. This is the
most arguable cell in the file and it is worth 1.1 points of the category.

## The instrument could not reach this category, so I built the probe

`tools/charge_probe.py` neuters a key of `Brain._drive_w`. **Not one Imagination signal is in that
dict.** The eleven drive sources are `novelty / ignition / place / timing / reroute / help / percept /
recall / surprise / drive / taskset` (`cortex/brain.py:633-648`), and the tempting one is a false
friend: `surprise` there is the Oja-PCA reconstruction residual over the percept latent, set in
`_perceive_real` at `cortex/brain.py:1196-1206`, not a forward-model rollout error. So every band in
this file would have been capped at 40 by refusal F6 for want of an artifact rather than for want of
charge -- which is the failure mode the rubric exists to prevent, running in reverse.

I wrote `docs/coverage/artifacts/imagination_ab.py`: same construction path (`make_brain(seed=0,
persist=False)`, Policy 4), same pinning discipline as `charge_probe._PinnedWeights` (a property, so a
value assigned inside the tick is swallowed and cannot leak back), same exit-code contract. It adds one
observable charge_probe lacks -- a checksum of the arm's action-value table -- because rubric band 61-80
is "changes a decision, action, **or learning update**" and a world model can be decisive through what
the brain learns without moving it on the same tick. **It belongs in `tools/`**; it is here because that
was the write scope of this audit and because tracked, re-runnable evidence today beats correct filing
later. Moving it costs one edit to the `command:` line of the four artifacts that cite it.

**Control, so nobody has to take the harness on trust.** `python tools/coverage_score.py --rerun
drive_60` reproduces at this HEAD: exit 0, observables `(('0','60'),('22','60'))` -- the same 22/60
trajectory changes `COVERAGE_BASELINE.md:90` cites. `--rerun imag_jepa_60` also reproduces exactly. The
`map_position` observable took 32 distinct values on arm A of every run, so the zeros reported below are
zeros against a live observable, not against a constant.

## The one charge, and what it is really crediting

*(Auditor's section, as written. The band it argues for is 65; the adversarial pass above lowered it to
61 and its numbered claims are annotated below where the refuter measured them.)*

`latent_world_model` is the only faculty here the instrument accepts above 40. Neutering
`JepaWorldModel.prediction_error` to a constant 0.0 -- the model's *predictive content*, leaving
encode/predict/train_step alive -- changed the arm's **action-value table on 56/60 ticks and its
trajectory on 47/60** (`imag_jepa_60.txt`, exit 0). The route is `autonomous_loop.py:1314`: the
prediction error for the just-lived transition becomes the curiosity drive's learning-progress signal,
which becomes intrinsic reward `r_int` (`:1355`), which enters the value update. That is a learning
update and an action, so band 61-80. It is **not** 81+: there is no held-out evidence for it, here or
anywhere in this category.

Three caveats a refuter should press on, none of which I think overturns the band:

- **Shared credit.** The consumer is `SpikingCuriosityDrive`. If Affect or Perception also scores
  curiosity as one of its faculties, the same neuter supports both rows and the two auditors are
  splitting one mechanism. My defence is that the signal I pinned is the *forward model's error*, and
  without a forward model there is no such signal at all -- but this is the exact shape of the defect
  that cost Executive 9 points in the withdrawn baseline, so it deserves the cross-check.
  **REFUTER: checked and discharged.** No other category file rows curiosity, empowerment or the world
  model as a faculty, and `prediction_error` has one call site in `src/`. No double count.
- **The neuter is not perfectly surgical.** Pinning the error to 0 also changes the curiosity drive's
  own running learning-progress statistics. A stricter version would inject a *shuffled* error of the
  same magnitude. I expect the verdict to hold (56/60 is not a marginal divergence) but the number
  would move.
  **REFUTER: measured, and the prediction was wrong in substance.** The magnitude-matched shuffle
  (`imag_jepa_action_60.txt`, arm C) changes the value table on 56/60 ticks exactly as predicted -- and
  the selected action and the trajectory on **0/60**. The verdict holds only for the signal's presence,
  not for its content.
- **The rollout is not what is charged.** The path that *looks* like imagination -- rolling the world
  model forward to choose an action -- is `model_based_plan_rollout`, and it measured zero. What
  charges is the model's error, not its prospection.
  **REFUTER: confirmed, and it is narrower still.** What charges is the model's error *scale*. Not its
  prospection, and not its accuracy either.

**A P-violation on record, which is not a band but bears on the claim.** `train_step` propagates the
invariance error into the encoder by using the forward predictor matrix transposed --
`gz[k] = gp[k] + sum(gp[i] * self.Wz[i][k])` at `cerebellum/jepa_world_model.py:272-278`, with the
comment at `:267` stating outright that this is "backprop into We/be". That is weight transport, and it
runs on the live tick, in a project whose standing rule is to pivot any backprop consideration to e-prop
and whose P1 sibling `cortex/counterfactual_sim.py` goes to some length in its docstring to justify a
purely local rule. The specified fix is feedback alignment: a fixed random `Bz` in place of `Wz^T`. The
faculty still earns its band -- the band asks whether the output changes a learning update, and it does
-- but "the brain learns its world model with a local rule" is not true of this learner, and the
coverage number should not be read as endorsing that.

**REFUTER: this claim is verified, not taken on trust.** `jepa_world_model.py:272-278` computes
`gz[k] = gp[k] + sum(gp[i] * self.Wz[i][k])` inside `if not frozen:`, and the comment at `:267` reads
"VICReg + backprop into We/be". The transported matrix is the live forward predictor's own `Wz`. It runs
on the live tick, because `train_step` is called every tick from `autonomous_loop.py:1619`. The
P-violation stands as written.

## Three honest negatives, each with its number

- **`model_based_plan_rollout` -- CONSUMED-BUT-NOT-DECISIVE (exit 1).** The beam search at
  `autonomous_loop.py:981-991` searches action *sequences* over `jepa.predict` and the winner is added to
  the top-down bias at `:1018`. It is genuinely live: a non-empty plan on 56/60 ticks (measured
  `['ENGAGE', 'RELEASE', 'JOINT_A_ROTATE(-6deg)']` at tick 60), and the plan-following synapse `_plan_w`
  learned from 0.1 to 0.1465 over the run, so dopamine was reinforcing plan-following. Pin the plan to
  empty and **0/60 act-hold decisions, 0/60 trajectories and 0/60 value-table states differ.** A synapse
  that moves while the outcome does not is this repo's documented trap ("movement is not activation"),
  and here it is in its purest form: the brain is learning how much to trust a plan that changes nothing.
- **`counterfactual_do_operator` -- CONSUMED-BUT-NOT-DECISIVE (exit 1), but it changed a reply.** The
  do()-operator runs every wake tick (`brain.py:1956`), trains its latent forward model on the observed
  transition (`:1972`, and the weights verifiably move), and produced 4 distinct top actions and 17
  distinct imagined-surprise values over 40 ticks -- it is not a constant wearing a signal's clothes.
  Behaviourally it is inert by construction (`state()` at `:2747` says "annotation, NEVER commanded") and
  the A/B confirms 0/60 on all three behavioural observables. But the **secondary read did change**: with
  the simulator pinned to None the narrator's answer to "what if you engaged the arm right now?" loses
  its "imagining it: rolling a learned forward model over my options ... I'd lean 'X'" clause
  (`brain.py:3109-3120`) and keeps only the somatic marker. Rubric band 41-60 is "changes an internal
  read -- **a reply**, a memory write, a self-model term", so by the rubric's own wording this faculty
  earns 41-60 on measured evidence. It is scored 40 because F8 caps any band above 40 to a probe that
  exited 0, and `charge_probe`'s exit contract has no verdict for "changed a reply, changed no action".
  **That is a real hole in the instrument**, not a judgment call: a genuine band-41 signal is currently
  indistinguishable from telemetry. Closing it needs a third exit code, which is a rubric-level change,
  so it is flagged rather than quietly worked around.
  One substantive limitation regardless of band: the value function the rollout ranks against is
  `lambda z: sum(z)` -- the code's own comment at `brain.py:1957-1958` calls it a placeholder. The
  do()-operator is complete; the objective it optimises is a stand-in with no goal semantics, so "the
  best imagined action" currently means "the one with the most total imagined activation".
- **`dream_replay` -- CONSUMED-BUT-NOT-DECISIVE (exit 1), and weaker than I expected.** The noise-driven
  attractor replay at `brain.py:2313` runs with `learn=True` on every one of the 29 sleep ticks in a
  60-tick run, and it really does rewrite weights: 29-30 distinct `AttractorNet.W` checksums over 30
  forced nights. Replace `sleep()` with a no-op and 0/60 behavioural observables differ -- expected,
  since the attractor is read only by `accuracy()` -- but **its own readout `consolidation_acc` is also
  unchanged**, so over 29 nights at seed 0 the dream contributed nothing even to the one scalar it feeds
  into the self-model (`brain.py:2318`, `:2360`). At seed 0 that scalar took 2 distinct values over 30
  nights (0.3333, 0.4444); at seed 3 it took 8. The A/B says the seed-0 movement is not caused by
  dreaming. The companion telemetry `_shy` (offline downscale-prune) reported `pruned: 0` of 552
  synapses at seed 0 and `pruned: 3` at seed 3 -- effectively a floor.

## Two zeros under rule 4, stated as zeros

- **`hierarchical_abstract_world_model` (HJepaServe).** This is the *highest-priority* planning path in
  the loop (`autonomous_loop.py:945-960`: accumulate a K-tick Level-0 window, encode the abstract
  Level-1 latent, plan over it). Its guard is `self.hjepa_serve is not None`, and that is false on every
  live tick: the parameter defaults to None at `:118` and `live_brain.py:67`, `Brain.__init__` never
  passes it (`brain.py:197-219`), `learn_daemon.py` contains no reference to jepa or hjepa at all, and
  `:2163` asserts the default is None. Measured: `arm.hjepa_serve is None` on `make_brain(seed=0)`.
  Rubric rule 4 -- "a precondition that never fires scores 0, however good the code is" -- and the code
  is good: it is a complete, self-checked, GPU-trained-and-served two-level model. Zero is a statement
  about what runs, not about the module's quality, and this is the single cheapest capability gap in the
  category (see below).
- **`generative_recombination`.** The rubric names this as one of the four things Imagination scores, and
  I could not find a live implementation. What exists: `cortex/neural_sampling.py` is EXEMPT permanent
  because "the live tick has no such [energy] model and no query asking for samples from one"
  (`dormancy_gate.py:107-111`); `cortex/quality_diversity.py`'s archive is constructed only inside
  `self_improvement_suite.py:213`, and the live brain imports only `census` from that module
  (`brain.py:389`); `cortex/vsa_reasoning.py`'s `bind`/`bundle` *are* live, but their four consumers
  (`access_consciousness`, `crossmodal_binder`, `recursive_tom`, `reasoning_curriculum`) use them to
  compose a representation, not to synthesise novel content -- that is binding, and it belongs to
  Perception, Consciousness and Social. The nearest thing to recombination on a live path is the
  noise-driven attractor dream already scored above. **Honest null: nothing implements this faculty as a
  faculty.** Zero rather than UNSCORED, because "I looked and there is no mechanism" is a measurement.

## What I could not measure, and where a refuter should aim

- **No held-out evidence exists for any Imagination faculty**, so 81-100 is unreachable for the whole
  category by rule 3 regardless of wiring. The world model is trained and consumed entirely on the live
  self-directed stream; there is no unrehearsed distribution and no metric with a baseline it beat.
- **`mental_imagery` has no attributable neuter.** `Brain.imagine()` (`brain.py:3477`) is real and I
  verified it live: it returns a 64-dimensional synthesized percept and it is cue-sensitive (`cube` and
  `sphere` gave different vectors), and the reply quotes it. But it fires only when a chat line begins
  with `imagine` / `picture` / `visualize` (`brain.py:2883-2885`) -- never on an autonomous tick -- and
  the only way to neuter it is to disable the predictive-coding front end `_pc`, which is a Perception
  faculty, so any divergence would be unattributable. 30 is a judgment: live via a real entrypoint,
  output reaches a reply, no A/B. A refuter could reasonably argue for 20 on the grounds that a faculty
  which never runs unless a human asks for it is not part of the brain's own cognition.
  **REFUTER: that argument was taken; the row is now 20.** Reasoning and the counter-argument are in
  "The adversarial pass" above. The measurement is unchanged -- only the band cell moved.
- **`efference_copy_forward_model` scored 10 on runtime evidence, not a grep.** `cerebellum/forward_model
  .ForwardModel` is imported inside `live_brain.deliberate_ahead` (`:329`) and is its default predictor,
  which is enough to satisfy the dormancy gate -- the gate reads the import, not the construction.
  `deliberate_ahead` has no live-tick caller: `plan_latent` passes the JEPA as predictor, and the only
  other callers are that module's own self-check at `:622-623`. Measured: `live_brain._planner` was still
  None after 20 live ticks at seeds 0 and 3, so the class is never instantiated on the tick. The live
  loop deliberately uses *proprioceptive reafference* instead, and says so at
  `autonomous_loop.py:512-518`. So the efference-copy mechanism the docstring of
  `cortex/agency_monitor.py:27-33` refers to as running is not running.
- **The `W3` warning this file will trigger is a false positive.** `coverage_score` fingerprints an
  artifact by its `N/M ticks` lines and warns when two faculties share one, on the theory that they are
  two probe points on one serial chain. `imag_plan_60`, `imag_cf_60` and `imag_dream_60` all fingerprint
  as `(('0','60'),('0','60'),('0','60'))` because all three are all-zero negatives on independent
  signals with independent pins. The heuristic cannot distinguish "one chain" from "three separate
  nulls" -- worth noting as a hole, since a category full of honest negatives now looks like a category
  that double-counted.
  **REFUTER: confirmed a false positive, and the heuristic has the mirror-image hole too.** The three
  pins are independent and each artifact records its own signal live on the wired arm. Meanwhile the two
  charges that genuinely DO share a fingerprint -- this file's `latent_world_model` and Executive's
  `acc_conflict_monitoring_evc`, both `(0/60, 47/60, 23/60)` with the same onset tick -- are invisible
  to W3 because it only compares within one category file. False positives inside a category, false
  negatives across them.
  **W3 ADJUDICATOR, 2026-07-30: FALSE POSITIVE CONFIRMED a third time, on the full stdout, and all
  three bands stand.** The check the two notes above assert was performed: the three artifacts were
  compared line by line rather than by fingerprint. STATIC -- `imagination_ab.py:98-103` pins three
  different objects on three different modules: `arm._planned_seq = []`, `b._cfsim = None`, and
  `b.attractor.sleep` replaced by a no-op. RUNTIME, already in the recorded artifacts and never in the
  fingerprint -- `ticks the signal was live (A)` reads **56/60** for `model_based_plan`, **57/60** for
  `counterfactual` and **29/60** for `dream_replay` (the last being the sleep-tick count), and
  `secondary read changed by the signal` reads **True** for `counterfactual` and **False** for the other
  two. Three different liveness counts and a split secondary read cannot come from one signal probed at
  three points. What the three artifacts *do* share -- `distinct A values act / traj / value-table :
  1 / 32 / 58` -- is the un-neutered arm A's own baseline, identical because it is the same seed-0 arm
  in all three runs, so it describes the instrument and not the faculties. And no band is inflated for a
  mechanical reason as well: all three rows are at or below 40 (40, 35, 25), F8 only demands an exit-0
  artifact above 40, and all three artifacts are exit-1 negatives -- the one row in this file that does
  rest on an exit-0 artifact, `latent_world_model`, is not in the W3 group at all. The collision is the
  regex: `_fingerprint` matches only `(\d+)/(\d+) ticks`, and every discriminating line above either
  lacks the trailing `ticks` or is not a ratio.
- **The CROSS-CATEGORY collision this file flagged against `Executive.md` is also a FALSE POSITIVE, and
  it was adjudicated on a four-arm probe rather than on a read.** New evidence:
  `docs/coverage/artifacts/crosscat_jepa_vs_conflict_ab.py`, recorded as
  `crosscat_jepa_vs_conflict_60.txt` (exit 0). It runs A wired, B1 `jepa_prediction=0`, B2
  `conflict=0` and C both, at one seed, so it can make the comparison neither two-arm probe can:
  **B1 against B2.** No band moved in either file.
  **First, a correction to the claim two paragraphs up: these two artifacts do NOT share a
  `_fingerprint`.** STATIC, measured with `coverage_score._fingerprint` itself:
  `imag_jepa_action_60` yields a **9-tuple** and `exec_conflict_60` a **4-tuple**, because this file's
  probe also prints a value-table count, a greedy-policy count and a whole scramble arm in `n/N ticks`
  form while the Executive probe prints an eligibility counter. What the two share is the three
  HUMAN-named behavioural observables -- act/hold `0/60`, trajectory `47/60`, selected action `23/60`.
  So the collision was never machine-visible, and the new cross-category check added to
  `coverage_score.py` this pass does not catch it either; that is recorded in the check's own comment
  block rather than papered over.
  **STATIC: the two probes pin different objects, and not on the same channel.**
  `imagination_refute_ab.py:102` rebinds `JepaWorldModel.prediction_error` (cerebellum);
  `executive_ab_probe.py:121` rebinds `ExecutiveControl.conflict` (cortex). Neither is a `_drive_w`
  key. They also enter the loop in different PHASES: `_conflict` is computed at
  `autonomous_loop.py:715`, before selection, and reaches the gate on the same tick through the STN
  brake at `:740`; `prediction_error` is called at `:1314`, after the winner already exists (`:1310`
  keys the visit count on the chosen `action`), and reaches behaviour only through the intrinsic
  reward at `:1315` and hence a LATER tick's value table.
  **STATIC: and here is the honest mechanism behind the coincidence.** The jepa error feeds `r_int`,
  which updates the q-table, which is a term in the drive vector, which is the ARGUMENT of
  `self._exec.conflict(drive)` at `:715`. So neutering the world model changes the conflict scalar's own
  output. The two neuters are not one signal; they are two signals that arrive at **one junction**, the
  conflict/STN brake on the BG gate. That is test 3 of the adjudication -- "both reading one upstream
  drive" -- answered as a shared *channel*, not a shared *cause*.
  **RUNTIME: the harness reproduces both recorded artifacts before it is trusted.** Arm A against B1
  gives `0/60, 23/60, 47/60`, identical to `imag_jepa_action_60`; arm A against B2 gives
  `0/60, 23/60, 47/60`, identical to `exec_conflict_60`; and the control arm D reproduces
  `novelty_60.txt`'s `0/60, 0/60`. Three recorded artifacts reproduced by one harness.
  **RUNTIME, the decisive test: the two neutered arms are NOT the same run.** `B1-B2` differs on
  **1/60 selected actions and 7/60 trajectory positions**. The withdrawn baseline's Executive defect was
  byte-identical action traces on 122/122 wake ticks (`COVERAGE_BASELINE.md:12`); this is not that
  shape.
  **RUNTIME: both marginals are nonzero, so neither row is a downstream consequence of the other.**
  With the jepa error already zeroed, additionally killing `conflict` still changes **5/60 actions and
  24/60 positions**; with `conflict` already zeroed, additionally killing the jepa error changes
  **5/60 and 24/60**. `Executive.md:117-123` established this file's precedent for exactly this test and
  kept two rows on a **4/60** marginal; the marginal here is larger, in both directions.
  **RUNTIME, stated against myself, because it is the real finding.** The identical numbers are worse
  than a shared count: the flip **tick sets are byte-identical**, all 23 of them
  `(14, 32, 39, 40 ... 60)`, and at every displayed tick B1, B2 and C select the **same** alternative
  action (tick 14: A `JOINT_B_ROTATE(-6deg)`, all three neutered arms `HOLD`). The mechanism is that
  both perturbations flip the same knife-edge decision at tick 14 and then push the arm into the same
  alternative trajectory, after which "differs from A" is a property of that trajectory. So `23/60` and
  `47/60` are onset-and-trajectory artifacts -- the trap this file already records at lines 48-56 and
  `Executive.md:110-116` records for its `22/60`, now measured to be worse than either said, because
  the tick POSITIONS coincide and not merely the totals. **Neither row's 23/60 is independent evidence
  of anything about the other**, and a reader should treat both as one shared measurement of where this
  arm is fragile plus a small separable residual. The bands stand because they rest on the separable
  residual and on the band's mechanical test, not on the shared 23.
  **RUNTIME: the shared onset tick 14 is not either signal's first firing.** `prediction_error` was
  called **9 times** on ticks 1-13 and `_conflict` was nonzero on **9** of them, so both signals were
  live for a dozen ticks without mattering. Tick 14 is the first tick on which a perturbation
  propagated to the gate's winner, which is a fact about the arbitration, not about either faculty.
  **What this probe could NOT establish, and it is the gap a next pass should close.** Arm D, the
  unrelated-neuter control (`_drive_w['novelty']` pinned), was meant to test whether ANY perturbation
  lands on those same 23 fragile ticks -- which would prove the tick set is a harness constant outright.
  It diverged on **0/60** of every observable, faithfully reproducing `novelty_60.txt` (exit 2,
  UNKNOWN), so it is an inert control and the fragility reading remains an INFERENCE from the tick-14
  mechanism rather than a measurement. A control that perturbs the drive vector and does diverge --
  `taskset` is the obvious candidate at 17/60, though it is itself an Executive row -- would settle it.
- **Tick-count sensitivity is unexplored.** All four artifacts are at n=60, chosen to match the
  `drive_60` control. The ledger's own W-list records that `drive 4` exits 1 and `drive 60` exits 0 on
  the same faculty and seed. The three negatives here could plausibly turn positive at n=500; I did not
  run it, and a longer run is the first thing to try before treating them as settled.
  **REFUTER: still unexplored after the adversarial pass.** `imag_jepa_action_60.txt` is also n=60 and
  also seed 0. The scramble control's `0/60` on the selected action is therefore a statement about 60
  ticks, not about the world model in general: a longer run could find the content decisive. It is the
  first thing to run next, and until it is run the "content is inert" finding is bounded exactly as the
  three negatives are.

## Capability inventory (Policy 2)

- `cerebellum/jepa_world_model.py` -- the live latent world model: `encode`, `predict`,
  `prediction_error`, `make_goal_value`, online `train_step` (VICReg anti-collapse, with the weight
  transport noted above). The only charged Imagination faculty -- charged through the *scale* of its
  error, with its predictive content measured behaviourally inert at n=60 (see the adversarial pass).
  Sole provider of an action-conditioned latent transition function to the planner, the curiosity drive
  and `neuromodulation/empowerment.py`.
- `cerebellum/hjepa_serve.py` -- temporally abstract (Level-1) prospection over a K-tick window, served
  frozen from the torch tier. Unique capability: planning over a *regime* rather than a frame. Not
  instantiated.
- `cerebellum/spiking_jepa.py` -- the spiking-native world model. Unique in principle; measured NOT a
  drop-in replacement (`encode()` invariant across 24 distinct live-width patterns), which is why it is
  EXEMPT rather than adopted. Retiring it would lose the only spiking world model, so it stays.
- `cerebellum/forward_model.py` -- efference copy: reafference cancellation and exafference flagging,
  with online delta-rule gain identification. Unique capability (self-caused vs externally-caused
  sensation); no live consumer.
- `cortex/counterfactual_sim.py` -- the do()-operator: `imagine(z, a, horizon)` rollout, `counterfactual`
  ranking with imagined surprise, and a local three-factor forward-model update. Unique capability: the
  only mechanism that scores *candidate actions* by imagined outcome.
- `cortex/executive_control.py::plan_sequence` (via `autonomous_loop.py:981`) -- multi-step beam search
  over a transition function. Unique capability: rollout depth. Overlaps Executive's planning faculty by
  construction; the *rollout* is scored here, the deliberation there.
- `hippocampus/sleep_replay.py` -- noise-driven offline attractor replay with Hebbian learning and SHY
  downscale-pruning (`:294`, `:151`). Unique capability: offline generative reactivation. Its
  consolidation *purpose* is Memory's faculty; only the dream mechanism is scored here.
- `cortex/brain.py::imagine` -- top-down generative synthesis through the predictive-coding hierarchy
  with the live inference state saved and restored. Unique capability: percept synthesis with no input.

## Gaps, ranked by distance_to_100 x feasibility

1. **Instantiate the hierarchical world model** (0 -> plausibly 35+). The consumer is already written and
   is the highest-priority branch of the loop; the gap is that nothing constructs `HJepaServe` and hands
   it to `AutonomousArm`. Highest distance, lowest effort, and it is a *construction* change, not new
   capability -- but note the rubric's pre-declared ceiling: wiring cannot raise this category, and my
   own evidence agrees with it. `model_based_plan_rollout` is already wired into the bias vector and
   measured 0/60. Wiring a second, better rollout into the same inert consumer would most likely produce
   a second 35, which is why this is ranked first on cost and not on value.
2. **Give the do()-operator a real objective** (40 -> 41-60 in substance). Replace
   `lambda z: sum(z)` at `brain.py:1957` with the goal value the loop already computes
   (`jepa.make_goal_value(self._z_goal)`), and route the ranking somewhere that reads it. Cheap, and it
   converts a placeholder into a faculty.
3. **Make the plan bias decisive, or retire it** (35, either direction). P3 is explicit that a wire which
   does not charge is reverted rather than shipped as coverage, and `imag_plan_60` is that evidence.
   Either the nudge `_plan_w * gain` is too small against the rest of the bias -- measurable in an
   afternoon -- or the whole beam search is 3-deep decoration and should go.
4. **A third exit code in the charge probe** (unblocks 41-60 across the ledger). "Changed an internal
   read, changed no action" is a band the rubric defines and the instrument cannot express. This is not
   an Imagination gap, it is a gap in the thing measuring Imagination, and at least one row here is
   understated by 15 points because of it.
5. **Feedback alignment in `train_step`** (no band change, P1 compliance). Swap `Wz^T` for a fixed random
   `Bz` at `jepa_world_model.py:272-278` and check the update still correlates with the transported one,
   the same way `reasoning/eprop.py` is validated against BPTT.
6. **Held-out prospection** (unblocks 81+, largest and last). Nothing in this category has ever been
   evaluated off the stream it trained on. Until a world model predicts a distribution it did not live
   through, the ceiling for Imagination is 80 and no amount of wiring moves it.

**Added by the adversarial pass**, and on the evidence it outranks everything above it except the
held-out gap:

7. **Make the world model's CONTENT decisive, not just its scale** (61, and the reason 61 is not more).
   The scramble control is the measurement that names this: replay the model's own prediction errors in
   the wrong order and the arm behaves *identically* for 60 ticks. So the intrinsic-reward channel
   currently consumes the error as a magnitude, and any signal with the same magnitude distribution
   would do. Two concrete moves, both cheap enough to measure in one session, and both testable by
   re-running `imagination_refute_ab.py`, where success means **arm C diverges from arm A**: route the
   error per-transition rather than through `SpikingCuriosityDrive`'s running learning-progress
   statistic, which is where the timing information is averaged away (`autonomous_loop.py:1315`); and
   give the value update a term that reads `jepa.predict` for the action it is about to take instead of
   only the error of the action it already took. This is the difference between a world model and an
   exploration bonus, and it is the honest description of what this category's one charged faculty is
   missing.
8. **A cross-category fingerprint check in `coverage_score.py`** (instrument, not brain). W3 compares
   artifacts only within one category file, and the ledger now contains a demonstrated cross-category
   collision: `(0/60, 47/60, 23/60)` at the same onset tick for both this file's `latent_world_model`
   and `Executive.md`'s `acc_conflict_monitoring_evc`. Two unrelated signals producing one triple is
   evidence that the triple is a harness constant; a check that noticed it would have caught the four
   points this pass removed, mechanically, instead of needing a refuter.
   **DONE 2026-07-30, and it does NOT catch this collision -- which is itself the finding.** `W5` is
   added to `coverage_score.py` (a non-fatal warning, mirroring W3 across category files instead of
   within one) and across all 133 rows it reports exactly **one** cross-category group: the seven
   all-zero `charge_probe` negatives sharing `(0/60, 0/60)` in Perception, Consciousness,
   SpatialTemporal, Social and WholeBrain, none of them banded above 40, so no band rests on it. The
   pair that prompted the check is invisible to it because their `_fingerprint`s are a 9-tuple and a
   4-tuple and never collided; the shared triple was three human-named observables. Widening the regex
   would catch it and is still refused -- it changes what `--rerun` compares against for every artifact
   in the tree. So the honest state is unchanged in kind: W3 is a floor within a category, W5 is a floor
   across them, and the collision that motivated both was found by a human reading artifacts.

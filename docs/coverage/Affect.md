# Coverage evidence -- Affect

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| tonic_stress_caution_bias | 50 | src/snn_data_lake/neuromodulation/cortisol.py:51; src/snn_data_lake/neuromodulation/cortisol.py:38; src/snn_data_lake/brainstem/autonomous_loop.py:724; src/snn_data_lake/brainstem/autonomous_loop.py:655 | affect_cortisol_caution_60.txt | 61 claimed by opus-5 auditor:Affect (affect_ab_probe.py cortisol_caution 60); REFUTED to 50 by refuter:Affect / affect_refute_probe.py seeds 60 6 + dose 60 + onset 60 | 2026-07-30 |
| categorical_emotion | 40 | src/snn_data_lake/cortex/affect_core.py:157; src/snn_data_lake/cortex/brain.py:2637; src/snn_data_lake/cortex/brain.py:1487#_route_percept_to_action | affect_categorical_emotion_150.txt | opus-5 auditor:Affect / artifacts/affect_ab_probe.py categorical_emotion 150 + 8-label episodic-cue sweep | 2026-07-30 |
| somatic_marker_recall | 40 | src/snn_data_lake/hippocampus/affective_memory.py:161; src/snn_data_lake/cortex/brain.py:3183; src/snn_data_lake/cortex/brain.py:2873 | - | opus-5 auditor:Affect / live 60-tick brain + one respond() turn, instrumented recall_by_emotion | 2026-07-30 |
| dimensional_valence | 30 | src/snn_data_lake/cortex/brain.py:2105; src/snn_data_lake/cortex/brain.py:2638 | - | opus-5 auditor:Affect / live census, 60 wake ticks (31 distinct values) | 2026-07-30 |
| neuromod_fused_affect_core | 30 | src/snn_data_lake/neuromodulation/affect_integrator.py:134; src/snn_data_lake/cortex/brain.py:1941; src/snn_data_lake/cortex/brain.py:1945 | affect_integrator_150.txt | opus-5 auditor:Affect / artifacts/affect_ab_probe.py affect_integrator 150 | 2026-07-30 |
| tonic_stress_learning_gain | 30 | src/snn_data_lake/neuromodulation/cortisol.py:46; src/snn_data_lake/brainstem/autonomous_loop.py:1434 | affect_cortisol_learnrate_60.txt | opus-5 auditor:Affect / artifacts/affect_ab_probe.py cortisol_learnrate 60 + instrumented call census | 2026-07-30 |
| arousal_dimension | 22 | src/snn_data_lake/cortex/brain.py:2636; src/snn_data_lake/brainstem/autonomous_loop.py:1550; src/snn_data_lake/cortex/brain.py:2651 | - | opus-5 auditor:Affect / live census, 60 wake ticks + 31 stored affective episodes | 2026-07-30 |
| learned_affective_appraisal | 22 | src/snn_data_lake/amygdala/affective_appraisal.py:106; src/snn_data_lake/cortex/brain.py:1890; src/snn_data_lake/cortex/brain.py:2041 | affect_appraisal_150.txt | opus-5 auditor:Affect / artifacts/affect_ab_probe.py appraisal 150 + data/telemetry.jsonl 24,574 snapshots | 2026-07-30 |
| affective_contagion_empathy | 22 | src/snn_data_lake/cortex/affect_core.py:179; src/snn_data_lake/cortex/brain.py:3078; src/snn_data_lake/cortex/brain.py:1576 | affect_empathy_help_150.txt | opus-5 auditor:Affect / artifacts/affect_ab_probe.py empathy_help 150 + tools/charge_probe.py help 60 | 2026-07-30 |
| amygdala_fear_conditioning | 0 | src/snn_data_lake/amygdala/fear_conditioning.py:169; src/snn_data_lake/brainstem/autonomous_loop.py:1171; src/snn_data_lake/brainstem/arm_config.py:32 | - | opus-5 auditor:Affect / instrumented Amygdala.pair/threat over 60 wake ticks | 2026-07-30 |

**Mean over the ten faculties: 28.6.** The auditor's table scored **29.7**; the adversarial pass at
the bottom of this file refutes the one band-61 row to **50**, which is where the 1.1-point
difference comes from. Both numbers are a **downgrade** from the `38` this category carried in
the withdrawn baseline (`COVERAGE_BASELINE.md:32`), and the downgrade is the instrument working, not a
regression in the brain: nothing here got worse, the bar got applied. The withdrawn 38 rested on a
per-faculty table that was never committed and on an "Affect 68" claim resting on a single A/B
(`COVERAGE_BASELINE.md:48`) whose artifact does not exist in the tree. This file replaces both with
ten rows and seven recorded runs.

## What was measured, and how

Every band on this page comes from a **running brain**, not from reading code. Two harnesses:

- `tools/charge_probe.py` -- the sanctioned instrument. Validated here before use: `drive 60`
  reproduced **22/60 trajectory changes, exit 0**, the same control `COVERAGE_BASELINE.md:106` cites
  to show the observable is not degenerate. So a zero this page reports is a property of the signal,
  not of a dead harness.
- `docs/coverage/artifacts/affect_ab_probe.py` -- **written for this audit, and it is the thing a
  refuter should attack first.** It exists because `charge_probe` neuters one key of `Brain._drive_w`,
  and the *only* Affect faculty in that dict is the empathy `help` drive. The tonic stress state, the
  categorical emotion, the learned appraisal and the neuromodulator-fused core affect are all consumed
  elsewhere and are simply out of that probe's reach; scoring them from static reading is what
  `COVERAGE_RUBRIC.md:65-68` forbids. The new probe copies charge_probe's method deliberately --
  `make_brain(seed=0, persist=False)`, one signal pinned at its **consumer boundary** for the whole run
  (pinning the return, not zeroing state after the tick, for the same leak reason charge_probe uses a
  `_PinnedWeights` subclass), behavioural observables only (the action `_select` returned,
  `arm_stopped`, `map_position`). It carries two guards that make it refuse rather than guess: a
  **self-control** (A vs a second un-neutered A, which must diverge on 0 ticks -- it did, on every run
  recorded here) and an **eligibility check** (a signal that took one value all run is a constant, and
  neutering a constant proves nothing -> NOT-ELIGIBLE, no verdict).

## The one charged faculty, and why it is the only one

**Superseded in part: the band claimed in this section is 61 and the row now reads 50. Read
"ADVERSARIAL PASS" at the bottom of this file before quoting anything here.** The measurement below
stands and reproduces; what fell is the band it was read as licensing.

`tonic_stress_caution_bias` is the single Affect signal that reaches behaviour. `Cortisol.caution_bias()`
(0.6 x the stress level, measured over **25 distinct values** in 60 ticks) is summed into the
basal-ganglia caution term at `autonomous_loop.py:724`, which damps the dopamine that has to overcome
the tonic GPi for an action to be released. Pinning it to 0.0 changed **the selected action on 4/31
selections and the trajectory on 21/60 ticks**, against a self-control of 0/31 and 0/60. That is
rubric band 61-80: "neuter the signal, and the action measurably differs". It is **not** 81+ -- one
seed, 60 ticks, no held-out distribution, so `--rerun` reproducing it exactly (verified) proves
determinism, not generalization.

**Two objections to this row that I want on the record rather than buried.** (1) *Is cortisol an Affect
faculty?* I claim it is -- it is this brain's only body-state affect signal, and "stress shifts the
decision toward AVOID" is the somatic half of somatic marking. But `WholeBrain` (homeostasis) can
reasonably claim the same module, and if it does, one of the two categories is double-counting. (2)
*It is the highest row and it rests on my own harness.* Both true. The mitigation is that the harness
is tracked, the command is recorded, `--rerun` re-executes it, and the self-control is inside the
artifact, so the claim is checkable rather than assertable.

## Six honest negatives, all recorded

These are results, not failures, and attaching the artifact is what stops the same run being re-badged
as a charge later:

- `categorical_emotion` 150 ticks -- CONSUMED-BUT-NOT-DECISIVE, exit 1. The label took **8 distinct
  values** over the run (`AffectCore` defines 8 emotions), and pinning every one of them to `calm`
  moved 0/73 actions and 0/150 trajectories.
- `affect_integrator` 150 -- CONSUMED-BUT-NOT-DECISIVE, exit 1. The emergent-prototype id took **6
  distinct values** as the self-organising map grew, and forcing the fused core affect to (0.0, 0.0)
  changed no behaviour.
- `appraisal` 150 -- CONSUMED-BUT-NOT-DECISIVE, exit 1. Its one non-telemetry consumer chain changes
  nothing behavioural even when the appraisal is forced to exactly neutral.
- `cortisol_learnrate` 60 -- CONSUMED-BUT-NOT-DECISIVE, exit 1. The aversive learning-rate gain really
  does fire (`learn_rate_scale` returned != 1.0 on 9 of 31 calls) and still moves no action.
- `empathy_help` 150 -- CONSUMED-BUT-NOT-DECISIVE, exit 1. The help drive fires to 0.456 on a distress
  turn and decays through 44 distinct values, and pinning it moved 0/75 actions.
- `tools/charge_probe.py help 60` -- UNKNOWN (undirected source, no divergence), exit 2. On a bare wake
  tick `_help_drive` is 0.0 on 60/60, so the sanctioned probe's one Affect source never gets a chance
  to fire. Recorded as `affect_help_source_60.txt`.

The three 150-tick negatives share an identical observable fingerprint and will raise a `W3` warning
("probably one serial chain scored as 3 faculties"). Here that warning **misfires, and the reason is
worth writing down**: W3 detects one chain probed at N points by matching flip counts, which is sound
for positive results and vacuous for negatives -- three genuinely independent neuters that each move
nothing are all-zero by definition, so they *must* collide. The chains are distinct
(`AffectCore.recognize` -> episodic cue; `AffectiveAppraisal.appraise` -> coherence raster;
`AffectIntegrator.integrate` -> emergent prototype id) and the artifacts' `signal:` lines say so.

**Adjudicated 2026-07-30: FALSE POSITIVE CONFIRMED, now MEASURED rather than asserted, and all three
bands stand.** The paragraph above argued the three chains are distinct from their call sites. That is
the argument W3 exists to distrust, so it was tested: if these were three probe points on one chain,
one signal would be a function of another. RUNTIME, 150 wake ticks, `make_brain(seed=0,
persist=False)` on the daemon `sys.path` (tier `eprop-snn`), logging the three signals the three
artifacts each pin -- `_emotion` (`brain.py:2637`), `_emergent_emotion` (`brain.py:1941-1946`) and
`_appraised` (`brain.py:1890`) -- per tick:

- distinct values **5** (`anger, boredom, excited, fear, sadness`) / **6** (prototype ids `0..4`,
  `None`) / **3** (`(0.0, 0.0)`, `(0.0, 0.062)`, `None`). Three different value domains.
- the map between the two emotion-category readouts is **many-to-many in both directions**: all
  **5 of 5** label keys map to several prototype ids (`fear` -> `{0,1,2,3,4}`), and **4 of 6**
  prototype ids map back to several labels (`0` -> all five labels). Neither is a function of the
  other, which is exactly what a serial chain could not produce.
- they move on different ticks: **45 / 28 / 14** change ticks, with only 19 shared between the two
  emotion readouts and 8 between emotion and appraisal.

Reproduce with a 30-line census that ticks a brain and diffs the three attributes; no probe or neuter
is involved, so nothing was recorded as an artifact -- it is a read of three live attributes, and the
numbers above are the whole result. **Two further mechanical reasons no band here is inflated.** First,
half of the six-element fingerprint is the probe's **self-control** (`SELF-CONTROL A vs A` at
`0/73, 0/150, 0/150`), which is *required* to be zero for the run to be valid at all -- so three valid
runs of this probe collide on three of six elements by construction, before any faculty is considered.
Second, all three rows are at or below 40 (40, 30, 22) and F8 only demands an exit-0 artifact above 40;
the three colliding artifacts are exit-1 honest negatives, so no row draws its band from them. The
discriminating line is in the same stdout and outside the regex: `distinct values the signal took in A`
reads **8** with a sample of labels, **6** with a sample of `None`/ints, and **8** with a sample of
`None`/tuples.

One thing this measurement settles that the adversarial pass below could not: it flags
`neuromod_fused_affect_core` as "a second emotion-category system beside `categorical_emotion`" and
declines to merge them for want of evidence. The many-to-many mapping is that evidence, and it supports
not merging **for coverage**: these are two mechanisms computing an emotion category from different
inputs (`recognize` reads self-model valence and `|rpe|`; `integrate` reads threat, `rpe`, raphe mood
and cortisol), not one mechanism scored twice. That leaves the P2 question -- two mechanisms for one
capability, one of which should be adopted and the other retired -- fully open, and it is a question
about the code, not about this table.

## The three leads I was handed, adjudicated

1. **`threat()` returns exactly 0.0 -- CONFIRMED, and confirmed to be correct rather than broken.**
   Instrumented on a live brain: `Amygdala.threat` was called **31 times in 60 wake ticks and returned
   0.0 on 31/31**; `Amygdala.pair` was called **0 times**. `pair()`'s only live call site
   (`autonomous_loop.py:1171`) sits inside the safety-first early return that needs `distance < CONTACT`,
   and `arm_config.py:32` still has `SAFETY_CALIBRATED = False`. So `amygdala_fear_conditioning` scores
   **0 under rubric rule 4** -- a precondition that never fires -- while remaining P3 exempt-awaiting-a-
   prerequisite. The zero is a statement about the absence of a body, not about the code, and "fixing"
   it by binding `pair()` to disappointment would be the theater P0 forbids.
   *Incidental finding:* that module's own docstring cites `autonomous_loop.py:1108` and `:1102` for
   these two lines; they are now `:1171` and `:1165`. A ~63-line citation drift, exactly the failure
   mode `docs/coverage/README.md:75` says no cheap check catches.

2. **`derived.valence` pinned at 0.0 -- CONFIRMED on the deployed brain, REFUTED on a fresh one.**
   `data/telemetry.jsonl` currently holds 24,574 snapshots; `derived.valence` (which is
   `affective_appraisal[0]`, `brain_telemetry.py:246`) is **0.0 on every snapshot sampled, including all
   of the most recent 4,000**, so the 20,023-snapshot finding at `brain_telemetry.py:370` still stands
   today. But a fresh `make_brain(seed=0, persist=False)` gives appraisal valence **0.125 on 23 of 60
   ticks**. Both are real, and the difference is the actual finding: the teaching target at
   `brain.py:1896` is `clamp(_lp_mean - threat)`, `threat` is 0.0, and in the long-running daemon
   `_lp_mean` has settled to exactly 0.0 -- so `err_v` is exactly 0.0, no update ever touches `w_pos`
   or `w_neg`, and the readout `pos - neg` is structurally 0.0 forever. The valence channel is pinned
   by **exhausted learning progress**, not by a missing wire. A fresh brain still has progress to make
   and does move. That is why this row is 22 and not 30: its one non-telemetry consumer
   (`brain.py:2041` -> `RegionCoherence` -> synfire binding -> the workspace ignition threshold) is a
   real wire that in deployment carries a constant.

3. **P3 disposition of the amygdala -- agreed, and left alone.** Nothing on this page proposes wiring it.

## What I could not measure

- **Nothing here has held-out evidence, so no row can exceed 80** and none claims to. The category's
  ceiling on today's evidence is 61 as the auditor read it and **50** after the adversarial pass.
- **`categorical_emotion` and `somatic_marker_recall` are capped at 40 by the instrument, not by the
  measurement.** Both demonstrably change something the brain later reads, which is band 41-60's own
  definition. The emotion label is the affect half of the episodic recall cue (`brain.py:1447`): sweeping
  the eight labels at a fixed place returned **eight distinct (episode key, confidence) completions**,
  so it does select which memory is retrieved. The somatic marker fires **once per `respond()` turn and
  0 times per wake tick**, retrieves a genuinely charged episode (24 of 31 stored episodes have
  `|valence| >= 0.2`), and its text reaches the reply -- verified in the returned string. Band 41+ needs
  an A/B artifact with exit 0, and the A/B I *can* record is behavioural, where both come back
  non-decisive. So they sit at the top of band 21-40 with the internal-read evidence written down here.
  A probe that neutered a signal and diffed an *internal read* rather than an action would be the
  instrument change that could legitimately move these two rows.
- **`dimensional_valence` has no A/B of its own.** It is `lp - cortisol.level` (`brain.py:2105`), so a
  clean neuter is a self-model neuter, which is Consciousness's territory, not mine. Scored 30 on live
  census (31 distinct values in 60 ticks) plus call-site consumers; a refuter should read that as
  "unproven", not "proven weak".
- **Arousal, honestly: this brain has no arousal system.** Every arousal input is a proxy or a
  constant. The value fed to the categorical readout is `min(1, |rpe|)` (`brain.py:2636`), a
  reward-prediction-error magnitude wearing an arousal label. The arousal tag written onto every
  affective episode is `threat` (`autonomous_loop.py:1550`), measured at **exactly 0.0 on 31/31 stored
  episodes** -- so the "two-dimensional" somatic retrieval is one-dimensional in practice: five
  retrieval targets spanning arousal 0.0 to 1.0 all returned the same episode. The one dedicated
  arousal-like pool, locus coeruleus noradrenaline, sat at **1.0 on 55 of 60 ticks** (3 distinct values
  -- saturated). Hence 22.
- **`Cortisol.reflex_threshold_scale()` has zero live callers** -- 0 calls in 60 instrumented wake
  ticks. Cortisol's documented hypervigilance arm is dormant; only its caution and learning-rate arms
  run. Not scored as a separate row (it would be a padding row at 0-20 and it is a *method*-level
  dormancy case, which `substrate/method_dormancy_gate.py` is the right home for), but it is a real
  P3 item.

## Capability inventory (Policy 2: what provides what, before anything is retired)

- `neuromodulation/cortisol.py` -- tonic stress state. The ONLY behaviourally-charged affect signal in
  the brain. LIVE, and CHARGED **only in its saturated regime** -- the adversarial pass measured the
  action change at 3 of 6 seeds and those are exactly the 3 where `level` hit the `min(1.0, ...)`
  clamp at `cortisol.py:38`. In the graded regime it is behaviourally inert.
- `cortex/affect_core.py` -- dimensional -> categorical emotion via a spiking winner-take-all over a
  Russell circumplex, plus affective contagion. LIVE, non-decisive.
- `neuromodulation/affect_integrator.py` -- fuses four neuromodulators into a coupled (valence, arousal)
  core AND grows its own emotion prototypes rather than reading a hand-coded circumplex. LIVE,
  non-decisive.
- `amygdala/affective_appraisal.py` -- learned, three-factor, gradient-free (valence, arousal) appraisal
  of an arbitrary feature vector. LIVE; valence channel pinned at 0.0 in deployment.
- `amygdala/fear_conditioning.py` -- CS-US acquisition, discrimination, extinction, spontaneous
  recovery, multi-US categories. Imported and called every tick; never learns, because no aversive
  event exists to learn from.
- `hippocampus/affective_memory.py` -- affect-indexed episodic retrieval (Damasio somatic marker). LIVE
  on the chat path only; arousal axis of the key inert.

Nothing above is a duplicate of anything else -- there is no P2 merge candidate inside this category.

## Gaps, ranked by distance_to_100 x feasibility

1. **Give the arousal axis a real source.** It is currently `|rpe|` in one place and a constant 0.0 in
   another. `medulla/interoceptive_arousal.py` and `neuromodulation/locus_coeruleus.py` both exist;
   feeding a genuine arousal scalar into `affmem.store(arousal=...)` would make the somatic-marker key
   two-dimensional, which is the cheapest real gain in the category. Highest feasibility, and unlike
   the amygdala wire it changes no safety-adjacent semantics.
2. **Give the appraisal a teaching target that does not collapse.** The valence channel is pinned
   because `_lp_mean - threat` is exactly 0.0 in the deployed brain. A target that stays informative
   once learning progress saturates is a P1-shaped fix, not a new module.
3. **An internal-read A/B verb.** Two rows here are held at 40 by the absence of a probe that diffs a
   memory write or a retrieved key instead of an action. That is an instrument gap, and closing it
   would raise this category without touching the brain -- which is also exactly why it should be
   scrutinised before it is trusted.
4. **Emotion categories that condition anything.** Both category systems (fixed circumplex and grown
   prototypes) are measured non-decisive. An emotion that gated a policy, a learning rate, or a
   retrieval threshold would be the first genuine 61+ in the categorical half of this category.
5. **A body.** `amygdala_fear_conditioning` cannot leave 0 until `SAFETY_CALIBRATED` is true. Lowest
   feasibility, and correctly last.

---

# ADVERSARIAL PASS (refuter, 2026-07-30): 61 -> 50, category 29.7 -> 28.6

One row was above 40 and it is now below it. **Score before 29.7, score after 28.6, one band-41+
claim refuted of one examined.** Three new artifacts, all machine-written by
`coverage_score.py record` from `docs/coverage/artifacts/affect_refute_probe.py`:
`affect_refute_seeds_60x6.txt` (exit 1), `affect_refute_dose_60.txt` (exit 0),
`affect_refute_onset_60.txt` (exit 0).

## Two attack lines FAILED, and they are reported first

Refuting is the default, not the goal, so the attacks that missed belong on the record.

1. **Forgery: no.** `coverage_score.py --rerun affect_cortisol_caution_60` at this HEAD returns
   `recorded exit 0 / now 0`, observables `(('0','31'),('0','60'),('0','60'),('4','31'),('0','60'),
   ('21','60'))` identical, `reproduces`. The auditor's probe is tracked, deterministic and does what
   its stdout says.
2. **One-onset echo: no, and the auditor understated their own evidence.** The suspicion was the
   defect `Executive.md`'s item 6 ("the audited methodological finding is confirmed") and
   `WholeBrain.md:88-94` found -- `map_position` integrates its own motion, so a trajectory count is
   `N - onset + 1` and four category files quote the *same*
   arithmetic from a tick-39 split. That does apply to this row's `21/60`: **1 onset, 0
   reconvergences, first split at tick 39**, so `21/60` is one event and should never have been
   printed beside the action count as if it were a second observable. But the **action** count is
   not onset-derived: 4/31 with **3 independent onsets and 2 reconvergences**. The action sequence
   splits, rejoins, splits again. Three separate decisions changed, not one echoed. And the first
   flip is `HOLD -> JOINT_A_ROTATE(+6deg)` -- with stress removed the arm moves where the stressed
   brain held, which is the direction `cortisol.py:12` claims. The mechanism is real.

## What refutes the band: the charge lives only where the signal is CLAMPED

`affect_refute_seeds_60x6.txt`, 6 seeds x 60 ticks, self-control 0 at every seed:

(deliberately NOT a markdown table -- `coverage_score.py`'s `_table` raises `F3` on any pipe row in
an evidence file that is not six cells, which is the trap the auditor hit and fixed in this file and
which was refusing `Perception.md` earlier today; a prose sub-table costs the whole sweep its total)

- seed 0 -- max **0.600** -> gate `da` 15/31 -> action **4/31**, 3 onsets; trajectory 21/60
- seed 1 -- max 0.051 -> gate `da` 3/35 -> action **0/35**; trajectory **0/60**
- seed 2 -- max **0.600** -> gate `da` 29/34 -> action **16/34**, 3 onsets; trajectory 51/60
- seed 3 -- max 0.086 -> gate `da` 9/32 -> action **0/32**; trajectory **0/60**
- seed 4 -- max **0.600** -> gate `da` 27/32 -> action **15/32**, 9 onsets; trajectory 54/60
- seed 5 -- max 0.214 -> gate `da` 7/26 -> action **0/26**; trajectory **0/60**

The split is perfect and it is not noise: the three seeds where deleting the entire signal changes a
decision are exactly the three where `caution_bias` reached **0.600**, which is `MAXMOD * 1.0`, i.e.
the ticks where `Cortisol.step`'s `self.level = min(1.0, self.level)` clamp (`cortisol.py:38`) has
pegged the integrator at its ceiling. On the three seeds where the tonic state stayed in its
**graded** range (peak 0.05 to 0.21 of a possible 0.60), removing the whole signal moved **0 of 93
selections and 0 of 180 ticks** -- not a thin effect, an exactly-zero one, three times.

So the honest sentence is not "the tonic stress state charges behaviour". It is: **the stress signal
charges behaviour when, and only when, it has saturated into a constant 0.600.** The graded slow
modulator that `cortisol.py:1-15` describes -- integrate, decay, recover -- is the part measured
inert. This is rubric rule 4's shape one level in: not "the precondition never fires", but "the
precondition is the clamp, and it fires on half the seeds". A faculty whose entire behavioural
footprint sits at a fixed `MAXMOD = 0.60` (`cortisol.py:24`) is also a P1 observation, since the
charge is carried by the magic constant rather than by anything learned or baseline-relative.

## Why 50 and not 45, and not 61

Band 41-60 is earned **unconditionally**, and by more than the auditor claimed. **Measured:** the
neuter changes `da` -- the dopamine the BG gate weighs against its tonic GPi -- on **15 of 31 gate
calls** at seed 0 and on at least one call at **6 of 6 seeds**, including all three where no action
moved. I instrumented `BasalGangliaGate.select`'s actual argument rather than inferring it, and the
rate tracks the signal's size exactly as the dose-response predicts: 29/34 and 27/32 at the two
saturated seeds, 3/35, 9/32 and 7/26 at the three graded ones. So the internal read is charged at
every seed and the *decision* is charged at half of them -- which is the whole reason the row lands
inside 41-60 rather than at either edge of it.

**Static, and labelled as such because rule 1 of the refuter brief says to:** I read the rest of the
chain out of `da` rather than measuring it end to end --
`action_gating.py:110` computes the whole `gpi` vector *from* `da`, `autonomous_loop.py:758-759` takes
the runner-up-minus-winner margin off that vector into `_decision_margin`, and `brain.py:1846-1852`
divides it by its own running baseline and feeds the result into `_metacog.update(confidence=...)`.
So "the output alters something the brain itself later reads" holds on a measured first link and a
verified-by-reading remainder, which is what band 41-60 asks for and is strictly more than the
telemetry ceiling of band 21-40.

Band 61-80 asks for a decision change. One is measured, it is structurally sound, and it is **absent
on half the seeds for a systematic reason**.

**The precedent that fixes the band is a peer refuter's, and it was found the same day.**
`Executive.md`'s item 2 takes `task_set_sr_mapping` from **61 to 45** on exactly this defect:
*"A signal that changes no decision at all on half the sampled seeds has not earned that. Band 41-60
is what it has earned without reservation."* Charging at 2 of 4 seeds cost that row 16 points, so
charging at 3 of 6 cannot cost this one nothing. Two differences put this row a little above theirs
rather than level with it, and both are measured rather than argued:

- **Their inert seeds were inert at full strength; mine are not.** `exec_refute_taskset_elig.txt`
  closed the eligibility escape by showing the S-R drive term nonzero on 60/60 ticks at both dead
  seeds, peaking at 0.114 against seed 0's 0.141 -- present, removed, ignored. Here the three dead
  seeds peak at **0.051, 0.086 and 0.214 of a 0.600 range**, so the inertness is *explained* by the
  signal being small rather than unexplained at comparable magnitude. A graded influence that is too
  weak to move an argmax at 20% of its range is a different finding from one that is ignored at 80%.
- **A dose-response exists here and does not there.** 1, 2, 4 flips at x0.99, x0.9, x0.75 is direct
  evidence the magnitude does work, which no seed sweep can supply.

**50** is therefore the top of "changes an internal read": credited above that band's floor because a
real decision change is measured, and held below 61 because the regime in which it happens is the
clamp. A reading that treats the saturated regime as the one the faculty exists for would justify
55-60; nothing in the evidence justifies 61, and the rubric's adversarial default settles a tie
against the higher band. Note also which of the two peer 45s I am *not* leaning on: Executive's
refuter deleted its `stn_hyperdirect_brake` and `pfc_gain_metalearning` rows outright ("one cause,
counted twice"), so those two 45s no longer exist to compare against.

## A third control, reported because it cuts both ways

`affect_refute_dose_60.txt` scales `caution_bias` by 0.99, 0.9, 0.75, 0.5, 0.25, 0.0 at seed 0:
action flips **1, 2, 4, 4, 4, 4** (onsets 1, 2, 3, 3, 3, 3). Two readings, both true.

- *For the auditor:* the response is **monotone and graded** over the top quarter of the range -- the
  scalar's magnitude does work, so this is not a pure knife-edge, and the `x0.5`-equals-`x0.0`
  identity in `affect_refute_onset_60.txt` (4/31, first 23, 3 onsets, 2 reconvergences, byte for byte)
  is the observable saturating, not the mechanism being binary.
- *Against:* a **1% reduction already flips a decision**, and pinning the signal to a *constant* at
  its own run mean (0.2789, dynamics destroyed, magnitude kept) flips **6/31 actions with 5 onsets**
  -- more than deleting it. So `charge_probe`-style positive verdicts are cheap on this arm loop: the
  argmax and the GPi threshold sit close enough to a tie that almost any perturbation of a caution
  term will print a flip. That is an instrument finding about every 61 in this sweep, not only this
  one, and it belongs beside `README.md:72-75`'s known holes.

## Roster and double-count: checked, and the number is not where the padding is

- **Not padded in the direction that would matter.** Ten rows, and every arguably-over-granular row
  sits at 30 -- within 0.3 of the mean -- so merging them barely moves the score. Two are genuinely
  over-granular: `tonic_stress_learning_gain` is the **same state variable** as the refuted row (both
  artifacts census 25 distinct values because `caution_bias` is `0.6 * level` and the learn-rate arm
  samples `level` itself), which is the "one signal, several consumers" case `Executive.md`'s items 3
  and 4 deleted rows for ("one cause, counted twice"); and `neuromod_fused_affect_core` is a second
  emotion-category system beside
  `categorical_emotion`. Merging the first gives 9 rows and **28.4**; merging both gives 8 rows and
  **28.3**, against 28.6 as scored. Recorded, not applied:
  the auditor's granularity is defensible (two consumers, one decisive and one not, is information)
  and re-cutting a roster to move 0.2 is not measurement.
- **`affective_contagion_empathy` is scored in the wrong category.** `COVERAGE_RUBRIC.md:34` puts
  empathy under **Social**, not Affect. It is a 22 and below the mean, so moving it *raises* Affect
  to 29.3 -- which is precisely why it should be adjudicated by the sweep and not by whoever
  benefits.
- **A 61 that two other files say belongs here, and that I did not import.** The `drive` faculty
  (`drive_60.txt` / `wb_drive_60.txt`, exit 0, 22/60) was re-scoped out of Executive as *"the dominant
  homeostatic need -- interoceptive motivation, an Affect faculty"* (quoted at `WholeBrain.md:60-67`;
  Executive's own rewrite has since dropped the passage, so that citation is the surviving record of
  it). It is **not** an omitted faculty -- `WholeBrain.md`'s table carries it as
  `homeostatic_drive_to_action` at band **40**, capped there "by category and not by effect size",
  with that file's refuter stating outright that *"the row's evidence supports 61 in Affect"* and
  measuring 17/60 action flips on it. So the ledger has the faculty; what it does not have is an
  adjudication. I have deliberately **not** added the row: importing another category's charge would
  take this file to **31.5** on a faculty I did not measure, and I do not accept the scoping --
  `COVERAGE_RUBRIC.md:33` gives Affect "valence/arousal, appraisal, emotion categories, somatic
  marking", and a homeostatic need is motivation, not affect. Flagged as the one place where a
  sweep-level ruling moves this category's number by three points in either direction.
- **The caution sum is triple-claimed.** `autonomous_loop.py:722-729` sums `cort.caution_bias()`,
  `threat`, `intero`, `reliability_hold` and `_social_risk` into ONE scalar. This row claims the
  first; `Social.md`'s 61 claims `_social_risk` on `social_tom_caution_60.txt`; `WholeBrain.md:103-111`
  claims `interoceptive_caution_to_selection` and flags the separability test as open. Three
  categories, one argmax input, and the trajectory observable cannot tell them apart (21/60 here,
  24/60 there -- two onsets one tick apart). The sources are distinct and separately driven, so this
  is not automatically one faculty, but the sweep should know that ~3 points of its headline rest on
  five addends of a single `min(1.0, ...)`.

## What the refutation implies for the gap list

The cheapest honest gain in this category moved. It is no longer arousal (gap 1): it is **make the
stress signal decisive in its graded range**, because the measurement now says the wire is real and
only the clamped end of it does anything. The integrator reaches 0.600 by collecting `GAIN = 0.30`
per collision (`cortisol.py:23`, and `affect_refute_onset_60.txt` shows the live `aversive` argument
taking `1.0` from `coverage_reward`'s collision penalty at `autonomous_loop.py:498`), so three
collisions peg it. A P1-shaped fix -- integrate aversive events relative to their own running scale
instead of at a fixed gain into a hard clamp -- would give the same wire a graded effect and is worth
more than a new module.

## What I did not do

- **No multi-seed pass on the six negatives.** If seed sensitivity is this strong for the one
  positive, an exit-1 negative at seed 0 is weaker evidence than it looks: `categorical_emotion`,
  `appraisal`, `affect_integrator`, `cortisol_learnrate` and `empathy_help` may all be decisive at
  some seed nobody ran. Their bands (22-40) do not depend on the negative being universal, so no row
  moves on this, but "measured non-decisive" in this file means "at seed 0".
- **I did not re-adjudicate the two rows at 40.** `WholeBrain.md:113-129` lowered three analogous
  "held at 40 by an instrument gap" rows to 35 on the ground that a probe was recordable and was not
  recorded. Affect's auditor *did* record one (`affect_categorical_emotion_150.txt`, exit 1) and
  recorded none for `somatic_marker_recall`. Applying WholeBrain's standard uniformly would put both
  at 35 and this category at 27.6. I have no measurement of my own on either, and lowering a row on a
  cross-file analogy rather than on evidence is the mirror image of the inflation this pass exists to
  stop. Flagged for the sweep; not applied.
- **No held-out anything.** 50 is not a claim about generalization and 81+ remains unreachable here.

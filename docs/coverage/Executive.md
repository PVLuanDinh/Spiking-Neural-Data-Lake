# Coverage evidence -- Executive

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

**This file has been through the rubric's adversarial pass (`COVERAGE_RUBRIC.md:92-95`). The audited
score was 38.3 on ten rows; it is now 34.6 on eight. Three of the five band-41+ claims fell.** What
fell, and to what, is in "The refutation" below; the two that survived survived a harder test than the
one that produced them.

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| pfc_goal_top_down_bias | 61 | src/snn_data_lake/engine/autonomous_loop.py:1175#_pfc_bias; src/snn_data_lake/cortex/prefrontal_executive.py:350#top_down_bias; src/snn_data_lake/engine/autonomous_loop.py:1781#step | exec_pfcbias_60.txt | opus-5 refuter / executive_ab_probe.py pfc_bias 60 (exit 0, reproduces) + exec_refute_seeds_pfcbias (seeds 1,2,3 all diverge) + exec_refute_marg_pfcgain_given_pfcbias (absorbs pfc_gain_metalearning) | 2026-07-30 |
| acc_conflict_monitoring_evc | 61 | src/snn_data_lake/engine/autonomous_loop.py:890#_bg_select; src/snn_data_lake/engine/autonomous_loop.py:1102#_pfc_bias; src/snn_data_lake/cortex/executive_control.py:110#conflict; src/snn_data_lake/engine/autonomous_loop.py:915#_bg_select | exec_conflict_60.txt | opus-5 refuter / executive_ab_probe.py conflict 60 (exit 0, reproduces) + exec_refute_seeds_conflict (seeds 1,2,3 all diverge) + exec_refute_marg_stn_given_conflict (absorbs stn_hyperdirect_brake, 0/60 marginal) | 2026-07-30 |
| task_set_sr_mapping | 45 | src/snn_data_lake/engine/brain.py:2128#_c1_tick; src/snn_data_lake/engine/brain.py:1809#_drive_for; src/snn_data_lake/cortex/task_set.py:159#bias | exec_taskset_60.txt | opus-5 refuter / DOWNGRADED from 61 by exec_refute_seeds_taskset (0/60 on every observable at seeds 1 and 2) + exec_refute_taskset_elig (eligible on 60/60 ticks at those seeds) | 2026-07-30 |
| response_inhibition_nogo | 35 | src/snn_data_lake/basal_ganglia/action_gating.py:106#select; src/snn_data_lake/engine/autonomous_loop.py:928#_bg_select | - | opus-5 / live gate spy, 200 ticks (veto rate; counterfactual undefined) -- audited value, re-read not re-measured | 2026-07-30 |
| multi_step_plan_sequence | 30 | src/snn_data_lake/engine/autonomous_loop.py:1157#goal_value; src/snn_data_lake/cortex/executive_control.py:179#plan_sequence | exec_plan_60.txt | opus-5 refuter / executive_ab_probe.py plan 60 (exit 1 honest negative, reproduces) | 2026-07-30 |
| metacognitive_arbitration | 30 | src/snn_data_lake/engine/brain.py:2100#_update_metacog; src/snn_data_lake/engine/brain.py:2985#_enter_sleep; src/snn_data_lake/engine/brain.py:3604#respond | - | opus-5 / live trace, 200 ticks, both gates counted at their decision point -- audited value, citations re-verified | 2026-07-30 |
| pfc_working_memory_maintenance | 15 | src/snn_data_lake/cortex/prefrontal_executive.py:334#maintain; src/snn_data_lake/engine/autonomous_loop.py:1053#_pfc_bias | - | opus-5 / live trace, 200 ticks + the reverted-wire record at the citation -- audited value, citations re-verified | 2026-07-30 |
| task_set_shifting | 0 | src/snn_data_lake/cortex/task_set.py:213#observe; src/snn_data_lake/cortex/task_set.py:221#shift | - | opus-5 / live trace, 200 ticks (rubric rule 4) -- audited value, citations re-verified | 2026-07-30 |

**Mean over the eight rows: 277 / 8 = 34.6.** Two above 40 with a recorded, reproduced and now
multi-seed A/B; one at 45 that the audit had at 61; five at or below 35.

## The refutation, item by item

### 1. Every artifact is real. The forgery check found nothing.

All seven executive artifacts plus the `drive_60` harness control were re-executed with
`coverage_score.py --rerun`. Every one reproduced its recorded exit code **and** its full observable
fingerprint, including `exec_plan_60`'s exit 1 and `exec_pfcgain_200`'s 200-tick run. `drive_60`
reproduces its documented `0/60, 22/60`, so the harness is the same harness
`COVERAGE_BASELINE.md:90` describes. No band fell for want of evidence; the three that fell, fell to
new evidence.

### 2. task_set_sr_mapping: 61 -> 45. It is behaviourally inert on half the seeds tried.

This is the substantive refutation and it comes from the caveat the auditor named as their own
biggest: every audited figure was seed 0. Re-running the identical neuter at seeds 1, 2 and 3
(`exec_refute_seeds_taskset.txt`, exit 1):

(selected action / trajectory / act-hold flips, per 60 ticks -- laid out as a list rather than a
markdown table because `coverage_score._table` parses every pipe-prefixed line in the file and a
second table in the prose refuses the whole category with `F3`):

- seed 0, audited: 17/60 action, 22/60 trajectory, 0/60 act-hold
- seed 1: **0/60**, **0/60**, 0/60
- seed 2: **0/60**, **0/60**, 0/60
- seed 3: 16/60, 40/60, 0/60

At seeds 1 and 2 the two arms are identical on every tick of every observable. The obvious escape --
"the readout was not active at those seeds, so nothing was removed" -- is closed:
`exec_refute_taskset_elig.txt` (exit 0) measures the S-R readout's actual drive contribution
`_drive_w['taskset'] * clamp(task_pref[i])` on the wired arm and finds it **nonzero on 60/60 ticks at
seed 1 and 60/60 at seed 2**, peaking at 0.114 and 0.099 against seed 0's 0.141. So the signal was
present at full strength, was removed, and the arm did not care. That is
`CONSUMED-BUT-NOT-DECISIVE` at seeds 1 and 2 and `BEHAVIOURAL CHARGE OK` at seeds 0 and 3, on
comparable magnitudes.

Band 61 is "changes a decision, action, or learning update". A signal that changes no decision at all
on half the sampled seeds has not earned that. Band 41-60 is what it has earned without reservation:
the readout enters `_ext_drive` and hence the `drive` vector the BG gate reads on every tick at every
seed, so it demonstrably changes an internal read. **45.** The exit-0 artifact
(`exec_taskset_60.txt`) is retained on the row because it licenses more than I am claiming, which is
the honest direction; the artifact that caused the downgrade exits 1 and is named in `measured_by`
rather than in the artifact column, where `F8` would refuse it.

Note what this does *not* say. It is not a claim that the wire is dead -- it charges at two of four
seeds. It is a claim that the audited 61 was a property of one trajectory.

### 3. stn_hyperdirect_brake: 45 -> the row is gone. One cause, counted twice.

The audit stated in prose that "neutering `conflict` removes the STN brake too" and then scored the
brake as an independent 45-point faculty anyway. That is the defect the brief names -- a downstream
node of an already-scored signal, given its own row. `exec_refute_marg_stn_given_conflict.txt`
settles it mechanically with a three-arm test (A wired, B conflict neutered, C conflict + stn
neutered): **C vs B is 0/60 on selected action, 0/60 on trajectory, 0/60 on act/hold, and 0/60 on
`_decision_margin`, `_control_gain` and `_metacog_rel`.** Arms B and C are the same run. The
arithmetic says why: `stn = _stn_gain * max(0, conf - _conflict_base)` (`autonomous_loop.py:740`)
with `_conflict_base` initialised to 0.0 (`:378`) and updated only toward `conf`, so pinning
`conflict` to 0 pins `stn` to 0 for the whole run.

The brake's own mechanism is real -- `exec_refute_probe.py internal stn 60` shows `_decision_margin`
differing on 38/60 ticks (31 distinct values wired against 11 neutered) and `_metacog_rel` on 5/60,
so the audit's band-41-60 mechanism story was true. It is not a *second* faculty. Its evidence is
folded into `acc_conflict_monitoring_evc`, whose citation now includes `:740`.

### 4. pfc_gain_metalearning: 45 -> the row is gone. Same call, one flip in 200 ticks.

`gain = self.pfc_gain * (1.0 + ctl)` (`autonomous_loop.py:934`) is consumed at exactly one place,
`bias = self.pfc.top_down_bias(strength=gain)` (`:1000`). The learned `pfc_gain` is a scalar factor
of the same vector `pfc_goal_top_down_bias` neuters; a factor of a product feeding one consumer is not
a separate faculty from the product. `exec_refute_marg_pfcgain_given_pfcbias.txt` (200 ticks) reports
the residual honestly rather than assuming it away: pinning `pfc_gain` on top of the zeroed bias
still changes `_control_gain` on 112/200 and `_decision_margin` on 93/200, and it changes the
**selected action on 1/200 ticks** -- so by `charge_probe`'s `flips > 0` contract it is technically
separable, and `C vs A` is 149/200, identical to `B vs A`'s 149/200.

One action flip in 200 ticks, both solo (the audited `exec_pfcgain_200`) and marginal, is the exact
W4 degeneracy `docs/coverage/README.md:72-75` names: a long run until one divergence appears. The
audit was right to decline 61 on it; a row of its own on that evidence is the roster-padding hole the
same README calls the instrument's largest. Folded into `pfc_goal_top_down_bias`, whose citation now
includes `:1418`, the learning rule.

### 5. pfc_goal_top_down_bias 61 and acc_conflict_monitoring_evc 61 SURVIVE, and are now stronger.

Both were attacked on both available lines and held.

**Seeds.** `exec_refute_seeds_pfcbias.txt` and `exec_refute_seeds_conflict.txt` (both exit 0) run the
same neuters at seeds 1, 2, 3. Selected-action flips: pfc_bias **28, 7, 40** per 60 (seed 0: 39);
conflict **24, 4, 26** per 60 (seed 0: 23). Wide spread, never zero. The audited caveat is
discharged for these two rows: the charge is not a seed-0 artefact.

**Serial chain.** The audit kept these as two rows on a stated judgment call and offered 35.8 as the
merged alternative. I tested the merge instead of judging it.
`exec_refute_marg_conflict_given_pfcbias.txt` (exit 0) neuters `conflict` on top of an already-zeroed
goal vote: **4/60 selected actions and 28/60 trajectory positions still change.** Conflict has a
behavioural footprint that does not pass through the goal channel -- it runs through the STN brake and
the plan-nudge gain -- so the two rows are separable on evidence, not on preference, and the audited
two-row reading stands. The same test discriminates: it returns NESTED for stn (0/60) and SEPARABLE
for both conflict (4/60) and taskset (3/60, `exec_refute_marg_taskset_given_pfcbias.txt`), so a
NESTED verdict is a finding and not what the probe says about everything.

What survives with these two rows is nonetheless a thin thing, and the honest framing is: `C vs A` is
40/60 where `B vs A` is 39/60. Neutering the ACC monitor *in addition to* the goal vote barely moves
the total effect. Conflict's solo footprint is 23/60 actions; with the goal channel already removed it
is 4/60. (Those are different counterfactuals, so 23 - 4 is not a decomposition and I am not claiming
it is -- but the direction is unambiguous: most of what the two rows measure is one channel.) Two rows
is defensible because the marginal 4 exist. Two rows at 61 each is the upper end of what this evidence
supports, and if a future pass merges them the number is **30.9** on seven rows, not a higher one.

### 6. The audited methodological finding is confirmed, and the observable it recommends is sound.

`arm trajectory changed: n/N` really is `N - onset` in all seven audited artifacts, and the arm
integrates its own motion, so it is a restatement of *when* rather than *how often*. I confirmed the
replacement column is not the same artefact: `res["action"]` (`autonomous_loop.py:1632`, read back at
`brain.py:1823`) is the 7-way winner of `BasalGangliaGate.select` over
`ACTIONS = ("ENGAGE","RELEASE","HOLD") + _MOTOR` (`autonomous_loop.py:69`), i.e. the gate's decision,
not a position-derived quantity -- and it re-converges, which a position cannot. Every band above 40
in this file rests on that column.

### 7. An instrument defect this audit's own probe introduced. Reported, not worked around.

`coverage_score._fingerprint` matches every `n/N ticks` line, and `executive_ab_probe.py` prints its
eligibility count in that form. `exec_conflict_60` and `exec_evc_60` are the same experiment on all
three behavioural observables -- `0/60`, `47/60`, `23/60` -- but their fingerprints differ, because
the eligibility line reads `56/60` in one and `55/60` in the other. **So W3, the serial-chain
detector, does not fire on two artifacts that are the same run.** The audit disclosed the identity in
prose, so nothing was smuggled here; but the same template is used by `affect_ab_probe.py`,
`consciousness_ab_probe.py`, `social_tom_ab_probe.py` and `wb_refute_probe.py`, and in every one of
them a one-tick difference in an eligibility counter now defeats W3 for that category. The fix is in
`tools/**`, owned elsewhere this pass: either exclude the eligibility line from the fingerprint or
build the fingerprint from named observables rather than from every `n/N` in the stream.

**STATUS 2026-07-30: NOT FIXED, and the fingerprint widening of the same date made this half of the
defect slightly worse, deliberately.** `FLIPS` was widened from `N/M ticks` to any bare `N/M` counter,
which closed a different hole (probes printing `turns` fingerprinted as the empty tuple, so W3 could
not fire for them at all). But widening only ever makes a fingerprint **longer**, and this defect is
that a longer fingerprint has more places to differ: `exec_conflict_60` and `exec_evc_60` still do not
collide, for exactly the reason recorded above, and the widening adds nuisance counters to nineteen
other artifacts that could split a real collision the same way. Neither of the two fixes named in the
paragraph above -- excluding the eligibility line, or fingerprinting named observables -- was applied,
because both require the probes to agree on a name for their observables and that is a probe-side
change across six files. This remains the single largest reason W3 and W5 are a floor rather than a
census.

### 8. The CROSS-CATEGORY collision against Imagination: FALSE POSITIVE. `acc_conflict_monitoring_evc` stays 61.

`docs/coverage/Imagination.md:310-313` flagged that this file's `acc_conflict_monitoring_evc`
(`exec_conflict_60.txt`) and Imagination's `latent_world_model` (`imag_jepa_action_60.txt`) report the
same three behavioural numbers -- act/hold `0/60`, trajectory `47/60`, selected action `23/60` -- at the
same onset tick, and that W3 cannot see it because W3 runs inside `coverage_score.score`'s per-category
loop. Unlike the four adjudicated within-category warnings, **both artifacts exit 0 and both rows are
banded 61**, so these bands do rest on the colliding numbers. Adjudicated 2026-07-30 on new evidence,
`docs/coverage/artifacts/crosscat_jepa_vs_conflict_ab.py` recorded as
`crosscat_jepa_vs_conflict_60.txt` (exit 0): four arms at one seed -- A wired, B1 `jepa_prediction=0`,
B2 `conflict=0`, C both -- which is the B1-against-B2 comparison neither two-arm probe can make.
**Verdict: two separable mechanisms. No band changed in either file.** The full evidence is recorded in
`Imagination.md` under "The adversarial pass"; the four load-bearing items, from this file's side:

- **STATIC, different objects on different modules.** `executive_ab_probe.py:121` rebinds
  `ExecutiveControl.conflict`; `imagination_refute_ab.py:102` rebinds
  `JepaWorldModel.prediction_error`. Neither is a `_drive_w` key, and they enter the loop in different
  phases -- `_conflict` at `autonomous_loop.py:715` reaches the gate on the SAME tick via the STN brake
  at `:740`, while `prediction_error` at `:1314` is called after the winner exists and reaches
  behaviour only through `r_int` at `:1315` and a later tick's value table.
- **STATIC, the mechanism behind the coincidence, and it is a shared CHANNEL not a shared cause.** The
  jepa error feeds `r_int` -> the q-table -> the drive vector -> which is the argument of
  `self._exec.conflict(drive)` at `:715`. So the Imagination neuter changes this row's scalar's own
  output. Both perturbations arrive at one junction: the conflict/STN brake on the BG gate.
- **RUNTIME, decisive.** The two neutered arms are not the same run: `B1-B2` differs on **1/60 selected
  actions and 7/60 trajectory positions**. The withdrawn baseline's 9-point defect was byte-identical
  traces on 122/122 wake ticks (`COVERAGE_BASELINE.md:12`); this is not that. And both marginals are
  nonzero -- killing `conflict` on top of an already-zeroed jepa error still changes **5/60 actions and
  24/60 positions**, and the reverse likewise. That is the same separability test section 5 above used
  to keep two rows on a **4/60** marginal; this marginal is larger, both ways. The harness is trusted
  because it first reproduces `exec_conflict_60`, `imag_jepa_action_60` and `novelty_60` exactly.
- **RUNTIME, against the rows.** The coincidence is worse than section 6 recorded: not only are the
  counts equal, the flip **tick sets are byte-identical** (all 23: `14, 32, 39, 40 ... 60`) and both
  neutered arms select the same alternative action at each. Both perturbations flip one knife-edge
  decision at tick 14 -- where both signals had already been live for 9 ticks without mattering -- and
  then ride the same alternative trajectory. So `23/60` here and `23/60` in Imagination are **one**
  shared measurement of where this arm is fragile, plus a small separable residual. This row's 61 rests
  on the residual and on the multi-seed spread in section 5 (`24, 4, 26` per 60 at seeds 1-3), not on
  the shared 23. A future pass that wants to move this row should attack the residual.

## What I checked and left alone

- **Every citation in the table resolves to what it claims**, not merely to a line that exists (which
  is all `_cite_ok` can check). All 21 were opened: `prefrontal_executive.py:349` is
  `def top_down_bias`, `cortex/executive_control.py:80` is `def conflict`, `:149` is `def plan_sequence`,
  `task_set.py:159` is `def bias`, `:213` is the `_bad_streak >= patience` shift trigger, `:221` is
  `def shift`, `brain.py:1608` is the `taskset_a` drive term, `:1912` is `_task_pref =
  task_set.bias(feat)`, `:2401` is the once-per-night learn gate, `:2949` is `respond()`'s abstain
  branch, `autonomous_loop.py:1418` is the `_n_ovr >= 10 and _n_hab >= 10` guard on the `pfc_gain`
  update. No drift found.
  *(Disambiguation CORRECTED 2026-08-05, and it had been made worse before it was made better.
  `executive_control.py` then named two different modules — `cortex/executive_control.py` and
  `reasoning/brain_adapters/executive_control.py` (line counts this note used to carry had
  drifted twice and are dropped; the ambiguity itself is the record). A 2026-08-04 pass qualified
  this cite to the `reasoning/brain_adapters/` one; re-resolved, that file's `:80` was a numpy
  timing comment and its `:149` was `directory = os.path.dirname(directory)`. **`def conflict` is
  at `cortex/executive_control.py:80` and `def plan_sequence` at `:149`** — both exact. The bare
  cite was ambiguous-but-right; qualifying it to the wrong module made it unambiguous-and-wrong,
  which is the worse failure, because the next reader stops checking once a path looks specific.
  The ambiguity was RETIRED 2026-08-12: the adapter module was promoted+renamed to
  `cortex/competence_executive.py`, so the bare basename now resolves to exactly one module.)*
- **The rule-4 zero (`task_set_shifting`) and the three sub-40 rows are the audit's, re-read but not
  re-measured.** A refuter's mandate is the band-41+ claims; re-deriving conservative low bands would
  spend the budget in the direction that cannot inflate a score. If any of them is wrong it is wrong
  in the direction of being too low, and `task_set_shifting = 0` is corroborated in passing by
  `exec_refute_taskset_elig`: the S-R readout varies at every seed while the *set* never changes.
- **A gate problem in this directory that is nobody's category.** `python -m ruff check .` is one of
  the project's three gate commands and `docs/` is not excluded, so the probe scripts this sweep is
  adding are linted. `ruff check docs/coverage/artifacts/` currently reports **42 E501 and 1 F841**
  across the sibling probes written this pass. `exec_refute_probe.py` is clean under both
  `ruff check` and `ruff format --check`; the rest are not mine to edit, and whoever collects this
  ledger for a commit needs to fix them or the lint gate goes red on the evidence directory.
- **`response_inhibition_nogo` stays 35 with no artifact.** The audit's reason is structural, not a
  gap in effort: when `select()` vetoes, `winner is None` and there is no released channel to compare
  against, so the counterfactual is undefined rather than unmeasured. I looked for a formulation my
  three-arm harness could express and did not find one. It is not raised on the "the veto determines
  the executed action" argument, because that argument is available for every term in the drive
  vector.

## The arithmetic, both ways, so the row removals are auditable

`COVERAGE_RUBRIC.md`'s standing warning is that a row must never disappear to help a number. Two rows
disappeared here, so both totals are on the record:

- audited roster: 10 rows, sum 383, mean **38.3**
- taskset downgrade only, both duplicate rows kept: 10 rows, sum 367, mean **36.7**
- duplicate rows also folded in (this file): 8 rows, sum 277, mean **34.6**

The removals cost a further 2.1 points rather than buying any, which is what a de-duplication should
do to a roster whose duplicates were both above the mean. Both removed faculties remain named -- in
the surviving rows' `measured_by` and citations, in the sections above, and in the inventory below --
and their artifacts (`exec_stn_60.txt`, `exec_pfcgain_200.txt`) stay in the tree, so a later pass can
re-split them by refuting section 3 or 4 rather than by re-typing a band.

## Capability inventory (Policy 2 evidence: which file provides what)

- `cortex/executive_control.py` -- ACC Hopfield-energy conflict (`:80`), EVC arithmetic (`:118`),
  beam/tree sequence planner (`:149`). No learned weights, by design.
- `cortex/learned_evc.py:65` -- the dopamine-gated three-factor cost/value replacing `evc()`'s
  constants. Unique capability, no separable behavioural footprint (row 2).
- `cortex/prefrontal_executive.py` -- spiking WTA goal commitment (`:246`), latent-rollout planning
  (`:275`), top-down bias (`:349`), WM pool + goal stack (`:314`, `:333`) live only in the offline
  loops.
- `basal_ganglia/action_gating.py:99` -- the Go/NoGo gate that owns release. `stn` raises `eff_tonic`
  on every channel (`:109`), which is why it can only add caution, and why it moves
  `_decision_margin` only where the `max(0, ...)` clamp at `:96` bites.
- `cortex/task_set.py` -- n_sets S-R rule matrices in a persistent-activity LIF pool (`:159`,
  `:166`) plus an R-STDP shift vote (`:184`). The readout changes an internal read at every seed and a
  decision at half of them; the shift never fires.
- `cortex/metacognitive_monitor.py` via `brain.py:1851-1859` -- reliability, hold, explore,
  learn_gate. `_decision_margin` is its one executive input and reaches it (section 3).
- The learned control gain lives in `autonomous_loop.py:1418-1421`; the STN brake gain in `:745-749`.
  Both learn; neither is a separate faculty from the channel it scales.
- `cortex/autotelic_goals.py` is goal *generation*, reached only from
  `self_improvement_suite.census` and `reasoning/nightly_skills.py`, never from the live tick --
  WholeBrain's roster. Recorded here so it is not silently omitted from every category.

## Gaps, ranked by distance_to_100 x feasibility

1. **Make the task-set readout decisive, not merely present.** It is the one row this pass moved and
   the reason is that a 0.10-0.14 drive contribution is below the gate's resolution on half the
   seeds. The fix is leverage (scale the readout to the drive band actually being arbitrated, or let
   the held set gate the channel rather than add to it), not a bigger constant. Cheapest real
   point-mover, and it is now measured rather than guessed.
2. **Give the sequence planner leverage.** Fully live, learns its own weight, changes nothing
   (`exec_plan_60`, exit 1, 0/60 on all three observables). Either the plan competes for the goal
   itself or the row stays at 30.
3. **A reachable shift trigger for `task_set`.** `patience = 6` against a reward that is non-positive
   on 4/60 non-consecutive ticks is unreachable by about a factor of six. Per P1 the fix is a
   bad-evidence criterion relative to a running baseline, not a smaller 6.
4. **Fix W3's fingerprint (section 7) and W4's `max` over ratios.** Instrument work in `tools/**`
   that protects all eleven categories; until it lands, a 1-in-200 action count can present behind a
   38/200 trajectory count. **Partly addressed 2026-07-30:** `W5` now mirrors W3 ACROSS category files
   (section 8), which closes the structural blind spot but not the regex one -- the eligibility line
   section 7 names still defeats both checks, and `exec_conflict_60` and `exec_evc_60` still do not
   collide under `_fingerprint` for exactly the reason recorded there. W5's own comment block says so.
   **Second partial, 2026-07-30:** the `FLIPS` regex was widened to any bare `N/M` counter after a
   whole-ledger re-verification (all 82 artifacts re-executed). That closes the empty-fingerprint hole
   -- W3 can now fire for probes that print `turns` rather than `ticks`, and it immediately drew three
   Gf warnings and one new W5 group -- but it does **not** close the eligibility-counter hole this item
   is about; see the STATUS note appended to section 7. W4's `max` over ratios is untouched.
5. **Band 81+ remains structurally out of reach.** There is no held-out executive task in the tree,
   so nothing here can pass rubric rule 3 regardless of wiring. Multi-seed is *not* held-out
   evidence -- it is the same developer-constructed arm loop with a different random stream -- and
   the three seeds I ran do not lift anything above 61. `COVERAGE_RUBRIC.md:99-102` already declares
   the three-way arm decision as this category's ceiling; the surviving 61s sit on it.

## What I could not measure

- **Whether the surviving 61s are 61 or 80 within their band.** Four seeds is enough to kill a
  seed-0-only claim (it killed one) and not enough for a distributional argument.
- **Seeds for `pfc_gain` and the marginal tests.** The marginal experiments are seed 0 only, so
  "conflict is separable from the goal vote" rests on 4 flips in one 60-tick trajectory. It is the
  weakest surviving inference in this file and a future pass should re-run
  `exec_refute_marg_conflict_given_pfcbias` at seeds 1-3 before trusting the two-row split; if the
  marginal goes to 0 anywhere, the merge to 30.9 on seven rows is the honest outcome.
- **The NoGo veto's counterfactual.** Undefined, not merely unmeasured (above).
- **Sleep-side executive behaviour** beyond the single night a 200-tick run contains, and
  `respond()`'s abstain gate, which no tick-level probe reaches.
- **Anything at all about longer horizons.** Every figure here is 60 or 200 ticks, and
  `docs/coverage/README.md:72` records `drive 4` and `drive 60` returning opposite verdicts for the
  same faculty. Tick count remains a free parameter this instrument does not fix.

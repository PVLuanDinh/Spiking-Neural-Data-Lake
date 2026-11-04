# PRE-REGISTRATION PL1 -- does the one grower lose plasticity over a long non-stationary stream?

> **Status: FROZEN, pre-data (2026-11-04, integrator duo A, launcher D-208).** Frozen at the commit that adds this file
> (title without "(DRAFT, 2026-10-31)") and registers T127 in tasks.md. Its body is the draft
> lanes/research-1031/PREREG-PL1-plasticity-loss-1031.md (sha256 2be4917dca66923f...) copied verbatim except for this
> header, the title, and the new section 2a. The draft's reviews: policy-enforcer-28, -28b and -28c (0 MAJOR after 28c,
> launcher D-103; reports lanes/policy-enforcer-28*-REPORT.md), then duo B on round 3 (#7668, AGREE-WITH-FIXES) and on the round-4 fold (#7670, AGREE, all 5 closed).
> **Duo B is the ONLY independent reader of the round-4 fold** (recorded as B asked). Section 2a is the integrator's
> freeze fill and is read by duo B at the frozen sha before the pilot runs.
> - Nothing below has been run on seeds 31000 and up. Seed block claim: board #8078 (pilot 31000-31012, campaign
>   31100-31999).
> - Later changes are dated pre-data amendments appended at the end. `R` is filled into section 2a from the pilot by
>   section 2's derivation rule, before the campaign opens; nothing else in 2a changes after the pilot.
> - Base: origin/main 847706a7 (ARC 6 pushed). The draft's "Base: origin/main c4c10b28" line below is the draft's own.

Revised 2026-10-31 per policy-enforcer-28 (findings PL1-1 .. PL1-11). No hypothesis, delta or arm changed.
Revised round 2 per policy-enforcer-28b (PL1-2, PL1-6c, PL1-9, PL1-11, NEW-2, NEW-5, NEW-6, NEW-9). No hypothesis, delta or arm changed.
Revised round 3 per policy-enforcer-28c (ruling C futility; cap inflation from simulation; minor fixes). No hypothesis, delta or arm changed.
Revised round 4 per duo B review (board #7668 PL1, #7669 TAU1): one mde from SD_UCL at the seed count reached; futility clauses; outcome probabilities; TAU1 PC-sense block + cites.

Status: DRAFT, frozen only when committed on int3 by the integrator (row id assigned at compose; proposed T127).
Owner order 2026-10-31: "Pre-register step 1" (of lanes/research-1031/LACUNA-SNN-SURVEY-1031.md). Author: launcher-1028.
Scope: MEASUREMENT ONLY. No recycling mechanism is built or switched on by this row. Base: origin/main c4c10b28.

## 0. NULL-RECORD format (P8 amendment item 1)
`NULL-RECORD: PL1 units=<mean grown units at probe> ticks=<per run> tasks=<switches> seeds=<n> mde=<one-sided .05, power .8> retest=<state>`

## 1. Claim and why (literature -> SNDL)
Loss of plasticity (Dohare et al. 2024, Nature; Lyle et al.; Sokar et al. ReDo): a network trained for long on a
non-stationary stream learns a NEW task worse than a fresh network, via dormant units and representation-rank collapse.
SNDL's one grower is `Grower` (`growth/grower.py:1-13`, G3-T1), which WRAPS the percept engine
`NeurogenesisSelfImprove` through `PerceptProblem`; the measurement is taken through `Grower`/`PerceptProblem`, the
canonical entry. Each NG-SNN refit INHERITS the prior units (`ng_snn.py:468` -- `self.n_seeded = int(W.shape[0])  # units
inherited, not grown by this call`), candidates are contribution-grown, and `prune_unearned` (`ng_snn.py:329`, charged at
`neurogenesis_engine.py:738`) only adds and prunes; nothing revives a silent unit (tree check 2026-10-31).
`GrowthLoop` and `NeurogenesisSelfImprove` themselves retire at T6 (`growth/grower.py:4-5`); what survives is their refit
path as VALUES -- the design moves no refit code, so `_refit`, `NGSNN`, `candidate_lifecycle` and `placement` stay where
they are (`growth/grower.py:13-14`), which is where the measured mechanism lives.
P2 disclosure (engine re-evaluation 2026-10-31): `sleep/lifelong_consolidation.py` is a GROW <-> PRUNE <-> CONSOLIDATE
continual-learning loop with a non-regression (anti-forgetting) probe, built on `growth.growing_learner.GrowingLearner`,
not on `Grower`. It measures BACKWARD retention of old tasks; PL1 measures FORWARD plasticity on a new task. Not a
duplicate instrument; if step 2 is funded, that module's consolidate step is the P2 partner to check before any recycler.
The owner's goal is unbounded lifelong growth, so this is the failure to measure before growing further.
**H-PL1:** after a long task sequence, the aged grower learns a new task slower than a fresh grower, by at least delta.

## 2. Design
- Engine: the one grower through its canonical entry `Grower`/`PerceptProblem` (section 1), fed by the existing
  `SyntheticPerceptStream` (Markov over k modes).
  A TASK = one stream seed (its own mode features and transition matrix). Non-stationary sequence = T_age tasks of
  K ticks each, refit cadence unchanged.
- Arms, paired per seed s:
  - AGED: run T_age tasks, then the probe task P_s.
  - FRESH: a new engine (same config, same seed for init) on P_s only.
  - RESET-CONTROL (not the method; P8 item 2): the AGED engine's unit COUNT kept, every unit's weights
    re-drawn fresh, then P_s. Separates "capacity/size" from "state of the weights".
- Endpoint (one, the decision): learn-speed = mean held-out `predict_accuracy` over the first R refits on P_s
  (area under the early learning curve). D_s = learn-speed(FRESH) - learn-speed(AGED). delta = 0.05 accuracy.
- Descriptive only (no claim, so no multiplicity charge): active-unit fraction (unit fires on >=1% of held-out
  ticks), effective rank of the hidden state on a fixed held-out batch, mean |w| per grown unit, units grown.
- Parameters: K, T_age, k_modes, width are chosen by the integrator from the engine's current defaults and written into
  section 2a at freeze; none may change after the pilot. `R` is DERIVED from the pilot, not hand-picked: R = the first
  refit index at which the FRESH arm reaches 90% of its own final `predict_accuracy` on P_s (the point at which the early
  curve has settled), computed as a curve property of the pilot's FRESH arm and printed with that curve in section 2a
  before the campaign opens. The same R is used in every arm (AGED, FRESH, RESET-CONTROL) and every seed, so the endpoint
  is identical across arms.

## 2a. Parameter values (integrator at freeze, 2026-11-04; tree 847706a7)
Each value is the engine's current default where one exists; the two values with no default (K, T_age) are chosen
here with their reason. None changes after the pilot.
- **k_modes = 5**: `SyntheticPerceptStream(k_modes=5)` default (growth/neurogenesis_engine.py:136); chance = 0.2.
  Stream defaults kept too: peak 0.75, jitter 0.12, shuffle_transitions False.
- **width = 64**: the percept width is fixed by the stream, not a knob: `_mode_features` returns the
  `encode_multimodal` vector in [0,1]^64 (neurogenesis_engine.py:83-99). No grower width knob exists; the NG-SNN unit
  bound stays the engine default `l_max=None` (NGSNN derives it from the training rows; neurogenesis_engine.py:387-394).
- **Refit cadence = 20**: `NeurogenesisSelfImprove(refit_every=20)` default (neurogenesis_engine.py:386), unchanged
  (section 2: "refit cadence unchanged"). Engine defaults kept: buffer_cap 1000, eps_tol 0.28, T 15, growth_gate None,
  online_readout None. Drive = `SyntheticPerceptStream.drive` (observe, then one sleep tick, 1:1).
- **K = 1000 ticks per task**: equal to the engine's buffer_cap (1000), so each task's last refits train on a buffer
  holding only that task's rows (the age effect is carried by the inherited units, not by stale rows in the buffer), and
  each task gets 50 refits at cadence 20. The probe task P_s is K ticks in every arm.
- **T_age = 50 tasks** before the probe. Raising-ambition ladder: 20 / 50 / 200. 50 is picked: measured cost on this host
  (2026-11-04, stock grower, the stream above) is about 20 s per 1,000 ticks once the buffer is full, so one paired seed
  (AGED 51 tasks + FRESH 1 + RESET-CONTROL 1 = 53,000 ticks) is about 18 min serial; the pilot (13 seeds) about 4 h.
  The rung above (200 tasks, about 70 min per seed, days for a campaign cap) is ruled out by host budget, not by a
  policy; a null at 50 names that scale (P8 amendment: a null names its scale, and deletion waits for one 10x re-test).
- **Task seeds**: task i of seed s uses `SyntheticPerceptStream(seed = s * 1000 + i)` for i = 0 .. T_age - 1, and the
  probe P_s uses `seed = s * 1000 + 999`; the engine's init seed is s in every arm. Distinct per task, deterministic,
  and never inside another row's claimed block (31,000,000 and up).
- **n0 = 13** (recomputed at freeze with scipy.stats.chi2: ratio sqrt(12 / chi2.ppf(0.20, 12)) = 1.2398 <= 1.25;
  n = 12 gives 1.2546). Pilot seeds 31000-31012.
- **Cap inflation I = 1.0286** (computed at freeze, 2,000,000-path Monte Carlo, rng seed 20261104): Lan-DeMets
  O'Brien-Fleming, one-sided .05, looks at 25/50/75/100% of the cap; boundaries z = 3.7492 / 2.5359 / 2.0198 / 1.7208;
  familywise false-positive 0.0500; power at the cap 0.8000 with I = 1.0286 (0.7977 with the textbook 1.022, 0.7899
  with no inflation). Script: lanes/engine-1024/pl1-freeze/pl1_freeze_numbers.py. The full simulation guard (section 3,
  outcome probabilities under planted D = 0 and D = delta, with the real look roundings) still runs before data, with
  the cap set from the pilot's SD_UCL.
- **R**: OWED, from the pilot's FRESH arm by section 2's rule (first refit index at 90% of its own final
  `predict_accuracy` on P_s), printed here with that curve before the campaign opens.
- **Row id: T127.** **Pilot host:** the integrator host under HEAVY-RUN.lock (lanes/engine-1024/heavylock.ps1), not a
  lane; opencode lanes are paused (owner 2026-11-03).

## 3. Sample size (P1)
- Measured SD: none exists for this endpoint, so a pre-registered PILOT measures SD(D). Pilot seed count `n0` is
  DERIVED, not habitual (the amendment's incident): n0 is the smallest number of PAIRED seeds at which the 80%
  chi-square upper confidence bound on SD is <= 1.25 x the pilot's own point estimate, i.e. the first n with
  `sqrt((n-1)/chi2.ppf(0.20, n-1)) <= 1.25`. Computed with `scipy.stats.chi2`: **n0 = 13** (ratio 1.2398; n = 12 gives
  1.2546, above the bar), printed with the pilot's SD and the computed value in section 2a before the pilot runs.
  PILOT seeds from 31000 upward, n0 of them, assigned at freeze; pilot seeds are not reused in the campaign.
- Effect: delta = 0.05.
- Target: n* = ceil(((z.95 + z.80) * SD_UCL / 0.05)^2), with SD_UCL the 80% upper confidence bound of the pilot SD, so a
  lucky pilot cannot underpower the campaign; the point-estimate n* = ceil((2.486 * SD / 0.05)^2) is printed beside it
  for transparency only and does not set the campaign size.
- Sequential: 4 looks at information fractions of the CAP: 25/50/75/100% of cap = `ceil(0.25 cap)` / `ceil(0.5 cap)` / `ceil(0.75 cap)` / `cap` seeds, the rounding of every look registered at freeze (cap = ceil(I n*), so the
  100% look IS the cap). No look sits above the cap and no seed is added after an unregistered look.
- Spending function: Lan-DeMets O'Brien-Fleming, one-sided .05.
- Futility (ruling policy-enforcer-28c C, POLICIES.md P1 amendment item 2): non-binding (conditional power < .10), taken
  only at a look whose FIXED-SAMPLE mde at that look's seed count, computed from SD_UCL at the design's one-sided .05 and
  power .8, is <= delta. In practice only the CAP look qualifies, so looks 1-3 are EFFICACY-ONLY; a boundary-adjusted
  mde is reported beside each look as information and never gates futility or P8 item 4. The futility gate's mde is the
  SAME mde P8 item 4 then decides with (below); conditional power is computed under the claimed effect delta.
- Cap: ceil(n* x I), where the inflation I is DERIVED by the simulation guard as the smallest value giving power >= .8 at
  the cap under Lan-DeMets O'Brien-Fleming spending over the 4 looks (policy-enforcer-28c: the textbook 1.022 gives 0.798
  under LD spending, 2M-path Monte Carlo); I is printed at freeze and the look sizes use it. Power at the cap stays >= .8, and the simulation guard prints power at the cap
  and at each of the 4 looks (at the 100% look the last-look and at-cap numbers are the same, and both are printed).
- mde (one definition, P8 item 1): the fixed-sample, one-sided .05, power .8 mde AT THE SEED COUNT REACHED, from SD_UCL
  (the same mde the futility gate uses -- one mde, never two). P8 item 4's
  INCONCLUSIVE decision uses ONLY this mde; a boundary-adjusted mde (stricter at an interim OBF look) is reported beside
  each look as information and never decides item 4.
- Simulation guard before data: planted D=0 keeps the FAMILYWISE false-positive rate <= .05 across the 4 looks; planted
  D=delta crosses with power >= .8; and (POLICIES.md P1 amendment item 2, :193) under BOTH planted D=0 and planted
  D=delta it prints the probability of EVERY outcome: efficacy stop at each look, futility stop, uncrossed cap, and
  INCONCLUSIVE.
- An uncrossed cap or futility stop reports its mde; P8 item 4 decides (INCONCLUSIVE unless mde <= delta). Campaign seeds
  31100 upward.
- One endpoint, so no multiplicity correction.

## 4. What P8 requires of an invented method
1. Reachable RED (the measurement fails when it should):
   - PC-detect: plant dormancy -- zero the input weights of 50% of an aged engine's grown units. The dormant-unit
     detector must flag >= 90% of planted units (denominator = planted units) AND <= 5% of the AGED engine's NON-planted
     grown units (denominator = non-planted grown units scored; both counts are reported with the two rates). Both halves
     are measured on the aged engine, so the specificity half is not vacuous -- a fresh engine has ~0 grown units and a
     "5% of a fresh engine" bar could never go red. Fails -> the descriptive metrics are not reported.
   - PC-probe: plant stiffness -- AGED arm with refits disabled on P_s. The probe must separate it from FRESH at a stated
     bar: D_pilot(refits disabled) >= 2 x delta AND >= the pilot's own mde (one-sided .05, power .8, fixed-sample pilot
     seeds). That proves the learn-speed probe can read a plasticity loss at all. Fails -> PL1 halts, instrument broken
     (the halt rule is named here, not inferred).
   PC-detect runs before the pilot; PC-probe is read from the pilot's own seeds. Both are logged before the campaign
   opens.
2. Control that is not the method: FRESH (twin) and RESET-CONTROL (size held, weight state removed).
   - Chance floor, already built and used as the engine self-check's must-fail control
     (`neurogenesis_engine.py:1339-1341`, claim 2 "CONTROL THAT MUST FAIL"), reused here as the campaign floor (P8 item 2
     / P3): `SyntheticPerceptStream(shuffle_transitions=True)` (`neurogenesis_engine.py:126-137`: the next mode is drawn
     independently of the current one, so a grower fed by it can only ever reach chance) runs on BOTH arms. It is a floor
     check, not a second decision endpoint.
3. Attribution: if H-PL1 holds, RESET-CONTROL apportions it: AGED-vs-RESET = weight state; RESET-vs-FRESH = size.
4. Disclosure: synthetic stream, not the live brain's percepts (engine-level claim only); the grower's own
   contribution criterion can prune silent units, which can MASK dormancy -- unit counts are reported per arm.
5. Disposition:
   - H-PL1 crosses: fund step 2 (recycling as a mode of the one grower, its own pre-registration; P2: never a second grower).
   - Null (powered, D < delta): write the NULL-RECORD into `specs/022-unified-brain-arc/evidence/pl1-plasticity-loss.md`
     (the evidence file, named here; precedent `evidence/e1b-rpe-harm.md`). Named 10x axis = TASK-SEQUENCE LENGTH, stated
     in TICKS so it is a learning-rule axis: T_age x K (total ticks per run) >= 10 x the NULL-RECORD's `ticks=<per run>`.
     The ONE re-test that follows is itself a pre-registration, with its own "Sample size (P1)" section (its own pilot,
     looks, spending function, futility, cap), its own NULL-RECORD, and THE SAME endpoint, bar and control as this row --
     no lighter re-test.
   - The null's obligations are the NULL-RECORD and "funds nothing": PL1 is measurement-only, so nothing is built that a
     `scale-retest` row could name, and no such row is written here (a row naming no module is a phantom EXEMPT row,
     `dormancy_gate.py:1153-1176`; the instrument keeps its own `"permanent"` row, section 5). A `scale-retest` row
     attaches only if the step-2 recycling prototype already exists on the live path at null time, and then it keys that
     prototype -- not the instrument -- naming the NULL-RECORD and `since=<the arc's base on origin/main>` and expiring
     before the next pushed arc (`null_scale_gate` as called at `dormancy_gate.py:1185-1196`).
   - Underpowered: INCONCLUSIVE, funds nothing; item 4's decision uses the one fixed-sample mde defined in section 3,
     never a boundary-adjusted one.
   - A measured HARM direction (aged learns FASTER by > delta) is reported as a finding, not a defect: descriptive only,
     no bar and no test, so it cannot become a second decision endpoint.

## 5. P3 / build rules
The instrument lands as `instruments/plasticity_loss.py` (name checked with graph-exists at build), registered in
REGIONS + run_selfchecks, self-check = PC-detect on a tiny engine. Dormancy disposition is the gate's own token for a
governed harness, `"permanent"` (precedent: `instruments/rpe_charge`, `dormancy_gate.py:130-139`), and it carries
rpe_charge's re-decide clause: "Re-decide after the campaign: a harm disposition deletes the rule it measured, not this
harness." Its EXEMPT row therefore states a real retire condition (re-decide at the campaign's close), not the label
"campaign instrument". No brain-lake writes (scratch NORD_HOME). No envelope files. e-prop untouched.

## 6. Raising-ambition record (self-apply)
Axis: Depth. Picked rung: reproduce the field's central lifelong-learning failure on SNDL's own grower with the
expert's instrument (Dohare's fresh-vs-aged probe) plus positive controls. Next rung up: the same probe on the LIVE
brain's own percept stream over days -- ruled out now by P4b (no live-lake writes for a campaign) and the daemon /
host gates; it becomes the re-test venue if PL1 crosses. Outside judge: FRESH-twin learning curve on held-out ticks.

## 7. Not decided here (integrator at freeze)
Section 2a parameter values (K, T_age, k_modes, width); `R` is NOT one of them -- it is filled into 2a from the pilot by
section 2's derivation rule; exact row id; whether the pilot runs on an opencode lane or the integrator host.

## Amendments (dated, pre-data)

### Amendment 1 (2026-11-04, integrator duo A; duo B's minors on #8082, AGREE on the freeze at 69a838d5)
Pre-data: nothing has run on any PL1 seed. No hypothesis, arm, endpoint, delta, value or seed changes; three
statements in section 2a are corrected.
- m1 (the K = 1000 reason). Section 2a says each task's last refits train on a buffer holding only that task's rows.
  That holds at the task's FINAL refit alone, and even that buffer carries ONE cross-task row: `observe()` pairs the
  previous percept `_prev` with the current category (growth/neurogenesis_engine.py:533-537), so the first row of a task
  pairs the previous task's last percept with the new task's first category. Earlier refits in the task still hold
  rows of the previous task. The reason for K = 1000 stands as "a task fills the buffer once", not "the buffer is pure".
- m2 (seed range). The derived stream seeds `s * 1000 + i` (and `s * 1000 + 999` for the probe) span
  31,000,000-31,999,999 for s in 31000-31999. Board #8078 claimed only 31000-31999 (the engine-init seeds s); the
  claim now also covers 31,000,000-31,999,999 (board post on thread PL1, this date).
- m3 (wording). Section 2a's "None changes after the pilot" reads: nothing in section 2a changes after the pilot
  except `R`, which the pilot fills by section 2's rule.

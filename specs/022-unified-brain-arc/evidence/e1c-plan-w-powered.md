# E1c -- the powered production test of `_plan_w` harm (H-P1), owed by E1b's INCONCLUSIVE

> **Status: FROZEN, pre-data.** Frozen at the commit that dropped "(DRAFT pre-registration)" from the title, after
> policy-enforcer AGREE #5641. Nothing has been built or run in a brain. The design numbers in section 5 come from a
> brain-free resampling of E1b's recorded production data. Later changes are dated pre-data amendments, appended below
> section 12's heading. Any other change is a new pre-registration on a fresh block.
> - **Task:** T122 [E1], registered in tasks.md by the freezing commit.
>   Board claims #5612 (this file), #5613 (seeds 21000-21999), #5614 (T122).
> - **Go:** the owner's words in session A's chat on 2026-10-28, "_plan_w first", recorded at #5609. The button the
>   owner chose read: "Run the powered _plan_w follow-up (~485 seeds) first, then G1(c)-2."
> - **Owner ruling it inherits:** #4616 (2026-10-23), "the production tier alone decides a `_plan_w` deletion".
> - **Read against:** int3 27444d33 (origin/main c1df3887 plus 8 docs commits; the brain code equals c1df3887).

## 0. Raising-ambition record (self-apply, owner 2026-10-25)

- **Axis:** Depth. The job is to answer one owed question excellently, not to widen it.
- **Ladder.**
  - **R1:** size the cap the way E1b did, on a normal model, from the point SD 0.3097 (fixed n 485).
  - **R2:** size and check the design on the MEASURED shape. Resample E1b's 80 production differences through the
    harness's own decision function, with the decision rule's sign conjunct included, and size the cap at the
    pessimistic end of the SD's uncertainty.
  - **R3:** explain the heavy tail before the run (which seeds flip regime, and why), then stratify on that cause.
- **Pick: R2.**
  - R3 is ruled out. Strata chosen from the same 80 looked-at seeds would tune the design to data already seen, and
    P1 forbids design decisions taken after an unregistered look.
  - The owner's words also scope this row to the owed powered run.
  - A descriptive tail census is reported beside the result, on the fresh seeds, and is used for nothing.
- **Range.**
  - Clinical-trial group-sequential design: Lan-DeMets O'Brien-Fleming spending with non-binding futility. Kept from
    E1b.
  - Robust statistics: excess kurtosis is 6.2, so a Hodges-Lehmann shift is reported beside the t (no bar).
  - Adversarial design review: look for a decision rule that cannot fire at the planned n. This review FOUND one: the
    scaled sign conjunct (section 5.2).
- **Outside judge** (none of these is the method itself):
  - the `pfrozen` twin;
  - the A/A repeat arm;
  - E1b's brain-free compare fixture, where `pfrozen = real + 0.1` must read MET;
  - the resampled-null type I of the exact decision rule.

## 1. Question, and what a result licenses

**Question.** On the production tier, at E1b's scale (600 ticks, one synapse), does the learned `_plan_w` trajectory
lower held-out reward by at least DH = 0.035 against the same brain with `_plan_w` held at its init?

Background:
- E1 recorded t +2.82 on the no-encoder tier.
- E1b's production H-P1 stopped for futility at n = 80 with t -0.100, SD 0.3097 and mde 0.1275 > 0.035. That is
  INCONCLUSIVE (P8 null-scale item 4).
- POLICIES P8 amendment item 4: an underpowered result "licenses no deletion at any scale until a run with enough seeds
  (power .8 for the claimed effect) is done". **This row is that run.**

**What it can license:**
- Exactly one of: harm, powered no-harm, or INCONCLUSIVE (section 8).
- It says nothing about the no-encoder tier, about E1's signal on that tier, about ValenceLearner, or about any scale
  other than 600 ticks.
- It cannot write a NULL-RECORD for `_plan_w`, because the benefit endpoint E-P2 stays unpowered (section 4).

## 2. Territory (int3 27444d33; the brain code equals c1df3887)

- **The update.** `AutonomousArm._pfc_bias` runs `_plan_w <- clip(_plan_w + _plan_lr * _rpe * (_plan_elig -
  _plan_base), 0, 1)` and nudges the first planned action by `_plan_w * gain`.
- **Its modulator.** `_rpe` is written only by `AutonomousArm.step`, from `ValenceLearner.learn`. Critic and actor run
  in series (board #5606; the P2 reading is under review).
- **The pfrozen arm.** It holds `_plan_w` at its init 0.1 (`_plan_lr = 0`), as E1b section 1 defines it.
- **E1b production H-P1, n = 80.** The d_s values are reproduced exactly by the design measurement's instrument check
  (mean -0.00347, SD 0.3097, 35/80 > 0).
  - Median 0.0. Five exact zeros: the arms never diverged.
  - Robust SD (1.4826 x MAD) 0.1399. Excess kurtosis 6.16.
  - Eight seeds have |d| > 0.4. The largest |d| is 1.338 (seed 27).
  - Bootstrap SD: 95% CI (0.206, 0.400); 80% upper bound 0.345.

## 3. Arms and runs (production tier only, #4616)

- **Arms:** `real` and `pfrozen`, exactly as in E1b section 2, at E1b's harness settings (N_TICKS 600, held-out mask
  from tick 300, the encoder pins ENC, the perception packs).
- **One tree.** Every run is on ONE tree sha, recorded before the first brain run. A tree that moves VOIDs the campaign.
- **Controls, at look 1 only, as E1b section 4:**
  - `repeat`: an A/A twin of `real` on seed 21000, with 0 action flips and recorded series identical to `real` 21000;
  - `pc` on seeds 21000 and 21001, with flips > 0.
- **No in-brain planted harm exists for H-P1.** E1b did not build one either. H-P1's positive control is the brain-free
  compare fixture (section 7, R1). Disclosed under P8 item 4.
- **Validity.** E1b section 4's production items, unchanged. A failed item VOIDs the campaign and the report says so.
- **The no-encoder tier is not run.** It holds no decision (#4616), and running it would double the cost.

## 4. Endpoints

- **H-P1 (primary, harm).** Per seed, d_s = r_pfrozen(s) - r_real(s), where r is the mean held-out reward from tick 300
  (`rpe_charge.mean_r`). Claimed effect DH = 0.035 (E1b's frozen literal). The decision rule is in section 5.
- **E-P2 (beside, benefit).** The same differences read the other way, against DR = 0.0131. It is reported with its mde
  and decides nothing.
  - At SD 0.31 a powered E-P2 needs about 3,500 seeds. It is therefore expected to stay INCONCLUSIVE, so no NULL-RECORD
    for `_plan_w` can follow from this row.
- **Beside, no bar** (each look's verdict line; delegate D-12b, launcher-1028):
  - the per-seed sign count k/n;
  - the 10%-trimmed-mean contrast (10% cut from each tail of d_s). With excess kurtosis 6.2, a crossing carried by a
    few outlier seeds must be visible beside the t. The builder graph-exists-checks for an existing trimmed mean
    before writing one;
  - the Hodges-Lehmann shift with its 95% interval;
  - the count of exact-zero seeds;
  - the descriptive tail census (|d| > 0.4 and the first divergence tick of those seeds).

## 5. Sample size (P1)

### 5.1 The design measurement (brain-free, before the freeze)

- **Script and logs.** `derive_e1c.py` and its two logs, in the integrator's work folder `lanes\engine-1024\e1c-design`
  (outside the tree). It imports the harness from the int3 tree and asserts that the import is brain-free (no torch).
- **Instrument check first.** It reproduces E1b's 80 production d_s exactly (mean -0.00347, SD 0.3097, n = 80) and
  refuses to continue otherwise.
- **The walk.** Each campaign resamples those 80 values with replacement. It walks
  `rpe_charge.seq_decision(ds, DH, looks, sd_pre=0.3097)` look by look until the first stop.
- **Shapes resampled:**
  - null: d - mean(d);
  - shift: d - mean(d) + 0.035;
  - shift_up: (d - mean(d)) x 1.115 + 0.035, which puts the SD at the bootstrap 80% upper bound 0.345.
- **Runs:** 2,000 campaigns per cell, with seeds from zlib.crc32 of the cell name. The SE of each rate is at most about
  .011.

| cap (10 or 12 looks of 64) | harm rule | null: harm rate | shift: power | shift_up: power | E[n] under shift |
|---|---|---|---|---|---|
| 512 | t and k >= ceil(5n/8) (E1b frozen) | 0.0000 | 0.169 | 0.085 | 474 |
| 512 | t and k >= ceil(4n/8) | 0.0525 | 0.815 | 0.738 | 383 |
| 512 | t alone | 0.0450 | 0.823 | 0.725 | 381 |
| **640** | **t and k >= ceil(4n/8)** | **0.0465** | **0.886** | **0.812** | **436** |
| 640 | t and k >= ceil(5n/8) | 0.0000 | 0.137 | 0.065 | 595 |
| 640 | t alone | 0.0525 | 0.872 | 0.818 | 444 |
| 768 | t and k >= ceil(4n/8) | 0.0410 | 0.919 | 0.877 | 488 |
| 768 | t and k >= ceil(5n/8) | 0.0000 | 0.133 | 0.050 | 710 |

**The table above is SUPERSEDED for the design by 5.1b** (policy-enforcer #5635 M1). Its shifted shapes also moved the 5
exact-zero seeds, where `real` and `pfrozen` never diverged. A harm carried by `_plan_w` cannot move a seed whose two
arms are bit-identical. The table is kept as the record of what was first measured.

### 5.1b The design measurement on a zero-preserving shape (`derive_e1c-zp-2000.log`, 2,000 campaigns each)

- **The shapes.**
  - null0: the zeros stay at 0, and the 75 diverged seeds are centred so the overall mean is 0.
  - zp_shift: the diverged seeds are shifted by 0.03733, so the OVERALL mean harm is exactly DH = 0.035. This matches
    H-P1's estimand, the mean of d_s over all seeds.
  - zp_up: the same, with the diverged spread x1.115 (pooled SD 0.345).
- **Rule:** k >= ceil(4n/8).

| cap (looks of 64) | null0: harm / cap or futility powered / INCONCLUSIVE | zp_shift power (SD .310) | zp_up power (SD .345) |
|---|---|---|---|
| 640 (10) | .0020 / .9360 / .0620 | .8345 | **.7875** (below .8) |
| **704 (11)** | **.0050 / .8285 / .1665** | **.8800** | **.8335** |
| 768 (12) | .0030 / .6945 / .3025 | .8985 | .8540 |

- Policy-enforcer #5635 independently found .839 / .8055 at 640 (its own walk). That agrees within 2 SE.

### 5.2 Finding: E1b's scaled sign conjunct cannot fire on this shape

- **The rule.** `seq_decision` declares harm only when t crosses AND k >= ceil(K_MIN x n / 8) seeds are positive, with
  K_MIN = 5. At n = 704 that asks for 62.5% positive seeds.
- **What this shape gives.** A true mean harm of exactly DH makes only about 52.5% of seeds positive on the
  zero-preserving shape (58.75% on the superseded all-shift shape). The conjunct therefore fails more often as n grows.
  On the all-shift shape power was 0.169 at 512, 0.137 at 640 and 0.133 at 768, never .8 at any cap. On the
  zero-preserving shape it can only be lower.
- **Consequence.** A follow-up that keeps the conjunct can never be "a run with enough seeds (power .8)" under P8 item 4.
- **Scope of the finding.** It is about this design at large n. It does not impugn E1b's verdicts: E1b's H-P1 stopped
  for futility at n = 80 on conditional power, never at a crossing the conjunct could have blocked.

### 5.3 The design (frozen at the freeze commit)

- **Looks:** every 64 seeds, n = 64, 128, ..., 704 (11 looks).
- **Cap: 704** (re-picked on 5.1b; enforcer #5635 M1). It is the smallest 64-multiple cap whose zero-preserving power at
  the 80% upper SD clears .8 by at least 2 SE: .8335, with SE about .0083. 640 gives .7875.
  - Power at the measured SD is .880.
  - Harm rate on the zero-kept null is .005. The direction guard makes the test conservative there, and that is
    disclosed.
- **Boundary:** `rpe_charge.seq_boundaries` (Lan-DeMets O'Brien-Fleming, one-sided alpha .05) on those looks. The
  per-look t is 7.627, 4.639, 3.644, 3.094, 2.736, 2.480, 2.285, 2.130, 2.002, 1.896 and 1.804 (`looks_e1c-704.log`).
- **Harm at a look** = t above the look's boundary AND k >= ceil(4n/8), meaning at least half of the seeds are
  positive. This is the large-n form of E1b's direction guard.
  - On the zero-preserving shape it costs 2.5-3.3 points of power against t alone (enforcer #5635: .839 against .872 at
    the measured SD).
  - In exchange it refuses a mean carried by a few regime-flip seeds against the majority (R4b).
  - **(Open, section 11: D-1.)**
- **Futility: P1 item 2's rule, which OPENS at n = 640.** `futility_open` (sd_pre 0.3097) is True only at 640, where
  the mde is 0.0335 <= 0.035. It is non-binding, with conditional power < .10.
  - Under the zero-kept null, 71.7% of campaigns stop there POWERED, 16.7% stop there underpowered (INCONCLUSIVE),
    and 11.2% reach the cap powered.
  - The 10-look design of the first draft never opened. The record below is kept.
  - **(First draft, 10 looks / cap 640; superseded by the 704 design above.)** The bound is the policy's. Delegate D-12c
    asked for a rule that opens at any look >= 5. That rule was measured, and the delegate then reversed D-12c, keeping
    P1 item 2's rule.
    - POLICIES P1 amendment item 2: "a futility stop is taken only at a pre-registered look whose mde, computed from
      the pre-registered SD at the design's alpha, is at or below the claimed effect ... before such a look the
      campaign continues". `futility_open` implements exactly that rule, and at these looks no look before the cap
      qualifies.
    - An opening rule, conditional power < .10 at any look >= 5, was measured (`derive_e1c-fut5-2000.log`, 2,000
      campaigns each):
      - power .886 / .808 (shift / shift_up), so it keeps the .80;
      - but under the no-harm shape E[n] 451, and **81.6% of campaigns end at an underpowered futility stop**
        (INCONCLUSIVE). Only 13.7% end "powered, no harm", against 89.4% with the policy bound.
    - From look 8 (`derive_e1c-fut8-2000.log`), power at shift_up is .784 < .80.
    - So the opening rule is barred by P1 item 2. It would also turn the likely no-harm answer into INCONCLUSIVE,
      which is the result this run exists to replace.
- **The stop's "powered" read uses the OBSERVED SD.** At 704 the mde is 0.0309 at SD 0.3097, so an observed SD above
  about 0.35 makes the cap read INCONCLUSIVE. The rates are in 5.4.
- **Powered.** Read as E1b addendum 3 reads it: the mde at the stopped n with the observed SD, at or below 0.035.
- **Simulation guard (R5).** Before look 1, the frozen harness reproduces 5.1b's 704 row within 2 SE at 2,000
  campaigns, on the zero-preserving shapes.
- **Validity read: the exact-zero rate per look.** E1b had 5 of 80 (6.25%). If a look's cumulative exact-zero fraction
  exceeds 12.5%, the mean effect per diverged seed that the cap was sized for no longer holds.
  - The record then prints the zero-preserving power recomputed at the observed zero rate, beside the verdict.
  - The verdict rule is not changed, and no seed is added.

### 5.4 What each shape ends in under 5.3 (cap 704, 2,000 campaigns each, `derive_e1c-zp-2000.log`)

| zero-preserving shape | harm MET | futility, powered | futility, INCONCLUSIVE | cap, powered | cap, INCONCLUSIVE |
|---|---|---|---|---|---|
| null0 (mean harm 0) | .0050 | .7170 | .1665 | .1115 | .0000 |
| zp_shift (mean +0.035) | .8800 | .0395 | .0085 | .0715 | .0005 |
| zp_up (mean +0.035, SD .345) | .8335 | .0120 | .0680 | .0590 | .0275 |

- About one no-harm campaign in six ends INCONCLUSIVE at the futility look. That is the disclosed residual of
  reading "powered" on the observed SD (P8 item 4), not a red.
- The first draft's all-shift rates (cap 640, `derive_e1c-detail-2000.log`) are superseded by this table.

## 6. Seeds

- **Confirmatory block:** 21000-21703, in order. 21704-21999 stay unexposed for a repair.
- **Grep check, 2026-10-28, int3.** 0 seed/block/range hits for 21000-21999 in evidence, src, tools/*.py and tests.
  Board claim #5613 holds 21000-21999.
- **Not reused:** E1b's seeds 8-183, E1's 0-7, and the derive RNG seeds 8801-9595 (simulation streams only).
- **Internal sub-seeds (cross-campaign note, launcher check).** `engine/brain.py` builds `GridPlaceMap(seed=self._seed)`,
  and each grid module mi seeds its cells with run seed + 1000 x mi.
  - Within E1c, offsets of 1000 or more exceed the block's width (704), so no two E1c runs share a stream.
  - Across campaigns, E1c's mi = 1 streams (22000-22703) cover L1b-STALE's run seeds 22000-22039. Each campaign's
    paired test compares arms of the SAME seed, so neither result is affected. This is disclosed, and no block moves.
  - `sleep_replay`'s accuracy stream uses the constant seed_base 100 (`attractor.accuracy(self._patterns, 100)`), not
    the run seed, so no run-to-run collision arises there.
- **Repeat control:** seed 21000. **Planted controls:** E1b's PSEEDS (brain-free replay seeds), unchanged.

## 7. Reds (each constructed and RUN before look 1; a red that does not fire halts the campaign)

- **R1, compare fixture** (E1b section 4, brain-free), at n = 704 under section 5.3's rule:
  - `pfrozen = real + 0.1` on every seed must read MET;
  - `pfrozen = real` must read NOT MET;
  - a failed planted control must read VOID.
  If any of the three prints differently, the harness is blind.
- **R2, in-brain sensitivity** (look 1): `pc` flips > 0 on seeds 21000 and 21001.
- **R3, repeat** (look 1): the A/A pair on seed 21000 has 0 action flips and identical recorded series.
- **R4, rule-change red.** Fed the 80 E1b differences shifted by +0.035 and tiled to 704, the k >= ceil(5n/8) rule
  must read NOT MET while k >= ceil(4n/8) reads MET. This proves the rule actually changed in the harness.
- **R4b, the guard refuses what it exists to refuse** (enforcer #5635 m1). At n = 704, take 30% of seeds at +0.30 and
  70% at -0.04 (mean +0.062, so t crosses). It must read NOT MET under k >= ceil(n/2) and MET under t alone.
- **R5, simulation guard.** As in 5.3, on the zero-preserving shapes.

## 8. Disposition (fixed in advance; the production tier decides, #4616)

- **HARM MET** at a look:
  - The learned `_plan_w` trajectory is a measured harm on production at 600 ticks. P8 null-scale item 4: "a measured
    harm ... is deleted without a re-test".
  - The integrator writes the deletion row **(route (iii), enforcer #5635 M2; P1 no-hardcode).** The `_plan_w` update
    is deleted, AND the plan nudge `bias[...] += _plan_w * gain` leaves the live path with it.
    - Holding `_plan_w` at its init 0.1 would leave a hand-set constant arbitrating behaviour on a live path, and none
      of P1's exceptions applies.
    - Any replacement (for example a running-baseline form) comes back only through its own pre-registered row.
    - Until then `_planned_seq`'s nudge is off the live path. If that leaves the planner's sequence output with no
      consumer, it takes a `dormancy_gate.EXEMPT` entry naming that row (P3, "awaiting a prerequisite").
    - **Disclosed:** the deleted state (no nudge) is NOT the `pfrozen` arm (a nudge at 0.1), and E1c does not
      measure it. The replacement row's pre-registration measures it against the planner-off and planner-on states.
  - The owner is told before it lands. `_plan_w` is not an owner-drawn P4d part.
- **NOT MET at a futility or cap stop, POWERED** (mde <= 0.035):
  - "Powered: no harm of 0.035 or more on production at 600 ticks." `_plan_w` stays as it is.
  - E1's no-encoder +2.82 is recorded as not replicated on production at power.
  - No NULL-RECORD, because E-P2 is unpowered (section 4). This is a non-harm result, not a null of the method's
    charge.
- **NOT MET, UNDERPOWERED** (mde > 0.035; the design makes this rare): INCONCLUSIVE, with its mde recorded. Nothing
  changes.
- **VOID** (validity, a moved tree, or a red that did not fire): no verdict. Repair, or open a new pre-registration on
  21704 and up.
- **Refused in every branch:** "`_plan_w` charges reward" (E-P2 is unpowered); any no-encoder or cross-tier wording;
  any scale other than 600 ticks.

## 9. Build rows (lanes, one builder at a time under D-13 REVISED; graph-exists before each; no new mechanism)

- **B1.** The two carried harness fixes from E1b:
  - print `null_precondition` in `cross`;
  - the E-P1 denominator (look-1 count over the look's seed count).
- **B2.** An E1c campaign spec in `instruments/rpe_charge.py`:
  - production tier, arms `real`/`pfrozen`, the seeds, looks, cap and K_MIN of section 5.3;
  - a parameter of the existing sequential path, not a copy (P2);
  - R4 as a test.
- **B3.** The campaign driver: reuse E1b's run tree pattern (detached tree, one arm per subprocess) and `resource:timing`.
  - Run graph-exists first.
  - EXTEND the incumbents for the throttle: `lanes\engine-1024\battery-stall-watch.ps1` and E1b's `ram.ps1` /
    `procs.ps1`. No second watcher mechanism (P2).
  - Per-run results are written to disk as `results/production/<arm>_<seed>.json` as each run finishes, and a resumed
    driver skips the runs already done, as E1b's did on 2026-10-27. A restart loses only the run in flight.
- **Commits.** One change each. The enforcer reviews each build, and nothing runs before the freeze commit.

## 10. Cost and resources

- **Worst case:** 704 seeds x 2 arms = 1,408 runs, plus 3 control runs.
- **E1b's measured throughput:** 534 production and no-encoder runs in about 3.1 h of wall time, about 21 s per run.
  That puts the worst case at about 8.2 h on an unloaded host.
- **Resources.** `resource:timing` is held for the whole campaign. No battery or other brain campaign runs beside it.
- **Host (no reboot is possible; owner 2026-10-28):** delegate D-13 REVISED applies (launcher-1028,
  `lanes\DELEGATED-DECISIONS-1028.md` Round 4).
  - **ONE run at a time.** Each seed's `real` and `pfrozen` run back to back.
  - **Pilot.** The first run of the campaign is the pilot. Its wall time T sets this campaign's `faulthandler_timeout`
    and stall-watch thresholds as multiples of T. A run that times out is re-run once before it counts. A timeout under
    paging is an instrument reading, not a verdict.
  - **Hysteresis.** The driver does not launch a run while Available < 1.5 GB or commit > 85%. It resumes only once
    Available >= 2.5 GB and commit <= 80%. A pause is logged, never a verdict.
  - **The nonpaged pool is logged at every look.** A rise of 1 GB or more since the previous look pauses the campaign,
    and the integrator tells the launcher.
  - **Load and determinism (#5641 m1).** R3's A/A repeat runs under the campaign's own load, one run at a time, so the
    repeat floor is measured where the arms run. If a later ruling allows concurrent runs, R3 is first re-run at the
    maximum worker count beside other live workers, with each seed's two arms in the same launch batch. If it flips,
    the campaign stays at 1 worker and says so.
  - **P1 declared exceptions (#5641 nit 2).** The throttle constants are operational: 1.5 / 2.5 GB, commit 85% / 80%,
    the timeout multiples of T and the 1 GB pool rise. They arbitrate scheduling, never brain behaviour, and no
    endpoint reads them.

## 11. Settled before the freeze

Policy-enforcer #5635 (CHALLENGE) was folded. Its delta AGREE is #5641 (0/0/1, 2 nits, folded here). Delegate D-12b
accepted D-1. The items below stay as the record of what was open.

- **D-1. The harm rule's direction guard.**
  - Proposed: k >= ceil(4n/8), "at least half positive".
  - Alternatives:
    - (a) keep E1b's k >= ceil(5n/8). The power ceiling stays near 0.14, so the run can never be the powered run
      item 4 asks for;
    - (b) drop the guard, t alone: .872 / .818, type I .0525 (all-shift shape; enforcer #5635 zero-preserving:
      .872 / .8215 at 640).
  - The proposal keeps a direction guard that costs 2.5-3.3 points of power on the zero-preserving shape, bought back
    by the 704 cap. Delegate D-12b accepted it, conditional on the enforcer's AGREE.
- **D-2. Axis.** None is needed: a harm deletes without a re-test, and a powered non-harm deletes nothing.

## 12. Record

(empty until the campaign's results; appended below the freeze, never edited above it)

## Addendum 1 (2026-10-28, before look 1): the campaign's controls, and where R1 runs

Policy-enforcer #5688 m3 and wf-rv #5689 m3. Appended below the freeze; nothing above is edited, and no rule,
threshold, seed, look or disposition changes.

- Section 6 ends "Planted controls: E1b's PSEEDS (brain-free replay seeds), unchanged". Those seeds belong to E1b's
  E-V2 replay record (`e2v2.json`: the ValenceLearner endpoint, with its planted learn and planted harm rows).
- E1c decides H-P1 only (`_plan_w`, section 3). With the E1C design the compare path treats a missing `e2v2.json` as
  not pending (16c43820), and the driver never writes one. So no planted-harm control is part of this campaign BY
  CONSTRUCTION, and none is skipped. Section 6's line names E1b's replay seeds, kept unchanged for E1b.
- The campaign's own controls are section 3's look-1 `repeat` (seed 21000: 0 action flips, series identical) and `pc`
  (seeds 21000 and 21001: flips > 0). A failed one VOIDs the look.
- R1 (section 7) is the brain-free compare fixture at n = 704 through the E1C compare path: MET on a planted harm,
  NOT MET on none, and VOID on a failed control (9197ae17). Its VOID legs cover a failed `pc` and a failed `repeat`,
  the campaign's real controls (f4578fa9; each check, when disabled, fails its own leg), beside the planted-harm leg. "Run before look 1" means `test_e1c_r1*` passes inside the
  gate run of `tests/test_rpe_charge.py` at the GO commit itself. The GO note cites that gate log; a GO without it is
  refused.

## Addendum 2 (2026-10-28, before look 1): R5's band is 2 SE of the difference

wf-rv #5689 m5. Appended below the freeze; no rule, threshold, seed, look or disposition changes.

- Section 5.3 says R5 "reproduces 5.1b's 704 row within 2 SE at 2,000 campaigns". The reference rates in 5.4 (.8800 at
  scale 1, .8335 at scale 1.115) are themselves 2,000-campaign estimates from an independent RNG stream. R5's stream is
  named `r5` and is never bit-exact with the derivation's stream (#5675 m2).
- The compared quantity is therefore a DIFFERENCE of two independent estimates, with SD sqrt(2) x SE. With 2 SE of
  ONE estimate, a correct harness fails a row with P = .157, and fails either row with P = .29.
- Read "2 SE" as 2 SE of the difference: |p_R5 - p_5.4| <= 2 sqrt(2 p (1 - p) / 2000). That is about .0206 at .8800
  and .0236 at .8335. The false-FAIL rate is .046 per row and .089 for either row.
- On a FAIL, look 1 does not start, and R5 is NOT re-run under a new stream name. The integrator attributes the gap: a
  code or pin difference between the freeze commit's derivation and the GO tree. The attribution goes on the board
  before any GO. With no attributable difference, the halt goes to the delegate, never to a silent re-run.
- `lanes/e1c-plan-w-work/r5.py` implements this band (snapshot of the prior file: `e1c-b3-row/r5.py.pre-5689`).

### The bound driver set (launcher D-21; E1b addendum 10's pattern, plus a refusal)

- Bound, after wf-rv #5689's driver fixes (M1 procs.ps1, m5 r5.py, m6 campaign.sh floors and the watcher's per-set
  reset) passed their delta reads:
  - `lanes/e1c-plan-w-work/`: campaign.sh, driver_check.sh, plan.py, procs.ps1, r5.py, ram.ps1, reds.sh, run_arm.ps1,
    tool.ps1;
  - `lanes/engine-1024/`: battery-stall-watch.ps1, and unwind.py, added to D-21's list because the watcher runs it to
    capture each stack. sched_dump.py is not bound: it runs only under an xdist pytest controller, never in this
    campaign.
- `lanes/e1c-plan-w-work/DRIVER.sha256` holds the list below. The GO file names that manifest's own sha256
  (`DRIVER_SHA256=`). `campaign.sh` refuses (exit 3) at start and at every look unless the manifest is the GO's AND
  `sha256sum -c` passes. driver_check pins this: an intact set starts; an edited file, another manifest, and a GO
  without the field each refuse (`DRIVER-D21-GREEN.log`). The first red (`DRIVER-D21-RED.log`, rc 127) proves only that
  the function was absent. The discriminators are two one-line mutants of `check_driver`, each killed by exactly its
  own case (`DRIVER-D21-MUTANTS.log`):
  - skipping `sha256sum -c` lets an edited bound file start;
  - skipping the GO-vs-manifest comparison lets another manifest start.
- Disclosed limit: an edited campaign.sh could skip its own check. Its hash is in the manifest the GO pins, so the
  integrator and the enforcer re-run `sha256sum -c` from outside at each look's post.

- Bound 2026-10-28 16:21:06 -0500 (the manifest's write time). `sha256sum -c` re-passed at 16:34. Every value equals the bytes wf-rv #5698 hashed independently (its `driver-set.sha256`;
  11/11 match). `DRIVER.sha256` uses absolute paths (#5698 nit 5), because campaign.sh never changes directory before
  the check. Manifest sha256 (the GO's `DRIVER_SHA256=`):
  `7758468edf006487e214e1e8348912ada92396d910ccac283b2ec5266fef45c5`.

```
9a1f306ab1213381333475a29391cdcf9e65ff8c5f2f6a82d3ac2b6497126c72  /mnt/d/AI-project/lanes/e1c-plan-w-work/campaign.sh
8a905e26fcae9e427ec0f2341e05c4818461369df6a8ae4378902c40fc41d1c2  /mnt/d/AI-project/lanes/e1c-plan-w-work/driver_check.sh
2df10b1053378dc5ebaeec22d1a22770e3de66f6d6facf1520cc0703bcf8b6a4  /mnt/d/AI-project/lanes/e1c-plan-w-work/plan.py
430e1736cf73314929ea7de9916c13a6a0a57b8ef059581db59a0433af473bc2  /mnt/d/AI-project/lanes/e1c-plan-w-work/procs.ps1
ccb4e6879f6b8067fb4f3e81e1a1a6c29e18145efedd966477eb3c54c93636e2  /mnt/d/AI-project/lanes/e1c-plan-w-work/r5.py
123fa5e75ea6a1cac61b378590c4610b36b9e4c6bf8393cf6311aac16480d1c5  /mnt/d/AI-project/lanes/e1c-plan-w-work/ram.ps1
796fe1e1a337b0b42cd9e44a9dd3a20c2e656f06fbb457e63fd4689f43459013  /mnt/d/AI-project/lanes/e1c-plan-w-work/reds.sh
d72c08a8df794696b1faaf38b609a01bad530bbc46ca5d921c3b8608d4ade55d  /mnt/d/AI-project/lanes/e1c-plan-w-work/run_arm.ps1
c6982c19ec0bf72390373645d790ff2eb077b7622094a26699dbb11c244fcd41  /mnt/d/AI-project/lanes/e1c-plan-w-work/tool.ps1
81dc2bdda4f0214444b9d817fc198a74105169fe9dc00044913e9ace83cb2400  /mnt/d/AI-project/lanes/engine-1024/battery-stall-watch.ps1
facf2057c5a8e084a266f7019dc08722b11a4c26bd264a76a6c60cb5e0f43608  /mnt/d/AI-project/lanes/engine-1024/unwind.py
```

- procs.ps1's wider 'rpe_charge' also matches the driver's own compare and tool steps (enforcer #5696 nit). That is
  harmless: claim() runs it only at start and after a yield, when no step of this driver is running.

## Addendum 3 (2026-10-28, before look 1 and before R5 of record): futility only where powered; R5's full row; errata

Arc-4 panel #5706 (lenses A #5719, B #5721, C #5720). The futility change was ruled by delegate D-25 (launcher-1028,
under the owner's delegation; it is the delegate's decision, not the owner's words). P1 item 2 permits it as a dated
pre-data addendum: no E1c run, no GO and no R5 of record exist yet (`lanes/e1c-plan-w-work/logs/` is empty).

**A3-1. Futility stops at 640 only where it would read powered (D-25; #5719 M3, #5721 m2).**
- Under section 5.3 as frozen, about 1 in 6 no-harm campaigns stop for futility at 640 UNPOWERED, which is
  INCONCLUSIVE (section 5.4's own .1665). Section 8's "the design makes this rare" is therefore false. Corrected here.
- The rule, from int3 e2ad9e91: `seq_decision` takes one parameter, `futility_powered` (default False, the frozen
  behaviour; every E1b design is unchanged, and E1b's record renders byte-identically). `E1C` sets it True. A futility
  look that would NOT read powered continues to the cap. Futility is non-binding, so continuing spends no alpha.
- Its red: the frozen rule stops INCONCLUSIVE at 640 while the variant continues; a powered futility look stops
  under both (`test_e1c_futility_at_640_stops_only_when_it_would_read_powered`).
- The reference table under the variant: `sim_guard` on E1b's 80 production d_s, 2,000 campaigns per shape, stream
  "e1c-a3" (`arc4-fold-row/DERIVE-A3.log`). It replaces section 5.4's rows for E1c.

  | shape | harm MET | futility-powered | cap-powered | cap-under | INCONCLUSIVE |
  |---|---|---|---|---|---|
  | null (no harm) | .0020 (type I) | .7450 | .2460 | .0070 | .0070 |
  | shift (+0.035, SD 0.3097) | .8855 | .0400 | .0740 | .0005 | .0005 |
  | up (+0.035, scale 1.115) | .8145 | .0125 | .1100 | .0630 | .0630 |

  INCONCLUSIVE under no harm falls from .1665 to .0070. Lens A measured the same variant independently: .0065 / .0010
  / .0610.
- Read the harm column correctly (panel C #5747 Dm1, panel A #5748 m4). The variant only CONTINUES where the frozen
  rule stopped, so on every path its harm events include the frozen rule's; power and type I can only rise. Lens A
  paired 4,000 campaigns on the same draws: the variant never lost a frozen crossing (0 of 4,000), with paired power
  .8155 against .8165. ".8145 against the frozen .8335" compares two DIFFERENT streams. Section 5.3's .8335 is not
  reproduced on any other stream (.8105 to .8245), so the cap's real margin over .80 is about .015 to .025, not .033.
  Type I stays below the boundaries' .05, because futility is non-binding.
- Power at the upper SD against P1's .80 floor (panel B #5746 M1). The table's .8145 is 1.67 SE above .80 on 2,000
  campaigns, which is too thin to show section 5.3's "clears .8 by 2 SE".
  - The three independent 2,000-campaign runs of the variant (.8145 on e1c-a3, .8245 on r5, .8045 on lens B's
    stream) pool to .8145 over 6,000 campaigns, 2.9 SE above .80.
  - R5 now also checks the floor directly, at 20,000 campaigns (A3-2): .8233, with -2 SE = .8179 >= .80
    (R5-MUTANTS-3.log).
  - The cap of 704 stands.
- Why E1c's futility uses the point SD while D1's moves to the upper SD (panel C Dm7): E1c's futility first opens at
  640, past its upper-SD fixed-sample target of 601. So the point-SD rule never opens futility before the target.

**A3-2. R5 reproduces the whole 704 row, not two cells (#5719 M1, #5720 M1, #5721 M1).**
- Section 5.3 says R5 reproduces "5.1b's 704 row". The bound r5.py checked only the two harm-power cells. So a harness
  with the futility bound deleted passed R5.
- R5 now runs all three shapes at 2,000 campaigns (stream "r5") and makes FOUR reproduction checks, family-wise .05
  (Bonferroni over 4; z 2.576 two-sided, 2.326 one-sided). Each band is in SE of the DIFFERENCE (addendum 2):
  - shift harm, up harm and null futility-powered: two-sided;
  - null INCONCLUSIVE <= reference + z SE: one-sided.
- It adds TWO P1 floors, each at 20,000 campaigns on its own stream, so a thin margin is resolved instead of guessed
  (panel B #5746 M1, panel C Dm2):
  - up harm power - 2 SE >= .80;
  - null type I + 2 SE <= .005 (D-25). A fixed .005 bar on a 2,000-campaign rate would false-fail a correct harness 7
    to 24% of the time near .003 to .004.
- The band is FROZEN here at z 2.576 per check (delegate D-25a, which amends D-25 condition 4's "2 SE"). At 2 SE each of
  the 4 reproduction checks false-fails a correct harness with P = .046, and the family with P = 1 - .954^4 ~ .17. At
  z 2.576 (Bonferroni over 4) the family rate is about .04.
- The R5 of record uses a FRESH stream (D-25a), in two parts, both required:
  - the statistical checks (4 reproduction + 2 floors) run on `r5go-<first 12 hex of the GO's TREE_SHA>`, a name that
    cannot exist, or have been looked at, before the GO names the tree;
  - the code-identity replay runs on the pinned streams `r5` / `r5-floor-up` / `r5-floor-null`, compared with EXACT.
- An exploratory R5 ran BEFORE this freeze (D-25a condition 3), on the pinned streams, real E1C
  (`R5-MUTANTS-3.log`). It PASSED at BOTH bands:
  - shift harm: gap .0065, 2 SE band .0201, z 2.576 band .0259;
  - up harm: gap .0100, bands .0246 / .0317;
  - null futility-powered: gap .0085, bands .0276 / .0355;
  - null INCONCLUSIVE .0070 against .0123 / .0131.
  No row passes at z 2.576 that fails at 2 SE.
- A second exploratory look, the integrator's dry run of the two-part R5 on the non-record stream `dry-integrator-1820`,
  also PASSED at both bands (`R5-DRY-FRESH.log`): gaps .0085 / .0035 / .0135, INCONCLUSIVE .0030. It also showed that R5
  refuses without TREE_SHA. No `r5go-` stream has been run.
- A FAIL halts look 1 and is attributed, never re-run under a new stream.
- Every stream is named, so R5 is DETERMINISTIC (panel C Dm2). R5 therefore also pins the fold tree's own rates BIT FOR
  BIT, as a code-identity check (panel A #5748 m2: the statistical checks cannot see a changed futility threshold or an
  extra futility look). `r5.EXACT` comes from `R5-EXACT.log`. The R5 of record at GO is a replay on the GO tree; a
  changed number means the harness changed, and that is a halt.
- Its reds (`R5-MUTANTS-3.log`). Real E1C passes all 7 rows. Each mutant fails:
  - futility off: futility-powered and identity;
  - the frozen rule: INCONCLUSIVE and identity;
  - the 5/8 conjunct: harm, the floor and identity;
  - futility also at 576, FUTILITY_CP .05, FUTILITY_CP .30: identity only, the blind spot lens A named.
- Its reds, recorded (`arc4-fold-row/R5-MUTANTS.log`): the check set was written BEFORE this run. On stream "r5",
  real E1C PASSES. Each mutant design FAILS on its own check:
  - futility off (`sd_pre=None`): null futility-powered .0000;
  - the frozen futility rule: null INCONCLUSIVE .1525;
  - E1b's 5/8 sign conjunct: harm power .0035 / .0025.
  These runs are design evidence. The R5 of record runs at GO on the GO tree.

**A3-3. Errata (no rule changes).**
- Section 2: E1b has 10 seeds with |d| > 0.4, not 8 (8 holds at 0.45). The median is -0.00204, not 0.0. Exact zeros:
  5. (Computed from the 80 d_s, 2026-10-28.)
- Section 2: "the P2 reading is under review" is stale. bd46b17d settled it before the freeze: not a P2 pair.
- Section 6's seed-overlap note predates D1 A0d. L1b-STALE's block now runs through 22999, so all of E1c's mi = 1
  sub-seeds (22000-22703) may coincide with L1b-STALE run seeds. Each campaign compares arms of the SAME seed, so
  neither result is affected.
- Size (#5721 m1, m5):
  - the fixed-sample target is 485 seeds at SD 0.3097 and 601 at the upper SD 0.345;
  - the cap of 704 is 601 x 1.17. Only about 1.06 of that is the sequential (look) inflation: a normal-model run
    reaches .8083 at 640 (panel A #5748 m5). The rest comes from the k >= n/2 direction guard and the
    zero-preserving data shape;
  - the owner's "~485 seeds" was the fixed-sample target. Under no harm the frozen design expected about 646 seeds;
    the variant adds about 12; the worst case is 704 seeds (1,408 production runs plus 3 controls).
  The GO post states these to the owner.
- Harm branch (#5721 m3): if harm is MET and the plan nudge leaves the live path, `_planned_seq`'s
  `dormancy_gate.EXEMPT` entry names its expiry: the replacement row's pre-registration. A tasks.md row for that
  replacement is opened in the same commit as the deletion. If no pre-registration is opened within the next arc,
  the integrator raises it to the owner. No hand-set duration is introduced.
- R4 (#5719, #5720 m2, #5721 m8): R4's test keeps its synthetic 58/42 shape. The registered input (E1b's 80 d_s +
  0.035, resampled to 704) also discriminates: 405/704 positive, the half rule MET, five-eighths NOT MET (lens probes).
  Disclosed, not re-tested here.

**A3-4. The GO file (driver changes from the panel; their bound hashes go in addendum 4).** The GO now names, each as
its own key:
- `TREE=<40 hex>` (exactly one);
- `DRIVER_SHA256=<manifest sha>`;
- `GATE_LOG=<path>`: the GO commit's gate run of tests/test_rpe_charge.py, written by gate_each.ps1. Its `tree` line must
  be the run tree's. Its `tests:` line must be exactly `tests/test_rpe_charge.py`. Its pytest line must read `N passed in`
  with nothing failed, errored, deselected or skipped. A subset run is refused (panel A #5748 m1), and a CRLF GO
  resolves.

campaign.sh refuses without any of them, renews `resource:timing` before every launch and inside every wait (TTL >=
run timeout + 600 s), and refuses environment overrides of its work paths. Its pre-steps are TREE-STAMPED (panel A
#5748 M1). The fixture log, the tier reds (TR1/TR2) and R5 are skipped only when their own `logs/<step>.tree` equals
this run tree. A PASS left by an older tree, or by a hand run, re-runs.

## Addendum 4 (2026-10-28, before look 1 and before R5 of record): the re-bound driver set; errata to addendum 3

The arc-4 panel's delta reads changed 7 of the 11 bound driver files after addendum 2 bound them:
- lenses A, B and C, #5746 to #5776;
- the policy-enforcer, #5749 to #5765 and its read 3.
The integrator's folds are #5756 and #5773, then delegate D-31 (launcher-1028), lens A #5780's MAJOR (fold 4, #5782) and lens A #5786's MAJOR (fold 5).

So addendum 2's manifest (`7758468e...`) fails `sha256sum -c` on exactly those 7, and a GO that names it refuses. This
addendum binds the reviewed bytes. No E1c run, no GO and no R5 of record exists.

**A4-1. What changed since addendum 2.** Each change is listed with its red. Logs are in `lanes/engine-1024/arc4-fold-row/`.
- campaign.sh:
  - The GO now needs three keys (addendum 3 A3-4): `TREE=`, `DRIVER_SHA256=` and `GATE_LOG=`.
  - The `resource:timing` lease is renewed before every launch and inside every wait. Lens C #5747 Dn3 found that the
    POOL_OK wait had no case. Its red: `arc4-fold-row/DRIVER-5747DN3-MUTANT.log`, where removing `renew_now` from that
    wait fails exactly the new case.
  - Pre-steps are stamped with the run tree AND the GO's `DRIVER_SHA256` (policy-enforcer #5753 m1). So a driver-only
    re-bind on the same tree, such as an EXACT re-pin, re-runs R5. Its red: `arc4-fold-row/DRIVER-5753-MUTANT.log`,
    where the tree-only stamp fails exactly the new case.
  - Every earlier step log is kept, never overwritten (lens B #5762).
  - R5 under delegate D-31 (A4-2): `r5_step` runs R5 once; after a single-row FAIL (r5.py rc 6) it runs ONCE more on
    the pre-named `r5go2-` stream, and anything else stops with no GO. The `R5_FAIL_RULING` path is removed.
    - The first FAIL's marker (`logs/R5_NOGO.*`, recording its tree) goes up BEFORE the re-run, and comes down only
      when that re-run passes. So a death or refusal during the re-run still leaves a no-GO (lens A #5786 D4-M1).
    - After a no-GO, a later R5 needs a GO naming an `R5_ADDENDUM=<commit>` for each kept no-GO tree. That commit must
      be in the run tree, NOT in the no-GO's tree (lens A #5780: an ancestor check alone passed any old commit, such
      as the freeze), and must add an `## Addendum` heading to this file (lens B #5785 m1).
    - Only git's exit 1 means "not an ancestor". A git error never does, so an unknown tree fails closed
      (policy-enforcer #5784 m1, lens A #5786 m3).
    - A log that drew rates but has no rc line (the driver died) is a no-GO on the tree its header names, and one
      whose header names no tree fails closed. A refusal before any draw is not a no-GO.
    - A re-GO after a no-GO is refused even on the same tree until a newer addendum exists.
  - campaign.sh never passes r5.py's `--stream`, and r5.py refuses `--stream r5go*` (lens A #5771 m2), so neither record
    stream is named by hand.
  - ONE driver per work folder (lens A #5780 D3-M1, a MAJOR in addendum 2's bytes). A second start's exit trap
    released the live driver's lease and ended its stall watcher. Now:
    - an atomic `mkdir` lock holds the owner's pid, and a lock whose pid is gone is taken over;
    - TOKEN_HELD holds the lease owner's pid;
    - the trap's release, its watcher end and refuse_driver's STOPPED post act only for the owner.
  - Every earlier step log is kept, never overwritten (lens B #5762).
  - Reds for the logs, R5 and the lock (each fails exactly its own case):
    - `DRIVER-5762-MUTANT-nokeep.log`: with the overwrite restored, the keep case fails.
    - `DRIVER-D31-MUTANT-rerun-any.log`: a re-run after any FAIL.
    - `DRIVER-D31-MUTANT-second-single-ok.log`: a single-row r5go2 accepted.
    - `DRIVER-D31-MUTANT-nogate.log`: the no-GO gate disabled.
    - `DRIVER-D31-MUTANT-no-newer.log`: an old ancestor accepted.
    - `DRIVER-D31-MUTANT-si-takeover.log`: a live lock taken over.
    - `DRIVER-D31-MUTANT-release-unguarded.log` and `DRIVER-D31-MUTANT-endwatch-unguarded.log`: a non-owner's trap acting.
    - `DRIVER-5786-MUTANT-marker-after.log`: the marker written only after the re-run (the D4-M1 shape).
    - `DRIVER-5786-MUTANT-git-error-open.log`: a git error read as "not an ancestor".
    - `DRIVER-5786-MUTANT-no-addendum-check.log`: any newer commit accepted.
  - Both tier reds must refuse, with no brain built, or pre_steps stops (exit 5) before R5 and stamps nothing (lens A
    #5771 m3). Before this, only TR1 was read and a failure went on. This was in addendum 2's bytes.
    `DRIVER-5771-MUTANT-m3-nostop.log`: the old TR1-only rule fails the stop-message case and the no-stamp case.
  - `check_gate_log` strips the UTF-8 BOM and the CR of each line once, before its greps (lens C #5768).
    gate_each.ps1 (PowerShell 5.1) writes both, and the old `tests:` check refused every real gate log. Its red:
    `arc4-fold-row/DRIVER-5768-MUTANT-oldgate.log`. Run against a byte-for-byte copy of `VERIFY-a09fb08a.log`, the old
    check fails exactly the new case (got 3, want 0).
- r5.py implements A3-2's two-part R5:
  - a fresh `r5go-<tree>` stream for the statistical rows (`--second`: the pre-named `r5go2-<tree>`, D-31);
  - the pinned streams for the identity row.
  Its exit code is D-31's outcome, from `kind()`: 0 every row in; 6 identity and both floors in and exactly one
  reproduction row OUT; 4 any other FAIL. `R5-ROWS-D31.log` gives each outcome a case, and checks both stream names.
  In `R5-ROWS-MUTANT-kind-floor.log`, a `kind()` that counts a floor or identity row as "single" misses exactly those
  3 cases.
  It refuses without TREE_SHA (`arc4-fold-row/R5-REFUSE.log`, rc 1). Per-row reds (#5753 m2): each of `check`'s 7 rows
  fires alone on a synthetic input, and the identity row reads only the pinned replay. The three two-sided rows fire
  both below and above their bands (lens A #5771 m6). Logs: `arc4-fold-row/R5-ROWS-3.log` for the real run;
  `R5-ROWS-MUTANT-m6-onesided.log`, where a one-sided band misses exactly the 3 above-band mutants;
  `R5-ROWS-MUTANT-m2-norefuse.log`, where removing the `r5go-` refusal is caught.
- tool.ps1 refuses unless the run tree's HEAD equals TREE_SHA and the tree is clean.
- plan.py prints `e1c_disposition`, never `seq_decision`'s E1b status text.
- reds.sh names the tier reds TR1 and TR2, because E1c has its own R1 and R2.
- battery-stall-watch.ps1 skips the xdist scheduler dump in campaign mode (`-BatteryPid 0`). So a stall capture never
  injects `sys.remote_exec` into some other session's pytest controller.
- driver_check.sh: 118 cases, PASS (`arc4-fold-row/DRIVER-5786.log`).
  - It now honours `E1C_WORK_WSL`, so a scratch mutant run tests the scratch files. Before this change, one mutant run
    tested the real files and read PASS. `NOTE-5753.txt` discloses it, with a second void run (a broken sed left one
    mutant copy unmutated).
  - Its first output line names the W it tests (policy-enforcer #5765 n1).
  - Under a scratch W, its no-GO and override checks print SKIPPED instead of passing on the override (lens A #5771
    m5).
- Unchanged since addendum 2: procs.ps1, ram.ps1, run_arm.ps1 and unwind.py.
- unwind.py, reviewed for the first time this arc (lens A #5771 m7, lens C #5768, policy-enforcer #5765): read-only, it injects nothing and
  writes nothing. Three nits go to the closing record:
  - the watcher writes the stack files as UTF-16;
  - in campaign mode it also matches the venv launcher stub;
  - it relies on a private 3.14-only module.

**A4-2. An R5 FAIL: one standing, mechanical rule (delegate D-31, launcher-1028).** No delegate ruling decides a
FAIL's verdict, because a ruling made after seeing a FAIL is a forking path. This replaces the ruling route lens A #5771 m1
asked to have stated, and it answers policy-enforcer #5777 m1 and lens B #5776.
1. Identity PASS, both floors PASS, and exactly ONE reproduction row OUT: R5 runs ONCE more, on the stream named here
   in advance, `r5go2-<first 12 hex of TREE_SHA>`. The GO stands only if that run passes EVERY row. Otherwise there
   is no GO, and the integrator attributes the failure.
2. Identity FAIL, any floor FAIL (P1's power floor or D-25's type-I bar), or two or more rows OUT: no GO. The
   integrator attributes and fixes it, and a new R5 needs a new dated addendum. For each no-GO tree, campaign.sh
   checks that the named commit is in the run tree, newer than that tree, and adds an `## Addendum` heading to this
   file (A4-1). The integrator and the enforcer read the addendum itself at the GO. A no-GO from two single-row
   FAILs, where identity passed and `REF` matches `DERIVE-A3.log` so that chance is the only explanation left (about
   .0016), goes to the delegate before any addendum is written. The delegate decides whether a re-draw is
   pre-registered, never the verdict (lens B #5785 nit; lens C #5795 nits 1-2).
3. This supersedes two sentences in exactly one way: it names this one second stream in advance. The sentences
   are addendum 2's "On a FAIL, look 1 does not start, and R5 is NOT re-run under a new stream name" and addendum 3
   A3-2's "A FAIL halts look 1 and is attributed, never re-run under a new stream" (lens C #5788 nit 2).
Why: the family-wise chance FAIL is about .04. The single-row path then falls to about .04 x .04 = .0016, and with
two-row and floor FAILs on the first run the total chance no-GO is about .003 to .004 (lens C #5788 nit 3). A real
harness defect will most likely fail twice. A broken harness fails identity on every stream, so no re-run can pass
it. The attribution of any FAIL also checks `REF` against `DERIVE-A3.log`: once identity passes, a typo in the
reference table is the only cause other than chance (lens B #5776).

**A4-3. The bind.** `lanes/e1c-plan-w-work/DRIVER.sha256` holds the list below. Every value equals the bytes the delta
reads hashed. Manifest sha256 (the GO's `DRIVER_SHA256=`): `db48f4d36752615049586f5e71078917e95ae406e3a099ba77a9fb2758977eee`.
Read 5 AGREEs on exactly these bytes: policy-enforcer #5792, lens A #5793, lens B #5791, lens C #5795.
Bound 2026-10-28 19:17:26 -0500 (the manifest's write time); `sha256sum -c` passed 11/11 in WSL at the bind.

```
7880f08fe5c85e2ea0fc3a7936327a6bf33dc1ed2d950b96217d673f5d800465  /mnt/d/AI-project/lanes/e1c-plan-w-work/campaign.sh
d38dac36d3c8ae2f064b9fce2991c5f85008871f0a1a7a7956a825c6005fb6b1  /mnt/d/AI-project/lanes/e1c-plan-w-work/driver_check.sh
45bbc8d9a90be744adbe215f9b0e2d68eeeea475a4be549890995078ea4256ae  /mnt/d/AI-project/lanes/e1c-plan-w-work/plan.py
430e1736cf73314929ea7de9916c13a6a0a57b8ef059581db59a0433af473bc2  /mnt/d/AI-project/lanes/e1c-plan-w-work/procs.ps1
a867eabcfa2ee80fbdefd491a5d65a0d217141f7c63a8615bedfd8e600d776f2  /mnt/d/AI-project/lanes/e1c-plan-w-work/r5.py
123fa5e75ea6a1cac61b378590c4610b36b9e4c6bf8393cf6311aac16480d1c5  /mnt/d/AI-project/lanes/e1c-plan-w-work/ram.ps1
0b810d76ac1be4a1ce3f292761180e999a803a308f44d5a6da05a13965fcd3b9  /mnt/d/AI-project/lanes/e1c-plan-w-work/reds.sh
d72c08a8df794696b1faaf38b609a01bad530bbc46ca5d921c3b8608d4ade55d  /mnt/d/AI-project/lanes/e1c-plan-w-work/run_arm.ps1
bfbab3a5432a2be559cd1462c6c8f85f9b497e1ec210c28fd33d5803a2168703  /mnt/d/AI-project/lanes/e1c-plan-w-work/tool.ps1
b3fb7501831736637605e387ace1c54f01adfb8340fd36ebeb57f775c4e6dd4c  /mnt/d/AI-project/lanes/engine-1024/battery-stall-watch.ps1
facf2057c5a8e084a266f7019dc08722b11a4c26bd264a76a6c60cb5e0f43608  /mnt/d/AI-project/lanes/engine-1024/unwind.py
```

**A4-4. Errata to addendum 3 (no rule changes).** Addendum 3 is append-only after f58e1642, so it is corrected here.
- A3-1, "paired power .8155 against .8165": the frozen rule's power is .8155 and the variant's is .8165, on the same
  4,000 draws (lens A #5748 D-m4; policy-enforcer #5753 n1).
- A3-2, "R5 now runs all three shapes at 2,000 campaigns (stream "r5")": stream "r5" is only the code-identity replay.
  The statistical rows run on the fresh `r5go-` stream, as A3-2's later bullet states (lens B #5762 nit 1).
- A3-2, the band arithmetic (lens C #5768). At 2 SE the family false-fail rate is about .15, not .17, because the
  one-sided INCONCLUSIVE row fails at .023. z 2.576 (two-sided) and z 2.326 (one-sided) are each a per-check .01 level.
  That is not exactly "Bonferroni over 4", which would be .0125 per check. The stated family rate of about .04 stands.
- A3-2, a `r5go-` name "cannot exist, or have been looked at, before the GO names the tree": that holds by discipline,
  not by construction. The name can be computed once the stacked head exists. The integrator runs no `r5go-` stream
  before the GO, and the no-GO gate (A4-1) stops any further draw after a no-GO until a newer dated addendum.
- The bound r5.py's own docstring and comments repeat "Bonferroni over 4" and "cannot exist before the GO". They read
  as the two errata above correct them. The bytes are frozen under the convergence rule (#5773), and its text does
  not change what the code runs (lens C #5778 nit 1).
- A3-2 cites `R5-DRY-FRESH.log` for R5's refusal without TREE_SHA. That log holds only the dry PASS; the refusal is
  `R5-REFUSE.log`. `R5-MUTANTS-3.log` came from the r5.py before the two-part change. `R5-ROWS-3.log` covers the
  bound file's 7 rows.
- A3-2, "TWO P1 floors": only one is a P1 floor, the up row's harm power (- 2 SE >= .80). The other is delegate D-25's
  type-I bar (null type I + 2 SE <= .005) (#5753 n2).
- A3-2, the bullet "Its reds, recorded (`R5-MUTANTS.log`)" describes the superseded 5-check set. The bullet above it
  (`R5-MUTANTS-3.log`, 7 rows) is the current record (#5753 n3).
- A3-3, R4: the input is E1b's 80 d_s + 0.035 TILED to 704 (seed i takes d_(i mod 80)), as section 5's R4 states. It is
  not "resampled". The 405/704 count is for the tiled input (lens C #5747 Dn1).
- A3-4: each pre-step's stamp is now the tree plus the driver set (A4-1), not the tree alone.
- Section 10, "no reboot is possible": the owner rebooted at about 17:55 CDT on 2026-10-28
  (lanes/SAVEPOINT-REBOOT-1028/INDEX.md). D-13 REVISED's one-run-at-a-time stands until the launcher re-rules the host
  regime. A move off one-run-at-a-time changes the conditions the A/A repeat was read under, so R3 is re-run under the
  new regime before any later look is read (lens B #5776 nit 1).

## Addendum 5 (2026-10-28, before the GO file and before any E1c run): the host regime after the reboot (delegate D-34)

Section 10 names delegate D-13 REVISED, which was written when the host could not reboot. The owner rebooted at about
17:55 CDT (addendum 4 A4-4). Delegate D-34 (launcher-1028, `lanes\DELEGATED-DECISIONS-1028.md`) rules on the regime.
Because the frozen section 10 names the regime, this addendum lands BEFORE the GO file. No E1c run, GO or R5 of record
exists.

**A5-1. D-13 REVISED STAYS for the GO, the pilot and look 1.**
- One run at a time.
- Section 10's hysteresis, the nonpaged-pool check and the 85% commit line are unchanged.
- Why, from D-34: the reboot removed D-13's premise (idle Available is about 20 GB, and the nonpaged leak is gone). But
  campaign.sh is serial by construction: its run_one loop runs one brain at a time, and its claim refuses to start
  beside any live brain process. Section 10 also requires R3 to be re-run at the worker count before any concurrency.
  So concurrency is a build, not a setting.

**A5-2. What the pilot measures (policy-enforcer #5822 m2, m3).**
- The pilot unit is seed 1's two runs, `real` then `pfrozen`. Section 10's timeout rule keeps the FIRST run's T, and
  this addendum does not move it.
- T for the projection is the LARGER of the two runs' wall times.
- P is the peak private bytes of the WHOLE run subprocess tree, children included. The arc-4 battery found subprocess
  children (brian2 reference traces, about 1.7 GB each) that a per-process reading misses.
  - The tree is defined by ParentProcessId descent from the run's root pid, with no image-name filter, so a non-python
    child (a compiled binary, nvcc, a shell shim) counts.
  - P is the maximum, over samples, of the SUMMED private bytes of the tree, at a sample period of 1 s or less (and at
    most T/20).
- The GPU peak per arm is read per pid (nvidia-smi --query-compute-apps). A device-total reading may be cited only as
  an upper bound.
- P and the GPU peak are taken per arm. A5-4 uses the larger arm.
- The observer is out of band. It is not in the bound driver set, and it decides no verdict: P sets only A5-4's
  worker count.
  - The observer's path and sha256 are named in the pilot post, before the pilot runs.
  - One control comes before P is cited: a planted child process, preferably non-python, holding a known allocation,
    which the observer must report within its stated tolerance.

**A5-3. The projection, posted after the pilot.** T x 2 arms x seeds, both to the expected stop (about 658 seeds under
no harm: about 646, plus about 12 for the futility variant, addendum 3 A3-3) and to the 704 cap. Section 10's E1b
throughput (about 21 s per run) puts the cap at about 8.2 h, and the expected stop at about 7.7 h. The 48 h line of
A5-4 is crossed only if T exceeds about 131 s per run. The pilot's T replaces the estimate.

**A5-4. Serial or concurrent, decided on the projection.**
- An expected-stop projection of 48 h or less: the whole campaign stays serial.
- More than 48 h: the integrator may propose a concurrent driver, as its own fold that EXTENDS campaign.sh
  (graph-exists first), with all of:
  - looks composed by seed index, never by completion order;
  - R3 re-run at the maximum worker count, with each seed's two arms in the same launch batch (section 10, carried
    forward); a flip keeps 1 worker;
  - an identity control: one seed run alone and the same seed run beside N-1 others give a byte-identical result row;
  - workers = min(4, floor((Available - 4 GB) / P)), plus the GPU bound;
  - hysteresis, the 85% line and the pool check unchanged;
  - stall thresholds re-derived at N;
  - a panel before use, and a switch only at a look boundary;
  - a new DRIVER_SHA256 bound in a dated addendum before the switch. That addendum names the looks already read,
    and changes no rule (policy-enforcer #5822 n2).
- P1 declared exceptions (policy-enforcer #5822 m1; D-34 calls them the delegate's operational choices): the 48 h line,
  the 4 GB reserve and the ceiling of 4 workers. They are scheduling only, and no endpoint reads them.

**Raising ambition (self-applied, SNDL rule 2026-10-25).**
- Rung picked: adopt D-34 as ruled. Measure T and P first, then decide concurrency on the measurement.
- Rung above, ruled out: building the concurrent driver now. D-34 gates it on the 48 h projection, with its own panel.
  P8 item 2 also owes it a control that is not the method, the identity control, which needs the measured P.
- Outside judge: the pilot's measured T and P, and R3 at the worker count.

## Record of the campaign (2026-10-28): VOID at look 1

Appended below the freeze and its addenda (section 12). Nothing above is edited.

**Disposition: VOID, so there is no verdict.**
- The campaign stopped after look 1 (22:25 CDT, board #5840). Its pre-registered control R2 did not fire:
  `pc` on seed 21001 read 0 action flips, and every recorded series was byte-identical to `real_21001`.
- Section 3 and section 7 R2 make that a VOID.
- Delegate D-36: no resume, no START_LOOK = 2, no pc re-run and no pc-seed swap.

**What ran.**
- **GO:** #5825. TREE `1b6e53ab`, DRIVER_SHA256 `db48f4d3`, GATE_LOG `VERIFY-1b6e53ab.log` (194 passed).
- **The compiled `_native`, pinned after the GO:**
  - NATIVE_SHA256 `5c5b39c1fac3184e4d534a8522f617db785ed76fabb5fbc8bb3d41a8da80d297` (221,184 bytes, built 21:22:35).
  - The toolchain was read from the host: rustc 1.96.0 (ac68faa20), cargo 1.96.0 (30a34c682), maturin 1.15.0.
  - An out-of-band guard hashed it every 30 s. It was unchanged throughout.
- **Smoke 1 was a FAILED LAUNCH, not a run.** It failed at import with an ImportError on `snn_data_lake._native`. The
  cause was the environment: the fresh run tree had no compiled extension. It produced no result row. Smoke 2, after
  the build, passed (about 24 s).
- **R5 of record (D-31): PASS**, rc = 0, at 21:28:02, on stream `r5go-1b6e53ab933b`. All 7 rows were OK:

  | Row | Value |
  |---|---|
  | shift harm | .8765 vs .8855 +- .0259 |
  | up harm | .8165 vs .8145 +- .0317 |
  | null futility-powered | .7375 vs .7450 +- .0355 |
  | null type I | .0032 + 2 SE = .0041 <= .005 |
  | null INCONCLUSIVE | .0045 <= .0131 |
  | up power floor | .8207 - 2 SE = .8153 >= .8 |
  | code identity | EXACT |

- **The pilot (addendum 5).**
  - T = 26 s (real_21000 26 s, pfrozen_21000 25 s), which projects about 10 h serial.
  - P: pfrozen_21000 3,092 MB. GPU per pid is NA under WDDM, with a device upper bound of 897 MB.
  - **The real_21000 observer missed.** pilot_watch matched on arguments alone and latched onto the TR1 tier-red probe,
    reading P = 0 from 1 sample.
  - A supplement on seed 21003, with the same observer shas, read real 3,058 MB and pfrozen 3,063 MB.
- **Runs:** 131 runs at look 1, all rc = 0.

**Look 1, as seen. These numbers decide nothing** (D-36; E1d section 1).
- H-P1: t = 1.531 (34/64 > 0), SD 0.3542.
- Exact-zero 0.016.
- The tail census (|d| > 0.4) has 8 seeds.

**Why VOID: a brain property, not a harness defect** (#5845, accepted by D-37).
- The old pc stubbed `PrefrontalExecutive._decide_spiking` to winner 0. On production the race reaches behaviour only
  through `_gate_primary`'s gated write of the held goal.
- On seed 21001 no commitment ever clears `gate_threshold` (293/293 gate calls hold 0). So "force 0" equals the
  brain's own held goal and changes nothing.
- Stubbing winner 1 on 21001 flips 378 actions, so the channel is live.
- E1b's pc seeds (8 and 9, both tiers) and E1's (0 and 1) all flipped, so no E1b re-read is owed.

**Driver defect found at look 1.** The compare step wrote into a `reports\` folder that nothing created
(FileNotFoundError). Validity was computed before the write failed. The fix, with its red, is in E1d's driver bind.

**Reproduction tree.** E1c reproduces from `1b6e53ab`, not from HEAD: E1d's B3 moves `E1C_SEEDS` to its own block.

**P3 opening (D-37 item 4), recorded and fixed by no row here: the planner gate never clears on some seeds.**
- On calibration seeds 28000-28063 (tree `1b6e53ab`, pyd `5c5b39c1`), 9/64 (14.1%, Wilson 95% [7.6%, 24.6%]) are
  goal-frozen at their first held goal, which is 0 on 64/64 seeds.
- On those seeds the executive's race runs about 7,300 times per run and charges nothing (#5850).

**Successor:** E1d (T124), `evidence/e1d-plan-w-pc2.md`, FROZEN at `df6bcfde`. It has a held-goal positive control
(delegates D-37 and D-42) and uses seeds 23000-23703.

**Corrections and additions to this record** (stack-a #5934 MINOR-3, 2026-10-29; nothing above is edited):
- **R5's time.** R5 of record FINISHED before the first campaign run: `logs/r5.log` was written at 21:27:59, and
  `real_21000` started at about 21:28:01. The "21:28:02" above is when the integrator's waiter detected it, not when
  R5 ran.
- **The look-1 controls, all read** (the look-1 compare log, VALIDITY section):
  - R3, the repeat on 21000: 0 action flips, recorded series identical. It HELD.
  - `pc_21000`: 420 flips (first at tick 25). It HELD.
  - `pc_21001`: 0 flips. It FAILED, and that is the VOID.
- **The stop at #5840.**
  - The driver stopped on "STEP FAILED compare-production-look1" (exit 5), because the compare tool crashed writing
    its report into a `reports\` folder that nothing had created.
  - The compare had already printed "VALIDITY: VOID -- pc seed 21001 flips == 0" before the write failed.
  - The driver's own VOID halt branch (exit 6) never ran live.
- **The P3 opening above is carried in the integrator's openings ledger** (2026-10-28, D-37 item 4), as E1d section 9
  B0 requires.

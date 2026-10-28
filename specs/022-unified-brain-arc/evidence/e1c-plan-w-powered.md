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

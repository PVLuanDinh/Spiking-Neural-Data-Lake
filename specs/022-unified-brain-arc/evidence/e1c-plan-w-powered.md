# E1c -- the powered production test of `_plan_w` harm (H-P1), owed by E1b's INCONCLUSIVE (DRAFT pre-registration)

> **Status: DRAFT, pre-data.** Nothing has been built or run in a brain. The design numbers in section 5 come from a
> brain-free resampling of E1b's recorded production data. The file freezes at the commit that drops "(DRAFT
> pre-registration)" from the title, after the policy-enforcer has reviewed it. Later changes are dated pre-data
> amendments. Any other change is a new pre-registration on a fresh block.
> - **Task:** T122 [E1] (proposed). T119 and T120 are taken on int3; D1's cloud pre-registration proposes T121.
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
  - **The bound is the policy's** (delegate D-12c asked for one that opens; measured, then reversed by the delegate).
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

## 9. Build rows (lanes, after the reboot; graph-exists before each; no new mechanism)

- **B1.** The two carried harness fixes from E1b:
  - print `null_precondition` in `cross`;
  - the E-P1 denominator (look-1 count over the look's seed count).
- **B2.** An E1c campaign spec in `instruments/rpe_charge.py`:
  - production tier, arms `real`/`pfrozen`, the seeds, looks, cap and K_MIN of section 5.3;
  - a parameter of the existing sequential path, not a copy (P2);
  - R4 as a test.
- **B3.** The campaign driver: reuse E1b's run tree pattern (detached tree, one arm per subprocess) and `resource:timing`.
- **Commits.** One change each. The enforcer reviews each build, and nothing runs before the freeze commit.

## 10. Cost and resources

- **Worst case:** 704 seeds x 2 arms = 1,408 runs, plus 3 control runs.
- **E1b's measured throughput:** 534 production and no-encoder runs in about 3.1 h of wall time, about 21 s per run.
  That puts the worst case at about 8.2 h on an unloaded host.
- **Resources.** `resource:timing` is held for the whole campaign. No battery or other brain campaign runs beside it.
- **Host (no reboot is possible; owner 2026-10-28):** the throttled regime of delegate D-13 applies.
  - A pilot arm measures peak private bytes P and wall time T.
  - Workers = floor((Available - 2 GB) / P), between 1 and 4, re-derived at every look.
  - Pause launches while Available < 1.5 GB or commit > 85%.
  - Timeouts are set as multiples of T.
  - The nonpaged pool is logged at every look.
  - A pause is logged, never a verdict.

## 11. Open for the owner and the enforcer (before the freeze)

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

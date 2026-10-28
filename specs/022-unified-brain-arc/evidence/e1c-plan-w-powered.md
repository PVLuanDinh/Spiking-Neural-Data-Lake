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
- **Beside, no bar:**
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

### 5.2 Finding: E1b's scaled sign conjunct cannot fire on this shape

- **The rule.** `seq_decision` declares harm only when t crosses AND k >= ceil(K_MIN x n / 8) seeds are positive, with
  K_MIN = 5. At n = 640 that asks for 62.5% positive seeds.
- **What this shape gives.** On the measured shape, a true shift of exactly DH makes only 58.75% of seeds positive
  (58.75% on shift, 57.5% on shift_up). The conjunct therefore fails more often as n grows. Power is 0.169 at 512,
  0.137 at 640 and 0.133 at 768, and it is never .8 at any cap.
- **Consequence.** A follow-up that keeps the conjunct can never be "a run with enough seeds (power .8)" under P8 item 4.
- **Scope of the finding.** It is about this design at large n. It does not impugn E1b's verdicts: E1b's H-P1 stopped
  for futility at n = 80 on conditional power, never at a crossing the conjunct could have blocked.

### 5.3 The design (frozen at the freeze commit)

- **Looks:** every 64 seeds, n = 64, 128, ..., 640 (10 looks).
- **Cap:** 640, the smallest 64-multiple cap whose power under this rule is at least .8 at the 80% upper SD (512 gives
  .738). Power is .886 at the measured SD and .812 at the 80% upper SD. Type I is .0465 on the measured null shape.
- **Boundary:** `rpe_charge.seq_boundaries` (Lan-DeMets O'Brien-Fleming, one-sided alpha .05) on those looks.
- **Harm at a look** = t above the look's boundary AND k >= ceil(4n/8), meaning at least half of the seeds are
  positive. This is the large-n form of E1b's direction guard. It costs nothing measurable (.886 against .872 for t
  alone) and still refuses a mean carried by a few regime-flip seeds against the majority. **(Open, section 11:
  D-1.)**
- **Futility: none in practice.** `futility_open` (sd_pre 0.3097) is False at every look before the cap. The mde at
  576 is 0.0354, just above 0.035 (`looks_e1c.log`). The campaign therefore stops only at a harm crossing or at the
  cap.
  - The per-look boundaries t are 7.139, 4.392, 3.453, 2.933, 2.594, 2.352, 2.167, 2.020, 1.899 and 1.798.
- **The cap's "powered" read uses the OBSERVED SD.** At 640 the mde is 0.0323 at SD 0.3097. It goes above 0.035 when
  the observed SD exceeds about 0.336, and the cap then reads INCONCLUSIVE. Its rate per shape is in 5.4.
- **Powered.** Read as E1b addendum 3 reads it: the mde at the stopped n with the observed SD, at or below 0.035.
- **Simulation guard.** Before look 1, the frozen harness reproduces this file's 640 / k >= n/2 row within 2 SE at
  2,000 campaigns.

### 5.4 What each shape ends in under 5.3 (2,000 campaigns each, `derive_e1c-detail-2000.log`)

| resampled shape | harm MET | cap, powered (no harm) | cap, underpowered (INCONCLUSIVE) |
|---|---|---|---|
| null (mean harm 0) | .0495 | .8940 | .0565 |
| shift (+0.035) | .8795 | .1145 | .0060 |
| shift_up (+0.035 at SD 0.345) | .8110 | .0635 | .1255 |

- These rates come from different random streams than 5.1's table, and the two agree within 1 SE.
- At the pessimistic SD, one campaign in eight ends INCONCLUSIVE. That is the design's disclosed residual, not a red.

## 6. Seeds

- **Confirmatory block:** 21000-21639, in order. 21640-21999 stay unexposed for a repair.
- **Grep check, 2026-10-28, int3.** 0 seed/block/range hits for 21000-21999 in evidence, src, tools/*.py and tests.
  Board claim #5613 holds 21000-21999.
- **Not reused:** E1b's seeds 8-183, E1's 0-7, and the derive RNG seeds 8801-9595 (simulation streams only).
- **Repeat control:** seed 21000. **Planted controls:** E1b's PSEEDS (brain-free replay seeds), unchanged.

## 7. Reds (each constructed and RUN before look 1; a red that does not fire halts the campaign)

- **R1, compare fixture** (E1b section 4, brain-free), at n = 640 under section 5.3's rule:
  - `pfrozen = real + 0.1` on every seed must read MET;
  - `pfrozen = real` must read NOT MET;
  - a failed planted control must read VOID.
  If any of the three prints differently, the harness is blind.
- **R2, in-brain sensitivity** (look 1): `pc` flips > 0 on seeds 21000 and 21001.
- **R3, repeat** (look 1): the A/A pair on seed 21000 has 0 action flips and identical recorded series.
- **R4, rule-change red.** Fed the 80 E1b differences shifted by +0.035 and tiled to 640, the k >= ceil(5n/8) rule
  must read NOT MET while k >= ceil(4n/8) reads MET. This proves the rule actually changed in the harness.
- **R5, simulation guard.** As in 5.3.

## 8. Disposition (fixed in advance; the production tier decides, #4616)

- **HARM MET** at a look:
  - The learned `_plan_w` trajectory is a measured harm on production at 600 ticks. P8 null-scale item 4: "a measured
    harm ... is deleted without a re-test".
  - The integrator writes the deletion row: `_plan_lr = 0`, with `_plan_w` held at init, and the update removed as
    one change.
  - The owner is told before it lands. `_plan_w` is not an owner-drawn P4d part.
- **NOT MET at a futility or cap stop, POWERED** (mde <= 0.035):
  - "Powered: no harm of 0.035 or more on production at 600 ticks." `_plan_w` stays as it is.
  - E1's no-encoder +2.82 is recorded as not replicated on production at power.
  - No NULL-RECORD, because E-P2 is unpowered (section 4). This is a non-harm result, not a null of the method's
    charge.
- **NOT MET, UNDERPOWERED** (mde > 0.035; the design makes this rare): INCONCLUSIVE, with its mde recorded. Nothing
  changes.
- **VOID** (validity, a moved tree, or a red that did not fire): no verdict. Repair, or open a new pre-registration on
  21640 and up.
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

- **Worst case:** 640 seeds x 2 arms = 1,280 runs, plus 3 control runs.
- **E1b's measured throughput:** 534 production and no-encoder runs in about 3.1 h of wall time, about 21 s per run.
  That puts the worst case at about 7.5 h. E[n] is about 436 seeds (about 5 h) under the claimed shift.
- **Resources.** `resource:timing` is held for the whole campaign. No battery or other brain campaign runs beside it.
- **Prerequisite:** the host reboot (2026-10-28: nonpaged pool 16 GB, Available 2.4-2.75 GB).

## 11. Open for the owner and the enforcer (before the freeze)

- **D-1. The harm rule's direction guard.**
  - Proposed: k >= ceil(4n/8), "at least half positive".
  - Alternatives:
    - (a) keep E1b's k >= ceil(5n/8). The power ceiling stays near 0.14, so the run can never be the powered run
      item 4 asks for;
    - (b) drop the guard, t alone: .872 / .818, type I .0525.
  - The proposal keeps a direction guard at no measurable cost.
- **D-2. Axis.** None is needed: a harm deletes without a re-test, and a powered non-harm deletes nothing.

## 12. Record

(empty until the campaign's results; appended below the freeze, never edited above it)

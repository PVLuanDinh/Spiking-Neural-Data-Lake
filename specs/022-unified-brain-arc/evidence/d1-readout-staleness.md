# D1 -- L1b's owed co-adaptation control, read as representational drift (pre-registered 2026-10-27)

Requested by the owner, 2026-10-27, verbatim: *"start D1, pre-register the L1b drift control"*. The research map is
`research/representational-drift/MAP.md` (section 4, D1). The row L1b owes is `evidence/l1b-synaptic-online.md:312-313`:
*"Before that negative is generalized it owes one cell: a rule arm with the growth-gate verdict logged per refit row, or
a re-solve-beta-on-every-refit control."* D1 is both of those: every arm logs the gate verdict per row, and every row
carries a scoring-only re-solve. It proposes no rule, re-lands nothing and claims nothing beyond section 2.6.

Proposed registration, for the operator to place:
- task **T121**;
- red family **D**, row **D-R1** (the letter is unused in Part F; measured below);
- this file as the row's record.

The registration rows are NOT in this commit. This commit is the pre-registration alone (`docs/AGENT_COLD_START.md`
§6.2).

Written in a cloud clone at `81c1c33` (main `6612ce7` plus the research map). This environment has no cp314t
interpreter, no perception packs and no GPU, so **nothing below was run in a brain**. Every number in section 1 is read
from the tree or from a frozen record, with its location.

## 1. Territory (code at `6612ce7`, read 2026-10-27)

**The incumbent at HEAD is not the incumbent L1b ran against.** L1b ran on 2026-09-05. Its only readout was `beta`, re-solved at
open-gate refits and stale in between (`l1b-synaptic-online.md:22-25`). Since then two changes landed.

- **L1c (2026-09-06)** put an online delta readout on the live prediction:
  - `growth/neurogenesis_engine.py:409-417`: `NeurogenesisSelfImprove(online_readout=...)`.
  - `:557-558`: `self.online_readout.learn(m, S[0], cat)` on every wake transition.
  - `engine/brain.py:1572`: the brain constructs it as `online_readout="delta"`.

  The LIVE reader therefore co-adapts with any change in the hidden rates.
- **L1c Amendment 3** kept a SECOND, deliberately stale readout for the growth gate:
  - `neurogenesis_engine.py:418-425`: `self._batch_beta`, "the readout of the last OPEN-gate solve".
  - `:667-675`: a closed-gate refit scores the held-out slice with it, `m.decide(S_te, beta_inst)`.
  - `:770`: only an open-gate refit refreshes it.

  The closed-gate refit row is the growth gate's own evidence (`self.trajectory`, consulted at `:663-665`). So the stale
  reader L1b's harm was attributed to still exists at HEAD, but it now reads only the gate's evidence. It no longer
  reads the live prediction.
- **The scoring path was fixed after L1b** (`evidence/scoring-path-abstention.md`; `l1b-synaptic-online.md:245-255`).
  `NGSNN.decide` (`growth/ng_snn.py:591`) is now the one decision rule. The fix moved the incumbent's accuracy on 3 of
  8 seeds. **L1b's OJA arm was never re-read on the fixed path**; only the incumbent was.

**The mechanism L1b blamed, quoted.** `l1b-synaptic-online.md:266-277` reads: "input rows moving under a readout the
growth gate holds STALE", with three observations behind it:
- every collapse follows a growth row after which the T047 gate closes;
- OJA and BCM read identical accuracies on different substrates for 4-6 rows;
- "the incumbent shows the SAME mode with frozen `W`".

The alternative cause, "aligned rows lose the feature diversity the readout uses", was retracted as UNMEASURED
(`:276-277`).

**The rule's measured artifact** is `git show 4f26c9c:src/snn_data_lake/plasticity/grown_input_rule.py`. That is
289 lines, read for this record (`l1b-synaptic-online.md:346-347`). It applies:

    u_i <- normalize(u_i + mu*g*post_i*xn),   W_i <- ||W_i|| * u_i,   OJA: post_i = s_i,   mu = 0.05 (CHOSEN)

The plug point and the rule were deleted (`neurogenesis_engine.py:426-431`, `l1b-synaptic-online.md:294-306`). The
probe harness `l1b_probe.py` (`:90`) is not in the tree or its history: `git log --all -- '*l1b_probe*'` is empty.
**D1 rebuilds the harness.**

**The re-solve machinery already exists and is the model's own.**
- `NGSNN._solve_beta(S, Y)` (`growth/ng_snn.py:317-323`) is the regularized pseudoinverse that `fit` uses.
- `NGSNN._rates(X, W, b)` (`:280`) is the deterministic LIF rollout.

Existence check (§6.1). I ran `git grep -i -E "readout_staleness|shadow.?re-?solve|re-solve-beta" -- src tools
tests`. It returned nothing. The three in-tree uses of "drift" mean other things:
- `instruments/prior_drift_harness.py` (weights against the H01 prior);
- `infra/drift_monitor.py` (input PSI);
- `instruments/sleep_plasticity_ab.py` digests (equality only).

**None found by these searches.**

**The scale and the null-scale triage.** `evidence/null-scale-triage.md:69-70` records OJA's null-scale record
(paraphrased here, not quoted, so that `gates/null_scale_gate` does not parse this file as a second record; erratum E1):
the method is the Oja local rule on the grown units' input rows (L1b), at units 38, ticks 800, seeds 8, with mde
unknown and the re-test state CANDIDATE.

- AXIS = units, so a 10x re-test means at least 380 units.
- It is "NOT a measured harm" under the P8 amendment, item 4, because L1b pre-registered no harm bar.
- Triage row 2 (`:206`) says the stated cause "is gone since L1c landed". **Section 1's first bullet shows that is true
  for the live prediction and false for the gate's evidence.**

**The prior SD.** I recomputed it from `l1b-synaptic-online.md:179-210` (pre-correction scoring path).
- Paired (acc_OJA - acc_BATCH) over seeds 0-7: mean -0.1964, SD 0.2832, t -1.961. This reproduces the record's
  t = -1.959 to rounding.
- BCM: mean -0.1247, SD 0.2597, t -1.359.

At SD 0.28, a one-sided .05 test needs on the order of 50-60 seeds for a 0.10 effect at power .8. The exact sign-flip
FPR in `instruments/paired_contrast.py` refuses n > 40 (`MITM_MAX_N = 40`, `:63`, `:166`: "refused, never
approximated"). **This prior is for an endpoint D1 does not use** (between-substrate accuracy). D1's primary endpoint is
within-substrate (section 2.2), and its SD is unknown until the pilot. Section 2.5 makes feasibility a written branch,
not a hope.

**The red letter.** `grep -oE "\b[A-Z]{1,2}-R[0-9]"` over `PLAN-FINAL-2026-09-02.md` returns the families
G, S, N, L, C, R, W, A, X, H, U and B. There is no `D-R`; the six raw "D-R" hits are substrings of `PROPOSED-ROWS`.

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.1 The mechanism under test, its incumbents (P2), and what is new

**Under test.** Does a stale readout carry L1b's Oja harm? The hypothesis is the one stated at
`l1b-synaptic-online.md:266-277` (M-1).

**Adopted, not re-implemented.**
- The perturbation is L1b's rule at its measured artifact (`4f26c9c`), constants unchanged: mu = 0.05, ungated (g = 1;
  L1b withdrew the gate at its Stage 0, `:158-163`), post_i = s_i. It is applied once per wake percept, right after the
  transition is buffered, to the RETAINED model's `W` in place. That is L1b's call position (`:55-57`).
- The re-solve is `NGSNN._solve_beta` on the retained model's own `_rates` of THIS refit's training rows. The targets
  are encoded exactly as `fit` encodes them.
- The decision rule is `NGSNN.decide`.
- The statistic is `instruments/paired_contrast` plus `instruments/rpe_charge.seq_boundaries` (section 2.5).

**New, and confined to one instrument module.** The proposed module is `instruments/readout_staleness.py`. Its
disposition is `dormancy_gate.EXEMPT` `permanent`, in the `prior_drift_harness` shape: a measurement on scratch brains,
never on a tick. It contains five things:

1. **The SHADOW re-solve.** At every refit row, in every arm, the instrument computes
   `beta_sh = m._solve_beta(S_tr, Y_tr)` on the model `m` the row just scored, and
   `acc_sh = mean(m.decide(S_te, beta_sh) == y_te)` on the same held-out rows.
   - It is a scoring-only reading. Nothing reads it back: not the gate, not the trajectory, not the live beta, not the
     admission path.
   - So every arm's substrate is exactly what it would have been without the instrument. This is L1b's own panel
     lesson, `:100-102`: "a future twin must patch scoring only".
   - A test asserts it: the substrate hash `W`/`b`/`beta`/`_batch_beta` with the shadow on equals the hash with it off,
     per row, on a brain-free fixture.
2. **The perturbation operators**, OJA and RAND (section 2.3). They are applied through an INSTANCE-level wrapper on
   the engine object in a child process on a `throwaway_home()`. A class-level patch is refused: L1b's class-level
   `predict` patch leaked into the growth gate (`:96-99`, panel I-1).
3. **The gate log.** It records the T047 verdict (open/closed) on every refit row, which is L1b's other owed option.
4. **The drift readings.** These are descriptive (section 2.4).
5. **The driver**, which runs one arm per child process.

**Governance the instrument joins (resolved at the build commit, not here).**
- The OJA and RAND operators write weights, so they carry `GRADIENT_CLAIM` rows (LOCAL, never on a live path).
- They need a rule-census adjudication if `rule_composer --regen-census` counts them (`l1b-synaptic-online.md:36-41`).
- Their learn-verb methods need `EXEMPT_METHOD` entries.
- If the census ratchet refuses an instrument-local operator, that is recorded as a build finding before any cell runs.

### 2.2 Arms and endpoints

**Arms.** Each seed runs four arms in separate child processes on one in-brain stream (`make_brain(seed=s,
persist=False)`, 800 ticks, `torch.set_num_threads(1)`, as L1b ran). All four keep HEAD's engine configuration: delta
online readout, T047 growth gate, `refit_every` as the brain sets it.

| arm | input rows `W` | role |
|---|---|---|
| FROZEN | never move (the HEAD incumbent) | control: the gap that data drift alone opens |
| FROZEN_B | never move | A/A: must equal FROZEN bit for bit |
| OJA | L1b's Oja step per wake percept (section 2.1) | the row L1b owes |
| RAND | the same step with `xn` replaced by a fresh random unit vector `r_t` per percept (same `post_i`, same mu, norm-preserving) | **magnitude-matched, direction-random drift**: separates "any movement under a stale reader" from "Hebbian alignment" (the retracted diversity hypothesis) |

**Endpoints per refit row r.**
- `acc(r)`: the row's own scored accuracy. On a closed-gate row this is scored through `_batch_beta`; on an open-gate
  row through the fresh solve. It is exactly the number the gate reads.
- `acc_sh(r)`: the shadow.
- `gap(r) = acc_sh(r) - acc(r)`.

**Per seed and arm** (last third of the refit rows, L1b's window, `:32-33`):
- `GAP = mean gap(r)`;
- `ACC_SH = mean acc_sh(r)`;
- `ACC = mean acc(r)`.

**Same rows.** The majority AND persistence floors must be identical across the four arms at every seed to 1e-6
(L1b's rule, `:103-106`). A seed whose floors differ is EXCLUDED and counted. If more than one seed in a look block is
excluded, that look is NOT READ.

### 2.3 The conjuncts: two primaries in a fixed-sequence (gatekeeping) order

**P1: staleness carries the drift cost.**
- Per-seed difference: `d1_s = GAP_OJA,s - GAP_FROZEN,s`.
- MET iff the sequential test of section 2.5 crosses its upper boundary for `d1 > 0`.
- Claimed effect: **0.10**. That is about half of L1b's measured mean OJA-vs-frozen loss (-0.196): "carries the harm"
  needs at least half of it. This is a derivation, stated as one.
- The difference-in-differences is the point. `l1b-synaptic-online.md:274-276` records that the frozen incumbent also
  collapses under a stale readout, from data drift alone. P1 asks whether moving `W` adds staleness beyond that.

**P2: no residual harm once the reader is re-solved.** Read ONLY if P1 is MET; otherwise NOT READ.
- Non-inferiority: `d2_s = (ACC_SH_OJA,s - ACC_SH_FROZEN,s) + 0.10`.
- MET iff the sequential test crosses for `d2 > 0`.
- The margin 0.10 equals P1's claimed effect, by the same derivation.
- Fixed-sequence testing (P2 only after P1) keeps the familywise rate at .05 with no split. This is the P1 amendment's
  multiplicity correction, named.

**Secondary readings.** These are reported with their t and exact FPR. They carry no bar and support no disposition:
- S1 `GAP_RAND - GAP_FROZEN`: does direction-random drift also open a gap?
- S2 `ACC_SH_OJA - ACC_SH_RAND`: is there a Hebbian-direction effect once the reader is re-solved?
- S3 LIVE: `live_accuracy` for OJA vs FROZEN. This is HEAD's co-adapting delta reader, `neurogenesis_engine.py:552-558`.
- S4 units and the gate-verdict sequence per arm. OJA admitted more units than the incumbent in L1b (`:281-285`).
  Units are reported, never read as growth (§10).
- S5 the replication read: `ACC_OJA - ACC_FROZEN` on the pilot seeds 0-7. This is L1b's own contrast, now on the fixed
  scoring path at HEAD. It is reported against L1b's t -1.959 and is not a verdict.

### 2.4 The drift readings (descriptive; preconditions marked)

These use a FROZEN PROBE SET: the first 60 wake percepts buffered after the first model exists in that cell, stored
once and re-presented at every refit row.
- **Rate drift, surviving units only.** `||S_r - S_{r-1}||_F / ||S_{r-1}||_F` over units present at rows r-1 and r.
  Identity is tracked through the leading-row rule and `last_keep` (`neurogenesis_engine.py:707-750`). Added units are
  reported separately, never inside the drift.
  - FROZEN must read **exactly 0.0**. The rows are frozen and the LIF rollout is stateless and deterministic, so any
    nonzero value is an instrument defect (red R4).
- **Flip rate.** The fraction of probe rows whose `decide` label differs between the row's scoring beta and `beta_sh`.
  This is the probe-set face of the gap.
- **Readout null dimension.** `L - rank(beta_sh)`, numerical rank at tol 1e-6 x sigma_max. The null-space split of the
  literature (Rule, O'Leary & Harvey 2019) is defined only where this exceeds 0:
  `||dS P_row||^2 / ||dS||^2`, with `P_row` the projector onto `col(beta_sh)`.
  - **Precondition, measured:** if the null dimension is 0 on every row of every cell, the split is **NOT READ**. The
    record then states the structural reading: at this width every hidden-rate change is readout-visible.
  - L1b's widths (2-15 units against several categories) make this likely. It is recorded, never assumed.
- Spearman correlation, flip rate against `gap`, across the rows of each arm. Reported only.

### 2.5 Statistic and sample size (P1)

**Unit and test.**
- The unit is the seed.
- The test is the paired one-sided t (`paired_contrast.paired_t`), ties at |d| <= 1e-6 counted as zeros (L1b's), with
  the exact sign-flip FPR (`permutation_fpr_detail`) printed beside every look's verdict.
- No majority-of-seeds rule. `rpe_charge.seq_decision`'s `K_MIN` seed-count clause is NOT adopted (§8.8); only its
  boundaries are.

**Stage P, pilot. A fixed count, one-shot, with its power stated: none.**
- Seeds 0-7: L1b's seeds, so S5 is a like-for-like replication read.
- The pilot measures the paired SDs `SD_1` of `d1` and `SD_2` of `d2`, and runs every red in the brain.
- Its d values are **never** pooled into the confirmatory test.

**Design, frozen by dated Amendment A1 after the pilot and before any confirmatory seed.**
- The fixed-sample target is `n0 = min n with (t_{.95,n-1} + t_{.80,n-1}) * SD_1 / 0.10 <= sqrt(n)`, computed with
  `pool_selection.t_quantile`.
- The looks fall every 8 seeds.
- The boundaries are `rpe_charge.seq_boundaries(looks, 0.05)`: Lan-DeMets O'Brien-Fleming-type spending, recursive
  integration. That function is imported and **not lifted**. A P2 lift into a shared module is its own commit and is
  not required by this row (disclosed).
- The cap is n0 times the design's simulated inflation, rounded up to a look.
- Futility is non-binding: conditional power under 0.10 below .10. It is taken only at a look before the cap whose mde
  from `SD_1` is at or below 0.10 (P1 amendment item 2; `rpe_charge.futility_open`'s rule).
- P2's power at the cap is computed from `SD_2` and stated in A1.

**Feasibility branch, pre-registered.** If the cap exceeds **40**, the exact FPR refuses (section 1), and the
confirmatory campaign **does not run**. The record states the pilot's SDs, the target, the cap and the mde reachable at
40, and the owner decides among:
- a larger claimed effect;
- an approved approximate FPR;
- stopping.

Nothing runs past 40 on this pre-registration.

**Simulation guard (P1 amendment), run before the first confirmatory seed.** 10,000 draws at `SD_1`, using the design
in A1:
- a planted null (delta = 0) must keep the stopping-rule FPR at or below .05;
- a planted delta = 0.10 must cross with power of at least .8 at the cap.

Both numbers go into A1. If either fails, the design is re-derived before data, not after.

**Seeds.** Confirmatory seeds are drawn in order from the block **17000-17039**, so the cap is at most 40.
- R3 uses 16000-16999 (`r3-replay-growth-composition.md:329`).
- G3's `SEED_BLOCK` is (9000, 10000) (`growth/growth_loop.py:410`).
- The board's `SEEDS.md`, which G3 cites, is not in the tree. **The operator confirms the block is free before Stage C.**
- Pilot seeds 0-7 are not reused.

**Null-scale position.** D1 runs at L1b's scale (2-38 units, 800 ticks). It is **NOT the 10x re-test** of the
NULL-RECORD in section 1: that axis is units, which means at least 380, and this stream has never grown past 38. D1
leaves that record's `retest=CANDIDATE` unchanged. What D1 can supply is a measured SD, and so an mde, for the triage
row's `mde=unknown`, at this scale only.

### 2.6 The reds: constructed inputs, each RUN before the reading it guards

The reds fall in three stages. Brain-free ones live at the build (fixtures, in the module's self-check and in
`tests/test_readout_staleness.py`). In-brain ones run at Stage 0, on one seed. Statistical ones run at A1.

| id | guards | green on | constructed red (must fire) |
|---|---|---|---|
| R1 | the shadow is a reading of staleness | brain-free: a closed-gate refit with no change to `W` and the same training rows reads `gap` = 0 to 1e-6 | a planted rotation of `W` (30 degrees per row, norm kept) between solve and scoring must read `gap` > 0 |
| R2 | the shadow never feeds back | substrate hashes identical with the shadow on and off, every row | a deliberately leaking variant (the shadow written into `_batch_beta`) must change a hash |
| R3 | the operators move what they claim | OJA and RAND: per-row norms preserved to 1e-6; Sum\|dW\| > 0 | mu = 0 must read Sum\|dW\| = 0 and be flagged INERT; a norm-breaking variant must fail the norm check |
| R4 | the rate-drift instrument | FROZEN surviving units read 0.0 exactly; a planted rotation reads monotone over 5, 15 and 45 degrees | permuting surviving-unit identity on FROZEN must read > 0 |
| R5 | the null split | fixture with L > m: drift planted in `null(beta^T)` reads a row fraction < 0.05; drift in `col(beta)` reads > 0.95 | swapping the projector must reverse both readings |
| R6 | RAND is magnitude-matched | the per-percept angular-step distributions of OJA and RAND agree, median within 10% on the pilot | RAND at 10 x mu must fail the match |
| R7 | same rows | identical floors across the four arms at a seed | one arm on a different stream seed must be detected and excluded |
| R8 | the A/A | FROZEN vs FROZEN_B bit-identical: accuracy, `W` hash, trajectory | a FROZEN_B with a one-ulp perturbation of one `b` entry must be caught |
| R9 | the statistic | simulation guard (section 2.5) | the planted null and the planted effect as above |

**The positive control for the disposition's claimed cause** (§8.8, §9) runs at **Stage 0, before the pilot**. The
cause under test is a stale readout, and its instrument is `gap`. On one seed, in the brain, the instrument injects R1's
planted rotation into the retained model at the first closed-gate refit. That seed's `gap` must exceed the same seed's
un-planted `gap`.

If that fails, the instrument is **BLIND**: no reading of P1 is taken, and the instrument is repaired or deleted. That
result never supports "staleness does not matter".

### 2.7 Disposition (fixed in advance)

- **INSTRUMENT BLIND.** Any of R1-R8 fails to fire, or Stage 0's positive control fails. No verdict. Repair or delete
  the instrument; nothing about L1b changes.
- **INFEASIBLE AT EXACT FPR** (section 2.5). The pilot record stands, and the owner decides.
- **P1 MET and P2 MET.** *Licensed:* "At 2-38 grown units, 800 ticks, mu = 0.05 ungated, on this stream, L1b's Oja
  cost to the growth gate's evidence is carried by the gate's stale readout. With the readout re-solved on the moved rows,
  Oja is not worse than frozen rows by more than 0.10."
  - Consequences:
    - (a) The L1b negative is scoped in its own record by a dated cross-reference, written by the operator: L1b's file
      is frozen evidence and D1 does not edit it.
    - (b) The `_batch_beta` staleness is recorded as a **measured hazard** for any future input-weight rule. Such a rule
      must re-solve, or must co-adapt, the gate's reader. This is a design constraint offered to G5 / T113's G-R5.
    - (c) Oja stays a re-test CANDIDATE. **Nothing is re-landed.**
- **P1 MET and P2 NOT MET, or P2 INCONCLUSIVE.** *Licensed:* "staleness carries at least 0.10 of the cost; a residual
  cost under a re-solved reader is not excluded". S1 and S2 are then reported as descriptive, never as a cause.
- **P1 NOT MET at a powered stop** (mde at or below 0.10). *Licensed:* "at this scale, a stale readout does not carry
  L1b's Oja cost by 0.10 or more". L1b's M-1 cause statement is then contradicted at this scale. The operator writes the
  dated cross-reference; the hypothesis has no module to delete.
- **P1 INCONCLUSIVE** (cap or futility without power). Nothing changes, and the mde reached is recorded.

**Refused in every branch:**
- "Oja helps the brain";
- "drift is harmless";
- "the engine compensates for drift like cortex";
- "sleep re-aligns the reader";
- any reading beyond the stated scale, mu, gating and stream;
- any statement about BCM, which is not an arm (section 2.8).

### 2.8 What this row does NOT claim or do

- It does not re-land Oja, BCM or the engine's input-rule hook. The operators exist only inside the instrument, on
  scratch homes.
- It is not the null-scale 10x re-test (section 2.5).
- It does not test BCM. BCM did not fall past its bar in L1b (`:235`). Excluding it keeps one primary family; its U4
  candidacy (`null-scale-triage.md:196-199`) is unchanged.
- It does not test a three-factor gate. The ACh gate was withdrawn at L1b's Stage 0 and remains a separate candidate
  (U2).
- It does not change the live engine. The shadow, the gate log and the operators run only in instrument child
  processes. HEAD's `_batch_beta` design (L1c Amendment 3) is not modified, whatever the verdict. Proposing that change
  would be a separate row.
- It does not report a unit count as growth (§10).
- It does not measure on the live stream or the deployed daemon.

### 2.9 Stream, host, cost

**Stream.** The in-brain stream L1b ran: `make_brain` on a scratch home, 800 ticks, the operator's perception packs.
L1b's BATCH floors per seed (`:256`) are compared at Stage 0 on seeds 0-7:
- agreement is reported;
- a disagreement is reported as "the stream moved since 2026-09-05" (§9: "a polluted baseline reads clean"), not as a
  finding;
- it does not abort, because every D1 contrast is within HEAD.

**Host.** The operator's host. cp314t is required (`__init__.require_free_threaded`). This cloud session can build and
run the brain-free reds R1-R5 and R8's fixture only if a cp314t venv installs here. Otherwise it hands them to the
operator, named.

**Cost.** L1b's cells ran 51-63 s each (`:171`). Stage 0 measures ONE arm-seed at campaign size before anything else:
- pilot: 4 arms x 8 seeds = 32 cells;
- confirmatory: at most 4 x 40 = 160 cells.

At about 1 minute per cell and 4 at a time, the cap is roughly 45 minutes of wall time. The live lake's mtime and digest
are checked before and after every campaign (R3's K14 pattern).

### 2.10 Commit order (one change each; the reds before any verdict)

1. **This pre-registration, alone.**
2. The build:
   - `instruments/readout_staleness.py` and its brain-free reds R1-R5 and R8;
   - `tests/test_readout_staleness.py`;
   - registration: `REGIONS`, `run_selfchecks.CORE` (self-check under 60 s, no brain), `dormancy_gate.EXEMPT`
     (`permanent`), `GRADIENT_CLAIM`, `EXEMPT_METHOD`, and the census adjudication if one is required.
3. Stage 0: R7, the in-brain R8, Stage 0's positive control, the cost of one cell, and the stream-agreement report.
4. Stage P: the pilot (seeds 0-7), S5's replication read, R6, the SDs.
5. Amendment A1: the design, R9's simulation guard and the feasibility verdict. **Dated, before any confirmatory seed.**
6. Stage C: one commit per look, carrying that look's decision.
7. The verdict and disposition (section 2.7), then panel, fold and closing report (§6.2).

The operator's battery and push follow the project's rhythm.

### 2.11 Amendments (dated, appended; nothing above is edited in place)

#### A0 -- 2026-10-27: territory measured after the freeze, before the build and before any pilot seed

**What was run, and what was not.**
- A cp314t environment was set up in the cloud clone:
  - CPython 3.14.6 free-threaded; `Py_GIL_DISABLED` = 1, `_is_gil_enabled()` = False;
  - torch 2.14.1+cu130 from PyPI (the CPU wheel index is refused by this environment's network policy);
  - brian2 2.10.1, mujoco 3.11.0, numpy 2.5.4;
  - `pip install -e ".[dev,nir]"`.
- **P5 hazard.** G3 T0's capture ran torch 2.11.0+cu128 (`g3-t0-baseline.md:161`). Nothing measured here is campaign
  data.
- The bounded pre-checks at `6612ce7` (plus this file) all returned 0: ruff check, ruff format (src tests tools),
  `red_reachability --assert`, `dependency_ledger --assert`, `citation_guard`, `doc_command_guard --assert`,
  `doc_budget_gate --assert`, `dormancy_gate` and `method_dormancy_gate`. So did the self-checks of `growth.ng_snn`,
  `growth.neurogenesis_engine`, `growth.grower`, `readout.online_readout`, `instruments.paired_contrast`,
  `instruments.rpe_charge` and `instruments.pool_selection`.
- A **feasibility probe** ran FROZEN only, on seeds 96-99, 800 ticks each.
  - These seeds are outside the pilot (0-7) and outside the confirmatory block. They are excluded from all D1 data.
  - No OJA or RAND arm was run, and no accuracy, contrast, gap or shadow was read.
  - The probe read four things: the tier; per wake percept, whether any grown unit fires; per refit row, the gate
    state, the units, the categories in the buffer, and `L - rank(beta)`.
  - Its positive control ran in the same command: an all-zero `W` reads 0 firing units, and it did.
- **Tier.** Every probe cell ran `perception_tier = "stdlib"`, the no-encoder tier. This clone has
  `models/nmnist_pack.npz` and `shd_pack.npz` (tracked) but not the gitignored `eprop_*.pt` encoders.

| seed | percepts with a model | any unit fires | units | categories | refit rows | closed-gate rows | max `L - rank` |
|---|---|---|---|---|---|---|---|
| 96 | 377 | 1.000 | 4-6 | 4-8 | 20 | 0 | 2 (open rows, ticks 40-80) |
| 97 | 376 | 1.000 | 3-5 | 4-7 | 20 | 11 (sticky from tick 200) | 1 (tick 20); 0 on every closed row |
| 98 | 379 | 0.934 | 3 | 4-7 | 20 | 0 | 0 |
| 99 | 375 | 0.864 | 2-7 | 7-10 | 20 | 0 | 2 (tick 20) |

**Consequences, ruled before the build** (the session's calls, disclosed).

1. **P1's endpoint is re-defined.** As frozen in section 2.2, `gap` is read after each refit. On an open-gate row that is
   a fresh solve, so the gap is about 0 by construction. The gate closed on 1 of 4 probe seeds, so the frozen endpoint
   would mostly measure open rows and dilute any effect toward zero. **That would not be a test of the mechanism.**
   - **The new endpoint is the PRE-REFIT gap.** At the start of every refit row, before `_refit` changes anything, the
     instrument takes the split `marginal_holdout` is about to produce on the current buffer. It scores those held-out
     rows twice on the current `W`:
     - (a) with the gate's reader as it stands (`_batch_beta`, the last open-gate solve);
     - (b) with the shadow re-solve on this split's training rows.
   - `gap_pre(r) = acc_b - acc_a`. It exists on every row, open or closed, because it measures the staleness that
     accrued over the interval since the last solve.
     - On FROZEN, only the buffer moved, so it is data-drift staleness.
     - On OJA and RAND, both the buffer and `W` moved.
   - P1 keeps its contrast (`GAP_OJA - GAP_FROZEN`), its claimed effect (0.10) and its window (last third), read on
     `gap_pre`.
   - P2 uses `acc_b` from the same pre-refit reading: the re-solved reader on the moved rows.
   - The original post-refit closed-row gap becomes **S6**, reported with each arm's closed-row count (S4). Gate
     dynamics can differ by arm, and L1b's OJA admitted more units.
   - R1 and R2 apply unchanged to the pre-refit reading. R2 asserts that the reading leaves the split, the buffer, the
     model and every RNG exactly as `_refit` would have found them.
2. **The null split will mostly read NOT READ.** `L - rank(beta)` was 0 on every closed row and on most open rows. That
   is the structural reading section 2.4 anticipated. R5 stays, as the split's own positive control.
3. **The rule will act.** Grown units fired on 86-100% of percepts, so OJA's `post_i = s_i` is not structurally zero at
   this tier. R3's INERT check still runs on the operator's tier.
4. **Tier rule.** The pre-registered stream remains the operator's tier (section 2.9). Every cell records
   `perception_tier`. A cell on a different tier from its seed's other arms is EXCLUDED and counted, under the same rule
   as same rows. The stdlib-tier runs this clone can make (the build's smoke) are never data.
   - The E1 lesson applies (`learning-rules-1019.md:31-35`): a worktree without the encoders silently builds the
     no-encoder brain.
5. **Engine binding survives the one-grower merge.** T111a will retire `NeurogenesisSelfImprove` into `Grower`
   (`tasks.md:251`).
   - The instrument binds through `brain.grower.percept.engine`, whose forwarding is dynamic
     (`growth/grower.py:16-17`), not through the class.
   - It reads only `_refit`, `_model`, `_batch_beta`, `trajectory` and `_buf_x` / `_buf_y`.
   - If T111a lands before Stage 0, the binding is re-stated by a dated amendment before any cell, with every
     endpoint's meaning unchanged.
6. **D1 is not E3.** `learning-rules-1019.md:79-80` orders E3 after the grower merge and E2. E3 is a three-factor
   input-row rule against FREEZE, with an M-shuffle control and the gate verdict per row.
   - D1 evaluates no candidate rule for landing. It measures the gate reader's staleness under L1b's deleted rule, and
     its result is offered to E3's design.
   - Running D1 before E3 rests on the owner's request of 2026-10-27, disclosed here. E2's block ("one silent unit")
     is the deployed brain's (`l-exit.md:96-97`), not the scratch brains', where units fire (table above).

#### E1 -- 2026-10-27, erratum: one line of section 1 reformatted (an in-place edit, disclosed)

**What broke.** Section 1 quoted the triage's null-scale record verbatim, as an indented line starting
`NULL-RECORD:`. `gates/null_scale_gate.py` parses every such line as a record, read `mde=unknown` as a non-number
and turned `--assert` RED (R-FIELD) on this branch, a lint step of the battery.

**The fix.** The line is now a prose paraphrase carrying the same fields. It has no `NULL-RECORD:` prefix, and no
content changed. After the edit, `python -m snn_data_lake.gates.null_scale_gate --assert` is green.

**Why an in-place edit.** This is the one in-place edit to the frozen text. It is made because the quotation was
formatting in the territory section, not a criterion, and because the alternative was a permanently red gate.

#### A0b -- 2026-10-27: the build path, measured before the build (scratch probes; no D1 data)

A read-only audit of the build path ran engine-level probes on synthetic chains plus one scratch `make_brain` at the
stdlib tier. Seeds 0, 1 and 96, all engine-level or scratch; the 0 and 1 here are synthetic-engine seeds, not the
pilot's brain seeds. No arm, gap or contrast in this file came from them. It found three things the pre-registration
had not accounted for, ruled here before the build.

1. **A no-drift gap is not zero after a prune.**
   - When the last open-gate solve ran `prune_unearned`, `_batch_beta` is the beta re-solved on prune's own internal
     split (`growth/ng_snn.py:378`, `growth/candidate_lifecycle.py:218-219`). A shadow solved on the refit's training
     rows differs from it even with `W` unchanged: max |d beta| 0.02-0.32 and a gap of -0.039 to +0.077 on the
     synthetic chain. Without a prune, the shadow equals `_batch_beta` bit for bit.
   - Prunes fired on most open solves. Prune frequency depends on units, and OJA admitted more units in L1b, so the
     bias need not cancel across arms.
   - **Ruled:**
     - (a) At every open refit, the instrument records a CALIBRATION reading: the shadow against `_batch_beta` on that
       refit's own split, with the same `W`. It also records whether that solve pruned.
     - (b) `gap_pre` on each later row is reported raw and CALIBRATED, i.e. minus the calibration reading of the open
       solve that produced the reader in use.
     - (c) **P1 reads the calibrated `gap_pre`.**
     - (d) R1's green fixture is a solve that did not prune. R1 gains a second green: a pruned solve reads a
       calibrated gap of 0 to 1e-6 with `W` unchanged.
2. **A vocabulary term.**
   - `_batch_beta`'s width is fixed at its last open solve (max label + 1, `growth/ng_snn.py:395`). The live beta is
     widened online (`readout/online_readout.py:83-88`). The shadow takes `int(y_tr.max()) + 1`.
   - Held-out rows whose label the stale reader cannot output therefore add a staleness term unrelated to `W` (7-10
     of 60 rows on a probe row).
   - **Ruled:** each row logs `unseen_by_reader`. P1 is read on rows the stale reader can answer, with the all-rows
     reading reported beside it. The shadow mirrors the closed branch's reader fallback exactly
     (`growth/neurogenesis_engine.py:668-670`).
3. **Faults vanish silently.** `Brain._sleep_tick` swallows any exception from the grower's tick (`engine/brain.py`
   around `:3356-3360`); a planted raise in a wrapped `_refit` dropped one row silently. **Ruled:**
   - the wrapper keeps its own fault counter;
   - every row asserts that the trajectory advanced exactly as `_refit` alone would have made it advance;
   - a cell with any instrument fault is EXCLUDED and counted, and an arm with faulted cells on more than 1 seed in a
     look block makes that look NOT READ.

**Governance, corrected before the build.**
- `EXEMPT_METHOD` (section 2.10) is INERT for this module. `method_dormancy_gate` scans only modules reachable from a
  live root (`gates/method_dormancy_gate.py:293-298`), and an EXEMPT-permanent instrument is not one. It is dropped
  from the build.
- The OJA and RAND operators are declared **LOCAL** in the module's `GRADIENT_CLAIM`. They are local rules, and
  CONTROL would mislabel them. Each gets a `_CENSUS_ADJUDICATED` entry in `plasticity/rule_composer.py`, copying the
  `instruments/g5_part_d_adaptors.py` precedent, and `tools/rule_census.json` is regenerated through
  `rule_composer --regen-census` in the build commit.
- The shadow and the drift readings are declared `NOT_A_RULE`; the re-solve is `DIRECT_SOLVE`.
- Helper names avoid the census's learn-verb substrings (`fit`, `train`, `grad`, `update`, `learn`, `adapt`) unless
  declared.

**Determinism, made explicit.**
- The shadow path draws no RNG. Probed: torch global, Python `random`, numpy and the model's own generator states
  are identical before and after.
- It does write `NGSNN.last_abstentions_` and `last_unanswerable_` through `decide`. Nothing reads those before
  `_refit` overwrites them, but the instrument saves and restores them anyway, and R2's hash covers them.
- RAND draws from its own CPU `torch.Generator`, seeded by `zlib.crc32` of (seed, arm), never the global generator.

#### A0c -- 2026-10-28: relayed to the integrator; seed block and name changed before registration (pre-build, pre-pilot; no data)

Written by the int3 integrator (board claude-engine-1024), not by this file's cloud author. That author cannot be
messaged; its work reaches the integrator only as pushed branches.

**Owner, verbatim** (session B's chat, relayed at board #5611): "Yes, relay D1". The button the owner chose carried this
text: "Q3: yes, relay D1 (origin d1/readout-staleness-prereg @ 2e946d8d) to A. Register it as its own row after a dated
amendment fixes F1 (seed collision with OG1) and F2 (the name), with an enforcer review of the full pre-reg before any
run."

**How it arrived on int3.**
- The four commits from 6612ce72 to 2e946d8d were cherry-picked onto claude/int3 after bd46b17d (base origin/main
  c1df3887).
- The file above this amendment is byte-identical to 2e946d8d (`git diff 2e946d8d HEAD` on it is empty).
- The branch later gained 2875259c (a session record only). It is not brought.

**F1. The confirmatory seed block moves** (integrator P2 check #5602; policy-enforcer #5605 re-derived it).
- Section 2.5's block 17000-17039 overlaps OG1's registered blocks on origin/main (`og1-organoid-stream.md:191`: "Dev
  17000-17007, confirmatory 17010-17017").
- From this amendment on, confirmatory seeds are drawn in order from **22000-22039**. The cap stays at most 40.
- Grep check, 2026-10-28, on int3: 0 seed/block/range hits for 22000-22999 in evidence, src, tools/*.py and tests.
  Board claim #5629 holds 22000-22999.
- Blocks avoided: OG1 17000-17017, R3 16000-16999, sleep_plasticity_ab 16900-16999, G3 9000-9999, G5 11000-11999, E1b
  8-183, and the E1c draft's 21000-21999 (claim #5613).
- The pilot seeds 0-7 and the A0 probe seeds 96-99 are unchanged.
- Section 2.5's own clause "the operator confirms the block is free before Stage C" now reads against 22000-22039.

**F2. The name changes, to avoid two existing uses.** "D1" already names G5 Part D's dev cohort and its freeze (57 uses
in `g5-readout-variants.md`) and specs/011's design item D1. "D-R1" reads like a Part D red (PD-R*).
- From this amendment on, the row is **L1b-STALE**.
- Task **T121** is unchanged: it is free on int3 and is this file's own proposal.
- Red family **LS**, row **LS-R1**. On 2026-10-28, `LS-R` has 0 uses in PLAN-FINAL and 0 in evidence.
- In the frozen text above, "D1" means this row and nothing else. The file name stays, so citations hold.

**F3. One drift-reading function** (#5602 F3, widened by policy-enforcer #5605 m1).
- Three claimants would read activity drift on a fixed probe set: this row's section 2.4, G5 Part E's P-E0, and G9's
  criterion-1 instrument (Part E E.8 item 2).
- Whichever builds first owns ONE drift-reading function, and the others import it.
- Its correlation primitives come from `infra/_mathutils` (`pearson`; a Spearman is lifted there if needed), never from
  a new local copy.

**F4. Territory at c1df3887.** Arc 3 changed `instruments/rpe_charge.py` in five places: its docstring, E1b's
constants, `analyse`, `_series_differ` and `main`. `seq_boundaries` and `seq_decision`, which section 2.5 imports, are
unchanged.

**Owed before any run:**
1. A policy-enforcer review of this WHOLE file (A0, E1, A0b and A0c; enforcer #5605 m2).
2. The registration rows (tasks.md T121; Part F LS-R1).
3. A1 after the pilot, as section 2.5 states.
4. The host reboot.
5. Nothing holding `resource:timing`.

#### A0d -- 2026-10-28: the cap is no longer bounded at 40, and the pins are scoped to one host (pre-build, pre-pilot; no data)

Written by the int3 integrator. The two decisions below were taken by the launcher session under the owner's delegation,
board-noted as `launcher-1028` (`lanes\DELEGATED-DECISIONS-1028.md`, Round 2, D-11). The owner's delegation, verbatim
from the launcher's chat on 2026-10-28 (verified by A from that session's transcript): "You decide whatever happens in
the two nested session, only report to me here". **These decisions are the delegate's, not the owner's words.** The
owner can reverse either one. Both answer questions the cloud author's session record (2875259c, section 9) addressed to
the owner. Nothing else in that record changes this file (D-10).

**Cap (D-11 #3).** Section 2.5's feasibility branch ("If the cap exceeds **40** ... the confirmatory campaign **does not
run**") is RETIRED.
- The cap is what section 2.5 already derives: n0 from the pilot's `SD_1` by power (.8 for 0.10), times the
  design's simulated inflation, rounded up to a look. It is a P1 sequential design with no fixed bound.
- The confirmatory block 22000-22039 (A0c) becomes the block's FIRST 40 seeds. A cap above 40 draws on in order
  through 22999 (claim #5629). A cap above 1000 is refused and goes back to the owner.
- **The verdict is unchanged:** the paired one-sided t against `rpe_charge.seq_boundaries`, as frozen. Its validity
  rests on the frozen simulation guard, not on the FPR printed beside it.
- **The FPR printed beside it, past 40.** `paired_contrast.permutation_fpr_detail` refuses n > 40 (`MITM_MAX_N`) by
  design. Past 40, the line prints a seeded Monte-Carlo sign-flip FPR:
  - 200,000 sign-flip draws, each seeded by `zlib.crc32` of the row and look;
  - the printed SE is at most .0005 at .05;
  - it is labelled "MC" and is never presented as exact.
- **ONE mechanism (P2).** The Monte-Carlo mode is ADDED to `instruments/paired_contrast` beside the exact path. It is
  not a local copy in this row's instrument. It owes a graph-exists check and its own red: on fixtures with n <= 40 it
  must agree with the exact FPR within 3 SE, and a planted shift must read a small FPR. It is built in this row's build
  commit, before any confirmatory seed.
- The INFEASIBLE AT EXACT FPR disposition branch of section 2.7 is retired with the feasibility branch.

**Pins (D-11 #2; the delegate asked A to decide it here, under enforcer review).** Scope: this row only. The
repo-wide decision 2 of the cloud record stays parked (D-10).
- **Decision:** every run of this row, brain-free reds included, is on ONE host: the operator's Windows host, on the
  production tier.
- Any byte-identity check this row asserts (R2's substrate hash, the RNG-state probes, the A/A identity) is a
  **Windows-host pin**. On any other host it is SKIPPED, with the reason "host-dependent float fingerprint, record
  2875259c section 5.2", never asserted.
- **Reason.** The cloud record measured float fingerprints that differ between Windows and Linux with unit counts
  equal, cause unproven. A pin that holds on one host and not another is that host's pin, and asserting it elsewhere
  would red on the host, not the method.

**Not changed:**
- section 2.5's pilot (seeds 0-7) and looks every 8 seeds;
- the simulation guard;
- futility;
- P2's non-inferiority read, which A1 states;
- A0c.

#### A0e -- 2026-10-28: fold of policy-enforcer #5637 (pre-build, pre-pilot; no data)

Written by the int3 integrator. It folds the enforcer's CHALLENGE on the whole file (#5637: 0 BLOCK / 1 MAJOR / 2 MINOR).

**M1, pin scope (P8: no red may be unreachable where the battery runs).** A0d's "Windows-host pin" is NARROWED to
comparisons against a stored digest recorded on that host.
- R2 (shadow on against off, same run), the RNG before/after probes (same process) and R8's FROZEN/FROZEN_B identity
  (same host) are RELATIVE comparisons. A cross-host float difference cancels inside each of them.
- They are therefore ASSERTED ON EVERY HOST, the Linux CI lint job included.
- Only a stored cross-host digest, if one is ever kept, is skipped off its host, and the reason is printed.

**m1, arm-dependent row filters (a validity read, pre-data).** A0b's P1 reads rows the stale reader can answer and
subtracts an open-solve calibration. Both depend on the gate's dynamics, which differ by arm. For every arm and seed the
record therefore prints:
- the fraction of last-third rows excluded as unanswerable;
- the count of calibrations taken at a solve that pruned.

Balance read on the pilot: if OJA's mean excluded fraction differs from FROZEN's by more than 0.10 (absolute), the
pre-registered ALL-ROWS gap becomes P1's reading and the A0b reading moves beside it. A1 records which one leads before
any confirmatory seed.

**m2, sizing from an 8-seed pilot (P1 item 1, read conservatively).** A1 sizes n0 on the pilot's 80% upper confidence
bound of `SD_1`: SD_1 x sqrt(7 / chi^2_{.20,7}), about 1.35 x SD_1. The point SD is not used.
- The simulation guard runs at that upper SD with normal draws.
- It also runs one heavy-tail stress row: t with 3 df, scaled to the same SD.
- Power at the cap is stated for both rows. A cap below .8 on the stress row is recorded beside, and the cap is not
  silently raised past A0d's 1000 refusal.

**Unchanged:** the verdict, the boundaries, the futility rule, A0c and the remainder of A0d.

#### A0f -- 2026-10-28: registration, one wording fix and a cross-campaign seed note (pre-build, pre-pilot; no data)

Written by the int3 integrator. Policy-enforcer AGREE #5642 (on A0e: 0/0/0, 1 nit) cleared the registration rows. The
owner's condition, "an enforcer review of the full pre-reg before any run" (#5611), is met by #5637, #5638 and #5642.

- **Registered.** tasks.md now carries T121 (row L1b-STALE, red family LS, row LS-R1). Its runs queue behind E1c
  (T122). Like every SNDL brain campaign, they run one at a time under delegate D-13 REVISED.
- **Wording (#5642 nit).** In A0e m1, "the pre-registered ALL-ROWS gap becomes P1's reading" means the CALIBRATED
  all-rows gap. A0b's calibration is kept in the switch, and the raw all-rows gap is reported beside it.
- **Cross-campaign seed note.** `engine/brain.py` builds `GridPlaceMap(seed=self._seed)`, and grid module mi seeds its
  cells with run seed + 1000 x mi.
  - This row's run seeds 22000-22039 (and through 22999 under A0d) are also the mi = 1 sub-seeds of E1c's runs
    21000-21703 (T122).
  - Within this row, offsets of 1000 or more never land on another run of the same block's first 1000 seeds.
  - Each campaign's contrast compares arms of the SAME seed, so neither result is affected. Disclosed; no block moves.

#### A0g -- 2026-10-28: arc-4 panel fold (pre-Stage 0, pre-pilot; no data)

Written by the int3 integrator. It folds arc-4 panel lenses B (#5721) and C (#5720) on this file. No rule that a
result has been read against changes, because no Stage 0, pilot or confirmatory data exist.

**C M2. Stage 0's positive control is re-keyed to the endpoint P1 reads.**
- Section 2.6's Stage 0 control injects the planted rotation "at the first closed-gate refit" and reads the post-refit
  `gap`. A0 demoted that gap to S6, and A0b made P1 the CALIBRATED `gap_pre`. A0's own probe found zero closed-gate
  rows on 3 of 4 seeds, so the control as written might never fire, and it guards S6, not P1.
- Re-keyed: inject the planted rotation right after an OPEN solve. At the next refit row, whatever the gate state,
  require: calibrated `gap_pre` (planted) - calibrated `gap_pre` (unplanted) > a MARGIN, same seed and same row. A1
  fixes the margin from the pilot's calibration SD before any confirmatory seed (panel A #5748 m7: a bare ">" can pass
  on noise).
- Section 2.6's consequence is KEPT (panel C #5747 M1): if the control fails on its named seed, the instrument is BLIND,
  and no P1 reading is taken. Whether the control fires on other seeds is recorded as a validity read, counted under
  section 2.2's per-look exclusion cap (more than one excluded seed in a look block: that look is not read). It never
  filters which seeds P1 reads.

**B m6. Futility opens on the same SD the cap is sized on.**
- A0e sizes n0 on the pilot's 80% upper SD (about 1.35 x SD_1), while section 2.5 opens futility at the first look
  whose mde on the POINT SD reaches the claimed effect. Futility could therefore open before the fixed-sample target,
  which is how E1b ended INCONCLUSIVE.
- Fixed (P1 item 2): futility opens only at a look whose mde, computed from the UPPER SD that sizes the cap, is at or
  below the claimed effect. A1 states that look before any confirmatory seed.
- A1 also owes (panel B #5746):
  - if no look before the cap qualifies, futility is inert, and A1 says so;
  - the decision function is named as `rpe_charge.seq_decision` with this row's parameters (`k_frac=0.0`: no sign
    guard), so the build writes no second sequential rule (P2);
  - whether this row uses E1c's `futility_powered`, decided with its own red if adopted.
- Why E1c differs (panel C Dm7): E1c's futility first opens at 640, past its upper-SD fixed-sample target (601). So its
  point-SD rule never opens futility before the target. This row's point-SD rule could, so it moves to the upper SD.

**C8. Where the brain-free reds run.** A0e supersedes A0d's "one host" for the RELATIVE reds (R2, the RNG probes, R8):
they are asserted on EVERY host, the Linux CI lint job included. Only a stored cross-host digest is a Windows-host pin.

**C9 / B m6. The host reboot is retired.** A0c's owed "host reboot" is withdrawn. The owner said on 2026-10-28, in the
integrator's chat: "I cannot reboot PC since I'm working remotely". Runs use the D-13 REVISED and D-14 throttle instead.

**C15. A drift-function claimant was missing.** A0c F3's list of drift-function claimants omits the module that
research/representational-drift/MAP.md's D0 plans. It is added. When that module is built, P2 decides between it and
this row's readings.

**Build notes (wf-rv #5728, the integrator's review).**
- `GRADIENT_CLAIM` lives inside `instruments/readout_staleness.py` (the per-module shape), not as an integrator row.
- `instrument_engine` has NOT been run against the real engine or `marginal_holdout`; only its fixture engine is tested.
  That run is owed to Stage 0, before the pilot.
- The build is PARKED (delegate D-28). Branch `claude/l1b-stale-r3-1028` (edd29da4) carries over; the lane's original
  is kept at `claude/l1b-stale-1028-lane-orig` (2eef0119). Still open (policy-enforcer #5745 nit 2):
  - #5740's buffer digest (`_buf_x` / `_buf_y` in the substrate hash);
  - the real-engine `_inputs` split equivalence test (#5718 m2, carried to Stage 0);
  - the per-commit lint fold of round 3;
  - the registration rows and the census regeneration.

**Unchanged:** the verdict, the boundaries, A0c's seed block, A0d's cap rule, A0e's sizing on the upper SD and its
balance read, and A0f.

## 3. Record

*(empty until Stage 0)*

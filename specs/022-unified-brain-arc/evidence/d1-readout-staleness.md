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

**The scale and the null-scale triage.** `evidence/null-scale-triage.md:69-70` records OJA as follows:

    NULL-RECORD: Oja local rule on the grown units' input rows (L1b) | units=38 | ticks=800 | seeds=8 | mde=unknown | retest=CANDIDATE

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

*(none yet)*

## 3. Record

*(empty until Stage 0)*

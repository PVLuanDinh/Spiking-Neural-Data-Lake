# Phase 2's L1c — the readout learns online (2026-09-06)

The plan row (`PLAN-FINAL-2026-09-02.md` Phase 2, L1c: "Adopt an existing online local readout rule —
`laminar_local_readout.observe`'s single-sample local update, or `FrozenExpert`'s three-factor rule — P2: one
survives, named at pre-registration — with its state as a lake record"), the owner's 2026-09-05 standing
authorization (every decision below is the session's call under it, disclosed where it departs from a letter),
and what L1b just taught (`evidence/l1b-synaptic-online.md` §4): the grown readout `beta` is batch-solved only at
open-gate sleep refits and is STALE on every wake tick between them and through every closed-gate period — the
mechanism behind L1b's measured harm, and the reason the readout must co-adapt before any input-weight rule
returns. Lessons carried: a mechanism that touches the growth gate's evidence changes the admission path (the L1b
twin) — every such leak is measured by W-hash divergence and disclosed; a control that is not the method (a
SIGN-FLIPPED update) beside the reach control (the update neutered must equal the incumbent bit for bit); same
rows by measurement; the closing record commit is the last commit before the push.

## 1. Territory (code at `1e31239`, read 2026-09-05/06)

- **The readout today.** `NGSNN.beta` (L, m) is solved analytically (`_solve_beta`: an activity-regularized ridge
  on the LIF rates `S`) inside `fit` (per candidate block and once on all rows at the end), re-solved after
  `prune_unearned`, and never touched between sleep refits. `predict(X) = argmax(_rates(X) @ beta)`. The
  engine's `_refit` runs every 20 SLEEP ticks; a closed T047 gate scores the RETAINED model without a re-solve.
  The LIVE instrument (kept from L1b) scores `predict(x_{t-1})` against `cat_t` on every wake percept BEFORE the
  transition is buffered: on the incumbent it reads the stale beta (live accuracy 0.31–0.95 against a live
  persistence floor 0.93–0.98; `l1b-synaptic-online.md` §3.2).
- **Candidate A — the laminar L5 rule** (`cortex/laminar_local_readout.train_step`, the tree's ONE live-tick
  no-transport readout: `W2 -= lr · deltaᵀ h_mod / n` with `delta = softmax(logits) − onehot`; `observe` runs it on
  one sample at `lr = mu / EMA(‖[a4, 1]‖²)`, mu = 1.0 CHOSEN there, P1-derived normalizer; declared LOCAL).
  Ported onto NGSNN's tensors: `beta -= lr · sᵀ (softmax(s @ beta) − onehot(cat))` with `s = _rates(x_{t-1})`.
- **Candidate B — FrozenExpert's error-sign perceptron** (`substrate/similarity_sized_growth.FrozenExpert.learn`:
  on a misclassified sample `W[k] += lr·x` (LTP toward the should-have-fired class) and `W[pred] −= lr·x` (LTD);
  a 30-epoch batch loop with lr = 0.5 fixed; declared LOCAL "error-modulated Hebbian is three-factor"; the module
  lives in `substrate/`, not `cortex/` as the plan row says; no live path constructs it). Ported as the
  single-sample step on beta's columns: on `pred ≠ cat`: `beta[:, cat] += lr·s`, `beta[:, pred] −= lr·s`.
- **Both are local to the output synapses** (the error is the output layer's own; no weight is transposed, no
  hidden error is formed): LOCAL rows, no cosine owed (POLICIES e-prop SCOPE). Neither rule reads `W`.
- **The lake record.** `beta` already rides in the growth record (`growth_state_bytes`: W, b, beta, acc, floors,
  the trajectory tail); a wake-learned beta reaches the lake at the sleep flush (`_flush_growth`) and `close()` —
  the same lag as the substrate itself (disclosed). The record's `acc` keeps meaning the last refit's held-out
  accuracy of the weights at that fit.
- **P2 and the batch solve.** The plan's L1b row said the batch solve is "kept ONLY as a declared control off the
  live tick, else deleted"; L1b's record §1 scoped that honestly: `fit`'s internal solve is the ADMISSION
  instrument (it forms the residual the greedy pursuit fills) and stays; the SHIPPED readout is what L1c takes
  over between refits. What survives here is ONE readout, `beta`, moved by ONE online rule on every wake percept
  and RE-INITIALIZED by the batch solve at every open-gate refit (the sleep consolidation — the plan's own
  "consolidation stays in sleep", L2 row). That is a schedule composition, not two mechanisms for one job:
  between refits only the online rule writes beta; at a refit only the batch solve does. The session's call.

## 2. Pre-registration (written before any cell runs; the build follows it)

### 2.0 The stimulus, measured before the build (Part C: the world before the wire)

The engine's schedule emulated brain-free on the real `NGSNN` (a buffer of (percept_{t-1} → cat_t), a grow-on-top
refit on the whole buffer every 20 observations, the live instrument scored from the retained model BEFORE the
online step, the online readout continuing from each fresh batch solve; 4 seeds × 600 observations; archive
`l1c-2026-09-06/` this section's transcript):

| chain | batch-only (stale between refits) | DELTA | SIGN | DELTA sign-flipped |
|---|---|---|---|---|
| DRIFTING (the succession rule flips every 100) | 0.445 | **0.638** | 0.624 | 0.288 |
| STATIONARY | 0.886 | 0.843 | 0.800 | 0.589 |

The online readout is worth +0.19 when the batch solve goes STALE against a drifting world and costs 0.04–0.09
when the batch solve is fresh on a stationary one; the sign-flipped update destroys accuracy on both (the control
that is not the method has a reachable red). **Expectation for the live stream, stated before the cells:** the
brain's stream is persistence-shaped and stable (W2, L1a, L1b), the buffer holds every transition and a refit
lands every ~20 fed percepts, so the batch beta is rarely stale — L1c-R1 may read NOT MET in-brain, and the
non-regression R4 is the guard against the stationary cost. That is a reachable red in both directions; the
module's self-check carries the DRIFTING chain as its positive control and the STATIONARY loss as its disclosed
negative. On a null the module is deleted (P8 clause 5) — a capability the live world does not consume is
dormancy — and the drifting-chain numbers stay in this record as the condition under which the row re-opens
(a world that drifts).

### 2.0b Amendment 1 — the control re-run on the ENGINE's substrate FAILED at the ported step (written before
any further run; commit 2)

§2.0's emulation fitted `NGSNN` directly (T = 8, no prune, no marginal holdout) and grew 53–100 units; the real
engine on the same drifting chain (`NeurogenesisSelfImprove`, refit every 20 observations, `_refit`'s admission
and `prune_unearned`) grows 1–14 units. On THAT substrate the module's self-check — the drifting-chain positive
control, seed 0 — reads: batch-only 0.400, DELTA 0.386, SIGN 0.379 (mu = 1.0, the faithfully ported step). Per
refit interval the online readout ADAPTS after each regime flip (0.30 / 0.25 / 0.35 / 0.30 against the batch's
0.00 / 0.00 / 0.00 / 0.00 in the four intervals after the first flip) and THRASHES each fresh batch solution
(0.45 against 0.80; 0.05 against 0.45 right after refits): with ‖s‖² of 1–5 on a 2–5-unit readout, the
normalized step of mu = 1.0 is a near-full overwrite per sample (laminar's own effective step is 0.05 because its
presynaptic energy is ~20). **The positive control fails on the substrate the mechanism will run on.** The
emulation's +0.19 was a claim about a different substrate and is withdrawn as this arc's control.

**One derived step, pre-registered here, one shot:** the online readout's memory is ONE REFIT INTERVAL — the
batch solve consolidates at every open-gate refit, so the online rule need only carry what happened since the last
one — hence `mu = 1 / n_refit`, with `n_refit` the fed percepts per refit interval as the engine's own cadence
gives it (20 in the harness and in-brain: `refit_every = 20` sleep ticks against ~20 fed wake percepts per interval
in the L1b cells). `lr = (1/20) / (‖s‖² + 1)`. Not tuned against the red: it is the first and only step tried after
the ported one. **Decision rule for the re-run (seeds 0 and 1 of the self-check, DELTA and SIGN):** if the
drifting-chain control clears at this step (online > batch-only + 0.05 on both seeds, the sign flip below), the
in-brain cells run at it and the module's self-check carries it; **if it fails too, L1c is REFUSED ON MEASUREMENT
before any in-brain cell — the module is deleted, the record stands, and the negative is: on the engine's
substrate an online single-sample readout cannot beat the batch solve's consolidation even on a drifting world.**
Also disclosed: the widening check compares matmul outputs, which agree to float tolerance, not bitwise (the
self-check and test use `allclose`, atol 1e-6); the engine constructs the readout from a kind name so the module
is reachable from a live path before the survivor lands (the dormancy gate's reading, as in L1b).

### 2.0c Amendment 2 — the refusal withdrawn; the regimes measured; the ridge candidate; the cells restored
(written after the record panel on `619d596`, before any in-brain cell; commit 5)

**What the panel found and the session reproduced brain-free on the `bd53a42` artifacts (threads pinned, the
drifting chain, a refit every 20 observations, seeds 0 / 1):**

| regime | batch-only | DELTA (mu 1.0) | SIGN | DELTA sign-flipped | RIDGE (K = 20) |
|---|---|---|---|---|---|
| A ungated (§3's control; the batch re-solved at every refit) | 0.400 / 0.359 (13 / 7 units) | 0.386 / 0.397 | 0.379 / 0.407 | 0.298 / 0.234 (below) | **0.557 / 0.517 (+0.157 / +0.159)** |
| B the growth gate closes after the 3rd refit (beta held stale — L1b §4's mechanism) | 0.457 / 0.390 (5 / 2 units) | **0.588 / 0.517 (+0.131 / +0.128)** | 0.586 / 0.519 | 0.297 / 0.305 (below) | **0.745 / 0.538 (+0.288 / +0.148)** |
| C a one-category tail after 200 chain observations (the deployed brain's shape: `_refit` skips a one-class buffer) | 0.860 / 0.821 | 0.922 / 0.886 | — | — | 0.931 / 0.881 |

Three corrections follow. **(1) The refusal was premature (panel PH-1).** §3's control ran with no growth gate, so
its batch competitor was re-solved every 20 observations — the one regime in which an online readout has no job.
L1c was commissioned for the readout held STALE under a closed T047 gate (L1b §4) and the brain lives in a
near-constant stream whose one-category buffer makes `_refit` skip every re-solve (§4); in both regimes the ported
DELTA step clears the pre-registered bar on both seeds with the sign flip below. "The positive control fails on the
substrate the mechanism would run on" was true only of regime A. **(2) WIDTH is not the binding constraint (panel
I-5, PH-2, M-1).** A readout-only mechanism on the SAME 7–13 units — the engine's own `_solve_beta` re-solved at
every observation on the last K = 20 buffered transitions, moving only beta (the sliding ridge) — gains +0.157 /
+0.159 ungated where the LMS rules gain nothing; §2.0's +0.19 was a fact about the LMS rules on a 53–100-unit
direct fit (archived now: `emulation_direct_ngsnn.py` + `.txt`) and the null was a fact about the LMS rules, not the
substrate. The width sentence is demoted to a hypothesis and struck from the plan row. **(3) The sign-flipped
control is directional only at the ported step (panel M-2):** below mu 1.0 the flipped DELTA reads ABOVE the
incumbent on both seeds (the panel's sweep: 0.417–0.469 at mu 0.05–0.3 against 0.400), so R3 is valid only at mu = 1.0
and the derived step mu = 1/20 is dropped — inert, not a candidate. Also corrected: "updates fired on every transition"
holds for DELTA (580 of 580); SIGN moves on error only (344–398 of 580); the comparator of §3 is a whole-buffer batch
that reads below chance after every flip (a buffer capped at 40 reads 0.593 / 0.653 — but a cap changes admission and
is not a readout-only control).

**The re-filed pre-registration (the session's call under the authorization):**
- **Candidates in-brain:** DELTA (mu 1.0, the faithful port), SIGN (mu 1.0, the plan's second candidate, kept for the
  A/B as the row names it), and **RIDGE** — the engine's own ridge (`NGSNN._solve_beta`, the same regularization) on a
  sliding window of the last K transitions, re-solved on every wake percept, moving only beta (P2: it is the incumbent
  solve at a per-sample cadence with forgetting, not a second solver). K = 20 = one refit interval (the same
  derivation the derived step used: the online memory is one refit interval; in-brain the L1b cells fed ~395 percepts
  over 20 refits ≈ 20 per interval — measured, not assumed). Cost per wake percept: one K × L ridge solve (L ≤ 14).
- **Arms per seed (8 seeds × 800 ticks, 4 at a time, one command):** BATCH, BATCH_B (A/A), DELTA, SIGN, RIDGE,
  NEUTER_DELTA (mu 0: the reach control, bit-identical to BATCH), SIGNFLIP_DELTA (mu 1.0, sign −1: the LMS rules'
  control that is not the method — valid at this step), RIDGE_SHUF (the ridge on the same window with the window's
  labels permuted each solve: the ridge's control that is not the method). Same rows by identical floors and identical
  live persistence across the eight arms per seed.
- **The reds, each quoted and scored one by one (the plan's statistic):** L1c-R1 paired (live_acc_CAND −
  live_acc_BATCH) clears, per candidate; L1c-R2 NEUTER_DELTA == BATCH bit for bit on 8 of 8; L1c-R3 for the LMS
  candidates: paired (live_acc_SIGNFLIP − live_acc_BATCH) does not clear and reads below on a majority; for RIDGE:
  paired (live_acc_RIDGE_SHUF − live_acc_BATCH) does not clear; L1c-R4 paired (refit_acc_CAND − refit_acc_BATCH) not
  below −1.895; reported: units, W-hash divergence (the mechanism's effect on the admission path through the
  closed-gate rows, disclosed in §2.2), the live persistence floor, the A/B among candidates.
- **Decision rule:** a candidate LANDS if R1 clears and R2, R3, R4 hold; among candidates that land the largest R1 t
  survives and the others are deleted (P2); none → the module is deleted (P8 clause 5) and the record stands. The
  survivor is constructed by `_make_neuro`; the strict in-tree test carries the reading; on a deletion the test is
  not carried. Expectation stated before the cells (from the regimes above): RIDGE is favoured; DELTA is expected to
  clear where the brain's gate closes or its buffer goes one-class, and the live stream is such a stream.

### 2.1 The mechanism

`hippocampus/online_readout.OnlineReadout.learn(model, s, cat)` (a new governed module beside `ng_snn`), called from
`NeurogenesisSelfImprove.observe` immediately AFTER the live instrument has scored this transition and BEFORE it
is buffered — on the pair (`s = model._rates(x_{t-1})`, `cat_t`) the instrument just scored, so the instrument
never reads a pair it has learned from. `beta` is moved in place on the retained model. If `cat ≥ m`, beta is
widened with a zero column first (function-preserving; laminar's precedent). Step size for BOTH candidates (one
derivation, P1): `lr = MU / (‖s‖² + 1)` — the normalized-LMS form of laminar's `_observe_lr` (the `+1` is the
bias unit's presynaptic energy there; beta has no bias here, so `+1` is CHOSEN as the floor that keeps the
denominator ≥ 1 on a silent percept — disclosed), MU = 1.0 cited from laminar (CHOSEN there; the P1 debt named).
**Candidate DELTA:** `beta -= lr · sᵀ (softmax(s @ beta) − onehot(cat))`. **Candidate SIGN (the perceptron):** on
`argmax(s @ beta) ≠ cat`: `beta[:, cat] += lr·s`, `beta[:, argmax] −= lr·s`; else nothing. At an open-gate refit
`fit` re-solves beta on the buffer and `_refit` retains the new model — the online rule continues from that
solution. State beyond beta: none (no EMA; the step is per-sample normalized). GRADIENT_CLAIM: LOCAL, with the
read set stated (s, beta's own columns, cat); RuleSpec through `authored_increment`-style declaration is NOT
repeated here — the rule is a per-sample tensor update on the output layer, declared and census-adjudicated as
`OnlineReadout.learn`.

### 2.2 What the mechanism touches beyond the readout (disclosed before the cells)

The closed-gate refit scores the RETAINED model, whose beta the online rule has moved: the trajectory row and
the growth gate's evidence then read the ONLINE readout. The admission path may therefore diverge from the
incumbent's (as the L1b twin's did): every cell records the W hash and the unit trajectory, and §3 reads the
divergence explicitly. This is the mechanism acting (a better readout is real evidence for the gate), not a leak
— but the paired comparisons below are WHOLE-SYSTEM comparisons on identical rows, and attribution to the
readout alone rests on the neutered and sign-flipped arms, which share the incumbent's growth path only when
their beta never moves (NEUTER) or moves the wrong way (SIGNFLIP).

### 2.3 The cells — 8 seeds × 800 ticks, 4 at a time, one command, thread-pinned, same rows by measurement

Arms per seed: **BATCH** (the incumbent = the control), **BATCH_B** (A/A), **DELTA**, **SIGN**, **NEUTER_DELTA**
(the rule wired, lr pinned at 0: the reach control — must equal BATCH bit for bit in W, beta, live accuracy),
**SIGNFLIP_DELTA** (the DELTA update with its sign reversed: the control that is not the method — an online rule
moving beta the wrong way must read BELOW the incumbent's live accuracy). Same rows asserted by identical
majority AND persistence floors across all six arms per seed; the live persistence floor identical too.

### 2.4 The reds — each conjunct quoted, MET / NOT MET one by one (the plan's statistic: S = 8, paired
one-sided t > 1.895 AND exact sign-flip p ≤ 0.05, ties at |d| ≤ 1e-6, DEGENERATE below 5 non-tied; the
tie-pattern control printed beside every verdict)

- **L1c-R1 (the red):** paired (live_acc_CAND − live_acc_BATCH) clears the bar — the online readout beats the
  stale one on the wake stream it learns from, scored before learning. Reported beside it: (live_acc_CAND −
  live_persistence) per arm.
- **L1c-R2 (reach):** NEUTER_DELTA equals BATCH bit for bit (W hash, beta hash, live accuracy, units) on 8 of 8 —
  else the wire does something the rule does not, and no verdict is read.
- **L1c-R3 (the control that is not the method):** paired (live_acc_SIGNFLIP − live_acc_BATCH) does NOT clear
  and reads below zero on a majority of seeds (reported as a count, not a bar) — the direction of the update
  matters; if SIGNFLIP also beats the incumbent, the instrument, not the rule, is what moved.
- **L1c-R4 (non-regression):** paired (refit_acc_CAND − refit_acc_BATCH) not below −1.895 (the batch instrument
  at open-gate refits); units and W-hash divergence reported.
- **A/B:** paired (live_acc_DELTA − live_acc_SIGN), reported.

### 2.5 Decision rule

A candidate LANDS if R1 clears and R2, R3 and R4 hold; if both land, the larger R1 t wins and the other is deleted
from the module (P2); if neither, the module is DELETED (P8 clause 5) with the observe hook it needed, the
record stands, and the plan's L2 trigger falls to the T047 gate's own plateau (already restated). The survivor is
constructed by `_make_neuro`; the strict in-tree test in `tests/test_real_percept_stream.py` (arms BATCH / L1C /
SIGNFLIP through the same `_l_cell` harness) carries the reading — MET as a plain assert, NOT MET as a strict xfail
naming the numbers — and is NOT carried on a deletion (the L1b lesson).

### 2.6 Instrument control

BATCH's floors, units and refit accuracy against `l1b-synaptic-online.md` §3.2's BATCH cells (the same src at
`1e31239` for the incumbent path): expected identical to 1e-6; a difference is the instrument moving.

## 3. Results — the pre-registered one shot (Amendment 1), on the engine's substrate, brain-free

Drifting chain (the succession rule reverses every 100 observations), `NeurogenesisSelfImprove` with a refit every
20 observations, the live instrument scored before each online step; seeds 0 and 1; the measured module is
`git show bd53a42:src/snn_data_lake/hippocampus/online_readout.py`.

| step | seed | units | batch-only | DELTA | SIGN | DELTA sign-flipped | control |
|---|---|---|---|---|---|---|---|
| mu = 1.0 (ported) | 0 | 13 | 0.400 | 0.386 | 0.379 | — | FAILS (thrashes each fresh solve; adapts after each flip) |
| mu = 1/20 (derived, the one shot) | 0 | 13 | 0.400 | 0.407 | 0.340 | 0.417 | FAILS; the sign flip is NOT below |
| mu = 1/20 | 1 | 7 | 0.359 | 0.333 | 0.314 | 0.386 | FAILS; the sign flip is NOT below |
| mu = 1/20, STATIONARY chain | 0 | — | 0.872 | 0.871 | — | — | gap −0.002 (inert) |

At the ported step the online readout moves too much (a near-full normalized correction per sample on a 2–5-unit
readout); at the interval-derived step it moves too little to matter in EITHER direction — its sign-flipped
control reads no lower than the incumbent (0.417 / 0.386 against 0.400 / 0.359), so at that step the instrument
cannot even tell the update's direction. Between the two steps lies a tuning campaign the pre-registration forbade.
Updates fired on every observed transition (580 of 580 per run; Σ|dβ| 25.6 / 23.9 at the derived step, 494 at the
ported step): the rule ran; it did not help.

**The binding constraint, read from the same runs:** the engine admits 1–14 units on this 4-class chain (7 and 13
at the end; 1–2 for long stretches — §2.0b's per-interval table), and the live stream's incumbent holds 2–7
(`l1b-synaptic-online.md` §3.2). A readout over so few units has almost nothing to re-weight; the batch solve
already extracts what the substrate carries. Recency in the readout is worth +0.19 on a 53–100-unit substrate (§2.0,
the emulation) and nothing on the engine's. The row's premise — "synaptic first" on the readout — is upstream-blocked
by WIDTH, which is L2's and Phase 3's business, not a rule's.

## 4. Verdict and disposition — WITHDRAWN as premature (Amendment 2, §2.0c); superseded by the in-brain cells

The first writing of this section filed L1c as REFUSED ON MEASUREMENT before any in-brain cell, on the ungated
brain-free control of §3, and attributed the null to WIDTH. The record panel (§6, three lenses, one saying revise)
showed the control excluded the regimes L1c was commissioned for and the cause was an inference; the session
reproduced every one of its measurements (§2.0c) and re-files. The refusal is withdrawn; nothing about the
mechanism is decided by §3 alone; the in-brain cells of §2.3–2.5 (as amended in §2.0c) are the measurement.
The deployed-brain reading of the first writing (live accuracy 0.000 against persistence 0.9375 after the relaunch;
the cockpit later read 0.0 / 0.999 over 2985 transitions) stands as evidence that the stale-readout regime is the
one the brain LIVES in — a near-constant stream whose one-category buffer makes `_refit` skip every re-solve — and
is carried into §2.0c's regimes, not left as an aside. Whether the deployed unit fires on that stream (the one case
the rule cannot move) was not read and is owed.

## 5. Commits of this arc (stacked on `ea69088`)

| commit | what | landed on |
|---|---|---|
| `3b4de12` 1 | this record §1–2 (pre-registration; the emulation's stimulus reading) | before any run |
| `a832906` 2 | Amendment 1: the control failed on the engine's substrate at the ported step; the one derived step pre-registered | before the re-run |
| `bd53a42` 3 | the build AS MEASURED (its self-check red on this substrate) | the artifact of §3 |
| `(this commit)` 4 | the REFUSAL: the build deleted; §3–5; the plan's L1c row | P8 clauses 2 and 5 |

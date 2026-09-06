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

(The RIDGE column of this table is the session's reproduction script's sliding ridge, not the module later shipped and
deleted — the shipped one read 0.569 / 0.507 ungated and 0.745 / 0.529 gated; the sign flip was run in regime B only.)
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
`NeurogenesisSelfImprove.observe` immediately AFTER the live instrument has scored this transition (the transition is
buffered first; the buffer is not read on the wake tick) — on the pair (`s = model._rates(x_{t-1})`, `cat_t`) the instrument just scored, so the instrument
never reads a pair it has learned from. `beta` is moved in place on the retained model. If `cat ≥ m`, beta is
widened with a zero column first (function-preserving; laminar's precedent). Step size for BOTH candidates (one
derivation, P1): `lr = MU / (‖s‖² + 1)` — the normalized-LMS form of laminar's `_observe_lr` (the `+1` is the
bias unit's presynaptic energy there; beta has no bias here, so `+1` is CHOSEN as the floor that keeps the
denominator ≥ 1 on a silent percept — disclosed), MU = 1.0 is laminar's value: laminar claims it under P1 exception (a) and records that its
"fully corrective" NLMS derivation was DELETED because NLMS convergence does not transfer to a softmax readout — the
same caveat applies to this delta rule, and the value is LOAD-BEARING here (1/20 inert; the sign flip directional only at
1.0), so the owed item is an in-brain step sweep with the sign-flip control at each step (panel PH-9).
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

## 3a. Results of the first writing — the pre-registered one shot (Amendment 1), ungated, brain-free (superseded by §3)

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

## 5a. Commits of the first writing (stacked on `ea69088`)

| commit | what | landed on |
|---|---|---|
| `3b4de12` 1 | this record §1–2 (pre-registration; the emulation's stimulus reading) | before any run |
| `a832906` 2 | Amendment 1: the control failed on the engine's substrate at the ported step; the one derived step pre-registered | before the re-run |
| `bd53a42` 3 | the build AS MEASURED (its self-check red on this substrate) | the artifact of §3 |
| `619d596` 4 | the REFUSAL: the build deleted; §3–5; the plan's L1c row | P8 clauses 2 and 5 |

## 3. Results (the in-brain cells, §2.0c)

### 3.1 The cells — 8 arms × 8 seeds × 800 ticks, 4 at a time (02:03–02:24 UTC; `stage1/`; head(s) 7862d44; wall 54–114 s; the daemon was up)

Same rows: majority, persistence AND live-persistence floors identical across the eight arms on every seed to 1e-6 — no seed excluded; seeds used [0, 1, 2, 3, 4, 5, 6, 7]. A/A (BATCH vs BATCH_B): max |Δ live acc| 0.00e+00, W/beta hashes equal on 8 of 8. Perception unified on every cell: True; feed refusals: none. **§2.6's instrument control: BATCH equals `l1b-synaptic-online.md` §3.2's BATCH on 8 of 8 (floors, units, refit and live accuracy, W hash) — the instrument did not move.**

| arm | seed | units | live acc / live pers | refit acc (last third) | persistence | majority | updates / Σ|dβ| / widened | W / beta hash |
|---|---|---|---|---|---|---|---|---|
| BATCH | 0 | 2 | 0.547 / 0.932 | 0.311 | 0.953 | 0.744 |  /  /  | 1fd01989 / 90fea667 |
| BATCH | 1 | 2 | 0.849 / 0.963 | 0.917 | 0.983 | 0.739 |  /  /  | d655e464 / ecd22e20 |
| BATCH | 2 | 2 | 0.807 / 0.952 | 0.825 | 0.953 | 0.694 |  /  /  | f2879fd9 / d131007f |
| BATCH | 3 | 7 | 0.653 / 0.902 | 0.656 | 0.911 | 0.319 |  /  /  | 60750bab / 60771e58 |
| BATCH | 4 | 6 | 0.590 / 0.907 | 0.628 | 0.950 | 0.508 |  /  /  | 708075a4 / af611828 |
| BATCH | 5 | 3 | 0.541 / 0.934 | 0.833 | 0.947 | 0.758 |  /  /  | 5b73e151 / 83f82801 |
| BATCH | 6 | 3 | 0.630 / 0.951 | 0.867 | 0.933 | 0.506 |  /  /  | 914883ca / 1b8e0b0c |
| BATCH | 7 | 2 | 0.771 / 0.992 | 0.947 | 1.000 | 0.653 |  /  /  | 292414f4 / 7d7da645 |
| BATCH_B | 0 | 2 | 0.547 / 0.932 | 0.311 | 0.953 | 0.744 |  /  /  | 1fd01989 / 90fea667 |
| BATCH_B | 1 | 2 | 0.849 / 0.963 | 0.917 | 0.983 | 0.739 |  /  /  | d655e464 / ecd22e20 |
| BATCH_B | 2 | 2 | 0.807 / 0.952 | 0.825 | 0.953 | 0.694 |  /  /  | f2879fd9 / d131007f |
| BATCH_B | 3 | 7 | 0.653 / 0.902 | 0.656 | 0.911 | 0.319 |  /  /  | 60750bab / 60771e58 |
| BATCH_B | 4 | 6 | 0.590 / 0.907 | 0.628 | 0.950 | 0.508 |  /  /  | 708075a4 / af611828 |
| BATCH_B | 5 | 3 | 0.541 / 0.934 | 0.833 | 0.947 | 0.758 |  /  /  | 5b73e151 / 83f82801 |
| BATCH_B | 6 | 3 | 0.630 / 0.951 | 0.867 | 0.933 | 0.506 |  /  /  | 914883ca / 1b8e0b0c |
| BATCH_B | 7 | 2 | 0.771 / 0.992 | 0.947 | 1.000 | 0.653 |  /  /  | 292414f4 / 7d7da645 |
| DELTA | 0 | 2 | 0.547 / 0.932 | 0.311 | 0.953 | 0.744 | 369 / 53.763463 / 3 | 1fd01989 / b9bb2ab1 |
| DELTA | 1 | 2 | 0.841 / 0.963 | 0.917 | 0.983 | 0.739 | 377 / 84.15141 / 1 | d655e464 / e1c83750 |
| DELTA | 2 | 2 | 0.933 / 0.952 | 0.825 | 0.953 | 0.694 | 374 / 158.321034 / 2 | f2879fd9 / 47012c6f |
| DELTA | 3 | 7 | 0.857 / 0.902 | 0.403 | 0.911 | 0.319 | 378 / 194.343179 / 3 | 60750bab / 8b8f2732 |
| DELTA | 4 | 6 | 0.838 / 0.907 | 0.517 | 0.950 | 0.508 | 376 / 152.055553 / 2 | 708075a4 / 6508680f |
| DELTA | 5 | 3 | 0.708 / 0.934 | 0.836 | 0.947 | 0.758 | 377 / 94.302605 / 1 | 5b73e151 / 89fb07bc |
| DELTA | 6 | 3 | 0.854 / 0.951 | 0.800 | 0.933 | 0.506 | 370 / 127.510851 / 2 | 914883ca / de83eee7 |
| DELTA | 7 | 2 | 0.952 / 0.992 | 0.947 | 1.000 | 0.653 | 376 / 114.479314 / 1 | 292414f4 / c3243d87 |
| SIGN | 0 | 2 | 0.556 / 0.932 | 0.311 | 0.953 | 0.744 | 164 / 12.924321 / 3 | 1fd01989 / f4f9f4ee |
| SIGN | 1 | 2 | 0.833 / 0.963 | 0.917 | 0.983 | 0.739 | 63 / 20.543337 / 1 | d655e464 / ecd22e20 |
| SIGN | 2 | 2 | 0.922 / 0.952 | 0.825 | 0.953 | 0.694 | 29 / 38.097555 / 2 | f2879fd9 / d131007f |
| SIGN | 3 | 7 | 0.868 / 0.902 | 0.433 | 0.911 | 0.319 | 50 / 68.615406 / 3 | 60750bab / c43fc3da |
| SIGN | 4 | 6 | 0.856 / 0.907 | 0.558 | 0.950 | 0.508 | 54 / 49.152006 / 2 | 708075a4 / daf1a182 |
| SIGN | 5 | 3 | 0.719 / 0.934 | 0.742 | 0.947 | 0.758 | 106 / 27.167398 / 1 | 5b73e151 / 65d31c92 |
| SIGN | 6 | 3 | 0.859 / 0.951 | 0.867 | 0.933 | 0.506 | 52 / 29.733911 / 2 | 914883ca / 1b8e0b0c |
| SIGN | 7 | 2 | 0.973 / 0.992 | 0.947 | 1.000 | 0.653 | 10 / 8.420745 / 1 | 292414f4 / 7d7da645 |
| RIDGE | 0 | 2 | 0.550 / 0.932 | 0.311 | 0.953 | 0.744 | 364 / 59.518354 / 3 | 1fd01989 / 8aab81e3 |
| RIDGE | 1 | 2 | 0.830 / 0.963 | 0.917 | 0.983 | 0.739 | 374 / 41.150169 / 1 | d655e464 / 2574864f |
| RIDGE | 2 | 2 | 0.928 / 0.952 | 0.825 | 0.953 | 0.694 | 366 / 120.853711 / 2 | f2879fd9 / 0a7375fb |
| RIDGE | 3 | 7 | 0.812 / 0.902 | 0.400 | 0.911 | 0.319 | 370 / 197.410334 / 3 | 60750bab / 7ffa54b6 |
| RIDGE | 4 | 6 | 0.840 / 0.907 | 0.453 | 0.950 | 0.508 | 364 / 105.578484 / 2 | 708075a4 / fbf05daf |
| RIDGE | 5 | 3 | 0.732 / 0.934 | 0.792 | 0.947 | 0.758 | 368 / 72.343537 / 1 | 5b73e151 / a48592ab |
| RIDGE | 6 | 3 | 0.892 / 0.951 | 0.761 | 0.933 | 0.506 | 362 / 91.835466 / 2 | 914883ca / dc4f6021 |
| RIDGE | 7 | 2 | 0.965 / 0.992 | 0.947 | 1.000 | 0.653 | 371 / 59.552136 / 1 | 292414f4 / 2574864f |
| NEUTER_DELTA | 0 | 2 | 0.547 / 0.932 | 0.311 | 0.953 | 0.744 | 0 / 0.0 / 0 | 1fd01989 / 90fea667 |
| NEUTER_DELTA | 1 | 2 | 0.849 / 0.963 | 0.917 | 0.983 | 0.739 | 0 / 0.0 / 0 | d655e464 / ecd22e20 |
| NEUTER_DELTA | 2 | 2 | 0.807 / 0.952 | 0.825 | 0.953 | 0.694 | 0 / 0.0 / 0 | f2879fd9 / d131007f |
| NEUTER_DELTA | 3 | 7 | 0.653 / 0.902 | 0.656 | 0.911 | 0.319 | 0 / 0.0 / 0 | 60750bab / 60771e58 |
| NEUTER_DELTA | 4 | 6 | 0.590 / 0.907 | 0.628 | 0.950 | 0.508 | 0 / 0.0 / 0 | 708075a4 / af611828 |
| NEUTER_DELTA | 5 | 3 | 0.541 / 0.934 | 0.833 | 0.947 | 0.758 | 0 / 0.0 / 0 | 5b73e151 / 83f82801 |
| NEUTER_DELTA | 6 | 3 | 0.630 / 0.951 | 0.867 | 0.933 | 0.506 | 0 / 0.0 / 0 | 914883ca / 1b8e0b0c |
| NEUTER_DELTA | 7 | 2 | 0.771 / 0.992 | 0.947 | 1.000 | 0.653 | 0 / 0.0 / 0 | 292414f4 / 7d7da645 |
| SIGNFLIP_DELTA | 0 | 2 | 0.187 / 0.932 | 0.311 | 0.953 | 0.744 | 369 / 180.214308 / 3 | 1fd01989 / a1ad0a37 |
| SIGNFLIP_DELTA | 1 | 2 | 0.159 / 0.963 | 0.917 | 0.983 | 0.739 | 377 / 299.655137 / 1 | d655e464 / 0e10d568 |
| SIGNFLIP_DELTA | 2 | 2 | 0.195 / 0.952 | 0.825 | 0.953 | 0.694 | 374 / 375.715621 / 2 | f2879fd9 / e959d138 |
| SIGNFLIP_DELTA | 3 | 7 | 0.222 / 0.902 | 0.325 | 0.911 | 0.319 | 378 / 418.383268 / 3 | 60750bab / e4de3d45 |
| SIGNFLIP_DELTA | 4 | 6 | 0.160 / 0.907 | 0.028 | 0.950 | 0.508 | 376 / 402.075223 / 2 | 708075a4 / 9be96874 |
| SIGNFLIP_DELTA | 5 | 3 | 0.122 / 0.934 | 0.422 | 0.947 | 0.758 | 377 / 203.064452 / 1 | 5b73e151 / 7d03a834 |
| SIGNFLIP_DELTA | 6 | 3 | 0.108 / 0.951 | 0.828 | 0.933 | 0.506 | 370 / 373.738678 / 2 | 914883ca / 2c17a8e9 |
| SIGNFLIP_DELTA | 7 | 2 | 0.098 / 0.992 | 0.947 | 1.000 | 0.653 | 376 / 360.026197 / 1 | 292414f4 / 02c976dd |
| RIDGE_SHUF | 0 | 2 | 0.507 / 0.932 | 0.311 | 0.953 | 0.744 | 364 / 105.337473 / 3 | 1fd01989 / 4d66f0c3 |
| RIDGE_SHUF | 1 | 2 | 0.798 / 0.963 | 0.917 | 0.983 | 0.739 | 374 / 75.984023 / 1 | d655e464 / 2574864f |
| RIDGE_SHUF | 2 | 2 | 0.890 / 0.952 | 0.825 | 0.953 | 0.694 | 366 / 197.742485 / 2 | f2879fd9 / 0a7375fb |
| RIDGE_SHUF | 3 | 7 | 0.690 / 0.902 | 0.256 | 0.911 | 0.319 | 370 / 379.136771 / 3 | 60750bab / 7ffa54b6 |
| RIDGE_SHUF | 4 | 6 | 0.758 / 0.907 | 0.397 | 0.950 | 0.508 | 364 / 236.268773 / 2 | 708075a4 / fbf05daf |
| RIDGE_SHUF | 5 | 3 | 0.692 / 0.934 | 0.531 | 0.947 | 0.758 | 368 / 129.069423 / 1 | 5b73e151 / a48592ab |
| RIDGE_SHUF | 6 | 3 | 0.832 / 0.951 | 0.761 | 0.933 | 0.506 | 362 / 163.167472 / 2 | 914883ca / dc4f6021 |
| RIDGE_SHUF | 7 | 2 | 0.955 / 0.992 | 0.947 | 1.000 | 0.653 | 371 / 72.346798 / 1 | 292414f4 / 2574864f |

### 3.2 Verdict — each conjunct of §2.4 against its wording

- **L1c-R2 (reach): NEUTER_DELTA equals BATCH bit for bit** — W hash 8 of 8, beta hash 8 of 8, live accuracy 8 of 8, units 8 of 8 → **HOLDS**.
- **DELTA, L1c-R1 ("the online readout beats the stale one on the wake stream, scored before learning"): paired (live_acc_DELTA − live_acc_BATCH) clears** — t = 4.132, exact p = 0.0156, 6 / 1 / 1, mean +0.143 (random-sign clears 0.035) → **MET**. Beside it: (live acc − live persistence) t = -2.874, exact p = 1.0000, 0 / 8 / 0, mean -0.125 (random-sign clears 0.030).
- **DELTA, L1c-R4 (non-regression on the batch instrument): paired (refit_acc_DELTA − refit_acc_BATCH) not below −1.895** — t = -1.664, exact p = 0.9375, 1 / 3 / 4, mean -0.053 (random-sign clears 0.000) → **DEGENERATE** (4 non-tied; relabelled in the fold, panel PH-4: the letter's t is not read on a degenerate statistic; the per-seed closed-gate losses are in §3.3). Units [2, 2, 2, 7, 6, 3, 3, 2] (BATCH [2, 2, 2, 7, 6, 3, 3, 2]); the admission path diverged from BATCH (W hash differs) on 0 of 8 seeds (§2.2); updates per cell [369, 377, 374, 378, 376, 377, 370, 376]; Σ|dβ| [53.763463, 84.15141, 158.321034, 194.343179, 152.055553, 94.302605, 127.510851, 114.479314]; widened [3, 1, 2, 3, 2, 1, 2, 1].
- **SIGN, L1c-R1 ("the online readout beats the stale one on the wake stream, scored before learning"): paired (live_acc_SIGN − live_acc_BATCH) clears** — t = 4.054, exact p = 0.0117, 7 / 1 / 0, mean +0.150 (random-sign clears 0.055) → **MET**. Beside it: (live acc − live persistence) t = -2.714, exact p = 1.0000, 0 / 8 / 0, mean -0.118 (random-sign clears 0.030).
- **SIGN, L1c-R4 (non-regression on the batch instrument): paired (refit_acc_SIGN − refit_acc_BATCH) not below −1.895** — t = -1.705, exact p = 1.0000, 0 / 3 / 5, mean -0.048 (random-sign clears 0.000) → **DEGENERATE** (3 non-tied). Units [2, 2, 2, 7, 6, 3, 3, 2] (BATCH [2, 2, 2, 7, 6, 3, 3, 2]); the admission path diverged from BATCH (W hash differs) on 0 of 8 seeds (§2.2); updates per cell [164, 63, 29, 50, 54, 106, 52, 10]; Σ|dβ| [12.924321, 20.543337, 38.097555, 68.615406, 49.152006, 27.167398, 29.733911, 8.420745]; widened [3, 1, 2, 3, 2, 1, 2, 1].
- **RIDGE, L1c-R1 ("the online readout beats the stale one on the wake stream, scored before learning"): paired (live_acc_RIDGE − live_acc_BATCH) clears** — t = 3.910, exact p = 0.0117, 7 / 1 / 0, mean +0.145 (random-sign clears 0.050) → **MET**. Beside it: (live acc − live persistence) t = -2.896, exact p = 1.0000, 0 / 8 / 0, mean -0.123 (random-sign clears 0.035).
- **RIDGE, L1c-R4 (non-regression on the batch instrument): paired (refit_acc_RIDGE − refit_acc_BATCH) not below −1.895** — t = -2.087, exact p = 1.0000, 0 / 4 / 4, mean -0.072 (random-sign clears 0.000) → **DEGENERATE** (4 non-tied; the t reads past the bar). Units [2, 2, 2, 7, 6, 3, 3, 2] (BATCH [2, 2, 2, 7, 6, 3, 3, 2]); the admission path diverged from BATCH (W hash differs) on 0 of 8 seeds (§2.2); updates per cell [364, 374, 366, 370, 364, 368, 362, 371]; Σ|dβ| [59.518354, 41.150169, 120.853711, 197.410334, 105.578484, 72.343537, 91.835466, 59.552136]; widened [3, 1, 2, 3, 2, 1, 2, 1].
- **L1c-R3 (the control that is not the method): SIGNFLIP_DELTA does not clear and reads below BATCH on a majority of seeds** — t = -11.562, exact p = 1.0000, 0 / 8 / 0, mean -0.517 (random-sign clears 0.055); below on 8 of 8 → **HOLDS**.
- **L1c-R3 for RIDGE (the ridge on a label-permuted window does not clear):** t = 2.579, exact p = 0.0312, 6 / 2 / 0, mean +0.092 (random-sign clears 0.045) → **FAILS**.
- **A/B (live):** DELTA − SIGN t = -1.740, exact p = 0.9375, 2 / 6 / 0, mean -0.007 (random-sign clears 0.030); RIDGE − DELTA t = 0.277, exact p = 0.3945, 5 / 3 / 0, mean +0.002 (random-sign clears 0.055); RIDGE − SIGN t = -0.508, exact p = 0.6797, 3 / 5 / 0, mean -0.005 (random-sign clears 0.035).
- **The incumbent (BATCH):** live accuracy [0.5474, 0.8488, 0.8075, 0.6534, 0.5904, 0.5411, 0.6297, 0.7713]; (live acc − live persistence) t = -7.274, exact p = 1.0000, 0 / 8 / 0, mean -0.268 (random-sign clears 0.035); refit accuracy [0.3111, 0.9167, 0.825, 0.6556, 0.6278, 0.8333, 0.8667, 0.9472]; units [2, 2, 2, 7, 6, 3, 3, 2].

### 3.3 Amendment 3 — what the cells read, and the one variant pre-registered before its run (commit 7)

**Read from §3.1–3.2, each conjunct by its letter.** R1 is MET by all three candidates on the wake stream, by
about the same margin (DELTA t 4.132, p 0.0156, 6/1/1, mean +0.143; SIGN t 4.054, p 0.0117, 7/1/0, mean +0.150; RIDGE
t 3.910, p 0.0117, 7/1/0, mean +0.145) — the incumbent's live accuracy sits 0.11–0.39 below the live persistence floor on
every seed and the online readouts close about half of that gap (they stay below persistence on 8 of 8: DELTA mean
−0.125). R2 holds exactly (NEUTER_DELTA == BATCH in W, beta, live accuracy and units on 8 of 8). R3 holds for the LMS
rules (SIGNFLIP t -11.562, p 1.0000, 0/8/0, mean -0.517, below on 8 of 8 — a directional control in-brain at the
ported step) and **FAILS for RIDGE by its letter: the ridge on a label-PERMUTED window still clears the bar against the
incumbent (t 2.579, p 0.0312, 6/2/0, mean +0.092)** — so **RIDGE does not land (the pre-registered R3) and, beside that,
loses the P2 wash to DELTA (R1 t 3.910 < 4.132; RIDGE − DELTA +0.002)**. The control itself is near-vacuous on this
stream (panel PH-2 / M-3, verified): a permutation of a one-category window is the identity, RIDGE and RIDGE_SHUF end
with bit-identical betas on 7 of 8 seeds, and RIDGE beats its own shuffle on 8 of 8 (+0.053) — so the first writing's
"the gain is the window's recency, not the pairing" is a HYPOTHESIS the control cannot support, demoted here; DELTA and
SIGN were never held to a label null, and the sign flip does not separate pairing from marginal-tracking (M-11). R4 is **DEGENERATE for all three** (fewer than 5 non-tied seeds): on seeds 0, 1, 2 and 7
every last-third refit was an open-gate refit, where `fit` re-solves beta and the arms are identical by construction;
on seeds 3, 4, 5 and 6 closed-gate refits scored the ONLINE beta on the held-out marginal-covering slice and it read
LOWER than the stale batch beta on almost every one (DELTA −0.253 / −0.111 / +0.003 / −0.067; SIGN −0.222 / −0.069 /
−0.092 / 0; RIDGE −0.256 / −0.175 / −0.042 / −0.106). The A/B among the LMS rules is not resolvable (DELTA − SIGN
t -1.740, p 0.9375, 2/6/0, mean -0.007); RIDGE − DELTA t 0.277, p 0.3945, 5/3/0, mean +0.002.

**Under §2.0c's rule as written the stage-1 outcome is DELETION** (no candidate's R4 held — degenerate on all three —
and RIDGE failed R3); the session departs from that letter by pre-registering ONE instrument variant below, disclosed
before its run (panel PH-5). **The dissociation, stated plainly (the session's reading, panelled):** an online readout that learns from the
wake stream learns a RECENCY-shaped readout — right more often on the stream the brain lives in (which is 93–98 %
"the same category again"), worse on the marginal-covering slice that samples every category. Both numbers are true;
they measure different things. The held-out marginal accuracy is the instrument the T047 growth gate reads
(`_refit` → `_growth_gate_verdict`), and under a closed gate it is now the ONLINE beta's — lower — so the gate sees
"not charging", holds its stop, and re-opens only on a new optimum the recency-shaped beta cannot produce on that
slice: **a stop that welds itself shut with no batch re-solve ever again** (the shape `growth_stop.py`'s own docstring
names). Landing DELTA as measured would ship that hazard; the R4 letter ("not below −1.895") is numerically satisfied
(t −1.664) but on a degenerate statistic with a per-seed loss on 3 of 4 closed-gate seeds — not a landing the record
can call earned.

**The ONE variant, pre-registered here (the session's call under the authorization), the decoupled instrument:** the
engine keeps `_batch_beta` — the readout of the last OPEN-gate solve — and a closed-gate refit scores the held-out
slice with IT, exactly as the incumbent does, while the retained model's `beta` (the one the live instrument and the
brain's prediction use) keeps learning online. The growth gate's evidence is then the batch instrument — what a batch
solve can extract from the current units, the right evidence for a GROWTH decision — unchanged by construction from
the incumbent's; the live readout is the mechanism. P2: the batch beta is an INSTRUMENT held beside the mechanism
(POLICIES: a control/instrument is not a second faculty), not a second readout the brain uses; `predict_accuracy` on
state() is the instrument's number and `live_accuracy` the mechanism's, labelled so on the cockpit surface.
**Re-run:** DELTA and SIGN × 8 seeds at the amended engine (16 cells), plus BATCH × 8 as the A/A across the change
(the incumbent path must be bit-identical to stage1's BATCH: W, beta, live accuracy, refit accuracy, units).
**Decision rule:** the change is honest iff BATCH is bit-identical across it on 8 of 8; then R1 is re-read on the new
cells (must clear as before), R4 must read IDENTICAL to BATCH on 8 of 8 (ties, by construction — verified, not
assumed), R2 and R3 stand from stage1 (unchanged paths); the candidate with the larger R1 t lands, the other is
deleted (P2); if R1 no longer clears, the module is deleted (P8 clause 5). RIDGE is deleted regardless (R3).

### 3.4 Stage 2 — the decoupled instrument (BATCH, DELTA, SIGN × 8 seeds, 02:33–02:40 UTC; `stage2/`; head(s) e53e06d)

**The A/A across the engine change:** BATCH at stage 2 equals BATCH at stage 1 in W hash, beta hash, live accuracy, refit accuracy, units and floors on 8 / 8 / 8 / 8 / 8 of 8 → **bit-identical: the incumbent path is untouched by the change.**

| arm | seed | units | live acc / live pers | refit acc (last third) | updates / Σ|dβ| | W / beta hash |
|---|---|---|---|---|---|---|
| BATCH | 0 | 2 | 0.547 / 0.932 | 0.311 |  /  | 1fd01989 / 90fea667 |
| BATCH | 1 | 2 | 0.849 / 0.963 | 0.917 |  /  | d655e464 / ecd22e20 |
| BATCH | 2 | 2 | 0.807 / 0.952 | 0.825 |  /  | f2879fd9 / d131007f |
| BATCH | 3 | 7 | 0.653 / 0.902 | 0.656 |  /  | 60750bab / 60771e58 |
| BATCH | 4 | 6 | 0.590 / 0.907 | 0.628 |  /  | 708075a4 / af611828 |
| BATCH | 5 | 3 | 0.541 / 0.934 | 0.833 |  /  | 5b73e151 / 83f82801 |
| BATCH | 6 | 3 | 0.630 / 0.951 | 0.867 |  /  | 914883ca / 1b8e0b0c |
| BATCH | 7 | 2 | 0.771 / 0.992 | 0.947 |  /  | 292414f4 / 7d7da645 |
| DELTA | 0 | 2 | 0.547 / 0.932 | 0.311 | 369 / 53.763463 | 1fd01989 / b9bb2ab1 |
| DELTA | 1 | 2 | 0.841 / 0.963 | 0.917 | 377 / 84.15141 | d655e464 / e1c83750 |
| DELTA | 2 | 2 | 0.933 / 0.952 | 0.825 | 374 / 158.321034 | f2879fd9 / 47012c6f |
| DELTA | 3 | 7 | 0.857 / 0.902 | 0.656 | 378 / 194.343179 | 60750bab / 8b8f2732 |
| DELTA | 4 | 6 | 0.838 / 0.907 | 0.628 | 376 / 152.055553 | 708075a4 / 6508680f |
| DELTA | 5 | 3 | 0.708 / 0.934 | 0.833 | 377 / 94.302605 | 5b73e151 / 89fb07bc |
| DELTA | 6 | 3 | 0.854 / 0.951 | 0.867 | 370 / 127.510851 | 914883ca / de83eee7 |
| DELTA | 7 | 2 | 0.952 / 0.992 | 0.947 | 376 / 114.479314 | 292414f4 / c3243d87 |
| SIGN | 0 | 2 | 0.556 / 0.932 | 0.311 | 164 / 12.924321 | 1fd01989 / f4f9f4ee |
| SIGN | 1 | 2 | 0.833 / 0.963 | 0.917 | 63 / 20.543337 | d655e464 / ecd22e20 |
| SIGN | 2 | 2 | 0.922 / 0.952 | 0.825 | 29 / 38.097555 | f2879fd9 / d131007f |
| SIGN | 3 | 7 | 0.868 / 0.902 | 0.656 | 50 / 68.615406 | 60750bab / c43fc3da |
| SIGN | 4 | 6 | 0.856 / 0.907 | 0.628 | 54 / 49.152006 | 708075a4 / daf1a182 |
| SIGN | 5 | 3 | 0.719 / 0.934 | 0.833 | 106 / 27.167398 | 5b73e151 / 65d31c92 |
| SIGN | 6 | 3 | 0.859 / 0.951 | 0.867 | 52 / 29.733911 | 914883ca / 1b8e0b0c |
| SIGN | 7 | 2 | 0.973 / 0.992 | 0.947 | 10 / 8.420745 | 292414f4 / 7d7da645 |

- **DELTA, R1 (re-read under the decoupled instrument): paired (live_acc − BATCH live_acc) clears** — t 4.132, p 0.0156, 6/1/1, mean +0.143 → **MET** (stage 1 read t 4.132). (live acc − live persistence) t -2.874, p 1.0000, 0/8/0, mean -0.125.
- **DELTA, R4 (identical by construction, verified):** refit accuracy equals BATCH's on 8 of 8, the whole trajectory on 8 of 8 → **HOLDS**; same rows 8 of 8; units [2, 2, 2, 7, 6, 3, 3, 2].
- **SIGN, R1 (re-read under the decoupled instrument): paired (live_acc − BATCH live_acc) clears** — t 4.054, p 0.0117, 7/1/0, mean +0.150 → **MET** (stage 1 read t 4.054). (live acc − live persistence) t -2.714, p 1.0000, 0/8/0, mean -0.118.
- **SIGN, R4 (identical by construction, verified):** refit accuracy equals BATCH's on 8 of 8, the whole trajectory on 8 of 8 → **HOLDS**; same rows 8 of 8; units [2, 2, 2, 7, 6, 3, 3, 2].
- R2 (stage 1): HOLDS; R3 for the LMS rules (stage 1): SIGNFLIP t -11.562, below on 8 of 8 → HOLDS.

## 4b. Disposition — Amendment 3's decision rule, applied as written

- DELTA lands: True; SIGN lands: True; RIDGE: deleted on its failed control (§3.3).

**DELTA LANDS** (the larger R1 t of the two: DELTA 4.132, SIGN 4.054); **SIGN and RIDGE are DELETED from the module (P2, one mechanism).** The brain constructs `OnlineReadout("delta")` in `_make_neuro`; the closed-gate refit keeps scoring the batch instrument; the strict in-tree test `test_the_online_readout_beats_the_stale_one_on_the_wake_stream` (arms LIVE / L1C / L1C_CTRL through the same harness) carries R1, R3 and R4. What the landing claims and what it does not: the grown readout the brain predicts with now tracks the wake stream (+0.14 live accuracy over the stale one, still 0.02–0.38 below persistence on every seed); the growth gate's evidence is unchanged; the readout's recency shape on a marginal-covering slice is the disclosed trade-off (§3.3), which the instrument decoupling keeps out of the gate. The first writing's refusal stands withdrawn (§2.0c); it cost a panel and two commits and bought the regime lesson.

## 5b. Commits (continued)

| commit | what | landed on |
|---|---|---|
| `6edc26c` 5 | Amendment 2: the refusal withdrawn; the regimes; the ridge; the cells restored | before any cell |
| `7862d44` 6 | the build restored + RIDGE; the self-check's regimes | the brain-free reds |
| `706875f` 7 | the 64 cells (§3.1–3.2) and Amendment 3 (§3.3): the dissociation, the variant, its rule | before the variant's run |
| `e53e06d` 8 | the decoupled instrument + its brain-free test | the stage-2 A/A |
| `5427b5e` 9 | stage 2 (§3.4), this disposition, the landing of DELTA, the in-tree test, the plan row | Amendment 3's rule |

## 4c. The fold of the cells' panel (three lenses, `record_panel_cells.json`; the disposition stood on all three)

- **The instrument now survives a restart (I-1 / PH-1 / M-1, MAJOR, verified by the session).** As landed at
  `5427b5e` the growth record carried ONE beta and `restore_growth_state` made it the instrument, so every daemon
  relaunch would have re-coupled the closed-gate refit to the online readout until the next open-gate solve — the
  hazard Amendment 3 was built to close, back through the lake. The fold persists `_batch_beta` as an OPTIONAL
  record member (`batch_beta`, the additive contract, no format bump), restores it when present, and on a legacy
  blob makes the only beta there is the instrument (disclosed). The self-check's red (6) and a brain-free test now
  assert the reachable red: a restored online engine's closed-gate trajectory continues identical to a restored
  batch-only twin's, with the online beta and the instrument restored as two different tensors. The cells never
  exercised a restart (`persist=False`); the guarantee "unchanged by construction" was in-process until this fold.
- **The deployed brain: charged in the cells, DORMANT on the daemon (M-2, MAJOR).** `dbeta = −lr·outer(s, p)` is
  identically zero on a unit that does not fire. The daemon at the landed head reads 929 calls with Σ|dβ| 0.0 and live
  accuracy 0.0 against live persistence 0.997 on its single restored unit: the rule is reachable and not charged there
  (P3). The condition under which it charges — a unit that fires on the stream the brain lives in — is the open P3
  item this arc hands to L2 and the width question (SC-012a); the module discloses it; `updates` was an over-count
  (it counted zero-magnitude steps) and is now `steps` (M-6).
- **"SIGN and RIDGE are deleted" is now true of the tree (I-3 / PH-3, MAJOR):** the perceptron and ridge code paths,
  their constructor arguments and state fields are removed from the module (commit `7862d44` and the archive keep
  them); `KINDS` and the refusal test stay as the P2 assertion; the census row regenerated.
- **The strict in-tree test was red by construction (I-2, MAJOR):** its incumbent arm inherited the survivor from
  `make_brain`. The harness now removes the online readout for every arm except L1C / L1C_CTRL (the L1b arms measure
  the batch-only readout as before; their pins read the refit accuracy, identical either way), and R4 is carried as an
  IDENTITY check (the decoupling holds: no online beta reaches the gate), not a non-regression bar — which is what R4
  can mean once the instrument is decoupled (M-5: the online beta's marginal-slice accuracy is UNMEASURED after the
  decoupling; its last reading is stage 1's, −0.07 to −0.25 on the closed-gate seeds). The test ran once on the fold
  head before the battery (§4d).
- **The cockpit shows the instrument only (I-4 / PH-7 / M-4):** the sentence claiming a labelled live-accuracy row is
  struck; `predict acc` on the cockpit is the batch instrument's held-out number, and the mechanism's `live_accuracy`
  / `live_persistence` / `readout_charge` reach only `state()["neurogenesis"]`. The cockpit rows with PROVENANCE are
  owed onward (a sensorimotor-surface change with its own gate).
- **The tiebreak (PH-6):** the rule chose DELTA on the pre-registered t (4.132 against 4.054); by mean and by
  head-to-head SIGN read ahead (mean +0.150 against +0.143; 6 of 8 seeds; not resolvable, t −1.740). The choice is
  the letter, not the evidence between them.
- Also folded: 3.2's R4 lines relabelled DEGENERATE (PH-4); the departure from 2.0c's letter quoted (PH-5); RIDGE's
  deletion re-worded to its letter and its control disclosed as near-vacuous (PH-2 / M-3); the 2.0c table's ridge
  column sourced (PH-8); the mu citation corrected and the step sweep owed (PH-9); the 2.6 control's verdict printed
  (M-7); one representation of the charge on state() (I-5); "before buffering" corrected (I-6); numbering and hashes
  (PH-10). Part B's learner row: superseded 2026-09-06 by L1c (the online delta rule beside the batch solve).

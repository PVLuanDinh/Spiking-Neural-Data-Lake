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

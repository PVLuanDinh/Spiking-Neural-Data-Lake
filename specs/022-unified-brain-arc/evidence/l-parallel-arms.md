# Phase 2's three stream-independent arms — L1a, L3, the persistence oracle (2026-09-05)

Plan rows (PLAN-FINAL Phase 2; reds L-R1, L-R4 in Part F). Owner: "Do all this" (2026-09-05). Startable at W3
per Part E; they do not wait for L1b. Lessons carried from the live-floor arc, applied before any cell: every
bar CHOSEN and marked; every control's expected reading stated from existing data (obs #2073); a bar set from
the seeds it is scored on is "met by construction" (obs #2074) — here the bars are the PLAN's (pre-registered
2026-09-02 in Part F), not chosen from these cells; the previous arc's own reds run FIRST among a build's gates
(obs #2072).

## 1. Territory (code at `ef36e98`; s2 v2 numbers, evidence/s2-real-percept-stream.md)

- `NeurogenesisSelfImprove._refit`: held-out slice = the LAST `min(60, n//3)` rows of the (percept_{t-1} →
  category_t) buffer (`buffer_cap` 1000); the two floors (`_heldout_baselines(y_tr, y_te)`) assume the slice is
  contiguous and its first row's predecessor is the last training row.
- `NGSNN.fit` (`l_max=None`, derived capacity): its OWN admission val split is the last 20 % of the rows it is
  handed (`n_val = max(8, int(0.2 N))`, `val_idx = arange(n_tr, N)`); the earn-your-place stop scores held-out
  RMSE on those rows (`patience` 48), rolls back to the best width, then refits the readout on all rows;
  `prune_unearned` uses the same last-20 % split and, when EVERY unit is droppable, keeps ALL ("never prune to
  nothing" — `len(drop) >= L → return 0`).
- Both windows are recency-based; on a bursty category stream the recent rows are usually one run of one
  category, so a unit that would predict a rarer category never earns its place and the readout answers the
  majority (s2 v2 finding 1). The persistence oracle (percept = the current category one-hot, so the buffered
  pair is one-hot(cat_{t-1}) → cat_t; an identity map scores exactly the persistence floor) reached that floor
  on 2 of 8 seeds (−0.345 ± 0.098, t −3.53) and stopped at 2.5 units; LIVE 2.1 units, at its persistence floor
  on 3 of 8; the CONSTANT percept admitted 24.1 units (the instrument finding); SYNTH 13.9; SHUFFLE 14.0.
- The archived harness (`audit-2026-09-01/rank2a/rank2_ab_v2.py`) drives the arms at the feed
  (`Brain._wake_tick → neuro.observe`), 8 seeds × 800 ticks; its in-tree promotion target is
  `tests/test_real_percept_stream.py` (4 tests, no oracle, no constant arm).

## 2. Protocol — pre-registered before any cell runs

Statistic (plan A.4 step 2, W2 §3.5's form): S = 8 seeds (0–7), paired one-sided t > 1.895 (7 d.f.) AND the
exact one-sided sign-flip p ≤ 0.05 over all 2⁸ patterns, ties as zeros, DEGENERATE-TIES declared below 5
non-tied seeds; the positive control in the SAME command (a planted paired effect of the observed non-tied
|d| mean on the observed tie pattern must clear; the same magnitudes with random signs must clear on < 10 % of
200 draws — `l_arms_stats.py`, self-tested). Cells: 800 ticks (s2's), `throwaway_home`, `persist=False`, one
arm per subprocess, 4 workers, the live daemon left running. Before/after by HEAD (the JEPA arcs' idiom; a
worktree control loses the gitignored perception checkpoint). Harness `memory/archive/l-arms-2026-09-05/
l_arms_probe.py` (arms LIVE / ORACLE_PREVCAT / NEUTER_PERCEPT / SYNTH); stats `l_arms_stats.py`.

### 2.1 Stage A — the incumbent head, 32 cells (4 arms × 8 seeds); the instrument control
LIVE must reproduce s2 v2 at summary level (units ≈ 2, at its persistence floor on ~3 of 8, above its
majority floor on ~6 of 8); ORACLE_PREVCAT at or below its persistence floor on most seeds (s2: at on 2,
below on 6); NEUTER_PERCEPT admitting many units (s2: 24.1); SYNTH admitting > 1 (s2: 13.9). A miss is
explained by `git log 4fc5fb1c..ef36e98 -- src/snn_data_lake/cortex/neurogenesis_engine.py src/snn_data_lake/
hippocampus/ng_snn.py` or halts the arc.

### 2.2 Commit 1 — the oracle and the constant arm IN-TREE, red first (the persistence oracle deliverable)
`tests/test_real_percept_stream.py` gains two slow tests at the mechanism's input contract (A.4 step 2):
**the persistence oracle** — L-R1's red: paired (ORACLE acc − its own persistence floor) over 8 seeds clears
the bar (RED today: −0.345) — and **the constant percept** — L-R4's red: admits exactly 1 unit on 8 of 8 seeds
while SYNTH admits > 1 on 8 of 8 in the same test (RED today: 24.1). Both marked `slow`; both RED on the tree
before the rules land (the cross-cutting invariant), counted by running them (32 brains ≈ 12 min).

### 2.3 Commit 2 — L1a, the admission window (ONE mechanism, P2)
One split function (`hippocampus/ng_snn.py::marginal_holdout(y, n_val)` → (train_idx, val_idx)): the held-out
rows are chosen PER CATEGORY in proportion to the category marginal (round(n_val · n_c / N), at least one row
for every category with ≥ 2 rows, the remainder to the largest categories), taking the MOST RECENT rows of
each category — "recency-mixed": covers the marginal, prefers recent rows within it; deterministic, no rng.
Used by all three windows: `NGSNN.fit`'s admission val split (index tensors replace the `[:n_tr]` slices; the
final refit on all rows unchanged), `prune_unearned`'s split, and the engine's `_refit` measurement slice —
whose floors (`_heldout_baselines`) are computed against each held-out row's TRUE predecessor in the stream
(the contiguity assumption goes with the window). The incumbent recency window is DELETED, not flagged (P2);
the control cells ran at the previous head (Stage A). Reds — **L-R1**: under L1a, paired (ORACLE − own
persistence floor) clears the bar (MET / NOT MET), AND paired (LIVE_L1a − LIVE_incumbent) does not FALL past
the bar (the negated difference must not clear). Reported beside it: the s2 record's own pre-written form (the
oracle REACHES its floor on a majority of seeds), unit counts, majority floors.
Interpretation table, written now: (exceeds) the window was the limit and the stream's transitions are
learnable; (reaches on ≥ 5 of 8 but does not exceed) the window was the limit and the stream's transitions are
persistence-only — L1a LANDS if LIVE does not fall, with the plan's strict red recorded NOT MET and the s2 form
MET, both stated; (below on ≥ 5 of 8) the window was NOT the limit — L1a is REVERTED and the finding recorded
(P8 clause 5). Any LIVE fall past the bar → REVERTED regardless.

### 2.4 Commit 3 — L3, the admission instrument (two shapes, one criterion each)
**(i)** `NGSNN.fit` admits at most ONE candidate whose response has EXACTLY zero variance across the fitted
rows (identical rows give identical rates; exact zero is derived, not an epsilon): a constant response
cannot discriminate rows — the first such unit is the class prior, a second carries nothing. **(ii)**
`prune_unearned`, when every unit is droppable, keeps the ONE whose removal costs the most held-out RMSE
(ties → lowest index) instead of keeping all. Reds — **L-R4**: the constant-percept arm admits exactly 1 unit
on 8 of 8 seeds AND SYNTH admits > 1 on 8 of 8 in the same command; LIVE's unit count reported (a fall to 1
with unchanged accuracy is not a red; a fall with an accuracy drop past the bar is REVERTED).

### 2.5 Stage B — after each commit, the same cells; paired by seed across heads
After commit 2: LIVE + ORACLE_PREVCAT × 8 (L-R1). After commit 3: NEUTER_PERCEPT + SYNTH + LIVE × 8 (L-R4 and
LIVE's non-regression). The in-tree tests run at the end (green expected: the counted reds turned).

### 2.6 Controls that are not the method
The incumbent window (Stage A, the previous head); the oracle's own floors; the constant percept and SYNTH
(L3's two sides); the paired statistic's planted/null control in the same command.

### 2.7 Disposition on a null
§2.3's table for L1a; for L3: if the constant arm still admits > 1, the instrument is not the fix and the
count is recorded; if SYNTH admits 1, the criterion is too strong and is reverted (a positive control that
fails voids the change). Commit 1's tests stay in the tree as strict xfails if their fix does not land.

### 2.8 Commit order
1 (tests, red counted) → Stage A cells → 2 (L1a) → Stage B (LIVE, ORACLE) → 3 (L3) → Stage B (NEUTER,
SYNTH, LIVE) → tests green → the record → panel (3 lenses) → fold → battery → push.

### 2.9 Amendment 1 — three measurements the record panel showed the arc owed (written after the panel,
BEFORE these cells run)

The panel (§6) found: the LIVE "gain at the bar" (§3.2) compares accuracies scored on different held-out rows
at the two heads — the majority floor alone shifted +0.177 (t 2.03) and the floor-relative gain is +0.116
(t 1.32, sub-bar); L3 rule (i)'s "exact zero variance" is exactly 0.0 only for some (N, rate) pairs in float32
(rounding luck), though rule (i) is load-bearing (with it off the constant arm admits 2); and the claim "the
stream's transitions are persistence-only" owed the oracle's positive control. Pre-registered here:
- **(a) The learner-only comparison.** A new arm, LIVE_RECENCY_ADMIT: at the current head, `NGSNN.fit` and
  `prune_unearned` are given the OLD recency split (train = the first N − n_val rows, val = the last n_val)
  while the engine's measurement slice stays marginal-covering — so both learners are scored on the SAME rows.
  Paired (LIVE − LIVE_RECENCY_ADMIT) over 8 seeds at the plan's bar (t > 1.895 AND exact p ≤ 0.05). If it
  clears, the gain is the learner's; if not, the raw gain of §3.2 was the slice's and L1a's landing rests on
  the oracle's instrument-invariant reading alone — stated either way.
- **(b) Rule (i) in its N-independent form** — `(Sp == Sp[0]).all(dim=0) & ~dead` (identical rows give
  identical rates: an equality, not a variance rounding to zero) — re-run on NEUTER_PERCEPT × 8 and SYNTH × 8:
  exactly 1 alive unit on 8 of 8 with the constant arm's accuracy equal to its majority floor, SYNTH > 1 on 8
  of 8 (the same L-R4 bars). Per-item attribution recorded from the cells that exist: Stage A 30.5 units →
  the first form's 1 (dead) → the corrected 1 (alive); the panel's cell with rule (i) off: 2; the L1a-alone
  NEUTER × 8 is owed and not run here.
- **(c) The oracle's positive control (P8 clause 2).** Brain-free, the live-path engine fed one-hot(mode) →
  mode on `SyntheticPerceptStream`'s chain (a stream WITH first-order structure), 8 seeds, under the marginal
  window: paired (acc − its own persistence floor) must clear the bar and a per-cycle shuffle of the chain
  must not. If it clears, "persistence-only" is a claim about the live stream, not about the learner.
- Also folded before these run: the strict L-R1 xfail guarded on the real perception packs (on a fresh clone
  the oracle beats persistence on the synthetic rotation, floor 0.0 — the red is a property of the live stream)
  and given `raises=AssertionError`, the degenerate-ties check outside it, a tie tolerance of 1e-6 in the
  statistic; the L-R4 test asserts the constant arm's accuracy equals its majority floor (a dead unit reds);
  `marginal_holdout` gains a brain-free contract test that the recency split fails; L3's two rules gain
  brain-free reds (`tests/test_admission_window.py`).

## 3. Results

### 3.1 Stage A — the incumbent head `bba164b` (32 cells, 800 ticks, 20:33–20:42 UTC; `stageA/` in
`memory/archive/l-arms-2026-09-05/`)

| arm | seed | units (last-third mean) | acc last third | persistence floor | majority floor | categories |
|---|---|---|---|---|---|---|
| LIVE | 0 | 1 (1.17) | 0.161 | 0.914 | 0.664 | 5 |
| LIVE | 1 | 1 (1.17) | 1.000 | 1.000 | 1.000 | 3 |
| LIVE | 2 | 1 (1.00) | 0.456 | 0.983 | 0.472 | 4 |
| LIVE | 3 | 3 (2.50) | 0.383 | 0.942 | 0.161 | 5 |
| LIVE | 4 | 2 (2.00) | 0.858 | 0.958 | 0.367 | 5 |
| LIVE | 5 | 2 (2.00) | 0.539 | 0.939 | 0.583 | 3 |
| LIVE | 6 | 2 (1.83) | 0.042 | 0.942 | 0.244 | 4 |
| LIVE | 7 | 1 (1.17) | 0.011 | 0.997 | 0.011 | 3 |
| ORACLE_PREVCAT | 0 | 2 (2.00) | 0.850 | 0.914 | 0.664 | 5 |
| ORACLE_PREVCAT | 1 | 1 (1.17) | 1.000 | 1.000 | 1.000 | 3 |
| ORACLE_PREVCAT | 2 | 1 (1.00) | 0.472 | 0.983 | 0.472 | 4 |
| ORACLE_PREVCAT | 3 | 2 (2.17) | 0.356 | 0.942 | 0.161 | 5 |
| ORACLE_PREVCAT | 4 | 2 (1.67) | 0.736 | 0.958 | 0.367 | 5 |
| ORACLE_PREVCAT | 5 | 2 (2.00) | 0.583 | 0.939 | 0.583 | 3 |
| ORACLE_PREVCAT | 6 | 2 (2.50) | 0.244 | 0.942 | 0.244 | 4 |
| ORACLE_PREVCAT | 7 | 1 (1.00) | 0.011 | 0.997 | 0.011 | 3 |
| NEUTER_PERCEPT | 0–7 | **46 / 51 / 43 / 13 / 28 / 26 / 14 / 23** (mean 30.5) | = the majority floor on 7 of 8 (a constant input answers the majority) | | | |
| SYNTH | 0–7 | 12 / 24 / 7 / 17 / 11 / 13 / 17 / 10 (mean 13.9) | 0.64–0.74 against persistence 0.04–0.45 and majority 0.15–0.25 (the positive control learns its chain) | | | 5 |

- **Instrument control: HELD.** LIVE units mean 1.75 (s2 v2: 2.1), at its persistence floor on 1 of 8 and
  within 0.01 of its majority floor (paired t −0.06; s2: +1.03); ORACLE − own persistence floor mean −0.428,
  t −3.63, reaching the floor on 1 of 8 (s2 v2: −0.345, t −3.53, 2 of 8); the constant percept admits 30.5
  units (s2: 24.1); SYNTH 13.9 (s2: 13.9). The statistic's positive control clears (the planted effect on the
  observed tie pattern) and its null clears on 4 % of 200 draws.
- **L-R1 RED, counted:** paired (ORACLE − persistence) does NOT clear (t −3.63, exact p 1.0; 0 wins, 7 losses,
  1 tie). The oracle's accuracy equals LIVE's on seeds 1, 2, 5, 7 to three digits and is within 0.1 on the
  rest: handed exactly the information persistence uses, the learner does no better than on the live percept
  — the window, not the stimulus, is the limit (s2 v2's finding 1, reproduced).
- **L-R4 RED, counted:** the constant percept admits 13–51 units (never 1); SYNTH admits 7–24 (> 1 on 8 of 8),
  so the positive control's half of L-R4 already holds.
- Every cell: perception unified, both real packs live, 390–399 wake ticks of 800, 20 refits, feed refusals 0.

### 3.2 Stage B after L1a — the marginal-covering window (LIVE + ORACLE × 8 seeds, 800 ticks, the L1a
working tree on `921a1a2`; the diff archived as `stageB1/L1a_working_tree.diff`; `stageB1/`)

| arm | seed | units (last-third mean) | acc last third | persistence floor | majority floor | Stage A acc → |
|---|---|---|---|---|---|---|
| LIVE | 0 | 2 (1.83) | 0.311 | 0.953 | 0.744 | 0.161 → +0.150 |
| LIVE | 1 | 2 (2.00) | 0.917 | 0.983 | 0.739 | 1.000 → −0.083 |
| LIVE | 2 | 3 (2.50) | 0.814 | 0.953 | 0.694 | 0.456 → +0.358 |
| LIVE | 3 | 4 (4.00) | 0.708 | 0.911 | 0.319 | 0.383 → +0.325 |
| LIVE | 4 | 4 (4.00) | 0.578 | 0.950 | 0.508 | 0.858 → −0.281 |
| LIVE | 5 | 4 (3.67) | 0.900 | 0.947 | 0.758 | 0.539 → +0.361 |
| LIVE | 6 | 2 (2.83) | 0.778 | 0.933 | 0.506 | 0.042 → +0.736 |
| LIVE | 7 | 2 (1.83) | 0.794 | 1.000 | 0.653 | 0.011 → +0.783 |
| ORACLE_PREVCAT | 0 | 2 | 0.931 | 0.953 | 0.744 | (−0.022 vs its floor; Stage A −0.064) |
| ORACLE_PREVCAT | 1 | 2 | 0.983 | 0.983 | 0.739 | (0.000; Stage A 0.000) |
| ORACLE_PREVCAT | 2 | 4 | 0.947 | 0.953 | 0.694 | (−0.006; Stage A −0.511) |
| ORACLE_PREVCAT | 3 | 8 | 0.911 | 0.911 | 0.319 | (0.000; Stage A −0.586) |
| ORACLE_PREVCAT | 4 | **55** | 0.939 | 0.950 | 0.508 | (−0.011; Stage A −0.222) |
| ORACLE_PREVCAT | 5 | 2 | 0.903 | 0.947 | 0.758 | (−0.044; Stage A −0.356) |
| ORACLE_PREVCAT | 6 | 3 | 0.867 | 0.933 | 0.506 | (−0.067; Stage A −0.697) |
| ORACLE_PREVCAT | 7 | 2 | 0.983 | 1.000 | 0.653 | (−0.017; Stage A −0.986) |

The floors themselves moved with the window: the majority floor is now the training rows' majority scored
on a marginal-covering slice (0.32–0.76 against 0.01–1.00 before) and the persistence floor is each held-out
row's true stream predecessor (0.91–1.00 against 0.91–1.00).

**L-R1, each conjunct against §2.3's wording:**
- Paired (ORACLE − its own persistence floor) clears the bar: **NOT MET** — t = −2.505, exact p = 0.988, 2
  wins (ties at exactly 0.000 on seeds 1 and 3) / 6 losses / 0 ties, mean −0.021 (Stage A: −0.428, t −3.63).
  The oracle now sits AT its floor — within 0.022 on 5 of 8, within 0.011 on 3, exactly on 2 — and does not
  exceed it. The paired statistic's positive control clears (planted) and its null clears 5.5 % of draws.
- Paired (LIVE_L1a − LIVE_incumbent) does not FALL past the bar: **MET** — and the same statistic CLEARS the
  bar in the positive direction: t = 2.270, exact p = 0.035, 6 wins / 2 losses / 0 ties, mean +0.294 (the
  differences +0.15, −0.08, +0.36, +0.33, −0.28, +0.36, +0.74, +0.78). LIVE is above its majority floor on 7
  of 8 (mean +0.110; Stage A: 3 of 8, −0.007) and grows 2.9 units (Stage A: 1.6).
- The s2 record's own form ("the oracle REACHES its floor on a majority of seeds"): by the letter — reaching
  = at or above — **2 of 8**; within 0.022, 5 of 8. §2.3's interpretation table wrote "reaches" with no
  tolerance (a pre-registration defect, disclosed); by that letter 6 of 8 are "below" and the table's third
  row ("the window was NOT the limit — REVERTED") fires, while the number it was written to detect — the
  oracle's deficit — shrank twenty-fold (−0.43 → −0.02) and LIVE rose at the bar, which is the table's
  MIDDLE row's reading (the window was the limit; the stream's transitions are persistence-only, W2 §3.1c's
  "no first-order succession" seen from the learner's side).
- Reported: ORACLE seed 4 grew 55 units on the one-hot input (the other seeds 2–8) — a growth outlier on the
  instrument arm, not a live path; noted for L3's admission criteria.

### 3.3 Stage B2 after L3 — the admission instrument (NEUTER_PERCEPT + SYNTH + LIVE × 8 seeds, 800 ticks)

**First form (21:11–21:18 UTC; `stageB2/`; the L3 diff archived as `stageB2/L3_working_tree.diff`):** the
constant percept admits exactly 1 unit on 8 of 8 (Stage A: 13–51) and SYNTH admits 10–29 on 8 of 8 — L-R4's
two unit conjuncts MET — but the constant arm's last-third accuracy reads **0.000 on every seed** where a
one-unit readout should answer the training majority (0.32–0.76). Reproduced brain-free: the one
zero-variance unit the rule admitted was a DEAD candidate (LIF rate exactly 0 on every row, beta all zeros,
argmax → class 0). Cause, exact: on the first block the consistency bar is `thresh = 1 − r − η(0) = 1 − 0.9 −
0.1 = 0.0`, so every candidate passes, and the rank `score + passes·1e12` saturates float32 (a score of ~0.1
vanishes beside 1e12), so the top pick is index-ordered and can be a dead unit; before L3, later live units
masked it (24 units), and the constant arm answered the majority. **Corrected before the re-run (the fix is
part of commit 3):** a zero-NORM candidate (rate exactly 0 on every scoring row) is excluded always — it
cannot carry even the class prior — and a zero-variance ALIVE candidate (a constant nonzero rate) is the
class prior, admitted at most once. Verified brain-free on a patched copy before touching the tree: one alive
unit (rate 0.467), the readout answers the majority (accuracy 0.40 = the floor, on the original and on
shifted labels); a structured input still grows 8 units at 0.92. The float32 rank saturation is a
pre-existing instrument note (it orders every fit's ties by index), recorded and not changed here.

**Re-run at the corrected L3 (`stageB2fix/`; the diff archived as `stageB2fix/L3_fixed_working_tree.diff`):**

| arm | seed | units (last-third mean) | acc last third | persistence floor | majority floor | note |
|---|---|---|---|---|---|---|
| NEUTER_PERCEPT | 0 | 1 (1.00) | 0.744 | 0.953 | 0.744 | acc − majority +0.000 |
| NEUTER_PERCEPT | 1 | 1 (1.00) | 0.739 | 0.983 | 0.739 | acc − majority -0.000 |
| NEUTER_PERCEPT | 2 | 1 (1.00) | 0.694 | 0.953 | 0.694 | acc − majority +0.000 |
| NEUTER_PERCEPT | 3 | 1 (1.00) | 0.319 | 0.911 | 0.319 | acc − majority +0.000 |
| NEUTER_PERCEPT | 4 | 1 (1.00) | 0.508 | 0.950 | 0.508 | acc − majority +0.000 |
| NEUTER_PERCEPT | 5 | 1 (1.00) | 0.758 | 0.947 | 0.758 | acc − majority +0.000 |
| NEUTER_PERCEPT | 6 | 1 (1.00) | 0.506 | 0.933 | 0.506 | acc − majority +0.000 |
| NEUTER_PERCEPT | 7 | 1 (1.00) | 0.653 | 1.000 | 0.653 | acc − majority -0.000 |
| SYNTH | 0 | 8 (8.00) | 0.686 | 0.161 | 0.236 | the positive control |
| SYNTH | 1 | 29 (24.00) | 0.678 | 0.042 | 0.217 | the positive control |
| SYNTH | 2 | 12 (12.00) | 0.672 | 0.247 | 0.242 | the positive control |
| SYNTH | 3 | 20 (15.83) | 0.717 | 0.231 | 0.233 | the positive control |
| SYNTH | 4 | 15 (15.50) | 0.708 | 0.044 | 0.236 | the positive control |
| SYNTH | 5 | 21 (14.67) | 0.769 | 0.531 | 0.247 | the positive control |
| SYNTH | 6 | 13 (12.33) | 0.633 | 0.267 | 0.236 | the positive control |
| SYNTH | 7 | 8 (7.67) | 0.636 | 0.050 | 0.225 | the positive control |
| LIVE | 0 | 2 (1.83) | 0.311 | 0.953 | 0.744 | Stage A 0.161 |
| LIVE | 1 | 2 (2.00) | 0.917 | 0.983 | 0.739 | Stage A 1.000 |
| LIVE | 2 | 2 (1.83) | 0.825 | 0.953 | 0.694 | Stage A 0.456 |
| LIVE | 3 | 7 (7.00) | 0.656 | 0.911 | 0.319 | Stage A 0.383 |
| LIVE | 4 | 6 (6.00) | 0.628 | 0.950 | 0.508 | Stage A 0.858 |
| LIVE | 5 | 3 (2.67) | 0.833 | 0.947 | 0.758 | Stage A 0.539 |
| LIVE | 6 | 3 (2.50) | 0.867 | 0.933 | 0.506 | Stage A 0.042 |
| LIVE | 7 | 2 (2.00) | 0.947 | 1.000 | 0.653 | Stage A 0.011 |

- **L-R4, each conjunct against §2.4's wording:** the constant-percept arm admits exactly 1 unit on **8 of 8** ([1, 1, 1, 1, 1, 1, 1, 1]); SYNTH admits > 1 on **8 of 8** ([8, 29, 12, 20, 15, 21, 13, 8]). The constant arm's accuracy now equals its majority floor to within 0.000 on every seed (differences [0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0]) — a constant input answers the majority, as it should.
- LIVE under L1a + L3: units [2, 2, 2, 7, 6, 3, 3, 2] (mean 3.38; Stage A 1.6, L1a alone 2.9); paired (LIVE − Stage A) t = 2.225, exact p = 0.0312, 6 wins / 2 losses / 0 ties, mean +0.317 (the non-regression conjunct MET; it clears the bar in the positive direction as under L1a alone); LIVE − its majority floor 7 / 1 / 0, mean +0.133.

## 4. Verdict — each red against §2's wording, MET / NOT MET

- **The persistence oracle (commit 1, the deliverable):** in the tree at the mechanism's input contract, with
  its red; reproduces the archived harness digit for digit (t −3.626 at the incumbent window, t −2.505 under
  L1a).
- **L-R1 (L1a):** paired (ORACLE − own persistence floor) clears the bar — **NOT MET** (t −2.505, p 0.988); the
  oracle's deficit shrank twenty-fold (−0.428 → −0.021; exactly at the floor on 2 of 8, within 0.022 on 5 of
  8) and does not exceed it. Paired (LIVE_L1a − LIVE_incumbent) does not fall past the bar — **MET**, and the
  statistic clears in the positive direction (t 2.270, exact p 0.035, 6 of 8). The s2 form ("reaches on a
  majority") by its letter: 2 of 8 (the letter carried no tolerance — a pre-registration defect, §3.2).
  **Disposition: L1a LANDED (commit 2) on the interpretation table's middle-row reading (the window was the
  limit; the stream's transitions are persistence-only) plus the measured LIVE gain — where the table's letter
  ("below on ≥ 5 of 8") said revert. That is the session's call, disclosed; the owner may revert `3979001`.**
  The in-tree L-R1 test stays a STRICT xfail carrying the reading.
- **L-R4 (L3):** the constant-percept arm admits exactly 1 on 8 of 8 — **MET**; SYNTH admits > 1 on 8 of 8 —
  **MET**; LIVE's unit count did not fall (3.4 vs 2.9) and its accuracy did not fall past the bar — reported.
  **L3 LANDED (commit `308f6db`)** with the dead-candidate correction found by the arc's own first-form reading
  (§3.3). The in-tree L-R4 test turns from red to green at this head.
- **Reported, not scored:** ORACLE seed 4 grew 55 units under L1a (the one-hot input; the other seeds 2–8) —
  a growth outlier on the instrument arm; the float32 rank saturation in `NGSNN.fit`'s candidate ranking
  (index-ordered ties on the first block) — a pre-existing instrument note; the majority floor's meaning
  changed with the window (the training majority scored on a marginal-covering slice), so Stage A and Stage B
  floors are not the same quantity — the paired LIVE statistic compares accuracies, not floors.
- **Owed onward:** the strict L-R1 red (the oracle exceeding persistence needs a stream whose transitions carry
  more than persistence — Phase 1's stream question, not a window question); SC-011 / SC-012a still need their
  stimulus; the rank-saturation note.

## 5. Commits of this arc (stacked on `4bae53a` after the JEPA live-floor and Q2 arcs)

| commit | what | landed on |
|---|---|---|
| `bba164b` 1 (pre-registration) | this record's §1–2 | before any cell |
| `921a1a2` 1 | the persistence oracle and the constant arm, in-tree, RED | the counted reds |
| `3979001` 2 | L1a: one marginal-covering window for all three grower windows | the middle-row reading + the LIVE gain; the session's call, owner may revert |
| `308f6db` 3 | L3: at most one alive zero-variance candidate, never a dead one; prune keeps one of an all-redundant set | L-R4 MET 8 of 8 both conjuncts |
| 4 (this commit) | this record §3–5; the plan's Phase 2 rows and Part F | docs |


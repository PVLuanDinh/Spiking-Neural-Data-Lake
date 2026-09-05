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

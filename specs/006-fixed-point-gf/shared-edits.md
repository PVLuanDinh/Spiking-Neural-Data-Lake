# Shared-file edits required by specs/006-fixed-point-gf (for the coordinating agent)

I own `reasoning/**`, `tools/arc_*`, `tests/test_arc_fix.py`, and this spec dir. The following
touch files forbidden to me, or repo-wide actions reserved for the coordinator.

## 1. docs/coverage -- Gf faculty evidence line (T028; the verdict DID change it)

Measured on held-out ARC-AGI-1 evaluation at the final state (fix combinator retired, mint
intake adopted; `final_runs.log`): **18 of 400 = 4.50%**, up from the 12 of 400 = 3.00%
measured baseline at `95c3f806` (docs previously said 11 of 400 = 2.75%; the re-measured
baseline had already drifted to 12).

CONTAMINATION QUALIFIER the ledger must carry if it quotes the split: 20 of the 400
evaluation tasks were the mint's Phase-1/3 dev set (`scratchpad/oracle_dev.json`), and 3 of
the 6 newly gained IDs (00576224, 00dbd492, 0a2355a6) are in that dev set. The 3 CLEAN gains
(37d3e8b2, 84f2aca1, ae58858e) were never inspected during development. A maximally
conservative held-out claim is therefore 15 of 380 never-inspected tasks gained-vs-baseline
context; the plain benchmark figure is 18 of 400 with the qualifier. Suggested evidence line:

  ARC-AGI-1 evaluation 18 of 400 = 4.50% (tools/arc_score.py, pass@2; up from 12 of 400 at
  95c3f806 via the specs/006 mint merge; 3 of the 6 gained IDs overlap the mint's 20-task
  dev subset of this split -- the other 3, incl. one FUNCTIONAL affine rule, are clean).

## 2. Dormancy-gate bookkeeping (P3)

`reasoning/arc_mint.py` is now reachable from the LIVE path: ArcExpert.answer ->
arc_solver.synthesize -> _pool -> arc_mint.mint_records (the P3 cure of spec finding F7).
If `substrate/dormancy_gate.py` or `substrate/method_dormancy_gate.py` carries arc_mint
exemptions or method keys for the retired `candidates`/`mint` symbols, they need updating --
both gates are forbidden to me. Note `method_dormancy_gate.py`'s DOCSTRING (line ~26) cites
arc_mint's `("affine", 3, _fit_affine)` dispatch-table example; the table still exists (the
fitters now return `(rule, params)` tuples), so the example remains true.

## 3. run_selfchecks.py -- no edit needed, one heads-up

`reasoning.arc_mint` is already registered. Its self-check now scores the 20-task dev set
THROUGH make_solver at bench budgets: the smoke gate gains roughly 60-90 s of wall time.
If that is too slow for the gate, trimming the dev pass is a coordinator call, not mine.

## 4. Final gate set + commit (T002/T030 deltas, recorded in baseline.txt and tasks.md)

Full `run_selfchecks.py`, full `pytest` (>=70% coverage), and repo-wide ruff were NOT run by
me (concurrent specs/005 agent in the same tree at baseline time; the coordinator owns the
final gate set). Targeted greens on my scope: arc_solver / arc_mint / arc_bench self-checks,
tests/test_arc_fix.py 11 of 11, ruff clean on every touched file.

citation_guard: census.md / verdict.md / docs/DEPTH_CEILINGS.md cite tools/arc_census.py and
tools/arc_fixdepth_control.py -- both are new files and must be part of the feature commit.

## 5. graphify update . (T031)

Repo-wide regeneration; deferred to the coordinator to run once after the commit.

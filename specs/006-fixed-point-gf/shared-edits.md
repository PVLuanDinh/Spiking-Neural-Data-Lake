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

---

# DESIGN B additions (2026-08-03) -- docs/INVENTION.md section 3

Appended by the Design B implementation. Nothing below was applied.

## 6. Register the new module (REQUIRED -- the dormancy gate fails without it)

`src/snn_data_lake/reasoning/arc_residual.py` is new. CLAUDE.md's rule: register a new module in
`__init__.py` REGIONS **and** `run_selfchecks.py`, then it must be reachable from a live path or
listed in `dormancy_gate.EXEMPT`.

* `src/snn_data_lake/__init__.py` -- add `arc_residual` to the `reasoning` region list.
* `run_selfchecks.py` -- add `snn_data_lake.reasoning.arc_residual`. Its `main()` returns 0 on pass
  and 1 on fail, prints ASCII only, touches no benchmark data and no filesystem, and runs in well
  under a second.

No EXEMPT entry is needed: `arc_mint` imports it at module level and `arc_solver._pool()` imports it
per task, so it sits on the one live ARC solve path.

## 7. docs/DEPTH_CEILINGS.md -- the library A/B no longer exists

`ArcLibrary` is DELETED (P2, docs/INVENTION.md 3.4 item 1), with its `cap = 64`, `_defs`, `_seen`,
the `lib:` pool branch and the `library=` parameters of `_pool` / `synthesize` / `make_solver`.
`tools/arc_score.py`'s positional contract changed from `<training|evaluation> <on|off> [limit]` to
`<training|evaluation> [limit]`; a legacy `on|off` token is still ACCEPTED and prints a RETIRED
notice rather than silently meaning something new, so a quoted reproduction command still runs. The
doc's library-on / library-off paragraph now describes a mechanism that is gone.

Confirmed independently at HEAD `f3bf8c70` before deleting anything
(`specs/006-fixed-point-gf/designb/base_*.txt`):

| split | library ON | library OFF | solved-ID sets |
|---|---|---|---|
| evaluation | 18 of 400, 217 s, 4 composites learned | 18 of 400, 215 s | BYTE-IDENTICAL |
| training | 53 of 400, 132 s, composites learned | 53 of 400, 153 s | BYTE-IDENTICAL |

CORRECTION the doc should carry: `verdict.md` cites lib=on as costing 115 s MORE. At this commit it
did not -- the two arms were within noise, and training-on was the faster of the pair. Wall time on
this machine had other work in flight and is not a clean instrument here. The P2 case does not rest
on it and is unaffected: the library earned **0 of 0** solves on both splits.

## 8. docs/INVENTION.md -- section 3.4 item 2 is measured FALSE

The design states the induced conjunction subsumes the authored `_RECOLOR_UNITS` object enumeration
(and `_fit_colormap`), so both can be deleted after porting. For `_RECOLOR_UNITS` this is FALSE and
was caught causally on evaluation task `ae58858e` -- see `designb/verdict-b.md`. The six authored
object tuples are RESTORED and must stay. `_fit_colormap` was consequently not retired either: it is
a whole-grid cellwise closer, not a unit write, and nothing in Design B ports it.

## 9. docs/coverage/*.md -- deliberately NOT touched

The coordinator's ledger work is in flight there. Design B changes no faculty coverage citation: the
ARC-AGI-1 evaluation figure is **unchanged at 18 of 400 = 4.50%**, so the Gf evidence line in item 1
above still reads true as written.

## 10. Gates not run by me

Full `run_selfchecks.py` and full `pytest` were NOT run (other work in flight, per instruction).
Targeted greens on my scope: `tests/test_arc_fix.py` + `tests/test_arc_residual.py` +
`tests/test_arc_diagnostic.py` 28 of 28; `arc_solver` / `arc_mint` / `arc_residual` self-checks OK;
`ruff check` clean on `src/snn_data_lake/reasoning`, `tools`, `tests`.

citation_guard: `designb/verdict-b.md` cites `tools/arc_designb.py` and `tools/arc_probe_ids.py` --
both are new files and must be part of the feature commit.

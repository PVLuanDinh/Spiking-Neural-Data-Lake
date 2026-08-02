# Quickstart: validating Fixed-Point Gf

All commands from the repo root with the project venv:
`.venv-lake\Scripts\python.exe` (NEVER the `python` on PATH). ARC data must be present at
`data/arc/{training,evaluation}` (instruments SKIP with a message if absent -- a SKIP is not a
figure).

## Prerequisites

- Clean tree for the touched files (the baseline is invalid otherwise):
  `git diff --quiet HEAD -- src/snn_data_lake/reasoning/arc_solver.py src/snn_data_lake/reasoning/arc_mint.py src/snn_data_lake/reasoning/arc_scene.py src/snn_data_lake/reasoning/reasoning_cortex.py src/snn_data_lake/reasoning/arc_bench.py tools/arc_score.py`
- The three gates green:
  `.venv-lake\Scripts\python.exe run_selfchecks.py`
  `.venv-lake\Scripts\python.exe -m pytest`
  `.venv-lake\Scripts\python.exe -m ruff check .` and
  `.venv-lake\Scripts\python.exe -m ruff format --check src tests tools`

## Scenario 0 -- the baseline exists and predates everything (FR-007, SC-001)

```
type specs\006-fixed-point-gf\baseline.txt
```

Expected: a commit hash, gate status, and four `arc_score` figures with solved-ID lists, all
recorded before any implementation commit (verify: the baseline commit is an ancestor of, or equal
to, the first implementation commit's parent).

## Scenario 1 -- variable depth on synthetic ladder tasks (US1, SC-002)

```
.venv-lake\Scripts\python.exe -m pytest tests/test_arc_fix.py -k ladder -q
```

Expected: one fix program solves the drop-until-rest ladder at heights 3, 7, 19; recorded depths
differ across the three (the test asserts at least two distinct depths); with the combinator
disabled the same tasks are NOT solved.

## Scenario 2 -- the admission controls fail as designed (FR-008, SC-003)

```
.venv-lake\Scripts\python.exe -m pytest tests/test_arc_fix.py -k "admission" -q
.venv-lake\Scripts\python.exe -m snn_data_lake.reasoning.arc_solver
```

Expected: A1 echo ~0 and A2 identity-DSL 0 of N still asserted in the self-checks; A3
fix(identity) converged at depth 1 with 0 added solves; A4 cycle; A5 diverged. Every assert is on
the FAILING outcome -- if any of these five "controls" passes where it must fail, the run is red.

## Scenario 3 -- trajectory totality on the real split (SC-002)

```
.venv-lake\Scripts\python.exe tools\arc_score.py evaluation off
```

Expected: 400 of 400 tasks complete; the run prints the depth census (outcome counts `N of M` per
primitive class, converged-depth distribution); zero unclassified trajectories; wall time within
2x of the baseline figure.

## Scenario 4 -- the typed bottleneck (US2, SC-006)

```
.venv-lake\Scripts\python.exe -m pytest tests/test_arc_fix.py -k "record" -q
.venv-lake\Scripts\python.exe -m snn_data_lake.reasoning.arc_mint
```

Expected: `mint_records()` output reproduces `fn` behaviour from declarative fields on train
pairs; memo-flagged records refuse-on-unseen under composition (None propagates, never a guess);
the mint self-check scores its dev set THROUGH `make_solver()` (no direct application), keeping
its honest-null framing; grep evidence that exactly 1 of 1 end-to-end ARC solve paths remains
(the searcher).

## Scenario 5 -- determinism with fix programs (FR-012, SC-007)

```
.venv-lake\Scripts\python.exe -m snn_data_lake.reasoning.arc_solver
```

Expected: the existing two-run determinism check (identical solved sets) passes with `fix:` and
`mint:` entries in the searched space, including a library composite containing a fix step.

## Scenario 6 -- the A/B and the control arm (US3, SC-004, SC-005)

```
.venv-lake\Scripts\python.exe tools\arc_score.py evaluation on
.venv-lake\Scripts\python.exe tools\arc_score.py evaluation off
```

plus the fixed-K arm per contracts/fix_combinator.md C18 (K computed from census.md, never typed).

Expected: `verdict.md` contains the three-arm table (`N of 400` each), solved-ID diffs with every
lost ID accounted, and either (a) adoption with the P2 disposition and the retired path named, or
(b) the null sentence -- "null result; feature stops here" -- mirrored into
`docs/DEPTH_CEILINGS.md`. Both are successful outcomes of this scenario.

## Scenario 7 -- the live surface tells the truth (US4)

```
.venv-lake\Scripts\python.exe -c "from snn_data_lake.reasoning.reasoning_cortex import build_default_cortex; c = build_default_cortex(); print(c.answer('arc: 12|34 -> 12|34 ; test 56|78'))"
```

Expected: on a gap, the message describes the surviving mechanism (no "(<=3 steps)" claim if depth
is variable; unchanged wording if the feature was nulled and retired).

## Scenario 8 -- the gates, last (FR-013)

```
.venv-lake\Scripts\python.exe run_selfchecks.py
.venv-lake\Scripts\python.exe -m pytest
.venv-lake\Scripts\python.exe -m ruff check .
.venv-lake\Scripts\python.exe -m ruff format --check src tests tools
```

Expected: all green at every phase boundary, not only at the end.

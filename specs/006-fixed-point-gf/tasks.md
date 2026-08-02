---

description: "Task list for Fixed-Point Gf (Variable-Depth Computation + Typed Bottleneck)"
---

# Tasks: Fixed-Point Gf (Variable-Depth Computation + Typed Bottleneck)

**Input**: Design documents from `specs/006-fixed-point-gf/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/fix_combinator.md,
quickstart.md

**Baseline commit**: to be recorded by T001 (spec authored at `30d8e76a`; HEAD may have moved --
`specs/005-*` lands concurrently. The measured-at-capture figure is the baseline, never a quoted
one). Verify the tree is clean for every touched file before capturing anything -- specs/003
learned this the hard way (an uncommitted rewrite was measured and mislabelled as the baseline).

**Interpreter**: every command uses `.venv-lake\Scripts\python.exe`, never the `python` on PATH.

**Tests**: INCLUDED. Not optional -- the constitution binds three gates (`run_selfchecks.py`,
`pytest` >=70% coverage, `ruff check .` + `ruff format --check src tests tools`), and Standing
Invariant 5 makes the five admission controls (contract section 5) admission gates, not niceties.

**Organization**: grouped by user story. **Phase order deviates from spec priority order in one
place**: US3's baseline half executes FIRST (Phase 1), because no implementation commit may
precede `baseline.txt` (FR-007); and the trajectory census (T015) gates the merge work, because
Standing Invariant 1 requires a gate's output counted before anything leans on it. Story labels
still map to the spec.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1 combinator, US2 typed bottleneck/merge, US3 measurement/verdict, US4 surfaces
- Exact file paths in every description

## Path Conventions

Single project. Source at `src/snn_data_lake/reasoning/`, instruments at `tools/`, tests at
`tests/`, records in `specs/006-fixed-point-gf/`.

---

## Phase 1: Setup -- the baseline, before anything changes (US3 first half, FR-007, SC-001)

**Purpose**: every later claim needs a control captured at a named commit, BEFORE implementation.

- [ ] T001 Verify clean tree and name the commit: run `git diff --quiet HEAD -- src/snn_data_lake/reasoning/arc_solver.py src/snn_data_lake/reasoning/arc_mint.py src/snn_data_lake/reasoning/arc_scene.py src/snn_data_lake/reasoning/reasoning_cortex.py src/snn_data_lake/reasoning/arc_bench.py tools/arc_score.py` and ABORT if dirty; record `git rev-parse HEAD` as the first line of `specs/006-fixed-point-gf/baseline.txt`
- [ ] T002 [P] Record three-gate status at baseline into `specs/006-fixed-point-gf/baseline.txt`: `.venv-lake\Scripts\python.exe run_selfchecks.py`, `.venv-lake\Scripts\python.exe -m pytest`, `.venv-lake\Scripts\python.exe -m ruff check .`, `.venv-lake\Scripts\python.exe -m ruff format --check src tests tools` -- pass/fail each
- [ ] T003 Capture the four arbiter figures into `specs/006-fixed-point-gf/baseline.txt`: `.venv-lake\Scripts\python.exe tools\arc_score.py training on`, `training off`, `evaluation on`, `evaluation off` -- each with its solved-ID list and wall time, every count transcribed as `N of M` (expected near 40 of 400 training / 11 of 400 evaluation; whatever is measured is the baseline). If `data/arc` is absent the tool SKIPs -- a SKIP is not a baseline; download first (URL is printed by the tool)

**Checkpoint**: `baseline.txt` exists, names a commit, and predates every implementation commit.

---

## Phase 2: Foundational -- the trajectory runner (blocks all stories)

**Purpose**: the loop-with-classifier every story consumes (contract section 1).

- [ ] T004 Implement the trajectory runner in `src/snn_data_lake/reasoning/arc_solver.py`: `_fix_run(f, g)` -> `(outcome, depth, result)` per contract C1-C5 -- exact-equality convergence, revisit-set cycle detection, `_ok`-failure divergence, charge every application to the same node counter `synthesize()` uses so the standing `node_budget`/`time_budget` are the ONLY caps; exceptions inside `f` are `diverged`; NO max-iteration constant anywhere (grep-able guarantee: no new integer literal in the halting logic)
- [ ] T005 [P] Create `tests/test_arc_fix.py` with trajectory-classifier unit tests: identity converges at depth 1; a reflect-like step on an asymmetric grid classifies `cycle` with `result is None`; a tile-like grower classifies `diverged`; a gravity-like contraction converges with `depth > 1`; a tiny node budget yields `budget`; same `(f, g)` twice gives identical tuples (purity)

**Checkpoint**: `_fix_run` exists, unit-tested, no story work started.

---

## Phase 3: User Story 1 -- fix in the search, depth becomes task-dependent (P1)

**Goal**: `synthesize()` can search fix programs; depth varies with the task; controls can go red.

**Independent test**: quickstart Scenarios 1-3 (ladder tasks, admission controls, totality).

- [ ] T006 [US1] Wrap the pool in `src/snn_data_lake/reasoning/arc_solver.py::_pool`: for each pooled entry `(name, fn)` add `("fix:" + name, ...)` per contract C6-C8 -- every primitive eligible (Q2 assumption), `fix:` entry ordered directly after its base (MDL, C8), `dsl="identity"` gains NO fix entries (C9)
- [ ] T007 [US1] Add depth telemetry per contract C14-C15 in `src/snn_data_lake/reasoning/arc_solver.py`: `synthesize()` accepts an optional stats collector (default None, zero cost unused) recording per-trajectory (base name, outcome, depth); all prints `N of M`
- [ ] T008 [P] [US1] Add admission-control tests A3-A5 to `tests/test_arc_fix.py`, each asserting the FAILING outcome (contract section 5): `fix(identity)` converged depth 1 and adds 0 solves on the self-check task set; `fix(reflect)` cycle; `fix(tile)` diverged; plus re-assert A2 (`dsl="identity"` scores 0 with the combinator present)
- [ ] T009 [P] [US1] Add the synthetic ladder test to `tests/test_arc_fix.py`: three drop-until-rest tasks (heights 3, 7, 19) solvable by one `fix:` program; assert all three solve WITH the combinator, at least two distinct converged depths across them, and NONE solve with the combinator absent (proving the solve came from iteration, not existing depth) -- spec US1/AS-1
- [ ] T010 [US1] Extend the `__main__` self-check of `src/snn_data_lake/reasoning/arc_solver.py`: admission controls A3-A5 asserted failing; determinism check now runs with `fix:` entries in the space AND a library composite containing a `fix:` step (two budget-bound runs, identical solved sets -- FR-012, edge case "composite replay")
- [ ] T011 [US1] Gate check: all three gates green (`run_selfchecks.py`, `pytest`, `ruff check` + `ruff format --check src tests tools`) -- record pass in the task log before proceeding

**Checkpoint**: US1 independently testable -- quickstart Scenarios 1, 2, 5 pass.

---

## Phase 4: The census -- count the gate's output before anything leans on it (US1/US3, Standing Invariant 1)

**Purpose**: publish convergence reality BEFORE merge or adoption reasoning (research.md F8/R1).

- [ ] T012 [US3] Run the evaluation split with the combinator ON at bench budgets (`.venv-lake\Scripts\python.exe tools\arc_score.py evaluation off` -- library off isolates the combinator) collecting telemetry via T007; write `specs/006-fixed-point-gf/census.md`: trajectory outcomes per primitive class and converged-depth distribution, every count `N of M`, wall time vs baseline (must be within 2x, else trigger the R4 structural-filter decision and record it)
- [ ] T013 [US3] DECISION POINT recorded in `specs/006-fixed-point-gf/census.md`: if no primitive family shows non-trivial convergence (all converged depths == 1), the variable-depth hypothesis is DEAD ON CENSUS -- skip Phases 5 is NOT allowed (the merge is an independent P2 payment), but mark the Phase 6 A/B as expected-null and continue; otherwise record the mean winning-fix depth as the SOURCE for the control arm's K

**Checkpoint**: census published; adoption reasoning may now cite it.

---

## Phase 5: User Story 2 -- the typed bottleneck feeds the one searcher (P2)

**Goal**: minted primitives enter `_pool()` as typed records; mint stops solving end-to-end.

**Independent test**: quickstart Scenario 4.

- [ ] T014 [US2] Implement `mint_records(train)` in `src/snn_data_lake/reasoning/arc_mint.py` per contract C10 and data-model.md "Typed primitive record": emits ONLY LOO-validated records (name, unit, attribute, rule_form, rule_params, consumer, memo_flag, fn), deterministic MDL-then-name order, ASCII names; no grid or train-pair residue in any field
- [ ] T015 [P] [US2] Add the record-reconstruction test to `tests/test_arc_fix.py`: for every record emitted on the mint's dev tasks, the declarative fields (unit, attribute, rule_form, rule_params, consumer) reproduce `fn`'s output on every train pair (the bottleneck is auditable, C10)
- [ ] T016 [US2] Consume records in `src/snn_data_lake/reasoning/arc_solver.py::_pool` per C11: append `("mint:" + rec.name, rec.fn)` after fitted primitives; `fix:` wrapping applies to minted entries like any other (C6)
- [ ] T017 [P] [US2] Add memo-flag propagation tests to `tests/test_arc_fix.py` per C12: a memo-flagged record on an unseen attribute value returns None; the None propagates under composition and under `fix:` wrapping (no candidate, never a guess); the memo flag appears in the program description string
- [ ] T018 [US2] Port-before-retire inventory tests in `tests/test_arc_fix.py` (P2 guard, research.md R6): LOO validation still rejects a train-only lookup; MDL order still prefers functional forms over rank/lookup; memo-flagging still marks rank/lookup -- all asserted on the SURVIVING path (records scored through `make_solver`)
- [ ] T019 [US2] Rewire `src/snn_data_lake/reasoning/arc_mint.py::main` as the PRODUCER self-check: emit records, score the dev set THROUGH `make_solver()` (no direct program application in the harness), keep the honest-null framing of the current docstring, print counts `N of M`. Do NOT delete `mint()`'s direct-application internals yet -- deletion is T026, after the A/B (P2: A/B before deleting)
- [ ] T020 [US2] Gate check: all three gates green; `run_selfchecks.py` includes the rewired `reasoning.arc_mint` self-check

**Checkpoint**: US2 independently testable -- quickstart Scenario 4 passes; two solve paths still
exist in the tree (deletion gated on Phase 6).

---

## Phase 6: User Story 3 -- A/B, control arm, verdict (P3)

**Goal**: adopt on evidence or record a null and stop. Either outcome completes this story.

**Independent test**: quickstart Scenario 6; `verdict.md` exists with the three-arm table.

- [ ] T021 [US3] Create the fixed-K control instrument `tools/arc_fixdepth_control.py` (tracked -- a document may not cite an untracked script, per the citation_guard rule): same pool, `fix:` entries replaced by exactly-K-applications entries (no convergence check, take iterate K), K computed by READING `specs/006-fixed-point-gf/census.md`'s mean winning-fix depth and rounding to nearest int (C18 -- K is derived, never typed); explicit `--data`, SKIP-not-zero on missing data; usage line in the docstring
- [ ] T022 [US3] Run the record A/B at identical bench budgets and append to `specs/006-fixed-point-gf/verdict.md`: baseline (from baseline.txt) vs adaptive (`tools\arc_score.py evaluation on` and `off`, training both ways) vs fixed-K (`tools\arc_fixdepth_control.py evaluation`); solved-ID sets diffed -- gained IDs listed, lost IDs listed
- [ ] T023 [US3] Account every lost solved-ID individually in `specs/006-fixed-point-gf/verdict.md` (Q3 assumption, Standing Invariant 3): each lost ID gets a per-task explanation (e.g. node starvation by fix trajectories) or the adoption is REFUSED; if starvation is the cause, decide and record the R4 structural filter disposition instead of raising any budget
- [ ] T024 [US3] Write the verdict in `specs/006-fixed-point-gf/verdict.md` per C19: ADOPT only if adaptive strictly exceeds baseline with losses accounted; the word "adaptive" claimed only if adaptive strictly exceeds fixed-K (SC-005); otherwise the null sentence -- "null result; feature stops here" -- and the P2 disposition either way (what is retired, what survives). A null verdict still completes T025-T029; it changes only their content

**Checkpoint**: verdict recorded; every number in it traces to baseline.txt, census.md, or a
command in quickstart.md.

---

## Phase 7: User Story 4 + P2 disposition -- the surfaces tell the truth (P4)

**Goal**: live message, docs, and the one-path census match the verdict.

**Independent test**: quickstart Scenario 7; grep shows 1 of 1 end-to-end solve paths.

- [ ] T025 [US4] Update `ArcExpert.answer`'s honest-gap message in `src/snn_data_lake/reasoning/reasoning_cortex.py` to the surviving truth: on adoption, describe variable depth via fixed-point (drop the hardcoded "(<=3 steps)"); on null, keep the wording accurate to the retained mechanism (FR-017)
- [ ] T026 [US2] Execute the P2 retirement per the verdict in `src/snn_data_lake/reasoning/arc_mint.py`: on no-regression, delete the direct end-to-end application path (mint -> grid on test input outside the searcher) so `mint_records` is the only consumer-facing surface; verify with a grep census recorded in `specs/006-fixed-point-gf/verdict.md` that exactly 1 of 1 end-to-end ARC solve paths remains (SC-006); on a regression verdict, record REFUSED and leave both paths with the refusal reason (verify-first works when it says stop)
- [ ] T027 [P] [US4] Add the Ceiling-1 addendum to `docs/DEPTH_CEILINGS.md`: the three-arm table, the census summary, the verdict (null reported as null, same framing as the existing library-learning null), and a reproduce block naming `tools/arc_score.py` and `tools/arc_fixdepth_control.py`
- [ ] T028 [P] [US4] Update the Gf faculty's evidence line in `docs/coverage/` ONLY if the verdict changed it (held-out evidence band rules apply; a null changes nothing and the ledger stays put)

---

## Phase 8: Polish & cross-cutting

- [ ] T029 Sweep every print touched by this feature for `N of M` formatting and ASCII-only output in `src/snn_data_lake/reasoning/arc_solver.py`, `arc_mint.py`, `reasoning_cortex.py`, `tools/arc_fixdepth_control.py` (FR-011, FR-012)
- [ ] T030 Final gates: `.venv-lake\Scripts\python.exe run_selfchecks.py`, `.venv-lake\Scripts\python.exe -m pytest` (>=70% coverage), `.venv-lake\Scripts\python.exe -m ruff check .`, `.venv-lake\Scripts\python.exe -m ruff format --check src tests tools` -- all green
- [ ] T031 Refresh the knowledge graph per project rule: `graphify update .` from the repo root

---

## Dependencies

```text
Phase 1 (T001-T003)  ->  everything (no implementation before baseline.txt)
Phase 2 (T004-T005)  ->  Phases 3-7
Phase 3 (T006-T011)  ->  Phase 4 (census needs the combinator + telemetry)
Phase 4 (T012-T013)  ->  Phase 6 (K derives from census; Standing Invariant 1)
Phase 5 (T014-T020)  ->  Phase 6 (the A/B measures the merged pool)
Phase 6 (T021-T024)  ->  Phase 7 (T025-T028 content depends on the verdict; T026 gated on it)
Phase 7              ->  Phase 8
```

Story completion order: US1 -> (census) -> US2 -> US3 -> US4. US1 and US2 are independently
TESTABLE but US2's pool intake (T016) edits the same `_pool()` as T006 -- do them sequentially,
never in parallel branches of the same file.

## Parallel examples

- Phase 2: T005 (tests) alongside T004 review.
- Phase 3: T008 and T009 (same new test file, different test functions -- parallel only if merged
  carefully; otherwise sequential) while T006/T007 land in `arc_solver.py` sequentially.
- Phase 5: T015 and T017 in parallel after T014; T016 sequential in `arc_solver.py`.
- Phase 7: T027 and T028 in parallel (different docs).

## Implementation strategy

MVP = Phase 1 + Phase 2 + Phase 3 (US1): baseline, combinator, controls, ladder proof. That alone
answers "can this architecture iterate, honestly?" and is independently valuable even if every
later phase nulls. The census (Phase 4) is the cheap kill-switch before the merge investment.
US2 pays the standing P2 debt regardless of the A/B outcome; US3 decides adoption; US4 keeps the
surfaces honest under either verdict.

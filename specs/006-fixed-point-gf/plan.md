# Implementation Plan: Fixed-Point Gf (Variable-Depth Computation + Typed Bottleneck)

**Branch**: none (specs authored directly on the working branch; feature directory is the unit here)
| **Date**: 2026-08-02 | **Spec**: `specs/006-fixed-point-gf/spec.md`

**Input**: Feature specification from `specs/006-fixed-point-gf/spec.md`

**Note**: `.specify/feature.json` was deliberately NOT repointed at this directory -- it is shared
state and `specs/005-*` is being authored concurrently. Every speckit phase for this feature is
invoked with the directory named explicitly.

## Summary

Give the Gf faculty's ARC solve path variable-depth computation: a fix-point combinator in
`arc_solver.synthesize()`'s program grammar that applies one pooled primitive repeatedly until the
grid stops changing (exact equality -- no tolerance constant), classifying every trajectory as
converged / cycle / diverged / budget. Simultaneously close the recorded P2 debt between the two
ARC solve paths: `arc_mint`'s verified typed primitives enter the searcher's one pool through the
typed record (unit, attribute, rule, consumer, memo-flag) -- the Hourglass bottleneck, already
mostly built in the mint -- and the mint's direct end-to-end application path is retired. Adoption
is decided by `tools/arc_score.py` on the held-out evaluation split against a pre-implementation
baseline, with a fixed-depth control arm that separates "adaptivity earned it" from "more compute
earned it", and an explicit contract that a null result records and stops the feature.

## Technical Context

**Language/Version**: Python (the repo's `.venv-lake`; run everything with
`.venv-lake\Scripts\python.exe`, never the `python` on PATH)

**Primary Dependencies**: stdlib only for everything this feature touches (`arc_solver.py`,
`arc_mint.py`, `arc_bench.py` are stdlib-pure by design and stay that way). No new dependencies.

> **P7 VOID (annotated 2026-08-04).** The sentence above certifies "no new dependency" as a constitution check. **That is not a criterion under P7** (`docs/POLICIES.md`), which states that *"it would be another dependency" is NOT a valid reason to reject anything* and requires a rejection to cite a measurement on the real workload or a structural argument. No library was evaluated here, so the check scored nothing. Left in place rather than deleted because this file records a decision that was made; annotated because P7 names these certifications as live residue of the zero-dependency rule abolished 2026-07-20. `specs/010/plan.md` is the first plan to decline the certification on purpose.

**Storage**: none. Fix trajectories and typed primitive records are task-local and ephemeral;
nothing new is persisted to the one lake. Spec-directory artifacts (`baseline.txt`, census and
verdict records) are plain text files in `specs/006-fixed-point-gf/`.

**Testing**: the three standing gates -- `python run_selfchecks.py` (module `__main__` self-checks;
`reasoning.arc_solver`, `reasoning.arc_scene`, `reasoning.arc_mint`, `reasoning.arc_bench`,
`reasoning.reasoning_cortex` are all registered), `pytest` (>=70% coverage; new unit tests in
`tests/test_arc_fix.py`), `ruff check .` + `ruff format --check src tests tools`.

**Target Platform**: Windows dev box + the CI runner (no GPU on CI). CPU only (P5: symbolic search
over <=30x30 grids; the measured CUDA crossover N~24k-32k is irrelevant here).

**Project Type**: single project; changes confined to `src/snn_data_lake/reasoning/`, `tests/`,
docs, and this spec directory. `tools/arc_score.py` semantics are frozen (it is the arbiter).

**Performance Goals**: the full evaluation split at bench budgets (node_budget=4000,
time_budget=5.0 per task) must remain a sit-through-able run (baseline wall time is recorded in
`baseline.txt`; the adaptive run must stay within 2x of it, else the R4 structural filter is
triggered). No live-tick impact: `ArcExpert` is lazy and only runs on `arc:` queries.

**Constraints**: determinism (no RNG in the combinator or intake; two identical runs solve the
identical set); ASCII-only prints; counts printed `N of M`; no new numeric constant that
arbitrates behaviour (halting is exact-equality + revisit + existing resource guards -- see
research.md R4); no `loss.backward()` anywhere.

**Scale/Scope**: 800 ARC tasks (400 + 400); primitive pool ~40 base + fitted + library + minted
entries, roughly doubled by fix-wrapping; 3 source files touched, 1 test file added, 2 docs
updated.

## Constitution Check

*GATE: evaluated against `.specify/memory/constitution.md` v1.0.0 (defers to `docs/POLICIES.md`).*

| Principle | Verdict | How this plan satisfies it |
|---|---|---|
| P0 self-improvement | PASS | A new capability avenue (adaptive depth) pursued with its honesty guards: measured or it is fake; a null is recorded and stops the feature (FR-010). |
| P1 no-hardcode | PASS | Halting introduces NO numeric constant: exact equality + revisit detection; iteration cost counted against the EXISTING node/time budgets, whose "resource guards only" disposition is standing (`arc_solver.py:25`). The fixed-K control constant is DERIVED from run telemetry, not authored. The feature also RETIRES a P1 debt: depth stops being the typed-in `max_depth`-only story. |
| P2 non-additive | PASS (this feature is largely a P2 payment) | Two solve paths become one: mint -> typed record -> `_pool()`; direct-application path retired AFTER the A/B, with the port-before-retire inventory itemized (research.md R6). No flag-parked survivors (FR-010). |
| P3 no dormancy | PASS | Cures a live instance: `arc_mint` computes programs no live path consumes (spec F7). Post-merge, mint output is consumed by the live searcher. The census/telemetry is printed, not stored-and-ignored. |
| P4 one brain | PASS | No brain construction anywhere in this feature; `ArcExpert` remains the single live wire; nothing bypasses `make_brain()`. |
| P5 GPU where measured | PASS | CPU only; populations/grids orders of magnitude below the measured crossover; suite stays runnable without a GPU. |
| P6 mutual reach | PASS (no regression) | Gf's reach is unchanged structurally; the merge INCREASES internal reach (mint -> searcher). No new isolated build. |
| e-prop over backprop | PASS | Symbolic feature; no gradients introduced; e-prop paths untouched (FR-015). |
| Standing Invariant 1 (count the gate's output first) | PASS | The trajectory census (Phase 3) runs and publishes BEFORE the merge or any adoption decision leans on convergence behaviour. |
| Standing Invariant 2 (never silence a check) | PASS | arc_score/arc_bench semantics frozen; controls are added beside them, never relaxed. |
| Standing Invariant 3 (negative verdict is dangerous) | PASS | Lost solved-IDs individually accounted or adoption refused (Q3 assumption, FR-009). |
| Standing Invariant 5 (verify the verifier) | PASS | Five admission controls asserted on their FAILING outcome (FR-008). |
| Three gates | PASS | Green before and after every phase (FR-013). |

**Post-Phase-1 re-check (design artifacts)**: contracts introduce no new constants, no persistence,
no second path; the fix-eligibility filter, if ever triggered, is structural not nominal
(FR-004). No violations to justify -- Complexity Tracking left empty.

## Project Structure

### Documentation (this feature)

```text
specs/006-fixed-point-gf/
|-- spec.md              # /speckit-specify output
|-- plan.md              # this file
|-- research.md          # Phase 0 output
|-- data-model.md        # Phase 1 output
|-- quickstart.md        # Phase 1 output
|-- contracts/
|   `-- fix_combinator.md    # trajectory + typed-record + census contracts
|-- checklists/
|   `-- requirements.md
|-- baseline.txt         # created by T001-T003 (BEFORE implementation)
|-- census.md            # created by Phase 3 (trajectory census, published)
|-- verdict.md           # created by Phase 6 (A/B + control arm + P2 disposition)
`-- tasks.md             # /speckit-tasks output
```

### Source Code (repository root)

```text
src/snn_data_lake/reasoning/
|-- arc_solver.py        # fix combinator in the pool; minted-record intake in _pool();
|                        #   depth telemetry in synthesize(); self-check extensions
|-- arc_mint.py          # export the typed record (mint_records()); retire direct application;
|                        #   main() becomes a producer self-check scored through the searcher
|-- reasoning_cortex.py  # ArcExpert gap message updated to the surviving truth
`-- arc_bench.py         # UNCHANGED (arbiter; admission controls already live here)

tools/
|-- arc_score.py         # UNCHANGED (arbiter)
`-- arc_baseline.py      # UNCHANGED

tests/
`-- test_arc_fix.py      # NEW: trajectory classification, admission controls, typed-record
                         #   intake, determinism with fix programs, N-of-M formatting

docs/
|-- DEPTH_CEILINGS.md    # Ceiling-1 addendum: this feature's measurement, whatever the verdict
`-- coverage/            # Gf faculty evidence line updated only if the verdict changes it
```

**Structure Decision**: single project, in-place extension of the three reasoning modules. No new
module is created for the combinator -- it is a grammar form of the existing searcher, and a new
module would be additive surface area (P2 spirit). The only new file is the test file.

## Execution Order

Evidence validity dictates order (research.md R7); story labels map to spec.md.

1. **Phase 1 -- Baseline (US3's first half).** Clean-tree check, capture `baseline.txt` at a named
   commit: both splits, library on and off, solved-IDs, wall time, gate status. NOTHING else may
   land first.
2. **Phase 2 -- Foundational: the combinator (US1).** `fix` trajectory runner + classifier in
   `arc_solver.py`; pool wrapping; node accounting; depth telemetry; unit tests; the five
   admission controls asserted failing.
3. **Phase 3 -- The census (US1/US3, Standing Invariant 1).** Run the evaluation split with the
   combinator ON; publish `census.md` (trajectory outcomes and converged-depth distribution per
   primitive class) BEFORE any adoption reasoning. If no non-trivial convergence exists: record,
   jump to Phase 6 with a null verdict for the combinator arm.
4. **Phase 4 -- The typed-record intake (US2).** `arc_mint.mint_records()` (typed records out,
   nothing else); `_pool()` consumes them as `mint:*` primitives; memo-flag None-propagation test;
   mint `main()` rewired as producer self-check through the searcher; port-before-retire
   inventory verified; direct-application path retired.
5. **Phase 5 -- A/B + control arm (US3).** Adaptive run vs baseline vs fixed-K control (K derived
   from Phase 3/5 telemetry), solved-ID diffs, lost-ID accounting. Determinism check with fix
   programs.
6. **Phase 6 -- Verdict + surfaces (US3/US4).** `verdict.md` (adopted or null, P2 disposition);
   `docs/DEPTH_CEILINGS.md` addendum; `ArcExpert` message updated to the surviving truth; coverage
   ledger only if evidence line changed; all three gates green.

## Risks

- **Empty census** (no primitive family converges beyond depth 1): early null; that is the
  experiment working, and Phase 3's placement caps the cost before the merge work.
- **Node starvation / lost solves**: fix entries double the pool and trajectories eat nodes; the
  A/B's lost-ID accounting detects it; disposition is the structural filter (research.md R4) or
  refusal -- never a raised budget to force a win.
- **Concurrent `specs/005-*`**: if its implementation touches `reasoning/arc_*.py` before Phase 1
  here, re-baseline; if after Phase 1, rebase and re-baseline (the baseline is whatever HEAD
  measures, never a quoted figure).
- **Composite replay with fix steps**: library composites containing `fix:*` must replay
  deterministically; covered by an explicit test; semantic note (depth belongs to the task) in the
  contract.
- **Verdict pressure**: the failure mode this repo documents is green lights unable to go red.
  Mitigation is structural: controls asserted on failing outcomes, census published before
  adoption reasoning, null contractually terminal (FR-010).

## Complexity Tracking

> No Constitution Check violations -- table intentionally empty.

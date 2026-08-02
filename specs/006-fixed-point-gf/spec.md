# Feature Specification: Fixed-Point Gf (Variable-Depth Computation + Typed Bottleneck)

**Feature Directory**: `specs/006-fixed-point-gf`

**Created**: 2026-08-02

**Source**: Owner directive, 2026-08-02. Governed by `docs/POLICIES.md` (P0-P6), the authoritative
policy text; `.specify/memory/constitution.md` is its governance surface and defers to it by its own
rule. The Standing Invariants (`constitution.md`, "Standing Invariants") apply throughout.
Research basis: arXiv 2606.18206 (Fixed-Point Reasoners, "FPRM") and arXiv 2607.11696 (Hourglass
Reasoning) -- see `research.md` for what is portable and, just as importantly, what is not.

**Status**: Draft -- ready for `/speckit-plan`. Open questions Q1-Q4 recorded below with stated
assumptions; none blocks planning.

**Input**: User description: "Variable-depth computation via fixed-point convergence for the Gf
faculty's ARC solving path, plus a typed bottleneck between reasoning stages."

## Problem

The project's honest fluid-intelligence number is **11 of 400 = 2.75% (~3%)** on the held-out
ARC-AGI-1 evaluation split (`docs/DEPTH_CEILINGS.md`, "Ceiling 1"), measured by
`tools/arc_score.py` -> `reasoning/arc_bench.py` (pass@2, exact grid match). Earlier work moved that
number from 0.0%, and the cause of the remaining gap was proven **architectural**, not a tuning
matter. This feature tests the next architectural hypothesis: **compute depth is fixed and
open-loop**, and tasks whose natural solution is a procedure ("apply a step until nothing changes")
are structurally out of reach.

Two defects, precisely located:

1. **Every task gets the same depth.** `arc_solver.synthesize()` searches compositions of at most
   `max_depth=3` primitives, each applied exactly once (`arc_solver.py:770-859`). A task whose
   solution is "let objects fall until they rest", "grow the pattern until the border", or "propagate
   a colour until blocked" needs the SAME step applied a task-dependent number of times. Today that
   is expressible only if a developer hand-authors the whole iterated behaviour as one primitive
   (`_gravity` and `_fill_enclosed` are exactly this -- depth hidden inside hand code, which is the
   developer-supplied prior Chollet's developer-aware clause charges against the score). The FPRM
   result (arXiv 2606.18206) is the external evidence this matters: on ARC-AGI and other procedural
   benchmarks, the number of effective layers reached by looping determines solution quality, and
   fixed-point convergence is a working end-to-end halting mechanism that lets compute adapt to task
   difficulty.

2. **The typed bottleneck exists but its products never reach the searcher.** `arc_mint.py` already
   implements this project's analogue of Hourglass's (schema, rule) stage boundary: typed attribute
   readers over `arc_scene` measurements -> `induce_rule()` (MDL-ordered functional forms, validated
   by leave-one-out, memorisation-class forms flagged) -> typed consumers. But the mint applies its
   programs DIRECTLY, in its own harness, and `reasoning_cortex.py` never references it: the live
   path (`ArcExpert` -> `arc_solver.synthesize`) and the mint are **two additive solve paths** --
   the exact shape P2 forbids, and a debt `arc_mint.py`'s own docstring records: *"Composing minted
   primitives INTO arc_solver.synthesize() ... are the NEXT Phase-3 steps -- noted, not done here."*

This feature closes both at once, because they compose: the fix-point combinator gives the searcher
variable depth, and the typed bottleneck is the gate through which minted step-primitives (the
things worth iterating) enter the one searcher.

**What this feature is not.** Every other current ARC-AGI-2/3 leaderboard approach surveyed
(Tycho 2607.28287, ARCANA 2607.09059, PoTRE 2607.20268, PRO-LONG 2607.20064) is a harness around a
frozen frontier LLM. None is portable here: importing an LLM violates the brain-native contract and
teaches the brain nothing. Recorded in full in `research.md` so nobody re-proposes them.

## Findings already in hand (verified against source at `30d8e76a`, 2026-08-02)

**F1 -- Depth is a typed-in constant at every entry point.** `synthesize()` defaults
`max_depth=3, node_budget=4000, time_budget=None` (`arc_solver.py:770-778`); `make_solver()` sets
`time_budget=5.0` (`arc_solver.py:862`); the live `ArcExpert.answer()` calls with
`node_budget=1500, time_budget=2.0` and its honest-gap message hardcodes the text `"(<=3 steps)"`
(`reasoning_cortex.py`, class `ArcExpert`). P1 note: the node/time budgets are RESOURCE GUARDS
(the module says so: "a hard node budget and an optional wall-time budget (resource guards only)",
`arc_solver.py:25`) and stay; `max_depth=3` is the value this feature replaces with a
convergence-derived depth.

**F2 -- No iteration exists in the search; some primitives iterate internally.** Each primitive in a
composition is applied exactly once (`arc_solver.py:824-852`). `_gravity` settles objects in an
internal loop; `_fill_enclosed` floods internally. Iteration exists only where a developer authored
it into a primitive body -- the architecture itself cannot iterate.

**F3 -- The search's equivalence pruning makes "just search deeper" a dead end for iteration.**
`synthesize()` prunes any state already seen (`seen` set, `arc_solver.py:846-848`). A step whose
second application revisits an earlier state is pruned, and depth-k re-application of one primitive
costs k search levels. Iteration must be a COMBINATOR (one search step that runs the loop to its
fixed point), not deeper open-loop search. This is also the honest reading of FPRM: looping is an
inductive bias, not merely more layers.

**F4 -- Fixed-point halting is parameter-free in this domain.** ARC grids are finite discrete
states (at most 30x30 cells, 10 colours; `_MAXS = 30`, `arc_solver.py`). Convergence is EXACT
EQUALITY of successive grids -- no tolerance epsilon exists to tune. Non-convergence is detected
structurally: a revisited earlier state is a cycle; an invalid or oversize grid is divergence. The
P1 trap named in the directive (a tuned halting tolerance + a tuned max-iteration cap) therefore
does not arise: the halting rule needs NO numeric constant. The only caps are the already-standing
node/time resource guards, which the loop's applications are counted against.

**F5 -- The deciding benchmark exists and already has its controls.** `tools/arc_score.py`
(promoted 2026-07-30, explicit `--data`, SKIPs rather than printing a false 0/0) scores
`arc_bench.score()` on training/evaluation, library on/off, pass@2 exact match. `arc_bench`'s
self-check asserts an echo solver scores ~0 (the metric can go red), and `arc_solver`'s self-check
asserts the identity-only DSL scores 0 on real tasks and that two budget-bound runs solve the
IDENTICAL task set (determinism). `tools/arc_baseline.py` is the pre-build instrument for the
brain-side interfaces. The P2 arbiter for this feature is `tools/arc_score.py` on the evaluation
split.

**F6 -- The mint's typed bottleneck is real and partially sufficient.** `arc_mint.py` has typed
attribute readers (`_obj_attr`, `_region_attr`, unit builders), MDL-ordered rule forms
(const/mod-k/offset/affine before rank/lookup), leave-one-out validation, and memorisation-class
flagging (`_MEMO_KINDS`); a flagged rule returns None on an unseen attribute value -- no answer,
never a wrong guess. What crosses a mint stage boundary today is already only (unit type,
attribute, rule). The DELTA this feature owes is not building a bottleneck -- it is making that
bottleneck the interface through which minted primitives enter `synthesize()`'s one pool, and
retiring the mint's separate end-to-end application path.

**F7 -- arc_mint is dormant with respect to the live path.** `reasoning_cortex.py` contains no
reference to `arc_mint`; its importers are diagnostic/bench modules. Under P3's per-method rule the
mint computes programs no live path consumes. The merge in this feature is also the P3 cure.

**F8 -- The spiking/depth analogue of FPRM's signal-propagation problem is real here and must be
measured, not assumed away.** FPRM had to fix depth-induced signal degradation (pre-norm, residual
scaling) before looping paid. The symbolic analogue: many primitives NEVER converge under
iteration -- `_reflect`/rotations oscillate with period 2, `_tile` grows the grid until `_ok()`
rejects it, recolour rules can 2-cycle. A fix combinator must therefore classify every trajectory
as converged / cycle / diverged / budget-truncated, and the expected outcome is that only a MINORITY
of primitives are usefully iterable. The per-primitive trajectory census is a required deliverable
(SC-008), because "the loop exists but nothing converges" is this feature's most likely failure
mode and it must be visible, not silent.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A step can run to its own fixed point, and depth becomes task-dependent (Priority: P1)

The searcher gains ONE new composition form: fix(f) -- apply primitive f repeatedly until the grid
stops changing. The number of applications is decided by the task's own convergence, not by an
author. A gravity task on a tall grid iterates more times than on a short grid; the solver does not
know or care in advance.

**Why this priority**: This is the architectural hypothesis under test. Without it, US2's minted
step-primitives would still be applied a fixed number of times, and US3 would have nothing to
measure.

**Independent Test**: Construct synthetic tasks whose solution is "iterate step s until stable"
with different required depths (e.g. drop-by-one gravity on heights 3, 7, 19). One fix(s) program
must solve all of them, and the recorded iteration counts must differ across them.

**Acceptance Scenarios**:

1. **Given** a synthetic task solvable only by k>3 applications of one DSL primitive, **When**
   `synthesize()` runs with the fix combinator available, **Then** a fix program solves it, and the
   same task is NOT solved with the combinator disabled (proving the solve came from iteration,
   not from existing depth).
2. **Given** the identity primitive, **When** fix(identity) runs on any grid, **Then** it reports
   convergence after exactly 1 application (fixed point reached immediately) and contributes 0
   solves anywhere.
3. **Given** an oscillating primitive (reflect), **When** fix(reflect) runs, **Then** the trajectory
   is classified as a CYCLE and produces no candidate grid -- never the last iterate as a guess.
4. **Given** a growing primitive (tile), **When** fix(tile) runs, **Then** the trajectory is
   classified as DIVERGED at the first invalid/oversize grid and produces no candidate.
5. **Given** any evaluation task, **When** the solver runs, **Then** every fix trajectory
   terminates with one of: converged / cycle / diverged / budget -- there is no fifth outcome and
   no unbounded loop (halting is exact-equality + revisit detection + the standing node/time
   resource guards; no new numeric halting constant exists anywhere in the change).

---

### User Story 2 - Minted typed primitives enter the one searcher through the typed bottleneck (Priority: P2)

The mint stops being a second solver. Its verified products -- typed (unit, attribute, rule,
consumer) programs -- are handed to `synthesize()`'s primitive pool for THIS task, exactly like
fitted primitives are today. Only the typed record crosses the boundary; no grid, no train-pair
residue, no mint-internal state. Programs built on memorisation-class rules stay flagged, and their
refuse-on-unseen behaviour survives composition.

**Why this priority**: P2 makes the merge mandatory once the two paths are named, and the fix
combinator is most valuable over minted step-primitives (the mint can produce task-fitted steps
worth iterating). But US1 is testable without US2, so US2 is second.

**Independent Test**: A task the mint solves today (its dev harness) must be solved by
`synthesize()` alone after the merge, with the winning program naming the minted primitive; and
`arc_mint`'s direct end-to-end application path must be gone (exactly one reachable ARC solve
path in the tree).

**Acceptance Scenarios**:

1. **Given** a task where the mint induces a verified primitive, **When** `synthesize()` runs with
   the bottleneck wired, **Then** the minted primitive appears in the searched pool and the winning
   program can name it (`mint:<name>` in the program description).
2. **Given** a memorisation-flagged minted rule and a test input with an unseen attribute value,
   **When** the composed program reaches that value, **Then** the program yields no grid (the None
   propagates; the searcher treats it exactly like any failed primitive) -- never a wrong guess.
3. **Given** the merged tree, **When** counting reachable end-to-end ARC solve paths (a callable
   that maps a task to candidate grids), **Then** the count is exactly 1 of 1
   (`arc_solver.synthesize` via `make_solver`/`ArcExpert`), and `arc_mint.mint()` is reachable only
   as a primitive producer inside it.
4. **Given** the retirement, **When** the mint's unique capabilities are checked, **Then**
   leave-one-out validation, MDL ordering, and memo-flagging all still execute on the surviving
   path (port-before-retire, P2).

---

### User Story 3 - The A/B, the control arm, and the null verdict (Priority: P3)

The feature is adopted, or not, on `tools/arc_score.py` evaluation-split evidence against a
baseline captured BEFORE any implementation, at a named commit. A fixed-depth control arm decides
whether ADAPTIVITY specifically (as opposed to just more depth) earned anything. A null result --
no additional held-out solves, or adaptivity no better than fixed depth -- is a valid, successful
end state: it is recorded and the feature STOPS.

**Why this priority**: This is the honesty apparatus. It is third only because it needs US1's
mechanism to exist before it can measure it; its baseline half (capture before implementation) is
actually the FIRST work executed.

**Independent Test**: `specs/006-fixed-point-gf/baseline.txt` exists, names the commit, and
contains evaluation + training scores (library on and off) that were produced before any source
change; the final report contains the adaptive run, the fixed-K control run, and solved-ID diffs.

**Acceptance Scenarios**:

1. **Given** a clean tree at a named commit, **When** the baseline is captured, **Then**
   `baseline.txt` records `git rev-parse HEAD`, the four `arc_score` figures (2 splits x library
   on/off), and solved-ID lists -- all before any implementation task runs.
2. **Given** the implemented feature, **When** the evaluation split is scored, **Then** the result
   is reported as `N of 400` beside the baseline `11 of 400`, with the solved-ID sets diffed:
   gained IDs listed, lost IDs listed, and any lost ID individually explained or the adoption
   refused.
3. **Given** the adaptive result, **When** the control arm runs -- the same combinator forced to a
   FIXED depth K for every task, K = the mean converged depth observed in the adaptive run
   (rounded to nearest integer) -- **Then** the adaptive and fixed-K solved counts are compared;
   adaptivity is claimed only if adaptive strictly exceeds fixed-K.
4. **Given** any degenerate metric input, **When** the admission controls run, **Then** each FAILS
   as designed: the echo solver still scores ~0, the identity-only DSL still scores 0, and
   fix(identity) adds 0 solves. A control that cannot fail disqualifies the number it guards
   (Standing Invariant 5).
5. **Given** a null outcome (no new held-out solves, or adaptive <= fixed-K), **When** the verdict
   is recorded, **Then** the feature stops: the record goes into this directory and
   `docs/DEPTH_CEILINGS.md`, and the combinator is either retired with the record kept, or kept
   ONLY if it is strictly harmless (identical solved set, within resource budgets) -- it is NOT
   left behind an off-by-default flag, which would be additive (P2).

---

### User Story 4 - The live path tells the truth about its new shape (Priority: P4)

`ArcExpert` (the brain's live ARC interface) uses the surviving mechanism, and its user-facing
honest-gap message no longer claims "(<=3 steps)" once depth is variable. Documentation that
reproduces the Gf number (`docs/DEPTH_CEILINGS.md` reproduce block) names the new mechanism and the
control-arm result, whatever it was.

**Why this priority**: Smallest slice, pure consequence of the others -- but the project's honesty
discipline makes a stale claim in a live message a defect, not a nit.

**Independent Test**: Ask the live brain an `arc:` question that gaps; the gap message describes
the actual search (variable depth via fixed-point, or the unchanged wording if the feature was
nulled and retired).

**Acceptance Scenarios**:

1. **Given** the adopted feature, **When** `ArcExpert` gaps on a task, **Then** the message
   describes the surviving mechanism truthfully (no "(<=3 steps)" if depth is variable).
2. **Given** either verdict (adopted or null), **When** `docs/DEPTH_CEILINGS.md` is read, **Then**
   its Ceiling-1 section reports the new measurement with the same honesty framing as the existing
   entries (controls named, null reported as null).

---

### Edge Cases

- A fix trajectory that converges to a grid identical to the test input (fix reached, nothing
  changed): valid convergence in 1 step; the candidate is only useful if the task is identity-like;
  must not be special-cased.
- A step that converges on every training input but cycles on the test input: the program yields no
  candidate for that test input (honest gap), and this outcome is counted in the trajectory census.
- Convergence to different depths on different training pairs of the SAME task: allowed by
  construction (that is the point); the program is valid iff every training pair's fixed point
  equals its expected output.
- A minted primitive that is itself internally iterative AND gets wrapped in fix: permitted; the
  outer fix converges in 1 extra application if the inner iteration already reached stability.
- node_budget exhausted mid-trajectory: the trajectory is classified `budget`, the candidate is
  discarded, and the search truncates exactly as it does today ("budget exhausted -> truncate
  honestly", `arc_solver.py:830`).
- Grid at the 30x30 maximum where a growing step immediately diverges: classified diverged on the
  first invalid iterate; no candidate.
- The library learning a composite that CONTAINS a fix step: composites must record the combinator
  so replay is exact; a composite that replays to a different depth on a new task is correct
  behaviour (depth is the task's, not the composite's).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The searcher MUST support a fix-point combinator: repeated application of one pooled
  primitive until exact grid equality between successive iterates (convergence), with the iterate
  at convergence used as the step's output.
- **FR-002**: The combinator MUST classify every trajectory as exactly one of converged / cycle /
  diverged / budget. Cycle = any revisited earlier state; diverged = any invalid or oversize
  iterate; budget = node/time resource guard hit. Only converged trajectories may produce a grid;
  the others contribute no candidate and never surface a partial iterate as a guess.
- **FR-003**: Halting MUST introduce NO new numeric constant that arbitrates behaviour (P1). Exact
  equality and state-revisit detection are the halting rule; each application MUST be counted
  against the EXISTING `node_budget`/`time_budget` resource guards, which remain the only caps.
- **FR-004**: The combinator MUST operate over the one primitive pool (base + fitted + library +
  minted primitives) -- not a private list. Whether every pooled primitive is fix-eligible or the
  pool is filtered by a structural property (e.g. output-shape-preserving) is a design decision for
  `plan.md`, but any filter MUST be structural, not a hand-picked name list.
- **FR-005**: Minted primitives MUST enter the pool through the typed bottleneck: the only data
  crossing from mint to searcher is the typed record (unit type, attribute, rule form, rule
  parameters, consumer, memo-flag). Memo-flagged programs MUST retain refuse-on-unseen (None)
  behaviour under composition.
- **FR-006**: Exactly ONE end-to-end ARC solve path MUST remain reachable after the merge (P2).
  `arc_mint`'s direct program-application path is retired; its unique capabilities (leave-one-out
  validation, MDL ordering of rule forms, memorisation flagging) MUST be ported to / preserved on
  the surviving path BEFORE the retirement lands, and the A/B (FR-009) MUST precede any deletion.
- **FR-007**: The baseline MUST be captured before any implementation change: `git rev-parse HEAD`
  on a clean tree, `tools/arc_score.py` on both splits with library on and off, solved-ID lists,
  written to `specs/006-fixed-point-gf/baseline.txt`. Any implementation commit made before the
  baseline exists invalidates the feature's evidence.
- **FR-008**: The metric MUST keep its admission controls, and gain the combinator-specific ones:
  (a) echo solver ~0 (existing), (b) identity-only DSL 0 (existing), (c) fix(identity) converges in
  exactly 1 application and adds 0 solves, (d) fix(oscillator) is classified cycle, (e) fix(grower)
  is classified diverged. Each control MUST be able to FAIL (assert the failing outcome, not skip).
- **FR-009**: The A/B MUST compare, on the evaluation split at identical budgets: baseline vs
  adaptive-fix vs fixed-K control (K = mean converged depth of the adaptive run, rounded to nearest
  integer -- a derived value, not an authored constant). Adoption requires adaptive to strictly
  exceed baseline solved count with every lost solved-ID individually accounted for; the claim
  "adaptivity earned it" additionally requires adaptive to strictly exceed fixed-K.
- **FR-010**: A null result (no held-out gain, or adaptive <= fixed-K) MUST be recorded as a
  complete, successful outcome in this directory and in `docs/DEPTH_CEILINGS.md`, and MUST stop the
  feature. The combinator MUST NOT survive behind an off-by-default flag (additive, P2); it is
  retired with the record, or adopted on evidence -- nothing in between.
- **FR-011**: Every count in every new or touched print/report MUST read `N of M`, never `N/M`.
- **FR-012**: Determinism MUST be preserved: no RNG in the combinator or the bottleneck wiring;
  any seeded instance uses `random.Random(seed)`; two identical budget-bound runs MUST solve the
  identical task set (the existing self-check extended to cover fix programs). All prints
  ASCII-only.
- **FR-013**: The three gates MUST be green before and after every phase: `python run_selfchecks.py`,
  `pytest` (>=70% coverage), `ruff check .` + `ruff format --check src tests tools`, using
  `.venv-lake\Scripts\python.exe`.
- **FR-014**: The change MUST stay on CPU (P5): this is symbolic search over tiny grids; the
  measured CUDA crossover (N~24k-32k) is orders of magnitude above any quantity here. If any future
  widening changes that, re-measure -- but no GPU work is in scope.
- **FR-015**: No backprop anywhere (standing rule): the feature is symbolic; it MUST NOT introduce
  `loss.backward()` or any gradient path, and MUST NOT touch the e-prop learning rules.
- **FR-016**: Per-run depth telemetry MUST be reported: the distribution of converged depths and
  the trajectory-outcome census (how many converged / cycle / diverged / budget, per primitive
  class), printed `N of M`. This is the direct evidence that (a) depth actually varies and (b) F8's
  non-convergence expectation is quantified rather than assumed.
- **FR-017**: User-facing and doc claims MUST be updated to the surviving truth: `ArcExpert`'s gap
  message, `docs/DEPTH_CEILINGS.md`'s Ceiling-1 reproduce block, and the coverage ledger entry for
  the Gf faculty if its evidence line changes.

### Key Entities *(include if feature involves data)*

- **Fix trajectory**: the ordered iterates of one primitive under repetition on one grid; carries
  an outcome tag (converged / cycle / diverged / budget) and a depth (number of applications).
  Ephemeral -- never persisted.
- **Typed primitive record**: the only object crossing the mint->searcher boundary: (unit type,
  attribute name, rule form, rule parameters, consumer name, memo-flag). Deterministic, task-local,
  never persisted across tasks.
- **Depth census**: per-run aggregate of trajectory outcomes and converged-depth distribution;
  printed, and recorded in this directory for the A/B runs.
- **Baseline record** (`baseline.txt`): commit hash, gate status, four arc_score figures with
  solved-ID lists. Written once, before implementation; append-only thereafter.
- **Verdict record**: the A/B + control-arm outcome (adopted or null), solved-ID diffs, and the
  P2 disposition (what was retired). Lives in this directory and is mirrored into
  `docs/DEPTH_CEILINGS.md`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `baseline.txt` exists, names a commit, and predates every implementation change (its
  content reproduces `11 of 400` evaluation / `40 of 400` training at that commit, or records the
  re-measured figures if HEAD has moved -- whichever is measured, that is the baseline).
- **SC-002**: On the full evaluation split, 400 of 400 tasks complete with every fix trajectory
  classified (zero unclassified, zero non-termination) within the standing budgets.
- **SC-003**: All five admission controls fail-as-designed: echo ~0, identity-DSL 0 of N,
  fix(identity) depth 1 with 0 added solves, fix(oscillator) -> cycle, fix(grower) -> diverged.
- **SC-004**: The evaluation A/B is reported as `N of 400` vs baseline `11 of 400` with a solved-ID
  diff; adoption only on strict improvement with losses accounted; otherwise the null verdict is
  recorded and the feature stops. EITHER outcome satisfies this criterion -- an unrecorded outcome
  fails it.
- **SC-005**: The fixed-K control arm is run and reported beside the adaptive arm; the word
  "adaptive" appears in no claim unless adaptive strictly beat fixed-K.
- **SC-006**: After the merge, a census of reachable end-to-end ARC solve paths counts exactly
  1 of 1, and the mint's LOO/MDL/memo-flag behaviours are demonstrated on the surviving path.
- **SC-007**: All three gates green at every phase boundary; the determinism self-check passes with
  fix programs in the searched space (two runs, identical solved sets).
- **SC-008**: The depth census on the evaluation split shows a NON-CONSTANT converged-depth
  distribution (at least two distinct depths among converged trajectories of winning programs) --
  or, if it is constant, that fact is itself reported as evidence against the adaptivity
  hypothesis and feeds the null verdict.

## Assumptions

- The ARC-AGI-1 dataset is present at `data/arc/{training,evaluation}` (800 tasks); if absent, every
  instrument SKIPs with a message (existing behaviour) and no figure is recorded.
- The baseline figures at `30d8e76a` are expected to match the documented 11 of 400 / 40 of 400;
  if HEAD has moved by implementation time, the baseline is RE-MEASURED at the then-HEAD, and that
  measurement is the baseline -- the documented figure is never quoted as a stand-in (FR-007).
- The live `ArcExpert` budgets (node_budget=1500, time_budget=2.0) and the bench budgets
  (4000 / 5.0) both stay as they are: they are resource guards, and changing them mid-feature would
  confound the A/B.
- `specs/005-*` is being authored concurrently by another agent; this feature touches only
  `specs/006-fixed-point-gf/` at spec time. If 005's implementation lands in
  `reasoning/arc_*.py` first, the baseline re-measure (FR-007) absorbs it.
- `.specify/feature.json` was NOT updated to point at this directory (it is shared state under
  concurrent use); downstream speckit phases are pointed at this directory explicitly.
- The 1-D inducers in `gf.py` (`induce_program`, `OpenGf`, `synthesize_sequence`) already have their
  own variable-depth story (iterative deepening / `solve_unbounded`); they are out of scope (Q1).

## Out of Scope

- Any LLM, frozen or otherwise, anywhere in the solve path (brain-native contract; see the
  non-portability record in `research.md`).
- ARC-AGI-2/3 datasets and leaderboards; the yardstick stays ARC-AGI-1 as instrumented today.
- Variable-depth for the 1-D sequence inducers (`gf.py`) and for RAVEN (`rpm_solver.py`,
  `real_rpm_bench.py`).
- A spiking implementation of the loop itself. This feature is symbolic search, stated plainly, as
  the existing modules state it of themselves. The spiking analogue of depth-induced signal decay
  is RECORDED as research (F8, `research.md`) but not built.
- Resurrecting `probe_fluid.py` (saturated; banned as a signal) or the falsified
  "corpus abstraction" hypothesis. Neither may appear as evidence anywhere in this feature.
- Grammar/DSL widening beyond the fix combinator and the minted-primitive intake (no new
  hand-authored base primitives -- that is the developer-prior treadmill this feature exists to
  step off).

## Dependencies

- `docs/POLICIES.md` (P1 exceptions text, P2 guards) and `.specify/memory/constitution.md`
  (Standing Invariants 1-5, three gates).
- `tools/arc_score.py`, `tools/arc_baseline.py`, `reasoning/arc_bench.py` -- the instruments; no
  changes to their scoring semantics are permitted by this feature (admission controls may be
  ADDED beside them).
- `reasoning/arc_solver.py` (the searcher), `reasoning/arc_mint.py` (the typed bottleneck),
  `reasoning/arc_scene.py` (the measurement vocabulary), `reasoning/reasoning_cortex.py`
  (`ArcExpert`, the live wire).
- `data/arc/` download (external, gitignored).

## Open questions

Recorded verbatim with the assumption taken; none blocks `/speckit-plan`.

- **Q1 -- Scope of the combinator.** *"Should fixed-point variable depth also wrap the 1-D
  inducers (`gf.py`) or RAVEN?"* Could not be answered from the repo (no owner statement).
  **Assumption**: No -- ARC grid path only. The 1-D inducers already have `solve_unbounded`
  (unbounded iterative deepening), which is their variable-depth mechanism; duplicating the new
  combinator there without a benchmark that can arbitrate it would be additive.
- **Q2 -- Fix-eligibility of the pool.** *"Is every pooled primitive fix-eligible, or only a
  structurally-filtered subset (e.g. shape-preserving steps)?"* The repo gives no precedent.
  **Assumption**: every pooled primitive is eligible and the trajectory classifier is the filter
  (cycle/diverged outcomes are cheap and honest); `plan.md` may tighten this to a structural filter
  if the node-budget cost measured in Phase 0 demands it, but never to a name list (FR-004).
- **Q3 -- Adoption bar on lost solves.** *"If the new mechanism gains 4 tasks and loses 1, is that
  adoption?"* POLICIES.md P2 requires "no regression, gate-verified" but the arbiter here is a
  count with a set behind it. **Assumption**: strict count improvement AND every lost ID
  individually explained in the verdict record; an unexplained lost ID blocks adoption (Standing
  Invariant 3 -- the negative verdict is the dangerous one).
- **Q4 -- Where the retired mint harness's dev-set reporting goes.** *"arc_mint's main() prints a
  20-task dev-set report; does that reporting survive?"* **Assumption**: the report survives as a
  self-check of the PRODUCER (mint -> typed records -> pool), scored through the one searcher --
  it stops applying programs directly. Its honest-null framing ("functional apparatus contributes
  ZERO held-out solves today") is kept.

## Status

- Spec drafted 2026-08-02 against `30d8e76a`; findings F1-F8 verified against source the same day.
- Q1-Q4 recorded with assumptions; owner may override any of them before `/speckit-tasks` output
  is executed.
- Next: `/speckit-plan`.

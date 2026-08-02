# Phase 0 Research: Fixed-Point Gf (Variable-Depth Computation + Typed Bottleneck)

All source citations verified at `30d8e76a` (2026-08-02). External paper claims are taken from the
owner-supplied research brief (arXiv IDs named below); nothing in this feature depends on a paper
number being exactly right -- every decision below is re-justified against this repo's own code and
policies.

## R1. What is portable from FPRM (arXiv 2606.18206), and what is not

**The claim used.** Looped architectures carry an inductive bias toward step-by-step procedures;
the number of effective layers reached by looping determines solution quality on procedural
benchmarks (Sudoku, Maze, state tracking, ARC-AGI); and **fixed-point convergence works as an
end-to-end halting mechanism**, letting the model adapt its compute to task difficulty instead of
paying a worst-case fixed depth.

**Decision**: port the CONTROL STRUCTURE only -- iterate a step until its output stops changing,
and let the iteration count be the task's own property. Do not port the neural machinery (looped
transformer blocks, learned halting heads): the Gf ARC path here is symbolic program synthesis
(`arc_solver.py` states this of itself), and the honest place to add adaptive depth is a **fix
combinator in the program grammar**, not a neural loop bolted beside the searcher.

**Rationale**: the repo's measured Gf gap was proven architectural (`docs/DEPTH_CEILINGS.md`,
Ceiling 1: 0.0% -> 2.75% with the cause analysis; the 8-design adversarial panel). The specific
architectural lack this addresses is exact: `synthesize()` applies each primitive once, at most 3
deep (`arc_solver.py:770-859`), and the only iteration in the whole solve path is hand-authored
inside primitive bodies (`_gravity`, `_fill_enclosed`). FPRM's thesis maps cleanly: those
hand-iterated primitives are frozen depth; the combinator makes depth a runtime outcome.

**The depth-induced signal-propagation problem, translated.** FPRM had to repair signal
degradation at depth (pre-norm layers, residual scaling) before looping paid -- deep loops
otherwise destroy their own signal. The symbolic analogue is real and MUST be treated as a
first-class expected outcome, not an implementation surprise:

| FPRM (continuous) | this repo (discrete grids) |
|---|---|
| activations blow up / vanish at depth | iterates grow past 30x30 (`_tile`) or leave validity (`_ok` fails) |
| oscillation, no convergence | period-2 cycles (`_reflect`, rotations, some recolours) |
| needs normalization to converge | needs trajectory CLASSIFICATION: converged / cycle / diverged / budget |

The expectation to record now, so a later measurement cannot be spun: **most base primitives will
NOT converge under iteration** (reflections and rotations cycle by construction; tilers diverge by
construction; only contraction-like steps -- gravity moves, fills, prunes, recolour-to-stable --
can converge in more than the trivial 1 step). The per-primitive trajectory census (FR-016,
SC-008) is the deliverable that makes this visible. If the census shows no primitive family with
non-trivial convergent depth, the hypothesis dies at Phase 3 and that is a reportable null.

**Alternatives considered**:
- *Learned halting signal* (ACT-style, a spiking readout that decides "stop"): rejected for this
  feature -- it adds a trained component whose failure modes (never halting, always halting) need
  their own instrument, and the discrete domain gives exact halting for free. Recorded as the
  natural follow-on IF the combinator earns adoption and the census shows depth matters.
- *Raise `max_depth` instead* (just search deeper): rejected on measurement grounds -- depth-k
  repetition of one primitive costs k levels of a branching search (pool size ^ depth), and the
  `seen` pruning (`arc_solver.py:846-848`) actively removes revisit-shaped iteration. A loop is one
  search step; deeper open-loop search is exponentially many.

## R2. The non-portable ARC leaders -- recorded so nobody re-proposes them

Every other current ARC-AGI-2/3 leader in the surveyed set is a harness around a **frozen frontier
LLM**: Tycho (arXiv 2607.28287), ARCANA (arXiv 2607.09059), PoTRE (arXiv 2607.20268), PRO-LONG
(arXiv 2607.20064). They differ in how they prompt, sample, verify, or refine, but the reasoning
substrate in all four is a pretrained LLM that this project would have to import.

**Not portable, twice over.**
1. **Brain-native contract**: the substrate must be the brain's own machinery; an imported LLM is a
   second brain (P4 in spirit) and a wholesale hardcode (P1 in spirit) -- the capability would live
   in the import, not in anything this project built or can grow.
2. **It teaches the brain nothing** (P0): the point of the Gf program is that the faculty improves;
   a frozen model's score is not the brain's score, and no self-improvement avenue passes through
   it.

This is the same verdict `docs/DEPTH_CEILINGS.md` already records for hand-specified attribute
grammars (the developer-aware clause charges the prior against the score) -- an imported LLM is the
largest possible developer-supplied prior. **Do not re-propose these as adoption candidates.** The
two papers mined here (FPRM, Hourglass) were mined for CONTROL STRUCTURE and INTERFACE DISCIPLINE
respectively, which port without importing any weights.

## R3. The typed bottleneck: what Hourglass adds over what the MINT already has

**The claim used (arXiv 2607.11696).** Strict context isolation between reasoning stages:
Induction compresses examples into a schema; Deduction derives a rule; an Implementer compiles
them; an error-driven Refiner revises. **Only (schema, rule) crosses a stage boundary.** Strip the
LLM; the portable idea is the typed bottleneck.

**What already exists (do not rebuild -- P2).** `arc_mint.py` IS a typed bottleneck, per-primitive:
typed attribute readers over `arc_scene` measurements (`_obj_attr`, `_region_attr`,
`_object_units`, `_region_units`) -> `induce_rule()` with MDL-ordered functional forms
(const/mod-k/offset-mod10/affine before rank/lookup), leave-one-out validation, and
memorisation-class flagging (`_MEMO_KINDS`, refuse-on-unseen) -> typed consumers (recolour / crop /
reflect / tile / materialise). What crosses a mint stage boundary is already only the typed record.
The Hourglass idea is therefore **partly built**, and this spec says so plainly.

**The delta this feature owes** (and the part that is genuinely missing):
1. The mint's products stop at the mint. `reasoning_cortex.py` never references `arc_mint`; the
   mint applies programs directly in its own harness. The bottleneck exists WITHIN the mint but not
   BETWEEN the project's reasoning stages (perception -> mint -> searcher -> closer). Decision:
   the typed record becomes the intake through which minted primitives join `synthesize()`'s
   `_pool()` for the task at hand -- the same position fitted primitives occupy today
   (`_fitted_prims`, `arc_solver.py`).
2. No error-driven Refiner exists, and none is built here. Hourglass's Refiner revises the schema
   on implementation failure; the honest equivalent (re-mint with the failing pair's residual as a
   new observation) is scoped OUT of this feature -- it is a second hypothesis and deserves its own
   baseline. Recorded as follow-on, not smuggled in.

**Alternatives considered**:
- *Full Hourglass staging (four modules, schema objects between all stages)*: rejected -- it would
  re-implement as ceremony what `arc_scene` (measure) + `arc_mint` (induce) + `arc_solver` (search)
  already are. The stages exist; only the boundary discipline between mint and searcher is missing.
- *Passing mint-built GRIDS to the searcher* (let the mint pre-solve and hand over its output):
  rejected -- that is exactly what the bottleneck forbids (a grid is not a typed record), and it
  would keep two solve paths alive (P2 violation with extra steps).

## R4. Where the fix combinator lives in the search

**Decision**: pool-level derived primitives. For each fix-eligible primitive `f` in the pooled set,
the pool also offers `("fix:"+name, fix_f)` where `fix_f(g)` runs the trajectory
`g, f(g), f(f(g)), ...` to classification and returns the converged grid or None. A fix step is
therefore ONE search step, composable inside the existing depth (e.g. `crop > fix:gravity_down >
map` is a depth-3 program), and the `seen`-set pruning is untouched -- `fix_f` maps state to state
like any primitive.

**Rationale**:
- One search step per loop is the FPRM lesson in symbolic form (R1): the loop is a combinator, not
  depth.
- It preserves every existing guarantee by construction: totality (returns None on
  cycle/diverged/budget, exactly like a failing primitive today, `arc_solver.py:833-844`),
  determinism (no RNG; trajectory is a pure function of `f` and `g`), and MDL bias (a fix program
  is one name longer than its base, so smallest-first ordering still prefers the non-iterated
  solution when both work).
- Q2's assumption (spec.md) holds here: ALL pooled primitives are fix-eligible; the classifier is
  the filter. Wasted trajectories cost nodes, and nodes are already the budget currency. If
  Phase 0 measurement (tasks T00x) shows the node cost is prohibitive, the permitted tightening is
  a STRUCTURAL filter (e.g. only shape-preserving primitives, decided by comparing input/output
  shapes on the train pairs) -- never a hand-picked name list (FR-004).

**Halting needs no new constant (the P1 analysis, in full).** The directive names the trap: a
convergence tolerance and a max-iteration cap are exactly the tuned constants P1 forbids. Neither
exists here:
- *Tolerance*: grids are finite discrete states; convergence is `iterate == previous` (exact
  equality, the same `_tup` tuple comparison the searcher already uses for `seen`). There is no
  epsilon because there is nothing continuous to be within epsilon of.
- *Iteration cap*: termination is guaranteed WITHOUT one. Each iteration either (a) equals the
  previous iterate -> converged; (b) revisits any earlier iterate -> cycle, stop; (c) is invalid or
  oversize -> diverged, stop; or (d) is a NEW valid state -- and every new state costs one node
  against the EXISTING `node_budget` / `time_budget` resource guards (`arc_solver.py:25` names
  these "resource guards only", which is the standing, reviewed P1 disposition for them). The
  state space is finite, so even without budgets the trajectory must terminate by (a), (b) or (c).
  No new numeric constant arbitrates anything.

**Alternatives considered**:
- *fix over COMPOSITE bodies* (`fix(f>g)`): rejected for this feature -- the candidate space
  squares, and FPRM's own evidence is about looping ONE block. If the census shows single-step
  loops converge but never solve, this is the recorded next widening.
- *A global outer loop over the whole program* (apply the entire synthesized program repeatedly):
  rejected -- it changes the meaning of every existing program (silent behaviour change of solved
  tasks) instead of adding an explicit, named, searchable form.

## R5. The instrument, the baseline, and the control arms

**The deciding benchmark** (P2: "better requires a measurement that exists"):
`tools/arc_score.py <split> <on|off> [limit] [--data DIR]` -> `arc_bench.score()` -- pass@2, exact
grid match, SKIP-not-zero on missing data, prints solved-ID lists. It already carries two
admission controls that genuinely fail (echo solver ~0 asserted in `arc_bench`'s self-check;
identity-only DSL 0 asserted in `arc_solver`'s), plus a determinism check (two budget-bound runs,
identical solved sets). `tools/arc_baseline.py` covers the brain-side interfaces and is not the
arbiter here (its role was the ORIGINAL "before"; this feature's before/after lives at the solver
boundary where the mechanism changes).

**Baseline discipline** (FR-007): captured BEFORE any implementation commit, on a clean tree, at a
named commit, into `baseline.txt` -- both splits, library on and off, solved-ID lists, wall time.
The documented figures (11 of 400 evaluation, 40 of 400 training at the `docs/DEPTH_CEILINGS.md`
measurement commit) are expectations, not the baseline itself: HEAD has moved since (now
`30d8e76a`, and `specs/005-*` is landing concurrently), so the measured-at-capture figure is the
baseline whatever it reads. Lesson inherited from specs/003: verify the tree is CLEAN for the
touched files before capturing, because an earlier research pass measured an uncommitted rewrite
and mislabelled it as the baseline.

**Control arms** (the honesty apparatus this repo's history demands -- three green-only gates
shipped inside one feature is the standing cautionary tale):
1. **Admission controls that must FAIL**: echo ~0 and identity-DSL 0 (existing, re-asserted);
   fix(identity) converges at depth 1 and adds 0 solves; fix(reflect) classifies as cycle;
   fix(tile) classifies as diverged. Each is an assert on the FAILING outcome -- a control that
   cannot go red admits nothing (Standing Invariant 5).
2. **The fixed-depth null arm** (SC-005): re-run the evaluation split with the combinator forced to
   exactly K applications (no convergence check, take iterate K), K = mean converged depth of
   winning fix programs in the adaptive run, rounded to nearest integer. K is DERIVED from the
   adaptive run's own telemetry -- not an authored constant (P1) -- and the arm answers the only
   question that separates "adaptive compute" from "more compute": does halting-by-convergence beat
   the same average depth spent blindly? If adaptive <= fixed-K, adaptivity earned nothing: null.
3. **Null is a complete result** (FR-010): recorded in the verdict record and
   `docs/DEPTH_CEILINGS.md`, feature stops, no flag-parked combinator. The repo has prior art for
   reporting exact nulls with pride (library learning: "identical scores and identical solved-ID
   sets ... Reported as measured, not quietly dropped").

**Banned evidence, restated**: `probe_fluid.py` is SATURATED and banned as a signal; the "corpus
abstraction" hypothesis is FALSIFIED. Neither may appear in any figure, control, or justification
in this feature. The only Gf evidence admissible here is the ARC harness above.

## R6. The P2 retire decision, named before implementation

**The pair doing the same job**: end-to-end ARC solving exists twice -- (a)
`arc_solver.synthesize()` via `make_solver`/`ArcExpert` (live), and (b) `arc_mint.mint()` +
direct program application in its own harness (bench-only, dormant w.r.t. the live path, F7).

**The route**: MERGE, not adopt -- each holds something the other needs (the searcher has
composition, closers, library, the live wire; the mint has out-of-closure primitive induction with
LOO/MDL/memo-flag discipline). The ONE surviving mechanism is `synthesize()` with the mint as its
typed-primitive producer. Retired: the mint's direct application path (its `main()` harness becomes
a producer self-check scored THROUGH the searcher, per Q4's assumption).

**Port-before-retire inventory** (P2's guard, itemized so the A/B can check it): leave-one-out
validation of induced rules; MDL ordering of rule forms; memorisation-class flagging with
refuse-on-unseen (None) semantics under composition; the honest dev-set report including its
current null ("functional apparatus contributes ZERO held-out solves today").

**The A/B that precedes deletion**: evaluation + training splits, identical budgets, before (two
paths, baseline) vs after (one path, merged). Regression rule per Q3's assumption: strict solved
count improvement AND every lost solved-ID individually explained, else no adoption and no
deletion.

## R7. Execution-order note (deviation from spec priority order, and why)

The spec orders stories US1 (combinator) > US2 (merge) > US3 (measurement) > US4 (surface). The
PLAN executes the baseline half of US3 FIRST (nothing may change before `baseline.txt` exists,
FR-007), then US1, then US3's controls against US1 alone (the combinator's A/B is informative
before the merge lands and isolates ITS contribution from the mint intake's), then US2, then the
full A/B + verdict, then US4. Same lesson as specs/003 ("Phase order deviates from spec priority
order"): evidence validity dictates order, story labels still map to the spec.

## R8. Risks retired by research vs risks that remain

Retired by analysis:
- *P1 tolerance/cap trap*: does not arise -- discrete exact halting, existing resource guards (R4).
- *Non-termination*: impossible by construction -- finite states + revisit detection + budgets (R4).
- *Determinism*: trajectory is a pure function; no RNG anywhere in the design (FR-012).

Remaining, carried into plan.md Risks:
- *The census comes back empty* (no primitive family converges non-trivially): the hypothesis
  nulls early. Mitigation: none -- that is the experiment working. Cost is bounded by running the
  census (Phase 3) before the merge work (Phase 4).
- *Node-budget starvation*: fix trajectories consume nodes fast; at `node_budget=4000` the added
  pool entries could crowd out the existing search and LOSE currently-solved tasks. Detection: the
  A/B's lost-ID accounting (FR-009). Disposition if seen: the structural filter of R4, or refusal.
- *Concurrent specs/005 lands in the same files*: baseline re-measure absorbs drift (R5); if 005
  touches `arc_solver.py` itself, rebase and re-baseline before Phase 2.
- *Library composites containing fix steps replay differently*: intended semantics (depth belongs
  to the task), but the determinism self-check must cover it explicitly (edge case in spec.md).

# Trajectory census -- fix combinator on the held-out evaluation split

Published BEFORE any adoption reasoning (Standing Invariant 1; contract C15). Produced by
`tools/arc_census.py evaluation` (tracked; raw transcript in `census_run.log`). Run state:
baseline commit `95c3f806` + the Stage-A combinator (T004-T011) in the working tree; the mint
intake (T016) was deliberately NOT yet wired, so this census isolates the combinator exactly as
T012 requires (library off, node_budget=4000, time_budget=5.0 -- the bench budgets).

## Headline

| figure | value |
|---|---|
| tasks completed | 400 of 400 (SC-002: zero unclassified, zero non-termination) |
| trajectories classified | 500918 of 500918 |
| solved | 12 of 400 -- the BYTE-IDENTICAL solved-ID set as the baseline (0 gained, 0 lost) |
| winning programs containing a `fix:` step | 0 of 12 |
| mean winning-fix depth | NONE (0 of 0 winning fix trajectories) |
| wall time | 139 s vs baseline 337 s (within the 2x bound; see note below) |

## Trajectory outcomes per primitive class

| class | converged | cycle | diverged | budget |
|---|---|---|---|---|
| base | 209414 of 459009 | 38399 of 459009 | 210993 of 459009 | 203 of 459009 |
| fitted | 4755 of 41909 | 0 of 41909 | 37120 of 41909 | 34 of 41909 |
| library | 0 of 0 (library off) | 0 of 0 | 0 of 0 | 0 of 0 |
| mint | 0 of 0 (intake not yet wired) | 0 of 0 | 0 of 0 | 0 of 0 |

research.md F8/R1's recorded expectation is CONFIRMED and now quantified: most base primitives
do not converge under iteration (46 of 100 diverge, 8 of 100 cycle); fitted primitives diverge
89 of 100 times (fill/frame return None once inapplicable, which classifies diverged).

## Converged-depth distribution (all converged trajectories, 214169 of 214169)

depth 1: 80450 -- true fixed points on first application (identity-like on that grid)
depth 2: 128020 -- idempotent steps (act once, confirm once); the dominant convergence mode
depth 3-19: 5699 -- GENUINE multi-step contractions exist (gravity chains, deletion chains,
fill cascades), thinning roughly geometrically: 3616 at depth 3, 1198 at 4, 456 at 5, 231 at 6,
70 at 7, 57 at 8, 28 at 9, 20 at 10, and single digits from 11 to 19.

## T013 DECISION POINT (recorded before any merge/adoption reasoning)

1. The variable-depth hypothesis is NOT structurally dead: non-trivial convergence exists
   (5699 of 214169 converged trajectories at depth >= 3, reaching depth 19). The classifier
   works; depth genuinely varies; SC-008's non-constant depth distribution is satisfied in
   trajectory space.
2. BUT the combinator closes NOTHING: with fix twins for every pooled entry, the evaluation
   solved set is byte-identical to baseline (12 of 400, same IDs). Not one winning program
   uses a `fix:` step. Iteration reaches new grids, not new SOLVED grids -- the tasks the DSL
   misses are missing VOCABULARY, not depth (consistent with arc_diagnostic's DSL-coverage
   verdict and docs/DEPTH_CEILINGS.md Ceiling 1).
3. Fixed-K control arm (C18): K = round(mean winning-fix depth) is UNDERIVABLE -- there are
   0 of 0 winning fix trajectories to average. `tools/arc_fixdepth_control.py` therefore SKIPs
   by contract (a SKIP, not a zero). This is itself the strongest possible control result:
   there is no adaptivity claim to test because adaptivity never earned a solve.
4. Phase 5 (the typed-record merge, US2) proceeds regardless -- it is an independent P2
   payment, and tasks.md forbids skipping it. The Phase 6 A/B is hereby marked EXPECTED-NULL
   for the combinator arm.

## Budget-semantics note (recorded so the A/B cannot be spun)

A pool application charges 1 node and applies the primitive to every grid in the search state
(typically 3-5 grids); a fix application charges 1 node per application on ONE grid (contract
C2). Fix twins therefore consume the same 4000-node budget with fewer total grid applications,
which is why the census run is FASTER than baseline (139 s vs 337 s) while searching less. The
A/B holds CHARGED NODES equal, exactly as the contract defines budget parity.

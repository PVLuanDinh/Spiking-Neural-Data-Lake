# Verdict -- specs/006-fixed-point-gf (A/B, control arm, P2 disposition)

Recorded 2026-08-03. Every figure below traces to `baseline.txt` (commit `95c3f806`),
`census.md`/`census_run.log`, `ab_runs.log`, or `final_runs.log` in this directory. Budgets
identical across all arms: node_budget=4000, time_budget=5.0, pass@2 (`tools/arc_score.py`
semantics, frozen).

## The arms (evaluation split, held out; counts N of 400)

| arm | eval solved | training solved | source |
|---|---|---|---|
| baseline (no combinator, no mint intake) | 12 of 400 | 43 of 400 | baseline.txt |
| combinator only (fix twins, no mint) | 12 of 400 -- byte-identical ID set | not run (census isolates eval) | census.md |
| fixed-K control (C18) | SKIP -- K underivable from 0 of 0 winning fix trajectories | -- | ab_runs.log |
| merged (fix twins + mint intake) | 18 of 400 | 51 of 400 (2 lost, accounted below) | ab_runs.log |
| FINAL (mint intake adopted, combinator retired) | **18 of 400 = 4.50%** | **53 of 400 = 13.25%** | final_runs.log |

Library on/off: identical solved sets in every arm and split (the standing library null,
re-confirmed three more times; see baseline.txt finding 2).

## Solved-ID diffs (final state vs baseline)

Evaluation GAINED 6 of 6, LOST 0 of 12 -- every winner is a depth-1 minted record through the
one searcher (per-ID winning programs, probed at identical budgets):

| id | winning program | note |
|---|---|---|
| 00576224 | mint:tile_alt[3x3] | in the mint's 20-task dev subset (contaminated) |
| 00dbd492 | mint:recolor_region_by[side]:lookup | dev subset (contaminated); memo-flagged |
| 0a2355a6 | mint:recolor_object_by[nholes]:lookup | dev subset (contaminated); memo-flagged |
| 37d3e8b2 | mint:recolor_object_by[nholes]:rank | CLEAN -- never inspected; memo-flagged |
| 84f2aca1 | mint:recolor_region_by[size]:affine | CLEAN -- the first FUNCTIONAL held-out ARC solve |
| ae58858e | mint:recolor_object_by[shapecells]:lookup | CLEAN; memo-flagged |

CONTAMINATION DISCLOSURE: 20 of 400 evaluation tasks form `scratchpad/oracle_dev.json`, the
mint's Phase-1/3 dev set; 3 of the 6 gains are in it. Memo-flagged wins are visible as such in
the program description (`:lookup` / `:rank` in the name) per contract C12.

Training GAINED 10 of 10, LOST 0 of 43 in the FINAL state (all ten winners are depth-1
mint records: keep[holes], paint_perimeter[const], materialize[period], crop[rarest], and six
induced recolours -- five affine + one offset_mod10, all functional forms).

## Lost-solve accounting (T023, Standing Invariant 3)

The MERGED arm (fix twins present) lost 2 training IDs; each individually accounted:

- `0b148d64` -- baseline winner `keep_odd_color > crop_content` (depth 2). With fix twins in
  the pool, trajectory applications starve the 4000-node budget before depth-2 reaches it.
  Proven causal: same tree with fix entries absent solves it with the identical program.
- `496994bd` -- baseline winner `half_top > mirror_v` (depth 2). Same mechanism, same proof.

Disposition: NO budget was raised (the forbidden fix); no structural filter was needed --
the combinator that caused the starvation is retired outright (below), and the FINAL state
recovers both IDs: 0 of 43 training losses, 0 of 12 evaluation losses.

## The combinator verdict: NULL -- and a strict cost

- 0 of 12 evaluation winners used a fix step (census); 0 of 500918 trajectories unclassified.
- Mean winning-fix depth: NONE (0 of 0) -- the fixed-K arm is vacuous by measurement, so the
  word "adaptive" appears in NO claim (SC-005 satisfied by absence).
- The combinator's only measured effect was negative: 2 of 400 training solves starved.
- SC-008: the converged-depth distribution is non-constant in trajectory space (depths 1..19)
  but constant-absent among winners -- evidence AGAINST the adaptivity hypothesis, feeding
  this null. Iteration reaches new grids, not new solved grids: the unsolved tasks lack
  VOCABULARY, not depth (arc_diagnostic's standing verdict, independently reconfirmed).

**null result; feature stops here** (the fixed-point arm). Per FR-010 the combinator does NOT
survive behind a flag: `_fix_run`, `_fix_k_run`, the pool twins, the stats/fix_depth
parameters, and the fix-specific tests are DELETED from the live tree; the record survives in
this directory, `docs/DEPTH_CEILINGS.md`, and the feature-measurement commit.
`tools/arc_census.py` and `tools/arc_fixdepth_control.py` remain tracked as instruments of
record (SKIP-printing; reproducible at the feature-measurement commit).

## The merge verdict: ADOPT (the P2 payment stands on its own evidence)

Adaptive-vs-baseline bar (Q3): strict improvement (12 -> 18 of 400 evaluation, 43 -> 53 of 400
training) with every lost ID individually accounted AND recovered in the final state. The
typed record (unit, attribute, rule_form, rule_params, consumer, memo_flag) is the only
mint -> searcher crossing; `fn` is compiled FROM the declarative fields (auditable by
construction; reconstruction asserted in tests and the mint self-check).

Port-before-retire inventory (R6), asserted on the surviving path BEFORE deletion
(tests/test_arc_fix.py + arc_mint self-check):
- leave-one-out validation still rejects a train-only functional fit;
- MDL order still prefers functional forms over rank/lookup;
- memo-flagging still marks rank/lookup, refuses-on-unseen (None), and the flag is visible in
  the winning program description;
- the honest-null dev framing survives ("functional apparatus contributes ZERO held-out
  solves" on the 20-task dev set -- while 84f2aca1 is the first functional held-out solve on
  the never-inspected remainder).

## P2 disposition -- exactly ONE end-to-end ARC solve path (SC-006)

RETIRED: `arc_mint.candidates()` and `arc_mint.mint()` (the direct program-application path;
the bounded depth-2 composition harness retired with it -- composition is the searcher's job,
and every mint-earned solve is depth-1 through the searcher). RETIRED: the fix combinator (all
of it, above). SURVIVING: `arc_solver.synthesize()` (via `make_solver` on the bench, via
`ArcExpert` live), fed by `arc_mint.mint_records()` through `_pool()`.

Grep census: `def candidates(` and `def mint(` -- 0 of 0 matches in `arc_mint.py`;
`mint_records` consumers -- 1 of 1 (`arc_solver._pool`); end-to-end ARC solve paths reachable
in the tree -- **1 of 1** (`synthesize`). Enforced by
`tests/test_arc_fix.py::test_direct_application_path_is_retired`.

## Wall time

Final evaluation runs: 228 s (lib on) / 231 s (lib off) vs 337 s baseline lib-off -- within
the 2x bound (mint fitting cost is charged inside the same per-task budgets).

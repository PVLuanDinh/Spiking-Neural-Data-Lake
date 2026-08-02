# Phase 1 Data Model: Fixed-Point Gf

No entity in this feature is persisted to the one lake (P4 note: no new durable store; the single
durable store remains `brain_lake.bin`, untouched). Everything below is either task-local and
ephemeral, or a plain-text record in this spec directory.

## Entity: Fix trajectory (ephemeral, inside one `synthesize()` call)

The ordered iterates of one primitive under repetition on one grid.

| Field | Type | Meaning |
|---|---|---|
| `base` | str | pooled primitive name being iterated (e.g. `gravity_down`, `mint:recolor_area_mod3`) |
| `iterates` | implicit | `g0, g1 = f(g0), g2 = f(g1), ...` -- held only as current/previous + a revisit set of grid tuples |
| `outcome` | enum | exactly one of `converged` / `cycle` / `diverged` / `budget` |
| `depth` | int | number of applications performed (converged: applications to reach `g_k == g_{k-1}`) |
| `result` | grid or None | the converged grid iff `outcome == converged`, else None |

### Invariants

- Termination is structural: every iteration is (a) equal to previous -> `converged`; (b) present
  in the revisit set -> `cycle`; (c) invalid/oversize (`_ok` false) -> `diverged`; or (d) a new
  valid state, which costs one node against the standing `node_budget`/`time_budget` ->
  eventually `budget`. There is no fifth branch and no iteration-cap constant (P1; research.md R4).
- Only `converged` may surface a grid. A `cycle`/`diverged`/`budget` trajectory contributes no
  candidate -- never the last iterate as a guess (FR-002).
- Purity: the trajectory is a deterministic function of (`f`, `g0`). No RNG, no wall-clock in the
  logic (wall-clock appears only via the standing time_budget resource guard, exactly as today).
- Identity fixed point: if `f(g0) == g0` the trajectory is `converged` at `depth == 1`.

### State transitions

`running -> converged | cycle | diverged | budget` (terminal, exactly one).

## Entity: Typed primitive record (the bottleneck; task-local)

The ONLY object that crosses the mint -> searcher boundary (FR-005). Produced by
`arc_mint.mint_records(train)`; consumed by `arc_solver._pool()`.

| Field | Type | Meaning |
|---|---|---|
| `name` | str | stable ASCII name, prefixed `mint:` when pooled (e.g. `mint:recolor_obj_area_affine`) |
| `unit` | str | unit family the rule reads (`object` / `region` / `grid`) |
| `attribute` | str | arc_scene attribute measured (e.g. `area`, `holes`, `period`) |
| `rule_form` | str | one of the mint's forms: `const` / `modk` / `offset` / `affine` / `rank` / `lookup` |
| `rule_params` | tuple | the induced parameters (integers/small tuples; fully determines the rule) |
| `consumer` | str | the typed consumer (`recolor` / `crop` / `reflect` / `tile` / `materialize` / `paint`) |
| `memo_flag` | bool | True iff `rule_form` is memorisation-class (`rank`/`lookup`) |
| `fn` | callable | the compiled grid -> grid-or-None program (total; None on unseen attribute when memo-flagged) |

### Invariants

- No grid, train pair, or mint-internal state crosses the boundary -- the record's non-callable
  fields fully describe the program (auditable bottleneck; a test reconstructs `fn`'s behaviour
  from them on the train pairs).
- Memo-flagged records keep refuse-on-unseen: `fn` returns None on an attribute value outside the
  induced table, and the searcher treats None exactly as a failed primitive (FR-005, US2/AS-2).
- LOO-validated before emission: a record is only emitted if the mint's leave-one-out check passed
  (ported guard, research.md R6).
- Deterministic order: records are emitted in MDL-then-name sorted order so pool order (a search
  prior and tie-break) stays deterministic.

## Entity: Depth census (per run; published as `census.md` for the record runs)

| Field | Type | Meaning |
|---|---|---|
| `per_class` | table | for each primitive class (base / fitted / library / mint): count of trajectories by outcome, printed `N of M` |
| `depth_distribution` | table | converged depths -> counts, overall and for WINNING programs only |
| `mean_winning_depth` | derived | mean converged depth over winning fix programs -- the SOURCE of the control arm's K (rounded to nearest int) |

Invariant: the census for the record runs is published to this directory BEFORE any adoption
reasoning (Standing Invariant 1; plan Phase 3).

## Entity: Baseline record (`baseline.txt`; write-once, then append-only)

Same shape as `specs/003-.../baseline.txt` and `specs/004-.../baseline.txt`:

- `git rev-parse HEAD` on a verified-clean tree (clean specifically for
  `src/snn_data_lake/reasoning/arc_solver.py`, `arc_mint.py`, `arc_scene.py`,
  `reasoning_cortex.py`, `arc_bench.py`, `tools/arc_score.py`).
- Three-gate status at that commit.
- Four `arc_score` figures (training/evaluation x library on/off) with solved-ID lists and wall
  times, each printed `N of M`.

## Entity: Verdict record (`verdict.md`)

- The A/B table: baseline vs adaptive vs fixed-K, `N of 400` each, solved-ID sets diffed (gained
  list, lost list, each lost ID with its individual account or the word REFUSED).
- The adaptivity claim: allowed only if adaptive > fixed-K strictly (SC-005).
- The P2 disposition: what was retired, the port-before-retire inventory check, and the count of
  reachable end-to-end ARC solve paths (must read `1 of 1`).
- On null: the sentence "null result; feature stops here" plus where the record was mirrored
  (`docs/DEPTH_CEILINGS.md`).

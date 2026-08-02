# Contract: the fix combinator, the typed-record intake, and the census

Binding on the implementation; `tests/test_arc_fix.py` and the extended module self-checks are the
enforcement. File/function names below are the contract surface; internal helper names are free.

## 1. Trajectory contract (`arc_solver.py`)

A single function owns the loop (suggested name `_fix_run(fn, grid, nodes_ref, budget_state)`):

```
_fix_run(f, g0) -> (outcome, depth, result)
  outcome in {"converged", "cycle", "diverged", "budget"}
  result is a grid tuple iff outcome == "converged", else None
```

Rules:

- **C1 (halting, parameter-free).** Stop with `converged` when `f(g) == g` (exact tuple equality).
  Stop with `cycle` when an iterate equals ANY earlier iterate other than the immediate fixed-point
  case. Stop with `diverged` the moment an iterate is None, malformed, or fails `_ok` (which
  enforces the 30x30 bound). Stop with `budget` when the standing node/time guard trips. No other
  stop condition may exist; in particular NO max-iteration constant.
- **C2 (node accounting).** Every application of `f` increments the same node counter that
  `synthesize()` charges for a primitive application today, and the existing
  `node_budget`/`time_budget` checks apply to it unchanged.
- **C3 (no partial answers).** Non-`converged` outcomes yield None to the search, byte-for-byte
  the same handling as a primitive returning None today (`arc_solver.py:839-844` semantics).
- **C4 (purity/determinism).** `_fix_run` is a pure function of `(f, g0)` up to resource guards;
  no RNG; ASCII-only in any print.
- **C5 (exceptions).** An exception inside `f` is caught and treated as `diverged` (primitives
  must be total; today's per-application try/except semantics are preserved).

## 2. Pool contract (`arc_solver.py::_pool`)

- **C6.** For every entry `(name, fn)` in the pool (base + fitted + library + minted), the pool
  also exposes `("fix:" + name, fix-wrapped fn)` UNLESS a structural filter is active. Any filter
  MUST be a property computable from the train pairs or the primitive's input/output shapes --
  never a hand-picked name list (FR-004). The identity primitive, if pooled, is NOT exempted --
  its fix converges at depth 1 and is admission control (c).
- **C7.** `fix:` entries participate in search, `seen` pruning, library learning, and program
  descriptions exactly like any primitive. A library composite may contain `fix:` steps; replay
  runs the loop afresh (depth belongs to the task, not the composite).
- **C8 (MDL).** A `fix:` entry ranks after its base in the hand order (one position of extra
  cost), so smallest-first search prefers the non-iterated program when both close.
- **C9 (dsl="identity" control).** The identity-only control DSL gains NO fix entries -- it must
  keep scoring 0; the control's job is to stay degenerate.

## 3. Typed-record intake contract (`arc_mint.py` -> `arc_solver.py`)

- **C10.** `arc_mint.mint_records(train) -> list[TypedPrimitiveRecord]` (fields per
  data-model.md). It is the ONLY mint symbol `arc_solver` may import. No grid and no train-pair
  residue crosses; a test asserts the record's declarative fields reproduce `fn`'s behaviour on
  the train pairs.
- **C11.** `_pool()` appends minted entries as `("mint:" + rec.name, rec.fn)` after fitted
  primitives, in the mint's deterministic MDL-then-name order.
- **C12.** Memo-flagged records: `fn` returns None on unseen attribute values; under composition
  and under fix-wrapping this None propagates as C3. The flag itself appears in the program
  description so a memorisation-assisted solve is visible in solved-ID reporting.
- **C13 (retirement).** After the merge lands (and only after the A/B), no reachable call path
  applies a mint program to a test input except through `synthesize()`. `arc_mint.main()` becomes
  the producer self-check: it emits records, scores them THROUGH `make_solver()` on its dev set,
  and keeps its honest-null framing. The port-before-retire inventory (LOO, MDL order,
  memo-flagging) is asserted by tests BEFORE the direct path is deleted.

## 4. Telemetry and census contract

- **C14.** `synthesize()` exposes per-call trajectory statistics (outcome counts, converged
  depths) via an optional collector argument or return channel -- zero cost when unused, no global
  state. `tools/`-side census runs aggregate it.
- **C15.** All counts print `N of M` (never `N/M`); the census for record runs is written to
  `specs/006-fixed-point-gf/census.md` before adoption reasoning (Standing Invariant 1).

## 5. Admission controls (each asserted on its FAILING outcome)

| id | control | required outcome |
|---|---|---|
| A1 | echo solver on real tasks (existing, `arc_bench` self-check) | ~0, asserted |
| A2 | `dsl="identity"` on real tasks (existing, `arc_solver` self-check) | 0 of N, asserted |
| A3 | fix(identity) on any grid | `converged`, depth 1, adds 0 solves |
| A4 | fix(reflect_h) on any non-symmetric grid | `cycle`, no candidate |
| A5 | fix(tile_2x2-style grower) | `diverged`, no candidate |

## 6. A/B and control-arm contract

- **C16.** Baseline (`baseline.txt`) precedes every implementation commit and names its commit.
- **C17.** The record A/B holds budgets identical across arms (node_budget=4000, time_budget=5.0,
  the bench defaults) and reports solved-ID set diffs, not just counts.
- **C18.** Fixed-K arm: same pool, fix entries replaced by exactly-K-applications entries
  (`K = round(mean converged depth of winning fix programs in the adaptive record run)`); K is
  computed by the harness from the census, never typed in.
- **C19.** Adoption: adaptive > baseline strictly, all lost IDs individually accounted;
  "adaptive" as a claim additionally requires adaptive > fixed-K strictly. Anything else is a
  null: `verdict.md` records it, `docs/DEPTH_CEILINGS.md` mirrors it, the combinator does NOT
  survive behind a flag (P2), and the feature stops.

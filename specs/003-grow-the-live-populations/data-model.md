# Phase 1 Data Model: Grow the Live Populations

**Baseline**: `c6ca8c8d`. Entities are derived from spec.md Key Entities and Phase 0 research.

## Entity: Live population (`TimeCellClock`)

The only instance in scope. Constructed at `cortex/brain.py:744`, ticked at `:1933`, taught at `:1562`.

| Field | Type | Meaning | Growth-relevant rule |
|---|---|---|---|
| `n_cells` | `int >= 1` | population width | Currently a literal `64`. Becomes mutable. Mutated **last** in any growth operation, because it bounds `tick()`'s loop (`:122`) and sizes `reset()`'s arrays (`:110-111`), so every intermediate state stays in-bounds. |
| `thresholds` | `list[float]`, **strictly ascending** | **the learned state** *and* the structural definition | FR-001: existing entries bit-identical across growth. FR-012: ascending, asserted. Sortedness is *not* currently guaranteed — see Invariant 1. |
| `drive` | `float > 0` | ramp increment per tick | New input guard. Divides the readout; `learn_interval` already divides by it (`:168`, `:182`) while `decode()` does not — the latent unit bug. |
| `noise_sd` | `float >= 0` | ramp noise | Part of the persisted lineage key. |
| `seed` | `int` | RNG lineage | Part of the persisted lineage key. A vector learned under another seed is another lineage's state. |
| `rng` | `random.Random` | per-instance | Growth **must draw no random numbers** — this is what keeps `tests/test_time_cells.py:46` green. |

### Per-trial state (cleared by `reset()`, never persisted)

`v`, `fired`, `spike_tick`, `n_fired`, `t`, `_last_estimate`. These describe an interval in progress.
The interval ended when the process died, so restoring them against a fresh RNG would be a fake.

`n_fired` carries a **dual role**: a count *and* a sorted index into `thresholds`. This is why interior
insertion below the ramp is unsafe and why growth is restricted to a reset boundary.

### Invariants

1. **Ascending order.** `thresholds[i] < thresholds[i+1]` for all `i`. Violated today: 9 of 12 seeds
   invert somewhere by 4000 ticks. An inversion silently strands every higher cell, because `tick()`
   stops at the first uncrossed threshold. Enforced by making `learn_interval` refuse order-breaking
   updates.
2. **Length coherence.** `len(thresholds) == len(fired) == len(spike_tick) == n_cells`.
3. **Growth is RNG-neutral.** `rng.getstate()` identical before and after.
4. **Reset-boundary precondition for growth.** `n_fired == 0 and v == 0.0 and all(s is None for s in
   spike_tick)`.

### State transitions

```
                 refuse (not ascending / not at boundary / degenerate)
                 ┌──────────────────────────────┐
                 │                              ▼
  [width n] ──grow()──▶ [width n+1] ──earns?──▶ [kept]
                                    └──no──▶ [reverted to width n, learned state intact]
```

`grow()` returns the new threshold, or `None` for a refusal. Both outcomes increment a counter; a
refusal is data, not an error (FR-004, FR-011).

## Entity: Learned state record (persisted)

Written via `home.save_state` as `time_cells.json`. **Not** a lake record — see research R4.

| Key | Type | Purpose |
|---|---|---|
| `fmt` | `str` | format tag; mismatch → refuse |
| `n_cells` | `int` | persisted width |
| `seed` | `int` | lineage key; mismatch → refuse |
| `drive` | `float` | lineage key; thresholds are only interpretable relative to it |
| `noise_sd` | `float` | lineage key |
| `thresholds` | `list[float]` | the learned state; `len == n_cells` or refuse |

**Reconciliation on restore**: adopt persisted, wholesale — rebuild the object at the persisted width
and assign the persisted vector. Never mutate `n_cells` in place, never merge element-wise, never
truncate or pad. Refuse (and record the refusal) on: fmt mismatch, length mismatch, lineage mismatch,
`n_cells < 1`, or a non-ascending vector.

Bounding: overwrite **is** the prune policy, so no retention constant exists to tune (P1), and FR-014
is satisfied without depending on `specs/002`'s unbuilt lake prune.

## Entity: Observability record (`state()` additions)

| Key | Type | Satisfies |
|---|---|---|
| `time_cells_width` | `int` | FR-011 — nothing can observe width today |
| `time_cells_grown` | `int` | FR-011 — growth events |
| `time_cells_refused` | `int` | FR-011, SC-007 — makes a halt *observable* rather than inferred from absence |

## Entity: Admission evidence

Produced by `tools/timing_heldout.py`. Not persisted in the brain; recorded as a coverage artifact.

| Field | Meaning |
|---|---|
| teach set / test set | disjoint, gap-major grid over the population's own ladder, positional 1-in-5 stride |
| `d_s` | per-seed `RMSE(grown) − RMSE(ungrown)`; negative means the cell earned |
| `mean_d`, `sd`, `se`, wins, **ties** | paired statistic; ties printed unconditionally |
| null-arm result | insertion at the lowest-teach-residual gap; its measured zero licenses reading the main arm as contribution rather than as a benefit any cell would get |
| readout eligibility | the degenerate-ladder control on `R`; failure → exit 2 `NOT-ELIGIBLE` |

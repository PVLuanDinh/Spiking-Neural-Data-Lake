# Contract: `TimeCellClock` and the growth instrument

**Baseline**: `c6ca8c8d`. This is the interface contract the implementation must honour. It states
behaviour, not implementation.

## 1. `decode() -> (estimate, sd)`

**Changed.** The estimate becomes a population readout.

| Condition | Returned estimate |
|---|---|
| no cell has fired | `0.0` |
| saturated (all cells fired) | top nominal latency |
| otherwise | the geometric centre of the bracket formed by the last-fired and next-unfired nominal latencies |

All nominal latencies are in **ticks**, i.e. divided by `drive`.

`sd` is **unchanged**, byte for byte.

**Contract obligations**

- The estimate MUST change when any bracketing threshold changes. (FR-005)
- The estimate MUST be non-decreasing within an interval — fired cells latch. (`tests/test_time_cells.py:13`)
- Two clocks with the same seed MUST produce identical output. (`tests/test_time_cells.py:46`)
- The module self-check's `final_err < 5.0` MUST hold on the real ladder, and MUST **fail** on a
  degenerate ladder. (Standing Invariant 5)
- No numeric literal may be introduced.

**Known cost**: +45% RMSE against true elapsed at `n_cells=64` (1.0536 vs 0.7276, 16 seeds, horizon
60), closing to +4% at `n_cells=512`. Accepted deliberately as the price of measurability.

## 2. `grow(mode) -> float | None`

**New.**

| Mode | Placement | Purpose | Claimable? |
|---|---|---|---|
| interior | geometric midpoint of the widest log-gap | refine resolution — the only mode that can earn | yes, subject to the instrument |
| extend | `n_cells + 1` | extend range, addressing saturation | **no** — must be reported as clamp-widening under FR-006 |

**Preconditions** — all must hold, else return `None`:

- the vector is strictly ascending
- the clock is at a reset boundary
- the resulting vector would remain strictly ascending

**Postconditions on success**

- every pre-existing threshold is **bit-identical** (FR-001)
- `len(thresholds) == len(fired) == len(spike_tick) == n_cells`
- the vector is strictly ascending, asserted
- `rng.getstate()` is unchanged
- the growth counter increments

**On refusal**: no state changes; the refusal counter increments. A refusal is a normal outcome.

## 3. `learn_interval(...)`

**Changed**: an update that would break ascending order is **dropped**, not clamped. Clamping was
measured to collapse gaps to 1 ULP (2.220e-16 vs 5.298e-03 under refusal) with identical learning
coverage. The `0.5` floor and every other existing term are unchanged.

## 4. Persistence

`save` / `restore` of `{fmt, n_cells, seed, drive, noise_sd, thresholds}` via `home.save_state` /
`load_state`.

- Restore MUST adopt the persisted width wholesale, or refuse.
- Restore MUST refuse on fmt, length, or lineage mismatch, or a non-ascending vector, and MUST record
  the refusal.
- Restore MUST NOT run inside `TimeCellClock.__init__` — the module self-check would read the previous
  gate run's state.
- Restore MUST NOT share `brain.py:736-752`'s exception handler, which nulls five populations.
- Restore MUST be gated so brain construction stays a pure function of its seed.

## 5. `state()` additions

`time_cells_width`, `time_cells_grown`, `time_cells_refused`. Adding keys changes any digest computed
over `state()` — enumerate consumers first.

## 6. `tools/timing_heldout.py`

**CLI**: takes explicit path/param arguments; no value may be computed from no data.

**Exit codes**, matching the repo's existing probe contract:

| Code | Meaning |
|---|---|
| `0` | a verdict was reached — the effect and its dispersion are reported |
| `1` | ran, but no verdict is warranted (effect within its own dispersion) → recorded as an explicit null |
| `2` | `NOT-ELIGIBLE` — the readout failed the degenerate-ladder control or could not see a threshold perturbation. **No verdict is issued.** |

Exit 2 is not a soft 0. A probe that cannot ask the question must not answer it.

**Output discipline**

- print every count as `N of M`, never `N/M` — the artifact fingerprint harvests bare ratios positionally
- print tie counts unconditionally
- name the held-out split, the control, and the saturation regime in the output

**Mandatory arms**: main (interior insertion at the widest gap) and null-control (insertion at the
lowest-teach-residual gap). The null arm's measured zero is what licenses the main arm's reading.

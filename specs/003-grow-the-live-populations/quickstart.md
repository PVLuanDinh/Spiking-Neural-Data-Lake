# Quickstart: validating Grow the Live Populations

**Baseline**: `c6ca8c8d`. Run from the repo root. The deterministic gate env is `.venv-lake`.

## Prerequisites

```bash
python -m pip install -e ".[dev]"
```

## The three gates (must be green before and after every step)

```bash
python run_selfchecks.py
```

```bash
pytest
```

```bash
python -m ruff check . && ruff format --check src tests tools
```

## Scenario 1 — the readout reads its population (FR-005, SC-002)

The defining check of this feature. Perturb one threshold; the consumed estimate must move.

Expected **before** the change: the estimate sequence is bit-identical — max `|estimate − v|` is `0.0`
over 63 unsaturated ticks, and perturbing `thresholds[30]` by `+0.4` changes nothing.

Expected **after**: the estimate sequence differs, and RMSE against true elapsed falls monotonically
with width (measured target: ~1.05 at `n_cells=64` down to ~0.76 at 512).

## Scenario 2 — the degenerate-ladder control (Standing Invariant 5)

The single most important check. Run the module self-check with
`thresholds = [1.0 + 1e-6*i for i in range(64)]`, a ladder carrying no timing information.

**The self-check MUST FAIL.** A readout that passes is not reading its population and is disqualified,
however good its accuracy looks. This control disqualified the leading candidate during Phase 0 —
it scored err 0.995 where the real ladder scores 0.486.

## Scenario 3 — growth is non-destructive (FR-001, SC-001)

Teach a clock, snapshot `thresholds`, grow it, compare.

Expected: all pre-existing thresholds **bit-identical** (compare exact float values, not rounded);
vector strictly ascending; `len(thresholds) == len(fired) == len(spike_tick) == n_cells`;
`rng.getstate()` unchanged.

Expected refusals: growth away from a reset boundary, and growth on an already-inverted vector.

## Scenario 4 — order preservation (FR-012)

Run a taught clock for 4000 ticks in both the reset-driven and saturating regimes.

Expected **before**: 9 of 12 seeds invert somewhere. Expected **after**: 0 of 12, at 200 / 600 / 1200 /
4000 ticks, with learning coverage unchanged (~49.8 of 64 thresholds still move).

## Scenario 5 — the held-out verdict (FR-002, FR-010, SC-004)

```bash
python tools/timing_heldout.py --seeds 12
```

Expected: 12 paired seeds; the main arm and the null-control arm both reported; ties printed
unconditionally; counts printed as `N of M`.

Exit `2` (`NOT-ELIGIBLE`) is the **correct** result until Scenario 1 passes — the instrument refuses to
issue a verdict on a readout that cannot see its population. Exit `1` is an honest null and is a valid
end state for the feature. Exit `0` reports an effect with its dispersion.

## Scenario 6 — recording the artifact (FR-015)

```bash
python tools/coverage_score.py record --kind heldout --name time_cells_growth_heldout -- python tools/timing_heldout.py --seeds 12
```

Note the literal `--` separator; the recorder raises without it. Then verify reproduction:

```bash
python tools/coverage_score.py rerun --name time_cells_growth_heldout
```

Expected: `reproduces`. Be aware that `record` writes no `metric:`/`baseline:` keys, so this artifact
cannot support a band above 40.

## Scenario 7 — restart survival (FR-007, SC-006)

Grow a clock, persist, rebuild the brain, and read `state()`.

Expected: `time_cells_width` returns at the grown width with thresholds intact — or a **recorded
refusal** if the lineage key (`seed`/`drive`/`noise_sd`) or the vector shape disagrees. A silent
partial restore is a defect.

## Scenario 8 — the off-switch (FR-016, SC-009)

With growth disabled, live behaviour must match the pre-feature baseline and all gates must pass with
no assertion loosened.

Note the deliberate exception: the readout change and the `learn_interval` order fix are **not** behind
the switch. They are defect fixes owed regardless, and they do change live behaviour — the readout by
+45% RMSE at the live width, and by fixing the `drive` unit bug (which reported 19.34 for a true
elapsed of 40 at `drive=0.5`) and the saturation clamp (RMSE 292.9 at t=600).

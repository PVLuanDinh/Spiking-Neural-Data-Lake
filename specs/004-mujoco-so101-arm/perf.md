# Arm performance, determinism and environment (T073, T074, T076, T077)

Measured 2026-08-01 with `scratchpad/perf.py`.

| | |
|---|---|
| platform | Windows AMD64, CPython 3.11.15 |
| mujoco | 3.11.0 |
| model timestep | 0.002 s |

## T074 — per-tick arm cost

| substeps per `apply()` | ms / apply | ms / physics step |
|---|---|---|
| 1 (default) | 0.050 | 0.0502 |
| 10 | 0.209 | 0.0209 |
| 50 | 0.918 | 0.0184 |
| 200 | 3.707 | 0.0185 |

`state()` costs **0.021 ms**.

**The tick rate is not at risk.** The daemon's tick was measured at ≥240 ms under load
(2026-07-30), so even at 200 substeps the arm is ~1.5 % of a tick, and at the default of 1 it is
0.02 %. Per-step cost falls with batch size (0.050 → 0.018 ms) because the fixed Python-call
overhead amortises across the loop inside `apply()`.

Note the interaction with the brain-owned actuation rate: the cost of a tick now depends on a
value the brain adapts. The ceiling is one simulated second (500 steps ≈ 9 ms), so the worst
case is bounded and still small against the tick.

## T073 — determinism

A fixed 4000-step trace (seeded ctrl changes every 50 steps), hashed over `qpos` + `qvel`:

```
sha256 90518454da0ee008290aadbef4f9def7ad6037c13a1a79617120fc4a33e7be44
final qpos [-0.327639635, 0.23388347, 0.263985081, -0.201499004, -0.179642058, -0.174523773]
```

**Only one matrix cell is measured, and that is stated rather than glossed.** The task asks for
a comparison across the CI matrix; this machine is one of its cells. The hash above is the
reference value — a cross-OS claim requires the other cells to produce it, and until they run,
"deterministic across OSes" is a hypothesis, not a result. Re-run `scratchpad/perf.py` on each
cell and compare this digest.

What *is* established locally: the run is reproducible on this cell, and the planner, the
reach sweep and the coverage sweep are all seeded so they reproduce too.

## T076 — MUJOCO_GL

An unrecognised value **raises at import**, before any arm exists:

```
RuntimeError: invalid value for environment variable MUJOCO_GL: nonsense_value
```

This is MuJoCo's own behaviour and needs no code here. It is the right shape: the message names
the variable and the bad value, and it fails at startup rather than at the first attempt to
draw. Unset (the default on this machine) works, and offscreen rendering succeeds.

A *usable* MUJOCO_GL that still cannot obtain a context is a different case and is handled, not
raised: `substrate/arm_view.py` catches it and reports `arm_view_available: false` with the
reason. The daemon must keep thinking when it cannot see.

## T077 — mujoco resolves in both environments

| env | role | mujoco |
|---|---|---|
| `.venv-lake` | deterministic CPU gate (pytest, self-checks) | 3.11.0 |
| `.venv-cu128` | the daemon | 3.11.0 |

Same version in both, so the gate and the daemon are running the same physics. This matters
more than it looks: the arm is now a core dependency of the brain, so a version skew between
the environment that *proves* the safety envelope and the environment that *runs* it would mean
the proof did not cover the thing that runs.

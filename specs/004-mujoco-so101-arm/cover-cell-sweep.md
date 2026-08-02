# Coverage-cell sweep (T053)

**Measured 2026-08-01.** 60 000 poses sampled uniformly over the joint limits, TCP placed with
`mj_kinematics` (no integration, no contact solve). Reproduce with
`scratchpad/cover_sweep.py`.

## Workspace

| quantity | value |
|---|---|
| TCP span x / y / z | 0.8143 / 0.8805 / **0.7467** m |
| max horizontal radius | 0.4784 m (60 k samples) |
| `arm_config.MAX_REACH_MM` | 0.4771 m (4 k samples, deterministic, seed 0) |

The two reach figures differ by 0.27 %, which is sampling density. That is acceptable **because
this is a scale, not a gate**: a slightly low reach makes cells slightly finer and changes no
decision. It would not be acceptable for a limit.

## Cells reachable, by cell size

| cell (m) | 2-D cells | 3-D cells | 3-D / 2-D | |
|---|---|---|---|---|
| 0.100000 | 67 | 416 | 6.2 | incumbent's value |
| 0.050000 | 248 | 2 500 | 10.1 | |
| 0.030000 | 654 | 9 598 | 14.7 | |
| **0.023853** | **1 003** | **16 525** | **16.5** | **derived — reach / 20** |
| 0.020000 | 1 416 | 23 401 | 16.5 | |
| 0.010000 | 5 319 | 48 053 | 9.0 | |

## Why the incumbent's value could not be carried over

At 0.10 m the whole workspace is **67 planar cells**. The old arm reached 2.0 m, so 0.10 m was
1/20th of its reach; on a body reaching 0.478 m the same length is more than a fifth of the
workspace per cell. The coverage reward would have saturated almost immediately and then read
as "explored everything" for the rest of every run — a metric that stops discriminating is
worse than no metric, because it still produces a number.

What transfers is the incumbent's **resolution** (twenty cells per reach), not the length it
happened to equal. Hence `_COVER_CELLS_PER_REACH = 20` and a cell derived from
`MAX_REACH_MM`.

## Why the cell is 3-D

| joint (moved alone from home) | \|dz\| | \|dxy\| | dominant |
|---|---|---|---|
| shoulder_pan | 0.0000 | 0.5775 | horizontal |
| **shoulder_lift** | **0.4462** | 0.2699 | **vertical** |
| elbow_flex | 0.2890 | 0.3318 | horizontal |
| wrist_flex | 0.1500 | 0.1810 | horizontal |
| **wrist_roll** | **0.0154** | 0.0027 | **vertical** (negligible magnitude) |

**Correction to the task text.** T053 states "three of five rotate joints move the TCP mainly
in z". Measured, it is **two**, and one of those (`wrist_roll`, 15 mm) is too small to matter —
so the real count is effectively one. The conclusion is unaffected and if anything sharper:
`shoulder_lift` alone displaces the TCP **0.4462 m vertically**, the z-span is **0.7467 m**, and
a planar cell cannot observe any of it. At the derived cell size 3-D distinguishes **16.5×** as
many positions as 2-D.

The cell-count figures in T053 (791 and 12 741 at 0.020 m) also do not reproduce: this sweep
measures **1 416** and **23 401**. The earlier numbers appear to have been taken over a smaller
or differently-conditioned pose sample. The direction of the argument is unchanged.

## Not swept here

Cell size interacts with the reward only through how often a *new* cell is entered per tick,
and that depends on how far one command moves the arm — which is now a rate the brain adapts
rather than a constant (see the actuation-cadence decision in `spec.md`). A sweep of
cells-entered-per-tick is therefore only meaningful once that rate has been shown to converge,
which it has not yet been. Flagged rather than estimated.

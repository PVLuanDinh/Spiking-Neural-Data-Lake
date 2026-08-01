# Quickstart: validating the MuJoCo SO-101 arm

**Baseline**: `c6ca8c8d`, `mujoco==3.11.0` (already installed in `.venv-cu128` and `.venv-arm`).
Run from the repo root.

## Prerequisites

```bash
python -m pip install -e ".[dev]"
```

The daemon env is `.venv-cu128`; the deterministic gate env is `.venv-lake`. CI never renders.

## The three gates (green before and after every step)

```bash
python run_selfchecks.py
```

```bash
pytest
```

```bash
python -m ruff check . && ruff format --check src tests tools
```

Baseline for comparison: **277/277 self-checks**, pytest exit 0 at **92.53%** coverage, ruff clean.
The self-check count must still read 277 after the swap — delete the old arm and add the new one in the
same commit, or seven documents quoting that number go stale.

## Scenario 1 — the safety kernel still names the arm (blocking)

The highest-severity check in this feature, and the cheapest.

Assert every path in `_SAFETY_KERNEL` (`godel_machine.py:148-156`) exists on disk, and that the same
list at `self_improve.py:157-163` agrees.

Expected: pass. If the arm module was renamed without updating both literals, this fails — which is the
entire point, because otherwise the arm silently leaves the brain's immutable set while `NORD_SELFMOD=1`
is live and self-build fires every 8 nights.

## Scenario 2 — E-STOP latches, and is idempotent

Apply `EMERGENCY_STOP` once; then apply it 1000×.

Expected: state after 1000 applications is **bit-identical** to state after one. Unguarded, this drifts
**35.16°** over 20 000 repeats versus **0.0148°** latched once. The live proximity path re-issues the
command every tick with no guard, so this is a real regime, not a synthetic one.

Also expected: while latched, every command except `HOME`/`RETRACT_ALL` is a no-op; stepping continues;
sag stays ≤ 0.0256° / 0.175 mm and is stationary.

## Scenario 3 — HOME is bitwise reproducible

Drive the arm to several different poses, then `HOME` from each.

Expected: identical end-effector position every time, so an equality assertion in the shape of the
existing one survives. Reference: `EE ≈ (0.391362, −0.0000113, 0.226469)` m, `ncon=0`, `qvel=0`.

## Scenario 4 — the collision contract, both halves

**Live**: command the arm into the floor.
Expected: contact reported **and** the achieved joint angle short of the commanded one — commanded
1.745 rad, achieved 0.760, blocked by 0.985, contact on 4907/5000 ticks. This assertion is strictly
stronger than the one it replaces.

**Probe**: query a penetrating pose.
Expected: contact reported, `qpos` bit-unchanged, `qvel` zero — report without prevent. Reference
`ncon=24`, min distance −0.2323 m.

**Probe, out-of-limit pose**: expected a distinct out-of-limit verdict, not FREE. Without this, 65.9% of
the planner's grid is falsely labelled free.

## Scenario 5 — torque, and the question that started this (SC-002)

Read per-joint hold torque at a commanded pose, with and without a 500 g payload at the TCP.

Expected: N·m per joint against the ±3.350 N·m actuator limit, with any joint over its rating named.
Cross-check against what the repo already computes — `hardware` reports `shoulder_pitch` needing
15.0 kg·cm against 7.5 rated (`margin_x: 0.5`, `ok: false`). Corroboration strengthens the result; a
discrepancy is itself a finding and must be explained, not averaged away.

## Scenario 6 — clamping is visible

Command a joint past its limit.

Expected: the gate clamps, `clamped` is true, and the reported command matches what the arm actually
received. MuJoCo's own clamp is silent — `ctrl` stays at the out-of-range value while the body sits at
the rail — so a body-side clamp would report a command that never happened.

## Scenario 7 — the viewer

```bash
curl -o arm.png http://127.0.0.1:8137/api/arm.png
```

Expected: a decodable PNG, or a non-200 with a reason. Open the cockpit and confirm the panel shows the
pose **and a revision counter** — without the counter a still image is indistinguishable from a live one.

Render must happen on the daemon's tick thread. Cross-thread rendering returns *garbage frames* with only
a warning, and a lock does not fix it, so "it returned an image" is not evidence of correctness.

## Scenario 8 — no GL context (FR-012b)

Run with `MUJOCO_GL=disable`.

Expected: the arm still simulates, the viewer reports unavailable, and **the daemon keeps running**.
This is the CI configuration, so all six matrix cells exercise it for free.

## Scenario 9 — determinism

Two runs, same seed, same command sequence.

Expected: identical arm state. **Cross-OS bit determinism is currently UNVERIFIED** — every measurement
so far is one Windows binary on one CPU, and MuJoCo makes no cross-platform bit-identity guarantee.
Measure before asserting: hash a fixed 4000-step `qpos`/`qvel` trace on each matrix cell and compare.

## Scenario 10 — exactly one arm (SC-010)

Search the tree for a second arm implementation.

Expected: one. No flag selects between two — a flag that keeps both paths reachable is the failure P2
exists to prevent. Note `substrate/demo.py` carries a deliberately dependency-free standalone `Arm`;
decide explicitly whether it is allow-listed or deleted.

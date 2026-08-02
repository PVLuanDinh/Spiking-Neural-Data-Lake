# Contract: the SO-101 arm, its gate, and the viewer

**Baseline**: `c6ca8c8d`, `mujoco==3.11.0`. Behaviour, not implementation.

## 1. `apply(command) -> body`

The single mutation path. Ordered checks:

1. **Latch check** — if `stopped` and the command is not `HOME`/`RETRACT_ALL`, return current state
   unchanged. No stepping side effect, no re-latch.
2. **Vocabulary** — reject any token not in the joint table or the fixed set. A joint name that does not
   exist is rejected, never silently ignored.
3. **Clamp** — clamp the target to the joint's limits, set `clamped` accordingly. The gate clamps;
   the body must not be relied on to.
4. **Actuate** — write `ctrl`, step the body.
5. **Report** — compute `reached`, `blocked`, contact verdict, `tracking_error`, `deviation`.

### `EMERGENCY_STOP`

- Sets `stopped = True`. **Guarded**: the latch action (`ctrl := qpos`) runs only on the `False → True`
  edge.
- **MUST be idempotent.** Applying it 1000× MUST be bit-identical to applying it once. Unguarded, 20 000
  repeats drift 35.16° against 0.0148° latched once.
- Stepping CONTINUES while latched. Stopping the stepping leaves stored velocity that discharges the
  moment anything resumes.
- Held-latch sag is bounded and stationary: ≤ 0.0256° / 0.175 mm.

### `HOME` / `RETRACT_ALL`

- Hard reset: `mj_resetData`, `qpos := HOME_QPOS`, `qvel := 0`, `qacc := 0`, `ctrl := HOME_QPOS`,
  `mj_forward`. **Not** "command zeros and settle".
- Clears the latch.
- MUST be bitwise reproducible, so an equality assertion in the shape of the existing one survives.
- Known side effect to handle: resets `d.time` to 0.

## 2. Contact verdict

`None | "GROUND" | "SELF"`. `OBSTACLE` is retired — the vendored scene has one geom.

- **Live**: reads `ncon` off the stepped body. Reports contact the solver already prevented.
- **Probe** `probe_contact(qpos)`: private never-stepped body, `mj_kinematics` + `mj_collision`.
  Reports without preventing. MUST NOT be reachable from the live body.
- Labels resolve through `geom_bodyid` body names.
- The probe MUST gate on joint limits and return a distinct out-of-limit verdict.

## 3. `reached` / `blocked`

- `reached = max|qpos − ctrl| <= REACH_TOL`, where `REACH_TOL` is derived above the measured clean droop
  floor (≤ 0.0492°), not picked.
- `blocked = (not reached) and (contact or force saturation)`.
- An unsatisfiable command is a normal outcome, not an error.
- `blocked` MUST have a consumer. A verdict computed at 28% frequency and read by nobody is dormancy.

## 4. `tracking_error` vs `deviation`

Two quantities, because a position servo makes them different:

| quantity | meaning | property |
|---|---|---|
| `tracking_error` | expected servo droop | small, bounded, displayed |
| `deviation` | excursion attributable to external perturbation | reflex-bearing |

Neither existing assertion may be loosened. Any threshold that changes MUST be justified in-diff as a
re-derivation with its measurement cited.

## 5. Joint limits

Read from the model at import. `arm_config` owns policy, not a copy of the numbers. The gate clamps.

## 6. Safety kernel registration — blocking

Any rename or deletion of the arm module MUST, **in the same commit**, update all five sites:
`__init__.py` REGIONS, `run_selfchecks.py`, the dormancy gate, `godel_machine.py:151`, and
`self_improve.py:160`.

A test MUST assert every path in `_SAFETY_KERNEL` exists on disk. Without it, deleting the file silently
removes the arm from the brain's immutable set while self-build is live.

## 7. Real-hardware path

- The calibration gate keeps its no-bypass property. With calibration unset, no command reaches the wire
  by any path.
- Wire validation stays independent of the simulator's state.

## 8. Viewer

- `GET /api/arm.png` returns `image/png` bytes, or a non-200 with a reason when unavailable.
- The renderer is created and used on the daemon's tick thread **only**. Cross-thread rendering returns
  garbage frames with only a warning, and a lock does not fix it.
- Encoding is stdlib `zlib`+`struct`. PIL MUST NOT be used — it is present but undeclared.
- The panel MUST carry a revision counter. A still picture is otherwise indistinguishable from a live
  one, which is the failure `PROVENANCE` exists to prevent.
- If no GL context exists, the viewer reports unavailable and **the daemon keeps running**.

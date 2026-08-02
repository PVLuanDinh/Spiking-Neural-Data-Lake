# Phase 1 Data Model: MuJoCo SO-101 Becomes The Arm

**Baseline**: `c6ca8c8d`, `mujoco==3.11.0`. Entities derive from spec Key Entities and Phase 0.

## Entity: Arm body (`So101Arm` — the gate that owns a body)

| Field | Type | Notes |
|---|---|---|
| `model` | MjModel | loaded once from the vendored `so101/scene.xml` |
| `data` | MjData | the LIVE body; stepped every tick |
| `stopped` | bool | E-STOP latch. Guarded: latch action runs only on the `False → True` edge |
| `n_joints` | int | 6, from the model — never a literal |

**Invariants**

1. `data` is the only stepped body. The probe body is module-private and never stepped.
2. The latch is idempotent — applying `EMERGENCY_STOP` 1000× is bit-identical to applying it once.
3. `qpos` is never assigned from outside the gate. Teleporting the live body into a penetrating pose and
   resuming stepping ejects it at 36.11 rad/s.
4. Every command is clamped to `JOINT_LIMITS` by the gate *before* reaching `ctrl`, because MuJoCo's own
   clamp is silent and would let `state()` report a command the arm never received.
5. HOME is a hard reset (`mj_resetData` + `HOME_QPOS` + `mj_forward`), bitwise reproducible.

## Entity: Joint table (`arm_config.JOINTS`) — the single vocabulary

One ordered row per joint. **Limits are read from the model at import; only alias/servo/lever are
hand-written.** A model bump therefore cannot silently desynchronise limits from vocabulary.

| Column | Source |
|---|---|
| `qpos_index` | position in the table |
| `name` | MJCF joint name — canonical identity, resolved by `mj_name2id` |
| `alias` | the `arm_hardware` build-target name (`base_yaw`, …) |
| `token` | `JOINT_<NAME_UPPER>_ROTATE` |
| `lo_deg`, `hi_deg` | **read from `jnt_range`**, not transcribed |
| `servo`, `lever_mm` | hand-written build-target data |

`arm_hardware.JOINTS` becomes a derivation of this table, not a second literal.

## Entity: Command

A token from a closed vocabulary. **The regex is syntax; the table is vocabulary** — the joint-name
group must be checked for membership, so a regex-only site cannot admit a joint that does not exist.

| Kind | Form |
|---|---|
| joint motion | `JOINT_<NAME>_ROTATE(<±d>deg)` |
| effector | `ENGAGE`, `RELEASE` |
| latch | `EMERGENCY_STOP` |
| latch-clearing | `HOME`, `RETRACT_ALL` |
| no-op | `HOLD` — deliberate asymmetry, on the wire allow-list only |

## Entity: `body` snapshot — THE contract

Fixed in plan.md; restated here as the authority for producers and consumers.

| key | type | producer | consumers |
|---|---|---|---|
| `theta_deg` | list[6] float | gate | telemetry, cockpit |
| `ee_mm` | tuple[3] float | gate | cockpit, planner |
| `effector` | float 0..1 | gate | telemetry, cockpit |
| `stopped` | bool | gate | `autonomous_loop`, cockpit, `serial_arm` |
| `collision` | `None`\|`"GROUND"`\|`"SELF"` | verdict fn | `autonomous_loop:1283`, `cockpit:452,562`, `brain_telemetry:111` |
| `reached` | bool | gate | action selection |
| `blocked` | bool | gate | action selection |
| `clamped` | bool | gate | telemetry |
| `tracking_error` | float m | gate | telemetry, cockpit |
| `deviation` | float m | gate | `spiking_pid`, `ring_attractor`, `live_arm` |
| `backend` | str | gate | telemetry |
| `moves` | bool | gate | telemetry honesty — now **true** |

**Rulings.** `collision` keeps its name and `None | str` type — three live consumers and a registered
telemetry key depend on both. `OBSTACLE` is retired: the vendored scene has one geom (`floor`), so the
branch can never fire. `deviation` splits from `tracking_error` because a position servo makes them
different quantities and the old assertions cannot otherwise survive without loosening.

## Entity: Contact verdict

Computed from `data.ncon` and `contact[i].geom1/geom2`, resolved to **body names via `geom_bodyid`**
because 30 of 31 geoms are unnamed.

| Path | Data | Semantics |
|---|---|---|
| live | stepped `MjData` | reports contact the solver already prevented |
| probe | private, never-stepped `MjData` + `mj_kinematics` + `mj_collision` | reports without preventing |

The probe additionally gates on `jnt_range` and returns a fourth verdict for out-of-limit poses —
without it, 65.9% of the planner's grid is falsely FREE.

## Entity: Torque reading — new capability

| Field | Source |
|---|---|
| `qfrc_bias` | gravity + Coriolis hold torque, per joint, N·m |
| `actuator_force` | commanded actuator force, N·m |
| `forcerange` | ±3.350 N·m — the STS3215-at-12V match |
| payload | 500 g body welded at the gripper TCP site |

This is what `ArmSim` structurally cannot produce and the reason the replacement is worth its risk.

## Entity: Frame (viewer)

| Field | Notes |
|---|---|
| `png` | stdlib `zlib`+`struct`, RGB, ~1.90 ms encode |
| `pose_rev` | monotonic counter — without it a still picture is indistinguishable from a live one |
| `available` | False when no GL context; daemon must survive |

**Ownership invariant**: the renderer is created and used on the daemon's tick thread only. Cross-thread
use returns garbage frames with a warning, and a lock does not fix it.

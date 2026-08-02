# Port-before-retire: what ArmSim could do, and where it went (T065)

P2 requires that every unique capability of the retired mechanism be ported **before** deletion,
and that each be recorded. This is that record. `brainstem/arm_sim.py` was deleted in the same
change that updated all five registration sites (T066).

## Capabilities, one row each

| ArmSim capability | Disposition | Where it went |
|---|---|---|
| `apply(command)` — the one mutation path | **PORTED** | `So101Arm.apply()`, same ordering: latch → vocabulary → clamp → actuate → report |
| E-STOP latch (`stopped`, cleared only by HOME/RETRACT_ALL) | **PORTED, strengthened** | `So101Arm`; returns *before* `mj_step`, so a latched arm is bitwise frozen rather than merely un-commanded |
| `collision()` — reports without preventing | **PORTED** | `So101Arm.collision()`; MuJoCo's solver keeps the links solid, but the command still executes and the contact is still reported |
| `state()` | **PORTED, widened** | adds `ee_mm`, `reached`, `blocked`, `clamped`, `tracking_error`, `reach_mm` |
| `deviation()` | **PORTED, redefined** | measured against the settled pose rather than the command — contact no longer leaks into it (T052) |
| `trajectory_breach()` | **PORTED, re-derived** | per-joint against the model's own droop envelope, not the 2 m-reach constant `TRAJ_DEV_MAX` (T051) |
| `close()` | **PORTED as a no-op** | nothing to release; documented, because dropping the model would break a body that is still usable |
| `effector` | **PORTED, changed meaning** | was a flag that snapped to 1.0; is now a physical closure ratio on a real gripper joint, clamped to [0, 1] |
| `theta` (2 entries, **radians**) | **RENAMED** → `state()["theta_deg"]` (6 entries, **degrees**) | no shim, deliberately: a compat property would let every unmigrated reader silently get the wrong unit and joint count |
| `ee()` (2-D, dimensionless) | **RENAMED** → `ee_mm()` (3-D, millimetres) | same reasoning |
| `perturb(da_deg=...)` | **RENAMED** → `inject_disturbance(joint, deg)` | renamed so it cannot be mistaken for a control path; test-only (Q2) |
| `expected_ee` | **RETIRED** | referenced only inside arm_sim and the grid planner that specs/004 replaced. The successor concept is `_commanded`, which the new `deviation`/`trajectory_breach` already use |
| `l1`, `l2` (link lengths) | **RETIRED** | properties of a 2-link planar arm; the SO-101's geometry comes from the MJCF |
| `pybullet_contacts()` | **RETIRED — dead on arrival** | defined once, **called nowhere** in `src/`, `tools/` or `tests/`. Nothing to port: MuJoCo's own contact solver supersedes the optional backend entirely |

## Things that were NOT ArmSim capabilities, and are not claimed as ports

- Mass, gravity, torque, integration. ArmSim had none — `JOINT_*_ROTATE` teleported. These are
  new, and the benchmark is where the gain is measured, not this table.

## Collateral retirements

- **`tests/test_arm_sim.py`** — deleted with its subject.
- **`tools/arm_benchmark.py`'s incumbent half** — `probe_armsim()` removed. The A/B that
  licensed the swap already ran and is recorded in `benchmark.md`; re-printing its numbers as
  literals would turn a measurement into a transcription no run could refute.
- **`substrate/demo.py`'s private `Arm`** — a *third* implementation with its own `apply()`,
  `ee()`, `effector`, `stopped` and its own untested E-STOP. Its justification was
  dependency-freedom, which the project dropped in 2026-07. Now raises on construction; the demo
  drives the one real arm (T071).
- **`substrate/reaction_diffusion_compute.py`** — not deleted, marked `"retire"` in
  `dormancy_gate.EXEMPT`. Its only consumer was the grid planner. Recorded rather than deleted
  in the same change, so that deletion is a decision on its own merits.

## The five registration sites, all moved in one change

1. `snn_data_lake/__init__.py` REGIONS
2. `run_selfchecks.py` CORE list
3. `reasoning/godel_machine.py` `_SAFETY_KERNEL`
4. `reasoning/self_improve.py` `PROTECTED`
5. `reasoning/self_mod_harness.py` pinned count — back to **9**, exactly as that assertion's own
   text said it would be when arm_sim retired

Plus `cortex/quality_diversity.py` and `cortex/self_improvement_suite.py`, whose `forbidden`
tuples named `arm_sim` as a string and now name `so101_arm`.

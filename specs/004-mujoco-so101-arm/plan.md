# Implementation Plan: MuJoCo SO-101 Becomes The Arm

**Branch**: `004-mujoco-so101-arm` | **Date**: 2026-08-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/004-mujoco-so101-arm/spec.md`

**Baseline commit for every figure**: `c6ca8c8d`. `mujoco==3.11.0` is installed in `.venv-cu128`
(the daemon env) and `.venv-arm`; SO-ARM100 is cloned at `D:\AI-project\SO-ARM100`. Every number below
was produced by running code on this machine unless marked otherwise.

## Summary

Replace the 2-DOF kinematic `ArmSim` with a MuJoCo SO-101 body, keep every safety property, migrate
every consumer, and show the arm in the cockpit.

Phase 0 changed the shape of this in five ways:

1. **Report-without-prevent survives, but only as a probe.** All three flag routes to "let MuJoCo
   penetrate while still reporting" are measured dead. The working answer splits the contract by call
   site: the live body reports contact the solver already prevented; the planner queries a private,
   never-stepped body that reports without preventing.
2. **The brain can currently rewrite the E-STOP, and this feature can silently widen that.**
   `_SAFETY_KERNEL` is a literal path list naming `arm_sim.py`. Deleting that file without editing two
   separate literals in the same commit removes the arm from the brain's immutable set while
   `NORD_SELFMOD=1` is live. This is the highest-severity item in the plan and it is a registration
   problem, not a design problem.
3. **The E-STOP as first designed ratchets.** The live proximity path re-issues `EMERGENCY_STOP` every
   tick with no already-stopped guard; an unguarded latch drifts **35.16°** over 20 000 repeats versus
   **0.0148°** when latched once — ~2400×.
4. **Three research areas produced three incompatible `body` dicts.** The plan fixes the contract once,
   below, before any task is written.
5. **Two green assertions cannot survive as written**, and loosening them is forbidden. They need a
   split quantity, not a bigger threshold.

## Technical Context

**Language/Version**: Python 3.11 / 3.12 / 3.13 (CI matrix)

**Primary Dependencies**: `mujoco==3.11.0` becomes core (exact pin — the solver version *is* the
physics, and SC-012 plus every torque assertion are numeric asserts against one build). Brings seven
transitive packages. Verified: wheels exist for all six CI cells; no sdist compile.

**Storage**: no new store. Model assets vendored into the package.

**Testing**: `run_selfchecks.py`, `pytest` (>=70% cov), `ruff check`, `ruff format --check`.

**Target Platform**: ubuntu-latest + windows-latest, CPU. CI never renders.

**Project Type**: single Python package plus `tools/` probes.

**Performance Goals**: measured 2000 steps in 0.051 s (~105× realtime); render 1.84–2.07 ms/frame;
stdlib PNG encode 1.90 ms. Simulation is not a bottleneck.

**Constraints**: no new tuned constant (P1); same-seed determinism; no assertion loosened or silenced;
exactly one arm survives (P2); the five safety-envelope modules change only under the owner
authorisation already recorded in the spec.

**Scale/Scope**: one arm, 2 DOF → 6 DOF, dimensionless → metres.

## Constitution Check

*GATE: re-evaluated post-design.*

| Principle | Assessment | Verdict |
|---|---|---|
| **P0** self-improvement | Gives the brain a body with real torque, closing a gap the repo itself documented. Honesty guards are load-bearing: the two "sim is a toy" assertions get rewritten to state what is newly true, never removed. | **PASS** |
| **P1** no-hardcode | Joint limits are read from the model at import, not transcribed, and the limit of record is now named (`actuator_ctrlrange` for clamping, `jnt_range` for the probe). `HOME_QPOS` is repo-declared because the model has no keyframe. Three existing tuned constants (`TRAJ_DEV_MAX`, `_COVER_CELL`, `REACH_TOL`) are re-derived with published sweeps; the first two are currently *wrong*, not merely stale, and `_COVER_CELL` additionally changes dimensionality. | **PASS, with three re-derivations owed** |
| **P2** adopt-better, non-additive | One arm survives; no flag selects between two. Eligibility requires the benchmark (US1) that does not exist — sequenced first. Port-before-retire enumerated below. | **PASS, gated on US1** |
| **P3** no dormancy | Real risk: a `blocked` verdict computed at 28% frequency and read by nobody is dormancy with extra steps; likewise an `OBSTACLE` label that can never fire. Both are given consumers or retired explicitly. | **PASS, with two consumers owed** |
| **P4** one brain | No new Brain, no second frontend, cockpit still never constructs one. | **PASS** |
| **P5** GPU | Unaffected. MuJoCo physics is CPU; measured crossover keeps the live tick off the GPU. Rendering uses GL incidentally. | **PASS** |
| **P6** reach-and-be-used | The arm is consumed by the autonomous loop, telemetry, planner and cockpit. | **PASS** |
| **SI-2** never silence a check | Two assertions (`deviation() < 1e-9`, `< 1e-6`) cannot hold under a servo. They are **split into two quantities**, not loosened. Any threshold change is justified in-diff with its measurement. | **PASS by design** |
| **SI-5** verify the verifier | The planner probe is measured blind to joint limits (65.9% of grid cells falsely FREE) and gains a fourth verdict. | **PASS** |

**No violations requiring justification.** Complexity Tracking is empty.

## The `body` contract — fixed once, before any task

Three research areas proposed three incompatible dicts. This is the contract; tasks build to it.

| key | type | meaning | consumers |
|---|---|---|---|
| `theta_deg` | list[6] float | joint angles, degrees, MJCF order | telemetry, cockpit |
| `ee_mm` | tuple[3] float | TCP position, **millimetres** | cockpit, planner |
| `effector` | float 0..1 | gripper closure | telemetry, cockpit |
| `stopped` | bool | E-STOP latch | `autonomous_loop`, cockpit, `serial_arm` |
| `collision` | `None` \| `"GROUND"` \| `"SELF"` | contact verdict — **name and type unchanged** | `autonomous_loop:1283` (`== "GROUND"`), `cockpit:452,562`, `brain_telemetry:111` |
| `reached` | bool | commanded angle achieved within tolerance | new; consumed by action selection |
| `blocked` | bool | not reached **and** (contact or force saturation) | new; consumed by action selection |
| `clamped` | bool | command was outside joint limits and clamped by the gate | new; consumed by telemetry |
| `tracking_error` | float m | expected servo droop, always small | telemetry, cockpit |
| `deviation` | float m | perturbation-only excursion | `spiking_pid`, `ring_attractor`, `live_arm` |
| `backend` | str | `"mujoco"` | telemetry |
| `moves` | bool | joints actually change on the tick — now **true** | telemetry honesty |

**Rulings this encodes.** `collision` is NOT renamed to `contact` — three live consumers and a
registered telemetry key depend on the name and on `None | str` truthiness. `OBSTACLE` is **retired**
as a label: the vendored `scene.xml` worldbody contains exactly one geom (`floor`), so the branch could
never fire, and a branch that cannot fire is the computed-but-not-consumed defect. Retiring it requires
updating `cockpit.py:562` and `cpg_oscillator.py:62`, and **`wavefront_planner.py:299-300` asserts an
obstacle exists** — that assertion is currently green and must be handled in the same change (see
Risks). `deviation` splits from `tracking_error` because a position servo makes them different
quantities: droop is expected and bounded, perturbation excursion is the reflex-bearing signal.

## Project Structure

```text
specs/004-mujoco-so101-arm/
├── spec.md  plan.md  research.md  data-model.md  quickstart.md
├── contracts/so101_arm.md
├── checklists/requirements.md
└── tasks.md                      # /speckit-tasks, not created here

src/snn_data_lake/brainstem/
├── so101/                        # VENDORED, Apache-2.0: scene.xml, so101_new_calib.xml,
│                                 #   assets/*.stl, SO-ARM100-LICENSE  (15 files, 16.14 MB)
│                                 #   NOT named models/ or vendor/ -- both are gitignored and
│                                 #   hatchling honours .gitignore, so the wheel would silently
│                                 #   omit them while CI stayed green
├── so101_arm.py                  # NEW: the gate that owns a body
└── arm_sim.py                    # DELETED in the same commit (P2, and the 277 self-check count)

src/snn_data_lake/brainstem/arm_config.py   # gains JOINTS table + JOINT_LIMITS read from the model
tools/arm_benchmark.py                      # NEW: US1 A/B + torque + payload
```

**Structure Decision**: single project, existing layout. Assets vendored under the package so no
absolute out-of-repo path survives; hatchling already ships non-`.py` files under `src/snn_data_lake`,
so **no packaging change is required** — but the wheel is never built in CI today, so that trap has no
detector and one is added.

## Execution Order

| # | Work | Why here |
|---|---|---|
| 0 | **Safety-kernel registration**, in the same commit as any file rename/delete | Highest severity. Five sites, not three — `__init__.py` REGIONS, `run_selfchecks.py`, dormancy gate, **`godel_machine.py:151`**, **`self_improve.py:160`**. Plus a test asserting every `_SAFETY_KERNEL` path exists on disk, so this cannot silently recur. |
| 1 | Vendor assets + make `mujoco` core + CI `MUJOCO_GL=disable` + wheel-build smoke step | Everything else imports the model. |
| 2 | `arm_config.JOINTS` + `JOINT_LIMITS` from the model | The single vocabulary and limit source. |
| 3 | US1 benchmark (`tools/arm_benchmark.py`) — torque, 500 g payload, A/B | P2 eligibility. See the circularity note below. |
| 4 | US2 the gate: latch, HOME, clamping, contact verdict, probe | Safety core. |
| 5 | US3 migration: token grammar, **eight** producer/validator sites, consumers | The substance. |
| 6 | US4 cockpit panel + `/api/arm.png` | Visible payoff. |
| 7 | US5 retire `ArmSim`, port-before-retire complete, **replace the planner** | P2 closure. The planner replacement (FR-021) lands here because it consumes the probe from step 4 and the vocabulary from step 5. |

**On US1's apparent circularity**: the benchmark compares the two arms, so it appears to need the new
arm first. It does not need the *gate*. Step 3 builds the benchmark against a bare MuJoCo body and the
existing `ArmSim`, measuring what each can and cannot report. The gate (step 4) is what the benchmark
licenses, and it is what US5 retires `ArmSim` in favour of.

## Key design rulings (all measured)

- **Frozen = latch a bool + `ctrl[:] = qpos` ONCE + keep stepping.** Not "stop stepping" (leaves
  5.45 rad/s stored, discharged the instant anything resumes), not `ctrl[:] = 0` (on a position
  actuator that is a *move-to-zero command* — 25.89° of slew during the E-STOP itself), not limp
  (38.32° collapse under gravity). Held-latch sag is bounded and stationary at 0.0256° / 0.175 mm.
- **The latch must be guarded** (`if not self.stopped`) and provably idempotent. `autonomous_loop.py:1163-1172`
  re-issues `EMERGENCY_STOP` every tick with no guard; unguarded, 20 000 repeats drift 35.16°.
- **HOME = hard reset, not settle.** The model has no keyframe (`nkey=0`), so `HOME_QPOS = zeros(6)` is
  repo-declared. Reset gives bitwise-identical results across trials, so an equality assert in the exact
  shape of `arm_sim.py:201` survives. "Command zeros and settle" does not — it lands 0.031° off and is
  not bitwise reproducible across start poses.
- **Clamping belongs in the gate, not the body.** MuJoCo's clamp is silent: `d.ctrl[0] = 5.0` against a
  ±1.92 range leaves `ctrl` unmutated at 5.0 while the body sits at the rail — so a body-side clamp
  would let `state()` report a command the arm never received, which is exactly what FR-009 forbids.
- **Contact labels come from `geom_bodyid` body names**, because 30 of 31 geoms are unnamed and naming
  them would fork the vendored asset.
- **The planner probe must gate on `jnt_range`.** `mj_kinematics` + `mj_collision` returns `ncon=0` for
  out-of-range poses, so 65.9% of the planner's default grid is falsely labelled FREE. The probe gains a
  fourth verdict for out-of-limit.
- **The renderer is owned by the daemon's tick thread.** Cross-thread rendering does not fail cleanly —
  it emits a warning and returns *garbage frames*, and **a `threading.Lock` does not fix it** (measured:
  four threads under a lock returned two distinct wrong images). Single-thread ownership is the only
  measured-correct option.
- **stdlib `zlib`+`struct` PNG, ~20 lines.** PIL is present in `.venv-cu128` but appears in pyproject
  only under the `[hf]` extra — an undeclared transitive. Using it would make the viewer depend on a
  package the project does not declare.

## Risks

- **`wavefront_planner.py` is a rewrite, and it is now scoped work rather than an unknown** (clarified
  2026-08-01). Closed-form 2-link IK has no 6-DOF analogue and enumeration becomes ~2.6e9 cells, so the
  replacement samples instead. **This is the largest scope item in the feature** — a motion-planning
  rewrite living inside an arm-replacement feature — and it still holds the obstacle assertion being
  retired. Risk is no longer "will it be done" but "does it fit alongside everything else"; if it
  threatens the rest, splitting it into its own feature is the pressure valve.
- **The only dense reward changes scale.** `_COVER_CELL = 0.10` gives 1338 reachable cells today; the
  proposed 0.020 m was claimed to preserve that and **re-measures at 791** — the justification is wrong
  by 72%, and the count-preserving value is ≈0.0145. The cell is also 2-D while TCP z spans 0.743 m, so
  the reward goes blind on the axis the ground reflex lives on. Re-derive with a published sweep.
- **`TRAJ_DEV_MAX` is invalid in exactly the condition this feature's headline benchmark creates.** The
  proposed replacement borrows its ceiling from a constant bounding an *estimator's* dead-reckoning
  error — a category swap. Under 500 g the margin falls from 17.6× to 5.4×, and under contact measured
  `|qpos − ctrl|` reaches 49.65° (58× the bound) while `spiking_pid.py:227` acts on the breach. The
  bound must be derived under the loaded condition, and the plan must say what `trajectory_breach()`
  returns during contact.
- **The postural-recovery reflex may become a no-op that still counts.** `_A_POS` rebinds to
  `shoulder_pan`, a vertical-axis joint that cannot change TCP height.
- **The GROUND reflex may become unreachable.** HOME sits 226.5 mm above the floor with `ncon=0`, and
  floor contact could not be produced within joint limits in testing.
- **28% of the autonomous action space becomes press-into-yourself at force saturation.**
- **Cross-OS bit determinism is UNVERIFIED.** All measurements are one Windows binary on one CPU;
  MuJoCo makes no cross-platform bit-identity guarantee and CI is 2 OS × 3 Python. FR-010/SC-012 cannot
  be asserted as bitwise until measured on ubuntu — cheap experiment: hash a fixed 4000-step trace per
  cell.
- **HOME rewinds `d.time` to 0.** Anything keyed on the sim clock sees time run backwards.
- **`MUJOCO_GL` is process-global and read at import**, and an unrecognised value raises at import time.
  A daemon that renders and a self-check that must stay headless cannot share one setting.
- **A rendered PNG is the most persuasive, least provenance-carrying element on the page.** A picture of
  a stationary arm is indistinguishable from a picture of a moving one; the panel needs a revision
  counter or it becomes the exact lie `brain_telemetry.PROVENANCE` exists to prevent.

## Resolved decisions (clarified 2026-08-01)

All five open questions were settled in a clarification session; the answers are recorded in the spec's
Clarifications section and propagated into `tasks.md`.

1. **`wavefront_planner.py` is REPLACED, not ported and not retired.** A six-joint sampling planner over
   `probe_contact` replaces C-space enumeration (~2.6e9 cells at 6 DOF). This **moved motion planning
   into scope** and added FR-021 — the largest scope change from the session. The planner must use the
   probe as its single collision authority, reject out-of-limit configurations (the probe alone does
   not), and satisfy or explicitly replace the two assertions the old planner carried.
2. **The limit of record is `actuator_ctrlrange`** — the range that binds a position command — so
   `clamped` is true exactly when the command was altered. `jnt_range` remains the physical stop enforced
   by the solver and is what `probe_contact` gates on. The two are measured unequal and must not be
   conflated.
3. **The coverage cell becomes 3-D**, with its size re-derived from a published sweep. Three of five
   rotate joints move the TCP mainly in z (span 0.743 m), so a planar cell cannot observe them.
4. **`blocked` penalises the corresponding action in autonomous selection AND appears in telemetry.**
   Telemetry alone was explicitly rejected: a verdict firing on ~28% of in-limit targets that changes no
   outcome is dormant by P3's own guard.
5. **`substrate/demo.py`'s standalone `Arm` is DEFERRED** to implementation (T071) as low impact —
   decide there whether it is allow-listed or deleted under SC-010.

**Consequence for this plan**: FR-021 and FR-005a are new, and the planner is no longer a risk item to
be absorbed — it is scoped work. See the amended risk below.

## Complexity Tracking

No Constitution Check violations. Section intentionally empty.

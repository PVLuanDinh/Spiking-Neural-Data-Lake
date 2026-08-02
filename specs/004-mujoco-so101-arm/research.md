# Phase 0 Research: MuJoCo SO-101 Becomes The Arm

**Feature**: `specs/004-mujoco-so101-arm` | **Date**: 2026-08-01
**Baseline**: `c6ca8c8d`. `mujoco==3.11.0`. Every figure below was produced by running code on this
machine; nothing here is inferred from documentation.

---

## R1. The collision contract

**Question**: `ArmSim` REPORTS collision without PREVENTING it, by design and under test. MuJoCo
prevents penetration by construction. How does the contract survive?

**Decision**: **split the contract by call site, not by flag.**

- **Live body** — the verdict reads `data.ncon` off the stepped `MjData`. Contact is reported; the
  solver has already prevented it.
- **Planner probe** — `probe_contact(qpos)` writes into a module-private, **never-stepped** `MjData`
  and runs `mj_kinematics` + `mj_collision`. This *is* report-without-prevent, exactly, with no flag.
  MEASURED: `ncon=24`, min contact distance −0.2323 m, `qpos` bit-unchanged, `qvel` all zero.

**All three flag routes are measured dead.** Every flag that stops prevention also stops reporting:

| route | result |
|---|---|
| `contype = conaffinity = 0` on all 31 geoms | `ncon=0` — penetrates, reports nothing |
| `mjDSBL_CONTACT` | `ncon=0` — identical failure |
| `mjDSBL_CONSTRAINT` | `ncon=0`, min dist `nan`, **and silently disables joint limits on all six joints** |

**Rationale.** The planner's question is a counterfactual — "if the arm were here, would it collide?" —
and "prevent" is not a coherent answer when nothing is executing. The live arm's question is different,
and the old answer was an artifact of the toy. The repo already documented the damage
(`autonomous_loop.py:1274-1277`): *"P(collide at t+1 | collided at t) = 0.991, effectively an absorbing
state … 53 of 73 wake ticks spent underground."* Nothing is lost by preventing: driving hard into the
floor still reports `ncon>0` on **4907/5000 ticks (98.1%)**.

**Hazard if the split is got wrong**: teleporting the *live* data into a penetrating pose and resuming
`mj_step` ejects the body at **36.11 rad/s** from standstill. The probe `MjData` must be private by
construction, not by convention.

**Corrections from the adversarial pass** — three, all material:

- **The `OBSTACLE` label has no source.** The vendored `scene.xml` worldbody contains exactly one geom
  (`floor`). The old label came from a 2 m-reach point circle with no MJCF counterpart. **Retire the
  label** — and note `wavefront_planner.py:299-300` asserts an obstacle exists, a currently-green gate.
- **30 of 31 geoms are unnamed**, so labels must resolve through `geom_bodyid` body names rather than
  geom names; naming them would fork the vendored asset.
- **The probe is blind to joint limits.** `mj_kinematics` + `mj_collision` returns `ncon=0` for
  out-of-range `qpos`, so **65.9%** of the planner's default grid is falsely labelled FREE. The probe
  gains a fourth verdict for out-of-limit poses.

**Alternatives rejected**: keep the arm fully kinematic and never step (forfeits torque, gravity and
payload — i.e. the entire point of the feature); return a bool (silently disables the
`== "GROUND"` reflex and degrades the operator readout).

---

## R2. The gate on a stateful body

**Decision**: build **a gate that owns a body**, not a body that is gated.

**Frozen = latch a bool + `ctrl[:] = qpos` ONCE + keep stepping.** MEASURED at peak slew
(|qvel| = 5.454 rad/s):

| freeze strategy | result |
|---|---|
| `ctrl := qpos`, keep stepping | coasts 0.0252° / 4.24 mm, then stationary to 50 s sim |
| `ctrl := 0` | coasts **25.89° / 156.2 mm** — on a position actuator, ctrl=0 is a *move-to-zero command* |
| stop stepping | 0 mm moved, but leaves **5.45 rad/s stored**, discharged when anything resumes |
| actuation disabled (limp) | **38.32° / 175 mm** collapse under gravity |

Held-latch sag is bounded and stationary: 0.0256° / 0.175 mm, reached within 0.2 s sim and unchanged to
25 000 steps.

**The latch must be guarded.** The first contract put the latch action before the freeze early-exit, so
a repeated `EMERGENCY_STOP` re-executed `ctrl:=qpos; step` each time — contradicting its own "applied
once" ruling. `autonomous_loop.py:1163-1172` has **no already-stopped guard**, so a parked object or one
NaN distance emits one `EMERGENCY_STOP` per tick indefinitely. MEASURED drift: **0.0148° latched once vs
35.16° after 20 000 repeats (~2400×)**. Ruling: guard with `if not self.stopped`, and prove idempotence
with a self-check that applies it 1000× and asserts state bit-identical to once.

**HOME = hard reset, not settle.** The model has **no keyframe** (`nkey=0`), so `HOME_QPOS = zeros(6)`
is repo-declared. Reset + `mj_forward` is bitwise identical across trials —
`EE = (0.39136190000000015, −1.1255287011580572e−05, 0.22646874461625618)` m, `ncon=0`, `qvel=0` — so an
equality assert in the exact shape of `arm_sim.py:201` survives verbatim. "Command zeros and settle"
does not: it lands 0.031° off and is not bitwise reproducible across start poses.

**Joint limits: the MJCF is the source; `arm_config.py` owns the policy.** MEASURED `jnt_range`:

| joint | deg |
|---|---|
| shoulder_pan | ±110.0 |
| shoulder_lift | ±100.0 |
| elbow_flex | ±96.83 |
| wrist_flex | ±95.0 |
| wrist_roll | −157.21 .. +162.79 |
| gripper | −10.0 .. +100.0 |

Loaded at import from the pinned vendored model, not hand-copied (P1). **Clamping belongs in the gate**,
because MuJoCo's clamp is silent: `d.ctrl[0] = 5.0` against a ±1.92 range leaves `ctrl` unmutated at 5.0
while the body settles at the rail — a body-side clamp would let `state()` report a command the arm never
received.

**Unsatisfiable commands** return `reached: False` + `blocked: True`, not an error. MEASURED over 300
random in-limit targets: `ncon>0` predicts "commanded angle not reached" with **99.3%** agreement and
force saturation with **99.7%**. Clean servo droop floor ≤ 0.0492°; blocked cases sit at 5.96–87.5°
residual, stationary to 60 s sim.

**Open**: is the limit of record `jnt_range` or `actuator_ctrlrange`? MEASURED not equal.

---

## R3. The migration surface

**Naming: the MJCF wins.** It is the only name that exists outside the repo and must resolve via
`mj_name2id`; renaming forks the vendored asset and turns the version pin into a fork.
`arm_hardware.JOINTS` survives as an **alias column** and stops being a literal — it becomes a
derivation, so its existing assertions keep working against derived rows.

| qpos | MJCF (canonical) | alias | token | servo | lever mm |
|---|---|---|---|---|---|
| 0 | shoulder_pan | base_yaw | `JOINT_SHOULDER_PAN_ROTATE` | ST3215-1:345 | 0 |
| 1 | shoulder_lift | shoulder_pitch | `JOINT_SHOULDER_LIFT_ROTATE` | ST3215-1:345 | 300 |
| 2 | elbow_flex | elbow_pitch | `JOINT_ELBOW_FLEX_ROTATE` | ST3215-1:191 | 180 |
| 3 | wrist_flex | wrist_pitch | `JOINT_WRIST_FLEX_ROTATE` | ST3215-1:191 | 60 |
| 4 | wrist_roll | wrist_roll | `JOINT_WRIST_ROLL_ROTATE` | ST3215-1:147 | 0 |
| 5 | gripper | gripper | *(none — ENGAGE/RELEASE own it)* | SC-series | 0 |

**Token grammar**: `JOINT_<MJCF_NAME_UPPER>_ROTATE(<±d>deg)`. **The regex is syntax; the table is
vocabulary.** Today both `arm_sim.py:30` and `serial_arm.py:32` conflate them (`[AB]` is both), which is
precisely why other sites can mint tokens nobody validated. The joint-name group must be checked for
membership in the table, so a regex-only site cannot accept a joint that does not exist.

**F6 undercounted: there are EIGHT producers/validators, not three** — `interpreter.py:28-35`,
`arm_sim.py:30` + dispatch, `serial_arm.py:32-33`, `autonomous_loop.py:61-63` (no allow-list),
`nl_interface.py:38-57` (a fourth literal label set, plus a hardcoded English "shoulder"→rotate_a mapping
that becomes wrong), `cpg_oscillator.py:60` (joint A hardcoded), plus two more. **That undercount is the
finding.**

**Consequences the plan must carry**: the trajectory comparator's meaning changes (contact-blocked
commands leave `qpos` far from `ctrl`, so breach fires on contact as well as perturbation); the
postural-recovery reflex rebinds to a vertical-axis joint that cannot change TCP height; the 2-DOF
E-STOP proofs are exact-equality float tests that fail under physics.

---

## R4. Dependency, CI, assets

**`mujoco==3.11.0` core, exact pin, one line.** The pin follows the file's own recorded precedent
(`ruff==0.15.20`, *"a formatter's version is part of the format"*). For a physics engine the solver
version **is** the physics, and SC-012 plus every torque assertion are numeric asserts against one build.

**Wheel matrix verified against the real PyPI file list**: all three interpreters × both platforms ship
binary wheels. No sdist compile, no C++ toolchain.

**CI: add `MUJOCO_GL: disable`** to the self-check and test jobs. Headless import already survives, but
`disable` short-circuits GL at import, makes "CI never renders" declared rather than inferred, and
exercises FR-012b's viewer-unavailable branch on all six cells for free.

**Add a wheel-build smoke step.** CI installs with `pip install -e .`, so **the wheel is never built** —
and the asset-packaging trap fails only in the wheel.

**Assets: vendor them.** Licence is **Apache-2.0**; vendoring is permitted. 15 files, 16.14 MB on disk,
5.95 MB compressed, into `src/snn_data_lake/brainstem/so101/` with the upstream LICENSE.
**Do NOT name the directory `models/` or `vendor/`** — both are unanchored in `.gitignore` and hatchling
honours it, so the files would be silently omitted from the wheel while CI stayed green. **No packaging
change is required**: hatchling already ships non-`.py` files under the package tree.

**CI cost**: +7 to +17 s per cell (+1.2–2.8% of a >600 s suite), plus a cached 18.9 MB download that is
3.6% of the torch wheel every cell already pulls.

**Registration is FIVE sites, not three** — and the two omitted are the safety ones. See R6.

---

## R5. Rendering and the cockpit

**Decision**: demand-gated, **tick-thread-rendered** PNG served through the existing `_send`.

**F9 confirmed end to end**: a server whose `_send` is byte-for-byte the cockpit's served a
stdlib-encoded PNG and a real Chromium decoded it (`320x240`, `complete=true`, `onload` fired).

**Thread affinity is worse than F13 said: it fails ALWAYS and SILENTLY.** A `Renderer` built on one
thread and used from a worker emits only a *warning* (`WGL: Failed to make context current`) and then
**returns a frame** — a wrong one. In 4/4 trials worker frames sat ~22 away from *both* reference poses:
not stale, garbage. Four threads completed 80 renders in 0.009 s against a true cost of 1.36 ms —
physically impossible, i.e. they never rendered. **A `threading.Lock` does not fix it**: under a lock,
two threads returned one wrong image and two returned another.

What works, MEASURED: a `Renderer` created and used on **one single thread** — main thread 2.07 ms/frame,
or a dedicated non-main thread constructing its own renderer 1.84 ms/frame with poses correctly distinct.
So the renderer is owned by the daemon's tick thread, created lazily, and the HTTP route only stamps
demand and returns cached bytes.

**Encoder: stdlib `zlib` + `struct`**, ~20 lines, 1.90 ms at level 1. Verified by chunk walk, per-chunk
CRC32, zlib round-trip to identical pixels, Tk's unrelated PNG decoder, and Chromium. **PIL is present in
`.venv-cu128` but appears in pyproject only under the `[hf]` extra** — an undeclared transitive, so using
it would make the viewer depend on a package the project does not declare.

**Risks**: the GLFW hidden window shares a process with a live CUDA context (untested interaction);
`import mujoco` raises at import if `MUJOCO_GL` holds an unrecognised value; a rendered PNG of a
stationary arm is visually identical to one of a moving arm, so the panel needs a revision counter or it
becomes the exact lie `PROVENANCE` exists to prevent.

---

## R6. The finding that outranks the rest: the brain can rewrite the E-STOP

`godel_machine.py:148-156` defines `_SAFETY_KERNEL` as a tuple of **literal paths**, including
`"src/snn_data_lake/brainstem/arm_sim.py"` at `:151`. It flows into `self_mod_harness.py:59,68`,
`self_build_loop.py:11`, and `attempt_on_repo(..., protected=_SAFETY_KERNEL)`. It is **duplicated as a
second literal** at `self_improve.py:157-163`.

This is live, not theoretical: `learn_daemon.bat:14` sets `NORD_SELFMOD=1`, and `brain.py:2653-2654`
fires `_maybe_autonomous_self_build()` every 8 nights.

**Deleting `arm_sim.py` and creating a successor without editing both literals in the same commit
silently removes the arm-safety envelope from the brain's immutable set.**

**Ruling**: registration is five sites — `__init__.py` REGIONS, `run_selfchecks.py`, the dormancy gate,
`godel_machine.py:151`, `self_improve.py:160` — plus a test asserting every path in `_SAFETY_KERNEL`
exists on disk. That test is the mechanism that stops this recurring, and it is cheap.

---

## R7. Three constants that are currently wrong

- **`TRAJ_DEV_MAX = 0.15`** — documented as tied to a 2 m reach and consumed against a dimensionless EE
  distance. The proposed replacement borrowed its ceiling from a constant bounding an *estimator's*
  dead-reckoning error — a category swap. Under 500 g the margin falls 17.6× → 5.4×; under contact
  measured `|qpos − ctrl|` reaches **49.65°**, 58× the bound, while `spiking_pid.py:227` acts on the
  breach. Must be derived under the **loaded** condition, and the plan must say what
  `trajectory_breach()` returns during contact.
- **`_COVER_CELL = 0.10`** — the arm's only dense reward. The proposed 0.020 m was justified as
  preserving the reachable-cell count (1338); **re-measured it gives 791**, so the justification is wrong
  by 72% and the count-preserving value is ≈0.0145. The cell is also 2-D while TCP z spans 0.743 m, so
  the reward goes blind on the axis the ground reflex lives on.
- **`deviation()`'s assertions** — `< 1e-9` (`arm_sim.py:224`) and `< 1e-6` (`wavefront_planner.py:308`),
  both currently green. Measured servo droop at home, unloaded, fully settled: **3.31e-4 m** — fails by
  2.5–5 orders. Loosening is forbidden. The only shape that keeps both the assertion honest and the
  reflex alive is **splitting the quantity**: `tracking_error` (expected, bounded, displayed) and
  `deviation` (perturbation-only, reflex-bearing).

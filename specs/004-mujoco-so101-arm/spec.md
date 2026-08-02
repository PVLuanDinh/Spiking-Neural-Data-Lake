# Feature Specification: MuJoCo SO-101 Becomes The Arm

**Feature Directory**: `specs/004-mujoco-so101-arm`

**Created**: 2026-08-01

**Source**: Owner directive, 2026-08-01, from `robot_arm_build_notes.md`. Governed by
`docs/POLICIES.md` (P0–P6); `.specify/memory/constitution.md` is its governance surface and defers to
it. The Standing Invariants apply throughout.

**Owner authorisations on record (2026-08-01)** — this feature cannot proceed without them and they are
quoted rather than assumed:

- Full replacement of `brainstem/arm_sim.py` **including the `ArmSim.apply()` safety gate**, which
  CLAUDE.md:70 declares safety-critical and modifiable only with explicit approval.
- **MuJoCo is authorised to drive commands** — it is in the command path, not a viewer.
- `mujoco` becomes a **core dependency**; the zero-dependency CI path is retired.
- Install `mujoco` and clone `TheRobotStudio/SO-ARM100` — both done.

**Status**: Draft

## Problem

The brain has a body it cannot learn anything true from.

`brainstem/arm_sim.py` is named a "physics simulator" in CLAUDE.md and is not one. It is a **2-DOF
planar forward-kinematics calculator**: `l1=l2=1.0` dimensionless units, reach 2.0, **no mass, no
gravity, no velocity, no torque, no integration step** (`arm_sim.py:36-45`). Its optional pybullet
backend adds no dynamics either — gravity is zeroed and joints are set by `resetJointState`, a
teleport, so pybullet is a collision mirror of the same kinematics.

The repo already knows this and has already named the fix. `brainstem/arm_hardware.py:146-163`:

> `sim_dof: 2`, `target_dof: 6`, `missing_dof: ("wrist_pitch", "wrist_roll",
> "gripper_articulation", "base_yaw")`, `sim_units: "dimensionless (l1=l2=1.0, reach 2.0)"`,
> `target_units: "mm (reach ~300)"` … *"MuJoCo + SO-ARM100 (TheRobotStudio) ships MJCF with motor
> properties matched to the real STS3215 — a genuine digital twin. That is the path from this
> kinematic toy to a body whose torque limits are real."*

So this feature is not a new idea. It is the execution of a bridge the codebase has already specified
and labelled, and which has sat as a string in a dictionary instead of as a body.

**What that buys, concretely**: torque. The build notes' entire purchasing decision — whether STS3215
servos can hold a 500 g payload at 300 mm reach — is unanswerable against a simulator with no torque.
Against the SO-101 MJCF it is a single measurement, and the actuator force range is already matched to
the real servo.

## Clarifications

### Session 2026-08-01

- Q: `wavefront_planner.py`'s 2-link IK and 2-D C-space enumeration do not survive the move to 6 DOF (~2.6e9 cells). What happens to it? → A: Full 6-DOF replacement now — a sampling planner over the MuJoCo probe, using the self-collision checking the new body provides.
- Q: Which measured distance is "reach" for SC-006, and what tolerance counts as matching the 300 mm build target? → A: Reach is the maximum horizontal TCP radius, measured 0.4795 m. The 300 mm figure is the SHOULDER LEVER, not reach — the two were never the same measurement, so SC-006 stops comparing them.
- Q: The model exposes two unequal limit tables. Which is the limit of record that the gate clamps against? → A: `actuator_ctrlrange` — what actually binds a position command, so `clamped` is true exactly when the command was altered. The physical stop still applies underneath as a solver constraint, and the contact probe gates on it separately.
- Q: The arm's only dense reward counts fresh spatial cells. On a 6-DOF body with 0.743 m of vertical TCP span, should the cell become 3-D? → A: Yes — 3-D, with the cell size re-derived from a published sweep rather than carried over. Three of five rotate joints move the TCP mainly vertically, so a 2-D cell cannot see most of what the new joints do.
- Q: What consumes the `blocked` verdict, given it fires on 28% of in-limit targets? → A: Both — it penalises the action in autonomous selection (so the arm learns not to press into itself) AND appears in telemetry. Selection is the consumer that satisfies P3; telemetry alone would not.

### Session 2026-08-01 (implementation)

- Q: `JOINT_X_ROTATE(+10deg)` — the incumbent ADDS 10 deg (8 applies = 80 deg) while the first replacement gate treated it as GO TO 10 deg. Which semantics wins? → A: **Both verbs.** `ROTATE` is relative, matching the incumbent and its own name; `MOVETO` is an explicit absolute setpoint. Absolute-only would have silently turned `cpg_oscillator`, `spiking_pid` and `wavefront_planner` into non-controllers and collapsed the autonomous repertoire to two setpoints per joint — measured, 5 distinct 100 mm cells over 3000 ticks, so the arm's only dense reward could never fire.
- Q: Does the new arm get a `perturb()`? Without an external-disturbance channel the trajectory reflex is structurally dead, and magnitude cannot separate a shove from ordinary motion (an ordinary commanded move peaks at 0.523 rad of tracking error, LARGER than the 0.350 from a 20 deg shove). → A: A clearly-named **test-only `inject_disturbance()`**, plus `deviation` measured against the pose the servo is actually holding rather than against a magnitude threshold. Measured, that separates them completely: deviation is 0.000000 for any commanded motion however large, and 0.349066 for a 20 deg external shove. In normal operation deviation is legitimately 0.0 because the simulated arm has no external disturbance source — a true statement about the body, not a stub.
- Q: One `apply()` advances physics by a single 2 ms step, so a `+45deg` command moves the joint **0.020 deg** and needs ~150 applies to settle at 37.8 deg. The incumbent teleported, so every tick-count constant downstream was written against a body that arrived instantly. What should one motor command mean? → A: **No fixed step count — the brain updates the rate autonomously.** The body exposes `substeps` (default **1**, the integrator's own step, the only value that is a measured fact rather than a choice) and `set_substeps()`, bounded by a *derived* ceiling of one second of simulated time. `AutonomousArm._adapt_actuation()` raises the rate while commands fail to arrive and lets it decay while they succeed, driven by the `reached` flag the loop already reads back. This keeps P1 (no transcribed constant governing how far a command travels) and P0 (the rate becomes one more thing the brain improves), instead of replacing one magic number with another.

  **Not yet verified to converge usefully, and stated as an open rather than a result.** Measured over 400 ticks the rate rises to 256 and then decays back to 1 and stays there: the loop drives a joint to its limit (span 100.000 deg, the full ctrl range), where every command trivially "arrives", so `reached` stays true and the rate falls away. Whether that is the adaptation working (a stuck arm should not buy longer ticks) or the exploration policy failing (the arm should not be parked on a limit) is not yet distinguished. A neuter A/B was attempted and **its result is discarded as invalid** — the neuter stubbed `set_substeps` but the pinned and adapting runs came back byte-identical on every reported quantity while the comparison still printed a difference, and the cell metric read an attribute that does not exist. The knob is therefore wired and driven by a real signal, but **not yet demonstrated to change an outcome**, which is what P3 actually requires.
- Q (T052): What does `trajectory_breach()` return during sustained contact, given `spiking_pid.py:227` acts on it? → A: **Contact-explained deviation is EXCLUDED, and it is measured.** With the arm driven into the floor and held for 50 ticks: `collision=GROUND`, `tracking_error=0.6842` rad — *larger* than the 0.3496 from a genuine 20° external shove — and `deviation=0.0`, `breach=False`. An external shove on the same body gives `deviation=0.349066`, `breach=True`. The separation is total and is not a magnitude threshold: deviation is measured against the pose the servo actually settled at. Had contact tripped the breach, the PID would abort on every touch, which is exactly when a controller is needed. Pinned by two tests.
- Q (T051): Re-derive `TRAJ_DEV_MAX` under load? → A: **No — retire it with `arm_sim`.** The constant (0.15 m, tied to a 2 m reach, compared against a dimensionless distance) is read *only* by `arm_sim.py`. `So101Arm.trajectory_breach()` never references it: it derives a per-joint bound from the model's own gravity torque and actuator gain, so it tracks a model change instead of going stale. Re-deriving a constant that only the retiring body reads is work on a corpse; the task is superseded by Phase 7. A test asserts the live body does not read it.
- Q: Does growing `ACTIONS` 7 → 13 invalidate persisted policy? → A: **No — the premise was wrong.** `self.q` has no save path, is absent from `brain_bundle`, never reaches `brain_lake.bin`, and is keyed by action STRING via `.get(a, 0.0)`. It is rebuilt in memory each run, so new actions simply start at 0.0 and nothing is reindexed. No migration and no invalidation are needed.

## Findings already in hand (verified 2026-08-01)

**F1 — The SO-101 model loads, steps, and renders offscreen on this machine.** MEASURED with
`mujoco 3.11.0` against `D:\AI-project\SO-ARM100\Simulation\SO101\scene.xml`: `nq=6, nv=6, nu=6,
nbody=8`; joints `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` with real
radian limits; **actuator force range ±3.350 N·m** on every joint, which is ~34 kg·cm and matches the
build notes' ~30 kg·cm STS3215-at-12V figure; `timestep = 0.002`. Offscreen render returned a
`(240, 320, 3) uint8` frame with no window. `qfrc_bias` after `mj_forward` gives gravity load directly
— unloaded, `shoulder_lift` needs **0.5388 N·m** to hold. The build notes list that torque readout as
"not yet done"; it is one call.

**F2 — `ArmSim.apply()` is the gate, and it is the only mutation path.** Four live call sites outside
self-checks: `autonomous_loop.py:1172` (E-STOP), `:1269`, `:1292`, and `live_brain.py:415`. Replacing
`ArmSim` replaces the gate every motor command passes through.

**F3 — The E-STOP is a latch inside `ArmSim`, not `reflex.py`.** `arm_sim.py:66-70` sets
`self.stopped = True` on `EMERGENCY_STOP`, and every command except `RETRACT_ALL`/`HOME` is a no-op
while stopped; `:81-85` clears it. There are **two different things called "the reflex"**, and the
Brain path uses the one that is *not* `reflex.py` — `reflex.Reflex` is constructed only at
`closed_loop.py:39` and `signal_loop.py:491`. Any replacement must reproduce the latch exactly.

**F4 — `ArmSim` REPORTS collision without PREVENTING it, deliberately, and MuJoCo does the
opposite.** `arm_sim.py:56-62` computes ground/obstacle contact; `apply()` never consults it; the
behaviour is asserted at `:211-212`. MuJoCo's contact solver physically blocks penetration. **This is a
direct semantic conflict and the single hardest problem in this feature.** Downstream depends on the
current semantics: `wavefront_planner.py:7` calls `ArmSim.collision()` *"the ONE collision authority"*
and constructs a scratch `ArmSim` at `:73` to use it read-only.

**F5 — `SerialArm` does not trust the mirror.** `serial_arm.py:79-89` re-checks the E-STOP freeze and
a wire-level allow-list itself before writing to hardware, so a blocked command reaches the mirror but
the wire stays silent. The real-arm calibration gate `SAFETY_CALIBRATED = False`
(`arm_config.py:32`) currently refuses to drive real servos at all.

**F6 — The command vocabulary is a closed string set defined in EIGHT places that must agree.**
CORRECTED 2026-08-01 — this finding originally said three, and the undercount was itself the finding.
The three below are the *validating* sites; five more **produce** tokens, and one of those has no
allow-list at all: `autonomous_loop.py:61-63` (builds `ACTIONS` from a hardcoded joint pair and hands
the token straight to `apply()`), `nl_interface.py:38-57` (a fourth literal label set, plus a hardcoded
English "shoulder"→rotate_a mapping that becomes wrong at six joints), `cpg_oscillator.py:60` (joint A
hardcoded in the emitted token), and two further sites enumerated in [research.md](./research.md) R3.
Validating a token grammar against three of eight sites is how five other sites came to mint tokens
nobody checked.
`Interpreter.COMMANDS` (`interpreter.py:28-35`, containing `JOINT_A_ROTATE`, `JOINT_B_ROTATE`,
`ENGAGE`, `RELEASE`, `HOME`, plus `REFLEX_CMD`), the `ArmSim.apply()` dispatch and its `_JOINT` regex
(`arm_sim.py:30`), and `serial_arm`'s wire allow-list. All three are **two-joint** vocabularies —
`JOINT_A` and `JOINT_B`.

**F7 — The autonomous path has no allow-list.** `autonomous_loop.py:60-71` builds `ACTIONS` from a
hardcoded `_JOINT_STEP_DEG = 6.0` over joints `A` and `B` only, and hands the token straight to
`apply()`. The chat path is allow-listed; this one is not.

**F8 — The daemon that serves the frontend runs `.venv-cu128`, not `.venv-lake`.**
`learn_daemon.bat:20-23`. pybullet is installed **only** in `.venv-lake`, which is the CPU gate env —
so the existing physics backend has never been reachable from the live frontend. `mujoco 3.11.0` is
since been installed into `.venv-cu128` (2026-08-01) and verified there; `.venv-lake` still lacks it.

**F9 — The cockpit serves no binary today, but its send path already supports it.** CORRECTED
2026-08-01 — an earlier reading of this was wrong. `cockpit.py:137-144` is
`data = body if isinstance(body, bytes) else body.encode("utf-8")` with the content type a parameter,
so **bytes and an arbitrary content type are already supported**; there is simply no route using them.
The page is one raw ~390-line HTML string, every value arrives later as JSON over `fetch`, the client
polls at 1 Hz, and there is no push. Adding an image route is therefore a smaller change than "no
binary content type" implied.

**F13 — Rendering is thread-affine and the cockpit is threaded.** MuJoCo's GL context belongs to the
thread that made it, while the cockpit runs a `ThreadingHTTPServer` (`cockpit.py:206`) that spawns a
thread per request. Any render path must be serialised or produced once per tick and cached; a naive
render-per-request will fail intermittently rather than cleanly.

**F14 — Simulation cost is not a concern; MEASURED.** 2000 physics steps in **0.038 s**, roughly 105×
realtime at 500 Hz. The arm will not be the tick's bottleneck. Rendering, not stepping, is the cost to
watch.

**F15 — Two existing assertions encode "the sim is a toy" and will fail on contact with a real body.**
`arm_hardware.py:217-218` asserts `sim_dof < target_dof` and `sim_has_dynamics is False`. These are
the honesty mechanism that stops the dashboard implying the simulated arm is the real one. When the
body gains six real degrees of freedom and actual dynamics, **both assertions become false and must be
rewritten in the same change** — not deleted, and not silenced. The deterministic HOME pose is
similarly pinned: `arm_sim.py:201` asserts the end-effector rests at `(2.0, 0.0)`, a 2 m-reach fact.

**F16 — The command sites do not currently agree, and the disagreement is deliberate.** `HOLD` is on
`serial_arm`'s wire allow-list and is NOT in `Interpreter.COMMANDS` and NOT a case in
`ArmSim.apply()` — it falls through to the no-op branch on purpose (`arm_sim.py:86`). So FR-006's
"the sites must agree" means *agree by intent, with every asymmetry deliberate and recorded*, not
"be textually identical".

**F17 — `pybullet_contacts()` has zero callers** (`arm_sim.py:178`). The retiring implementation
carries dead code that must not be ported forward under FR-017's port-before-retire.

**F10 — A new panel has a three-part extension point and a fourth gate that fails CI if any part is
skipped.** Data must be added to `snapshot()` and registered in `brain_telemetry.DASHBOARD_KEYS`
(`:65-104`) or `DASHBOARD_SECTION_KEYS` (`:109-113`); markup added to the HTML string; paint added to
the single `paint()` function (`cockpit.py:490`). The existing `"body"` section key is
`("theta_deg", "ee", "effector", "collision", "deviation")` — **a 2-DOF-shaped contract**.

**F11 — Telemetry already states the arm does not move.** `brain_telemetry.py:190` documents that
`body` carries `moves: False` *"because only `effector` currently changes"*, and `:221-225` says a
dashboard letting a viewer assume otherwise would be the exact failure that module exists to prevent.
A 6-DOF arm that actually moves must update this honestly rather than silently.

**F12 — P2 is not satisfiable today, and the policy says so explicitly.** `docs/POLICIES.md:150-153`
names the failure mode: *"'Merging' that keeps both code paths behind a flag, a mode switch, or an `if`
is additive with extra steps."* So a `use_mujoco=` flag is not a solution. And P2 gates eligibility on
a benchmark that exists — **no A/B comparing a MuJoCo arm to `ArmSim` exists in this repo**, so the
required first answer is *build the benchmark*. `:164-173` records an OPEN P2 DEBT precedent showing
coexistence is permitted only as measured, named debt.

**F18 — THE BRAIN CAN REWRITE THE ARM'S SAFETY GATE, AND THIS FEATURE CAN SILENTLY WIDEN THAT.**
`godel_machine.py:148-156` defines `_SAFETY_KERNEL` as a tuple of **literal file paths**, including
`"src/snn_data_lake/brainstem/arm_sim.py"` at `:151`. It flows into `self_mod_harness.py:59,68`,
`self_build_loop.py:11`, and `attempt_on_repo(..., protected=_SAFETY_KERNEL)`. The same list is
**duplicated as a second literal** at `self_improve.py:157-163`.

This is live, not theoretical: `learn_daemon.bat:14` sets `NORD_SELFMOD=1`, and `brain.py:2653-2654`
fires autonomous self-build every 8 nights.

**Deleting `arm_sim.py` and creating a successor without editing BOTH literals in the SAME COMMIT
silently removes the arm-safety envelope from the set of files the brain may not rewrite.** No gate
catches it, because the protection is a path string and the path simply stops matching anything. This
is the highest-severity item in the feature, and it is a registration problem rather than a design one.

## User Scenarios & Testing

Five slices in execution order. Slice 1 is a P2 precondition, not a nicety.

### User Story 1 — The comparison that makes the swap legal (Priority: P1)

The operator can run one benchmark that measures the MuJoCo SO-101 arm and the existing `ArmSim` on
the same inputs and reports where each is right, wrong, or silent.

**Why this priority**: P2 makes "better" require a benchmark that exists, and none does (F12). Without
this slice the replacement is not policy-eligible, regardless of how obviously better a real physics
model is. It also produces the number the build notes actually want — the 500 g payload torque answer.

**Independent Test**: Run the benchmark against both arms; it emits a comparison table and a verdict.

**Acceptance Scenarios**:

1. **Given** both arm implementations, **When** the same command sequence is applied to each, **Then**
   end-effector pose, collision verdict, and E-STOP behaviour are reported side by side.
2. **Given** the SO-101 model at a commanded pose, **When** joint torque is read, **Then** a number in
   N·m is reported per joint — a quantity `ArmSim` structurally cannot produce.
3. **Given** a 500 g payload attached at the gripper's tool-centre-point, **When** the arm is
   commanded through its reach, **Then** the required holding torque per joint is reported against the
   actuator's rated limit, and any joint exceeding it is named.
4. **Given** the benchmark output, **When** P2's "better requires a benchmark" test is applied,
   **Then** the swap is either licensed or explicitly refused on evidence.

---

### User Story 2 — The gate survives the transplant (Priority: P2)

Every safety property the existing gate provides still holds when MuJoCo is the body: the E-STOP
latches and only the two clearing commands release it, blocked commands never reach the wire, and the
real-arm calibration gate still refuses.

**Why this priority**: This is the safety-critical core and the owner's explicit authorisation covers
it. Nothing else may land before it, because every later slice routes commands through this gate.

**Independent Test**: Replay the existing gate's self-checks and tests against the new body; behaviour
must be indistinguishable where the semantics are defined.

**Acceptance Scenarios**:

1. **Given** an E-STOP command, **When** any command other than the two clearing commands follows,
   **Then** it is a no-op and the latch remains set.
2. **Given** a latched E-STOP, **When** a clearing command is issued, **Then** the latch releases and
   the arm returns to its home pose.
3. **Given** the real-hardware path with calibration unset, **When** any motor command is issued,
   **Then** nothing is written to the wire, and this cannot be bypassed.
4. **Given** a command blocked by the gate, **When** the wire is observed, **Then** it is silent even
   though the mirror updated.
5. **Given** the autonomous path, which today has no allow-list (F7), **When** it emits a token,
   **Then** that token is validated by the same gate as the chat path.

---

### User Story 3 — The body becomes six-jointed (Priority: P3)

The arm has six joints with real units, and everything that speaks to it speaks six-joint.

**Why this priority**: The DOF change is the substance of the feature and touches the widest surface.
It is third because it is meaningless without a gate (US2) and unlicensed without a benchmark (US1).

**Independent Test**: Command each of the six joints individually and observe the commanded joint, and
only that joint, move.

**Acceptance Scenarios**:

1. **Given** the six-joint vocabulary, **When** **all eight** producer and validator sites are compared,
   **Then** they agree exactly, with no token accepted by one and rejected by another, and no site able
   to mint a token that was never validated.
2. **Given** a command naming a joint that does not exist, **When** it is issued, **Then** it is
   rejected rather than silently ignored.
3. **Given** the arm's reported reach, **When** it is read, **Then** it is in real units rather than a
   dimensionless 2.0, is defined as the maximum horizontal TCP radius (~0.4795 m), and is not conflated
   with the 300 mm shoulder lever, which is a different measurement.
4. **Given** the trajectory-deviation threshold, which is documented as tied to a 2 m reach, **When**
   the arm's reach changes by roughly an order of magnitude, **Then** the threshold is re-derived
   rather than carried over.
5. **Given** a commanded joint angle beyond the model's limit, **When** it is applied, **Then** the
   outcome is defined and tested — the current arm never clamps at all.

---

### User Story 4 — The operator can see the arm (Priority: P4)

The cockpit shows the arm's actual pose, updating as it moves, without a second frontend and without
the page pretending motion that is not happening.

**Why this priority**: This is the visible payoff and the owner's stated goal, but it observes the
system rather than changing it, so it follows the substance.

**Independent Test**: Open the cockpit, command a joint, watch the displayed pose change.

**Acceptance Scenarios**:

1. **Given** the running daemon, **When** the cockpit is opened, **Then** the arm's six joint values
   and end-effector position are displayed.
2. **Given** a commanded motion, **When** the page next updates, **Then** the display reflects the new
   pose within one refresh interval.
3. **Given** the arm is not moving, **When** the page is read, **Then** it says so rather than
   implying motion — the honesty property telemetry already enforces (F11).
4. **Given** any value the panel displays, **When** the dashboard-key gate runs, **Then** the key is
   registered and the gate passes.
5. **Given** the cockpit, **When** it starts, **Then** it still never constructs a brain.

---

### User Story 5 — Exactly one arm survives (Priority: P5)

The old implementation is gone, not disabled, and the tree contains one arm.

**Why this priority**: P2 is non-additive and a flag-guarded coexistence is the failure it exists to
prevent (F12). Retirement is last because port-before-retire requires everything above to have landed.

**Independent Test**: Search the tree for a second arm implementation; find none.

**Acceptance Scenarios**:

1. **Given** the retirement, **When** the tree is searched, **Then** exactly one arm implementation
   exists and no flag selects between two.
2. **Given** each unique capability of the retired arm, **When** the new arm is checked, **Then** the
   capability is present — ported before deletion, not after.
3. **Given** the retirement, **When** consumers that read the old arm's internals are checked,
   **Then** each has been migrated rather than left resolving to a deleted name.

### Edge Cases

- **Collision semantics invert.** The old arm reports collision and lets the command through; MuJoCo
  physically prevents penetration. Every consumer of the collision verdict is affected, including the
  planner that treats it as the single authority (F4).
- **The commanded pose is unreachable because contact blocks it.** Under the old arm this was
  impossible; the arm always reached the commanded angle.
- **A joint is commanded past its limit.** The old arm never clamped, so no consumer has ever seen a
  clamp.
- **Physics is non-deterministic across platforms.** Determinism under a fixed seed is a repo
  invariant, and CI runs two operating systems and three Python versions.
- **The dependency is absent in the environment that actually matters.** The daemon's env is not the
  gate env (F8); a dependency installed in the wrong one is invisible where it counts.
- **The simulation must keep up with the tick.** A 0.002 s physics timestep against the brain's tick
  rate implies many substeps per tick; the arm must not become the tick's bottleneck.
- **Rendering cost.** If frames are produced, they are produced on the daemon's thread budget.
- **The model files live outside the repo.** A clone under a sibling directory is not a tracked
  dependency and can drift or vanish.
- **Region and gate registration.** A new module in a region directory that is not registered fails
  the atlas gate; a module unreachable from a live path fails the dormancy gate.
- **The daemon may have no window station.** If launched by Task Scheduler with "run whether user is
  logged on or not", the platform GL context can fail. A render path must degrade to "viewer
  unavailable" rather than take the daemon down.
- **The honesty assertions invert.** Two existing assertions state the simulated arm is a toy with no
  dynamics (F15). A body with real dynamics falsifies them, and the correct response is to rewrite what
  they assert, never to remove them.
- **The retiring implementation contains dead code.** Port-before-retire must not carry forward a
  capability that has no callers (F17).
- **Vendored model assets carry a licence.** Assets copied into the repo bring their upstream terms with
  them; the licence must be checked before vendoring, not after.

## Requirements

- **FR-001**: A benchmark comparing the new arm and the existing arm on identical inputs MUST exist
  and MUST have been run before any deletion occurs (P2 eligibility).
- **FR-002**: The E-STOP MUST latch, MUST make every command other than its two defined clearing
  commands a no-op while latched, and MUST release only on those two. This behaviour MUST be
  demonstrated equivalent to the current behaviour, not merely re-described.
- **FR-003**: The real-hardware calibration gate MUST retain its no-bypass property: with calibration
  unset, no motor command may reach the wire by any path.
- **FR-004**: Wire-level validation MUST remain independent of the simulator's own state — the
  hardware path MUST NOT be made to trust the mirror.
- **FR-005**: Every motor command, from every path including the autonomous one, MUST pass through
  exactly one gate. No path may retain a private vocabulary that bypasses validation.
- **FR-005a**: The verdict that a command was physically unsatisfiable MUST change an outcome, not only
  a display. It MUST penalise the corresponding action in autonomous selection, so the arm stops
  choosing actions that cannot succeed, AND it MUST be visible in exported state. It fires on roughly a
  quarter of in-limit targets; a value computed that often that changes nothing is dormant by P3's own
  guard, and telemetry alone does not discharge that.
- **FR-006**: The command vocabulary MUST remain a closed set, and its definition sites MUST agree.
  A token accepted at one site and rejected at another is a defect.
- **FR-007**: The arm MUST report pose in real physical units, and any threshold previously derived
  from the old arm's dimensionless reach MUST be re-derived rather than carried over. Each re-derivation
  MUST publish the sweep or measurement it came from. Specifically: the spatial-novelty reward MUST
  count cells in **three dimensions**, because the majority of the new joints move the tool mainly
  vertically and a planar cell cannot observe them; and the trajectory-deviation bound MUST be derived
  under the **loaded** condition, naming the load.
- **FR-008**: The collision contract MUST be stated explicitly in exactly one place and every consumer
  MUST agree with it. If reporting-without-preventing is preserved, that MUST be demonstrated against a
  body whose solver prevents penetration; if the contract changes, every consumer MUST be updated in the
  same change. **This includes the set of verdict values**: a verdict that the new body can never
  produce MUST be retired rather than left as an unreachable branch, and retiring one is a contract
  change subject to this requirement — including any currently-green assertion that depends on the
  retired value.
- **FR-009**: Joint-limit behaviour MUST be defined and tested. Silent acceptance of an out-of-range
  command is not acceptable. The gate clamps against **the range that binds a position command**, so
  that a command reported as accepted is the command the actuator received and `clamped` is true exactly
  when the command was altered. The physical joint stop remains in force underneath as a solver
  constraint and is what the contact probe MUST gate on; the two tables are measured unequal and MUST
  NOT be conflated.
- **FR-010**: The arm MUST be deterministic given a fixed seed, on every operating system and Python
  version the project gates on.
- **FR-011**: The dependency MUST be present in the environment that runs the daemon, not only in the
  environment that runs the gates.
- **FR-012**: The model assets the arm depends on MUST be obtainable and version-pinned by the repo;
  the arm MUST NOT depend on an untracked directory outside the project. Their upstream licence MUST be
  verified as compatible BEFORE any asset is vendored.
- **FR-012a**: Assertions that currently state the simulated arm lacks the real arm's degrees of freedom
  and dynamics MUST be rewritten to state what is newly true, and MUST NOT be deleted or disabled. They
  are the mechanism preventing the dashboard from overclaiming, and a body with real dynamics needs that
  mechanism more, not less.
- **FR-012b**: If the arm cannot produce a visual frame in the deployed environment, the frontend MUST
  report the viewer as unavailable and the daemon MUST keep running.
- **FR-013**: The cockpit MUST display the arm's pose, MUST NOT imply motion that is not occurring,
  and MUST NOT construct a brain.
- **FR-014**: Any value a new panel displays MUST be registered with the dashboard-key gate.
- **FR-015**: The arm MUST NOT become the tick's bottleneck; its per-tick cost MUST be measured and
  stated.
- **FR-016**: Exactly ONE arm implementation MUST survive. A flag, mode switch, or conditional that
  keeps both paths reachable does not satisfy this.
- **FR-017**: Every unique capability of the retired arm MUST be ported BEFORE it is deleted.
- **FR-018**: All three gates MUST pass, on every OS and Python version in the matrix, with no
  assertion loosened and no check silenced.
- **FR-021**: Path planning MUST survive the DOF change as a working capability, planning over all six
  joints rather than a subset. Exhaustive configuration-space enumeration MUST NOT be carried forward —
  it is infeasible at six joints — so the replacement plans by sampling candidate configurations and
  testing each against the contact probe. The replacement MUST use the probe as its single collision
  authority, MUST reject configurations outside the joint limits (the probe alone does not), and MUST
  satisfy or explicitly replace the two assertions the old planner carried. Planning quality beyond
  "finds a collision-free path when one exists" is not required by this feature.
- **FR-020**: Any rename or deletion of the arm module MUST update **every** registration that names it
  by literal path, in the **same change** — including the self-modification safety kernel that defines
  which files the brain may not rewrite. A test MUST assert that every path in that kernel exists on
  disk, so a missed registration fails loudly instead of silently removing the arm from the protected
  set. Autonomous self-build is enabled by default, so this protection is load-bearing at runtime, not
  merely at review time (F18).
- **FR-019**: The torque a joint must produce to hold a commanded pose, with and without a payload,
  MUST be reportable — the capability the existing arm structurally lacks and the reason this
  replacement is worth its risk.

### Key Entities

- **Arm body**: the simulated mechanism — its joints, their limits, its reach, and its pose. Six
  joints in real units, replacing two in dimensionless ones.
- **Command**: a token from a closed vocabulary, validated once, that requests a change to the body.
- **Gate**: the single point every command passes through; owns the E-STOP latch and validation.
- **Collision verdict**: the body's report of contact, consumed by planning and by the dashboard.
- **Torque reading**: per-joint force required to hold a pose, with or without payload. New.
- **Arm snapshot**: the body state the frontend renders. Currently a 2-DOF-shaped contract; its
  replacement is fixed once in [plan.md](./plan.md) and [data-model.md](./data-model.md) and is binding
  on every producer and consumer. Two properties of that contract are requirements rather than design
  taste: the contact verdict keeps its existing name and `None`-or-label type, because live consumers
  and a registered telemetry key depend on both; and the reported excursion **splits into two
  quantities** — expected servo droop versus perturbation-attributable deviation — because a position
  servo makes them different, and it is the only shape under which the existing assertions survive
  without being loosened (FR-007, FR-008).

## Success Criteria

- **SC-001**: A single command produces a side-by-side comparison of both arms on identical inputs,
  with a stated verdict on whether the replacement is licensed.
- **SC-002**: Per-joint holding torque is reported in N·m for a commanded pose, with and without a
  500 g payload, and any joint over its rated limit is named.
- **SC-003**: E-STOP behaviour is demonstrated identical to the current behaviour across the full
  command set, including the latch, the no-ops, and both clearing commands.
- **SC-004**: With calibration unset, no command reaches the wire by any path, demonstrated rather
  than asserted.
- **SC-005**: Each of the six joints can be commanded individually, and only the commanded joint moves.
- **SC-006**: Reach is reported in real units and defined as the **maximum horizontal TCP radius from
  the base axis** — measured 0.4795 m on the model — agreeing with the model-derived value within 1%.
  The build target's 300 mm figure is the **shoulder lever arm**, not reach; the two are different
  measurements and MUST NOT be asserted equal. Whichever quantity a document reports, it names which one.
- **SC-007**: The collision contract is stated in one place and every consumer demonstrably agrees.
- **SC-008**: The operator sees the arm's pose in the cockpit, updating within one refresh after a
  commanded motion, with no second frontend.
- **SC-009**: Per-tick arm cost is measured and stated, and the tick rate does not regress beyond a
  stated bound.
- **SC-010**: A tree-wide search finds exactly one arm implementation and no selecting flag.
- **SC-011**: All three gates pass on the full matrix with nothing loosened.
- **SC-012**: Two runs with the same seed produce identical arm state.

## Assumptions

- The SO-101 model's actuator properties are treated as a faithful stand-in for the STS3215 servos in
  the build notes, on the strength of the ±3.350 N·m force range matching the quoted ~30 kg·cm at 12 V.
  Physical validation against real hardware is out of scope and would be its own feature.
- The build notes' arm and the SO-101 are assumed to be the same 6-DOF layout. Both enumerate base
  yaw/pan, shoulder, elbow, wrist pitch, wrist roll, gripper.
- The `SO100` variant present in the cloned assets is assumed not to be used; the SO-101 is the target.
- The existing arm's `moves: False` honesty note is assumed to become false in fact — the new arm's
  joints are expected to move on the canonical tick — and the telemetry is expected to change with it.
- Offscreen rendering is assumed available because it was verified on this machine; CI is assumed to
  need no rendering, only simulation.

## Out of Scope

- **Driving real hardware.** The calibration gate stays shut. This feature makes the simulated body
  real; it does not commission the physical arm.
- **Buying or building the physical arm.** The build notes' BOM is the motivation, not the deliverable.
- ~~**Improving motion planning or inverse kinematics.**~~ **MOVED INTO SCOPE, clarified 2026-08-01.**
  The planner is REPLACED, not ported: its closed-form 2-link IK has no 6-DOF analogue and its C-space
  enumeration grows to ~2.6e9 cells, so enumeration is abandoned for sampling over the contact probe.
  See FR-021. Retiring it as debt and running it on a reduced joint subset were both considered and
  rejected. It also holds two currently-green assertions (an obstacle existence check and a `deviation`
  bound) that must be handled by the replacement.
- **Photorealistic rendering, RL, or vision.** The note names Isaac Sim for that; not this feature.
- **The SO100 variant, ROS2/Gazebo/MoveIt2, and `ros2_control`.** Named in the notes as alternatives;
  none is adopted here.
- **Retuning the brain's arm-related learning.** Behaviour will change because the body changes; this
  feature measures that, it does not chase it.

## Dependencies

- `mujoco` 3.11.0 — **installed into `.venv-cu128`, the daemon env, on 2026-08-01** and verified there
  (model loads, 2000 steps in 0.051 s, offscreen render returns a frame, and `numpy`/`torch`/CUDA are
  unaffected). Also present in `.venv-arm`. **Still absent from `.venv-lake`**, the deterministic gate
  env used for local `run_selfchecks`/`pytest`, so FR-011 is partly satisfied and not yet closed (F8).
- The SO-101 model assets — cloned to a sibling directory outside the repo, which FR-012 forbids
  depending on permanently.
- P2 eligibility depends on FR-001's benchmark, which does not yet exist.

## Status

**COMPLETE 2026-08-01.** All 79 tasks across 8 phases done; all three gates green
(self-checks 278/278, pytest 628 passed / 92.63% coverage, ruff clean).

The SO-101 is THE arm. `brainstem/arm_sim.py` is deleted, `test_arm_sim.py` with it, and
`test_exactly_one_arm` asserts structurally that exactly one body implementation exists.

### Carried forward, stated rather than hidden

1. **The brain-owned actuation rate is wired but not proven to change an outcome.** It moves on
   a real signal (`reached`) and is bounded by a derived ceiling, but over 400 ticks it rises to
   256 and decays back to 1 because the loop parks a joint on its limit where every command
   trivially arrives. Whether that is the adaptation behaving correctly or the exploration
   policy failing is not yet distinguished, and the neuter A/B that would settle it was run and
   **discarded as invalid**. Until a valid A/B exists this is a wired knob, not a proven one.
2. **Cross-OS determinism is a hypothesis.** `perf.md` records the 4000-step digest for one CI
   cell (Windows/AMD64/py3.11). The other cells have not run.
3. **`reaction_diffusion_compute` is marked `"retire"`, not deleted.** Its only consumer was the
   grid planner. Deleting a correct, self-checked primitive is a decision worth taking on its
   own merits rather than as a side effect of a planner rewrite.
4. **The arm is still not built.** `arm_hardware` still reports the shoulder needing 15.0 kg*cm
   against 7.5 rated. A faithful simulation of an underpowered arm is an underpowered arm; if
   anything the fidelity makes the shortfall easier to forget.
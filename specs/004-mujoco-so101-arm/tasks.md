---

description: "Task list for MuJoCo SO-101 Becomes The Arm"
---

# Tasks: MuJoCo SO-101 Becomes The Arm

**Input**: Design documents from `/specs/004-mujoco-so101-arm/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/so101_arm.md, quickstart.md

**Baseline**: `c6ca8c8d`. `mujoco==3.11.0` already installed in `.venv-cu128` and `.venv-arm`.
SO-ARM100 cloned at `D:\AI-project\SO-ARM100` (Apache-2.0). Gate baseline: **277/277** self-checks,
pytest exit 0 at **92.53%** coverage, ruff clean.

**Tests**: INCLUDED. The constitution makes three gates binding, and this feature replaces the safety
gate every motor command passes through.

**Organization**: grouped by user story. Phase order follows plan.md's Execution Order.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelizable (different files, no incomplete dependency)
- **[Story]**: US1–US5 map to spec.md user stories
- Exact file paths in every description

---

## ⚠️ Read before starting

**The highest-severity item is a registration problem, not a design problem.** `_SAFETY_KERNEL`
(`godel_machine.py:148-156`) is a tuple of literal paths naming `arm_sim.py`, duplicated at
`self_improve.py:157-163`. Deleting that file without editing **both** literals in the **same commit**
silently removes the arm from the brain's immutable set while `NORD_SELFMOD=1` is live and self-build
fires every 8 nights. T009 adds the test that makes this impossible to miss; T066 is where it bites.

**The `body` contract is fixed in plan.md and data-model.md.** Do not invent keys. Three research areas
proposed three incompatible dicts; the contract is settled and tasks build to it.

---

## Phase 1: Setup

- [X] T001 Verify baseline: `git diff --quiet HEAD -- src/snn_data_lake/brainstem/` and record `git rev-parse HEAD` into `specs/004-mujoco-so101-arm/baseline.txt`; abort if the arm tree is dirty
- [X] T002 [P] Record the three gates at baseline (`python run_selfchecks.py`, `pytest`, `python -m ruff check .`) into `specs/004-mujoco-so101-arm/baseline.txt` — expect 277/277, exit 0, clean
- [X] T003 [P] Record the current `body` snapshot shape from a live daemon `GET /api/all` into `specs/004-mujoco-so101-arm/baseline.txt` as the pre-migration contract
- [X] T004 Vendor the SO-101 assets into `src/snn_data_lake/brainstem/so101/`: `scene.xml`, `so101_new_calib.xml`, `assets/*.stl`, and the upstream `SO-ARM100-LICENSE` (Apache-2.0). 15 files, 16.14 MB
- [X] T005 Verify the vendored directory name is not gitignored: confirm `so101/` is tracked by `git status --porcelain src/snn_data_lake/brainstem/so101/` — `models/` and `vendor/` are unanchored in `.gitignore:75-76` and hatchling honours it, so those names would silently omit the assets from the wheel
- [X] T006 Add `mujoco==3.11.0` to `[project] dependencies` in `pyproject.toml` — exact pin, following the `ruff==0.15.20` precedent, because the solver version IS the physics and SC-012 is a numeric assert against one build
- [X] T007 [P] Add `MUJOCO_GL: disable` to the self-check and test job envs in `.github/workflows/ci.yml` — declares "CI never renders" and exercises FR-012b's viewer-unavailable branch on all six cells
- [X] T008 Add a wheel-build smoke step to `.github/workflows/ci.yml`: build the wheel and assert the vendored `so101/scene.xml` is inside it. CI installs with `pip install -e .`, so the asset-packaging trap has no detector today

**Checkpoint**: assets vendored and packaged, dependency pinned, CI declares its GL posture.

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ CRITICAL**: no user story work may begin until this phase completes.

- [X] T009 Add `test_safety_kernel_paths_exist` to `tests/test_self_mod_safety.py`: assert every path in `godel_machine._SAFETY_KERNEL` exists on disk, and that the duplicate list at `self_improve.py:157-163` names the same set. This is the mechanism that stops the arm silently leaving the brain's immutable set (FR-020)
- [X] T010 Add the ordered `JOINTS` table to `src/snn_data_lake/brainstem/arm_config.py` per data-model.md: `qpos_index`, `name` (MJCF, canonical), `alias`, `token`, `servo`, `lever_mm`. Only alias/servo/lever are hand-written
- [X] T011 Add `JOINT_LIMITS` to `src/snn_data_lake/brainstem/arm_config.py`, **read from the model's `jnt_range` at import** from the vendored MJCF — never transcribed, so a model bump cannot desynchronise limits from vocabulary (P1, FR-009)
- [X] T012 Add `HOME_QPOS = zeros(6)` to `src/snn_data_lake/brainstem/arm_config.py` with a comment recording that the model has no keyframe (`nkey=0`), so this is repo-declared rather than model-derived
- [X] T013 [P] Add `test_joint_limits_match_model` to `tests/test_arm_config.py`: assert the imported limits equal the MJCF `jnt_range` and that a model fingerprint matches, so a silent asset swap fails loudly
- [X] T014 Convert `arm_hardware.JOINTS` in `src/snn_data_lake/brainstem/arm_hardware.py` into a derivation of `arm_config.JOINTS` (alias/servo/lever columns) rather than a second literal, keeping its existing assertions at `:194-203` working
- [X] T015 [P] Add `test_vocabulary_sites_agree` to `tests/test_arm_vocabulary.py`: enumerate every producer/validator of arm command tokens and assert each accepts exactly the tokens in `arm_config.JOINTS` plus the fixed set. There are **eight** such sites, not three
- [X] T016 Re-run all three gates with T009–T015 applied and append the result to `specs/004-mujoco-so101-arm/baseline.txt`

**Checkpoint**: single vocabulary, single limit source, and the safety-kernel test exists.

---

## Phase 3: User Story 1 — The comparison that makes the swap legal (Priority: P1)

**Goal**: one benchmark measuring both arms on the same inputs, plus the torque answer the build notes wanted.

**Independent Test**: `python tools/arm_benchmark.py` emits a comparison table and a verdict.

### Tests for User Story 1

- [ ] T017 [P] [US1] Add `test_benchmark_reports_both_arms` to `tests/test_arm_benchmark.py`: assert the benchmark emits a row for each arm on the shared command subset and never silently omits one (FR-001)
- [ ] T018 [P] [US1] Add `test_benchmark_torque_is_reported` to `tests/test_arm_benchmark.py`: assert per-joint N·m is present for the MuJoCo arm and explicitly reported as unavailable for `ArmSim` (FR-019)

### Implementation for User Story 1

- [ ] T019 [US1] Create `tools/arm_benchmark.py` with explicit CLI arguments and no value computed from no data; it must run against a bare MuJoCo body and the existing `ArmSim`, so it does not depend on the gate (FR-001)
- [ ] T020 [US1] Implement the shared-input comparison in `tools/arm_benchmark.py`: end-effector pose, contact verdict, and E-STOP behaviour side by side, naming what each arm structurally cannot report (SC-001)
- [ ] T021 [US1] Implement per-joint hold-torque readout in `tools/arm_benchmark.py` from `qfrc_bias` and `actuator_force`, in N·m against the ±3.350 N·m actuator limit (FR-019, SC-002)
- [ ] T022 [US1] Implement the 500 g payload case in `tools/arm_benchmark.py` — attach a payload body at the gripper TCP site via a repo-authored overlay that `<include>`s the vendored model, never by editing the vendored file (FR-012, SC-002)
- [ ] T023 [US1] Cross-check the benchmark's payload result against the repo's existing figure in `src/snn_data_lake/brainstem/arm_hardware.py` — `shoulder_pitch` 15.0 kg·cm required vs 7.5 rated, `margin_x: 0.5`, `ok: false`. Corroboration strengthens the claim; a discrepancy is a finding to explain, not average away
- [ ] T024 [US1] Write the P2 verdict into `specs/004-mujoco-so101-arm/benchmark.md`: whether the replacement is licensed, naming the inputs and what each arm could not report. A refusal legitimately stops the feature here (FR-001, FR-010)

**Checkpoint**: the swap is licensed on evidence, or it is not. The build notes' torque question is answered.

---

## Phase 4: User Story 2 — The gate survives the transplant (Priority: P2)

**Goal**: every safety property holds with MuJoCo as the body.

**Independent Test**: replay the gate's self-checks against the new body; behaviour indistinguishable where semantics are defined.

### Tests for User Story 2

- [ ] T025 [P] [US2] Add `test_estop_latches_and_freezes` to `tests/test_so101_arm.py`: assert every command except `HOME`/`RETRACT_ALL` is a no-op while latched (FR-002, SC-003)
- [ ] T026 [P] [US2] Add `test_estop_is_idempotent` to `tests/test_so101_arm.py`: apply `EMERGENCY_STOP` 1000× and assert state bit-identical to applying it once. Unguarded this drifts 35.16° over 20 000 repeats vs 0.0148° (FR-002)
- [ ] T027 [P] [US2] Add `test_home_is_bitwise_reproducible` to `tests/test_so101_arm.py`: HOME from several distinct poses, assert identical end-effector position each time — an equality assert in the shape of `arm_sim.py:201` (FR-002)
- [ ] T028 [P] [US2] Add `test_clamp_is_visible` to `tests/test_so101_arm.py`: command past a joint limit, assert `clamped` is true and the reported command matches what the arm received. MuJoCo's own clamp is silent (FR-009)
- [ ] T029 [P] [US2] Add `test_live_contact_reports_and_prevents` to `tests/test_so101_arm.py`: drive into the floor, assert verdict `"GROUND"` AND achieved angle short of commanded — strictly stronger than the assertion it replaces (FR-008)
- [ ] T030 [P] [US2] Add `test_probe_reports_without_preventing` to `tests/test_so101_arm.py`: query a penetrating pose, assert contact reported, `qpos` bit-unchanged, `qvel` zero (FR-008)
- [ ] T031 [P] [US2] Add `test_probe_rejects_out_of_limit` to `tests/test_so101_arm.py`: assert an out-of-range pose returns the out-of-limit verdict, not FREE — without this 65.9% of the planner grid is falsely free (FR-008, FR-009)
- [ ] T032 [P] [US2] Add `test_calibration_gate_no_bypass` to `tests/test_serial_arm.py`: with calibration unset, assert no command reaches the wire by any path (FR-003, SC-004)

### Implementation for User Story 2

- [ ] T033 [US2] Create `src/snn_data_lake/brainstem/so101_arm.py` as a gate that owns a body: load the vendored model once, hold the live `MjData`, expose `apply()` as the only mutation path (FR-005)
- [ ] T034 [US2] Implement the ordered checks in `src/snn_data_lake/brainstem/so101_arm.py` per `contracts/so101_arm.md` §1: latch check → vocabulary → clamp → actuate → report
- [ ] T035 [US2] Implement the E-STOP latch in `src/snn_data_lake/brainstem/so101_arm.py`: set `stopped`, run `ctrl[:] = qpos` **only on the False→True edge**, and KEEP STEPPING. Not "stop stepping" (leaves 5.45 rad/s stored), not `ctrl:=0` (a move-to-zero command, 25.89° of slew), not limp (38.32° collapse) (FR-002)
- [ ] T036 [US2] Implement HOME/RETRACT_ALL in `src/snn_data_lake/brainstem/so101_arm.py` as a hard reset (`mj_resetData`, `qpos:=HOME_QPOS`, `qvel:=0`, `qacc:=0`, `ctrl:=HOME_QPOS`, `mj_forward`), and handle the `d.time` rewind to 0 so nothing keyed on sim time sees time run backwards (FR-002)
- [ ] T037 [US2] Implement gate-side clamping in `src/snn_data_lake/brainstem/so101_arm.py` against **`actuator_ctrlrange`** — the range that binds a position command (clarified 2026-08-01) — setting `clamped` true exactly when the command was altered. `jnt_range` remains the physical stop enforced by the solver and is what `probe_contact` gates on; the two are measured unequal and must not be conflated (FR-009)
- [ ] T038 [US2] Implement the contact verdict in `src/snn_data_lake/brainstem/so101_arm.py` from `data.ncon` with labels resolved via `geom_bodyid` body names (30 of 31 geoms are unnamed). Verdict is `None | "GROUND" | "SELF"`; `OBSTACLE` is retired (FR-008)
- [ ] T039 [US2] Implement `probe_contact(qpos)` in `src/snn_data_lake/brainstem/so101_arm.py` against a module-private, **never-stepped** `MjData` using `mj_kinematics` + `mj_collision`, gated on `jnt_range`. It must be unreachable from the live body by construction — teleporting the live data into a penetrating pose ejects it at 36.11 rad/s (FR-008)
- [ ] T040 [US2] Implement `reached` / `blocked` / `tracking_error` / `deviation` in `src/snn_data_lake/brainstem/so101_arm.py` per `contracts/so101_arm.md` §3–§4, deriving `REACH_TOL` above the measured 0.0492° droop floor rather than picking it (FR-007, FR-009)

**Checkpoint**: the safety gate is rebuilt and demonstrably equivalent where semantics are defined.

---

## Phase 5: User Story 3 — The body becomes six-jointed (Priority: P3)

**Goal**: six joints in real units, and everything that speaks to the arm speaks six-joint.

**Independent Test**: command each joint individually; only that joint moves.

### Tests for User Story 3

- [ ] T041 [P] [US3] Add `test_each_joint_moves_alone` to `tests/test_so101_arm.py`: command each of the six joints, assert only the commanded joint changes (SC-005)
- [ ] T042 [P] [US3] Add `test_unknown_joint_rejected` to `tests/test_so101_arm.py`: assert a token naming a nonexistent joint is rejected, not silently ignored (FR-006)
- [ ] T043 [P] [US3] Add `test_reach_in_real_units` to `tests/test_so101_arm.py`: assert reported reach is in real units, is defined as the **maximum horizontal TCP radius** (~0.4795 m, within 1% of the model-derived value), and is NOT asserted equal to the 300 mm shoulder lever — they are different measurements (clarified 2026-08-01) (FR-007, SC-006)

### Implementation for User Story 3

- [ ] T044 [US3] Implement the token grammar `JOINT_<MJCF_NAME_UPPER>_ROTATE(<±d>deg)` in `src/snn_data_lake/brainstem/so101_arm.py`, checking the joint-name group for membership in `arm_config.JOINTS`. **The regex is syntax; the table is vocabulary** — a regex-only site must not be able to admit a joint that does not exist (FR-006)
- [ ] T045 [US3] Update `src/snn_data_lake/brainstem/interpreter.py` `COMMANDS` to the six-joint vocabulary, sourced from `arm_config.JOINTS` rather than a literal (FR-006)
- [ ] T046 [US3] Update the wire allow-list in `src/snn_data_lake/brainstem/serial_arm.py` to the six-joint vocabulary, preserving `HOLD`'s deliberate asymmetry and keeping validation independent of the mirror (FR-004, FR-006)
- [ ] T047 [US3] Give the autonomous path an allow-list in `src/snn_data_lake/brainstem/autonomous_loop.py`: `ACTIONS` must be built from `arm_config.JOINTS` and validated by the same gate as the chat path — today it hands a hardcoded token straight to `apply()` (FR-005)
- [ ] T048 [US3] Update `src/snn_data_lake/cortex/nl_interface.py`: its label set is a fourth literal and its hardcoded English "shoulder"→rotate_a / "elbow"→rotate_b mapping becomes wrong at six joints (FR-006)
- [ ] T049 [US3] Update `src/snn_data_lake/brainstem/cpg_oscillator.py`, which hardcodes joint A in its emitted token (FR-006)
- [ ] T050 [US3] Update the remaining producer/validator sites found by T015 under `src/snn_data_lake/` so all eight agree, and record the full list in `specs/004-mujoco-so101-arm/vocabulary-sites.md` (FR-006)
- [ ] T051 [US3] Re-derive `TRAJ_DEV_MAX` in `src/snn_data_lake/brainstem/arm_config.py` **under the loaded (500 g) condition**, with the load named and the derivation published. The current value is tied to a 2 m reach and consumed against a dimensionless distance (FR-007)
- [ ] T052 [US3] Define and document what `trajectory_breach()` returns during contact in `src/snn_data_lake/brainstem/so101_arm.py` — measured `|qpos − ctrl|` reaches 49.65° under contact, 58× the bound, and `spiking_pid.py:227` acts on the breach. Either contact-explained deviation is excluded, or the latch is accepted and recorded. Silence here is the defect (FR-007)
- [ ] T053 [US3] Make the coverage cell **3-D** in `src/snn_data_lake/brainstem/autonomous_loop.py` and re-derive its size with a published sweep written to `specs/004-mujoco-so101-arm/cover-cell-sweep.md` (clarified 2026-08-01). Carrying the old value forward is forbidden: the proposed 0.020 m re-measures at 791 reachable cells, not the claimed 1338, and 3-D at that size gives 12741. Three of five rotate joints move the TCP mainly in z (span 0.743 m), so a planar cell cannot observe them (FR-007)
- [ ] T054 [US3] Update the postural-recovery reflex in `src/snn_data_lake/brainstem/autonomous_loop.py`: `_A_POS` rebinds to `shoulder_pan`, a vertical-axis joint that cannot change TCP height, making the reflex a no-op that still counts (FR-005)
- [ ] T055 [US3] Migrate every consumer of the arm state shape to the `body` contract in `data-model.md` — `theta_deg`, `ee_mm`, `collision`, `deviation` readers across `src/` and `tools/` (FR-006)
- [ ] T056 [US3] Replace the 2-DOF exact-equality E-STOP proofs in `src/snn_data_lake/brainstem/autonomous_loop.py` and `src/snn_data_lake/hippocampus/ring_attractor.py` with assertions valid under physics — **without loosening them into vacuity**; state what each now proves (FR-002)

**Checkpoint**: six joints, real units, one vocabulary, every consumer migrated.

---

## Phase 6: User Story 4 — The operator can see the arm (Priority: P4)

**Goal**: the cockpit shows the real pose, honestly.

**Independent Test**: open the cockpit, command a joint, watch the pose change.

### Tests for User Story 4

- [ ] T057 [P] [US4] Add `test_png_encoder_roundtrip` to `tests/test_arm_view.py`: encode a known array with the stdlib encoder, assert per-chunk CRC32 and a zlib round-trip to identical pixels (FR-013)
- [ ] T058 [P] [US4] Add `test_viewer_unavailable_degrades` to `tests/test_arm_view.py`: with no GL context, assert the viewer reports unavailable and the process keeps running (FR-012b)

### Implementation for User Story 4

- [ ] T059 [US4] Implement the stdlib PNG encoder (`zlib` + `struct`, RGB, filter 0, one IDAT) in `src/snn_data_lake/substrate/arm_view.py`. **PIL must not be used** — it is present in `.venv-cu128` but declared only under the `[hf]` extra, so using it would depend on an undeclared package (FR-013)
- [ ] T060 [US4] Implement renderer ownership in `src/snn_data_lake/substrate/arm_view.py`: created and used on the daemon's tick thread ONLY, lazily, with frames cached. Cross-thread rendering returns garbage frames with only a warning, and **a lock does not fix it** (FR-013)
- [ ] T061 [US4] Add the render call to the daemon tick in `learn_daemon.py`, demand-gated so frames are produced only when the viewer has been requested (FR-015)
- [ ] T062 [US4] Add the `GET /api/arm.png` route to the `do_GET` if-chain in `src/snn_data_lake/substrate/cockpit.py`, returning bytes with `image/png` through the existing `_send`, and a non-200 with a reason when unavailable (FR-013, FR-012b)
- [ ] T063 [US4] Add the arm panel to the HTML string and `paint()` in `src/snn_data_lake/substrate/cockpit.py`, including a **revision counter** — a still picture is otherwise indistinguishable from a live one, the exact failure `PROVENANCE` exists to prevent (FR-013, SC-008)
- [ ] T064 [US4] Register every new panel key in `src/snn_data_lake/substrate/brain_telemetry.py` (`DASHBOARD_KEYS` / `DASHBOARD_SECTION_KEYS`) and confirm the no-orphan-panel gate passes (FR-014)

**Checkpoint**: the arm is visible, and the page cannot imply motion it is not showing.

---

## Phase 7: User Story 5 — Exactly one arm survives (Priority: P5)

**Goal**: the old implementation is gone, not disabled.

**Independent Test**: search the tree; find one arm.

- [ ] T065 [US5] Port every unique capability of `src/snn_data_lake/brainstem/arm_sim.py` BEFORE deletion and record each in `specs/004-mujoco-so101-arm/port-ledger.md`: E-STOP latch, trajectory comparator and its arming rule, collision authority, deterministic HOME. Do NOT port `pybullet_contacts()` — it has zero callers (FR-017)
- [ ] T066 [US5] Delete `src/snn_data_lake/brainstem/arm_sim.py` and update ALL FIVE registration sites **in the same commit**: `src/snn_data_lake/__init__.py` REGIONS, `run_selfchecks.py`, `src/snn_data_lake/substrate/dormancy_gate.py`, `src/snn_data_lake/reasoning/godel_machine.py:151`, `src/snn_data_lake/reasoning/self_improve.py:160`. T009's test must still pass (FR-016, FR-020)
- [ ] T067 [US5] Run `python run_selfchecks.py` and confirm the count still reads **277** after the swap; the old and new modules must be exchanged in one commit or the seven documents quoting that number go stale (SC-011)
- [ ] T068 [US5] **REPLACE** `src/snn_data_lake/reasoning/wavefront_planner.py` with a six-joint sampling planner (clarified 2026-08-01 — porting and retiring were both rejected). Abandon C-space enumeration (~2.6e9 cells at 6 DOF); sample candidate configurations and test each against `probe_contact`. It MUST use the probe as its single collision authority, MUST reject out-of-limit configurations (the probe alone does not), and MUST satisfy or explicitly replace the obstacle assertion at `:299-300` and the `deviation() < 1e-6` assertion at `:308` (FR-021, FR-008, FR-016)
- [ ] T068a [US5] Add `test_planner_finds_collision_free_path` to `tests/test_wavefront_planner.py`: assert the replacement finds a collision-free path when one exists, plans over all six joints, and never returns a path containing an out-of-limit or contacting configuration (FR-021)
- [ ] T069 [US5] Update `src/snn_data_lake/substrate/cockpit.py:562` and `src/snn_data_lake/brainstem/cpg_oscillator.py:62` for the retired `OBSTACLE` label, and confirm the contract is stated in exactly one place with every consumer demonstrably agreeing — record the consumer list in `specs/004-mujoco-so101-arm/vocabulary-sites.md` (FR-008, SC-007)
- [ ] T070 [US5] Rewrite the two honesty assertions in `src/snn_data_lake/brainstem/arm_hardware.py:217-218` (`sim_dof < target_dof`, `sim_has_dynamics is False`) to state what is newly true. **Do not delete or disable them** — a body with real dynamics needs the anti-overclaim mechanism more, not less (FR-012a)
- [ ] T071 [US5] Decide and record whether `src/snn_data_lake/substrate/demo.py`'s standalone dependency-free `Arm` is allow-listed or deleted under SC-010 (FR-016)
- [ ] T072 [US5] Add `test_exactly_one_arm` to `tests/test_arm_vocabulary.py`: assert a tree-wide search finds one arm implementation and no flag selecting between two (FR-016, SC-010)

**Checkpoint**: P2 closed. One arm.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T073 [P] Measure cross-OS determinism: hash a fixed 4000-step `qpos`/`qvel` trace on each CI matrix cell and compare, recording the result in `specs/004-mujoco-so101-arm/determinism.md`. **Currently UNVERIFIED** — all measurement is one Windows binary on one CPU, and MuJoCo makes no cross-platform bit-identity guarantee. FR-010/SC-012 cannot assert bitwise until this exists
- [ ] T074 [P] Measure and record per-tick arm cost in `specs/004-mujoco-so101-arm/perf.md`, asserting the tick rate does not regress beyond a stated bound (FR-015, SC-009)
- [ ] T075 [P] Make `blocked` **penalise the corresponding action** in autonomous selection in `src/snn_data_lake/brainstem/autonomous_loop.py`, so the arm stops choosing actions that cannot succeed, AND surface it in `src/snn_data_lake/substrate/brain_telemetry.py` (clarified 2026-08-01 — telemetry alone was rejected as not discharging P3). It fires on ~28% of in-limit targets (FR-005a, P3)
- [ ] T076 [P] Verify `MUJOCO_GL` handling in `src/snn_data_lake/brainstem/so101_arm.py`: an unrecognised value raises at import, and the daemon (which renders) and self-checks (which must not) cannot share one process-global setting
- [ ] T077 [P] Confirm `mujoco` resolves in BOTH the daemon env (`.venv-cu128`) and the gate env (`.venv-lake`), and that the GLFW hidden window coexists with the live torch CUDA context under `NORD_GPU=1` — record in `specs/004-mujoco-so101-arm/perf.md`. Untested interaction, and it only manifests in the daemon (FR-011)
- [ ] T078 Run `ruff format` over `src/snn_data_lake/brainstem/`, `src/snn_data_lake/substrate/`, `tools/` and `tests/`, then the full three-gate suite, and confirm 277/277 with no assertion loosened and coverage above the 70% floor (FR-018, SC-011)

---

## Dependencies

```
Phase 1 (Setup: assets, pin, CI)
   ↓
Phase 2 (Foundational: safety-kernel test, vocabulary, limits) ── BLOCKING
   ↓
Phase 3 US1 (benchmark) ── P2 eligibility; may REFUSE the swap and stop the feature
   ↓
Phase 4 US2 (the gate)
   ↓
Phase 5 US3 (6-DOF migration)
   ↓
Phase 6 US4 (cockpit)
   ↓
Phase 7 US5 (retire ArmSim) ── the five-site commit
   ↓
Phase 8 (Polish)
```

**Hard gates**

- T009 must exist before T066, or the arm can silently leave the safety kernel.
- T024 refusing the swap **stops the feature** — a legitimate outcome under FR-001/FR-010.
- T066 must edit all five registration sites in one commit; T009 is the detector.
- T068 may conclude the planner is unportable; that is recorded debt, not a failure.
- T073 gates any bitwise claim in FR-010/SC-012.

## Parallel Opportunities

- **Phase 1**: T002, T003 after T001; T007 alongside T006.
- **Phase 2**: T013, T015 in parallel after T010–T012.
- **Phase 3**: T017, T018 in parallel; T019–T024 sequential (one file).
- **Phase 4**: T025–T032 all parallel (eight independent test additions); T033–T040 sequential.
- **Phase 5**: T041–T043 parallel; T045, T046, T048, T049 parallel (different files).
- **Phase 6**: T057, T058 parallel; T059–T064 sequential.
- **Phase 8**: T073–T077 all parallel; T078 last.

## Implementation Strategy

**MVP is Phase 1 + Phase 2 + Phase 3.** The smallest increment that delivers value is the vendored
model, the single vocabulary, the safety-kernel test, and the benchmark — because the benchmark answers
the question that motivated the build notes (can STS3215 servos hold 500 g at 300 mm) and it licenses or
refuses everything after it. None of that touches the live arm path.

**Incremental delivery**

1. Phases 1–2 — assets, vocabulary, and the test that protects the safety kernel. Ship.
2. Phase 3 — the benchmark and the torque answer. Ship. **May legitimately end the feature.**
3. Phase 4 — the gate, off the live path until US3 wires it.
4. Phase 5 — the migration. This is the wide, risky one.
5. Phases 6–7 — visible payoff, then retirement.

**What would legitimately stop this early**: T024's benchmark refusing the swap, or T068 concluding the
planner cannot be ported. Both are recorded outcomes, not failures.

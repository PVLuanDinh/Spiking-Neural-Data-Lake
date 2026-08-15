---
description: Task list for 014-live-sensor-transduction
---

# Tasks: Live Simulated Sensory Transduction

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/sensor-endpoint.md](contracts/sensor-endpoint.md),
[quickstart.md](quickstart.md)

**Branch**: `014-live-sensor-transduction` | **Gate env**: `.venv-lake`

## Format: `[ID] [P?] [Story] Description`

- **[P]** = parallelizable (different files, no dependency on an incomplete task)
- **[US1]..[US4]** = user story from spec.md
- Setup / Foundational / Polish phases carry NO story label

## Path Conventions

Repo root `D:\AI-project\spiking-neural-data-lake`. Source under `src/snn_data_lake/`, tests under
`tests/`, evidence under `specs/014-live-sensor-transduction/evidence/`.

---

## THE GATE BATCH — run at EVERY commit, not only at the end

```bash
.venv-lake/Scripts/python.exe run_selfchecks.py && .venv-lake/Scripts/python.exe -m pytest -q && .venv-lake/Scripts/python.exe -m ruff check . && .venv-lake/Scripts/python.exe -m ruff format --check . && .venv-lake/Scripts/python.exe tools/p4c_ratchet.py --assert && .venv-lake/Scripts/python.exe tools/p6_gate.py --assert && .venv-lake/Scripts/python.exe tools/citation_guard.py --assert && .venv-lake/Scripts/python.exe tools/red_reachability.py --assert && .venv-lake/Scripts/python.exe tools/dependency_ledger.py --assert && .venv-lake/Scripts/python.exe -m snn_data_lake.substrate.import_manifest
```

`ruff format --check` is in the batch deliberately — a prior arc shipped drift without it.

## SAFETY SCOPE — binding on every task below

`brainstem/reflex.py` is the ONLY safety-envelope file any task may touch, and only to ADD a channel
rule. `serial_arm.py`, `so101_arm.py`, `arm_config.py`, `interpreter.py` are **not touched by any
task in this list**. If a task appears to require touching one, that is an error in this list —
stop and report rather than editing.

## ORDERING CHANGE FROM research.md D-5, AND WHY

D-5 sequenced the charge probe with the other dispositions in C5. **It moves earlier: the probe is a
GO/NO-GO gate at the end of C3 (Phase 6), before the envelope is touched.** FR-012's revert must
never have to unwind a safety-envelope commit. Consequence, stated rather than hidden: at C3 only
two of the three attribution paths exist (amygdala, spike/perception) — the reflex path requires
channel 8. So the probe runs **twice**: a two-path GO/NO-GO at C3, and full three-path attribution
after C4. A two-path null stops the lane before `reflex.py` is opened at all.

---

## Phase 1: Setup — capture baselines BEFORE any change

Verify-first: a "no change" claim later is only checkable against a number recorded now.

- [ ] T001 Create `specs/014-live-sensor-transduction/evidence/` for the recorded reds, tables and probe output referenced by later tasks
- [ ] T002 [P] Record the pre-change gate batch result (all ten commands, full output) to `specs/014-live-sensor-transduction/evidence/baseline-gate.txt`
- [ ] T003 [P] Record baseline `docs/coverage/` band values for `amygdala_fear_conditioning` and every row this lane could touch, to `specs/014-live-sensor-transduction/evidence/baseline-coverage.txt` — this is the file T060 diffs against
- [ ] T004 [P] Record baseline `tools/p4c_ratchet.py` and `tools/p6_gate.py` measured counts to `specs/014-live-sensor-transduction/evidence/baseline-ratchets.txt`, so a later "unchanged" claim is measured not inferred
- [ ] T005 [P] Record the baseline live `threat()` behaviour — run the brain for >= 80 wake ticks and record the call count and every returned value to `specs/014-live-sensor-transduction/evidence/baseline-threat.txt`; the module claims 38 calls / 0.0 every time, and this lane's whole premise rests on it
- [ ] T006 [P] Record baseline `git diff --name-only main -- src/snn_data_lake/brainstem/` (expected: empty) to `specs/014-live-sensor-transduction/evidence/baseline-envelope-diff.txt`

**Checkpoint**: baselines exist for gate, coverage, ratchets, threat, envelope diff.

---

## Phase 2: Foundational — the reading and its queue (BLOCKING for all stories)

Both the simulator (C1) and the cockpit endpoint (C2) produce into the same queue; the tick consumer
(C3) drains it. The queue is shared plumbing, so it lands before either producer.

**This phase produces into a queue that nothing consumes yet. That is intentional** — C3 is the
activation commit, and both feeds must exist before the thing that makes them live.

- [ ] T007 Define the `SensorReading` value in `src/snn_data_lake/substrate/sensor_feed.py` with fields per data-model E-1/E-4: `distance_m` (raw metres), `temp_k` (raw kelvin), `simulated` (bool, **required, never defaulted**), and a `wire()` method emitting the `"distance_m, temp_k"` format `parse_esp32_line` expects
- [ ] T008 Implement the bounded `SensorQueue` (submit/drain) in `src/snn_data_lake/substrate/sensor_feed.py`, mirroring the `Acts` queue pattern in `src/snn_data_lake/substrate/cockpit.py`; drain happens on the tick thread only
- [ ] T009 Make a `SensorReading` constructed without an explicit `simulated` value raise, in `src/snn_data_lake/substrate/sensor_feed.py` — **absence is a failure, never a default**; defaulting to `False` makes a dropped marker read as REAL, defaulting to `True` would mislabel a real reading once hardware exists
- [ ] T010 [P] Register `substrate.sensor_feed` in `src/snn_data_lake/__init__.py` REGIONS and in `run_selfchecks.py` (CORE tier) — a module in neither is invisible to the three P3 gates
- [ ] T011 [P] Write the module `__main__` self-check in `src/snn_data_lake/substrate/sensor_feed.py` asserting round-trip `SensorReading -> wire() -> parse_esp32_line` returns the original raw values
- [ ] T012 Write `tests/test_sensor_feed.py` covering queue submit/drain, bounded back-pressure, and the T009 required-marker raise **with its demonstrated red** (construct the no-marker case, record the failure output to `evidence/red-marker-required.txt`)

**Checkpoint**: `sensor_feed` registered, self-checked, gate batch GREEN. **COMMIT C0.**

---

## Phase 3: User Story 1 (producer half) — the daemon simulates an environment (P1) — COMMIT C1

**Goal**: the daemon produces a stream of simulated readings in the wire format, default-OFF.

**Independent test**: enable the simulator, observe readings appear on the queue; disable it and
observe behaviour byte-identical to today.

- [ ] T013 [US1] Implement `SimulatedEnvironment` in `src/snn_data_lake/substrate/sensor_feed.py` with **per-instance `random.Random(seed)`** — never module-level seeding; specs/015 just repaired exactly that leak and reintroducing one would undo it in the same week
- [ ] T014 [US1] Give `SimulatedEnvironment` a `forced` hook (drive straight to a chosen distance) in `src/snn_data_lake/substrate/sensor_feed.py` — without it the charge probe cannot reach the danger zone and every downstream measurement is vacuous
- [ ] T015 [US1] Use `arm_config.CONTACT` as the simulator's danger limit in `src/snn_data_lake/substrate/sensor_feed.py` — **read it, do not redefine it**. P1 exception (d): a raw physical limit inside the envelope stays FIXED. No new safety constant, nothing learned or baseline-relative
- [ ] T016 [US1] Wire the producer into `src/snn_data_lake/substrate/learn_daemon.py` beside `acts = Acts()` (~`:263`), **default-OFF** per FR-014, gated on an explicit opt-in; construct nothing at module scope (the promotion of this file already exposed three module-scope bugs)
- [ ] T017 [US1] Write `tests/test_sensor_sim.py` asserting determinism: same seed → identical reading sequence; different seed → different sequence
- [ ] T018 [US1] Add to `tests/test_sensor_sim.py` the FR-014 default-OFF assertion — with the simulator not enabled, the daemon's tick behaviour and state report are unchanged from baseline
- [ ] T019 [US1] Re-derive every `file:line` cite touched by this commit AFTER the last edit, then run the gate batch

**Checkpoint**: simulator produces, default-OFF proven, gate batch GREEN. **COMMIT C1.**

---

## Phase 4: User Story 2 — the cockpit accepts an afferent sensor reading (P1) — COMMIT C2

**Goal**: `POST /api/sensor` per [contracts/sensor-endpoint.md](contracts/sensor-endpoint.md).

**Independent test**: POST a reading, observe it enqueued and the documented status codes returned,
without the request thread touching brain, reflex or arm.

- [ ] T020 [US2] Add the `POST /api/sensor` branch to `do_POST` in `src/snn_data_lake/substrate/cockpit.py`, mirroring the `/api/act` pattern at `:360-372` — **validate on the request thread, enqueue, never touch the brain**
- [ ] T021 [US2] Return **404** when `serve()` was given no sensor queue, in `src/snn_data_lake/substrate/cockpit.py` — this is what makes the capability structurally absent rather than merely disabled, exactly as `/api/act` does
- [ ] T022 [US2] Validate `simulated` present and `true` (else 400) in `src/snn_data_lake/substrate/cockpit.py` — the endpoint cannot originate a real reading (`pyserial` absent), so accepting `false` would be offering to mislabel
- [ ] T023 [US2] Validate `distance_m` finite and non-negative (else 400) in `src/snn_data_lake/substrate/cockpit.py` — the sentinel rejection of data-model E-2, applied at the boundary
- [ ] T024 [US2] Return **202** (not 200) on accept in `src/snn_data_lake/substrate/cockpit.py` — the reading is queued, not acted on; 200 would let the cockpit claim an effect the brain has not had (P4c: what the cockpit REPORTS must be what the brain MEASURES)
- [ ] T025 [US2] Thread the sensor queue through `serve()`'s signature in `src/snn_data_lake/substrate/cockpit.py` and pass it from `src/snn_data_lake/substrate/learn_daemon.py:265`
- [ ] T026 [US2] Write `tests/test_cockpit_sensor.py` covering the four reachable reds from the contract: missing `simulated` → 400; `{"distance_m": -1, ...}` → 400; non-finite → 400; no queue → 404. **Record each red's output** to `evidence/red-endpoint-*.txt`
- [ ] T027 [US2] Add to `tests/test_cockpit_sensor.py` the assertion that the request thread never constructs a Brain and never calls reflex or arm code — the property the `/api/act` precedent exists to preserve
- [ ] T028 [US2] Re-derive every `file:line` cite touched by this commit AFTER the last edit, then run the gate batch

**Checkpoint**: endpoint validates and enqueues; nothing consumes yet. Gate batch GREEN. **COMMIT C2.**

---

## Phase 5: User Story 1 (activation half) — the tick consumes the feed (P1) — COMMIT C3

**THE ACTIVATION COMMIT.** This is where the amygdala can first fire, so the SIMULATED marker and
the `fear_conditioning.py` disposition amendment land here, in the same commit — not after.

**Independent test**: drive a simulated reading; observe it transduced through the EXISTING chain and
reaching `AutonomousArm.step`'s `distance` argument, with the marker intact at every hop.

### The two measured Phase-1 findings — these are the highest-value tasks in the lane

- [ ] T029 [US1] Implement the **idle-sentinel guard** in the tick consumer in `src/snn_data_lake/substrate/sensor_feed.py`: a negative / sentinel `distance_m` is a MISSING reading (like `temp_k <= 0`), never a close one. Without it, `-1.0 < CONTACT (0.04)` is True and `math.isfinite(-1.0)` is True, so `AutonomousArm.step:1473` fires and the amygdala learns that **an empty room is the threat**
- [ ] T030 [US1] Demonstrate T029's red: remove the guard, feed `"-1, -1"`, confirm the contact branch fires and `pair()` is called, **record the output** to `evidence/red-sentinel.txt`, restore the guard. A guard never seen to fail is not a guard
- [ ] T031 [US1] Add a comment in `src/snn_data_lake/substrate/sensor_feed.py` recording WHY the guard is in the consumer and not in `normalize()` — modifying `normalize()` would change the feature vector every existing caller receives, and the non-goals forbid rewriting the governed chain (P2: reuse, do not modify)
- [ ] T032 [US1] Assert in `tests/test_sensor_feed.py` that the safety path receives **RAW METRES** and the spike path receives the normalized feats, and that the two are never substituted — `CONTACT = 0.04` is metres while `distance01 = distance_m / 3.0`, so feeding normalized to a metres threshold moves the trip point from 4 cm to 12 cm, a silent 3x loosening
- [ ] T033 [US1] Demonstrate T032's red: substitute the normalized value for the raw one, confirm the assertion fails and the effective trip point moves, **record it** to `evidence/red-raw-vs-normalized.txt`

### The consumer — feed the one that exists, do not build one

- [ ] T034 [US1] Drain the sensor queue on the TICK thread in `src/snn_data_lake/substrate/learn_daemon.py` beside the existing acts drain (~`:292`), never on the request thread
- [ ] T035 [US1] Route each drained reading through the **EXISTING** chain — `parse_esp32_line -> encode_reading -> spikes_for` from `src/snn_data_lake/substrate/esp32_encoder_path.py`. **Do not write a second parser and do not modify that module** (P2: exactly one transduction mechanism)
- [ ] T036 [US1] Pass the raw `distance_m` to `AutonomousArm.step`'s `distance` argument (`src/snn_data_lake/brainstem/autonomous_loop.py:1469`) in place of the always-safe `safe_d` at `:2269`, only when a simulated reading is present. **The consumer at `:1473`/`:1479` already exists — this task feeds it, it does not build it**

### The owner's ruling, as tasks rather than prose

- [ ] T037 [US1] Propagate the SIMULATED marker to the brain state report in `src/snn_data_lake/substrate/learn_daemon.py` — carried as a field ON THE READING, never re-derived from a global "is the simulator on?" flag (a global can be true while a specific reading came from elsewhere)
- [ ] T038 [US1] Propagate the SIMULATED marker into any lake record written, under its own record tag in `brain_lake.bin` (P4b — no second store), in `src/snn_data_lake/substrate/learn_daemon.py`
- [ ] T039 [US1] Render the SIMULATED marker next to the value in `src/snn_data_lake/substrate/cockpit.py` — P4c: a simulated threat must never be presentable as real
- [ ] T040 [US1] Write `tests/test_sensor_marker.py` with **THREE separate reds** — strip the marker at the state-report hop, the lake-record hop, and the cockpit-display hop, one at a time, confirming failure at each; record to `evidence/red-marker-hop-{state,lake,cockpit}.txt`. A single end-to-end assertion can pass while an intermediate hop drops the marker and a later hop re-derives it
- [ ] T041 [US1] **Amend** the DISPOSITION note in `src/snn_data_lake/amygdala/fear_conditioning.py` — do not silently overtake it. It must state: a simulated body satisfies the prerequisite for LEARNING but **not** for SCORING; owner authorized 2026-08-13; the semantics are unchanged because the amygdala still conditions on CONTACT (nociception), not on the reward signal the note pre-refuses
- [ ] T042 [US1] Re-derive every `file:line` cite touched by this commit AFTER the last edit, then run the gate batch

**Checkpoint**: feed reaches the existing consumer, marker survives all hops, gate batch GREEN.
**COMMIT C3.**

---

## Phase 6: THE CHARGE GATE — GO / NO-GO before the envelope is opened

**This phase decides whether the lane continues.** It runs before `reflex.py` is touched so that
FR-012's revert never has to unwind a safety-envelope commit.

- [ ] T043 **Verify the verifier first**: confirm the probe harness is sensitive at all, using the external positive control in `tools/charge_probe.py` (47/60 flips with all drive sources neutered), or demonstrate this probe's own sensitivity. **A null from a harness that has never reported a difference is not a null** (Standing Invariant 5)
- [ ] T044 Run the neuter-A/B for the **amygdala path**: simulator ON and driven into the danger zone vs readings produced but not consumed, same seed. Record to `evidence/charge-amygdala.txt` whether `pair()` firing changes an outcome the brain would otherwise reach
- [ ] T045 Run the neuter-A/B for the **spike / perception path** separately, recording to `evidence/charge-spike.txt`. Per-item attribution is required — **"something changed" is not attribution** (P8 guard 3)
- [ ] T046 Write the GO/NO-GO verdict to `specs/014-live-sensor-transduction/evidence/charge-verdict.md`, naming which path charges and which does not, and disclosing that the reflex path is not yet measurable because channel 8 does not exist
- [ ] T047 **If BOTH paths are null: STOP and execute FR-012** — revert C1–C3, restore both dormancy exemptions with corrected reasons, do NOT proceed to Phase 7, and report the null plainly. Deletion on a null is the licence working, not failing

**Checkpoint**: GO (at least one path charges) → Phase 7. NO-GO → FR-012 revert, lane ends here.

---

## Phase 7: User Story 3 — the feed reaches the safety envelope (P1) — COMMIT C4

**THE ONLY COMMIT THAT TOUCHES A SAFETY-ENVELOPE FILE.** Only `brainstem/reflex.py`, only to ADD.

**Independent test**: drive a simulated obstacle to close range; observe a protective action; confirm
every pre-existing stop still fires exactly as before.

- [ ] T048 [US3] Add channel **8** (simulated proximity) to `DEFAULT_RULES` in `src/snn_data_lake/brainstem/reflex.py`. **Strictly above 6 and 7 — never insert, never renumber.** Rules are index-keyed into a raw sample list, so a layout shift would silently re-map motor current and force/torque onto the wrong data
- [ ] T049 [US3] Write `tests/test_reflex_channel_pin.py` pinning `DEFAULT_RULES[6] == (-50.0, 800.0, "STOP")` and `DEFAULT_RULES[7] == (-300.0, 300.0, "WITHDRAW")`, and that indices 6 and 7 still carry motor current and force/torque
- [ ] T050 [US3] Demonstrate T049's red: shift the frame layout by one, confirm the pin test fails, **record it** to `evidence/red-channel-pin.txt`, restore. Nothing in the tree pins this today — without the red the test merely restates the source
- [ ] T051 [US3] Build the **8-row caution matrix** in `tests/test_reflex_caution.py`: for each combination of ch6 breach / ch7 breach / neither / both, compare the reflex outcome WITH the simulated channel present against WITHOUT it, same input sequence and seed
- [ ] T052 [US3] Sweep ch8 across safe → danger in `tests/test_reflex_caution.py` rather than testing one value — absence of a warning in one case is not absence in general (Standing Invariant 4)
- [ ] T053 [US3] Assert in `tests/test_reflex_caution.py` that no row fires LATER or selects a LESS severe action than baseline. The claim is **"same or more protective, never less"** and it must be a MEASURED TABLE, not research.md D-3's structural argument
- [ ] T054 [US3] Record the completed matrix to `specs/014-live-sensor-transduction/evidence/caution-table.md` — SC-004 requires it demonstrated, and a structural argument that was never run is still an argument
- [ ] T055 [US3] Assert the non-finite row explicitly in `tests/test_reflex_caution.py`: a non-finite simulated reading yields severity `inf` and fires. This behaviour is INHERITED from `reflex.py`, not re-implemented — confirm no new code was added for it
- [ ] T056 [US3] Verify envelope scope by diff, not by promise: `git diff --name-only main -- src/snn_data_lake/brainstem/` must list `reflex.py` and **nothing else**. Also assert `arm_config.SAFETY_CALIBRATED is False` and `serial_arm.assert_calibrated()` still refuses. Record to `evidence/envelope-diff.txt`
- [ ] T057 [US3] Re-derive every `file:line` cite touched by this commit AFTER the last edit, then run the gate batch

**Checkpoint**: channel 8 live, 6/7 pinned with a red, caution table measured, envelope diff clean.
**COMMIT C4.**

---

## Phase 8: User Story 4 — the exemption is settled honestly (P2) — COMMIT C5

- [ ] T058 [US4] Re-run the charge probe with **all three** paths now available (reflex, amygdala, spike/perception) and record full per-item attribution to `specs/014-live-sensor-transduction/evidence/charge-attribution.md`
- [ ] T059 [US4] Settle the `esp32_encoder_path` exemption in `src/snn_data_lake/substrate/dormancy_gate.py:255` per what was actually PROVEN — the stated reason ("AWAITING HARDWARE... pyserial absent") is broader than the real blockage, since only `read_serial_lines` needs `pyserial` and `parse -> encode -> spikes` never did. **Correct or lift it based on the probe result, never on the existence of a wire** (Standing Invariant 2: removing an exemption because a wire exists rather than because it charges is silencing a check)
- [ ] T060 [US4] **Diff `docs/coverage/` against T003's baseline and assert NO BAND ROSE.** This is the condition the owner's authorization was given under — a checked condition, not an intention. A band that rose on simulated evidence fails the lane
- [ ] T061 [US4] Settle the `fear_conditioning` disposition consistently with T041 and T060: learning enabled, scoring frozen at band 0, in `src/snn_data_lake/substrate/dormancy_gate.py` and `src/snn_data_lake/amygdala/fear_conditioning.py`
- [ ] T062 [US4] Re-run T005's threat measurement and record the new `threat()` call count and value distribution to `evidence/threat-after.txt`, alongside the baseline. This is the lane's headline number and it must be measured, not inferred
- [ ] T063 [US4] Disclose in `specs/014-live-sensor-transduction/evidence/charge-attribution.md` the honest limit, unprompted (P8 guard 4): **a brain that has only ever been hurt in simulation has learned about a simulation.** `threat()` becoming non-zero is evidence the mechanism works, NOT evidence of a validated threat model of the world
- [ ] T064 [US4] Verify `tools/p4c_ratchet.py` and `tools/p6_gate.py` counts against T004's baseline — this lane adds no cross-region import, so the ratchets should be unchanged; **measure it rather than asserting it**
- [ ] T065 [US4] Re-derive every `file:line` cite touched by this commit AFTER the last edit, then run the gate batch

**Checkpoint**: exemptions settled on evidence, coverage band verified unchanged. **COMMIT C5.**

---

## Phase 9: Polish & Cross-Cutting

- [ ] T066 [P] Verify every new criterion in this lane has a RECORDED red in `specs/014-live-sensor-transduction/evidence/` — enumerate them and confirm none is missing. A criterion without a demonstrated red is DELETED, not weakened (P8 guard 1)
- [ ] T067 [P] Confirm no second transduction mechanism was introduced: grep for any new parsing of the wire format outside `esp32_encoder_path.py`, and confirm that module is unmodified (`git diff main -- src/snn_data_lake/substrate/esp32_encoder_path.py` empty)
- [ ] T068 [P] Confirm no module-level seeding was introduced: grep the lane's diff for `random.seed(` and `torch.manual_seed(` at module scope
- [ ] T069 [P] Confirm no second durable store: anything persisted goes to `brain_lake.bin` under its own record tag (P4b)
- [ ] T070 Run `graphify update .` to keep the knowledge graph current after the source changes
- [ ] T071 Final full gate batch from a clean checkout, plus `git diff --name-only main -- src/snn_data_lake/brainstem/` one last time before push

---

## Dependencies & Execution Order

### Phase dependencies

```
Phase 1 (Setup, baselines)
   |
Phase 2 (Foundational: reading + queue)          <- BLOCKS everything
   |
   +-- Phase 3 (US1 producer, C1) ---+
   |                                  |          <- C1 and C2 are independent:
   +-- Phase 4 (US2 endpoint, C2) ---+              both produce into the queue,
                                      |             nothing consumes yet
                                      v
                    Phase 5 (US1 activation, C3)  <- THE ACTIVATION COMMIT
                                      |
                                      v
                    Phase 6 (CHARGE GATE)         <- GO/NO-GO. NO-GO ends the lane.
                                      |  GO
                                      v
                    Phase 7 (US3 envelope, C4)    <- the ONLY envelope commit
                                      |
                                      v
                    Phase 8 (US4 dispositions, C5)
                                      |
                                      v
                    Phase 9 (Polish)
```

### User story dependencies

- **US1** spans two commits (C1 producer, C3 activation) with US2's C2 landing between them, because
  the tick consumer drains a queue that BOTH feeds fill. Building the consumer first would mean
  building it twice.
- **US2** (C2) is independent of US1's producer — both write to the Phase-2 queue.
- **US3** (C4) depends on Phase 6 returning GO.
- **US4** (C5) depends on US3 for the third attribution path.

### Parallel opportunities

- **Phase 1**: T002–T006 all parallel (independent evidence files)
- **Phase 2**: T010, T011 parallel after T007–T009
- **Phase 3 / Phase 4**: C1 and C2 are independent commits and may be developed in parallel, though
  each still lands with its own green gate batch
- **Phase 6**: T044 and T045 parallel (different attribution paths, separate evidence files)
- **Phase 9**: T066–T069 all parallel

### NOT parallelizable

- Anything in Phase 5 after T034 (single tick path, ordering matters)
- Everything in Phase 7 (single envelope file)
- Phase 6 gates Phase 7 absolutely — no overlap

---

## Implementation Strategy

### MVP

**Phases 1–3 (through C1)** is the smallest honest increment: a deterministic, default-OFF simulated
environment, registered and gated, producing into a queue. It proves the producer without activating
anything.

**But the MVP does not settle the feature's question.** FR-010's charge probe (Phase 6) is what
decides whether any of it stays. Shipping C1 alone would be a wire with no verdict.

### Incremental delivery

1. C0 (foundational) → C1 (producer) → C2 (endpoint): three green commits, nothing activated
2. C3: activation. Amygdala can fire. Marker enforced. Disposition amended
3. **Phase 6 GATE**: GO/NO-GO. A null here reverts C1–C3 and the envelope is never opened
4. C4: envelope, only on GO
5. C5: dispositions settled on measured evidence

### The failure mode this ordering exists to prevent

The lane's natural pull is to reach the envelope early — it is the authorized, interesting part.
Sequencing the charge gate before C4 means a null result costs three reversible commits instead of a
revert that touches `reflex.py`.

---

## Notes

- **The consumer already exists.** `AutonomousArm.step` (`autonomous_loop.py:1469`) calls
  `am.pair(pat)` at `:1479` when `distance < CONTACT` at `:1473`. It has never fired because the live
  caller at `:2269` passes `safe_d`, derived from `SAFE_DISTANCE = CONTACT * 3.0`, documented at
  `brain.py:71` as "never in a reflex/danger zone". **Unreachable by construction, not broken.** No
  task builds a consumer; tasks feed this one.
- **`esp32_encoder_path.py` is not modified by any task.** P2: reuse the one transduction mechanism.
- Cite discipline: every `file:line` in the lane's docs and commit messages is re-derived AFTER the
  last edit of the commit that touches it.
- Line counts and cite line numbers above were measured at branch HEAD and will drift as commits
  land; re-derive before quoting them.

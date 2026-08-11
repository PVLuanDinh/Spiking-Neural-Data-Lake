# Tasks: Finish the GeNN Port of the Burst+x_tar MNIST Learner

**Input**: Design documents from `/specs/012-finish-genn-port/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/module-contract.md, quickstart.md

**Tests**: No separate test tasks â€” per project convention the module's own self-check carries
the falsifiable checks (contract: module-contract.md), and the measured runs ARE the
verification. Tasks below reference them explicitly.

**Organization**: By user story. NOTE the deliberate phase order US1 -> US4 -> US2 -> US3:
governance (US4) runs BEFORE the finish (US2) so every later commit lands inside the gates â€”
P4c's "written, registered, self-checked and gated as part of the work that creates it".

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

**Purpose**: Evidence trail exists before the first measurement.

- [X] T001 Create `specs/012-finish-genn-port/evidence.md` with the E4 record skeleton
      (preflight / cpu-baseline / parity / control / determinism / payoff-300 / payoff-1600 /
      red-demo sections, each "PENDING") per data-model.md

**Checkpoint**: evidence.md exists; every later task appends verbatim output to it.

---

## Phase 2: Foundational

No foundational tasks â€” the pre-flight IS User Story 1 and everything else depends on it
directly.

---

## Phase 3: User Story 1 - Toolchain pre-flight (Priority: P1) â€” MVP, BLOCKING

**Goal**: The `[genn]` toolchain verified working in `.venv-lake`, or the blocker recorded
verbatim and the feature stopped at zero port commits.

**Independent Test**: quickstart.md Â§0 â€” install, import, smoke compile/run all pass, or STOP.

- [X] T002 [US1] Repoint the `[genn]` extra in `pyproject.toml` line ~244 to
      `pygenn @ https://github.com/genn-team/genn/archive/refs/tags/5.4.0.zip` (keep `ml_genn`;
      research.md R1). Do NOT commit until T005 verifies the install.
- [X] T003 [US1] Run `.venv-lake\Scripts\pip install -e .[genn]` from repo root; record the
      output tail + `pygenn.__version__` in evidence.md Â§preflight. If the install fails,
      attempt the documented fallbacks IN ORDER (VS developer environment for compiler
      discovery; legacy build deps `pip install pybind11 psutil pkgconfig "setuptools>=61"`;
      research.md R1-R3) â€” each attempt's verbatim failure recorded.
- [X] T004 [US1] Write `specs/012-finish-genn-port/genn_smoke.py` (tracked, so citation_guard
      stays green): minimal GeNNModel â€” one SpikeSourceArray(4), one LIF(2), one dense synapse
      â€” `build()` (nvcc compile), `load()`, 10x `step_time()`, print recorded spikes + backend
      name, ASCII only. Run it in `.venv-lake`; record output + any GIL-re-enable
      RuntimeWarning in evidence.md Â§preflight (research.md R3).
- [X] T005 [US1] STOP GATE: If T003+T004 both pass â€” run the three gates
      (quickstart.md Â§1), commit `pyproject.toml` + `evidence.md` + `genn_smoke.py` +
      spec-directory artifacts as "specs/012: the [genn] extra stands up". If EITHER fails
      after fallbacks â€” revert T002, set the docs/P4C_BACKLOG.md row to BLOCKED with the
      verbatim error, commit ONLY the evidence + ledger change, and HALT the feature awaiting
      an owner decision (spec US1-AS3).

**Checkpoint**: pygenn imports and compiles on the RTX 5070, or the feature is cleanly stopped.

---

## Phase 4: User Story 4 - The module enters governance (Priority: P2)

**Goal**: The (still-skeleton) module governed: moved, registered, dispositioned, linted â€”
so the finish lands inside the gates.

**Independent Test**: All three gates green with the module registered (spec US4-AS1/AS2)
while it is still a stub â€” governance is proven independent of the port being finished.

- [X] T006 [US4] `git mv src/snn_data_lake/reasoning/snn_mnist_stdp_genn.py
      src/snn_data_lake/plasticity/snn_mnist_stdp_genn.py` (zero code importers exist â€”
      verified 2026-08-07; research.md R7)
- [X] T007 [US4] Register the module: REGIONS entry under `plasticity` in
      `src/snn_data_lake/__init__.py`; self-check registration in `run_selfchecks.py`;
      `dormancy_gate.EXEMPT` entry in `src/snn_data_lake/substrate/dormancy_gate.py` with the
      R7 reason string
- [X] T008 [US4] Add `substrate/import_manifest.py` entries for the cross-tier reaches
      `plasticity/snn_mnist_stdp_genn -> reasoning/snn_mnist_stdp` and
      `-> reasoning/snn_mnist_stdp_fast` (existing pattern in that file)
- [X] T009 [US4] Check `substrate/method_dormancy_gate.py` against the module's public
      functions; add `EXEMPT_METHOD` entries with dispositions ONLY if the learning-verb
      heuristic matches (research.md R7)
- [X] T010 [US4] Run all three gates (module now INSIDE ruff coverage â€” fix any lint);
      commit as "P4c: snn_mnist_stdp_genn enters governance (still a stub)"

**Checkpoint**: gates green, module governed, diff so far contains no behavior change.

---

## Phase 5: User Story 2 - The port, finished and measured (Priority: P2)

**Goal**: Exact-mirror learner on GeNN; parity, control, determinism, constructed red â€” all
recorded.

**Independent Test**: quickstart.md Â§2-Â§4 + Â§6.

- [X] T011 [P] [US2] CPU oracle baseline (independent of the port): run
      `reasoning/snn_mnist_stdp_fast` with NORD_M=300 NORD_TRAIN=6000 NORD_TEST=1000
      SKIP_RATE=1, seed 0 (quickstart.md Â§2); record acc + wall-clock in evidence.md
      Â§cpu-baseline. [P] with T012-T015 â€” different files, no dependency.
- [X] T012 [US2] In `src/snn_data_lake/plasticity/snn_mnist_stdp_genn.py`: extract the C++
      rule bodies (pre_dynamics/pre_spike/post_spike) to module-level string constants
      importable without pygenn; `burst_xtar_rule()` consumes them unchanged (contract:
      module-contract.md; research.md R6)
- [X] T013 [US2] Same file: custom NEURON model â€” vars V/theta/winnerFlag, params
      beta/thetaPlus/thetaDecay/thresh/learn; sim_code `V = V*beta + Isyn` (+ theta decay when
      learn); threshold_condition = flagged winner; reset = winner's theta bump; plus the
      three custom updates: WTA argmax(V - theta) reduction writing winnerFlag + fired flag,
      global V reset after a fired step, per-image state reset (V, preTrace, spike counts)
      (research.md R5; data-model.md E2 ordering contract)
- [X] T014 [US2] Same file: encoding upload â€” reuse
      `reasoning.snn_mnist_stdp_fast.encode_latency` verbatim, pack per-neuron sorted
      timestamps into SpikeSourceArray startSpike/endSpike/spikeTimes (data-model.md E1);
      per-image re-upload path
- [X] T015 [US2] Same file: row-normalization custom update (per-row g sum -> scale to
      W_NORM); host train loop (`custom_update("wta"); step_time(); reset-if-fired` x T=30,
      per data-model.md E2); assign/eval host math mirroring fast_assign/fast_eval; env
      contract GENN_RUN/GENN_LR0/GENN_SEED + shared NORD_*/FAST_* names; ASCII output with
      final `acc=` / `train=` lines (contract: module-contract.md)
- [X] T016 [US2] Same file: self-check per contract â€” no-toolchain path asserts codegen
      integrity (rule text carries `lr * (preTrace - xtar)`, clamp [0, wmax], params byte-equal
      to FAST_LR/FAST_XTAR/WMAX/PRE_DECAY), toolchain path additionally constructs the model
      objects (no nvcc, no training); `GENN_RUN` gating so the gate NEVER trains; stub print
      preserved; docstring's stale "cannot run on the dev box" replaced by the measured
      environment facts (FR-009)
- [X] T017 [US2] Run the three gates; verify the self-check ALSO passes with pygenn absent
      (quickstart.md Â§1 stub-path check â€” run the module's self-check subprocess with the
      import forced off); commit as "the GeNN port is finished (unmeasured)"
- [ ] T018 [US2] Parity run (quickstart.md Â§3): NORD_M=300 NORD_TRAIN=6000 NORD_TEST=1000
      GENN_RUN=1, seed 0; record acc + wall-clock + delta vs Â§cpu-baseline in evidence.md
      Â§parity. Bar: <= 3 pp. If outside the band: debug the port, NOT the band â€” the band
      moves only with an owner decision.
- [ ] T019 [US2] Untrained control (quickstart.md Â§4): GENN_LR0=1, same config; record in
      evidence.md Â§control. Bar: < 20%.
- [ ] T020 [US2] Determinism check: encode one image twice + run it twice; assert identical
      spike trains both times; record in evidence.md Â§determinism (spec US2-AS3)
- [ ] T021 [US2] Constructed red (quickstart.md Â§6): temporarily mutate one rule constant ->
      self-check FAILS -> revert; record the demonstration in evidence.md Â§red-demo
- [ ] T022 [US2] Gates green; commit evidence.md + any fixes as "the GeNN port is measured:
      parity/control/determinism/red"

**Checkpoint**: the mandated build is BUILT and MEASURED at reference scale.

---

## Phase 6: User Story 3 - The payoff runs (Priority: P3)

**Goal**: Full-60k runs at M=300 and M>=1600, wall-clock + accuracy recorded â€” what the port
measurably bought.

**Independent Test**: quickstart.md Â§5.

- [ ] T023 [US3] Payoff run 1: NORD_TRAIN=60000 NORD_TEST=10000 NORD_M=300 GENN_RUN=1; record
      acc + wall-clock in evidence.md Â§payoff-300
- [ ] T024 [US3] Payoff run 2: same with NORD_M=1600; record in evidence.md Â§payoff-1600
      (VRAM sanity per spec Assumptions: ~5 MB weights â€” completion at recorded cost IS the
      measurement)
- [ ] T025 [US3] Write the honest comparison paragraph in evidence.md: GPU vs the CPU loop's
      measured wall-clock, where the crossover sits, whether the port beat the CPU at which
      widths (P5: a number, not a doctrine); commit as "the payoff is measured"

**Checkpoint**: the port's reason-to-exist has numbers attached, favorable or not.

---

## Phase 7: Polish & Close-out

- [ ] T026 Update the `snn_mnist_stdp_genn` row in `docs/P4C_BACKLOG.md`: MANDATED BUILD ->
      BUILT+MEASURED with evidence pointers (or the BLOCKED record from T005 stands); include
      the R8 non-duplication record so the P2 question is answered IN the ledger
- [ ] T027 [P] Add the CHANGELOG.md entry (accuracy figures, wall-clock, the extra fix, the
      promotion) per project convention
- [ ] T028 Run `graphify update .` from repo root (AST-only) so the graph reflects the move +
      finish
- [ ] T029 Final full-gate run (all three, quickstart.md Â§1); commit remaining docs as
      "specs/012 close-out"; push to origin/main per the project's arc-end convention

---

## Dependencies

- Phase 3 (US1) BLOCKS everything â€” T005's STOP gate is the only exit on failure.
- Phase 4 (US4) depends on US1 success only (the move/registration needs no toolchain, but
  landing governance before the finish is the P4c ordering â€” plan.md Structure Decision).
- Phase 5 (US2) depends on Phase 4 (the finish lands in the governed file). T011 (CPU
  baseline) is independent of T012-T016 and may run in parallel with them; T018-T021 are
  sequential measurements after T017.
- Phase 6 (US3) depends on Phase 5 (no point timing an unverified learner).
- Phase 7 depends on all prior phases (or on the T005 BLOCKED path, in which case only
  T026/T029 apply).

## Parallel Opportunities

- T011 (CPU baseline, ~minutes of compute in reasoning/) alongside T012-T016 (editing
  plasticity/) â€” different files, no shared state.
- T027 (CHANGELOG) alongside T026 (ledger) â€” different files.
- Everything else is deliberately serial: one module, one-change-per-commit, measurements
  ordered by what they validate.

## Implementation Strategy

MVP = Phase 3 alone (US1): after T005 the project either has a working GeNN substrate (worth
having even if the session ends there) or a clean BLOCKED record. Each subsequent phase is an
independently green, committed increment: governed stub -> finished-and-measured port ->
payoff numbers -> ledger close-out.

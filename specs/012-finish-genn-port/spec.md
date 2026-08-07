# Feature Specification: Finish the GeNN Port of the Burst+x_tar MNIST Learner

**Feature Branch**: `012-finish-genn-port`

**Created**: 2026-08-07

**Status**: Draft

**Input**: User description: "Finish the GeNN port of snn_mnist_stdp_genn (MANDATED BUILD, owner
2026-08-07, ledger row docs/P4C_BACKLOG.md). The module carries the tree's only copy of the
burst_xtar_rule GeNN codegen (the CPU-verified v0.17 burst+x_tar plasticity rule as C++), but the
port is a skeleton: no adaptive threshold, no lateral inhibition, no train/assign/evaluate loop.
What finishing owes: the [genn] extra stood up in the one gate env; the codegen completed; the
full P4c entry bar; measurement — parity vs the CPU-verified 76.0% at M=300/6k, an untrained
control near chance, determinism verified, and the full-60k payoff run with wall-clock measured."

## Why (context the lane starts from)

The owner's direct answer (2026-08-07) to the finish-or-retire question for
`snn_mnist_stdp_genn` was **finish**. It left the RETIRE class permanently as the one MANDATED
BUILD of the 16-row resolution. The ledger row (docs/P4C_BACKLOG.md, "the remaining RETIRE-class
holdouts") enumerates what finishing owes so this lane starts from the ledger, not archaeology.

The module exists because the CPU-verified learning rule hit a CPU ceiling: the v0.17
burst-encoding + x_tar LTP rule reaches 76.0% at M=300 neurons / 6k train images on the CPU
reference (`snn_mnist_stdp_fast.py`), and the O(T) Python simulation loop is the bottleneck that
keeps it at toy scale. The project's standing rule is **never toy scale** — a toy ceiling is a
reason to scale up. GeNN collapses the simulation loop into compiled GPU kernels; the port is the
route to running the same rule at full-dataset, larger-population scale on the RTX 5070.

Environment facts measured at spec time (2026-08-07, this box): nvcc CUDA 13.3 present, Visual
Studio 2022 Community present (`cl.exe` not on PATH), RTX 5070 present, the one gate env
`.venv-lake` is Python 3.14 free-threading, `pygenn` NOT installed, `[genn]` extra already
declared in pyproject (`pygenn`, `ml_genn`). Whether pygenn can install and compile models under
this exact stack is UNKNOWN and is the blocking pre-flight.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The toolchain exists before anything is built on it (Priority: P1)

As the project, I need the GeNN toolchain stood up and verified in the ONE gate environment
before any port work happens, because every downstream deliverable is unmeasurable without it,
and a capability reported without measurement is fake.

**Why this priority**: Blocking pre-flight (the specs/011 `T000` pattern). The module's own
docstring claims "cannot run on the dev box" — written before the box had a GPU and CUDA. That
claim is stale and must be replaced by a measurement in either direction. If the toolchain cannot
be stood up (e.g. no pygenn build for the free-threaded 3.14 ABI, or CUDA 13.3 unsupported by
GeNN), the whole feature STOPS and reports the measured blocker to the owner — no port work
proceeds on an unverifiable substrate, and no second environment is created without an owner
decision (a second env is the exact shape `.venv-cu128`'s retirement closed).

**Independent Test**: In the gate env, install the `[genn]` extra, import pygenn, build and run a
minimal GeNN model end-to-end on the GPU. All three succeed, or the failure is recorded verbatim
and the feature stops.

**Acceptance Scenarios**:

1. **Given** the gate env `.venv-lake`, **When** the `[genn]` extra is installed, **Then**
   `import pygenn` succeeds in that env and the installed version is recorded.
2. **Given** pygenn imports, **When** a minimal model (one source population, one LIF population,
   one synapse) is built, compiled and stepped, **Then** it completes on the CUDA backend and its
   output is sane (spikes recorded, no NaN), proving nvcc + host compiler + driver agree.
3. **Given** any of the above fails, **When** the failure is reproduced once more, **Then** the
   exact error is recorded in the lane's evidence file and the feature halts awaiting an owner
   decision — partial port work is NOT committed.

---

### User Story 2 - The CPU-verified rule runs on the GPU and earns the same result (Priority: P2)

As the project, I need the complete burst+x_tar learner running on GeNN — deterministic burst
encoding in, adaptive threshold and winner-take-all competition inside, label assignment and
evaluation out — reproducing the CPU reference's accuracy at the reference configuration, so the
tree's only GPU plasticity codegen is a measured capability instead of a skeleton.

**Why this priority**: This is the mandated build itself. Parity at M=300/6k is the honest
correctness bar: same rule, same encoding, same data budget, same seed discipline — the GPU may
differ in numerics (continuous-time vs discrete dynamics) but must land in the same accuracy band
for the port to count as the same mechanism.

**Independent Test**: Run the module end-to-end at the reference configuration on the GeNN box;
compare accuracy against the CPU-verified figure; run the untrained control; run the determinism
check. All three criteria have constructed red states.

**Acceptance Scenarios**:

1. **Given** the finished port at the reference configuration (M=300, 6k train / 1k test, the
   v0.17 default hyperparameters, fixed seed), **When** train + assign + evaluate completes,
   **Then** test accuracy is within 3 percentage points of the CPU reference's 76.0%.
2. **Given** the same run with learning disabled (lr = 0, the untrained twin), **When**
   evaluated, **Then** accuracy is near chance (below 20%), proving the measurement can fail and
   that learning — not the harness — produces the score.
3. **Given** the same image encoded twice, **When** the spike trains are compared, **Then** they
   are identical (the deterministic-anchor property the module's design rests on).
4. **Given** the module's self-check on a box WITHOUT the toolchain (CI), **When** it runs,
   **Then** it degrades to a stub that states the requirement and exits cleanly (suite stays
   runnable with no GPU), while the rule's codegen text is still cross-checked against the CPU
   reference's constants so CI exercises something falsifiable.

---

### User Story 3 - The payoff run: past the CPU ceiling (Priority: P3)

As the project, I need the port run at the scale it exists for — the full 60k MNIST training set
and a larger population — with wall-clock and accuracy recorded against the CPU reference, so the
port's reason to exist (escaping the CPU O(T) loop) is a measurement, not a docstring claim.

**Why this priority**: The port's WHY. Depends on Stories 1–2. Its outcome is recorded honestly
either way: if the GPU does not beat the CPU at some width, that is a crossover measurement (the
crossover is a number, not a doctrine), recorded, not smoothed.

**Independent Test**: One full-60k training run at the reference M and one at a scaled-up M;
wall-clock and accuracy recorded for both, beside the CPU reference's wall-clock at its
configuration.

**Acceptance Scenarios**:

1. **Given** the finished port, **When** trained on the full 60k set at M=300, **Then** the run
   completes, and accuracy + wall-clock are recorded beside the CPU reference's figures.
2. **Given** the full 60k set and a scaled population (target M >= 1600), **When** trained,
   **Then** the run completes on the 12.8 GB card and its accuracy + wall-clock are recorded —
   this is the regime the CPU loop cannot practically reach, so completion at recorded cost IS
   the payoff measurement.
3. **Given** the recorded measurements, **When** the ledger row is updated, **Then** the row
   states what the port measurably bought (or measurably did not), and the stale "cannot run on
   the dev box" docstring claim is replaced by the measured facts.

---

### User Story 4 - The module enters governance (Priority: P2)

As the project, I need the finished module inside the governed tree — registered, gated,
dispositioned — because every build is part of the brain and a build the gates cannot see is not
a build the project can claim.

**Why this priority**: The ledger row names the full P4c entry bar as part of what finishing
owes. Runs with Story 2 (the entry bar is met by the same change that completes the port, not a
later cleanup).

**Independent Test**: The three gates pass with the module registered; the dormancy gates give it
a disposition; lint covers it.

**Acceptance Scenarios**:

1. **Given** the finished module, **When** the gates run, **Then** it is registered in the region
   map and the self-check runner, carries a dormancy disposition (live-path reachable or
   EXEMPT-with-reason), and all three gates are green on this box AND on a GPU-free path.
2. **Given** the governed module, **When** ruff runs, **Then** the module is inside lint coverage
   (not excluded), clean.

---

### Edge Cases

- **pygenn has no build for the free-threaded Python 3.14 ABI**: the pre-flight measures this
  first. Source build is attempted (CUDA + MSVC are present); if that fails, STOP + owner
  decision. No silent second environment.
- **CUDA 13.3 vs GeNN's supported CUDA range**: GeNN releases lag CUDA majors. The pre-flight's
  minimal-model compile is exactly the test; a version-mismatch failure is recorded verbatim.
- **`cl.exe` not on PATH**: GeNN needs a discoverable host compiler at model-compile time. The
  environment requirement (developer prompt / vcvars) is documented where the module states its
  requirements; the pre-flight proves whichever route is chosen actually compiles.
- **GeNN's built-in LIF has no adaptive threshold**: the CPU rule's theta (homeostatic threshold
  growth) is load-bearing for prototype separation. If the built-in neuron cannot express it, the
  port owes a custom neuron model — the same codegen route the plasticity rule already uses. The
  parity criterion (US2-AS1) is what catches an inadequate shortcut.
- **Numerics differ between backends** (continuous-time exponential decay vs the CPU's discrete
  multiply-by-beta): the parity band (3 pp) plus the untrained control bound the acceptable
  drift; a port outside the band is not "close enough", it is red.
- **MNIST is not on the box / cannot download**: the CPU reference's loader is reused; its
  existing cache/download behavior is the dependency, not new work. A fetch failure surfaces as
  the loader's own error.
- **CI (no GPU, no toolchain)**: the stub-degrade path stays; the self-check must still assert
  something falsifiable there (rule-text/constants cross-check vs the CPU reference), so CI green
  is not vacuous.
- **A second GPU MNIST trainer already exists** (`training/mnist_genn.py`, supervised
  gradient-based, exports to the core's serving format; unverified scaffold): different question
  — supervised gradient training for serving weights vs unsupervised local plasticity at scale.
  Same-vocabulary-different-question, the ledger's documented failure shape. No merge is owed by
  THIS lane; if a future benchmark puts them on the same question, that is a new P2 event.
  Likewise `paradigm_b_genn.py` (coincidence query, no learning at all).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 (pre-flight, blocking)**: The `[genn]` toolchain MUST be installed and verified in the
  ONE gate environment: extra installs, pygenn imports, and a minimal model compiles and runs on
  the GPU — all three measured before any port work. On failure: record verbatim, STOP, owner
  decision. No second environment without an owner decision.
- **FR-002**: The port MUST implement the complete learner the CPU reference implements:
  deterministic burst-latency encoding uploaded as exact spike timestamps, the v0.17 burst+x_tar
  plasticity rule (the existing codegen), adaptive threshold (theta) equivalent to the CPU
  reference's, winner-take-all lateral inhibition, label assignment from class-conditional
  responses, and evaluation — end-to-end from dataset to accuracy figure.
- **FR-003**: The port MUST reproduce the CPU reference's accuracy at the reference
  configuration (M=300, 6k/1k, v0.17 defaults, fixed seed) within 3 percentage points of 76.0%.
- **FR-004**: The port MUST carry an untrained control (learning disabled) that scores near
  chance (< 20%) under the identical harness — the constructed red proving the criterion can fail.
- **FR-005**: Encoding determinism MUST be verified: identical input produces identical spike
  trains across runs.
- **FR-006**: The payoff MUST be measured: full-60k training at the reference M and at a scaled
  M (>= 1600), accuracy + wall-clock recorded beside the CPU reference's figures. An unfavorable
  result is recorded, not smoothed.
- **FR-007**: The module MUST meet the governed-module entry bar: region-map + self-check-runner
  registration, a dormancy disposition (live-path reachable or EXEMPT-with-reason), inside lint
  coverage, all three gates green.
- **FR-008**: The suite MUST stay runnable without a GPU: on a box without the toolchain the
  module degrades to a stub that exits cleanly, and its no-toolchain self-check still asserts
  something falsifiable (codegen text/constants cross-checked against the CPU reference).
- **FR-009**: The stale docstring claim ("cannot run on the dev box") MUST be replaced by the
  measured environment facts, and the ledger row MUST be updated with the outcome so the next
  session starts from the ledger.
- **FR-010**: Existing invariants hold: ASCII-only output, per-instance seeding, no module-scope
  side effects, e-prop invariant untouched (the rule is local plasticity; nothing here learns by
  backprop), one change per commit with the full gate green.

### Key Entities

- **The port** (`snn_mnist_stdp_genn`): the GPU learner; owner of the tree's only GeNN
  plasticity codegen (`burst_xtar_rule`).
- **The CPU reference** (`snn_mnist_stdp_fast` + constants from `snn_mnist_stdp`): the verified
  oracle — its 76.0% at the reference configuration is the parity target; its encoder semantics
  are the determinism contract.
- **The gate environment** (`.venv-lake`): the ONE env; the pre-flight's subject.
- **The ledger row** (docs/P4C_BACKLOG.md): where the mandate lives and where the outcome lands.
- **The evidence file** (this spec's directory): recorded measurements — pre-flight results,
  parity run, control run, payoff runs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The GeNN toolchain is verified working in the one gate env (install + import +
  minimal compile/run all pass), or the blocker is recorded verbatim and the feature stopped at
  zero port commits.
- **SC-002**: GPU accuracy at the reference configuration is within 3 pp of the CPU reference's
  76.0%, with the run's config + seed recorded.
- **SC-003**: The untrained control scores < 20% under the identical harness.
- **SC-004**: The determinism check passes: identical input, identical spike trains, two runs.
- **SC-005**: Full-60k runs at M=300 and M>=1600 complete with accuracy + wall-clock recorded
  beside the CPU reference's figures; the record states plainly whether the GPU beat the CPU
  loop and at what width.
- **SC-006**: All three gates green with the module registered and dispositioned; CI-shaped
  (no-GPU) self-check green AND falsifiable; zero regressions elsewhere in the tree.
- **SC-007**: Ledger row + module docstring updated to the measured state; no stale claims
  remain.

## Assumptions

- **Finishing means governing**: the ledger's "full P4c entry bar" is read as the module
  entering the governed tree in this lane (registration, disposition, lint) — not a later pass.
  The exact target region (it is a plasticity demonstration; `plasticity/` is the natural home)
  is a plan-phase decision.
- **The CPU reference stays where it is**: `snn_mnist_stdp_fast`/`snn_mnist_stdp` remain in
  `reasoning/` (their own ledger rows govern their promotion); the governed port reaching into
  `reasoning/` for shared constants uses the existing cross-tier import pattern. Their lanes are
  not this lane.
- **Parity band 3 pp**: the backends' neuron dynamics are not bit-identical (continuous vs
  discrete time); 3 pp on a 76.0% reference is tight enough to catch a broken rule (the pure
  latency ablation sits 7.2 pp below) and loose enough to absorb honest numeric drift. The
  untrained control catches a harness that flatters.
- **Chance bar at 20%**: chance is 10% on MNIST; < 20% allows the mild above-chance drift an
  untrained-but-normalized network shows while still being unreachable by a working learner.
- **`[genn]` extra composition unchanged**: it already declares `pygenn` + `ml_genn`; `ml_genn`
  belongs to the sibling trainer and is left alone. Version pins are added only if the
  pre-flight measures a version sensitivity.
- **Full 60k on this card fits**: dense 784 x M weights at M=1600 is ~5 MB — memory is not the
  constraint at these widths; compile time and simulation wall-clock are what get measured.
- **No branch workflow**: the project commits to `main` with the full gate green per commit; the
  "Feature Branch" header names the spec directory, not a git branch.

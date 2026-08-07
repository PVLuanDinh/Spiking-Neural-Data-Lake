# Implementation Plan: Finish the GeNN Port of the Burst+x_tar MNIST Learner

**Branch**: `012-finish-genn-port` | **Date**: 2026-08-07 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/012-finish-genn-port/spec.md`

## Summary

Finish `snn_mnist_stdp_genn` (MANDATED BUILD, owner 2026-08-07): stand up the GeNN toolchain in
the one gate env (blocking pre-flight — the `[genn]` extra is currently uninstallable because
pygenn is not on PyPI; repoint it to the GeNN 5.4.0 source archive), then complete the port by
EXACT-MIRRORING the CPU reference's semantics (hard global WTA + adaptive threshold + burst+x_tar
LTP + per-image normalization) via GeNN custom neuron model + reduction custom updates, measure
parity (±3 pp of a same-box CPU baseline at M=300/6k), the lr=0 untrained control (<20%),
encoding determinism, and the full-60k payoff runs (M=300 and M>=1600) with wall-clock — then
promote the module into `plasticity/` with the full P4c entry bar. All design decisions and their
evidence: [research.md](research.md).

## Technical Context

**Language/Version**: Python 3.14 free-threading (cp314t), the ONE gate env `.venv-lake`

**Primary Dependencies**: pygenn (GeNN 5.4.0, source install from GitHub tag — R1), torch (core,
for the shared encoder + MNIST loader), numpy (core). `[genn]` extra repointed to a PEP 508
direct reference.

**Storage**: N/A (no lake interaction; evidence records land in this spec directory — the module
trains, measures, prints, and its results go to the ledger + evidence file, not the lake)

**Testing**: the three gates (`run_selfchecks.py` in isolated subprocesses, `pytest` >=70%,
`ruff check .`) + this module's own self-check (codegen-integrity assert, no toolchain needed)
+ the measured runs (parity / control / determinism / payoff) recorded in evidence.md

**Target Platform**: Windows 11, RTX 5070 (sm_120), CUDA 13.3, VS 2022 Community host compiler;
CI = GPU-free free-threaded interpreter (stub path must stay green + falsifiable)

**Project Type**: single governed module inside the existing package + pyproject extra fix +
gate registrations

**Performance Goals**: none promised — MEASURED: the payoff run records wall-clock vs the CPU
loop at M=300 and M>=1600 on full 60k; an unfavorable crossover is a recorded finding (P5)

**Constraints**: suite runnable with NO GPU; ASCII-only output; no module-scope side effects;
per-instance seeding; one change per commit, full gate green; safety envelope untouched

**Scale/Scope**: one module (~400 lines finished), 1 pyproject line, 3 registration sites
(REGIONS, run_selfchecks, dormancy EXEMPT), import_manifest entries, ledger + docstring update

## Constitution Check

*GATE: evaluated against constitution v1.1.0 (P0–P8 + Standing Invariants). Re-checked after
Phase 1 design — no violations; no Complexity Tracking entries needed.*

- **P0 (purpose)**: direct hit — the port is a named avenue of capability the tree already owes;
  honesty guards carried: nothing is credited until measured (SC-002..SC-005), and the
  measurement can fail (untrained control, parity band, STOP semantics).
- **P1 (no hardcode)**: all rule constants come FROM the CPU reference's module (env-tunable
  there); the port introduces no new magic constants — parameters are passed into the codegen
  from the shared source of truth. The 3 pp band and 20% chance bar are criteria thresholds
  with recorded justification (spec Assumptions), not behavior-arbitrating runtime values.
- **P2 (non-additive)**: the port supersedes NOTHING (the CPU reference answers the
  CPU-verified-oracle question and remains the oracle; the GPU module answers the scale
  question). Sibling GeNN modules are different-question — recorded in research.md R8 as the
  ledger requires. The shared encoder is REUSED, not reimplemented.
- **P3 (no dormancy)**: dispositions declared up front (R7): dormancy_gate EXEMPT-with-reason,
  method gate checked, import_manifest entries for cross-tier reaches. No fake consumer is
  wired to dodge the gate (honesty guard: do NOT force-wire).
- **P4/P4b**: no Brain construction anywhere in the module; no second durable store (results
  are printed + recorded in the spec's evidence file, no sidecar state files beyond GeNN's own
  build directory, which is a compiler artifact, not memory).
- **P4c (every build is part of the brain)**: this lane EXECUTES P4c for the module — promotion
  into `plasticity/`, REGIONS + selfchecks registration, ruff coverage, entry bar green.
- **P5 (GPU where measured)**: the whole feature is a P5 measurement; the hard constraint
  (suite green with no GPU) is FR-008 and R6's design.
- **P6**: not implicated — the module is a training/measurement harness, not a faculty bus
  participant; no workspace instantiation, no action-gate wiring. p6_gate is a ratchet on
  structure that this change does not touch (verified at implement time by the gate itself).
- **P7**: the dependency (pygenn) ships with its measurement (the pre-flight + parity + payoff
  runs state exactly what it bought); the currently-declared-but-uninstallable extra is FIXED,
  closing a declared-but-never-importable defect of the exact shape P7 calls out.
- **P8**: no invented method — the rule is the CPU-verified v0.17 rule; the port's criteria
  carry reachable reds (constructed mutation red for the codegen check, lr=0 control for the
  learning claim, STOP for the toolchain claim).
- **Standing invariants**: e-prop untouched (local plasticity, no `.backward()`, no weight
  transport); safety envelope untouched; runtime evidence outranks static reading (the stale
  "cannot run here" docstring is replaced by measured facts, FR-009); never silence a check
  (the stub path stays falsifiable — R6).

## Project Structure

### Documentation (this feature)

```text
specs/012-finish-genn-port/
├── plan.md              # This file
├── research.md          # Phase 0 (complete)
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   └── module-contract.md   # Phase 1 — env/CLI/stub contract of the finished module
├── checklists/requirements.md
├── evidence.md          # Created during implementation: every measured run, verbatim
└── tasks.md             # Phase 2 (/speckit-tasks — NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
src/snn_data_lake/
├── plasticity/
│   └── snn_mnist_stdp_genn.py      # MOVED from reasoning/ + FINISHED (the one code change)
├── reasoning/
│   ├── snn_mnist_stdp.py           # UNTOUCHED oracle (constants, loader, rate baseline)
│   └── snn_mnist_stdp_fast.py      # UNTOUCHED oracle (encoder, FastStdpNetwork, 76.0% ref)
├── substrate/
│   ├── dormancy_gate.py            # +1 EXEMPT entry with reason
│   └── import_manifest.py          # +entries: plasticity/snn_mnist_stdp_genn -> reasoning/*
├── __init__.py                     # REGIONS: register under plasticity
run_selfchecks.py                   # +1 registration
pyproject.toml                      # [genn] extra -> direct reference (R1)
docs/P4C_BACKLOG.md                 # ledger row updated with the outcome (FR-009)
```

**Structure Decision**: single-module promotion + registrations. The module moves ONCE (git mv +
finish in the same lane, separate commits: move-then-finish keeps each commit's diff readable
and the gate green at both points). No new packages, no new test files beyond what pytest
coverage requires — the module's own self-check carries the falsifiable checks per project
convention.

## Complexity Tracking

No constitution violations to justify. The one structurally notable choice — custom updates +
host-driven per-step WTA instead of built-in models — is forced by the parity criterion and is
documented in research.md R5 with the rejected simpler alternatives and why each fails the bar.

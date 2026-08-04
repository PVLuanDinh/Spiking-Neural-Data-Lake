# Implementation Plan: Whole-brain neurogenesis -- every region a producer AND a consumer

**Branch**: `007-whole-brain-neurogenesis` | **Date**: 2026-08-03 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/007-whole-brain-neurogenesis/spec.md`

## Summary

The spec requires every region to both grow its own population and be changed by what other regions
grow, with consumption as the acceptance criterion. Phase 0 research established the fact that
determines the entire approach:

**Function-preserving growth emits exactly zero at growth time, by construction.** New units are
zero-filled, so they contribute nothing until trained. This is what "function-preserving" means --
`growable.py`'s own docstring says "they emit 0 until trained and the EXISTING outputs are
unchanged" -- and it is why the measured baseline is 88 of 93 fields identical. It is not a bug in
any mechanism, and no choice of mechanism escapes it.

Therefore the approach is **not** "add growth to fifteen regions and read the widths". It is:

1. **Fix the consumer, not the producer.** Growth already reaches the brainstem on 211 of 300 ticks.
   It arrives at ~1e-4 magnitude and is then destroyed by an int cast at `brain.py:1603` and
   absorbed into a summation. The product exists; consumption discards it.
2. **Charge the grown units** so they stop emitting zero. Growth must be coupled to the learning
   path, or it stays permanently inert whatever else is built.
3. **Gate breadth behind a per-region floor.** A region grows only once it has a derived halting
   condition that has been A/B'd. "No derivable floor, so it does not grow" is an accepted outcome,
   and it is the clause that stops this feature shipping fifteen dormant growers.

Sequenced against the spec's stories: P1 builds one honest cross-region consumer, P2 extends
production only where a floor exists, P3 makes the relation mutual.

## Technical Context

**Language/Version**: Python >=3.11 (CI matrix 3.11 / 3.12 / 3.13)

**Primary Dependencies**: `numpy` and `torch`, both unconditional core dependencies. This feature
proposes no new dependency.

**Storage**: the one durable lake (`brain_lake.bin`) plus opt-in per-component sidecars under
`~/.snn_data_lake/`. Grown-width persistence stays opt-in (`NORD_TIME_CELLS_RESTORE`) for the
determinism reason under Constraints.

**Testing**: `pytest` (>=70% line coverage; currently 92.85%), `python run_selfchecks.py`
(currently 278 of 278), `python -m ruff check .`. All three stay green.

**Target Platform**: Windows 11 and Linux CI. The deterministic CPU gate env `.venv-lake` is
authoritative; `.venv-cu128` is the GPU tier. The suite MUST remain runnable without a GPU.

**Project Type**: Python library plus an always-on daemon serving one cockpit frontend.

**Performance Goals**: the live tick is the budget. The measured cost model is
`ms/tick = 56.60 + 0.0804 * N_episodes`, since brought to ~27.8 ms per tick. Growth widens
populations and pushes directly against this, so any region admitted to growth must report its
per-tick cost at the GROWN width, not only the ungrown one.

**Constraints**:
- Determinism is non-negotiable: identical seed, identical structure. Making grown-width restore
  unconditional previously produced nondeterministic construction in ~42 of 100 trials.
- Growth decisions must consume zero random numbers (R1, measured bit-identical `rng.getstate()`
  across 50 attempts).
- E-prop / forward-mode credit assignment on any live learning path; never `loss.backward()`.
- ASCII-only `print()`; counts written `N of M`, never `N/M`.
- The safety envelope does not widen. Growth stays refused on arm / dof / joint / motor / actuator /
  effector / estop seams.

**Scale/Scope**: 17 regions, of which **15 populated** (191 modules) and 2 empty (`pons`,
`olfactory`, excluded by owner decision), **plus** the `reasoning` package's 146 modules which sit
outside the region map and are in scope. Live populations are 24-240 units wide.

## Constitution Check

*GATE: must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Assessment | Status |
|---|---|---|
| **P0** self-improvement | Neural growth is one of P0's three named avenues; this feature is what makes it real rather than merely allocated. P0's honesty guard -- "growth is NOT improvement; a grown unit that changes no outcome is dormant" -- is adopted verbatim as the acceptance criterion (FR-001, FR-002). | PASS |
| **P1** no hardcode | The halting condition is `resolvable()`, measured self-extinguishing and constant-free. The rejected alternatives were rejected *because* they smuggled constants -- a top-p quantile needs `p`. Remaining offenders are named, not hidden: each per-region floor must be derived rather than declared. | PASS with named debt |
| **P2** non-additive | Phase 0 selects no survivor and retires nothing yet. This is compliance, not evasion: P2 requires "A/B before deleting", and the M25 decision gate IS that A/B. Three dead stubs are approved for deletion now. **The plan commits that when a shared mechanism is chosen, the bespoke `time_cells` path BECOMES it; a flag keeping both is forbidden.** | PASS |
| **P3** no dormancy | The feature's entire purpose. The per-region floor gate exists so breadth cannot ship dormant growers. Every admitted region carries a neuter-A/B, never a reachability claim. | PASS |
| **P4** one brain | Every probe constructs via `cortex.brain.make_brain()`. No new composition. `replication_guard` stays green. | PASS |
| **P5** GPU where measured | Live populations are 24-240 wide against a measured CUDA crossover of N~24,576-32,768, so this feature stays on CPU. **If any region's grown width approaches that, P5 requires re-measurement rather than assumption.** | PASS |
| **P6** mutual reach and use | Story P3 is P6 for this subsystem. The sibling defect (`brain.py:1623` discarding the winning faculty's identity) is tracked in `docs/P6_INTERCONNECT.md`; FR-005 exists so consumers built here are compatible with that fix rather than around it. | PASS |

**Standing Invariants:**

| Invariant | How this plan satisfies it |
|---|---|
| 1. Count a gate's output before a rule leans on it | A region's floor must be built AND A/B'd before that region may grow. No region is admitted on the strength of the rule alone. |
| 2. Never silence a check to make it pass | The 88-of-93 baseline stays the scoreboard. FR-002 names the forbidden observables so the tautological test cannot be re-run as evidence. |
| 3. The negative verdict is the dangerous one | "This region has no derivable floor" is recorded as UNKNOWN-and-does-not-grow, never as "growth is impossible here". |
| 4. Absence of warning is not absence of defect | Name-based growth scans undercount -- `GrowingLearner` recruits inline in `observe`, not in a `grow*` method. Every inventory reports a floor, not a census. |
| 5. Verify the verifier | **Every A/B in this feature ships with a matched A/A control.** Phase 0 caught two false positives this way; without the control both would have been reported as findings. |

**Violations requiring justification**: none. The one deliberate deferral is in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/007-whole-brain-neurogenesis/
├── plan.md              # This file
├── spec.md              # /speckit-specify output
├── research.md          # Phase 0 -- four decisions, all measured
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1 -- how to run the acceptance A/Bs
├── contracts/
│   └── growth-consumption.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 (/speckit-tasks -- NOT created here)
```

### Source Code (repository root)

```text
src/snn_data_lake/
├── cortex/
│   ├── time_cells.py            # the one grower that earns; widest_log_gap() fix lands here
│   ├── brain.py                 # :1603 int-cast quantiser -- the P1 prerequisite defect
│   └── neurogenesis_engine.py
├── hippocampus/
│   ├── growing_learner.py       # de facto shared mechanism, embedded by 4 regions
│   └── function_preserving_growth.py
├── substrate/
│   ├── growable.py              # registry never registered; the M25 decision gate targets this
│   ├── dormancy_gate.py         # P3 CI gate -- stays green
│   └── replication_guard.py     # P4 CI gate -- stays green
├── reasoning/                   # 146 modules, in scope, outside REGIONS
│   ├── readout_ensemble.py          # :63 dead grow() -- approved deletion
│   └── developmental_loop_brain.py  # :101 dead grow() -- approved deletion
└── brainstem/
    └── autonomous_loop.py       # :696 blind add -- where a real consumer must NOT go

tests/
└── test_neurogenesis_charge.py  # new: the A/B + matched A/A acceptance harness

tools/
└── growth_census.py             # new: per-region producer/consumer inventory, reported "N of 15"
```

**Structure Decision**: no new top-level structure. This feature is a consumer fix plus a gated
extension of an existing mechanism, so it lands inside the existing anatomical regions. One new test
module and one new census tool are the only additions. A new *package* would itself be a P2
violation, because the growth mechanisms it would host already exist -- in five places.

## Implementation Sequence

Ordered so that each step's output is the next step's precondition, and so the feature can be
stopped honestly at any boundary.

**Story P1 -- one honest cross-region consumer** (the whole feature in miniature)

1. Build the acceptance harness FIRST: A/B plus matched A/A, hermetic (`persist=False`, isolated
   `NORD_HOME`), recording raw unrounded internals alongside `state()` leaves. Without this, nothing
   later is evidence. The same A/B run with `persist=True` yields 18 of 102 differing leaves, so
   harness discipline is load-bearing rather than hygiene.
2. Fix `brain.py:1603` to consume the float rather than `max(1, int(round(el)))`. Prerequisite:
   today the quantiser destroys 100% of the refinement (`_learned_latency` differs on 0 of 300).
3. Fix `widest_log_gap()` to select among RESOLVABLE gaps, so site selection and the halting
   condition agree.
4. Identify or build the cross-region consumer that BRANCHES on the timing estimate. Governed by
   research.md Open Question 1 -- **if no natural consumer exists, report P1 blocked and amend the
   spec. Do not invent an artificial branch.**
5. Charge the grown units: couple growth to the learning path so new units stop emitting zero.

**Story P2 -- production where a floor exists**

6. Build `tools/growth_census.py`: per-region producer/consumer inventory, reporting
   "N of 15, 2 regions empty and excluded", plus the reasoning package separately.
7. Delete the three approved dead-growth sites.
8. Run the M25 decision gate; retire or adopt `GrowthCoordinator` on its measured answer.
9. Per region, in descending module count: derive a resolution floor, A/B it, and admit that region
   to growth only if the A/B passes. Record "no derivable floor" outcomes as accepted results.

**Story P3 -- mutual enhancement**

10. Suppression testing per producer/consumer pair; report the count of regions that BOTH produce
    and consume.
11. Ensure consumers are identity-sensitive (FR-005), compatible with the P6 fix rather than
    around it.

## Complexity Tracking

| Deferral | Why needed | Simpler alternative rejected because |
|---|---|---|
| No P2 survivor selected in Phase 0 | Selecting one cannot move the consumption number, because function-preserving growth emits zero for every candidate. Choosing now would be a coin-flip dressed as a decision, and would retire a working mechanism on an argument rather than a measurement. | "Just adopt `growable`, it is the most general" was the survey's recommendation. Rejected on measurement: its headline signal-to-noise was 0 over 0 with the probe's own hand-written weight supplying the signal; it is a registry never registered (16 occurrences, 1 file, consumer is a lambda in its own `main()`); and `docs/AUDIT_BACKLOG.md` M25 already has it queued for retirement. |

## Post-design Constitution re-check

Re-evaluated after the Phase 1 artifacts: **PASS, unchanged.** The design adds no new mechanism, no
new dependency, and no new brain composition.

The risk it does carry is research.md Open Question 1 -- whether a legitimate cross-region timing
consumer exists naturally, or would have to be invented to pass the test. Inventing one would
satisfy the letter of SC-001 while violating P3's honesty guard and Standing Invariant 2. If no
natural consumer is found, the correct outcome is to report P1 as blocked and amend the spec --
never to add an artificial branch so the number moves.


> **BASELINE CORRECTED 2026-08-03.** Any `0 of 88` / `88 of 93` / `5 of 93` figure in this
> document is a SEED-0 measurement and is NOT representative. Ten-seed sweep: growth moves a
> non-tautological field on **7 of 10 seeds** at the shipped width (mean 1.10) and **9 of 10**
> at init 32 (mean 1.30). SC-001 already passes on most seeds. See the third AMENDMENT in
> spec.md.

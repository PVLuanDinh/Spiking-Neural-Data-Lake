---

description: "Task list for whole-brain neurogenesis -- every region a producer AND a consumer"
---

# Tasks: Whole-brain neurogenesis -- every region a producer AND a consumer

**Input**: Design documents from `/specs/007-whole-brain-neurogenesis/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/growth-consumption.md, quickstart.md

**Tests**: REQUIRED, and not as a style preference. This feature's acceptance criterion IS a
measurement (FR-001, FR-002, SC-001), and the measurement harness is a blocking prerequisite for
every claim the feature makes. A task that ships without its A/B and matched A/A control has
produced no evidence.

**Organization**: grouped by user story so each is independently implementable and testable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: US1 / US2 / US3, mapping to the spec's P1 / P2 / P3 stories

## Path Conventions

Single project. `src/snn_data_lake/<region>/`, `tests/`, `tools/` at repository root. Interpreter is
`.venv-lake\Scripts\python.exe` with `PYTHONPATH=src` for every gate and measurement.

## The rule that governs every task below

Growth emits **zero** at creation, by construction (research.md). No task may claim consumption on
the basis of a wider population. Only a neuter-and-measure A/B with a matched A/A control at `0`
counts, and the observables named in contracts/C3 are forbidden as evidence.

---

## Phase 1: Setup

**Purpose**: nothing to scaffold -- this feature lands inside existing regions. These tasks pin the
baselines the feature is scored against, so that later movement is attributable.

- [ ] T001 Record the pre-feature baseline into `specs/007-whole-brain-neurogenesis/baseline.md`: the three gate results (`run_selfchecks.py`, `pytest`, `ruff check`) with exact counts, plus current per-tick cost
- [ ] T002 [P] Verify the safety envelope refuses growth by calling `is_safety_seam` in `src/snn_data_lake/substrate/growable.py` with arm/dof/joint/motor/actuator/effector/estop names, recording the returned values in `specs/007-whole-brain-neurogenesis/baseline.md`

---

## Phase 2: Foundational (BLOCKING -- no story may start until these pass)

**Purpose**: build the instrument before trusting any number from it. Standing Invariant 5 --
verify the verifier. Phase 0 caught two false positives with the A/A control alone; without it this
feature would report success it has not earned.

- [ ] T003 Create the charge A/B harness `tools/growth_charge_ab.py`: run the brain twice from an identical seed for equal ticks with growth enabled and disabled, comparing every scalar field of `Brain.state()`; construct only via `cortex.brain.make_brain()` per P4
- [ ] T004 Add the matched A/A control mode to `tools/growth_charge_ab.py` (growth-ON vs growth-ON) which MUST print `0 of N`; a nonzero A/A means the harness has run nondeterminism and every A/B through it is unreadable
- [ ] T005 Add hermetic isolation to `tools/growth_charge_ab.py`: `persist=False` plus an isolated home directory. The same A/B with persistence enabled yields 18 of 102 differing leaves, nearly all cross-run contamination
- [ ] T006 Add raw-internals capture to `tools/growth_charge_ab.py` reading `arm._ext_drive` and `arm._conflict` unrounded alongside `state()` leaves, because `state()` rounds to 3 decimals and the real effect is ~1e-4
- [ ] T007 Encode the forbidden-observable list from contracts/C3 into `tools/growth_charge_ab.py` so that `time_cells_grown`, `time_cells_refused`, `time_cells_width`, `elapsed_time` and `timing_sd` are reported but EXCLUDED from the headline count; the tool must print the honest score as "N of 88 non-tautological"
- [ ] T008 Reproduce the documented baseline with the finished harness and record it in `specs/007-whole-brain-neurogenesis/baseline.md`: expect 5 of 93 differing, all 5 forbidden, honest score 0 of 88, A/A at 0

**Checkpoint**: T008 must reproduce the documented baseline before any story starts. If it does not,
the harness is wrong and fixing it takes priority over every task below.

---

## Phase 3: User Story 1 -- growth that already happens starts changing what the brain does (P1)

**Goal**: make the capacity that already exists change one downstream outcome. If consumption cannot
be made to work where growth already exists and is already earned, it will not work anywhere.

**Independent test**: disable growth, run, compare; at least 1 non-tautological field differs with
the A/A control at 0.

- [ ] T009 [US1] Fix the quantiser at `src/snn_data_lake/cortex/brain.py:1603`: consume the float elapsed estimate instead of `max(1, int(round(el)))`, which today destroys 100 percent of the refinement (`_learned_latency` differs on 0 of 300 ticks)
- [ ] T010 [US1] Re-run `tools/growth_charge_ab.py` after T009 and record whether `_learned_latency` now varies; this is the prerequisite check -- if it still differs on 0 of 300, the fix did not land and no consumer downstream can possibly see growth
- [ ] T011 [P] [US1] Fix site selection in `src/snn_data_lake/cortex/time_cells.py` so `widest_log_gap()` selects the widest log gap AMONG RESOLVABLE gaps; today it argmaxes over gaps equal to within 8.882e-16 and can pick a site the floor then refuses
- [ ] T012 [US1] Apply the Open Question 1 verdict: select the cross-region consumer to target, satisfying contracts/C2 -- outside cortex, branches rather than sums, not permutation-invariant, and natural to what the module already does
- [ ] T013 [US1] Wire the selected consumer to the timing estimate, per contracts/C2. **If the verdict is ONLY_ARTIFICIAL_POSSIBLE, do NOT wire anything**: instead amend `specs/007-whole-brain-neurogenesis/spec.md` to record P1 as blocked and state which success criterion is unachievable as written
- [ ] T014 [US1] Couple growth to the learning path so grown units stop emitting zero, using e-prop / forward-mode credit assignment only -- never `loss.backward()` on a live path
- [ ] T015 [US1] Create `tests/test_neurogenesis_charge.py` asserting SC-001: at least 1 non-tautological field moves, the matched A/A control reads 0, and the forbidden observables are excluded from the count
- [ ] T016 [US1] Assert graceful degradation in `tests/test_neurogenesis_charge.py` (FR-004): the brain functions with growth disabled and no consumer requires that growth has happened
- [ ] T017 [US1] Run all three gates and record results; `run_selfchecks.py`, `pytest`, and `ruff check .` must all stay green

**Checkpoint**: SC-001 moves from 0 of 88 to at least 1 of 88, or P1 is honestly reported blocked.
Both are acceptable outcomes of this phase. Inventing a consumer to make the number move is not.

---

## Phase 4: User Story 2 -- every region can grow its own population (P2)

**Goal**: extend production, but only where a derived floor exists. This phase's gating is what
prevents the feature shipping 15 dormant growers.

**Independent test**: per region, growth occurs under load and is refused when unearned, with
refusals counted.

- [ ] T018 [P] [US2] Create `tools/growth_census.py` reporting per-region producer and consumer status, distinguishing OWN growth code from a BORROWED mechanism the region merely embeds, and printing "N of 15, 2 regions empty and excluded" with the reasoning package reported separately
- [ ] T019 [US2] Make `tools/growth_census.py` state in its own output that name-based scanning reports a FLOOR and not a census, because `GrowingLearner` recruits inline in `observe` rather than in a `grow*` method
- [ ] T020 [P] [US2] Delete the dead growth stub at `src/snn_data_lake/reasoning/readout_ensemble.py:63` whose body is a bare `return`
- [ ] T021 [P] [US2] Delete the dead growth stub at `src/snn_data_lake/reasoning/developmental_loop_brain.py:101` whose body is `return  # linear: no hidden layer to grow`
- [ ] T022 [US2] Collapse the four duplicated `(1,0)` readout splits in `src/snn_data_lake/reasoning/growable_readout.py:176`, `laminar_readout.py:205`, `laminar_readout.py:483` and `laminar_local_readout.py:149` to one, which the source itself documents as "REUSED verbatim"
- [ ] T023 [US2] Run the M25 decision gate from `docs/AUDIT_BACKLOG.md`: register the two seams that already grow (`src/snn_data_lake/hippocampus/growing_learner.py:112`, `src/snn_data_lake/cortex/multimodal_encoder.py:198`), wire one real cross-module producer/consumer pair, and A/B whether the consumer's output moves for a reason other than width
- [ ] T024 [US2] Honour the M25 verdict: on a 0 result retire `GrowthCoordinator` and `register_seam` from `src/snn_data_lake/substrate/growable.py`; on a nonzero result record the measurement that justifies keeping it
- [ ] T025 [US2] Select the single surviving growth mechanism per P2 and record the A/B that justifies it in `specs/007-whole-brain-neurogenesis/research.md`; the bespoke `src/snn_data_lake/cortex/time_cells.py` path must BECOME that mechanism, never sit beside it behind a flag
- [ ] T026 [US2] Derive and A/B a resolution floor for the next region by module count, `src/snn_data_lake/substrate/`, admitting it to growth only if the A/B passes; record "no derivable floor" as an accepted outcome
- [ ] T027 [P] [US2] Derive and A/B a resolution floor for `src/snn_data_lake/hippocampus/`, admitting it to growth only if the A/B passes
- [ ] T028 [P] [US2] Derive and A/B a resolution floor for `src/snn_data_lake/neuromodulation/`, admitting it to growth only if the A/B passes
- [ ] T029 [P] [US2] Derive and A/B a resolution floor for `src/snn_data_lake/plasticity/`, admitting it to growth only if the A/B passes
- [ ] T030 [P] [US2] Derive and A/B a resolution floor for `src/snn_data_lake/cerebellum/`, admitting it to growth only if the A/B passes
- [ ] T031 [US2] Derive and A/B a resolution floor for the `src/snn_data_lake/reasoning/` package, which holds the most growth code in the tree and sits outside the region map
- [ ] T032 [US2] Assert the determinism clauses in `tests/test_neurogenesis_charge.py`: R1 growth decisions consume zero random numbers (`rng.getstate()` bit-identical across attempts), and SC-004 identical seeds produce identical structure in 20 of 20 trials
- [ ] T033 [US2] Assert R4 in `tests/test_neurogenesis_charge.py`: ripple order is a deterministic total order, pinned explicitly rather than relying on insertion-ordered dicts
- [ ] T034 [US2] Record per-tick cost at the GROWN width for every admitted region in `specs/007-whole-brain-neurogenesis/baseline.md`, against the `ms/tick = 56.60 + 0.0804 * N_episodes` budget
- [ ] T035 [US2] Verify per P5 that no admitted region's grown width approaches the measured CUDA crossover of N~24,576-32,768; if one does, re-measure rather than assume the CPU tier still wins
- [ ] T036 [US2] Run all three gates and confirm `substrate/dormancy_gate.py` and `substrate/replication_guard.py` stay green

**Checkpoint**: SC-002 rises above 2 of 15 producers with the denominator stated, SC-007 reaches
exactly 1 surviving mechanism, and every admitted region carries a passing floor A/B.

---

## Phase 5: User Story 3 -- regions enhance each other, continuously and in both directions (P3)

**Goal**: make the producer/consumer relation mutual and always-on.

**Independent test**: suppress one region's contribution while it otherwise runs identically; at
least one OTHER region's outcome changes.

- [ ] T037 [US3] Add suppression-test mode to `tools/growth_charge_ab.py`: suppress a single region's contribution while leaving it present and running, per the spec's US3 independent test
- [ ] T038 [US3] Run the suppression test per producer/consumer pair and record each result as a ConsumptionRecord per data-model.md, including the matched A/A delta which must read 0
- [ ] T039 [US3] Extend `tools/growth_census.py` to report the count of regions satisfying BOTH producer and consumer, with producer-only regions reported as not yet complete rather than counted
- [ ] T040 [US3] Assert FR-005 identity-sensitivity in `tests/test_neurogenesis_charge.py`: for any cross-region signal carrying an identity, scrambling identities while preserving magnitudes must change the consumer's outcome
- [ ] T041 [US3] Verify the consumers built here are compatible with the pending `src/snn_data_lake/cortex/brain.py:1623` identity fix tracked in `docs/P6_INTERCONNECT.md`, so the sibling defect is not solved twice in incompatible ways
- [ ] T042 [US3] Assert FR-003 in `tests/test_neurogenesis_charge.py`: consumption is continuous, not one-shot -- a consumer reading grown capacity only at construction does not satisfy the feature

**Checkpoint**: SC-003 shows at least 1 other region's outcome changing per completed region, and
the BOTH count is reported honestly with its denominator.

---

## Phase 6: Polish and cross-cutting concerns

- [ ] T043 [P] Update `docs/coverage/` with the measured result, citing `file:line` per the evidence discipline; a faculty that cannot be measured stays UNSCORED counting 0 and no row is deleted to raise a score
- [ ] T044 [P] Correct the specs/003 record in `specs/003-grow-the-live-populations/` to state that the held-out benefit arm contributes 0 of 196 refusals and that `resolvable()` produces every refusal
- [ ] T045 [P] Purge the "LIF growth" misnomer per FR-016 from any document this feature produces; `TimeCellClock` is a threshold ladder over one shared accumulator with no per-cell membrane potential, leak, or input weight
- [ ] T046 Register any new module in `src/snn_data_lake/__init__.py` REGIONS and `run_selfchecks.py`, or add it to `substrate/dormancy_gate.py` EXEMPT with a reason
- [ ] T047 Run `tools/citation_guard.py` and confirm exit 0 with STRICT 0 unexempted; do not add files to the exemption list to make it pass
- [ ] T048 Run all three gates a final time and record the results in the commit message with exact counts

---

## Dependencies

```
Phase 1 (T001-T002)
   |
Phase 2 (T003-T008)  BLOCKING -- the harness is the instrument
   |
   +--> Phase 3 US1 (T009-T017)   MVP
           |
           +--> Phase 4 US2 (T018-T036)   needs US1 to prove consumption works
                   |
                   +--> Phase 5 US3 (T037-T042)   needs producers AND consumers to exist
                           |
                           +--> Phase 6 (T043-T048)
```

**Story independence**: US1 is genuinely standalone and is the MVP. US2 depends on US1 only for the
proof that consumption is achievable -- if US1 reports blocked, US2's floor work is still valid but
its regions must be admitted as producer-only. US3 requires both.

**Hard ordering inside US1**: T009 (fix the quantiser) blocks T010, which blocks T012-T013. Wiring a
consumer before the quantiser is fixed measures nothing, because the refinement is destroyed before
it reaches any consumer.

## Parallel execution opportunities

- **Phase 1**: T002 runs alongside T001.
- **Phase 2**: none. The harness tasks build one file in sequence, and T008 validates the whole.
- **US1**: T011 (site selection) is independent of the T009-T013 consumer chain.
- **US2**: T020, T021 are independent deletions. T027-T030 are four independent per-region floor
  derivations that touch different regions and can run concurrently once T026 establishes the
  pattern.
- **Phase 6**: T043, T044, T045 touch different documents.

## Implementation strategy

**MVP is Phase 1 + Phase 2 + US1.** That delivers a brain whose existing growth finally changes an
outcome, plus the harness that proves it -- valuable even if no new region ever grows.

**Stop-honestly boundaries.** The feature can be halted after any checkpoint with a truthful report:
after US1, "growth is charged in 1 place"; after US2, "N of 15 regions produce under a derived
floor"; after US3, "N of 15 both produce and consume". None of these requires overstating the
others.

**The failure mode to watch.** If SC-001 starts passing, check FIRST that T012-T013 did not invent
the consumer. The test is: would this consumer exist, and read this quantity, if the feature had
never been written? If not, T013's blocked branch is the correct path -- and a blocked P1 honestly
reported is a better outcome than a passing number that means nothing.

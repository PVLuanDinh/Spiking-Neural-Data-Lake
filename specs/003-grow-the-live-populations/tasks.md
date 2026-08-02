---

description: "Task list for Grow the Live Populations (Autonomous LIF Growth)"
---

# Tasks: Grow the Live Populations (Autonomous LIF Growth)

**Input**: Design documents from `/specs/003-grow-the-live-populations/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/time_cells.md, quickstart.md

**Baseline commit**: `c6ca8c8d`. Verify the tree is clean for `cortex/time_cells.py` and
`cortex/brain.py` before capturing any baseline figure — an earlier research pass measured an
uncommitted rewrite and mislabelled it as the baseline.

**Tests**: INCLUDED. Not optional here — the constitution makes three gates binding
(`run_selfchecks.py`, `pytest` at >=70% coverage, `ruff`), and Standing Invariant 5 makes the
degenerate-ladder control an admission gate rather than a nicety.

**Organization**: Tasks grouped by user story. **Phase order deviates from spec priority order** —
see the note below.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US1–US4 map to the spec's user stories
- Exact file paths in every description

## Path Conventions

Single project. Source at `src/snn_data_lake/`, probes at `tools/`, tests at `tests/`.

---

## ⚠️ Phase order deviates from spec priority order

The spec prioritises US1 (growth) as P1. **US2 (readout) must be implemented first.** Measured, 20
paired seeds at HEAD: `decode()` reads population *width* directly (`time_cells.py:143`), so appending
a cell flips a saturated read out of the clamp branch — up to **4.367 ticks** of change to
`_elapsed_time` at horizon 70, with no cell ever firing. That is pure clamp-widening, exactly the
artefact FR-006 disqualifies. US2 also changes its magnitude (4.367 → 1.633), so any US1 evidence
recorded before US2 is invalidated by US2.

Story *labels* still map to the spec. Only the execution order differs.

---

## Phase 1: Setup

**Purpose**: Capture the baseline before anything changes, so every later claim has a control.

- [X] T001 Verify baseline: run `git diff --quiet HEAD -- src/snn_data_lake/cortex/time_cells.py src/snn_data_lake/cortex/brain.py` and abort if dirty; record `git rev-parse HEAD` into `specs/003-grow-the-live-populations/baseline.txt`
- [X] T002 [P] Confirm the three gates are green at baseline: `python run_selfchecks.py`, `pytest`, `python -m ruff check .` — record pass/fail in `specs/003-grow-the-live-populations/baseline.txt`
- [X] T003 [P] Record baseline readout RMSE vs true elapsed at widths 64/128/256/512, horizon 60, 16 seeds, into `specs/003-grow-the-live-populations/baseline.txt` — expect the flat 0.7276 that proves growth is currently worthless

**Checkpoint**: baseline captured and attributable to a named commit.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Two defect fixes owed regardless of growth, plus the control that admits any readout.

**⚠️ CRITICAL**: No user story work may begin until this phase is complete. FR-012 is unsatisfiable
without T004, and no readout may be adopted without T007.

- [ ] T004 Make `learn_interval` order-preserving in `src/snn_data_lake/cortex/time_cells.py`: compute the proposed value, and apply it only if it lies strictly between its neighbours (lower bound `0.5` at index 0, `+inf` above the top); otherwise drop the update. Do NOT clamp — clamping was measured to collapse gaps to 1 ULP (2.220e-16) versus 5.298e-03 under refusal (FR-012)
- [ ] T005 Add a `drive > 0` input guard beside the existing `n_cells` check in `src/snn_data_lake/cortex/time_cells.py`
- [ ] T006 [P] Add `test_learn_interval_preserves_ascending_order` to `tests/test_time_cells.py`: run 4000 ticks in both the reset-driven and saturating regimes across 12 seeds, assert the vector is strictly ascending at 200/600/1200/4000 ticks. Expect 0 of 12 inverted after T004 (9 of 12 invert before it) (FR-012)
- [ ] T007 Add `test_degenerate_ladder_control` to `tests/test_time_cells.py`: run the module's own accuracy assertion with `thresholds = [1.0 + 1e-6*i for i in range(64)]` and assert it **FAILS**. This is the Standing Invariant 5 admission gate — a readout that passes this is not reading its population and is disqualified (FR-005)
- [ ] T008 [P] Verify `learn_interval` coverage is unchanged by T004 in `tests/test_time_cells.py` — assert roughly 49.8 of 64 thresholds still move over the measured window, so the fix refuses order-breaking updates without suppressing learning
- [ ] T009 Re-run all three gates with T004–T008 applied (`python run_selfchecks.py`, `pytest`, `python -m ruff check .`) and append the result to `specs/003-grow-the-live-populations/baseline.txt`

**Checkpoint**: the invariant holds, the admission gate exists, and two latent defects are closed.

---

## Phase 3: User Story 2 — The population's own output can register the population (Priority: P2, executed FIRST)

**Goal**: Replace the ramp readout with one that reads the population, so growth becomes measurable at all.

**Independent Test**: Perturb one threshold; the consumed estimate must change. At baseline it is bit-identical.

### Tests for User Story 2

- [ ] T010 [P] [US2] Add `test_decode_depends_on_thresholds` to `tests/test_time_cells.py`: perturb `thresholds[30]` by `+0.4`, assert the estimate sequence over 60 ticks differs from unperturbed (FR-005, SC-002)
- [ ] T011 [P] [US2] Add `test_decode_rmse_falls_with_width` to `tests/test_time_cells.py`: assert RMSE at n=512 is strictly below n=64 over horizon 60 — the C2 criterion, which the baseline readout fails (flat 0.7276) (FR-005)
- [ ] T012 [P] [US2] Add `test_decode_divides_by_drive` to `tests/test_time_cells.py`: at `drive=0.5` and true elapsed 40, assert the reported estimate is ~40, not ~19.34

### Implementation for User Story 2

- [ ] T013 [US2] Replace the estimate branch of `decode()` in `src/snn_data_lake/cortex/time_cells.py` with the geometric-mid population readout per `contracts/time_cells.md` §1: `0.0` when nothing fired, top nominal latency when saturated, else `sqrt(thresholds[k-1]*thresholds[k])/drive`. Leave `sd` byte-for-byte unchanged (FR-005)
- [ ] T014 [US2] Remove the `estimate = base` saturation clamp in `src/snn_data_lake/cortex/time_cells.py` — it is F9's fake-capability regime (measured RMSE 292.8993 at t=600) (FR-006)
- [ ] T015 [US2] Update the `decode()` docstring in `src/snn_data_lake/cortex/time_cells.py` to describe what the readout now actually does; the current text claims a "rank-order + tempotron-style readout" while reading the analog accumulator
- [ ] T016 [US2] Run T007's degenerate-ladder control in `tests/test_time_cells.py` against the new readout in `src/snn_data_lake/cortex/time_cells.py` and confirm it still FAILS — if it passes, the readout is disqualified and must be replaced, however good its accuracy looks (FR-005)
- [ ] T017 [US2] Run all three gates; confirm the module self-check's accuracy assertion, monotonicity (`tests/test_time_cells.py:13`), Weber/scalar timing (`:20`) and same-seed determinism (`:46`) all pass **without any assertion being loosened** (SC-009)
- [ ] T018 [US2] Measure the saturation-regime split on the live brain in `tools/timing_heldout.py` (or a probe under `tools/`): across >=3 seeds and >=600 ticks, report the proportion of clock-advancing ticks that are unsaturated vs saturated, and write it to `specs/003-grow-the-live-populations/regimes.md`. Baseline expectation is 92.5–95.8% unsaturated. Every later claim must name the regime it was measured in (SC-003)

**Checkpoint**: the readout reads its population. Expect +45% RMSE at n=64 (≈1.05 vs 0.7276) — the accepted, documented cost of measurability. US1 evidence is now valid.

---

## Phase 4: User Story 1 — One population can gain a cell without losing what it learned (Priority: P1)

**Goal**: Non-destructive, refusable growth, observable in exported state.

**Independent Test**: Teach a clock, snapshot thresholds, grow, compare — all pre-existing values bit-identical.

### Tests for User Story 1

- [ ] T019 [P] [US1] Add `test_growth_preserves_learned_thresholds` to `tests/test_time_cells.py`: teach a clock, snapshot, grow 64→65→66, assert every pre-existing threshold is **bit-identical** (exact float comparison, not rounded) (FR-001, SC-001)
- [ ] T020 [P] [US1] Add `test_growth_is_rng_neutral` to `tests/test_time_cells.py`: assert `rng.getstate()` is identical before and after growth — this is what keeps same-seed determinism green
- [ ] T021 [P] [US1] Add `test_growth_refuses` to `tests/test_time_cells.py`: assert `None` is returned and no state changes when away from a reset boundary, and when the vector is already non-ascending; assert a prune can never reduce the population to zero cells (FR-004)
- [ ] T022 [P] [US1] Add `test_growth_maintains_length_coherence` to `tests/test_time_cells.py`: assert `len(thresholds) == len(fired) == len(spike_tick) == n_cells` after growth

### Implementation for User Story 1

- [ ] T023 [US1] Implement `grow(mode)` in `src/snn_data_lake/cortex/time_cells.py` per `contracts/time_cells.md` §2. Interior mode inserts at the geometric midpoint of the widest log-gap; extend mode appends at `n_cells + 1`. Mutate `n_cells` **last** so every intermediate state stays in-bounds for `tick()` (FR-001, FR-003)
- [ ] T024 [US1] Enforce the reset-boundary precondition in `src/snn_data_lake/cortex/time_cells.py` (`n_fired == 0`, `v == 0.0`, all `spike_tick` `None`) — away from a boundary, interior insertion either raises `TypeError` at the deviations loop or fabricates a crossing time that shifts the live-consumed `sd` by +0.005256 with zero ticks elapsed (FR-004)
- [ ] T025 [US1] Add the strict-ascending assertion and the length-coherence assertion to `grow()` in `src/snn_data_lake/cortex/time_cells.py` (FR-012)
- [ ] T026 [US1] Add `grown` and `refused` counters to `TimeCellClock` in `src/snn_data_lake/cortex/time_cells.py`; increment on every outcome, including refusals (FR-011)
- [ ] T027 [US1] Enumerate every consumer of `state()` before changing it — at minimum `docs/coverage/artifacts/ab_method.py`'s behavioural digest, `substrate/cockpit.py`, and any golden comparison in `tests/` — and record the list in `specs/003-grow-the-live-populations/state-consumers.md`
- [ ] T028 [US1] Export `time_cells_width`, `time_cells_grown`, `time_cells_refused` from `state()` in `src/snn_data_lake/cortex/brain.py`, following the existing naming convention (FR-011, SC-001, SC-007)
- [ ] T029 [US1] Make a growth *failure* distinguishable from a growth *refusal* in `src/snn_data_lake/cortex/time_cells.py` and `src/snn_data_lake/cortex/brain.py`: an exception on the growth path must surface as its own observable outcome, never collapse into the refusal counter. Add a test to `tests/test_time_cells.py` asserting the two are distinguishable in exported state (FR-013)

**Checkpoint**: growth is non-destructive, refusable, and observable. Still off by default.

---

## Phase 5: User Story 3 — The added cell is measured, and may be found worthless (Priority: P3)

**Goal**: A held-out verdict, with a control, that is permitted to be a null.

**Independent Test**: `python tools/timing_heldout.py --seeds 12` returns 0, 1, or 2 with an effect and its dispersion.

### Tests for User Story 3

- [ ] T030 [P] [US3] Add `test_split_is_disjoint_and_covers_every_gap` to `tests/test_timing_heldout.py`: assert teach and test sets are disjoint and that all 63 gaps hold at least one held-out interval — a literal trailing-20% split covers only 4 of 63 (FR-009)
- [ ] T031 [P] [US3] Add `test_instrument_refuses_blind_readout` to `tests/test_timing_heldout.py`: pass a readout that ignores thresholds and assert exit code 2 `NOT-ELIGIBLE`, not a null (FR-005)

### Implementation for User Story 3

- [ ] T032 [US3] Create `tools/timing_heldout.py` taking the readout `R(clock, tau)` as a parameter, with explicit path/param CLI arguments and no value computed from no data (FR-002)
- [ ] T033 [US3] Implement the interval universe in `tools/timing_heldout.py`: 10 points per gap at `lo + (hi-lo)*(p+0.5)/10`, ordered gap-major, deterministic positional 1-in-5 stride (`index % 5 == 4` → TEST). No RNG. Expect 630 intervals → 504 teach / 126 test at n=64 (FR-009)
- [ ] T034 [US3] Implement the FR-005 self-test in `tools/timing_heldout.py`: perturb one mid threshold, require `R`'s output to change, else print `NOT-ELIGIBLE` and exit 2. Also run the degenerate-ladder control on `R` as an admission precondition (FR-005)
- [ ] T035 [US3] Implement the paired statistic in `tools/timing_heldout.py` — mean/sd/se/wins/ties over per-seed differences, `|mean/se|` reported at both the 2.0 and 3.0 bars. Port the arithmetic from `scratchpad/grow_on_top_nonstationary.py:130-137` with attribution in a comment; do NOT import it (gitignored, and `tools/citation_guard.py:320` forbids citing it) (FR-002, SC-004)
- [ ] T036 [US3] Implement the mandatory null-control arm in `tools/timing_heldout.py` — insertion at the lowest-teach-residual gap. Its measured zero is what licenses reading the main arm as contribution rather than as a benefit any added cell would receive (FR-006, SC-005)
- [ ] T037 [US3] Enforce output discipline in `tools/timing_heldout.py`: print every count as `N of M` never `N/M` (the artifact fingerprint harvests bare ratios positionally), print tie counts unconditionally, and name the held-out split, the control, and the saturation regime from `specs/003-grow-the-live-populations/regimes.md` in the output (FR-015, SC-003)
- [ ] T038 [US3] Handle the probe environment in `tools/timing_heldout.py`: insert the repo root on `sys.path` (probes run with `PYTHONPATH=<root>/src` only, which drops the brain to a blind stdlib perception tier) and do not assume accumulated durable state (`NORD_HOME` is forced to a fresh temp dir)
- [ ] T039 [US3] Run the instrument at 12 seeds and record the artifact: `python tools/coverage_score.py record --kind heldout --name time_cells_growth_heldout -- python tools/timing_heldout.py --seeds 12` (note the literal `--`), then verify with `rerun` (FR-015)
- [ ] T040 [US3] Write the verdict — benefit or explicit null — into `docs/coverage/README.md`, naming the held-out split, the control, and the saturation regime. A null stops the feature here and that is success, not failure (FR-008, FR-010, SC-005)

**Checkpoint**: a real number with a control, or an honest null. US4 proceeds only on a benefit.

---

## Phase 6: User Story 4 — The brain grows itself, and can be told no (Priority: P4)

**Goal**: Growth survives restart, happens autonomously, and halts observably.

**Independent Test**: Grow, restart the daemon, confirm the grown width and learned thresholds return.

### Tests for User Story 4

- [ ] T041 [P] [US4] Add `test_persist_restore_round_trip` to `tests/test_time_cells.py`: grow, save, restore, assert width and every threshold survive bit-identically (FR-007, SC-006)
- [ ] T042 [P] [US4] Add `test_restore_refuses_foreign_lineage` to `tests/test_time_cells.py`: assert restore refuses and records the refusal on fmt mismatch, length mismatch, seed/drive/noise_sd mismatch, and a non-ascending vector (FR-007, FR-013)

### Implementation for User Story 4

- [ ] T043 [US4] Implement save/restore of `{fmt, n_cells, seed, drive, noise_sd, thresholds}` via `home.save_state`/`load_state` in `src/snn_data_lake/cortex/time_cells.py`. Do **not** use the lake: `_T_GROWTH` is 180.0 of 187.5 MB (96%) of it, `load_growth` costs 0.400 s and 187.5 MB resident per Brain construction versus 0.110 ms, and overwrite is the prune policy so no retention constant is introduced (FR-007, FR-014)
- [ ] T044 [US4] Add the restore hook at Brain level in `src/snn_data_lake/cortex/brain.py`, immediately after the construction block, in its **own** `try` that does not share the `:736-752` handler (which nulls five populations). Never inside `TimeCellClock.__init__`, or the module self-check reads the previous gate run's state (FR-013)
- [ ] T045 [US4] Implement wholesale width adoption in `src/snn_data_lake/cortex/brain.py`: rebuild the object at the persisted width and assign the persisted vector. Never mutate `n_cells` in place, never merge element-wise, never truncate or pad (FR-001, FR-007)
- [ ] T046 [US4] Gate the restore so brain construction stays a pure function of its seed in `src/snn_data_lake/cortex/brain.py` — an unconditional load previously made construction non-deterministic in ~42% of trials
- [ ] T047 [US4] Implement the autonomous growth trigger behind an off-switch in `src/snn_data_lake/cortex/brain.py`, attempting growth only at reset boundaries and only while added cells still earn their place (FR-008, FR-016)
- [ ] T048 [US4] Implement the halt condition in `src/snn_data_lake/cortex/time_cells.py` so a sustained no-benefit regime stops growth, and the halt is visible via the refusal counter rather than inferred from absence (FR-011, SC-007)
- [ ] T049 [US4] Measure the durable footprint of growth per unit time and record the rate with a bounded/unbounded verdict in `specs/003-grow-the-live-populations/bounding.md` (FR-014, SC-008)

**Checkpoint**: the brain grows itself, survives restart, and can be refused.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T050 [P] Confirm the P3 dormancy gate's treatment of a flag-gated `grow()` by running `substrate/dormancy_gate.py` and `substrate/method_dormancy_gate.py` — the plan flags this as a check owed, not an assumption
- [ ] T051 [P] Verify the off-switch in `src/snn_data_lake/cortex/brain.py`: with growth disabled, assert live behaviour matches the pre-feature baseline recorded in `specs/003-grow-the-live-populations/baseline.txt`, and all gates pass with no assertion loosened (FR-016, SC-009)
- [ ] T052 [P] Correct the stale ledger entries in `docs/coverage/README.md:239`, which still quotes the retired integer ladder `[1.0, 2.0, 3.0, 4.0]`, and note that `time_cells_ab_eprop.txt` is 9 commits stale and predates the three commits that made `learn_interval` learnable
- [ ] T053 Run `ruff format src tests tools` over `src/snn_data_lake/cortex/time_cells.py`, `src/snn_data_lake/cortex/brain.py`, `tools/timing_heldout.py` and `tests/test_time_cells.py`, then the full three-gate suite one final time

---

## Dependencies

```
Phase 1 (Setup)
   ↓
Phase 2 (Foundational) ── BLOCKING for everything
   ↓
Phase 3 US2 (readout) ── BLOCKING for US1 evidence and US3
   ↓
Phase 4 US1 (growth) ────┐
   ↓                     │
Phase 5 US3 (instrument) ┘  ← needs both a growth mechanism and a population-sensitive readout
   ↓
Phase 6 US4 (autonomous) ← proceeds ONLY on a benefit verdict from T040
   ↓
Phase 7 (Polish)
```

**Hard gates**

- T004 blocks all growth work — FR-012 is unsatisfiable without it.
- T007 blocks T013 — no readout may be adopted before the control that can disqualify it exists.
- T016 can invalidate T013 and force a different readout.
- T018 must land before any claim in Phase 5, which has to name the regime it was measured in.
- T040 returning a null **stops the feature** before Phase 6. That is a success condition.

## Parallel Opportunities

- **Phase 1**: T002, T003 in parallel after T001.
- **Phase 2**: T006, T008 in parallel; both depend on T004.
- **Phase 3**: T010, T011, T012 in parallel (all test additions, written before T013). T018 last.
- **Phase 4**: T019–T022 in parallel (four independent test additions).
- **Phase 5**: T030, T031 in parallel. T032–T038 are sequential — one file.
- **Phase 6**: T041, T042 in parallel.
- **Phase 7**: T050, T051, T052 in parallel; T053 last.

## Implementation Strategy

**MVP scope is Phase 2 + Phase 3 (US2), not US1.** That is the deviation worth stating plainly: the
smallest shippable increment is the two defect fixes plus the readout replacement. It closes a live
saturation bug (RMSE 292.8993 at t=600), a `drive`-scaling unit bug (19.34 reported for a true elapsed
of 40), and an invariant violation that strands cells on 9 of 12 seeds — and it is worth shipping
whether or not a single cell is ever grown.

**Incremental delivery**

1. Phases 1–2 — two bug fixes and an admission gate. Ship.
2. Phase 3 — the readout. Ship. Growth becomes measurable for the first time.
3. Phase 4 — growth exists, off by default. Ship.
4. Phase 5 — the verdict. **May end the feature honestly.**
5. Phase 6 — only on a benefit.

**What would make this feature stop early, legitimately**: T016 finding no admissible readout, or T040
returning a null. Both are recorded outcomes under FR-010, not failures.

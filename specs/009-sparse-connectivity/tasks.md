---

description: "Task list for sparse connectivity -- bound fanout independently of population size"
---

# Tasks: Sparse connectivity

**Input**: Design documents from `/specs/009-sparse-connectivity/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/sparse-wiring.md, quickstart.md

**Tests**: REQUIRED. The acceptance criterion IS a measurement, and every other quantity in this
feature is maximised by deleting connections. A task that ships a saving without its function
measurement has produced no evidence.

**Organization**: grouped by user story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelizable (different files, no dependency on an incomplete task)
- **[Story]**: US1 / US2 / US3

## Path Conventions

`src/snn_data_lake/<region>/`, `tests/`, `tools/` at repo root. Interpreter
`.venv-lake\Scripts\python.exe` with `PYTHONPATH=src`.

## The rule that governs every task below

**Function is the constraint; the saving is the benefit.** Memory, density, fanout and ceiling all
improve monotonically as connections are deleted, so none of them is evidence on its own. Every
SPARSE verdict carries its dense baseline at the same width AND load.

---

## Phase 1: Setup

- [x] T001 Record the pre-feature baseline: connectivity census (171 structures, 99.57 percent dense;
      103 true synaptic, 99.82 percent) and the dense ceiling of 40,000 units in 12.8 GB. DONE in
      `specs/009-sparse-connectivity/research.md`
- [x] T002 Pre-register the tolerance BEFORE any sweep: mean 0.02 / min 0.05 vs dense at the same
      width, load and seed set; F* must be bracketed; grid-floor F* reported censored; a drop above
      0.20 is a CLIFF. DONE in research.md Decision 2

---

## Phase 2: Foundational (BLOCKING)

- [x] T003 Validate `retained()` as a metric that can LOSE, with two independent nulls both reading
      0: a row-shuffle preserving nnz and per-row value multiset, and a learning-destroyed null at
      matched fanout. DONE -- research.md Decision 1
- [x] T004 Create `tools/fanout_bisect.py`: bracketed bisection against a module's own metric, with
      the pre-registered tolerance and the censoring rule encoded, refusing to report an unbracketed
      F*. DONE -- self-check 4 of 4 properties
- [x] T005 Make `tools/fanout_bisect.py` reject a metric that cannot decrease and a dense baseline
      that is already failing -- both produced meaningless results during research. DONE -- a
      baseline below 0.5 raises rather than anchoring the comparison

---

## Phase 3: US1 -- one module works at reduced fanout (P1)

**Goal**: the smallest slice that can falsify the whole idea.

- [x] T006 [US1] `_wire()` in `src/snn_data_lake/hippocampus/episode_consolidation.py`: symmetric
      uniform random k-out from `random.Random(seed)`, separate from the instance's numpy generator
      so wiring cannot consume draws replay depends on. DONE
- [x] T007 [US1] `_zero_offmask()`: route every weight update through ONE pruning point, so a future
      update site cannot silently restore all-to-all. DONE -- replaced 4 `fill_diagonal` sites
- [x] T008 [US1] `degree()` reporting the REALISED degree, which symmetrisation raises above the
      requested F. DONE
- [x] T009 [US1] `tests/test_sparse_connectivity.py`: function FIRST, savings second, determinism,
      symmetry, mask containment, dense-unchanged. DONE -- 7 of 7 pass
- [x] T010 [US1] Measure function at N=256 and N=1,024: F=128 gives 23.4 percent density at
      `retained` 1.000 against a dense baseline of 1.000. DONE
- [ ] T011 [US1] Re-run T010's comparison through `tools/deep_fingerprint.py` with a matched A/A
      control, so the "no downstream change" half is measured with the right instrument rather than
      inferred from the module metric alone
- [ ] T012 [US1] Run all three gates and record the results

**Checkpoint**: function within tolerance at reduced fanout, or the approach is wrong and the
feature stops. **Reached.**

---

## Phase 4: US2 -- the saving grows with population (P2)

**Goal**: bend the memory curve, verified at widths the bound was NOT tuned on.

- [ ] T013 [US2] **Measure whether THIS brain's stored load scales with its population.** Instrument
      episode count against `dim` over a long live run. **The whole feature turns on this**: at fixed
      load F* falls 176 to 128; at proportional load uniform F* grows as N^0.68
- [ ] T014 [US2] Bisect F* at N=4,096 and N=16,384 using `tools/fanout_bisect.py`, confirming the
      research figures independently
- [ ] T015 [US2] Verify at a width at least 4x beyond the tuned one (SC-008); everything above 10^4
      is currently EXTRAPOLATED and must stay labelled as such
- [ ] T016 [US2] [P] Confirm top-k-per-row holds F* at 96 to 128 under PROPORTIONAL load beyond
      4,096 -- it is the result that makes proportional load survivable
- [ ] T017 [US2] Fit and record connection memory against population; report the ceiling with the
      resource that binds it, never as "unbounded"
- [ ] T018 [US2] Re-measure the GPU crossover under bounded fanout (P5) -- every prior crossover
      number assumed a fanout this feature changes

**Checkpoint**: SC-001 and SC-003 hold at the brain's REAL load, not only at fixed load.

---

## Phase 5: US3 -- the whole brain sparse, dense retired (P3)

**Goal**: the end state, and the only point at which the population ceiling actually moves.

- [ ] T019 [US3] Build task metrics for the 9 of 12 owner classes that have none. Each must be able
      to DEGRADE and must pass a learning-destroyed null
- [ ] T020 [US3] [P] Thin `hippocampus/sleep_replay.py` `AttractorNet` -- **first resolving its
      contaminated evidence**: the assembly-derived mask is an ORACLE on it (untrained masked net
      scores 1.000), so it needs a clean metric before any verdict
- [ ] T021 [US3] [P] Thin `cortex/predictive_coding.py` `PredictiveCodingNetwork` (1,152 cells)
- [ ] T022 [US3] [P] Thin `cerebellum/jepa_world_model.py` (2,512 cells)
- [ ] T023 [US3] [P] Thin `plasticity/criticality_regulator.py` (1,536 cells; `separability()` already
      demonstrated to fall)
- [ ] T024 [US3] Record `SpikingGridCAN` as OUT OF SCOPE with its measurement -- connectivity is
      already O(N) in a real torus-phase metric
- [ ] T025 [US3] Record every module that genuinely needs dense connectivity as a DOCUMENTED
      EXCEPTION with the measurement showing it (FR-003)
- [ ] T026 [US3] **Retire the dense path per site once F is measured there** -- this closes the P2
      deviation recorded in plan.md Complexity Tracking. Exactly one mechanism survives
- [ ] T027 [US3] Whole-brain A/B sparse vs dense through `tools/deep_fingerprint.py` with a matched
      A/A control at 0

**Checkpoint**: "N of M sparse, K documented exceptions, U unmeasured" with all four numbers, and
SC-006 at exactly 1 mechanism.

---

## Phase 6: Polish

- [ ] T028 [P] Update `docs/coverage/` with the measured result, citing `file:line`
- [ ] T029 [P] Record in `docs/POLICIES.md` P5 that the crossover was re-measured under bounded
      fanout, or that it was not and why
- [ ] T030 Run `tools/citation_guard.py` (exit 0, STRICT 0) and all three gates; record exact counts

---

## Phase 7: The durable lake (added 2026-08-04 after a scope audit)

**Why this phase exists.** 009's Phase 0 census searched for 2-D numeric weight structures, so THE
ONE DURABLE LAKE was never in scope -- a store is not a matrix and the census could not see it. That
was defensible until P4b declared the lake part of the brain. Audited 2026-08-04:

```
009 covers 1 of 12 connectivity-bearing modules and 0 of the lake's 4 record types
_T_SPIKE                     : 0 writers anywhere in the tree
save_growth                  : a flat bytes blob -- no structure, no mask, no adjacency
EpisodeConsolidator W/_mask  : never serialised
```

**What the audit CLEARED, so it is not work**: sparse wiring survives a restart with no lake record
at all. Measured -- wiring identical True, weights identical True, retained 1.000 before and after --
because the mask is a pure function of (seed, dim, fanout) and the weights rebuild from the lake's
episodes. The lake stores what was LEARNED; the seed regenerates HOW IT IS WIRED. That split is
correct and needs no task.

- [ ] T031 A retention-policy change silently REWIRES the store, and nothing detects it. Measured:
      `capacity 64 -> dim 256` versus `capacity 96 -> dim 384` gives a different mask shape and a
      different network over the same episodes. Changing `EPISODE_RETENTION_DAYS` in
      `src/snn_data_lake/cortex/brain.py` looks like a bookkeeping tweak and is actually a rewiring.
      Record the (seed, dim, fanout) a store was built with, and report -- loudly -- when a restored
      brain rebuilds a different network over the same lake
- [ ] T032 [P] `_T_SPIKE` (0x02) has **0 writers** anywhere in the tree: a reserved record type
      nothing uses, which is dormant by P3's own definition sitting in the lake's tag space. Either
      wire it or retire the tag with a reason recorded in `src/snn_data_lake/substrate/home.py`
- [ ] T033 [P] `save_growth()` takes a flat bytes blob with no structure, so a module whose learned
      wiring is NOT reproducible from a seed could not persist it through the current lake API.
      `EpisodeConsolidator` is safe only because its wiring is seeded -- that is a property of this
      module, not a guarantee the lake offers. Decide whether the lake should carry structure, and
      record the answer either way
- [ ] T034 Extend the Phase 0 census to the lake itself: report the 4 record types, their writers,
      their readers, and their record counts, so "N of M sparse" is joined by an honest statement of
      what the durable store holds and who reads it
- [ ] T035 Re-scope SC-007's denominator. It currently counts connectivity-bearing MODULES; after
      P4b the lake is part of the brain, so the feature must say what it covers of the whole brain
      rather than of the matrices it happened to census

**Checkpoint**: the feature can state its coverage over the brain INCLUDING its durable store, not
only over the weight matrices its census could see.

---

## Dependencies

```
Phase 1 (T001-T002)  DONE
   |
Phase 2 (T003-T005)  T003 DONE; T004-T005 BLOCK Phase 4
   |
   +--> Phase 3 US1 (T006-T012)   MVP -- T006-T010 DONE
           |
           +--> Phase 4 US2 (T013-T018)   T013 is the gating measurement
                   |
                   +--> Phase 5 US3 (T019-T027)
                           |
                           +--> Phase 6 (T028-T030)
```

Phase 7 (T031-T035) is INDEPENDENT of the user stories and can start immediately: it audits the
durable store rather than the weight matrices, and nothing in US1-US3 blocks it. T031 is the urgent
one -- a retention-policy change silently rewires the store TODAY.

**T013 gates everything after it.** If this brain's stored load scales with its population, the
uniform-random bound grows as N^0.68 and US2's memory claim weakens from linear to O(N^1.68); the
top-k-per-row path (T016) becomes the design rather than an alternative.

## Parallel execution opportunities

- **US2**: T016 is independent of T014-T015.
- **US3**: T020-T023 are four different modules and can run concurrently once T019 establishes the
  metric pattern.
- **Phase 6**: T028 and T029 touch different documents.

## Implementation strategy

**MVP is Phase 1 + 2 + US1** -- and it is already reached: one module thinned to 23.4 percent
density with function intact, which is what makes the rest worth doing.

**Stop-honestly boundaries.** After US1: "1 module sparse at 2 widths, everything else unmeasured."
After US2: "the bound holds at the brain's real load" or "it does not, and here is the exponent."
After US3: "N of M sparse." None of these requires overstating the others.

**The debt to close.** SC-006 currently reads 2 mechanisms, not 1. `fanout=None` keeps the dense path
alive because P1 forbids defaulting F. That is a real P2 deviation, recorded in plan.md rather than
argued away, and T026 is where it closes.

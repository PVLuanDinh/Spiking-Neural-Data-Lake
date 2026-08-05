# Implementation Plan: Sparse connectivity -- bound fanout independently of population size

**Branch**: `009-sparse-connectivity` | **Date**: 2026-08-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/009-sparse-connectivity/spec.md`

## Summary

Connectivity is all-to-all -- `nnz/row = N - 1` exactly, 98.99 percent dense across the live brain --
which caps the population near 40,000 in 12.8 GB of VRAM and makes a million-neuron brain need
8.0 TB of connections. Phase 0 measured the one thing the design turns on:

**The knee is set by STORED LOAD, not by width and not by density.** Both of those were asserted
during research -- one of them by me in the main session -- and both were falsified on the same
module with the same metric.

```
FIXED load (40 episodes)          PROPORTIONAL load (P = N/8)
   N       F*   knee density         dim    uniform F*   top-k-per-row F*
 256      176       68.8%            256          160            96
1024      160       15.6%           1024          320            96
4096      160        3.9%           2048          512           128
16384     128        0.78%          4096        1,024             -
```

At fixed load F\* FALLS 176 to 128 over a 64x width range. At proportional load the uniform knee
grows as `N^0.68` while top-k-per-row stays at 96 to 128 -- which is the result the design rests on.

So the approach is: **measure F per deployment, never default it**; wire uniform random k-out
because the module's own semantics admit nothing else; and treat function preservation as the
constraint with scale as the benefit, because every memory and fanout number improves monotonically
as connections are deleted and would be maximised by destroying the brain.

## Technical Context

**Language/Version**: Python >=3.11 (CI matrix 3.11 / 3.12 / 3.13)

**Primary Dependencies**: `numpy`, already a core dependency. No new dependency.

> **P7 VOID (annotated 2026-08-04).** The sentence above certifies "no new dependency" as a constitution check. **That is not a criterion under P7** (`docs/POLICIES.md`), which states that *"it would be another dependency" is NOT a valid reason to reject anything* and requires a rejection to cite a measurement on the real workload or a structural argument. No library was evaluated here, so the check scored nothing. Left in place rather than deleted because this file records a decision that was made; annotated because P7 names these certifications as live residue of the zero-dependency rule abolished 2026-07-20. `specs/010/plan.md` is the first plan to decline the certification on purpose.

**Storage**: unchanged. Sparse wiring is in-memory structure; nothing about the durable lake changes.

**Testing**: `pytest` (>=70 percent coverage), `run_selfchecks.py`, `ruff check`. Nulls use
`tools/deep_fingerprint.py` with a matched A/A control -- never `Brain.state()`, a measured
false-null instrument.

**Target Platform**: Windows 11 + Linux CI. `.venv-lake` is the authoritative deterministic gate.

**Performance Goals**: connection memory `O(N*F)` rather than `O(N^2)`. Measured ceiling at fixed
load: 12,500,000 units in 12.8 GB at 8 B/synapse against the dense baseline of 40,000 -- **312x**,
clearing SC-004's 10x.

**Constraints**:
- Function preservation is the acceptance criterion. Tolerance PRE-REGISTERED: mean within 0.02 and
  minimum within 0.05 of the dense baseline at the same width, load and seed set.
- Identical seeds produce identical wiring (FR-010), 20 of 20.
- The effector envelope does not widen.
- ASCII-only `print()`; counts `N of M`.

**Scale/Scope**: 12 connectivity-bearing owner classes, 103 true synaptic structures, 381,422 cells
at 99.82 percent density. `EpisodeConsolidator` owns 17.2 percent and is the P1 target;
`PcCortexServe` owns 80.4 percent but is `O(in_dim * d1)` over an N-MNIST sensor frame, not `O(N^2)`
over a population, so thinning it saves bytes without bending the curve.

## Constitution Check

| Principle | Assessment | Status |
|---|---|---|
| **P0** self-improvement | Removes the quadratic ceiling that makes neural growth unaffordable at scale. specs/007's growth costs `O(N)` new connections per unit under all-to-all wiring; bounded fanout is its precondition. | PASS |
| **P1** no hardcode | **F is not defaulted.** The knee tracks stored load, so any baked-in number would be a magic constant. F is measured per deployment by bracketed bisection against the module's own metric. Every closed form attempted in research relocated the constant it was meant to remove. | PASS |
| **P2** non-additive | **DEVIATION -- see Complexity Tracking.** `fanout=None` currently keeps the dense path alongside the sparse one. | **DEVIATION, justified and tracked** |
| **P3** no dormancy | Sparse wiring is measured by charge: `retained()` at the same width and load, with the mask proven to actually zero weights (`test_weights_never_escape_the_mask`). | PASS |
| **P4** one brain | No new composition. | PASS |
| **P5** GPU where measured | Every prior crossover assumed a fanout this feature changes. Re-measurement is deferred to the rung where it matters -- at 10^4 the synapse is 0.1 percent of the VRAM budget and the binding constraint is the neuron representation. | PASS, deferred with reason |
| **P6** mutual reach | Not a P6 feature. Thinning must not sever a cross-region path; the per-module metric is what detects it. | PASS |

**Standing Invariants:**

| Invariant | How this plan satisfies it |
|---|---|
| 1. Count a gate's output before a rule leans on it | F\* must be BRACKETED -- a failing F strictly below it. An F\* at the grid floor is reported "censored", never as a value. |
| 2. Never silence a check to make it pass | The tolerance was written to a file before the sweep and inherited unchanged by every adversarial re-run. |
| 3. The negative verdict is the dangerous one | A module that cannot be thinned is a DOCUMENTED EXCEPTION with its measurement (FR-003), and the feature reports "N of M sparse, K exceptions". |
| 4. Absence of warning is not absence of defect | Caching the BinLog passed its module self-check standalone and failed only in-process, where it silently defeated crash recovery. Module self-checks are not sufficient evidence. |
| 5. Verify the verifier | `retained()` is validated by two independent nulls, both reading exactly 0: a row-shuffle preserving nnz and per-row value multiset, and a learning-destroyed null at matched fanout. |

## Project Structure

```text
specs/009-sparse-connectivity/
├── plan.md, spec.md, research.md, data-model.md, quickstart.md
├── contracts/sparse-wiring.md
├── checklists/requirements.md
└── tasks.md

src/snn_data_lake/hippocampus/
└── episode_consolidation.py     # P1 target: _wire(), degree(), _zero_offmask()

tests/
└── test_sparse_connectivity.py  # 7 tests: function FIRST, savings second

tools/
└── fanout_bisect.py             # NEW: bracketed bisection that measures F* per deployment
```

**Structure Decision**: no new package. Wiring is a property of each module's own weight matrix.

## Implementation Sequence

**US1 -- one module works at reduced fanout** (DONE, measured)

1. `_wire()` -- symmetric uniform random k-out from `random.Random(seed)`, separate from the
   instance's numpy generator so wiring cannot consume draws replay depends on.
2. `_zero_offmask()` -- every weight update routes through one pruning point.
3. `degree()` -- reports REALISED degree, which symmetrisation raises above the requested F.
4. Tests: function first, savings second, determinism, symmetry, mask containment, dense unchanged.

**US2 -- the saving grows with population** (NOT DONE)

5. Build `tools/fanout_bisect.py`: bracketed bisection against the module's own metric, with the
   pre-registered tolerance and censoring rules encoded.
6. Verify at a width at least 4x beyond the tuned one (SC-008). Research verified 10^4; every rung
   above is EXTRAPOLATED.
7. **Measure whether this brain's stored load scales with its population.** The whole feature turns
   on it -- at fixed load the bound falls, at proportional load uniform-random grows as `N^0.68`.

**US3 -- the whole brain sparse, dense retired** (NOT STARTED)

8. Per-module task metrics for the 9 of 12 owner classes that currently have none.
9. Thin each, or record it as a documented exception with its measurement.
10. Retire the dense path once F is measured per site (closes the P2 deviation).

## Complexity Tracking

| Deviation | Why | Simpler alternative rejected because |
|---|---|---|
| **P2: `fanout=None` keeps the dense path** | P1 forbids defaulting F, and the knee tracks stored load rather than width, so no single number is correct across deployments. Until F is measured per site, dense must remain reachable or the module cannot run at all. | Picking a default F would satisfy P2 by violating P1 with a magic constant -- the exact defect this project calls its signature. The deviation closes at task 10, when F is measured per site and dense is deleted. **This is a real P2 debt, recorded rather than argued away.** |
| `SpikingGridCAN` excluded | Its connectivity is already `O(N)` in a real torus-phase metric, not `O(N^2)`. | Thinning it would be work with no ceiling benefit. |

## Post-design Constitution re-check

**PASS with one tracked P2 deviation.**

The risk this plan carries: **US1's result is verified at one module and two widths.** The scale
ladder marks 10^5 and above EXTRAPOLATED, and the binding constraint there is not the synapse at all
-- it is the neuron representation at 646.5 B/neuron of Python objects, which is specs/008's
vectorization work. So 009 removes the quadratic ceiling and 008 removes what binds next; neither
alone reaches the target population, and the plan should not imply otherwise.

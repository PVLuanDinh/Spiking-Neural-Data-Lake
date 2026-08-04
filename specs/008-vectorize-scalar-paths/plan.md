# Implementation Plan: Vectorize the brain's scalar numeric paths

**Branch**: `008-vectorize-scalar-paths` | **Date**: 2026-08-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/008-vectorize-scalar-paths/spec.md`

## Summary

The spec asked to convert scalar numeric paths to multidimensional arrays and sparse tensors, with
an unbounded population on the GPU tier. Phase 0 measured one fact that determines the whole
approach:

**A function drop-in is SLOWER at every width. The win is in the DATA FORMAT.**

```
width  scalar us   numpy PRE-array   numpy +convert   drop-in verdict
   16      0.909            0.494            1.663    0.55x  SLOWER
   64      2.821            0.496            4.051    0.70x  SLOWER
  512     20.058            0.533           26.763    0.75x  SLOWER
```

The per-call list-to-array conversion costs more than the Python loop it replaces. With the data
already resident as arrays the same computation is 1.84x faster at width 16 and 37.6x at 512. The
live width is 16 on 97.06 percent of 437,268 `dot` calls over 200 ticks.

So this plan is **not** a list of sites to convert. It is a **perimeter**: define where lists become
arrays, move that perimeter outward, and let call sites become array-native as they fall inside it.
Converting one function at a time is not a smaller version of this -- it is the opposite of it, and
it measures as a regression at every step.

Sequenced: P1 establishes the perimeter and the instrument on one subsystem; P2 widens it; P3 takes
the array-native representation to the GPU where scale requires it.

## Technical Context

**Language/Version**: Python >=3.11 (CI matrix 3.11 / 3.12 / 3.13)

**Primary Dependencies**: `numpy` and `torch`, both unconditional core dependencies. No new
dependency is proposed.

**Storage**: the one durable lake. **Phase 0 established the lake CANNOT back an unbounded
population** -- `StreamingHub` costs 273.4 B resident per channel. A new mmap-able fixed-width state
file is required, reusing the lake's shape (append-only, fixed-width, crash-safe, one file) so the
one-durable-store invariant is preserved by construction rather than by reinterpreting a knowledge
log as a state store.

**Testing**: `pytest` (>=70% coverage, currently 92.85%), `run_selfchecks.py` (278 of 278),
`ruff check`. **Plus `tools/deep_fingerprint.py` for every null** -- `Brain.state()` is a measured
false-null instrument (0 of 149 through `state()` where a deep walk finds 1,713 of 98,320).

**Target Platform**: Windows 11 + Linux CI. `.venv-lake` is the authoritative deterministic CPU
gate; `.venv-cu128` is the GPU tier (RTX 5070, 12.8 GB). **CI has no GPU** and the suite must remain
runnable without one.

**Performance Goals**: the live tick, measured at 74.26 ms. Neuter-and-measure puts `dot` at
**16.02 percent of tick time -- an Amdahl ceiling of 1.191x**, against cProfile's 7.6 percent.
Profiler attribution is not time, and no site may be claimed on profile rank alone.

**Constraints**:
- Determinism is a KERNEL CHOICE: 13 of 15 CUDA ops are bit-stable over 100 repeats; only
  atomics-based accumulate is not. **Avoid atomics on the live path.**
- CPU and GPU do not agree with each other and the gap grows with N (rel 1.504e-06 at 1M,
  1.464e-04 at 16M). Each tier is internally reproducible; neither proves the other.
- E-prop / forward-mode credit assignment on any live learning path; never `loss.backward()`.
- ASCII-only `print()`; counts written `N of M`.
- The safety envelope does not widen.

**Scale/Scope**: live populations 16-240 today; the spec's target is unbounded. **The connectivity
law is UNMEASURED and every ceiling divides by it** -- constant fanout gives one set of numbers, a
fraction-of-width gives quadratic memory and a completely different set.

## Constitution Check

*GATE: must pass before Phase 0. Re-checked after Phase 1 design.*

| Principle | Assessment | Status |
|---|---|---|
| **P0** self-improvement | Vectorization is infrastructure, not an improvement avenue in itself. It is scaffolding under P0's third clause and is labelled as such: it claims no faculty band and no new capability. | PASS |
| **P1** no hardcode | Introduces no behaviour-arbitrating constant. `nnz < 2^31` is a declared resource assertion tied to int32 indices, named rather than hidden. | PASS |
| **P2** non-additive | Every converted site REPLACES its scalar implementation; a flag keeping both is forbidden. The perimeter makes this structural -- inside it there is one representation, not two behind a switch. | PASS |
| **P3** no dormancy | Gate Zero exists to prevent converting a site nothing calls. A vectorized function never invoked is worse than the loop it replaced. | PASS |
| **P4** one brain | All measurement via `cortex.brain.make_brain()`. No new composition. | PASS |
| **P5** GPU where measured | Re-measured at target scale and recorded in POLICIES.md: CUDA is 3.0x at 1M and 12.2x at 4M, versus 0.3x at 240. The live tick at its DESIGN width is a GPU target; at today's width it is not, and this plan does not pretend otherwise. | PASS |
| **P6** mutual reach | Not a P6 feature, but it must not make the interconnect worse: array-native boundaries must carry structure, not collapse it into new scalars. | PASS |

**Standing Invariants:**

| Invariant | How this plan satisfies it |
|---|---|
| 1. Count a gate's output before a rule leans on it | Gate Zero measures the Amdahl ceiling BEFORE any site is converted. A site with a 1.19x ceiling cannot deliver more than 1.19x, whatever its microbenchmark says. |
| 2. Never silence a check to make it pass | Where exact equality fails, FR-002's measured-difference protocol applies. Loosening an assertion to accommodate a conversion is forbidden. |
| 3. The negative verdict is the dangerous one | A DECLINE must carry its measurement. "Not converted" without a number is not a decline, it is an omission. |
| 4. Absence of warning is not absence of defect | `Brain.state()` reported a clean null where 1,713 leaves had moved. Every null in this feature uses the deep fingerprint. |
| 5. Verify the verifier | Every A/B ships a matched A/A control. Phase 0's swap harness silently failed to install on its first run -- `_SWAPPED: False`, 0 calls -- and would have measured the unmodified code as a pass. |

**Violations requiring justification**: none.

## Project Structure

### Documentation (this feature)

```text
specs/008-vectorize-scalar-paths/
├── plan.md              # This file
├── spec.md              # + 2 amendments (GPU required; unbounded population)
├── research.md          # Phase 0 -- 4 decisions, all 4 probes refuted and corrected
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   └── conversion.md
└── tasks.md             # Phase 2 (/speckit-tasks -- NOT created here)
```

### Source Code (repository root)

```text
src/snn_data_lake/
├── substrate/
│   ├── _mathutils.py            # dot at 16.02% of tick; live width 16 on 97.06% of calls
│   └── streaming_hub.py         # _scan per-record struct.unpack_from -- ships independent of tiering
├── cerebellum/
│   └── jepa_world_model.py      # :189 carries 96.7% of dot calls as a 16x16 matvec of scalar dots
├── cortex/
│   ├── crossmodal_binder.py     # :117 -- ranked FIRST by cProfile, NOT first by neuter-and-measure
│   └── lif_neuron.py            # :31 step -- elementwise, exact class, the GPU candidate
└── reasoning/
    └── vsa_reasoning.py         # :94 -- INTEGER reduction, exact by construction, no A/B needed

tools/
├── deep_fingerprint.py          # EXISTS -- the instrument every null must use
└── convert_gate.py              # new: Gate Zero + the T0-T3 ladder as a runnable check
```

**Structure Decision**: no new package. The perimeter is a property of existing modules, not a new
layer. A new `arrays/` package would itself be the additive-with-extra-steps pattern P2 forbids.

## Implementation Sequence

**P1 -- perimeter and instrument, on one subsystem**

1. Build `tools/convert_gate.py`: Gate Zero (neuter-and-measure the Amdahl ceiling) plus the T0-T3
   ladder. Without it, no conversion has evidence.
2. Convert `jepa_world_model.py:189` -- the 16x16 matvec written as 16 scalar dots, carrying 96.7
   percent of all `dot` calls. Measured 1.086x end-to-end with 0 of 149 state fields changed;
   re-verify against the deep fingerprint, not `state()`.
3. Establish the perimeter around that subsystem: arrays in, arrays out, conversion at the boundary
   only.
4. Vectorize `streaming_hub.py::_scan` -- independent of everything else and backed by real data.

**P2 -- widen the perimeter**

5. Classify every remaining hot site with T0: integer-reduction and elementwise sites need no A/B;
   transcendental and float-reduction sites take the full ladder.
6. Convert or DECLINE each, with the measurement attached either way.
7. **Measure the connectivity law** -- count nonzero synapses per neuron at three or more population
   widths and fit the functional form. Every ceiling in the spec's amendments depends on it.

**P3 -- GPU tier, where scale requires it**

8. CSR int32 synaptic matrix, dense f32 spike vector, post-major, one cuSPARSE SpMV. Chunked
   construction at 2e7-2e8 nnz, justified by construction peak alone.
9. Avoid atomics-based accumulation so the GPU path stays bit-stable.
10. Golden-output reproducibility gate for the GPU path, separate from the CPU gate. Neither proves
    the other and no gate may be written as if it does.

## Complexity Tracking

| Deferral | Why | Simpler alternative rejected because |
|---|---|---|
| Tiering design not committed | Every ceiling divides by an unmeasured connectivity law. Constant fanout and fraction-of-width differ by orders of magnitude. | "Design for constant fanout 100 and adjust later" bakes an unverified assumption into the storage layout -- the most expensive place to be wrong. |
| State/synapse split kept as a direction, not a number | Its 132x ceiling headline was refuted as an artifact by its own refuter. | Quoting 132x would put a refuted number into a design document. |

## Post-design Constitution re-check

**PASS, unchanged.** No new mechanism, dependency or composition.

The risk this plan carries: the perimeter approach is all-or-nothing per subsystem in a way that
site-by-site conversion is not. A half-converted subsystem pays conversion cost at every internal
boundary and can measure SLOWER than either endpoint. The mitigation is that P1 establishes the
perimeter on ONE subsystem and measures it end to end before the pattern spreads -- and if that
subsystem cannot be made faster end to end, the approach is wrong and the feature should stop rather
than widen.

# Data Model: sparse connectivity

**Feature**: specs/009-sparse-connectivity | **Phase**: 1 | **Date**: 2026-08-04

The entities are the wiring, the bound that sets it, and the evidence that the module still works.
The evidence entity is first-class here rather than an afterthought, because every other quantity in
this feature improves monotonically as connections are deleted.

---

## WiringMask

Which units connect to which. Replaces the implicit all-to-all assumption.

| Field | Meaning | Validation |
|---|---|---|
| `fanout` | Requested connections per unit | `None` means dense (the historical path). An integer is clamped to `[1, dim-1]`. **Never defaulted to a number** -- see FanoutBound |
| `symmetric` | Whether `mask == mask.T` | MUST be true. The covariance rule is symmetric, so a one-way connection is not a thinned version of the dense dynamics -- it is a different system |
| `diagonal` | Self-connections | MUST be all-false. A unit driving itself is not connectivity |
| `realised_degree` | What was actually built | Symmetrising raises degree above `fanout` (a unit keeps its own picks plus whoever picked it). `degree()` reports the MEASURED value, not the requested one -- the two differ and only one is honest |
| `seed_source` | Which RNG drew it | MUST be `random.Random(seed)`, NOT the instance's numpy generator. Sharing a stream would let wiring consume draws that replay noise depends on, coupling two things that must be independently reproducible |

**State transitions**: wiring is built once at construction and never changes. A mask that could
change at runtime would make `retained()` incomparable between two points in the same run.

**The invariant that makes it real**: every weight update routes through one pruning point. A new
update site that forgets to prune would silently restore all-to-all connectivity while the mask sat
there looking correct -- so the mask is applied centrally, not per-site.

---

## FanoutBound

The number of connections per unit, and how it is arrived at.

| Field | Validation |
|---|---|
| `value` | **MUST be measured per deployment, never defaulted.** Measured F\*: 176 at N=256, 160 at 1,024 and 4,096, 128 at 16,384 -- at FIXED load |
| `derivation` | A PROCEDURE, not a formula: a graph-connectivity FLOOR (necessary, never sufficient), a hardware-budget CEILING (self-extinguishing), and a bracketed bisection against measured function that sets the operating point |
| `bracketed` | **MUST be true.** A failing F strictly below F\* on the grid. An F\* at the grid floor is reported "censored at F >= floor", never as a value |
| `driver` | **Stored LOAD, not width and not density.** Both alternatives were asserted and both falsified: 8.3x density spread at fixed N=192; fanout doubling 512 to 1,024 at constant density 0.2501 |

**Why no closed form**: every formula attempted in research relocated the constant it was meant to
remove. A quantile needs `p`; a spectral-gap threshold needs the threshold. The procedure terminates
on measured function instead, which is the only self-extinguishing criterion available.

---

## TaskMetric

What a module is FOR, measured its own way.

| Field | Validation |
|---|---|
| `can_degrade` | **MUST be able to get WORSE.** Any metric that improves monotonically with parameter count is arithmetic, not function. `mean_assembly_coupling()` is REJECTED for exactly this: it sums surviving weights, so thinning lowers it whether or not recall degrades |
| `validated_by_null` | MUST read ~0 under a control that destroys the learning while preserving the structure. `retained()` passes two: a row-shuffle preserving nnz and per-row value multiset reads 0.0000 against dense 1.0000 at N = 1,024 / 2,048 / 4,096; a learning-destroyed null reads 0.0000 at F = 160 / 320 / 640 and at dense |
| `not_an_oracle` | The metric MUST NOT leak the answer. Assembly-derived wiring on `AttractorNet` is an ORACLE -- with two disjoint assemblies the co-activity mask IS the ground-truth partition, so an UNTRAINED masked net scores 1.000 |

**Coverage today**: 3 of 12 owner classes have a metric demonstrated to degrade. 9 of 12 need one
built, and all nine are UNVERIFIED -- derived by reading source, never executed.

---

## ThinningEvidence

What a claim of "still works" must carry.

| Field | Validation |
|---|---|
| `tolerance` | **Fixed BEFORE the sweep.** Mean within 0.02 AND minimum over seeds within 0.05 of dense at the same width, load and seed set |
| `dense_baseline` | MUST be measured at the SAME width and load, and MUST be non-trivial. A baseline that is already failing makes "95 percent of dense" a weak claim -- this was measured against a collapsing baseline once and the result was meaningless |
| `seeds` | At least 5 |
| `cliff_vs_knee` | A drop above 0.20 between adjacent grid points is a CLIFF, reported as such |
| `width_verified` | The width function was actually measured at. Everything above it is EXTRAPOLATED and **must be labelled** -- this project has treated a toy-scale measurement as a target-scale fact three times |

---

## ModuleVerdict

Per connectivity-bearing module.

| Verdict | Meaning |
|---|---|
| `SPARSE` | Thinned, with F\* bracketed and function within tolerance |
| `DOCUMENTED_EXCEPTION` | Genuinely needs dense connectivity, WITH the measurement showing it. A legitimate outcome, not a failure |
| `OUT_OF_SCOPE` | Connectivity is already `O(N)` -- e.g. `SpikingGridCAN` has a real torus-phase metric |
| `UNMEASURED` | No task metric yet. **The default, and it must not be reported as anything else** |

**Reporting rule**: always "N of M sparse, K documented exceptions, U unmeasured", with all four
numbers stated. Today: 1 sparse, 0 exceptions, 1 out of scope, 10 unmeasured, of 12.

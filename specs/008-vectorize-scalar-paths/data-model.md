# Data Model: vectorize the brain's scalar numeric paths

**Feature**: specs/008-vectorize-scalar-paths | **Phase**: 1 | **Date**: 2026-08-04

The entities here are representations and the boundaries between them. The plan's central finding --
a function drop-in is slower at every width because conversion costs more than the loop it replaces
-- makes the BOUNDARY the primary entity, not the array.

---

## Perimeter

The region of the call graph inside which data is array-native. Not a module, not a package: a
property of a connected set of call sites.

| Field | Meaning | Validation |
|---|---|---|
| `subsystem` | The connected set of modules inside it | MUST be closed under calls: if A inside calls B outside on a hot path, B is on the perimeter and pays conversion |
| `entry_conversions` | Where lists become arrays | Each one costs; the count is the thing being minimised |
| `internal_conversions` | Conversions strictly inside | **MUST be 0.** A conversion inside the perimeter means the perimeter is drawn wrong |
| `measured_end_to_end` | Tick cost with the perimeter in place vs without | MUST be faster, or the perimeter is not justified |

**The failure mode this entity exists to prevent**: a half-converted subsystem pays conversion at
every internal boundary and measures SLOWER than either endpoint. Measured: `+convert` is 0.55x to
0.75x at every width tested. A perimeter that is not closed is not a partial win, it is a loss.

---

## ConversionSite

A numeric site proposed for conversion.

| Field | Meaning | Validation |
|---|---|---|
| `amdahl_ceiling` | Fraction of tick time, from neuter-and-measure | MUST come from neutering, NOT from a profiler. cProfile put `dot` at 7.6 percent and ranked `crossmodal_binder` first; neuter-and-measure puts `dot` at 16.02 percent, a 1.191x ceiling |
| `live_width_histogram` | The widths actually seen in a live run | MUST be measured, not assumed. `dot`'s live width is 16 on 97.06 percent of 437,268 calls -- the wide case does not exist |
| `exactness_class` | integer-reduction / elementwise-unfused / transcendental / float-reduction | Determines which tiers of the ladder apply |
| `verdict` | CONVERTED or DECLINED | **A DECLINE MUST carry its measurement.** "Not converted" without a number is an omission, not a decision |

**State transitions**: `proposed -> gate-zero -> classified -> (converted | DECLINED)`. A site may
be DECLINED at any stage and that is a successful outcome. A site may NOT skip gate-zero: a
conversion with no ceiling measurement cannot know what it is buying.

---

## ExactnessClass

What a site's numerics permit, which decides what must be proven.

| Class | Guarantee | Evidence required |
|---|---|---|
| **integer reduction** | Exact by construction | None. `vsa_reasoning.py:94` measured 20,000 of 20,000 identical at D=512, still exact at 4,096 and 65,536 |
| **elementwise, unfused, pinned dtype** | Bit-identical, CPU and CUDA | T0 only. `{+,-,*}` at 16,000,000 of 16,000,000 f64 and 64,000,000 of 64,000,000 f32 |
| **transcendental** | NOT exact | Full ladder. `sigmoid` is 87.42 percent bit-identical in f32 at N=16M; `exp` 69.80 percent |
| **float reduction** | NOT exact above width ~16 | Full ladder plus T3 margin bound |
| **anything fused** | Exactness VOID | Re-prove against the fused kernel that ships. Eager-vs-fused is 68.17 percent bit-identical -- but 16,000,000 of 16,000,000 spike decisions identical |

**The trap this table encodes**: "elementwise means exact" is true only unfused. Fusion is exactly
what target scale forces, so the guarantee evaporates when it is most needed. Spike decisions
surviving fusion intact is what makes FR-002 workable anyway.

---

## SparseStructure

The synaptic connectivity, at the scale where dense does not fit.

| Field | Validation |
|---|---|
| `format` | **CSR with int32 columns.** Measured at N=1M, F=100, nnz=100M with identical checksums: CSR32 804 MB / 4.17 ms, CSR64 1,208 MB / 4.97 ms, COO 2,000 MB / 12.05 ms plus a 4.0x coalesce blowup |
| `nnz` | MUST assert `< 2^31`. int32 columns are the reason CSR32 wins; exceeding it silently corrupts |
| `majorness` | **post-major**, paired with a dense f32 spike vector and one cuSPARSE SpMV. Pre-major CSR fed a dense spike vector silently computes a DIFFERENT QUANTITY |
| `construction_chunk` | 2e7 to 2e8 nnz. Justified by construction PEAK alone -- chunking measured 1.04x to 5.33x SLOWER at 100M nnz |
| `connectivity_law` | **UNMEASURED.** Constant fanout or fraction-of-width? Every ceiling divides by it, and fraction-of-width makes memory quadratic |

**Blocking**: no storage layout may be committed until `connectivity_law` is measured. It is the one
unknown that changes answers by orders of magnitude rather than percentages.

---

## SpikeVector

Which neurons fired this tick.

| Field | Validation |
|---|---|
| `representation` | **Dense float32**, not an int32 index list. This REVERSES the original probe finding and cannot be mixed with pre-major CSR |
| `density` | **MEASURED at 10.49 percent** on this brain (2,379,940 LIF steps, 249,560 spikes) -- 5x the biological 2 percent the first tiering argument assumed |

**Why dense wins here**: at 10.49 percent density the index list saves little and costs gather/scatter
overhead, and scatter is the one CUDA op measured non-deterministic (atomics). Dense float32 keeps
the kernel bit-stable, which the determinism decision requires.

---

## EquivalenceEvidence

What a conversion must produce to be accepted.

| Field | Validation |
|---|---|
| `instrument` | **`tools/deep_fingerprint.py`, NOT `Brain.state()`.** Measured: `state()` reports 0 of 149 where the deep walk finds 1,713 of 98,320 |
| `aa_control` | **MUST read 0.** Nonzero means run nondeterminism and the A/B through it is unreadable |
| `swap_charged` | **MUST be proven** -- call count > 0. Phase 0's harness silently failed to install (`_SWAPPED: False`, 0 calls) and would have measured unmodified code as a pass |
| `integer_leaves_moved` | MUST be 0 for a conversion claiming equivalence. Integers moving means discrete state changed, not floats drifting |
| `divergence_growth` | MUST NOT be super-linear in tick count. Measured linear: 1,713 at 300 ticks, 1,947 at 1500 |

**The rule**: a null measured through `Brain.state()` is not a null. Positive findings through it
remain valid -- a difference seen through a coarse instrument is still a difference. Only the
ABSENCES were never evidence.

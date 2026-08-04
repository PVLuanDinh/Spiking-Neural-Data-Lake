# Feature Specification: Vectorize the brain's scalar numeric paths

**Feature Branch**: `008-vectorize-scalar-paths`

**Created**: 2026-08-03

**Status**: Draft

**Input**: User description: "Change all scalar data format in the brain project to multidimensional
arrays and sparse tensors where applicable... Highly optimized for GPU/CPU vectorization, fast matrix
operations, and seamless integration with deep learning libraries. Remember to fix the cons: Can
waste memory on zero-values (non-spiking neurons) unless stored as sparse arrays (like COO or
Compressed Sparse Row formats)."

---

## Why this is worth doing, in one measurement

The live brain tick costs **74.26 ms**, and a profile over 100 ticks shows where it goes:

```
495,761 calls   builtins.sum                          1.219 s tottime
4,966,400 calls cortex/crossmodal_binder.py:117       0.833 s
3,942,048 calls substrate/_mathutils.py:61            0.578 s
  916,902 calls cortex/vsa_reasoning.py:94            0.201 s
  579,616 calls cortex/lif_neuron.py:31               0.191 s
```

Roughly **50,000 generator-expression calls per tick**. The hottest site is a matrix-vector product
written as a Python double loop. Replacing it with an array operation is byte-identical and:

```
D=64  N=32     72.5 us ->  2.6 us     28.4x
D=128 N=64    260.7 us ->  3.3 us     79.2x
D=256 N=128   964.1 us ->  4.0 us    238.7x
D=512 N=256  3658.0 us ->  9.1 us    402.7x
```

This is not a speculative optimization. The work is already being done; it is being done one scalar
at a time.

## What this feature deliberately does NOT assume

**Sparse is not free, and this codebase's spike codes are mostly dense.** The same crossmodal spike
code above is `+1/-1` bipolar at **48 percent positive**. COO or CSR there would cost MORE memory
than a dense array, plus indexing overhead. The owner's stated concern about wasting memory on
non-spiking zeros is real and site-specific -- so occupancy is MEASURED per site and the
representation follows the measurement, rather than sparse being applied by default.

**The GPU is not the win here.** The measured CUDA crossover for this project is N about 24,576 to
32,768; live populations are 24 to 240 units wide, and forcing the live tick onto the GPU previously
measured 0.01x at width 38. The win in this feature is CPU vectorization. Any site claimed for GPU
needs its own measurement at that site's real width.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The hot path gets faster without behaving differently (Priority: P1)

The two hottest numeric sites are converted to array operations. The brain produces the same outputs
it did before, and the tick gets measurably cheaper.

**Why this priority**: It carries most of the available win, it is the smallest slice that proves
the conversion method is safe, and it establishes the equivalence harness every later conversion
depends on. If byte-identity cannot be preserved at these two sites, the whole approach needs
rethinking before it spreads.

**Independent Test**: Run the brain before and after on the same seed; outputs identical, tick time
lower. Fully testable without converting anything else.

**Acceptance Scenarios**:

1. **Given** the same seed and tick count, **When** the brain runs before and after conversion,
   **Then** every observable output is identical, and any difference that does appear is shown not
   to move any gate or seeded outcome.
2. **Given** the converted site at its live width, **When** its cost is measured, **Then** it is
   faster than the scalar version it replaced, stated as a ratio.
3. **Given** the conversion is complete, **When** the scalar implementation is searched for,
   **Then** it is gone -- exactly one implementation of each converted site survives.

---

### User Story 2 - Every remaining hot site is converted or explicitly declined (Priority: P2)

The rest of the profile's hot sites are each either converted, or recorded as measured-not-worth-it
with the number that says so.

**Why this priority**: It completes the owner's "all scalar data formats" instruction honestly. A
site that is slower or behaviour-changing when vectorized must be declined loudly rather than
skipped silently, or the feature reports coverage it does not have.

**Independent Test**: The profile is re-run; every remaining scalar hot site appears in a ledger with
either a conversion and its speedup, or a decline and its reason.

**Acceptance Scenarios**:

1. **Given** the post-conversion profile, **When** the hot sites are listed, **Then** each is marked
   converted-with-speedup or declined-with-measurement, and none is unaccounted for.
2. **Given** a site where the array version is slower at the live width, **When** it is evaluated,
   **Then** it is DECLINED and the measurement is recorded -- not converted for consistency.

---

### User Story 3 - Spike data over time uses a shape that can be processed in bulk (Priority: P3)

Anything that processes spikes across time steps uses a batch/time/channel-shaped layout rather than
per-step scalar loops, with dense or sparse storage chosen per site from measured occupancy.

**Why this priority**: This is the structural half of the owner's directive and the part that opens
the door to bulk processing and external spiking libraries. It is third because it is only safe once
the equivalence harness (P1) exists and the site inventory (P2) is complete.

**Independent Test**: A spike-over-time path is exercised; its representation is the declared layout,
its occupancy is recorded, and its storage choice matches what the occupancy implies.

**Acceptance Scenarios**:

1. **Given** a spike path processing multiple time steps, **When** its representation is inspected,
   **Then** it carries a batch/time/channel shape rather than a per-step scalar loop.
2. **Given** any site proposed for sparse storage, **When** its occupancy is measured, **Then**
   sparse is used only if it measurably reduces memory or time at that site's live width, and dense
   is used otherwise with the occupancy recorded.

---

### Edge Cases

- **Floating-point associativity changes.** A left-fold Python `sum` and a pairwise array sum are
  not the same operation. Byte-identity CANNOT be assumed and must be measured per site. Where it
  differs, the difference must be shown not to move any gate or seeded outcome, or the site is not
  converted.
- **A site is slower when vectorized.** At small widths, array-call overhead can exceed the loop it
  replaces. This is expected at some sites and is a decline, not a failure.
- **Empty and single-element inputs.** Array paths must handle zero-length and length-one inputs
  that scalar loops handled implicitly.
- **Ragged data.** Percepts in this brain are not all the same width; a shape-based representation
  must handle ragged input or explicitly refuse it.
- **Sparse chosen where data is dense.** The failure this feature must not commit: applying COO/CSR
  to a 48-percent-occupied code and reporting it as an optimization.
- **Conversion silently disabling a live path.** A vectorized function that is never called is worse
  than the loop it replaced.
- **Determinism across platforms.** Array libraries may use different summation orders on different
  builds. Gates must stay reproducible on the environments CI actually runs.

---

## Requirements *(mandatory)*

### Functional Requirements

**Equivalence -- the safety property**

- **FR-001**: Every converted site MUST produce output equivalent to the scalar implementation it
  replaces, verified on the same seed against the pre-conversion output.
- **FR-002**: Where exact equality does not hold because of summation order, the system MUST measure
  the difference and demonstrate it moves no gate and no seeded outcome. A site whose difference
  cannot be shown harmless MUST NOT be converted.
- **FR-003**: The determinism property MUST be preserved: identical seed produces identical
  structure and identical gate results.

**Performance -- the reason for doing it**

- **FR-004**: Every converted site MUST be measurably faster at that site's LIVE width, not only at
  large synthetic widths. The measurement MUST be recorded.
- **FR-005**: A site that is slower when vectorized MUST be DECLINED, and the measurement recorded.
  Consistency of style is not a reason to convert.
- **FR-006**: The end-to-end tick cost MUST be measured before and after and reported as a ratio.

**Representation -- the owner's directive**

- **FR-007**: Numeric paths that operate over collections MUST use array operations rather than
  per-element Python loops, wherever FR-004 is satisfied.
- **FR-008**: Paths processing spikes across time steps MUST use a batch/time/channel-shaped layout.
- **FR-009**: Sparse storage MUST be used only where measured occupancy shows it reduces memory or
  time at that site's live width. Occupancy MUST be recorded for every site considered.
- **FR-010**: Every hot site in the profile MUST be accounted for as converted-with-speedup or
  declined-with-measurement. Silent omission is not permitted.

**Governance**

- **FR-011**: Exactly ONE implementation of each converted site MUST survive. A switch selecting
  between scalar and array paths is forbidden.
- **FR-012**: The conversion MUST NOT introduce behaviour-arbitrating constants.
- **FR-013**: A converted path MUST remain on a live path and be exercised; it MUST NOT become a
  parallel unused fast path.
- **FR-014**: The suite MUST remain runnable without a GPU, and no site moves to a GPU without a
  measurement at that site's real width.

### Key Entities

- **Conversion site**: A numeric path being considered. Has a live width, a measured scalar cost, a
  measured array cost, an equivalence result, and a disposition of converted or declined.
- **Occupancy record**: For any site considered for sparse storage, the measured fraction of
  non-zero entries at the live width, and the resulting dense-or-sparse decision.
- **Equivalence record**: Whether output is exactly equal, or differs only in a way shown to move no
  gate and no seeded outcome, with the measurement attached.
- **Site ledger**: The complete accounting of hot sites, each with its disposition. This is the
  feature's coverage evidence.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Tick cost falls measurably from its 74.26 ms baseline, reported as a before/after
  ratio on the same machine and seed.
- **SC-002**: Every converted site produces equivalent output, verified on the same seed: at least
  as many sites verified as converted, with 0 sites converted without a verification.
- **SC-003**: Generator-expression call count per tick falls from its measured baseline of about
  50,000, reported as "N of M sites converted".
- **SC-004**: Determinism holds: identical seed produces identical structure in 20 of 20 trials.
- **SC-005**: All three gates stay green -- self-checks, tests at or above the coverage floor, and
  lint -- before and after.
- **SC-006**: Every hot site from the baseline profile is accounted for: converted with a speedup
  ratio, or declined with the measurement that justified declining. 0 sites unaccounted.
- **SC-007**: Occupancy is recorded for 100 percent of sites where sparse storage was considered,
  and no site uses sparse storage without an occupancy measurement supporting it.
- **SC-008**: Exactly 1 implementation survives per converted site; 0 flag-selected dual paths.

---

## Assumptions

- "All scalar data formats" is read as **numeric paths that operate over collections**, not literally
  every scalar variable. A loop counter, a threshold, or a single accumulator is not a vectorization
  target, and converting one would add indirection for no measured gain.
- "Where applicable" is decided by FR-004 and FR-009 -- measurement, not judgement. This is the
  clause that keeps the feature from becoming a style migration.
- The array library is the one already in the project's unconditional core dependencies; this
  feature introduces no new dependency.
- The target is CPU vectorization. The training and perception tiers already use the GPU where it
  was measured to pay, and this feature does not change that boundary.
- Sites already using array operations are out of scope; this feature converts scalar loops.
- Equivalence is measured against the CURRENT behaviour, correct or not. This feature changes
  representation, never semantics -- a bug found during conversion is reported, not fixed in the
  same change, so equivalence stays checkable.

## Out of Scope

- Moving the live tick to the GPU. Measured 0.01x at width 38; revisit only if populations approach
  the measured crossover.
- Adopting an external spiking-network library as a runtime dependency of the live path. The
  batch/time/channel layout in US3 makes that possible later; adopting it is a separate decision
  with its own measurement.
- Fixing behaviour defects discovered during conversion. They are recorded and handled separately,
  so that equivalence remains the acceptance test.

---

## AMENDMENT 2026-08-03: GPU is REQUIRED, sparse is MANDATORY, and the target is 1M+ neurons

**Owner correction**: the live tick is meant to grow past N=24,576 and far beyond; the neuron
population should already be at least 1 million. The original spec measured the CURRENT 24-240
widths and concluded CPU-only with GPU out of scope. **That inverted the causality: 24-240 is the
defect this project is trying to fix, not the constraint the design must respect.**

This is also P5 operating exactly as written. P5 says the crossover is decided by measurement and
**"Re-measure when populations grow."** This is that re-measurement.

### Measured, toy scale through target scale

Elementwise LIF tick, ms per tick, RTX 5070 (12.8 GB), torch 2.11.0+cu128:

```
         N   numpy CPU   torch CPU   torch CUDA   CUDA speedup
       240       0.003       0.013        0.037         0.3x   <- the regime the original spec measured
    24,576       0.017       0.055        0.033         1.7x   <- the documented crossover, confirmed
   100,000       0.056       0.154        0.082         1.9x
 1,000,000       4.761       0.265        0.088         3.0x
 4,000,000      18.639       2.310        0.189        12.2x
16,000,000      77.674      15.711        1.552        10.1x
```

Against the numpy path this feature was originally scoped to build, CUDA at 1M is **54x faster**
(4.761 ms vs 0.088 ms), and at 16M it is **50x** (77.674 ms vs 1.552 ms).

### Sparse is not an optimization at this scale -- it is the only representation that fits

Synaptic connectivity, fanout 100:

```
         N   dense weight matrix   sparse COO   sparse matvec
   100,000              40 GB        0.120 GB       13.1 ms
 1,000,000            4000 GB        1.200 GB       11.3 ms
```

A dense 1M x 1M weight matrix is **4000 GB** against 12.8 GB of VRAM -- impossible by a factor of
about 300. Sparse is mandatory, not preferable.

Spike vectors at biological firing rates:

```
  rate 1.0%   active   10,000   dense 4.00 MB   COO 0.12 MB   33.3x smaller
  rate 2.0%   active   20,000   dense 4.00 MB   COO 0.24 MB   16.7x smaller
  rate 5.0%   active   50,000   dense 4.00 MB   COO 0.60 MB    6.7x smaller
```

**The earlier "sparse would cost MORE memory here" finding is superseded.** It was measured on a
D=64 bipolar code at 48 percent occupancy -- a toy width. At 1M neurons with 1-2 percent firing,
COO is 17x to 33x smaller. Both measurements are correct at their own scale; only one of them is
about the system being built.

### What changes in this specification

- **FR-014 is REPLACED.** It said no site moves to GPU without a measurement at its real width, and
  that the suite must run without a GPU. The second half stands -- CI has no GPU and the
  deterministic gate must remain CPU-runnable. The first half is now SATISFIED for the live tick by
  the table above: **the live tick is a GPU target**, because its design width is 1M+, not 240.
- **FR-009 is STRENGTHENED.** Sparse storage was permitted only where occupancy justified it. At the
  synaptic matrix it is now REQUIRED, because dense does not fit in any available memory. Occupancy
  measurement still governs the spike vectors and every other site.
- **Out of Scope is CORRECTED.** "Moving the live tick to the GPU" was listed out of scope on the
  strength of a 0.01x measurement at width 38. That measurement is real and irrelevant: it describes
  the toy. Moving the live tick to the GPU is now IN scope and is the point.
- **A scale target is added.** The population is a first-class requirement, not an emergent property:
  the live tick must reach **at least 1,000,000 neurons**, and the representation must not be the
  thing that stops it.

### New success criteria

- **SC-009**: The live tick runs at a population of at least 1,000,000 neurons.
- **SC-010**: At 1,000,000 neurons the tick uses the GPU tier and is faster than the CPU path,
  reported as a ratio measured on this machine.
- **SC-011**: Synaptic connectivity at 1,000,000 neurons fits in available VRAM, which requires a
  sparse representation; the memory figure is recorded.
- **SC-012**: The deterministic CPU gate still runs the full suite without a GPU. Scale is the
  live-path target; reproducibility on CI hardware is not negotiable and is what keeps the gates
  meaningful.

### The honest note on determinism at scale

GPU reductions are not order-deterministic in general, and this project's gates depend on identical
seeds producing identical structure. That tension is real and unresolved here: the deterministic
CPU gate remains authoritative (SC-012), and any claim that the GPU path is reproducible must be
measured, not assumed. If the two cannot be reconciled, the correct outcome is a CPU gate at small
width plus a separately-validated GPU path at scale -- not a weakened determinism requirement.

---

## AMENDMENT 2026-08-03 (second): 1,000,000 is NOT a cap. The population is unbounded.

**Owner correction**: "the 1 million neuron population is not the cap, an unbounded and ever
expanding population is the goal." The previous amendment set 1M as a target, which is still a
fixed number and therefore still the wrong shape of requirement. Unbounded is a STRUCTURAL
constraint, not a bigger target: it forbids any fixed allocation anywhere on the growth path, and it
means VRAM cannot be the store.

### Measured: where the walls actually are

**VRAM caps the population well below "unbounded", and the cap depends on connectivity** (sparse
COO, 12 B per synapse):

```
 fanout   bytes/neuron   max N in 12.8 GB
     10            120        106,666,666
    100          1,200         10,666,666
   1000         12,000          1,066,666      <- biological fanout: ~1M, the number quoted
  10000        120,000            106,666      <- cortical pyramidal: ~107k
```

The 1M figure is roughly what VRAM allows at biological fanout. It is a hardware artifact, not a
design goal -- which is exactly why it must not be the requirement.

**Growth must be chunked, never reallocate-and-copy.** Warmed, 20 reps:

```
           N   realloc+copy   chunk append    ratio
     100,000       0.0208 ms      0.0089 ms     2.3x
   1,000,000       0.0195 ms      0.0488 ms     0.4x    (noise)
   4,000,000       0.0608 ms      0.0346 ms     1.8x
  16,000,000       0.5647 ms      0.0538 ms    10.5x
```

Realloc-and-copy is O(N) per growth event, so it must lose at scale, and by 16M it does by 10.5x.
The small-N ratios are noise. (An earlier unwarmed run reported 247x at 100k and 1.9x at 1M --
non-monotonic, the signature of allocator warmup rather than of the mechanism. Discarded.)

**Host and disk are fast enough to be the real store**: host-to-GPU 16.3-18.0 GB/s, GPU-to-host
20.1-20.3 GB/s over pinned memory.

### The architecture unbounded actually implies

**Bound the ACTIVE set, not the population.** At biological firing rates only a small fraction of
neurons spike per tick, and only those need to be resident:

```
        total N     2% firing     active-set VRAM
      1,000,000        20,000             0.02 GB
    100,000,000     2,000,000             2.40 GB     <- comfortably resident
  1,000,000,000    20,000,000            24.00 GB     <- exceeds 12.8 GB
  6,000,000,000   120,000,000           144.00 GB
```

So a 12.8 GB card supports a total population into the hundreds of millions before even the firing
subset overflows. Tiering the whole population onto storage the project already has:

```
  GPU-resident only              :      10,666,666 neurons
  + 64 GB host                   :      64,000,000
  + 64 GB host + 2 TB disk       :   1,730,666,666
  + 64 GB host + 8 TB disk       :   6,730,666,666
```

This maps onto the existing architecture rather than requiring a new one: the project already keeps
exactly ONE durable store (the spike lake), and the lake becomes the population's backing store with
VRAM as its hot cache.

### What this replaces in this specification

- **SC-009 is REPLACED.** "The live tick runs at a population of at least 1,000,000 neurons" is a
  cap by another name. The requirement is that **no fixed limit exists in the design**: the
  population grows until storage is exhausted, and the component that stops it must be a measured
  resource, never a constant or an allocation shape.
- **Growth must be chunked/paged.** No growth path may reallocate-and-copy the population, because
  that is O(N) per growth event and makes ever-expanding quadratic in total.
- **VRAM is a cache, not the store.** Residency is decided by activity: the firing set is hot, the
  rest is paged. The durable population lives in the one lake.
- **Sparsity is now doubly load-bearing.** It is required for the synaptic matrix to fit at all
  (dense 1M x 1M is 4000 GB), and it is what makes the active-set argument work -- 2 percent firing
  is what keeps a billion-neuron population's hot set inside a 12.8 GB card.

### New success criteria, replacing SC-009

- **SC-009**: No fixed population limit exists in the design. Growing the population past any
  previously-run size requires no code change, and the ceiling that is eventually hit is a MEASURED
  resource exhaustion, reported with the resource named.
- **SC-013**: Growth is chunked: the per-growth-event cost does not scale with total population
  size. Measured across at least a 100x range of population sizes.
- **SC-014**: Population residency is decided by activity, not by capacity. The resident set is
  reported as "N of M neurons resident" and M is not bounded by VRAM.
- **SC-015**: The active-set fraction is measured, not assumed. If real firing rates are far above
  the 1-2 percent biological figure, the tiering argument weakens proportionally and the measured
  rate must be reported alongside the ceiling it implies.

### The honest caveat

The active-set argument assumes firing rates stay low. That is biologically motivated but is NOT
yet measured on this brain -- SC-015 exists to force it. If this brain's populations fire at 20
percent rather than 2 percent, every ceiling above drops by 10x and the tiering design needs
revisiting. The measurement is cheap and must be taken before the design is committed, not after.

### SC-015 MEASURED IMMEDIATELY, and the assumption was wrong by 5x

The caveat above asked for the firing rate to be measured before the design is committed. Taken at
once, instrumenting `LIFNeuron.step` over a 200-tick live run at seed 0:

```
LIFNeuron.step calls over 200 ticks : 2,379,940
calls that produced a spike         :   249,560
MEASURED FIRING RATE                : 10.49%
```

**Not 2 percent. 10.49 percent** -- about 5x the biological figure the tiering argument assumed.
The consequence, at 12 B per synapse and fanout 100:

```
                 total N     active set    active VRAM   verdict
MEASURED 10.49%    1,000,000       104,859       0.13 GB   fits
                 100,000,000    10,485,978      12.58 GB   fits, with 0.22 GB to spare
               1,000,000,000   104,859,786     125.83 GB   EXCEEDS VRAM by 10x

assumed 2%       100,000,000     2,000,000       2.40 GB   fits
               1,000,000,000    20,000,000      24.00 GB   exceeds by 1.9x
```

**The active-set ceiling on this card is therefore about 100,000,000 neurons, not the ~500,000,000
the 2 percent assumption implied.** Still unbounded in the sense that matters -- the population
itself lives on disk and the design carries no fixed limit -- but the hot-set ceiling is 5x lower
than the previous paragraph claimed, and 100M is uncomfortably close to the 12.58 GB of 12.8 GB it
consumes.

Two consequences the plan must carry:

1. **The active set itself needs tiering above ~100M total population**, not just the cold
   population. Hot/warm/cold rather than hot/cold.
2. **Firing rate is now a first-class design parameter.** At 10.49 percent it is high enough that
   reducing it -- if the brain's dynamics permit -- buys ceiling proportionally. Whether 10.49
   percent is correct for this brain or is itself a defect is a SEPARATE question this spec does not
   answer, but it is worth asking: biological cortex runs far sparser, and a 5x gap is the kind of
   number that is usually telling you something.

This is recorded as a measurement, not a target. It replaces the assumed figure everywhere above.

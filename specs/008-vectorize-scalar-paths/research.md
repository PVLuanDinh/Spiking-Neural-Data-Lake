# Phase 0 Research: vectorize the brain's scalar numeric paths

**Feature**: specs/008-vectorize-scalar-paths | **Date**: 2026-08-03

Four probes, each followed by an independent adversarial refuter that re-ran the decisive
measurement at a larger width with the missing control. **All four probes were refuted on at least
their headline.** Corrected verdicts win. Every number carries the width it was taken at; a number
without a width is UNVERIFIED.

**The standing hazard**: this feature has now measured-at-toy-scale-and-concluded-about-target three
times -- sparse memory at D=64, GPU speed at width 38, and chunking at 100M nnz. Each was a correct
measurement of the wrong thing.

---

## The finding that reorders the feature

**A function drop-in is SLOWER at every width. The win is in the DATA FORMAT, not the function.**

Measured directly in the main session, `substrate/_mathutils.py:61 dot`, 20,000 reps per cell:

```
width  scalar us   numpy PRE-array   numpy +convert   drop-in verdict
   16      0.909            0.494            1.663    0.55x  SLOWER
   64      2.821            0.496            4.051    0.70x  SLOWER
  512     20.058            0.533           26.763    0.75x  SLOWER
```

`+convert` is what a pure function swap actually pays: the list-to-array conversion on every call.
It costs more than the Python loop it replaces, at **every width measured**. `PRE-array` is the
same computation when the data already lives as an array -- **1.84x faster at width 16 and 37.6x at
512**.

The live width is not the wide case anyone imagines. Instrumented over 200 live ticks:

```
dot calls: 437,268    width 16: 424,416 (97.06%)    width 64: 11,192 (2.56%)
```

**Consequence for the plan**: the spec's US1 -- "convert the two hottest sites first" -- measures
SLOWER at each step and would be abandoned as a failed premise. The owner's original instruction
was the correct one: change the DATA FORMAT so arrays flow end to end, and let the call sites follow.
Converting functions one at a time is not a smaller version of that; it is the opposite of it.

---

## Decision 1: The equivalence protocol

**Decision**: Gate Zero, then a tiered ladder. Gate Zero decides whether a site is worth converting
at all; the tiers decide whether the conversion is safe.

- **GATE ZERO -- prove the site is CHARGED and worth it, before proving it is safe.** Neuter the
  site (a constant-returning stub, rebinding EVERY consumer module, not just the defining one) and
  time the tick. This is the step that produces a DECLINE.
- **T0 CLASSIFY (~6 s).** Exact-by-construction ONLY for elementwise over `{+, -, *}` in unfused
  eager execution at a pinned dtype, and reductions with an INTEGER accumulator. Everything else --
  including every transcendental -- falls through.
- **T0b FUSION CLAUSE (GPU tier only).** A T0 exactness proof is void the moment the kernel is fused.
- **T1 CHARGE + LIVE WIDTH (~16 s).** Shadow probe computing both answers side by side over 300 live
  ticks. **If the call count is 0, the DRIVER is wrong until proven otherwise.**
- **T2 REACH (~38 s).** Deep UNROUNDED object-graph fingerprint A/B with an A/A control reading 0.
- **T3 BOUND (~6 s), thresholded float reductions only.** Report the margin; above ~1e3 the sign
  cannot flip and the site is PROVEN harmless rather than observed harmless.
- **T4 GATES, once per BATCH.** Both gate arms with printed proof the swap reached the subprocess.

**Rationale, and three results that each killed a simpler protocol:**

*Gate Zero, at the site in front of us.* Neuter-and-measure puts `dot` at **16.02 percent of real
tick time** -- an Amdahl ceiling of **1.191x** -- while cProfile reported 7.6 percent and ranked
`crossmodal_binder.py:117` first. Profiler attribution is not time. And `jepa_world_model.py:189`
carries **96.7 percent of all dot calls** as a 16x16 matvec written as 16 scalar dots; converting
that ONE site measured **1.086x end-to-end with 0 of 149 state fields changed**.

*Why T0's class rule is restricted.* The original rule was "elementwise, no reduction -> exact,
no A/B needed". Re-run at N = 16,000,000 with A/A controls both reading clean:

```
add/sub/mul                      f64  16,000,000 of 16,000,000 (100.0000%)
sigmoid 1/(1+exp(-x))            f64  15,550,805 of 16,000,000 ( 97.19%)   f32 87.42%
exp(x)      [softmax, lif decay] f64  14,964,856 of 16,000,000 ( 93.53%)   f32 69.80%
tanh(x)                          f64  14,645,398 of 16,000,000 ( 91.53%)   f32 84.65%
```

`_mathutils.sigmoid` is elementwise, has no reduction, and diverges on nearly 1 call in 3 in float32.
The original rule would have shipped it with no A/B at all.

*Why T0b exists.* Elementwise `{+,-,*}` IS bit-identical CPU-vs-CUDA at 16,000,000 of 16,000,000
(f64) and 64,000,000 of 64,000,000 (f32). But at target scale you FUSE, and fusing breaks it:

```
f64 eager-vs-fused bit-identical 10,907,607 of 16,000,000 (68.17%)   maxabsdiff 8.882e-16
SPIKE DECISIONS identical        16,000,000 of 16,000,000            0 threshold flips
```

Exactness is a property of unfused eager execution and evaporates exactly when scale forces the
optimisation. The 0 spike-decision flips is what makes FR-002 workable anyway.

*Why T2 must NOT use `Brain.state()` -- and this indicts earlier work in this repo.*

```
deep unrounded fingerprint   A/A 0 of 98,320   B/B 0 of 98,320   A/B 1,713 of 98,320 differ
same three runs via state()  A/A 0 of 149      B/B 0 of 149      A/B     0 of 149
```

`Brain.state()` rounds to 3-6 decimals and is blind to a 1e-16 perturbation. **It is a FALSE NULL
instrument.** Containment was good -- two prefixes, max divergence 2.775558e-16, 0 differing integer
leaves, growth linear in tick count (1,947 of 291,532 at 1500 ticks) -- but the null through
`state()` was not evidence of anything.

**This matters beyond this feature.** specs/007's charge A/Bs were all measured through
`Brain.state()`. Their POSITIVE findings stand (a difference seen through a coarse instrument is
still a difference). Their NULLS do not: "0 non-tautological fields" may be a false null, and the
seed-0 baseline was measured the same way. specs/007's methodology needs re-running against a deep
fingerprint before any of its nulls are trusted.

**What would falsify this**: a site where Gate Zero says the Amdahl ceiling is large, T0 classifies
it exact, and conversion still changes a discrete outcome.

---

## Decision 2: The sparse representation

**Decision**: **CSR with int32 column indices** for the synaptic matrix -- not COO, not int64 CSR;
assert `nnz < 2^31`. **Dense float32** for the spike vector -- not an int32 index list.
**Post-major CSR plus a single cuSPARSE SpMV** for propagation. Build in chunks of 2e7 to 2e8 nnz.
**DECLINE** block-diagonal partitioning.

Parts 2 and 3 REVERSE the original finding, and the two designs cannot be mixed: a pre-major CSR fed
a dense spike vector silently computes a different quantity. Pick one coherent design.

**Rationale.** At N=1,000,000, F=100, nnz=100,000,000, with checksums IDENTICAL between formats
(5220773.0), so this is a pure cost comparison:

```
COO   : 1,999,901,360 bytes | spmv 12.0462 ms | coalesce peak 8,024.8 MB for a 1,999.9 MB structure
CSR64 : 1,208,000,008 bytes | spmv  4.9718 ms
CSR32 :   804,000,004 bytes | spmv  4.1668 ms
```

COO loses on all three axes at every width measured: 2.49x the memory, 2.9x the SpMV time, and a
4.0x construction blowup. This is the one part that survived every attack.

Chunking is justified by **construction peak alone**, not by any runtime win -- the refuter measured
chunking 1.04x to 5.33x SLOWER at 100M nnz and only 6.6x faster at 800M nnz, which is the toy-scale
trap in its third appearance.

**What would falsify this**: a measured fanout that pushes nnz past 2^31 at a population the design
must support, which would force int64 columns and change the memory table.

---

## Decision 3: The tiering design

**Decision**: **Do not commit a tier design yet.** The connectivity law that every ceiling divides
by has not been measured, and the repo contains a rule that would make it quadratic.

1. **SHIP NOW, independent of tiering**: vectorize `substrate/streaming_hub.py::_scan`'s per-record
   `struct.unpack_from` loop. It is a hot site regardless of how tiering resolves.
2. **MEASURE the connectivity law first.** Count actual nonzero synapses per neuron at three or more
   population widths and FIT the functional form -- constant, fraction-of-width, or mixed. Every
   ceiling in the spec's amendments assumed constant fanout 100. If connectivity is a FRACTION of
   width, memory is quadratic and every ceiling is wrong.
3. **The existing lake CANNOT back the population.** Neither `brain_lake.bin` (BinLog) nor
   `lake.spkl` (StreamingHub) can serve as the population store -- StreamingHub costs 273.4 B
   resident per channel. Reuse the lake's SHAPE (append-only, fixed-width, crash-safe, one file) for
   a new mmap-able fixed-width state file, so the one-durable-store invariant is preserved by
   construction rather than by reinterpreting a knowledge log as a state store.
4. **KEEP the state/synapse split as a DIRECTION and discard every ceiling number attached to it.**
   The 132x separation headline (6.10 B/neuron state vs 805 B/neuron synapses) was refuted as an
   artifact by its own refuter.

**What would falsify this**: a measured connectivity law that IS constant fanout, which would
restore the spec's amendment ceilings as written.

---

## Decision 4: The determinism split

**Decision**: **Determinism is a KERNEL CHOICE, not a device property.** 13 of 15 CUDA ops are
already bit-stable across 100 identical repeats; only the atomics-based spike accumulate is not.

Independently reproduced in the main session before the probes returned:

```
repeat the SAME op 100x on identical input -- DISTINCT results:
  N=1,000 / 1,000,000 / 16,000,000    sum: 1    elementwise LIF: 1
  sparse matvec N=1M fanout 100, 100 runs: 1 distinct
  deterministic-algorithms mode: already 1 distinct either way; 0.0130 -> 0.0140 ms
```

So the spec's recorded "GPU reduction order is not deterministic in general" is TRUE as a general
statement and FALSE for the kernels this feature would use. The tension it recorded as a hard
blocker is not one, PROVIDED atomics-based accumulation is avoided.

What remains true and unresolved: **CPU and GPU do not agree with each other**, and the gap grows
with N.

```
CPU(numpy) vs GPU(torch), identical input:
     N=1,000 and 100,000    absdiff 0.000e+00   (bit-identical)
     N=1,000,000            absdiff 3.750e-01   rel 1.504e-06
     N=16,000,000           absdiff 5.858e+02   rel 1.464e-04
```

**The honest split**: each tier is internally reproducible; they do not agree with each other. The
CPU gate proves structure and small-width numerics. The GPU path proves its own numerics against its
own recorded golden outputs. **Neither proves the other**, and no gate should be written as if it
does.

**What would falsify this**: a driver or hardware change that makes a currently-stable kernel vary,
which is why the GPU path needs its own reproducibility check rather than an inherited assumption.
Determinism was measured on ONE card, ONE driver, ONE torch build.

---

## The scale ladder

| Population | Representation | Tier | Binding constraint |
|---|---|---|---|
| ~10^3 (today, width 16-240) | Python lists | CPU | Interpreter overhead; conversion cost EXCEEDS compute -- vectorizing here LOSES |
| 10^6 | CSR int32 + dense f32 spikes | GPU | Fits VRAM at fanout 100 (804 MB); crossover already passed |
| 10^7 | CSR int32, chunked construction | GPU | Construction peak, not steady-state memory |
| 10^8 | CSR int32 + state/synapse split | GPU + host | **UNKNOWN -- depends on the unmeasured connectivity law** |
| 10^9+ | undecided | undecided | Connectivity law; no ceiling here is trustworthy until it is measured |

The last two rows are deliberately empty of numbers. Every ceiling previously written into the
spec's amendments assumed constant fanout 100, and that assumption is exactly what Decision 3 says
must be measured before it is trusted.

---

## Open questions carried into the design

1. **The connectivity law.** Constant fanout, or a fraction of width? Every ceiling divides by it.
   Cheapest experiment: count nonzero synapses per neuron at three population widths and fit.
2. **Does specs/007 survive a deep fingerprint?** Its nulls were measured through `Brain.state()`,
   now shown to be a false-null instrument. Cheapest experiment: re-run the growth charge A/B
   against the deep object-graph fingerprint at the same seeds.
3. **Where does the data-format conversion boundary go?** The drop-in loses at every width, so
   arrays must flow end to end -- but the perimeter where lists become arrays is undesigned.

## What this research REFUTED

- **"Vectorize the hot sites for a large win."** A drop-in is 0.55x to 0.75x -- SLOWER at every
  width. The win requires changing the data format.
- **"cProfile identifies the hot site."** It ranked `crossmodal_binder` first at 4.97M calls;
  neuter-and-measure puts `dot` at 16.02 percent with a 1.191x ceiling. Attribution is not time.
- **"Elementwise operations are exact, so they need no A/B."** True for `{+,-,*}` unfused; false for
  every transcendental (sigmoid 87.4 percent in f32) and false for anything fused (68.2 percent).
- **"`Brain.state()` shows whether a change reached the brain."** A/B reads 0 of 149 through
  `state()` and 1,713 of 98,320 through a deep fingerprint. It is a false-null instrument.
- **"GPU reductions are non-deterministic."** 13 of 15 ops bit-stable over 100 repeats.
- **"COO is the natural sparse format."** Loses on memory, speed and construction peak at every
  width.
- **"The existing lake can back an unbounded population."** It cannot; StreamingHub costs 273.4 B
  resident per channel.
- **"Chunking is faster."** 1.04x to 5.33x SLOWER at 100M nnz. It is justified by construction peak
  alone.

---

## T4 GATE RESULT 2026-08-04: the numpy swap passes the full suite

Decision 1's protocol ends at T4 -- both gate arms, with printed proof the swap reached the
subprocess. Run in the main session with `_mathutils.dot` replaced by a numpy pairwise dot via a
`sitecustomize.py` on `PYTHONPATH` (the repo itself untouched):

```
pytest exit                : 0
tests collected            : 687
[SWAP] numpy dot calls     : 9,543,154
coverage                   : 92.81 percent  (baseline 92.85)
```

**Left-fold to pairwise summation breaks 0 of 687 tests.** The 9,543,154 call count is the part
that makes this evidence rather than an assumption -- see below.

**This does NOT mean the change is invisible.** The deep fingerprint measures 1,713 of 98,320 leaves
moving on the same swap, contained to two prefixes, max divergence 2.775558e-16, 0 integer leaves,
growth linear in tick count. So this is precisely FR-002's case: the difference is REAL, MEASURED,
and shown not to move any gate or seeded outcome. It is not the FR-001 bit-identity case, and at
target width bit-identity is not available at all.

**The instrument failed twice before it worked, and both failures looked like success.**

1. The first harness silently did not install -- `_SWAPPED: False`, `0` calls. Running pytest
   against it would have exercised the UNMODIFIED code and reported a clean pass. The call counter
   is the only reason this was caught.
2. The second attempt appeared to die: log frozen at 73 percent, no python process, no exit record.
   It had in fact completed; the notification arrived late. A run was nearly re-done on a false
   reading of its own absence.

Both failure modes present as "no output", which is why C4 requires proof the swap is CHARGED rather
than proof it did not fail. An uninstrumented swap and a dead process are indistinguishable from a
passing run.

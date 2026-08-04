# Contract: converting a scalar site to an array-native one

**Feature**: specs/008-vectorize-scalar-paths | **Phase**: 1 | **Date**: 2026-08-04

Obligations a conversion must satisfy. Written as rules rather than signatures, because the plan
deliberately does not choose a module layout -- the perimeter is a property of call sites, not a
package.

---

## C1. Gate Zero -- prove the site is worth converting BEFORE proving it is safe

1. **The ceiling comes from neuter-and-measure, never from a profiler.** Replace the site with a
   constant-returning stub, rebinding EVERY consumer module rather than only the defining one, and
   time the tick. cProfile ranked `crossmodal_binder.py:117` first at 4,966,400 calls; neutering put
   `_mathutils.dot` at 16.02 percent of tick time with a **1.191x** Amdahl ceiling. Attribution is
   not time.
2. **The live width histogram is measured, not assumed.** `dot` runs at width 16 on 97.06 percent of
   its 437,268 calls. A microbenchmark at width 512 describes 0 percent of real traffic.
3. **A site whose ceiling is below the noise floor of the tick is DECLINED**, and the decline carries
   its number.

## C2. The perimeter, not the function

1. **A conversion that leaves data as Python lists is FORBIDDEN.** Measured: the drop-in is 0.55x at
   width 16, 0.70x at 64, 0.75x at 512 -- slower at every width, because the per-call conversion
   costs more than the loop. Converting the function while leaving the format is a regression with
   extra steps.
2. **The perimeter must be CLOSED.** Zero conversions strictly inside it. A conversion inside means
   the perimeter is drawn in the wrong place.
3. **The perimeter is measured END TO END**, at the tick, not at the site. A site 37x faster inside a
   perimeter that pays conversion at its edges can still be a net loss.
4. **P2 applies at the boundary**: the scalar implementation is DELETED, not flag-guarded. Two
   representations behind a switch is the additive pattern P2 forbids.

## C3. Exactness -- what must be proven depends on the class

| Class | Obligation |
|---|---|
| integer reduction | None. Exact by construction |
| elementwise, unfused, pinned dtype | T0 classification only |
| transcendental | Full ladder. `sigmoid` diverges to 87.42 percent bit-identical in f32; `exp` to 69.80 percent |
| float reduction | Full ladder plus a T3 margin bound |
| **anything fused** | **Exactness proofs are VOID.** Re-prove against the kernel that actually ships |

**The clause that is easiest to get wrong**: "elementwise is exact" holds only in unfused eager
execution. At target scale you fuse, and fusing drops bit-identity to 68.17 percent. Spike decisions
survived at 16,000,000 of 16,000,000, which is why FR-002's measured-difference route remains open.

## C4. Evidence -- the strictest clause, because every past failure here was an evidence failure

Every acceptance claim MUST ship:

| Requirement | Reason |
|---|---|
| **Proof the swap is CHARGED** (call count > 0) | Phase 0's harness silently failed to install -- `_SWAPPED: False`, 0 calls -- and would have measured unmodified code as a clean pass |
| **`tools/deep_fingerprint.py`, not `Brain.state()`** | `state()` rounds to 3-6 decimals and reported 0 of 149 where the deep walk found 1,713 of 98,320. It is a measured false-null instrument |
| **A matched A/A control reading 0** | Nonzero A/A means run nondeterminism; the A/B through it is unreadable |
| **0 integer leaves moved** | Integers moving means discrete state changed, not floats drifting |
| **Divergence growth at most linear in ticks** | Measured linear at 1,713 (300 ticks) to 1,947 (1500). Super-linear means the perturbation compounds |
| **The exact command, and the WIDTH it was run at** | This feature measured-at-toy-scale-and-concluded-about-target three times |

**FORBIDDEN as evidence**: a null through `Brain.state()`; a benchmark without warmup and
`torch.cuda.synchronize()`; a speedup measured at a width the site never sees; a profiler rank.

## C5. GPU tier

1. **No atomics-based accumulation on the live path.** 13 of 15 CUDA ops are bit-stable across 100
   identical repeats; the atomics-based spike accumulate is the exception. Determinism here is a
   kernel choice, and choosing wrong forfeits it.
2. **The GPU path carries its own reproducibility gate** with recorded golden outputs. The CPU gate
   cannot execute it -- CI has no GPU.
3. **Neither tier proves the other.** CPU and GPU disagree by rel 1.504e-06 at 1M and 1.464e-04 at
   16M, and the gap grows with N. No gate may be written as though passing one covers the other.
4. **CSR int32 post-major with a dense f32 spike vector and one cuSPARSE SpMV.** Mixing pre-major CSR
   with a dense spike vector silently computes a different quantity -- this is a correctness clause,
   not a performance preference.
5. **Assert `nnz < 2^31`.** int32 columns are why CSR32 wins; exceeding the range corrupts silently.

## C6. What this contract deliberately does not specify

- **Where the perimeter goes.** P1 establishes one on a single subsystem and measures it; the shape
  generalises from that measurement, not from this document.
- **The tiering design.** Blocked on the unmeasured connectivity law. Committing a storage layout to
  an assumed constant fanout would bake an unverified assumption into the most expensive place to be
  wrong.
- **Whether `dot` itself is ever converted.** Its 1.191x ceiling may not survive Gate Zero against
  the cost of moving its perimeter. `jepa_world_model.py:189` carries 96.7 percent of its calls and
  may be the whole answer.

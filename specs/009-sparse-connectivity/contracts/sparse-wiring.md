# Contract: thinning a module's connectivity

**Feature**: specs/009-sparse-connectivity | **Phase**: 1 | **Date**: 2026-08-04

Obligations a module must satisfy to be reported SPARSE. Written as rules rather than signatures,
because the wiring is a property of each module's own weight matrix and no shared base class is
imposed.

---

## C1. Function is the constraint, not the benefit

1. **The module's OWN task metric is measured against its dense baseline at the same width, load and
   seed set.** Not a proxy, not a connection count, not a coupling sum.
2. **The metric MUST be able to get worse.** A quantity that rises with parameter count regardless
   of learning is arithmetic. `mean_assembly_coupling()` is rejected on exactly this ground.
3. **The metric MUST be validated by a null that destroys learning while preserving structure**, and
   the null MUST read ~0. Without it, a metric that merely counts weights passes as function.
4. **The dense baseline MUST be non-trivial.** Comparing to a baseline that is already failing makes
   the tolerance meaningless -- a thinning test was once run against `retained() = 0.125` and proved
   nothing.
5. **The metric MUST NOT be an oracle.** If the wiring is derived from the same signal the metric
   scores, an untrained network can score perfectly. Measured: assembly-derived wiring on
   `AttractorNet` scores 1.000 UNTRAINED, because the co-activity mask IS the class partition.

## C2. The bound is derived, never declared

1. **No default fanout.** The knee tracks stored LOAD, so no single number is correct across
   deployments. A default would be the magic constant P1 forbids.
2. **F\* MUST be bracketed** -- a failing F strictly below it on the grid. Unbracketed, "it passed at
   the smallest F I tried" is not a knee.
3. **An F\* at the grid floor is reported "censored"**, never as a value.
4. **The tolerance is fixed BEFORE the sweep** and recorded. A threshold chosen after seeing results
   is unfalsifiable.
5. **A drop above 0.20 between adjacent grid points is a CLIFF**, reported as such rather than
   smoothed into a knee.

## C3. The wiring itself

1. **Symmetric, zero diagonal.** A one-way connection under a symmetric learning rule is a different
   system, not a thinned one.
2. **Deterministic from the seed alone**, in 20 of 20 trials. A random topology that is not
   reproducible takes the gates' seeded-structure property with it.
3. **Drawn from a stream independent of the module's other randomness.** Wiring must not consume
   draws that replay or noise depend on.
4. **Uniform random k-out unless the module's indices carry real structure.** `encode()` draws
   assemblies with `rng.sample(range(dim))`, so index order is an arbitrary permutation and any
   ring, distance kernel or small-world construction over it is decoration. `SpikingGridCAN` is the
   counter-case in-tree: it has a real torus-phase metric and already exploits it.
5. **Every weight update routes through ONE pruning point.** A per-site prune can be forgotten by the
   next update site, silently restoring all-to-all while the mask still looks correct.
6. **`degree()` reports the REALISED degree**, which symmetrisation raises above the requested F.
   Reporting the request as though it were the result is a lie about what was built.

## C4. Evidence

Every SPARSE verdict MUST ship:

| Requirement | Reason |
|---|---|
| Dense baseline at the same width AND load | Load is the driver; a baseline at a different load is not a comparison |
| At least 5 seeds, mean and minimum both reported | A mean alone hides a seed that collapsed |
| The width function was VERIFIED at | Everything above it is extrapolation and must be labelled |
| Proof the mask actually zeroes weights | A mask that is stored but not applied makes the store sparse only on paper |
| Nulls via `tools/deep_fingerprint.py` with a matched A/A control at 0 | `Brain.state()` is a measured false-null instrument: 0 of 149 where a deep walk found 1,713 of 98,320 |

**FORBIDDEN as evidence**: a memory saving on its own (maximised by deleting everything); a fanout
count on its own (same); a metric that cannot decrease; a tolerance fitted after the sweep; a
baseline at a different load or width.

## C5. Honest reporting

1. Counts as `N of M`, never `N/M`.
2. Modules reported as **"N sparse, K documented exceptions, U unmeasured, of M"** -- all four
   numbers. `UNMEASURED` is the default and must never be reported as anything else.
3. A module that genuinely needs dense connectivity is a **documented exception with its
   measurement**, not a failure and not a gap to be forced.
4. **No ceiling is ever described as "unbounded".** Bounded fanout turns a quadratic ceiling into a
   linear one; it does not remove it. Report the number and the resource that binds it.

## C6. What this contract does not specify

- **Which F any module uses.** Measured per deployment; a number here would be the constant C2
  forbids.
- **A shared base class or protocol.** Wiring is a property of each module's own matrix, and imposing
  a hierarchy would be a structure this evidence does not justify.
- **When the dense path is deleted.** It closes the P2 deviation, and it happens per site once F is
  measured there -- not on a schedule.

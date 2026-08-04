# Phase 0 Research: sparse connectivity

**Feature**: specs/009-sparse-connectivity | **Date**: 2026-08-04

Four probes, each followed by an independent adversarial refuter. **Fifteen claims went in plausible
and came out false**, several of them the headline of the analysis that produced them -- including
one of mine. Corrected verdicts win. Every number carries the width AND the stored load it was taken
at, because load turned out to be the variable everything depends on.

---

## Decision 1: The module inventory and each module's own task metric

**Decision**: `hippocampus/episode_consolidation.py` `EpisodeConsolidator` at `dim=256` is the P1
target. `retained(cue_frac=0.5, thresh=0.9)` and `recall_episode()` exact-match are the metrics.
`mean_assembly_coupling()` and nearest-prototype similarity are REJECTED.

**Rationale**: a live-brain census (200 ticks, depth 18) found 171 two-dimensional structures,
392,148 cells, **99.57 percent dense**. Separating true synaptic connectivity from state and buffers
leaves 103 structures, 381,422 cells, **99.82 percent dense**, across 12 owner classes -- so the
all-to-all finding is not inflated by state.

`EpisodeConsolidator` is the target because it is the only large, true, square N x N recurrent
structure over a single population, `nnz/row = N - 1` exactly, it owns 17.2 percent of connectivity
memory, and its metric provably falls. At `dim=256`, 40 episodes, top-|w| thinning:

```
k/row      256    128     64     32     16      8      4      2
retained  1.000  0.975  0.925  0.550  0.150  0.000  0.000  0.000
recall     40/40  40/40  40/40  35/40  32/40  25/40  20/40  11/40
false alarms on 8 never-stored cues: 0 of 8 at EVERY k
```

`retained()` is validated as a metric that can LOSE, by two independent nulls both reading exactly
zero: a row-shuffle preserving nnz and the per-row value multiset gives `0.0000` against dense
`1.0000` at N = 1,024, 2,048 and 4,096; and a learning-destroyed null (same wiring, same fanout,
same parameter count, weights from episodes never probed) gives `0.0000` at F = 160, 320, 640 and at
dense. The metric requires the learning, not the connectivity.

**Alternatives considered**: `PcCortexServe` owns 80.4 percent of connectivity memory and lost on
measurement -- its 2312 input dimension is the N-MNIST sensor frame, not a neuron population, so its
memory is `O(in_dim * d1)` and thinning it saves bytes without bending the quadratic curve.
`mean_assembly_coupling()` lost because it is a sum over surviving weights: thinning lowers it
arithmetically whether or not recall degrades. It measures the manipulation, not the function.

**What would falsify this**: a module with a larger true N x N structure whose own metric also
degrades under thinning.

---

## Decision 2: The fanout knee -- and BOTH candidate invariants are false

**Decision**: **Function survives thinning easily at fixed load.** The invariant is **neither density
nor connections-per-unit**. Both were asserted during this research -- one of them by me, in the main
session -- and both were falsified on the same module with the same metric. **The variable that sets
the knee is STORED LOAD.**

**The pre-registered tolerance**, written to a file BEFORE any sweep ran and inherited unchanged by
every re-run:

> F PASSES iff, over seeds 0..4 at the same width and stored load, the module's own metric is within
> **0.02 mean** and **0.05 minimum** of the dense baseline. F\* is the smallest passing F for which
> every larger grid point also passes, and MUST be BRACKETED by a failing F below it. An F\* at the
> grid floor is reported as "censored", never as a value. A drop above 0.20 between adjacent points
> is a CLIFF, not a knee.

**At FIXED load (40 episodes), across a 64x width range:**

```
     N     F*   knee density   bracketed?
   256    176       68.8%      yes (160 -> 0.965 fails)
 1,024    160       15.6%      yes (144 -> 0.975)
 4,096    160        3.9%      yes (128 -> 0.975)
16,384    128        0.78%     yes (112 -> 0.335, a CLIFF)
```

Connections-per-unit **falls 176 to 128** while density falls 68.8 to 0.78 percent. That satisfies
FR-005 and SC-001 more strongly than "flat", at 4x the width SC-008 requires.

**At PROPORTIONAL load (P = N/8, activity a = 0.08 held), the answer inverts:**

```
   dim    load   uniform F*   knee density   top-k-per-row F*
   256      32        160        62.75%            96
   512      64        192        37.57%            96
 1,024     128        320        31.28%            96
 2,048     256        512        25.01%           128
 4,096     512      1,024        25.01%             -
```

Uniform F\* grows as **N^0.68** -- 6.4x over a 16x span -- and knee DENSITY goes constant, which is
the outcome that kills the feature. **But top-k-per-row holds F\* at 96 to 128 across the same
range**, which is the result the design rests on.

**Both invariants die on measurement**: at fixed width N=192, two assembly sizes give knee densities
of 0.0681 and 0.5654 -- an **8.3x spread at fixed N**, which no density law survives. Between 2,048
and 4,096 under proportional load, knee fanout doubles 512 to 1,024 while density is constant at
0.2501, which no connections-per-unit law survives.

**My own error, recorded**: the main session measured fanout as `N^0.329` and concluded it grew
sublinearly. That sweep held stored load at 40 episodes while sweeping width 256 to 2,048, so the
store was relatively emptier at every step. It measured the load ratio and attributed the result to
width. The tolerance was also fitted after the sweep rather than before.

**What would falsify this**: a module whose knee moves with width at genuinely fixed load ratio.

---

## Decision 3: The topology, and how the bound is DERIVED

**Decision**: **uniform random k-out**, seeded per instance with `random.Random(seed)`. The bound is
derived by a **PROCEDURE, not a formula**: a graph-connectivity FLOOR (necessary, never sufficient),
a hardware-budget CEILING (self-extinguishing), and a **bracketed bisection against measured
function** at the deployment width and load, which sets the operating point.

**No closed-form expression for F is adopted, because every one attempted relocated the constant it
was supposed to remove.**

**Rationale**: `EpisodeConsolidator.encode()` draws assemblies with
`rng.sample(range(self.dim), self.K)`. Unit index order is therefore an arbitrary permutation and
**no metric exists over it** -- any ring lattice, distance kernel, small-world construction or block
partition over those indices is decoration. Uniform random k-out is not a fallback here; it is the
only topology the module's own semantics admit. The opposite case exists in-tree and confirms the
reasoning: `SpikingGridCAN` has a real torus-phase metric and already exploits it.

**What would falsify this**: a module whose unit indices carry real structure, where a
distance-dependent topology beats uniform random at equal fanout.

---

## Decision 4: What bounded fanout buys, and what breaks next

**Decision**: at fixed load, a population ceiling of **12,500,000 in 12.8 GB VRAM** at 8 B/synapse
(8,333,333 at 12 B/syn CSR; 8,263,395 charging eligibility traces) against the measured dense
baseline of **40,000** -- **312x, 208x and 207x**, all clearing SC-004's 10x.

**What breaks next is NOT the synapse.** In order: **(1) the neuron representation, (2) tick time,
(3) whether stored load scales with the population.**

**The guardrail that matters most here**: cost cannot set F. Every population ceiling is
`budget / bytes_per_neuron(F)`, a quotient that rises monotonically as F falls **by construction**,
regardless of what thinning does to the brain. It cannot get worse. Function preservation is the
constraint; cost measures only the benefit side. F comes from Decisions 2 and 3; Decision 4 only
says whether it fits.

**What would falsify this**: a measured neuron-state floor that is itself quadratic in population.

---

## The scale ladder

Condition: `EpisodeConsolidator`, uniform random k-out, activity a = 0.08, **stored load FIXED at 40
episodes**, tolerance mean 0.02 / min 0.05, seeds 0..4. **This is the only condition in which the
bound was verified.** Connection memory only, 12.8 GB VRAM budget.

| Population | Fanout F | Connection memory (8 B / 12 B per syn) | Function | Binding constraint |
|---|---|---|---|---|
| **10^4** | 160 | 12.8 MB / 19.2 MB (dense 800 MB, 62.5x) | **VERIFIED** -- bracketed at 4,096 and 16,384, which straddle this rung | Not the synapse (0.1 percent of budget). The **neuron representation**: 6.5 MB of Python-object state at 646.5 B/n that cannot enter VRAM at all |
| **10^5** | 128 | 102.4 MB / 153.6 MB (dense 80 GB, 781x) | **EXTRAPOLATED** 6.1x beyond the largest measured width | **Neuron representation** -- unreachable in VRAM until specs/008 vectorizes to the measured 9.0 B/n floor |
| **10^6** | 128 | 1.02 GB / 1.54 GB | **EXTRAPOLATED** | Neuron representation, then tick time |
| **10^7** | 128 | 10.2 GB / 15.4 GB | **EXTRAPOLATED**, and at 12 B/syn it exceeds VRAM | Synapse memory finally binds |

**The extrapolated rows are marked because this project has treated a toy-scale measurement as a
target-scale fact three times.** Function is VERIFIED only at 10^4.

---

## Modules that need dense connectivity (FR-003 documented exceptions)

None confirmed yet. `SpikingGridCAN` is a candidate for a DIFFERENT reason -- its connectivity is
already `O(N)` in a torus-phase metric, not `O(N^2)`, so it is out of scope rather than exceptional.
`AttractorNet` is unresolved: its `accuracy()` does fall under thinning, but the assembly-derived
mask is an ORACLE on it (see below), so its evidence is contaminated and it needs a clean metric
before it can be judged.

---

## Open questions carried into the design

1. **Does this brain's stored load scale with its population?** The whole feature turns on it. At
   fixed load the bound falls; at proportional load uniform-random grows as N^0.68 while
   top-k-per-row stays at 96 to 128. Cheapest experiment: instrument episode count against `dim`
   over a long live run.
2. **Does top-k-per-row survive at 10^5?** It is the result that makes proportional load survivable
   and it is measured only to 4,096.
3. **`AttractorNet` needs an uncontaminated metric** before it can be classified.

---

## What this research REFUTED

- **"The fanout knee is flat in population."** Flat only at FIXED load. Under proportional load the
  uniform knee goes 160, 192, 320, 512, 1,024 -- `N^0.68`.
- **"The invariant is connections-per-unit"** AND **"the invariant is density."** Both asserted, both
  dead: 8.3x density spread at fixed N; fanout doubling at constant density.
- **My own `N^0.329` fanout law** -- confounded by holding episode count fixed while width grew.
- **"Assembly-derived (co-activity) wiring holds function at 1 to 4 connections per unit."** Dead
  twice: on `EpisodeConsolidator` at proportional load it has NO knee at any fanout tested
  (`retained() = 0.0000` at every k up to 512 at dim 4,096), because co-active partners per unit
  reach 3,942.3 of 4,095 (96.3 percent). And on `AttractorNet` it is an **ORACLE** -- with two
  disjoint assemblies the co-activity mask IS the ground-truth class partition, so an UNTRAINED
  masked net scores 1.000. A metric that leaks the answer.
- **"A closed-form F can be derived."** Every attempted formula relocated the constant it was meant
  to remove.
- **"Cost can choose the fanout."** Every ceiling is `budget / bytes(F)`, monotone in F by
  construction. It cannot get worse, so it cannot be evidence.

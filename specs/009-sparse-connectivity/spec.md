# Feature Specification: Sparse connectivity -- bound fanout independently of population size

**Feature Branch**: `009-sparse-connectivity`

**Created**: 2026-08-04

**Status**: Draft

**Input**: Bound per-neuron fanout so it does not grow with population size, so that memory becomes
O(N*F) rather than O(N^2) and the population ceiling stops being quadratic.

---

## Why this exists

The brain's connectivity is all-to-all. Measured 2026-08-04 by constructing one connectivity-bearing
module at six widths and training each with 40 episodes:

```
  dim N   nnz/row   density   nnz/N^2
     16      15.0     93.8%    0.9375
     32      28.0     87.5%    0.8750
     64      57.5     89.8%    0.8984
    128     127.0     99.2%    0.9922
    256     255.0     99.6%    0.9961
    512     511.0     99.8%    0.9980
```

**Fanout is not a constant. It IS the width** -- `nnz/row = N - 1` exactly -- and density RISES
toward 100 percent as the population grows, so scaling makes the problem worse rather than better.
A census of the running brain agrees independently: 172 weight structures, 396,813 nonzeros of
400,878 cells, **98.99 percent dense overall**.

What that costs, at roughly 8 bytes per synapse:

```
store              max population    (the previous spec assumed)   error
VRAM 12.8 GB               40,000                    10,666,666    267x
+ 64 GB host               97,979                    64,000,000    653x
+ 2 TB disk               509,901                 1,733,333,333  3,399x
+ 8 TB disk             1,004,987                 6,733,333,333  6,700x
```

**A fully-connected million-neuron brain needs 8.0 TB of connections.** And storing them
efficiently does not help: at 99.8 percent density, a sparse format's index overhead is strictly
worse than dense storage. **The blocker is architectural, not representational** -- which is why it
needs its own feature rather than a storage fix inside another one.

---

## The risk that defines this feature

**This is a behavioural change, not a storage change.** Thinning an all-to-all network to a few
percent density changes what the brain computes. A brain that scales but can no longer do what the
dense one did has not satisfied this feature -- it has traded a capability for a number, which is
the most expensive kind of false progress available here.

So the acceptance criterion is **function preserved at reduced fanout, measured against the dense
baseline at the same width**. Scale is the benefit; function is the constraint.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One module keeps working with far fewer connections (Priority: P1)

A single connectivity-bearing module is rewired so each unit connects to a bounded number of others
instead of all of them, and it still performs its own task as well as it did dense.

**Why this priority**: It is the smallest slice that can falsify the whole idea. If function cannot
survive thinning on one module at one width, no amount of topology cleverness will save it at scale,
and the feature should stop. It also produces the measurement everything else depends on -- the
fanout knee.

**Independent Test**: Run the module's own task at full connectivity and at reduced fanout, same
width, same seed. Compare task performance. Fully testable without touching any other module.

**Acceptance Scenarios**:

1. **Given** a module at its current width, **When** its task performance is measured at full
   connectivity and at reduced fanout, **Then** the reduced-fanout performance is within a stated
   tolerance of the dense baseline, and the tolerance was fixed before the measurement was taken.
2. **Given** fanout is swept from full down to very sparse, **When** task performance is plotted
   against fanout, **Then** a knee is identified from the data rather than a round number being
   chosen.
3. **Given** the same seed, **When** the wiring is generated twice, **Then** it is identical.
4. **Given** reduced fanout, **When** memory is measured, **Then** it is lower than dense at the
   same width, stated as a ratio.

---

### User Story 2 - The saving grows with population size (Priority: P2)

The bounded fanout holds as the population grows, so the memory curve bends from quadratic to
linear and the ceiling moves by orders of magnitude rather than percent.

**Why this priority**: This is the actual goal. It is second because it is only meaningful once P1
shows function survives at all -- a linear memory curve for a brain that no longer works is not
progress.

**Independent Test**: Measure connections per unit at three or more populations spanning at least a
16x range, and fit. Flat means the goal is met; tracking width means it is not.

**Acceptance Scenarios**:

1. **Given** the rewired module at several populations, **When** connections-per-unit is measured,
   **Then** it does NOT grow with population size, demonstrated across at least a 16x range.
2. **Given** the same populations, **When** total connection memory is measured, **Then** it grows
   about linearly with population rather than quadratically.
3. **Given** the largest tested population, **When** task performance is measured, **Then** it is
   still within tolerance of dense -- **the fanout bound must hold at widths it was NOT tuned on**.

---

### User Story 3 - The whole brain runs sparse, and the dense path is gone (Priority: P3)

Every connectivity-bearing module uses the bounded-fanout wiring, the dense implementations are
deleted, and the brain as a whole still does what it did.

**Why this priority**: It is the end state and the only point at which the population ceiling
actually moves. It is last because a partially-converted brain gives neither the memory saving nor a
clean comparison, and because module-by-module knees may differ.

**Independent Test**: Run the full brain sparse and dense from the same seed and compare
whole-brain outcomes, then count the surviving dense implementations.

**Acceptance Scenarios**:

1. **Given** the full brain, **When** every connectivity module is rewired, **Then** whole-brain
   behaviour is compared against the dense baseline and any difference is stated as a number.
2. **Given** the conversion is complete, **When** dense implementations are searched for, **Then**
   exactly one connectivity mechanism survives -- not two behind a switch.
3. **Given** a module whose function cannot survive thinning, **When** it is evaluated, **Then** it
   is recorded as a documented exception with its measurement, and the feature reports "N of M
   modules sparse" rather than claiming completeness.

---

### Edge Cases

- **Function degrades gracefully rather than sharply.** There may be no clean knee -- performance
  may decline smoothly, making "where to stop" a judgement about acceptable loss rather than a
  measurement. The spec must say what loss is acceptable BEFORE the sweep, or the threshold will be
  chosen to fit the result.
- **A module needs all-to-all by nature.** An attractor network's capacity is a function of its
  connectivity; thinning it may reduce what it can store. This is a legitimate exception, not a
  failure to try hard enough.
- **Sparsity helps function.** Plausible and must not be assumed: sparse networks sometimes
  generalise better. If measured, it is a finding, and it must not be used to excuse a thinning that
  degrades a different module.
- **The wiring is random and unseeded.** Any random topology must be reproducible from the seed or
  the gates lose determinism.
- **Fanout bound tuned at one width, applied at another.** The previous feature measured at toy
  scale and concluded about target scale three separate times. A bound validated only at width 512
  says nothing about width 100,000.
- **Thinning changes learning dynamics, not just capacity.** A weight update rule tuned on dense
  connectivity may behave differently when most connections do not exist.
- **The population still cannot reach the goal.** Even at bounded fanout there is a ceiling. It must
  be reported honestly rather than described as unbounded.

---

## Requirements *(mandatory)*

### Functional Requirements

**Function preservation -- the constraint**

- **FR-001**: Each rewired module MUST perform its own task within a stated tolerance of its dense
  baseline, measured at the same width and seed.
- **FR-002**: The tolerance MUST be fixed BEFORE the sweep that measures it. A threshold chosen
  after seeing the results is not a criterion.
- **FR-003**: A module whose function cannot survive thinning MUST be recorded as a documented
  exception with its measurement. Counting it as converted is forbidden.
- **FR-004**: Whole-brain behaviour MUST be compared sparse versus dense, and any difference stated
  as a number rather than asserted to be acceptable.

**Bounded fanout -- the goal**

- **FR-005**: Connections per unit MUST NOT grow with population size, demonstrated across at least
  a 16x range of populations.
- **FR-006**: The fanout bound MUST be DERIVED -- from measured function, a connectivity budget, or
  a graph property -- never declared as a round number.
- **FR-007**: Total connection memory MUST grow about linearly with population rather than
  quadratically, stated as a fitted relationship.
- **FR-008**: The bound MUST hold at populations it was NOT tuned on.

**Honesty and governance**

- **FR-009**: Exactly ONE connectivity mechanism survives. The sparse path replaces the dense one;
  keeping both behind a switch is forbidden.
- **FR-010**: Identical seeds MUST produce identical wiring, in 20 of 20 trials.
- **FR-011**: The effector envelope does not widen. Existing safety refusals stay.
- **FR-012**: Any claim that a change has NO downstream effect MUST be measured with the deep
  fingerprint and a matched control reading zero -- never with the rounded state summary, which is a
  measured false-null instrument.
- **FR-013**: The population ceiling MUST be reported as a number with the resource that binds it,
  never as "unbounded".
- **FR-014**: Every performance and memory number MUST carry the population size it was measured at.

### Key Entities

- **Connectivity budget**: The bounded number of connections a unit may have. Derived, not declared.
  The central quantity of this feature.
- **Fanout knee**: The point where thinning starts costing function. Found from a sweep.
- **Module task performance**: What a module is FOR -- recall accuracy, pattern completion,
  prediction error. Not a proxy, and not a connection count.
- **Documented exception**: A module that genuinely needs dense connectivity, with the measurement
  that shows it. A legitimate outcome.

---

## Success Criteria *(mandatory)*

- **SC-001**: Connections per unit do not grow with population, across at least a 16x range.
  Baseline today: connections per unit equals population minus one, at every width measured.
- **SC-002**: Every rewired module performs within its stated tolerance of its dense baseline at the
  same width.
- **SC-003**: Connection memory grows about linearly with population. Baseline today: quadratic.
- **SC-004**: The population that fits in available memory rises by at least 10x against the
  measured baseline of 40,000 in the fastest store.
- **SC-005**: Identical seeds produce identical wiring in 20 of 20 trials.
- **SC-006**: Exactly 1 connectivity mechanism exists at completion.
- **SC-007**: Modules are reported as "N of M sparse, K documented exceptions", with all three
  numbers stated.
- **SC-008**: Function is verified at a population at least 4x larger than the one the bound was
  tuned on.

---

## Assumptions

- "Function" means each module's own task, measured its own way. There is no single brain-wide
  score that would make this comparable across modules, and inventing one would hide exactly the
  per-module degradation this feature must detect.
- Tolerance is per module and stated in advance. Modules differ in how much loss is acceptable --
  a memory store losing recall matters differently from a router losing precision.
- Some modules will legitimately need dense connectivity. The feature reports them rather than
  forcing them, and a feature that converts most modules is a success.
- Biological sparsity (order 10^4 connections per neuron against 10^11 neurons, small-world and
  distance-dependent topologies) is offered as a REFERENCE, not a target. Whether it transfers to
  these modules is an open question for the plan, not an assumption in the spec.
- The existing determinism, safety and evidence rules carry over unchanged.

## Dependencies and Relationship to Other Work

- **specs/008 (vectorization) does NOT depend on this and is not blocked by it.** Its perimeter
  work, equivalence ladder and profiling findings stand on their own.
- **But specs/008's sparse-format recommendation only becomes correct once THIS feature lands.** It
  benchmarked a sparse format at an assumed fanout of 100; at the brain's actual 99.8 percent
  density, that format is worse than dense. The two must not be solved twice in incompatible ways.
- **specs/007 (whole-brain neurogenesis) interacts directly.** Growing a population under all-to-all
  connectivity costs O(N) new connections per new unit and rising. Bounded fanout makes growth
  affordable, so this feature is a precondition for growth at any scale.

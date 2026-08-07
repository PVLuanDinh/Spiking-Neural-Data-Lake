# Feature Specification: Order neurogenesis by cortical architecture, from a prior that must be beatable

**Feature Branch**: `011-h01-neurogenesis-prior`

**Created**: 2026-08-06

**Status**: Draft

**Input**: User description: order neurogenesis by real cortical architecture, using the H01
human-cortex reconstruction as a BEATABLE PRIOR rather than as a target.

---

## Why this feature exists

Measured against the tree on 2026-08-06, not read: the growth engine produces **dimensionless
units**. A newly grown neuron has no position, no cortical layer, no excitatory/inhibitory identity
and no propagation delay. Tree-wide, the terms `coord`, `position`, `xyz`, `excitat` and `inhibit`
match zero times in the growth engine; the only `layer` hits there mean *the persistence layer*, i.e.
software; and no file anywhere references Dale's Principle or a conduction delay. New neurons are
**ordered by nothing** — they are appended to a population.

At the same time the tree already contains a faithful layered cortical microcircuit (e and i pools
across L2/3, L4, L5, L6) which is permanently exempted from the dormancy gate **because it has no
input port**: it accepts a scalar background rate and no signal. So the project holds a layered
circuit nothing can drive, beside a growth process that produces unplaced units.

Both are open work the project has already named. This feature closes the gap between them, and it
takes its ordering from a measured reconstruction of real human cortex rather than from a guess.

## The one constraint everything else follows from

The reconstruction supplies **constants measured in one patient, one cubic millimetre, one cortical
region**. Pinning those constants into the brain is precisely the hardcoding the project forbids.

Therefore the reconstruction is admissible **only as an initialisation prior that the brain's own
learning then moves**, and the feature's headline claim is not "the neurons are placed correctly" but:

> **THE SEED MUST BE BEATABLE.** Grow from the prior, let learning run, and measure the drift. If the
> brain never moves off the seed, the seed is a hardcode wearing a citation. If it moves and does
> worse, the prior was wrong for this architecture. Both outcomes are informative. Neither is
> available if the numbers are pinned.

That statement is also this feature's reachable red, and it is reachable **from both directions** —
which is what makes it a criterion rather than a hope.

## Relationship to feature 007 (Whole-brain neurogenesis)

007 asks **which regions can grow**. This feature asks **how what grows is ordered**. They compose and
must not duplicate: 007 owns the per-region growth interface; 011 owns the placement, identity and
seeding of the units that interface produces. **If implementing 011 requires a second growth
mechanism, 011 is wrong and must be re-planned** — exactly one growth mechanism survives.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A grown neuron has a place in the cortex (Priority: P1)

Today a neuron the brain grows during sleep is an anonymous entry in a list. After this story it is
born somewhere: it carries a position, it belongs to a cortical layer, and the layer it belongs to
determines what kind of neuron it is allowed to be. An operator inspecting the brain can ask "where
did last night's growth happen, and in which layers" and get an answer that is not fabricated.

**Why this priority**: Everything else in the feature depends on a unit having a place. It is also
the smallest slice that delivers value on its own — a spatially organised population is inspectable
and reportable even before it changes any behaviour.

**Independent Test**: Grow a population from a fixed seed, then read back the per-unit layer and
position. Every unit has both; the layer distribution is not uniform and is not all-one-layer.

**Acceptance Scenarios**:

1. **Given** a brain grown for one sleep cycle, **When** the operator requests the growth report,
   **Then** every newly grown unit carries a layer label and a position, and no unit carries the
   fall-through layer purely because a conversion failed.
2. **Given** the prior's boundary geometry expressed in one unit and the brain's positions expressed
   in another, **When** the two are combined, **Then** the mismatch is refused loudly rather than
   silently assigning every unit to the fall-through layer.
3. **Given** a growth run seeded identically twice, **When** both complete, **Then** the placements
   are identical.

---

### User Story 2 - Sign is a property of the cell, and it never changes (Priority: P1)

Each neuron is born excitatory or inhibitory, decided by the layer it is born into and the type it is
born as. Every connection it makes afterwards carries that sign. A cell cannot excite one partner and
inhibit another, and cannot change sign as a result of learning.

**Why this priority**: Equal-highest with Story 1. This is the single biological invariant the tree
most conspicuously lacks, it is cheap to state, and it is structural — once enforced it constrains
every later mechanism for free. It is also independently valuable: a network obeying it is more
biologically defensible even if placement never improves any measured outcome.

**Independent Test**: Grow a population, then attempt to make one cell's outgoing connections carry
mixed signs. The attempt fails. Separately, run learning to convergence and confirm no cell's sign
has changed.

**Acceptance Scenarios**:

1. **Given** a grown population, **When** every outgoing connection is inspected, **Then** all
   connections from any one cell carry the same sign.
2. **Given** a full learning run, **When** signs are compared before and after, **Then** no cell has
   changed sign, while connection magnitudes have changed.
3. **Given** a cell whose type carries no sign in the source data (for example a non-neuronal cell),
   **When** it is grown, **Then** it is excluded from the signed population rather than defaulting to
   excitatory.

---

### User Story 3 - The prior is beaten, or it is exposed (Priority: P1)

The brain starts from the measured cortical statistics and then learns. An operator can ask, at any
point, how far the brain has moved from the values it started with, and whether moving helped. If the
brain has not moved at all, the report says so — and that is a finding about the seed, not a success.

**Why this priority**: Equal-highest, and it is the reason the other two are permitted at all. Without
it the feature is a hardcoding exercise with a citation attached. It is independently testable because
the drift measurement can run against any seeded population, including a deliberately wrong seed.

**Independent Test**: Run three arms — seeded-from-prior, seeded-randomly, and seeded-from-a-
deliberately-wrong-prior — and compare both the drift and the downstream outcome. The measurement
distinguishes them.

**Acceptance Scenarios**:

1. **Given** a brain seeded from the prior and trained, **When** drift is measured, **Then** it is
   non-zero and the direction of movement is reported per statistic.
2. **Given** a brain whose seeded values are pinned so learning cannot move them, **When** drift is
   measured, **Then** the criterion FAILS and names pinning as the cause.
3. **Given** a brain seeded from the prior that ends up performing worse than the randomly seeded
   control, **When** the comparison is reported, **Then** the criterion FAILS and names the prior as
   the cause rather than reporting success.

---

### User Story 4 - The layered circuit can finally be driven (Priority: P2)

The layered cortical microcircuit stops being an isolated simulation and becomes something the rest
of the brain can send a signal into and read a response out of.

**Why this priority**: Lower than P1 only because Stories 1–3 deliver value without it. But it is the
structural blocker for the feature's long-term point, and it is work the project needs whether or not
any external reconstruction is ever adopted — which is why it is in scope here rather than deferred.

**Independent Test**: Send two different input patterns into the circuit and confirm the responses
differ, with the difference attributable to the input rather than to the seed.

**Acceptance Scenarios**:

1. **Given** the circuit and two distinguishable input patterns, **When** each is presented, **Then**
   the responses differ.
2. **Given** the circuit and one input presented twice, **When** responses are compared, **Then** they
   are identical.
3. **Given** the input port removed, **When** the same test runs, **Then** it fails — confirming the
   test measures the port and not something else.

---

### User Story 5 - Connection strengths start from measured structure, with their bias declared (Priority: P3)

Initial connection strengths are drawn from the measured distribution of real multi-synapse
connections rather than from a uniform guess — separately for excitatory and inhibitory populations,
because in the real tissue strong connections are overwhelmingly an inhibitory phenomenon.

**Why this priority**: Lowest of the five. The measured distribution carries a known population bias
that cannot be removed from the published data, and the quantity it describes is a structural proxy
for a physiological one. It is worth doing, and it is worth doing last.

**Independent Test**: Draw initial strengths for both populations and compare the resulting histograms
against the published distribution and against each other. The inhibitory tail is heavier.

**Acceptance Scenarios**:

1. **Given** seeded strengths, **When** the excitatory and inhibitory distributions are compared,
   **Then** the inhibitory population has the heavier multi-connection tail.
2. **Given** a report of any seeded strength, **When** its provenance is requested, **Then** the
   report states the population it was measured over and the exclusion applied to that population.

---

### Edge Cases

- **A unit is grown at a position outside the measured volume.** The prior covers a finite block of
  tissue; the brain may grow beyond it. The system must place the unit by an explicit stated rule and
  report how many units fell outside, rather than silently clamping them to an edge layer.
- **The fall-through layer absorbs everything.** The prior's layer assignment has a catch-all bucket
  for anything not captured by an explicit boundary. A units-vs-scale mismatch causes *every* unit to
  land there, and the result looks like a valid assignment. This must be detected as a failure.
- **A layer's population is zero or one.** Statistics seeded per layer must behave at an empty layer
  and a single-member layer without dividing by zero or by a near-zero denominator.
- **A layer that the source says has no excitatory cell bodies.** The upper-most layer contains
  inhibitory cells and support cells but no excitatory somata. A seeding rule that assumes every layer
  contains both signs will fabricate a population that does not exist.
- **The source's own categories do not match the circuit's.** The reconstruction distinguishes seven
  regions including white matter; the circuit models four layers with two populations each. The
  mapping is a decision that must be recorded, not an equivalence that can be assumed.
- **Growth continues after seeding.** The prior describes an adult endpoint. Growth continues past it.
  The system must define what the prior means once the population exceeds the measured one.
- **Two units are grown at the same position.** Positions are derived from a finite measured set;
  collisions are possible and must not corrupt the layer assignment or the identity of either unit.

---

## Requirements *(mandatory)*

### Functional Requirements

**Placement and identity**

- **FR-001**: The system MUST assign every newly grown unit a spatial position and a cortical layer at
  the moment of its creation, not as a later annotation pass.
- **FR-002**: The system MUST assign every newly grown neuron a fixed sign (excitatory or inhibitory)
  determined by its layer and type, and MUST exclude from the signed population any type the source
  data does not sign.
- **FR-003**: The system MUST guarantee that all outgoing connections from a single cell carry that
  cell's sign, and MUST prevent any learning process from changing a cell's sign.
- **FR-004**: The system MUST derive cortical depth by the same geometric convention as the source —
  a radial distance from a curvature centre that may lie outside the measured volume — and MUST NOT
  treat any single Cartesian axis as depth.
- **FR-005**: The system MUST state, for every unit grown outside the measured volume, the explicit
  rule by which it was placed, and MUST report the count of such units in the growth report.

**Provenance and the beatable seed**

- **FR-006**: Every value taken from the source MUST be stored in a **single named location** together
  with its units, its provenance, and its published accuracy bound, so that a second dataset can
  replace all of them in one edit.
- **FR-007**: The system MUST NOT present any seeded value as a measurement of this brain. A value
  derived by combining a source value with anything imported from elsewhere MUST be reported as
  derived, naming both inputs.
- **FR-008**: The system MUST permit learning to move every seeded statistic, and MUST measure and
  report the drift from the seeded value per statistic.
- **FR-009**: The system MUST report a seeded population that has not moved from its seed as a
  **failure of the seeding approach**, not as a success.
- **FR-010**: The system MUST support seeding from a randomly initialised control and from a
  deliberately incorrect prior, so the prior's contribution can be attributed rather than assumed.

**Correctness of ingestion**

- **FR-011**: The system MUST reject, loudly and at ingestion, any combination of source geometry and
  brain positions whose length scales are inconsistent, rather than proceeding to a plausible-looking
  assignment.
- **FR-012**: The system MUST use the source's own layer labels rather than re-deriving them, because
  independent re-derivations of the same cells disagree by 10–20% on absolute population sizes while
  agreeing on the excitatory fraction.
- **FR-013**: The system MUST distinguish the fraction of a neuron's *incoming synapses* that are
  excitatory from the fraction of *neurons* that are excitatory, and MUST NOT substitute one for the
  other.
- **FR-014**: The system MUST carry each source value's accuracy bound into any report that quotes it,
  including the substantially higher error on inhibitory measurements.

**Integration**

- **FR-015**: The layered circuit MUST accept an input signal from the rest of the brain and expose a
  response, such that different inputs produce different responses.
- **FR-016**: The system MUST NOT introduce a second layered-population mechanism. The existing
  layered circuit either becomes the substrate the growth process places units into, or it is retired.
- **FR-017**: Placement MUST demonstrably change the brain's own behaviour. A population that is
  correctly placed but whose placement changes no outcome MUST be reported as inert.
- **FR-018**: The system MUST produce identical placements, signs and seeded values for identical
  seeds.

**Explicitly excluded**

- **FR-019**: The system MUST NOT assign propagation delays from this source. The source measures
  geometry and contains no conduction velocity in any form; a delay derived from it would require an
  external velocity relation and MUST be labelled as literature-sourced if ever introduced.
- **FR-020**: The system MUST NOT use the source as a growth schedule, as a fixed target, or as a
  validation oracle (see Assumptions for why each is refused).

### Key Entities

- **Grown unit**: a neuron produced by the growth process. Carries a position, a cortical layer, a
  fixed sign, and a provenance marker distinguishing seeded attributes from learned ones.
- **Cortical layer**: one of the ordered depth regions, plus a catch-all region and a non-cortical
  region. Determines which signs and types may be born there.
- **Prior**: the complete set of externally measured statistics — per-layer population composition,
  excitatory fraction, connection-strength distributions — each with units, provenance and accuracy
  bound. Replaceable as a unit.
- **Drift record**: per statistic, the seeded value, the current value, the distance between them, and
  the direction of movement.
- **Layered circuit**: the existing multi-population cortical model, extended with an input port and
  a response readout.

---

## Success Criteria *(mandatory)*

Each criterion states the input that makes it FAIL. A criterion with no reachable red is deleted
rather than weakened.

- **SC-001 (placement exists)**: Every unit in a grown population carries a layer and a position, and
  the layer distribution spans more than one layer.
  **RED**: grow with the layer assignment disabled — units carry no layer and the criterion fails.

- **SC-002 (the fall-through trap)**: No more than a stated fraction of units land in the catch-all
  layer.
  **RED**: supply the boundary geometry in the wrong length unit — every unit lands in the catch-all
  layer and the criterion fails. *This red must be run, because it is the failure mode that otherwise
  produces a plausible-looking result.*

- **SC-003 (sign is per cell)**: Across a grown population, every cell's outgoing connections carry a
  single sign.
  **RED**: construct one cell with mixed outgoing signs — the criterion fails and names the cell.

- **SC-004 (sign survives learning)**: After a full learning run, no cell has changed sign while
  connection magnitudes have changed.
  **RED**: allow a sign flip during learning — the criterion fails. A run in which *no magnitude*
  changed also fails, so the criterion cannot pass vacuously on a brain that never learned.

- **SC-005 (the seed is beatable — the headline)**: After learning, the drift from the seeded values is
  non-zero, and the seeded arm performs at least as well as the randomly seeded control on the same
  task.
  **RED, direction 1**: pin the seeded values so learning cannot move them — drift is zero and the
  criterion fails, naming pinning.
  **RED, direction 2**: seed from a deliberately incorrect prior — the seeded arm underperforms the
  random control and the criterion fails, naming the prior.
  *Both reds must be constructed and run. A criterion that can only fail one way is not measuring
  what this feature claims.*

- **SC-006 (placement charges behaviour)**: With placement neutralised — positions and layers pinned
  to a single value — the brain's own output differs from the placed run on a majority of ticks.
  **RED**: if the outputs are identical, placement is inert and the criterion fails. *This is the P3
  bar: a census of correctly placed neurons is not evidence.*

- **SC-007 (the circuit is drivable)**: Two distinguishable inputs to the layered circuit produce
  different responses; the same input twice produces identical responses.
  **RED**: remove the input port — the first half fails, confirming the criterion measures the port.

- **SC-008 (provenance survives)**: Every externally sourced value reported anywhere carries its units,
  its origin and its accuracy bound.
  **RED**: strip the bound from one value — the criterion fails and names it.

- **SC-009 (determinism)**: Two runs from the same seed produce byte-identical placements, signs and
  seeded values.
  **RED**: introduce a placement decision that depends on iteration order — the criterion fails.

- **SC-010 (single mechanism)**: Exactly one layered-population mechanism exists in the tree after this
  feature.
  **RED**: add a second — the criterion fails.

- **SC-011 (inhibitory tail)**: Seeded connection strengths give the inhibitory population a heavier
  multi-connection tail than the excitatory population.
  **RED**: seed both from one pooled distribution — the criterion fails.

---

## Assumptions

**On the source's standing**

- The source is treated as **one sample, not ground truth**: one patient, one cubic millimetre, one
  cortical region, from surgical tissue whose clinical context is not published with the data. Every
  published error bar on it is a *reconstruction* error bar; the biological variance is undefined and
  undefinable from a single specimen.
- **It is refused as a growth schedule** because it is an adult terminal snapshot and contains no
  information about the order in which cells arrived or the rules that produced them. It can only be a
  target approached, never a timeline followed.
- **It is refused as a fixed target** because pinning its constants is the hardcoding the project
  forbids.
- **It is refused as a validation oracle** because scoring the brain against it would measure
  conformance to a sample of one.
- The defensible position, and the one this spec takes: **a biased real prior is more informative than
  an unbiased absence of one, provided it is labelled as such and remains replaceable.** Under any
  weaker condition the feature manufactures numbers carrying the authority of a measurement and none
  of its provenance.

**On the mismatch between source and model**

- The source is human temporal cortex; the existing layered circuit is calibrated on other species and
  other cortical areas. Adopting the source's per-layer composition makes the model more like this one
  human's temporal lobe and less like the circuit as validated. **This is accepted deliberately**, and
  it is the reason SC-005 exists — if the change does not earn its place against a control, it is
  reverted.
- The source's category scheme and the circuit's population scheme do not correspond one-to-one. The
  mapping is recorded as a decision with its rationale, not asserted as an equivalence.
- The uppermost layer contains inhibitory and support cells but **no excitatory cell bodies**, which
  contradicts a simplifying assumption in the existing circuit. The source is followed here.

**On what is not attempted**

- Propagation delays are out of scope until a conduction velocity has a cited external source. The
  reconstruction supplies path length to high precision and no velocity whatsoever.
- Connection strengths seeded from the source describe a per-fragment order statistic over a
  population from which one class of strongly multi-synaptic inhibitory cell was excluded by
  construction. This bias is carried in the provenance rather than corrected, because the data needed
  to correct it is not published.
- Absolute per-layer population *sizes* from the source are not used, because independent
  re-derivations disagree by 10–20%. The excitatory *fraction*, which is stable across those same
  re-derivations, is used instead.

**On scope**

- The safety envelope is read-only in this feature.
- Feature 007 owns the per-region growth interface; this feature owns placement and identity of what
  that interface produces. No second growth mechanism is introduced.
- The external data is reference material, not learned state, and therefore lives with the code as
  versioned constants rather than in the brain's durable store.

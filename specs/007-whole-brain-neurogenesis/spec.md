# Feature Specification: Whole-brain neurogenesis -- every region a producer AND a consumer

**Feature Branch**: `007-whole-brain-neurogenesis`

**Created**: 2026-08-03

**Status**: Draft

**Input**: User description: "neurogenesis should be able to grow new live population neurons for all
parts and builds of the brain, each part of the brain can do neurogenesis" and "every part of the
brain should be the consumer and producer to enhance everything together always".

---

## Why this feature is not "add grow() to nine regions"

The obvious reading of the request is: nine regions lack growth, so add growth to nine regions. That
reading is measurably wrong, and building it would make the brain worse.

The single region that already grows was put through a participation test on 2026-08-03, seed 0,
300 ticks, arm A growth on versus arm B growth off, comparing every scalar field of the brain's
state:

```
time-cell width   grow ON: 69    grow OFF: 64
fields that DIFFER: 5 of 93      identical: 88 of 93
```

All 5 differing fields sit on the growth mechanism's own side of the neuter. Three are its own
counters (grown, refused, width). Two are the return value of the very clock being manipulated.
Every downstream measure -- learned latency, reasoning accuracy, dominant drive, perception accuracy,
integration competence -- is byte-identical. **The brain grows and nothing notices.**

Replicating that nine more times produces nine more dormant mechanisms, which P3 counts as a
regression, not a feature. So consumption is the acceptance criterion of this feature and allocation
is merely its precondition. A region that grows units nothing reads has NOT satisfied this spec.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Growth that already happens starts changing what the brain does (Priority: P1)

The brain already grows capacity in two regions and the growth is invisible to everything else.
Before any new region grows anything, the capacity that already exists must demonstrably change a
downstream outcome. This is the whole feature in miniature: if consumption cannot be made to work
where growth already exists and is already earned, it will not work anywhere.

**Why this priority**: It is the only story that can fail for a reason that invalidates the rest. It
also delivers value alone -- a brain whose existing growth finally does something is better than
today's brain, even if no new region ever grows. And it is the smallest slice that proves the
acceptance test itself is honest.

**Independent Test**: Disable growth, run the brain, and compare a downstream outcome against the
growth-enabled run. At least one field that is neither a growth counter nor the grown component's own
output must differ. Fully testable without touching any other region.

**Acceptance Scenarios**:

1. **Given** a brain with growth enabled and a matched brain with growth disabled, run for an equal
   number of ticks from an identical seed, **When** their full observable states are compared,
   **Then** at least one non-tautological downstream field differs, and the difference is stated as
   a number rather than asserted.
2. **Given** the same comparison, **When** the differing fields are listed, **Then** no claim of
   success rests solely on a growth counter or on the grown component's own return value.
3. **Given** growth is disabled, **When** the brain runs, **Then** it still functions -- consumption
   must degrade gracefully to the ungrown width, never require growth to have happened.

---

### User Story 2 - Every region can grow its own population (Priority: P2)

Each region of the brain gains the ability to grow its live population under a gate it has to earn,
rather than inheriting growth by embedding another region's learner or having no growth path at all.

**Why this priority**: This is the literal request, and it is second only because growth without
consumption is the defect this spec exists to avoid. Once Story 1 proves consumption works, breadth
becomes valuable rather than harmful.

**Independent Test**: For each region, run the brain and observe whether that region's population
count increases under load and refuses to increase under an unearned condition. Testable per region
without the other regions being finished.

**Acceptance Scenarios**:

1. **Given** a region under sustained load it cannot serve at its current width, **When** the brain
   runs, **Then** that region's live population increases.
2. **Given** a region under load it CAN already serve, **When** the brain runs, **Then** the region
   refuses to grow, and the refusal is counted rather than silent.
3. **Given** any region on the safety envelope, **When** growth is attempted, **Then** it is refused
   -- the effector envelope does not widen, whatever else grows.
4. **Given** two runs from an identical seed, **When** both complete, **Then** the resulting widths
   are identical; growth must not make construction depend on what a previous run left behind.

---

### User Story 3 - Regions enhance each other, continuously and in both directions (Priority: P3)

Every region both offers what it has grown to the rest of the brain and is itself changed by what
other regions have grown. The relation is mutual and always-on, not a one-way broadcast and not a
one-time handshake at startup.

**Why this priority**: It is the "enhance everything together" half of the request and the most
valuable end state, but it is only meaningful once regions both produce (Story 2) and can be changed
by production (Story 1). Attempting it first would wire a bus with nothing on it.

**Independent Test**: Suppress one region's contribution while leaving it present and running, and
observe whether other regions' outcomes change. Repeated per pair, this measures mutual reach
without requiring all regions to be complete.

**Acceptance Scenarios**:

1. **Given** a region that has grown capacity, **When** its contribution is suppressed but it
   otherwise runs identically, **Then** at least one OTHER region's outcome changes.
2. **Given** the full brain running, **When** the set of regions that both produce and consume is
   counted, **Then** that count is reported as "N of M regions", and regions that only produce are
   reported as not yet satisfying this story.
3. **Given** any cross-region signal, **When** it carries an identity as well as a magnitude,
   **Then** the consumer's outcome depends on the identity -- scrambling identities while preserving
   magnitudes must change something.

---

### Edge Cases

- **A region grows and its consumer is itself.** The most likely way to fake this feature. A region
  reading its own grown capacity satisfies nothing; the consumption test must be cross-boundary.
- **Growth saturates immediately.** Today's grower reaches its ceiling inside the first 100 ticks and
  stays flat for the next 1100. A region that grows once and never again is nearly indistinguishable
  from a region sized correctly at construction; the spec must say whether saturation is success.
- **Growth never fires at all during a test run.** Absence of growth is not evidence that growth is
  impossible -- it may mean the run never posed a demand. A null result must distinguish "refused"
  from "never asked".
- **Two regions grow in response to the same pressure.** Without coordination, one demand can be
  answered twice, spending capacity twice for one need.
- **Grown capacity is consumed, then the producer shrinks or is disabled.** Consumers must not
  break when capacity they had adapted to disappears.
- **Unbounded growth.** A region that always earns growth eventually exhausts memory or slows the
  tick past usability. Growth needs a ceiling that is derived rather than declared.
- **Determinism versus persistence.** Grown width that survives a restart makes the next run's
  starting state depend on the previous run's history. Measured before: making restore unconditional
  produced nondeterministic construction in about 42 of 100 trials.

---

## Requirements *(mandatory)*

### Functional Requirements

**Consumption (the acceptance criterion)**

- **FR-001**: The system MUST demonstrate, for every region claimed complete, that disabling that
  region's growth changes at least one downstream outcome that is neither a growth counter nor the
  grown component's own output.
- **FR-002**: The system MUST name the forbidden observables explicitly and refuse them as evidence.
  For the existing grower these are: the grown count, the refused count, the population width, and
  the grown component's own returned values. Evidence drawn only from these is not evidence, because
  the disable switch gates the only thing that changes them -- the test cannot return another answer.
- **FR-003**: Consumption MUST be continuous rather than one-shot: a consumer that reads grown
  capacity only at construction does not satisfy this feature.
- **FR-004**: The system MUST degrade gracefully when growth is disabled or has not yet occurred --
  no consumer may require that growth has happened.
- **FR-005**: Cross-region signals that carry an identity MUST have consumers whose outcome depends
  on that identity, not only on its magnitude.

**Production**

- **FR-006**: Each region MUST be able to grow its own live population, rather than growing only by
  embedding another region's learner.
- **FR-007**: Growth MUST be earned: a region MUST refuse to grow when its current width already
  serves the demand, and refusals MUST be counted rather than silent.
- **FR-008**: Growth MUST be on by default and always available while the brain runs, not confined
  to a special mode or a maintenance phase.
- **FR-009**: Growth MUST be bounded, and the bound MUST be derived from measured demand or resource
  headroom rather than declared as a fixed number.
- **FR-010**: Growth MUST be refused anywhere on the safety envelope. The effector envelope does not
  widen under this feature.

**Mutuality**

- **FR-011**: Every region MUST be both a producer and a consumer. The system MUST report the count
  of regions satisfying BOTH, and MUST report producer-only regions as not yet complete.
- **FR-012**: The system MUST report an honest inventory that distinguishes a region that defines its
  own growth from one that merely hosts a borrowed mechanism.

**Honesty and governance**

- **FR-013**: Exactly one growth mechanism MUST survive. If a shared mechanism is introduced, the
  existing bespoke path MUST become that mechanism and be retired -- not preserved beside it behind
  a switch.
- **FR-014**: Growth and consumption thresholds MUST be earned or baseline-relative, not fixed
  behaviour-arbitrating constants.
- **FR-015**: Identical seeds MUST produce identical structure. Any persistence of grown state MUST
  NOT make construction depend on what a previous process left on disk.
- **FR-016**: The system MUST NOT describe what it grows in terms stronger than what it builds. The
  existing mechanism is a threshold ladder over one shared accumulator -- there is no per-unit
  membrane potential, leak, or input weight -- and MUST NOT be called a population of spiking
  neurons in any report or document produced by this feature.
- **FR-017**: A region reported as complete MUST have been measured, not inferred. Reachability,
  import, presence in a snapshot, and a moving value are all insufficient.

### Key Entities

- **Region**: A part of the brain. Has a live population that may grow, a demand signal that
  justifies growth, and a set of other regions whose output changes its behaviour.
- **Grown unit**: Capacity added at runtime. Has a state of earned-or-refused, and a consumption
  state of charged-or-dormant. A grown unit that is dormant is a liability, not an asset.
- **Demand signal**: The measured evidence that current width is insufficient. Must be relative to
  the region's own running baseline rather than an absolute threshold.
- **Consumption record**: For each region, which other regions read its production and what outcome
  moved when it was suppressed. This is the feature's ledger and its acceptance evidence.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Disabling growth changes at least 1 downstream outcome that is not a growth counter
  and not the grown component's own output. Today this number is 0 of 88 non-tautological fields.
- **SC-002**: The count of regions that BOTH grow their own population and are read by another
  region rises from its measured baseline of 2 of 15 populated regions, and is reported as
  "N of 15, 2 regions empty and excluded" with the denominator always stated. The reasoning package
  is counted alongside the regions and reported separately, since it is absent from the region map.
- **SC-003**: For every region reported complete, a suppression test shows at least 1 other region's
  outcome changing. Regions failing this are reported as producer-only, not as complete.
- **SC-004**: Two runs from an identical seed produce identical structure in 20 of 20 trials.
- **SC-005**: Growth refuses when unearned: the refusal count is greater than 0 in a run where
  growth also occurs, demonstrating the gate discriminates rather than rubber-stamps.
- **SC-006**: The brain remains usable with growth enabled continuously -- the per-tick cost does not
  regress beyond its measured baseline by more than the project's existing tick budget allows.
- **SC-007**: Exactly 1 growth mechanism exists at completion; the count of parallel growth paths is
  0 of the pre-existing alternatives.
- **SC-008**: Every claim of regional completeness is backed by a recorded measurement that
  reproduces on re-run. Claims without a reproducing measurement count as 0.

---

## Assumptions

- **"All parts of the brain" is RESOLVED, not assumed** (owner decision, 2026-08-03):
  - **Empty regions are EXCLUDED.** Two regions in the map contain zero modules. Populating them
    would mean building two new brain organs from nothing, which is a different feature from giving
    neurogenesis to organs that exist. The denominator for every count in this spec is therefore
    **15 populated regions**, and every report MUST state it as "N of 15, 2 regions empty and
    excluded" rather than silently shrinking the denominator to flatter the ratio. Populating them
    is deferred to its own spec and does not block this one.
  - **The reasoning package is IN SCOPE.** It is absent from the region map, holds 146 modules, and
    contains more growth code than any other package in the tree -- 8 growth-like definitions, none
    ever measured for consumption. It is also where the fluid-intelligence and abstract-reasoning
    work lives, so growth there is the most direct route to "support thinking and reasoning".
    Leaving it out would exempt the single largest body of growth code in the repo from the very
    dormancy check this feature exists to apply.
- "Always" is read as "on by default whenever the brain is running", consistent with growth's
  current opt-out default -- not as "growing on every tick", which saturation makes meaningless.
- Growth saturating at a ceiling is treated as SUCCESS provided refusals are counted, since a
  correctly sized population should stop growing. A region that never saturates is treated as
  suspicious rather than healthy.
- The existing earn-gate behaviour is treated as correct and worth preserving: it currently grows 5
  and refuses 19 at seed 0, which demonstrates discrimination rather than rubber-stamping.
- The safety envelope's existing refusal of growth on arm, joint, motor, actuator and emergency-stop
  seams is treated as correct and out of scope for widening.
- Learning on any live path uses the project's existing forward-mode credit assignment rather than
  backpropagation through time.

## Dependencies and Relationship to Other Work

- **Sibling defect, deliberately NOT solved here.** The brain's global workspace has the identical
  shape of failure in a different subsystem: a broadcast carries both a magnitude and the identity of
  the winning faculty, and the consumer keeps the magnitude and discards the identity. Measured
  2026-08-03: the winning identity genuinely varies across ticks (105 versus 39 across 144
  broadcasts), yet scrambling identities while preserving magnitudes leaves the entire brain state
  byte-identical. That fix is tracked as interconnect work, not here. It is named because both are
  produce-without-consume defects, and solving them separately in incompatible ways is the failure
  mode this note exists to prevent -- FR-005 exists so this spec's consumers are built compatible
  with that fix rather than around it.
- This feature depends on the existing region map and safety-seam refusal already present in the
  tree; it does not introduce them.

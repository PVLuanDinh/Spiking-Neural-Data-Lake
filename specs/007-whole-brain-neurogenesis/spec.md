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

---

## AMENDMENT 2026-08-03: User Story 1 is BLOCKED as written

This amendment is the pre-registered response to research.md Open Question 1. The probe returned
**ONLY_ARTIFICIAL_POSSIBLE**, and tasks.md T013 requires that this be recorded rather than worked
around. Nothing was wired.

### Finding 1: no natural cross-region consumer of the timing estimate exists

Four differently-shaped searches -- by branch site, by produced quantity, by uncertainty consumer,
and an independent AST census over parameter names and return shapes -- put forward **18 candidate
sites**. All 18 fail at least one clause of contracts/C2. The near-misses and their killing clause:

| Candidate | Dies on |
|---|---|
| `cerebellum/synfire_chain.py:170` | magnitude -- see Finding 2 |
| `basal_ganglia/action_gating.py:114` | C2.3 -- `range(1, steps+1)` forces an int, the same defect as `brain.py:1603` |
| `midbrain/multisensory_integration.py:142` | C2.2 -- returns a graded float, no boolean in its public API |
| `neuromodulation/locus_coeruleus.py:144`, `acetylcholine.py:75` | C2.4 -- both branch on salience or content mismatch, about WHAT not WHEN; also 0 live callers |
| `brainstem/autonomous_loop.py:1398` | C2.2 -- blind add into `bias` |
| `cortex/prefrontal_executive.py:276` | C2.1 -- cortex-internal, and quantised anyway |

The structural reason, verified independently rather than taken from the probe:

```
_elapsed_time / _timing_sd / _learned_latency   : 14 occurrences, 1 of 1 files (cortex/brain.py)
state() keys "elapsed_time" / "timing_sd"       : 1 of 1 files across all of src/
```

**There is no reader of any timing quantity outside cortex today.** The one live reader of the
refined value adds it to both non-HOLD actions EQUALLY -- the code's own comment says so -- and it
is then folded by `autonomous_loop.py:698`'s blind add, which research.md already disqualifies as
transport.

### Finding 2: growth does not reduce timing uncertainty -- measured

This is the more consequential result, and it was not anticipated by the spec, the plan, or
specs/003. Decoded sd at tick 30, seed 0, across a 16x range of population width:

```
n_cells    sd
    64     0.827713114
   128     0.850267196
   256     0.801530835
   512     0.804310127
  1024     0.813205575
```

Flat, and non-monotonic -- sd rises at 128 and again at 1024. Growing the population **does not make
the clock more certain**. Corroborating the same fact from the other direction: against an ungrown
twin, growth saturating at 64 -> 68 gives `max |delta sd| = 1.741e-02` where the nearest consumer
separatrix needs 0.107 (6.1x too small) and the live operating point needs 1.157 (66x too small).

The consequence is stark. Most natural consumers of a clock would want its PRECISION -- that is what
a confidence gate, a precision-weighted update, or an ignition threshold reads. Growth does not
improve precision. Growth's only real product is the point ESTIMATE (`max |delta est| = 1.0136`
ticks; est differs on 6 of 60 ticks while sd differs on 47 of 60 as noise), and the point estimate
is exactly what `brain.py:1603`'s `max(1, int(round(el)))` destroys.

So the value chain is: **growth refines an estimate -> an int cast discards the refinement -> the
one downstream reader adds it equally to every option.** Each link is independently broken.

### What this changes

- **SC-001 is unachievable via the time-cell route.** Not because a consumer is missing, but because
  growth's product is not the quantity a consumer would want, and the quantity it does refine is
  quantised away.
- **US1 is BLOCKED as written.** No consumer is wired. Adding a branch that reads timing purely so a
  wider population changes an outcome would satisfy SC-001 and violate P3's honesty guard and
  Standing Invariant 2. The pre-registered test -- would this consumer exist, and read this
  quantity, if the feature had never been written? -- answers no for all 18 candidates.

### What remains achievable, and is NOT blocked

1. **T009's quantiser fix stands on its own merit.** It is a P1 hardcode defect independent of this
   feature: a float refined by the clock is cast to an int before any consumer sees it.
2. **T011's site-selection fix stands.** Independent of consumption.
3. **US2's floor work stands entirely.** Deriving per-region floors and admitting regions to growth
   only on a passing A/B is unaffected -- and the honest outcome for a region with no derivable
   floor was already an accepted result.
4. **The deletions and the M25 gate (T020-T024) stand.**
5. **Retargeting US1 is open.** `time_cells` was chosen as the P1 producer because it is the only
   grower that EARNS. The other real producer -- `hippocampus/growing_learner.py`, embedded by 4
   regions -- was never probed as a producer for consumption. Its grown prototypes may reach a
   consumer that the timing estimate cannot. That is a new probe, not a re-run of this one.

### The honest headline

**A brain that grows more timing cells does not tell time better.** It shifts its point estimate,
which nothing downstream is allowed to see. That is worth knowing before building a shared growth
interface for fifteen regions, and it is the result this probe existed to find.

---

## AMENDMENT 2026-08-03 (second): the retarget is DEAD too, and both producers have now failed

The hippocampus retarget named in the previous amendment was probed to the same standard. Verdict
**RETARGET_DEAD**, on three independent kill paths, any one sufficient.

### Kill 1: growth does not improve the learner on the only distribution that matters

On 130 percepts captured from a live brain over 260 ticks, replayed through capped learners
(held-out reconstruction fidelity, label-free):

```
units      1          3          5        6..40
fid    0.991838   0.992471   0.991724   0.991724 (saturated)
```

Flat within 6.3e-4, **non-monotonic**, and the best value is at 3 units while it actually grows to
5. One prototype already codes every live percept at cosine 0.9918 -- there is no variance left for
growth to exploit. **This is the same shape that killed the time cells** (sd flat and non-monotonic
across a 16x width range). Two different producers, two different quantities, identical result.

The synthetic "improvement" is a counting identity. In the separable regime the headline accuracy
equals the closed form `min(cap,K)/K` to 2.2e-16, and reproduces **bit-identically with all
learning disabled** (`lr=0.00`): learning contributes exactly 0.0 of the rise. Even uniform random
vectors improve 0.0500 -> 0.4809 on count alone. In the overlapping regime, where an honest residue
does exist, learning's share is about 7 percent of the lift.

This is the specs/003 defect in a new costume, and it is worth naming precisely: **a metric that
cannot decrease when units are added is not measuring learning.** My own first pass at this fell
into it -- a mean-best-cosine metric rose monotonically 0.216 -> 0.332 with width, which is
guaranteed since a max over a larger set cannot shrink. Re-measured with a metric that penalises
over-segmentation (held-out cluster recovery), the same runs invert completely:

```
vigilance  n_units  cluster-recovery F1
   0.50        6.0      1.000000
   0.85        9.8      0.963978    <- the shipped default already over-grows
   0.96      283.2      0.193112
   0.99      300.0      0.097384
```

Growth makes it monotonically WORSE. (Synthetic, K known -- held loosely, and the live measurement
above is the one that decides.)

### Kill 2: the growth gate is a hardcoded constant, at every site

Contract C1.1 requires a halting condition computed from the region's own measured dispersion.
`growing_learner.py:88` is `if best_id < 0 or best_sim < self.vigilance:  # novel -> neurogenesis`,
and every vigilance in the tree is a literal:

```
plasticity/lifelong_consolidation.py:92      0.9
neuromodulation/affect_integrator.py:85      0.82
cortex/brain.py:1331                         0.86
hippocampus/growing_learner.py:59            0.85   (default)
substrate/similarity_sized_growth.py:144     1.0
reasoning/schema_induction.py:94             0.7
```

**0 of 31 `vigilance =` occurrences derive from stdev, variance or dispersion.** There is also no
refusal counter, so C1.3 fails as well. The contract's own disposition applies: `admitted = False`,
recorded as "no derivable floor", which is a passing outcome rather than a defect to work around.

### Kill 3: the one real consumer is permutation-invariant

`brainstem/autonomous_loop.py`'s `self.q[unit]` passes C2(a), (b) and (d) -- genuinely cross-region,
terminates in a motor action token, documented long before specs/007. It fails C2(c) bit-exactly:
consistently relabelling unit ids while leaving the partition unchanged produces **0 of 300**
differing actions, reproduced on 10 of 10 seeds by two independent probes, with an instrument check
confirming the relabelling shim fired. C2.4 names this as its own disqualifier.

Worse for the premise: at the repo default exploration, a **3-unit cap beats full growth by +43.9
cells**, and full growth beats a random-id shuffle on only 5 of 10 seeds. The consumer does not just
ignore growth; more growth is actively worse.

---

## CORRECTION to contract C1.7: the safety-seam refusal is NOT enforced

`contracts/growth-consumption.md` C1.7 states that any seam matching arm / dof / joint / motor /
actuator / effector / estop "refuses unconditionally". **That overstates the tree, and the error is
mine.** Measured:

```
is_safety_seam('arm.learner')             = True
is_safety_seam('autonomous_loop.learner') = False     <- the SAME object, different name string
```

It is a substring match on the name, so the same learner is safety-critical or not depending on
which label is passed. And it is called from exactly 1 site -- `growable.py:139`, inside
`register_seam` -- which nothing in the tree ever calls. **The guard cannot refuse anything, because
no growth path registers a seam.**

**What this does NOT mean**: the arm is not less safe. Arm safety comes from `ArmSim.apply()`'s gate
and the reflex E-STOP, both live, tested and untouched by any of this. What is unenforced is the
growth-specific seam check, which is unreachable for the same reason `GrowthCoordinator` is --
a registry that is never registered.

**Task T002 is corrected accordingly.** As written it would have called `is_safety_seam` directly,
received `True`, and reported the envelope protected -- a green result from a function nobody calls.
That is precisely the verify-the-verifier failure this feature exists to catch, and it was sitting
in my own task list.

## Where this leaves specs/007

**Both measured producers have now failed, for different reasons.** Time cells: growth does not
improve the quantity. GrowingLearner: growth improves nothing live, degrades structure recovery, is
gated by a constant, and its only cross-region consumer is permutation-invariant.

That is not a reason to weaken the acceptance criterion. It is a finding about the tree: **no
existing growth mechanism in this brain produces something a consumer can use.** The feature as
specified is not blocked by a missing wire; it is blocked by producers whose growth does not carry
information.

US2's floor programme and the standalone P1 fixes (T009, T011) remain valid and unaffected. What
does NOT follow is that a third producer should be probed next -- two failures with the same shape
suggest measuring WHY growth carries no information in this architecture before hunting a third
candidate.

---

## AMENDMENT 2026-08-03 (third): the founding baseline was a SEED-0 ARTIFACT, and US1 is UNBLOCKED

**The `0 of 88 non-tautological fields` figure that motivates this entire specification is wrong.**
It was measured at seed 0 and never swept. Ten seeds, growth ON vs OFF, 300 ticks, hermetic, matched
A/A control at 0 throughout:

```
init 64 (shipped)  SC-001 passes 7 of 10 seeds   per-seed [0,0,2,1,2,1,1,2,2,0]  mean 1.10
init 32            SC-001 passes 9 of 10 seeds   per-seed [1,1,2,1,2,1,1,2,2,0]  mean 1.30
fields moved: acc_conflict (6 of 10 seeds), learned_latency (5 of 10 seeds)
```

**Growth already changes non-tautological downstream outcomes on 7 of 10 seeds at the shipped
width.** "The brain grows and nothing notices" is false as a general statement; it is true only at
seed 0, which is the seed every measurement in this feature happened to use.

### What this reverses

- **SC-001 is NOT unachievable. It already passes on 7 of 10 seeds.** The correct baseline is a
  distribution, not a point: `mean 1.10 non-tautological fields, passing on 7 of 10 seeds`.
- **US1 is UNBLOCKED.** The second amendment recorded it BLOCKED because no natural cross-region
  consumer of the timing estimate could be found. That search was sound and its 18-of-18 result
  stands, but its premise -- that nothing downstream moves -- was seed-0. `acc_conflict` and
  `learned_latency` do move, on most seeds, through the existing path.
- **The "growth is dormant under P3" claim is withdrawn.** A mechanism that changes a downstream
  outcome on 7 of 10 seeds is charged, not dormant. P3 is satisfied here and was never violated.

### What still stands

- **`n_cells=64` at `cortex/brain.py:757` is still a P1 defect** and now a measured one: initialising
  at 32 raises the pass rate from 7 of 10 to 9 of 10 seeds and the mean from 1.10 to 1.30. The
  effect is real and MODEST -- an earlier single-seed run suggested 0 -> 1, which overstated it.
  The initial width should be earned rather than declared, exactly as the growth increments are.
- **The six coarsening boundaries are all still real.** They were each measured independently.
  What is withdrawn is the inference that they are why growth appears inert -- growth is not inert.
- **US2's per-region floor programme is unaffected.**

### How this happened, and the rule it violated

The constitution's Standing Invariant 5 is "verify the verifier", and the project's own standing
rule is to sweep seeds before letting a numeric result become load-bearing. I measured once at seed
0, wrote the number into spec.md, plan.md, research.md, quickstart.md, tasks.md and the contract,
and built three amendments on top of it. Every downstream artifact inherited a point estimate as if
it were a property of the brain.

The tell was available and ignored: a baseline of exactly 0 is the kind of clean number that should
invite a second seed rather than a specification.

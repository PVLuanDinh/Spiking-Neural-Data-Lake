# Feature Specification: The Unified Brain Arc — self-constructing, lake-resident substrate

> Policy authority: `docs/POLICIES.md` (run `python -m snn_data_lake.substrate.policy_manifest --audit` for the enforcement manifest).

**Feature Branch**: `022-unified-brain-arc`

**Created**: 2026-08-19

**Status**: Draft

**Input**: Six owner directives (2026-08-19), unified: (a) the brain builds itself from a single-neuron seed through an H01-ordered multi-region structure toward and past human scale; (b) neurons live in the ONE durable lake, queried on demand, remembering everything; (c) learning is a COMPOSED palette of local rules — the composition, not any single rule, is the brain-replication claim; (d) the brain improves itself through an owner-gated proposer/evaluator loop; (e) the finished system is a full representation of the human brain and consciousness — every region charged, every consciousness mechanism instrumented with a dissociation; (f) where research finds no existing method, the method is INVENTED under P8 with its full debt. The converged design (3 critic rounds + frozen verification, 12→8→4→0 blockers) is `SYNTHESIS.md` in this directory — the evidence base for every requirement below; where this spec and that document disagree, this spec wins and the disagreement is a defect to fix here.

## Clarifications

### Session 2026-08-19

- Q: Which brain regions should the growth system be able to GROW from the seed, versus remain hand-built scaffold? → A: The pinned 10 (cortex, hippocampus, thalamus, basal ganglia, cerebellum, amygdala, hypothalamus, midbrain, epithalamus, claustrum); scaffold = brainstem (envelope), medulla, pons, olfactory, insula. OWNER-RATIFIED.
- Q: Do you re-authorize the sensor hardware lane absorbed from specs/014, under its original two GO/NO-GO conditions? → A: Re-authorized, 014's two conditions carried verbatim as GO/NO-GO gates. OD-1 RESOLVED.
- Q: Which engine should generate candidate self-modification proposals in the S3 proposer loop? → A: "The brain should not rely on outside model other than itself" — the proposer is INTERNAL-ONLY: no external LLM, API, or third-party model in the loop. Candidate generation = declared mutation operators + the transform registry + the brain's own faculties as they mature. This supersedes SYNTHESIS.md L6's proposer-model-tiering note (spec wins).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The brain's parts talk, and the lake can prove what it holds (Priority: P1, stage S0)

The owner runs the daemon and, for the first time, the brain's faculties exchange rich typed signals instead of one scalar: serotonin's patience reaches a consumer, interoceptive state reaches the workspace, and the P6 faculty-to-faculty count — the access-consciousness instrument — moves off its honest zero. The durable lake gains an engine: indexed, delta-based, crash-provable, so "replay from tick T" is a lookup and a restart provably restores the brain.

**Why this priority**: Every later stage consumes the typed-signal bus and the lake engine. It is also the measured worst deficit: 0 faculty-to-faculty reads, full-file lake scans, 97.6% of the lake spent on repeated snapshots.

**Independent Test**: Neuter one declared port's payload (pinned-constant substitution) and show the consumer's output changes when live and stops changing when neutered; kill the daemon mid-batch and show recovery to the published watermark; delete the derived index and rebuild it from the lake with logical-dump equality.

**Acceptance Scenarios**:

1. **Given** the typed wiring manifest with JoinPolicy declared per input port, **When** two producers feed one summing join without a declared policy, **Then** the manifest gate goes red.
2. **Given** the lake engine with sealed segments, **When** the last segment is truncated at a random byte offset (injected torn write, with an uncorrupted positive-control run in the same command), **Then** recovery halts at the last valid record and the brain restarts from the watermark.
3. **Given** the region charge gate sourced from live reachability plus port-edge charge attribution, **When** a region's only wiring is a registration with no charged consumer, **Then** that region still counts UNWIRED.
4. **Given** the ignition actuator, **When** theta is commanded across a sweep, **Then** read-back asserts every commanded value took effect (positive control), and the ignition scorer's antimode and fitted-logistic inflection agree.

---

### User Story 2 - A seed grows a multi-region brain that learns by composed rules (Priority: P2, stage S1)

The owner seeds the substrate with a single unit. The growth policy — one shared per-node program reading local organizer fields — emits region identities over the pinned growable set (cortex, hippocampus, thalamus, basal ganglia, cerebellum, amygdala, hypothalamus, midbrain, epithalamus, claustrum), grows a functioning micro-brain, and stops by a measured criterion rather than a step count. Every learning rule in the composed palette carries per-rule neuter attribution: the owner can see what each rule contributes.

**Why this priority**: Self-construction is directive (a); it requires S0's bus and engine but nothing from later stages. The random-prior control arm exists BEFORE the H01 treatment arm (control-first, per standing policy).

**Independent Test**: Grow from seed with the random prior; assert region-coverage over the growable set rises from 0; neuter each palette rule in turn and show attributed outcome changes; run the growth past its measured optimum and show the stopping criterion trips (its reachable red).

**Acceptance Scenarios**:

1. **Given** a single-unit seed and organizer fields, **When** growth runs, **Then** grown units carry (neuromere, zone) region labels validated against `brain_atlas.region_of` semantics, and the scrambled-field spatial null scores measurably worse (same command).
2. **Given** the composed palette on one synapse population, **When** rule A is neutered, **Then** the attribution ledger shows A's contribution isolated (clause-6: all live readers enumerated, isolated counterfactual reported).
3. **Given** the WM attractor build, **When** K is derived from the population's own STP constants, **Then** measured retention equals round(K)±1, the tau_f sweep moves K monotonically, and the frozen-STP null REFUSES to emit K.
4. **Given** the sleep phase, **When** replay-only / Benna-Fusi-only / structural-clock-only are neutered in the same command, **Then** per-capability retention deltas separate, and the arm is quiesced with the E-STOP latch persisting across the phase boundary.

---

### User Story 3 - The H01 prior orders the cortex, and the population outgrows RAM (Priority: P3, stage S2)

The owner ingests the H01 c3 graph (~16k signed neurons) as a beatable prior: the seeded arm must beat both a random and a deliberately-wrong prior, and learning must move the brain off the seed. The substrate's population crosses the derived binding width and neurons page from the lake by activation, within an asserted per-tick rounds budget.

**Why this priority**: Directive (a)'s H01 stage and directive (b)'s paging both become measurable only at this scale; both consume S1's growth and S0's engine.

**Independent Test**: The three-arm drift harness (seeded/random/wrong) separates with its two reds; a spike arriving at a paged-out neuron is provably not dropped (constructed red); paging beats forced-total-residency at the derived width or the paging layer is deleted per its own null clause.

**Acceptance Scenarios**:

1. **Given** the H01 ingest, **When** units are joined base↔agglo instead of neuron_id↔agglo_seg, **Then** the construction-time gate refuses (silently-empty-graph trap made loud), and nm-scale coordinates failing the unit gate refuse ingest.
2. **Given** lake-resident neurons at the binding width, **When** a tick's fetch set is assembled, **Then** it issues as one async batch within the integer rounds budget, and the budget assert can go red (constructed overload).
3. **Given** the cerebellar tissue built at S1, **When** the adopted rule runs with the Bouvier control, **Then** the green/red pair reproduces, and one global window constant across two channels with different measured latencies FAILS (sign-inversion red).

---

### User Story 4 - The brain proposes its own improvements, and its instruments say what it is (Priority: P4, stage S3)

The proposer/evaluator loop runs: candidate diffs against declared mutation surfaces, evaluated by a hash-pinned cascade on harness-only evidence, archived (never champion-replaced), with owner approval required for any diff touching the evaluator, invariants, surface declaration, or safety envelope. The consciousness instrument battery reports what the brain measurably has — ignition rate against published bands, metacognitive calibration above derived floors, integration regime arms separated — and never claims what it cannot measure.

**Why this priority**: Directive (d) and the instrument half of (e); consumes everything before it.

**Independent Test**: A candidate diff touching evaluator bytes is refused and routed to the owner; a fabricated candidate-emitted score is ignored (harness-only evidence); the ANNECS growth curve reports jointly with per-goal time-to-master; each L7 instrument's red fires on its constructed input.

**Acceptance Scenarios**:

1. **Given** the proposer with ≥2 registry entries, **When** a lineage fails three cadence hits, **Then** the circuit-breaker state is a cockpit row, never a silent latch.
2. **Given** the metacognition instrument, **When** label entropy is below the derived floor, **Then** it REFUSES to emit calibration; **When** the second-order layer is lesioned, **Then** first-order performance is preserved while calibration collapses (the relay-proof double dissociation).
3. **Given** the report channel, **When** one named intermediate is perturbed, **Then** the report changes, and the scrambled-intermediate control (same command) does not.

---

### Edge Cases

- A spike arrives at a neuron that is paged out mid-delivery (the L2 constructed red; bounded staleness, never silent drop).
- Growth proposes a unit in a fixed-scaffold region (refused; scaffold set is pinned and disclosed).
- The free-threaded interpreter silently re-enables the GIL via an unsafe extension (startup assert fails hard).
- The lake's IMMORTAL append rate exceeds its ceiling (red, before disk pressure becomes the detector).
- Both workspace instantiations submit coalitions during cutover (instance resolution is an S0 deliverable; the composite is declared before any ignition number).
- The daemon restarts mid-stage from a commit push (watermark handoff; the lake carries the brain across — P4b is the cutover mechanism).
- A candidate self-modification fabricates its own passing log (scored only on harness-produced evidence; the fabrication is the discard reason, recorded).

## Requirements *(mandatory)*

### Functional Requirements

**L0 — Lake engine**
- **FR-001**: The lake MUST be served by derived, provably-rebuildable indexes: delete-and-rebuild yields logical-dump equality, verified with an injected-torn-write positive control.
- **FR-002**: Persistence MUST be delta-based with the dirty set derived from spikes; full snapshots exist only as compaction artifacts; every record carries magic+version+length+CRC; commit is watermark-published, never file-length.
- **FR-003**: Record classes MUST be labeled IMMORTAL or COMPACTABLE at introduction; IMMORTAL carries a bytes/tick ceiling with a reachable red; growth deltas are immortal as RECONSTRUCTIBLE, never as whole-substrate blobs.
- **FR-004**: The lake writer MUST be a consumer on the same ring every faculty reads (no signal exists the lake did not see), with a cross-process file lock on the writer path.

**L1 / L1.5 — Kernel and boundary**
- **FR-005**: The tick MUST be two-phase (propagate/deliver/rotate) with fixed partition count P per lineage, integer/fixed-point synaptic accumulation, and byte-identical spike trains across thread counts at fixed P (broken-control arm in the same command).
- **FR-006**: Startup MUST assert the GIL is genuinely disabled and fail hard otherwise.
- **FR-007**: Snapshots MUST capture per-instance RNG stream positions; restore MUST assert stream-position equality (G5), as an S0 exit gate.
- **FR-008**: The afferent/efferent boundary MUST expose declared TypedPorts per modality from the pinned modality inventory, an action-out port with an efference-copy field, a report channel, and a stimulus port with an EpisodeDriver owning trial schedules; each side has a neuter-based exit gate.
- **FR-009**: The safety envelope runs ON the substrate tick, monotone, through all phase transitions; the arm is quiesced and the effector port closed during sleep; the E-STOP latch persists across phases. Envelope files change only with explicit owner approval.

**L2 — Paging**
- **FR-010**: Neurons MUST be pageable from the lake by activation within an integer per-tick rounds budget (asserted, red-capable), with block-aligned unit records, a pinned hot set, and a delta tier for growth; the memory hierarchy is three explicit tiers (device/host/lake).
- **FR-011**: A spike arriving at a paged-out neuron MUST NOT be silently dropped (constructed red exists before the mechanism); if paging never beats forced-total residency at the derived binding width, the layer is DELETED (null clause).

**L3 — Typed composition**
- **FR-012**: Every inter-faculty signal MUST flow through a TypedPort (name, shape, dtype, precision, unit, encoding, kind, region) in a lake-committed wiring manifest with CI type-checking; `port.region` MUST equal `brain_atlas.region_of(declaring_module)`.
- **FR-013**: Every input port MUST declare a JoinPolicy and every output port a DeliveryPolicy; silent fan-in summation is a manifest red.
- **FR-014**: Port consumption MUST be mechanically charge-measurable: pinned-payload neuters per port, cleanup-margin records per consumption, and replay-based A/B as the standing harness.
- **FR-015**: The region charge gate MUST source from live reachability PLUS port-edge charge attribution (registration alone can never green it); exemptions carry date+owner+expiry with a ratcheting count ceiling; the S0 exit is an integer floor of regions charged.
- **FR-016**: The workspace MUST be resolved to one instance (or one declared composite) before any ignition number is reported; coalition records carry proposer ids, two-component activation, and typed channels whose producers are declared.
- **FR-017**: Recall MUST satisfy `recall(cue) → k episodes with margins`, beating a scrambled-cue control from a degraded cue (margin derived from measured similarity distributions).

**L4 — Composed learning**
- **FR-018**: Every learning rule MUST carry a GRADIENT_CLAIM declaration, declared read/write sets, and per-rule neuter attribution under the composition arbitration contract; a rule contributing nothing measurable in its declared scope is retired (P3).
- **FR-019**: The palette MUST include the LPL spine (erratum-corrected), three-factor extensions with declared producers, ESPP for temporal streams (same/different oracle named), inhibitory STDP, structural plasticity on the two-clock design, and the cerebellar rule ON BUILT TISSUE (granule/PF/Purkinje + pn relay; MLI sign-legal pathway; window identity pinned to measured channel latency).
- **FR-020**: WM capacity K MUST be derived (K = T_max/t_s from the built attractor population's own constants), never picked; the frozen-STP null refuses to emit K; nothing in src/ reads the theta/gamma ratio as a K source (permanent static check).

**L5 / L5.5 — Development and sleep**
- **FR-021**: Growth MUST run from a single-unit seed over the PINNED growable set (10 regions, OWNER-RATIFIED 2026-08-19) via one shared per-node program reading only local fields (locality mask raises on non-local access), emitting (region, cell-type) in two stages with two discretization regimes; fixed-scaffold regions are pinned and disclosed.
- **FR-022**: The H01 prior MUST be beatable: three-arm harness (seeded/random/wrong) with two reds; c3 pinned; unit gate refuses nm-inversions; layer computed, not read; CC-BY attribution on PROVENANCE.
- **FR-023**: The growth stopping criterion MUST derive from |LP| and charge, with an arm independent of the executive path, and a reachable red (growth past a measured optimum trips it).
- **FR-024**: Sleep MUST own consolidation (replay + Benna-Fusi actually reached), the structural slow clock, and lake compaction, with per-capability neuters and a phase-level aggregate control.

**L6 — Self-improvement**
- **FR-025**: Proposals MUST be diffs against named parents on declared mutation surfaces, evaluated by a staged cascade on harness-only evidence with resource terms, archived with MAP-Elites descriptors; the evaluator, invariant set, surface declaration, and envelope are hash-pinned outside the surface; diffs touching them require owner approval. **The proposer is INTERNAL-ONLY (owner, 2026-08-19): no external LLM, API, or third-party model generates candidates — generation is declared mutation operators + the transform registry + the brain's own faculties as they mature (P4: one brain; the brain relies on no model other than itself).**
- **FR-026**: The curriculum MUST consume curiosity: goals as (embedding, achievement-function), |LP| utility with an exploration floor, the lake as goal history with a vector index; ANNECS reported jointly with per-goal time-to-master; the open-endedness observer frozen in advance.
- **FR-027**: The incumbent objective `coverage_reward` carries its two measured properties (saturation curve reported; `_covered` lake-persisted at S0) and is joined, never silently replaced, by |LP| goals.

**L7 — Consciousness instruments**
- **FR-028**: Every consciousness mechanism MUST carry a dissociation instrument with a reachable red: ignition (actuator landed whole, subject resolved, dual-derived gate, published-band checks — a 100% ignition rate is a saturated detector), metacognition (derived floors, relay-proof double dissociation), integration (ported perturb-and-compress, staged on G5, never called Phi), attention schema (G3/G4 preconditions, train-without protocol), agency (efference-copy mismatch), binding (membership-matched primary null), report-order (perturbed-intermediate red).
- **FR-029**: The system MUST report only what it measures: access structure claimed and measured; phenomenal experience never claimed from a passing criterion (P8 sentience clause).

**Cross-cutting**
- **FR-030**: Every module lands in governed `src/snn_data_lake/` registration-complete in its own commit; every commit reaches `main` with the full gate green; cutover follows §3.5 of SYNTHESIS.md (shutdown barrier, watermark handoff, per-record-tag migration dispositions with round-trip reds over live records).
- **FR-031**: The eight P8 inventions live in `docs/INVENTION_REGISTER.md` (new budgeted file), each with RED, external control, per-item attribution, unprompted disclosure, deletion-on-null, and clause-6 anti-rigging; governance text lands only in named budgeted homes.

### Key Entities

- **TypedPort / WiringManifest**: the named, typed, region-tagged signal surface; the manifest is the anatomical inventory and a lake-committed artifact.
- **Lake record classes**: unit blocks, synapse deltas, episodes, coalitions, goals, proposals, snapshots — each IMMORTAL or COMPACTABLE, versioned, CRC-framed.
- **Region label (neuromere, zone)**: the prosomeric coordinate emitted by growth and consumed by placement and the connection policy.
- **Goal**: (embedding, achievement-function) pair with |LP| history in the lake.
- **Proposal**: diff + parent pointer + score dict + artifact hash, archived in the lake.
- **Instrument**: a scorer + actuator + declared subject + reachable red + external control, reported on the cockpit with provenance.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001** (S0): P6 faculty-to-faculty reads > 0 on genuinely charged signals (neuter-verified), from a measured baseline of 0; the integer floor of charged regions is met; p6_gate reads the manifest in the same commit that lands it.
- **SC-002** (S0): Lake rebuild-from-log proves logical-dump equality; kill-mid-batch recovers to the watermark; wake-tick p50/p95 hold as ratchets against the pinned-host baseline with the duty cycle reported.
- **SC-003** (S0): G5 green — identical spike trains from snapshot-restore with RNG stream positions asserted; ignition baseline recorded with actuator read-back at 100% of commanded values.
- **SC-004** (S1): A single-unit seed grows a multi-region micro-brain: region-coverage over the pinned growable set rises from 0, boundary-width beats the gradient-only control, and every palette rule shows nonzero isolated attribution on at least one outcome.
- **SC-005** (S1): Derived K equals held items ±1 on the built WM population; sleep-dependent retention beats the sleep-neutered control with per-capability separation.
- **SC-006** (S2): The seeded arm beats random AND wrong priors on the drift harness; learning moves the brain off the seed (the beatable-prior red can fire in both directions).
- **SC-007** (S2): At the derived binding width, paging meets its integer rounds budget and beats forced-total residency — or the layer is deleted and the deletion recorded (both outcomes satisfy this criterion; silence satisfies nothing).
- **SC-008** (S3): The proposer proposes >1 candidate per cadence with the circuit-breaker surfaced; a surface-violating diff is refused and routed to the owner; ANNECS + time-to-master reported jointly.
- **SC-009** (S3): Every L7 instrument has fired its constructed red at least once (the reds are reachable in fact, not in prose), and the cockpit's consciousness rows report only measured quantities.
- **SC-010** (always): Every commit on `main` is full-gate green; the anatomical charge gate never regresses (exemption ceiling only tightens); the safety envelope hash is unchanged except by owner-approved commits.

## Assumptions

- The pinned measurement host is the owner's current machine (RTX 5070, D: NVMe at 4.0 GB/s measured); all performance exits are ratchets against baselines recorded there, not absolute numbers.
- The H01 c3 agglomeration is the pinned prior source; the ~16k signed-neuron graph is the seed target; the cheapest first slice (<7 GB) is ingested before the full 30.6 GB edge list.
- The live daemon keeps running throughout; the lake carries the brain across every stage (P4b is the cutover mechanism); one daemon, one lake, one brain at all times (P4).
- Free-threaded parallelism is budgeted at 2–3×, not N×; the GPU serves widths past the measured crossover only.
- specs/007 (whole-brain neurogenesis interface) and specs/011 (H01 placement prior) are COMPOSED by this feature, not duplicated; exactly one growth mechanism survives (011's own clause). specs/014 is ABSORBED into L1.5.
- Oscillation-dependent derivations that assume time-continuous dynamics are refused-with-reason where they do not transfer to the T=1 tick (theta/gamma K is the recorded example).

## Dependencies / Owner decisions required

- **OD-1 — RESOLVED (owner, 2026-08-19)**: the absorbed sensor hardware lane is RE-AUTHORIZED with specs/014's two conditions carried verbatim as GO/NO-GO gates before the hardware lane activates (the SIMULATED feed needed no new authorization).
- **OD-2**: Promotion of any self-modification diff touching the evaluator, invariants, mutation-surface declaration, or safety envelope requires per-instance owner approval (FR-025).
- **OD-3**: The POLICIES.md text additions (determinism contract, retention rule, region charge gate) land only after the e-prop consolidation debt is paid to free doc-budget headroom — sequenced as this feature's first governance commit.

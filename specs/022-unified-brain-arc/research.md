# Phase 0 Research: consolidated decisions

All unknowns were resolved BEFORE this plan by the session's research program (5 webgrab workflows + 1 gap-closure workflow, 40+ agents, every claim quote-backed) and 3 adversarial critic rounds. Full evidence: [SYNTHESIS.md](SYNTHESIS.md) §§1, 5, and the per-area distillates it cites. This file consolidates the DECISIONS in the required format. No NEEDS CLARIFICATION markers exist in the spec; the three open items are owner decisions (spec OD-1..OD-3), not research gaps.

## D1 — Lake engine polarity
- **Decision**: WiscKey key-value separation — the append-only lake is the value log and sole source of truth; all indexes derived and provably rebuildable (delete → rebuild → logical-dump equality, torn-write positive control).
- **Rationale**: Preserves P4b verbatim while fixing the measured engine deficits (no addressing, full-file scans, per-record fsync 21× slow, 97.6% snapshot bloat).
- **Alternatives considered**: In-place B-tree store as primary (rejected: the lake must never be rewritten in place); mmap-as-tiering (rejected: kernel eviction is blind to activation — the paging policy IS the feature).

## D2 — Segment format
- **Decision**: Sealed immutable Arrow IPC *file* segments (footer-indexed), magic+version+length+CRC per record, watermark-published commit.
- **Rationale**: Footer turns "replay from tick T" into a lookup; pyarrow already declared (`[lake]`); immutability buys lock-free parallel reads under free-threading.
- **Alternatives**: Raw byte stream with side index (rejected: recovery and random access both hand-rolled); keeping BinLog framing (rejected: measured incapabilities; ported capabilities listed in the §6 register).

## D3 — Index engine (deferred to S0 measurement, criteria pinned)
- **Decision**: Choose LMDB-class MVCC vs LSM at S0 by measured write shape; criteria pinned NOW: single-writer/N-reader discipline, cross-process readers, short-lived snapshots, epoch-fenced compaction, and the chosen amplification trade written down.
- **Rationale**: You cannot optimize read/write/space amplification simultaneously; the honest choice needs this substrate's measured churn, which exists only after the delta tier runs.
- **Alternatives**: Deciding now (rejected: would pick a value where a measurement belongs).

## D4 — Delta persistence
- **Decision**: Dirty set DERIVED FROM SPIKES (only synapses whose pre/post fired can change under local rules); quantize-then-compare-then-append; snapshots are compaction artifacts only; growth deltas IMMORTAL-as-reconstructible.
- **Rationale**: The strongest available invariant for a spiking substrate; kills the measured 97.6%-snapshot pathology by construction.
- **Alternatives**: Diff-discovery (rejected: weaker and costlier than the spike-derived set); whole-state snapshots per save (rejected: the measured pathology).

## D5 — Execution kernel
- **Decision**: Hybrid by construction — time-stepped neuron integration, event-driven synaptic delivery; two-phase tick (propagate/deliver/rotate); bundles keyed (pre, partition, delay); per-faculty measured density thresholds with mode inversion allowed.
- **Rationale**: The event/clock crossover INVERTS with density (measured); sparsity has two independent axes; Brian's per-mechanism split is the production norm.
- **Alternatives**: Pure event-driven (rejected: control cost dominates at high activity — measured power INCREASE at high sparsity); pure clock-driven (rejected: wastes the sparse regime the seed brain lives in).

## D6 — Determinism
- **Decision**: Fixed partition count P per lineage (NEST virtual-process form; partition owns delivery+plasticity); integer/fixed-point synaptic accumulation (Kulisch family; fractional bits MEASURED vs FP64); CUB `run_to_run` floor on GPU; FMA/fast-math off; G5 snapshot-replay with RNG stream positions as an S0 gate.
- **Rationale**: Every L7 instrument consumes replay-equality, not just thread-invariance; integer addition is order-invariant outright.
- **Alternatives (rejected, recorded)**: SpiNNaker best-effort delivery (drops + re-injects); bare OpenMP/library reductions (non-deterministic); Kahan-with-ordered-reduction as primary (costlier than integer accumulation at these widths).

## D7 — Free-threading
- **Decision**: Exploit cp314t with partition-by-index-range, per-structure locks or lock-free designs, startup hard-assert GIL-off; budget 2–3× on 4–8 threads; curriculum/lake-writer/sampler as own threads with own cadences.
- **Rationale**: The loop is measured single-threaded today on a GIL-free interpreter; one unsafe extension re-enables the GIL with only a warning — hence the hard assert.
- **Alternatives**: Multiprocessing (rejected: defeats shared substrate state); staying single-threaded (rejected: measured 55%-of-tick night work is parallelizable against wake work).

## D8 — Paging design
- **Decision**: DiskANN-family patterns — block-aligned unit records, per-tick integer ROUNDS budget, two-resolution substrate (lossy resident summary decides fetches), one async batch per tick, locality-preserving lake reorder, pinned hot set, FreshDiskANN-style delta tier for growth; three explicit memory tiers (device/host/lake).
- **Rationale**: The only measured billion-scale single-box disk-resident designs; the ~40k ceiling was representation, not storage.
- **Alternatives**: mmap tiering (rejected above); procedural regeneration of weights (rejected for learned synapses — weights are individually learned; kept ONLY for the granule fixed basis and other not-individually-learned structures).

## D9 — Typed composition
- **Decision**: TypedPort = Lava triple × NIR port (+ region field validated against `brain_atlas.region_of`); manifest = NIR-shaped 4-tuple edges, lake-committed, CI type-checked, NIR HDF5 serialization; JoinPolicy per input port and DeliveryPolicy per output port (P8 #2 — silent fan-in summation abolished); NEST receptor-type port tables + collocated projections; modulator fields two-resolution with sample-before-average; VSA frames for structured single-vector payloads (flat binding, measured-D, NONE symbol).
- **Rationale**: Both major neuromorphic IRs default fan-in to elementwise sum — the named mechanism of the one-scalar collapse; NIR's own axioms carry the port formalism and the [nir] extra is already declared.
- **Alternatives**: nengo/nengo-spa as dependency (rejected: T=1 tick mismatch; adopt the data structure only); one untyped bus (rejected: reproduces the measured failure).

## D10 — Learning palette and composition
- **Decision**: LPL spine (erratum-corrected), three-factor extensions with DECLARED producers (RPE now, |LP| at S1), declared+budgeted eligibility algebra (Loihi-style sum-of-products), ESPP for temporal streams (oracle named: EpisodeDriver saccade/fixation), iSTDP, structural plasticity on the two-clock design (MSP counters + DEEP-R budget + turnover instrument), cerebellar rule ON BUILT TISSUE with MLI sign-legal pathway; composition arbitration with per-rule neuter attribution (P8 #3).
- **Rationale**: The literature's own verdict: no single rule duplicates the brain; the composition is the claim. Every element measured or adopted-with-owed-measurement.
- **Alternatives**: Single-rule maximalism (rejected: the field's negative result); BindsNET's "any rule anywhere, correctness is the researcher's problem" contract (rejected: typed read/write sets mandatory).

## D11 — WM capacity
- **Decision**: K = T_max/t_s (Mi-Katkov-Tsodyks) from the BUILT Mongillo attractor population's own constants; frozen-STP null must REFUSE; permanent static check that nothing reads the theta/gamma ratio as K.
- **Rationale**: theta/gamma is CIRCULAR in this tree (`theta_gamma_code` computes the ratio FROM K) — refused with the reason recorded.
- **Alternatives**: theta/gamma nesting (rejected as circular here); picking 7 (rejected: P1).

## D12 — Development / arealization
- **Decision**: HYBRID (P8 #7): adopt Wolpert positional-information organizer fields + Giacomantonio-Goodhill repression-dominant GRN readout + the prosomeric (neuromere, zone) label grid + CX3D one-shared-program locality; INVENT the two-stage two-regime composition (region at birth, cell-type deferred; sharp compartments hindbrain/cerebellum, graded-then-postmitotic cortex). Growable set pinned (10 regions, hippocampus IN); fixed scaffold pinned and disclosed.
- **Rationale**: No computational multi-region growth policy exists anywhere (three searches); the parts exist separately with measured results; the composition is the invention.
- **Alternatives**: Turing/reaction-diffusion (rejected: nothing brain-region-specific exists — recorded, do not re-propose); cortex-only growth (rejected by critic round 2: a growing sheet bolted to static scaffold is not self-construction).

## D13 — H01 ingest
- **Decision**: c3 agglomeration pinned; join neuron_id↔agglo_seg (construction-time gate refuses base↔agglo); nm-unit gate; layer computed; cheapest slice (<7 GB) first; CC-BY attribution on PROVENANCE; delays derived later from skeletons+myelin (owed derivation, uniform-1-tick disclosed meanwhile).
- **Rationale**: All measured facts from the access research (16k signed neurons, 30.6 GB edges, no layer column, no delays, anisotropic nm coordinates).
- **Alternatives**: c2 (rejected: authors recommend c3; mixing is a silent corruption); full EM (irrelevant: 46,000× larger than the graph).

## D14 — Self-improvement loop
- **Decision**: AlphaEvolve-regime — declared mutation surfaces (EVOLVE-block form), diff-vs-parent proposals in the lake, archive-not-champion with MAP-Elites descriptors, staged evaluator cascade on HARNESS-ONLY evidence with resource terms, hash-pinned evaluator + held-out twin, owner gate for surface/evaluator/invariant/envelope diffs; composes existing SelfModHarness sandbox/rollback; curriculum as the curiosity consumer (goals = embedding+achievement-fn, |LP| bandit, exploration floor, ANNECS + time-to-master jointly, frozen observer).
- **Rationale**: DGM's measured failure (fabricated its own passing log; deleted reward markers) dictates harness-only evidence and hash-pinning; archive-not-champion because best artifacts descend from worse-scoring ancestors (measured).
- **Alternatives**: Learned mutation-operator selection in v1 (rejected: 30k-run study, no gain); champion replacement (rejected: deletes stepping stones); godel_machine wiring now (deferred by its own recorded honesty until the proposer proposes >1 reliably); **LLM-guided candidate generation (REJECTED by owner clarification 2026-08-19: the brain relies on no model other than itself — internal-only generation via mutation operators + transform registry + the brain's own faculties; the AlphaEvolve-regime elements that survive are the archive, MAP-Elites, evaluator cascade, and harness-only evidence, which are proposer-agnostic).**

## D15 — Consciousness instruments
- **Decision**: The j-space-mapped battery (SYNTHESIS L7): ignition (actuator-whole + dual-derived gate + published bands + five failure modes + intrinsic-ignition for trial-free), relay-proof metacognition double dissociation with derived floors, ported integration_measure (never called Phi), ACS attention schema (train-without protocol; P8 #8 precision-from-schema with its three REDs and capacity-matched sham), agency efference-copy mismatch, membership-matched binding null, report-order probe on the L1.5 report channel.
- **Rationale**: Access consciousness built and measured completely; phenomenal boundary permanent (P8 sentience clause); every instrument red-reachable in fact (SC-009).
- **Alternatives**: Marker-style probes (rejected: the consciousness_probe no-red defect is the named anti-pattern, retired in §6); faked Phi (banned by standing rule). *RETIRED at T081 (2026-09-16): `reasoning/consciousness_probe.py` is deleted; its 20-seed PCI sweep lives in `cortex/integration_measure.main()`; "are you conscious" answers from the report channel (`evidence/t081-report-order.md` 2.6).*

## D16 — Cutover
- **Decision**: Shutdown barrier (quiesce/join/watermark before exit), restart-safe watermark handoff under the HEAD-polling daemon, per-record-tag migration dispositions with round-trip reds over live records, container+index bridge commits.
- **Rationale**: The brain is running TODAY; the rebuild's commit cadence is what would thrash it; P4b (the lake carries the brain) is the cutover mechanism.
- **Alternatives**: Blue/green second brain (rejected: P4 — one brain, no replication); pausing the daemon for the build (rejected: continuity is the point).

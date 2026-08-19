# Phase 1 Data Model

Entities, their fields, validation rules, and relationships. Storage-level encodings live in [contracts/lake-records.md](contracts/lake-records.md); the port schema in [contracts/manifest.md](contracts/manifest.md).

## TypedPort
The atomic signal surface (spec FR-012).
- `name: str` — unique within the declaring faculty.
- `region: str` — MUST equal `brain_atlas.region_of(declaring_module)` (manifest assert).
- `shape: tuple[int, ...]`
- `dtype`, `precision_bits: int|None` (None = unbounded), `unit: str` (physical meaning in the type).
- `encoding: {VEC_DENSE, VEC_SPARSE, SCALAR_DENSE, SCALAR_SPARSE, EVENT}`
- `kind: {IN, OUT, REF}` — REF is the flagged escape hatch; default value-copy.
- `direction-specific policy`: IN ports declare `join_policy ∈ {sum, max, concat, typed-merge, arbitrate}` (no default); OUT ports declare `delivery_policy` (graded availability legal, gated on learned association).
- Validation: every IN port unreferenced by an edge is a manifest warning; an edge into an undeclared join is a RED.

## WiringManifest
Lake-committed, versioned, NIR-HDF5-serialized (spec FR-012, contracts/manifest.md).
- `edges: list[(src_faculty, src_port, dst_faculty, dst_port)]` — 4-tuples, port-to-port.
- `validate_structure()` → `infer_types()` → `check_types()` at CI.
- The manifest IS the anatomical inventory: region coverage derives from it plus charge attribution (never registration alone).
- Relationships: consumed by the ported `p6_gate` (same-commit port), the region charge gate, and the neuter harness.

## LakeRecord (framing common to all classes)
- `magic`, `version: u16`, `length: u32`, `crc32` — recovery = forward scan, halt at first bad CRC, truncate to last-known-good; commit = watermark publish, never file length.
- `record_class`: one of the classes below; each class labeled `IMMORTAL` or `COMPACTABLE` at introduction.
- `IMMORTAL` classes carry a bytes/tick ceiling (red-capable assert).
- Migration: one disposition per persisted tag (read-as-is / upcast-on-read / one-time-rewrite), each with a round-trip red over live records.

### UnitBlock (IMMORTAL-as-current + delta chain)
- `unit_id: u64`, `region_label: (neuromere: u8, zone: enum)` (the prosomeric coordinate — distinct from TypedPort's `region: str`, which is the REGIONS-map key), `cell_type`, `position (nm-scale ints, unit-gated)`, `sign (Dale)`, `state fields (SoA layout mirrored)`, `out_edge_ids`, `weights`.
- Block-aligned: one activation hop = one block read (paging contract).
- Validation: region label from the growth program's stage-1 emission; refuses fixed-scaffold regions.

### SynapseDelta (IMMORTAL as reconstructible)
- `(pre_id, post_id, rule_id, old→new quantized weight, tick, epoch)`.
- Dirty-set membership derived from spikes; quantize-compare-drop-if-unchanged.

### Episode / Coalition
- Coalition: `{coalition_id, tick, proposer_ids[], content_ref, base_level, current, channels{salience, valence±, arousal, ...}}` — channel producers DECLARED (interoceptive/homeostatic ports).
- Episode: consolidated coalition sequences; recall via `recall(cue) → k episodes with margins` (degraded-cue red).

### Goal (S1+)
- `(embedding, achievement_function_ref)`, `|LP| history`, outcomes; vector-indexed in the lake.

### Proposal (S3)
- `{parent_ptr, diff, surface_id, score_dict (incl. resource terms), artifact_hash, run_id, map_elites_coords}`.
- Validation: diff confined to declared surface; evaluator/invariant/envelope byte-touch → owner routing.

### Snapshot (COMPACTABLE, bounded)
- Full state + per-instance RNG stream POSITIONS (G5); restore asserts stream-position equality.

## RegionLabel
- `(neuromere: u8, zone: {floor, basal, alar, roof})` — the prosomeric coordinate.
- Emitted by growth stage-1 (GRN fixed point over local organizer fields); consumed by placement AND the connection policy (one coordinate, two consumers).
- Discretization regime per territory: sharp (hindbrain/cerebellum) vs graded-then-postmitotic (cortex).

## Instrument
- `{scorer, actuator, declared_subject, reachable_red (constructed input), external_control, provenance}` — cockpit-reported.
- Validation: an instrument whose red has never fired is not yet an instrument (SC-009); a saturated detector (100% rates) is a defect signal.

## GrowthProgram (per-node, shared)
- One MLP + GRN table shared by all nodes; reads ONLY local fields (locality mask raises on non-local access).
- Emissions: stage-1 `RegionLabel` at birth; stage-2 `cell_type` deferred.
- State per node: field vector p, dilution counters (2 floats, halved on division), calcium EWMA + element counters (structural clocks).

## State transitions (the ones with invariants)
- Unit: `proposed → grown(region-labeled) → placed → charged | pruned` — pruned units append a prune record (the brain can notice loss).
- Segment: `active → sealed → (compactable only) compacted-with-proof`.
- Proposal: `proposed → evaluated → archived | rejected(reason) | owner-routed`; never `silently-latched` (circuit breaker is a cockpit row).
- Sleep phase: `wake → sleep(arm quiesced, E-STOP latch persists) → wake` — envelope tick monotone across transitions.

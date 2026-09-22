# Implementation Plan: The Unified Brain Arc — self-constructing, lake-resident substrate

**Branch**: `022-unified-brain-arc` | **Date**: 2026-08-19 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/022-unified-brain-arc/spec.md`; converged design evidence in [SYNTHESIS.md](SYNTHESIS.md) (9 layers, 4 stages, 8 P8 inventions, adoption/port registers — 3 critic rounds + frozen verification, 12→8→4→0 blockers).

## Summary

**Current execution authority (combined 2026-09-22):** follow
[the finalized plan](PLAN-FINAL-2026-09-02.md), particularly its combined B/R/A architecture,
N workstream and Parts E/F. This older four-stage summary is historical where later measured
dispositions supersede it (including the retired paging layer); it is not a parallel roadmap.
Brian2 acceptance, durable recall/replay/earned growth and offline derived-lake annealing are
FR-037-FR-040 / SC-015-SC-018 / T101-T107; NS/LSM keeps FR-036 / SC-014 / T096-T100.

Build, in four staged slices (S0–S3), the substrate in which the brain constructs itself from a single-neuron seed over a pinned multi-region growable set, lives in the ONE durable lake (indexed delta engine; activation-driven paging past a derived binding width), learns by a composed palette of local rules with per-rule neuter attribution, improves itself through an owner-gated proposer/evaluator loop, and carries a consciousness-instrument battery whose every mechanism has a dissociation with a reachable red. Technical approach per layer is the SYNTHESIS adoption register; where no method exists, the eight registered P8 inventions apply with full debt. Everything lands in governed `src/snn_data_lake/` registration-complete per commit; every commit reaches `main` full-gate green; the safety envelope is untouched.

## Technical Context

**Language/Version**: Python 3.14.6 free-threaded (cp314t, `Py_GIL_DISABLED=1`; startup hard-asserts the GIL is off — FR-006). Gate env `.venv-lake`.

**Primary Dependencies**: torch 2.11.0+cu128 / numpy; brian2 2.10.1 (replaces SNNTorch as equation/event reference, isolated worker plus Torch execution bridge; final-plan B1 governs acceptance); pyarrow (core, Arrow IPC segments); sqlite (stdlib, derived index selected at S0); nir (`[nir]` extra) for wiring-manifest serialization; mujoco 3.11.0 (exact-pinned, envelope). No whole-program codegen rewrite is implied; governed local learning and the free-threaded parent remain required.

**Storage**: `brain_lake.bin` remains the ONE durable store (P4b) — upgraded in place to sealed Arrow-IPC segments + CRC framing + watermark commit, with derived (provably rebuildable) indexes beside it. Per-record-tag migration dispositions with round-trip reds over live records (§3.5 of SYNTHESIS.md).

**Testing**: `python run_selfchecks.py` + `pytest` (≥70% coverage) + `python -m ruff check .` + `ruff format --check` (BOTH lint commands — recorded trap); the policy gate battery (dormancy ×3, replication, p4c/p6 ratchets, dependency ledger, red_reachability, gradient_claim, eprop transport, policy_manifest) asserts per commit; new gates land WITH their `--self-check` (a gate nobody verifies is a gate nobody trusts).

**Target Platform**: Windows 11 pinned measurement host (RTX 5070, D: NVMe 4.0 GB/s measured) for all performance ratchets; CI matrix ubuntu/windows cp314t (CPU torch) for correctness — CI never hosts performance claims.

**Project Type**: Single governed package (`src/snn_data_lake/`), anatomical-region layout; this feature adds modules across existing regions plus `docs/INVENTION_REGISTER.md`.

**Performance Goals**: Wake-tick p50/p95 ratchets vs pinned-host baseline (current: p50 62ms / p95 162ms, duty cycle reported); integer per-tick disk rounds budget (derived from re-measured IOPS at S2); free-threading budgeted 2–3× on 4–8 threads; GPU engaged past the measured crossover (262,144) only.

**Constraints**: Determinism — byte-identical spike trains across thread counts at fixed partition count P (integer accumulation; G5 snapshot-replay with RNG stream positions); the safety envelope runs on the substrate tick monotonically through all phases; the live daemon keeps running (cutover = watermark handoff, one brain one lake throughout); IMMORTAL lake classes carry a bytes/tick ceiling; doc-budget homes named before any governance text lands (OD-3 sequencing).

**Scale/Scope**: Seed (N=1) → multi-region micro-brain (S1, N≈10²–10³) → H01-seeded cortex (S2, ~16k signed neurons, sparse) → paging regime at the derived binding width (resident bytes > host RAM). Human-scale is a disclosed trajectory, not a stage.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

`docs/POLICIES.md` is the authoritative policy text. Audit run this session (2026-08-19, HEAD green):

```bash
python -m snn_data_lake.substrate.policy_manifest --audit
```

Result: P3/P4/P6/P7/P8/e-prop all PASS; UNCLAIMED_GATE none; NOT_IN_CI none.

Per-principle evaluation of THIS plan:

- **P0** — this feature IS the avenue list built: self-building code (L6 proposer), knowledge acquisition (L1.5 + recall contract), neural growth (L5 multi-region), plus new avenues (developmental self-construction, lake-resident substrate) joining the list. Honesty guards carried: every capability ships with its measurement; every criterion has a reachable red (FR-011/023/028; SC-009).
- **P1** — no new fixed constants: K derived (FR-020), delays derived, thresholds derived (ignition antimode + logistic agreement), rounds budget derived, binding width derived; the four honest exceptions unchanged. Constants that resist replacement are treated as bugs in the blocking constraint.
- **P2** — replacement-oriented throughout with the port register (SYNTHESIS §6): unique capabilities port FIRST, one mechanism survives (exactly one growth mechanism, per 011's own clause; `theta_gamma_code` loses ownership of K but keeps its job; scalar workspace tuples retire into coalition records with the P6-fixed rank key regression-guarded).
- **P3** — the feature's core IS charge measurement: JoinPolicy + port neuters + cleanup margins make dormancy mechanically visible; the region gate is charge-sourced (registration alone can never green it); every new module registers in REGIONS + run_selfchecks in its own commit.
- **P4/P4b/P4c** — one brain, one lake, one daemon through cutover; the lake writer on the ring makes P4b structural; everything lands governed from the first commit (no reasoning/ staging).
- **P5** — GPU by measured width; suite stays runnable without a GPU (CI has no card); determinism contract covers both tiers.
- **P6** — the typed manifest + declared JoinPolicy topologies are P6's structural answer; p6_gate ports to the manifest in the same commit that lands L3 (S0), so the ratchet never goes blind.
- **P7** — adoptions each carry their owed measurement (research.md); rejections carry measurements or structural arguments (recorded, do-not-re-propose).
- **P8** — eight inventions in `docs/INVENTION_REGISTER.md`, each with RED / external control / attribution / disclosure / deletion-on-null / clause-6 anti-rigging.
- **Cross-cutting** — e-prop not backprop (palette is local/e-prop-shaped; cerebellar rule's transport-freedom measured at adoption); verify-first (every stage exit is a measurement); one change per commit, full gate green; cite the SYMBOL; safety envelope owner-approval only.

**Violations requiring justification: none.**

## Project Structure

### Documentation (this feature)

```text
specs/022-unified-brain-arc/
├── spec.md              # The specification (committed ee6796f4)
├── SYNTHESIS.md         # Converged design evidence (committed ee6796f4)
├── plan.md              # This file
├── research.md          # Phase 0: consolidated decisions (adoptions/rejections/inventions)
├── data-model.md        # Phase 1: entities, records, validation
├── quickstart.md        # Phase 1: per-stage validation guide
├── contracts/           # Phase 1: manifest, lake-record, and gate contracts
└── tasks.md             # Phase 2 (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
src/snn_data_lake/
├── substrate/           # L0 lake engine (segments, index, watermark, journaller ring);
│                        #   L2 paging (unit blocks, residency, rounds budget);
│                        #   L3 manifest + TypedPort + Join/DeliveryPolicy + charge instruments;
│                        #   region charge gate; cutover barrier; existing gates extended
├── cortex/              # L5 growth policy host (arealization program, placement generalization);
│                        #   executive faculty port; integration_measure port (L7)
├── hippocampus/         # recall contract consumer; existing growers composed into the growable set
├── thalamus/            # workspace instance resolution; coalition records; ignition actuator+scorer;
│                        #   pulvinar/MD transthalamic route
├── cerebellum/          # L4 cerebellar tissue build (granule/PF/Purkinje) + adopted rule (S1/S2)
├── basal_ganglia/       # proposer/evaluator surfaces (L6, composing SelfModHarness path)
├── brainstem/           # SAFETY ENVELOPE — untouched except owner-approved; effector port wiring AROUND it
├── medulla/             # interoceptive producers (valence/arousal ports)
├── midbrain/            # RPE third-factor producer (dopamine_nuclei port)
├── pons/                # gains its first module: the pn mossy relay (L4 cerebellar input)
├── insula/              # NEW region — opened by T021a (atlas + connectome registration)
├── hypothalamus/        # sleep clock (L5.5 pacer)
├── plasticity/          # L4 palette: LPL spine, STP/WM, composition arbitration, structural two-clock
├── neuromodulation/     # serotonin/glia first registrants (RPE lives in midbrain/, |LP| in basal_ganglia/)
└── connectome.py        # pulvinar/MD nodes; atlas gate (NODES→module resolution)

tests/                   # per-commit: unit + the new gates' self-checks + G5/determinism/torn-write reds
docs/INVENTION_REGISTER.md   # NEW budgeted file (registered in doc_budget_gate.BUDGETS same commit)
```

**Structure Decision**: anatomical-region layout preserved (P4c: the directory is a fact about the filesystem; governance comes from registration + gates). All L0–L3 infrastructure lands in `substrate/` beside the existing lake/gates it extends or replaces via the port register. The only NEW package is `insula/` (opened by T021a with full registration); `pons/` (existing, empty) gains its first module. No `reasoning/` or `research/` additions — governed tree from the first commit.

## Complexity Tracking

No constitution violations to justify. The one deliberate scope split: S3's proposer/evaluator composes the EXISTING SelfModHarness sandbox/rollback rather than rebuilding it (P2 — the unique capability, byte-exact rollback, already exists and ports forward).

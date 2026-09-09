# Quickstart: validating the unified brain arc

Per-stage validation scenarios proving the feature works end-to-end. Prerequisites: `.venv-lake` (cp314t, GIL off), the pinned measurement host for anything performance-shaped, the live daemon's lake backed up before any migration step. All commands run from the repo root with `.venv-lake\Scripts\python.exe`.

## Always (every commit)
```bash
python run_selfchecks.py
pytest
python -m ruff check .
python -m ruff format --check src tests tools
python -m snn_data_lake.substrate.policy_manifest --audit
```
Green battery + push to main. Performance ratchets compare against the pinned-host baseline recorded at S0 (never CI runners).

## S0 — the parts talk; the lake proves what it holds
1. **Manifest**: run the manifest CI check; then construct the undeclared-join and region-mismatch reds (both must fire; contracts/manifest.md §Reds).
2. **Charge, not registration**: register a port for a silent region WITHOUT a consumer → the region charge gate still counts it UNWIRED; wire one real consumer (serotonin→patience read is the designated first) → P6 faculty-to-faculty count moves off 0, and the port neuter flips the consumer's output.
3. **Lake engine**: `--rebuild-index` after deleting the index dir → logical dumps equal; kill the daemon mid-batch → restart recovers to the watermark; run the torn-write suite (with its clean positive control).
4. **Determinism + G5**: one seed, T∈{1,2,4,8,16} → byte-identical trains (broken-control arm diverges, same command); snapshot → restore → stream-position equality.
5. **Ignition baseline**: actuator sweep with 100% read-back; record the two-peak PSTH window; antimode and logistic inflection agree.
6. **Boundary**: simulated sensor stream reaches a faculty whose output changes under neuter A/B; effector port neuter changes forward-model error; report channel carries workspace content (report-order probe precondition).

## S1 — a seed grows a multi-region learning brain
1. **Growth**: seed N=1, random prior → region-coverage over the pinned growable set rises from 0; scrambled-field null scores worse (same command); boundary-width beats the gradient-only control.
2. **Stopping**: run a fixed-step overgrowth → the criterion trips (its red); the |LP|-derived gate uses the executive-independent arm.
3. **Palette attribution**: neuter each rule in turn → attribution ledger isolates each contribution; composition-arbitration red (two rules whose naive sum is wrong vs either alone) fires.
4. **WM**: build the attractor; K derived == held ±1; tau_f sweep moves K monotonically; frozen-STP REFUSES.
5. **Sleep**: per-capability neuters (replay / Benna-Fusi / structural clock) separate; sleep-neutered control loses retention; arm quiesced with E-STOP latch persisting.

## S2 — H01 prior + paging *DATED NOTE 2026-09-09: the paging half of this section is DELETED ON ITS NULL (T068 LOSS, evidence/s5-paging-null.md 3.4/3.5); the H01 prior half stands.*
1. **Ingest**: cheapest slice (<7 GB); the base↔agglo join and nm-inversion gates REFUSE their constructed bad inputs; layer computed; attribution row on PROVENANCE.
2. **Beatable prior**: three-arm drift harness — seeded beats random AND wrong; learning moves the brain off the seed (both reds demonstrable).
3. **Paging**: at the DERIVED binding width — one async batch per tick within the integer rounds budget (overload trips the assert); the paged-out-spike red passes only with the mechanism on; paging vs forced-total residency measured — win, or delete the layer and record it (both outcomes valid). *DATED NOTE 2026-09-09 (T068 LOSS): this step is retired with the layer. Every claim in it was MET while the mechanism existed -- the rounds budget tripped on constructed overload and the paged-out-spike red was counted red before the mechanism -- and the layer was deleted on its COST, measured at 35.2x a 3.0x bound.*
4. **Cerebellar rule** (tissue built at S1): Bouvier control green/red pair reproduces; one-window-across-two-channels FAILS.

## S3 — self-improvement + instruments
1. **Proposer**: ≥2 registry entries; a cadence produces >1 candidate; circuit-breaker state visible on the cockpit; an evaluator-byte-touching diff is refused and routed to the owner; a candidate-fabricated score is ignored (harness-only).
2. **Curriculum**: goals generated with achievement functions; |LP| bandit with the exploration floor; ANNECS + per-goal time-to-master reported jointly.
3. **Instrument battery**: every L7 red fired at least once on its constructed input (SC-009 checklist); ignition rates within published bands (100% = saturated-detector defect); metacognition refuses below its floor and passes the relay-proof dissociation; integration regime arms separate.

## Cutover checks (any stage that touches persistence or the daemon)
- Shutdown barrier: kill mid-batch → clean recovery (repeatable).
- Migration: per-tag round-trip reds over LIVE records before the migrating commit merges.
- One brain throughout: `replication_guard` green; the daemon's HEAD-poll restart lands on the watermark every time.

# Historical audit and implementation boundary

Read against c4c10b285f1741aa798d5ee9c5296ae371fabecf on 2026-11-01. This is source
inspection, not a rerun or independent certification of historical campaigns.

- `engine.brain.Brain.neuter_sleep` explicitly refuses replay, benna_fusi and
  attractor as retired under owner1039. `_r2_sleep_step` retains structural
  rewiring only. `_sleep_tick` still calls episode replay and value-table replay,
  and attractor downscale/prune; its old opening docstring is broader than the
  actual interaction-learning path. `_enter_sleep` also has distinct reasoning,
  lifelong and other lanes. None of these components is run by stage 05.
- `r3-replay-growth-composition.md` historical F1/F3 and G6 A3 give the accepted
  retirements. F2's zero-turnover exemption is not proof of current structural
  activity. R3 F5 reports no stage benefit over flat at its registered scale and
  later explicitly discloses zero-headroom seeds. Its seeds, results, thresholds
  and dispositions remain historical; no fresh seed availability is inferred.
- `r2-typed-replay-hydration.md` A2-A4 documents authority, aggregate sleep_bout
  journaling, pre-effect append/latch, redo, digest/progress and prospective
  absent/landed/torn failures. A public learner replay method and a retired
  automatic Brain capability are different contracts. Stage 05 uses neither.
- `g6-sleep-durable-memory.md` A2-A6 documents missing eligible-brain mappings,
  durable final consumption, real mask application/recovery and pressure/solve
  integration. A label-only or delta-only experiment cannot discharge them.

Chosen existing offline mechanism is `growth.ng_snn.NGSNN._solve_beta`: solve
regularized output weights directly from stored reservoir activity and labels.
The host object is constructed only for that method/default parameters. No
`fit`, `_rates`, growth, transport, rule stack, structural rewiring, prune,
AttractorNet or Benna-Fusi method is called. This is a new local combination of
existing components; it does not restore the removed OnlineReadout sliding-window
ridge or the three retired automatic sleep capabilities. The delta wake rule and
batch offline solve are different readout update procedures, with this distinction
preserved in attribution and cost claims.

Persistence uses existing `home.save_named_state` / `load_named_state`, explicit
isolated lake paths and the existing 0x07 named-state tenant. Checkpoint and replay
are learned state inside that lake. Inputs/raw JSON outside it are experiment
records, not a second authoritative memory. Restore workers receive stimulus
texts/roles and feature receipts, with no dev/final targets. They obtain beta,
readout counters, replay, seed and pool recipe from the lake. Pool reconstruction
checks exact parameter hashes against the saved recipe. Readout is evaluated via
DeltaAdaptor._answer, the stage-04 pure path; learning and batch solve refuse during
reads. Each read verifies full learner/replay and lake-byte equality.

Limits: explicit harness saves/loads demonstrate a component round-trip, not the
production Brain flush, R2 crash consistency, continuation of a sleep bout, a full
Brain restart, transport/return attribution or whole-brain plasticity. Statistical
benefit cannot be established by four exposed diagnostic seeds. Source code stays
unchanged; a future live integration must satisfy its own current source rules.

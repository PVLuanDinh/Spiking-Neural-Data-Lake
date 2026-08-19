# Contract: gates this feature adds or extends

Every gate lands WITH `--self-check` (proven able to fail) in its own commit, registered in CI and, where package-shaped, in `run_selfchecks.py`. Counting before adopting: each gate's output is counted against the tree BEFORE its rule binds (cross-cutting invariant).

| gate | asserts | its reachable RED (proven at landing) |
|---|---|---|
| Manifest type-check (extends CI lint) | structure/types/regions/joins of the wiring manifest | undeclared-join; region mismatch |
| Region charge gate | integer floor of regions charged; exemption ratchet | a registered-but-uncharged region counts UNWIRED |
| p6_gate (ported) | faculty-to-faculty ratchets, sourced from the manifest | gate/manifest divergence; ratchet regression |
| Determinism gate | byte-identical spike trains across T∈{1..16} at fixed P; V_c==0, V_ermv==0 | broken-control arm (float accumulation) diverges — same command |
| G5 replay gate | snapshot-restore stream-position equality + identical trains | omit RNG positions from a snapshot → red |
| Lake recovery gate | torn-write truncation to last-good; watermark recovery | injected mid-record corruption (positive control: clean run) |
| Index rebuild gate | logical-dump equality after delete-and-rebuild | planted index divergence |
| Rounds-budget assert | per-tick disk rounds ≤ integer budget | constructed overload trips it |
| IMMORTAL ceiling assert | bytes/tick per immortal class ≤ ceiling | constructed append flood |
| Atlas-node gate | every connectome.NODES id resolves to ≥1 referencing module in its region, or dated EXEMPT | the pre-existing cb_ctx/pn phantoms are the day-one count |
| Growth coverage red | regions-grown / pinned growable set | cortex-only placement measures below floor TODAY |
| Boundary-sharpness red | emitted region labels vs modal-across-seeds; gradient-only control in same command | fields without GRN → wide boundaries |
| Stopping-criterion red | growth past measured optimum trips the gate | fixed-step overgrowth run |
| K-derivation guards | derived==held ±1; frozen-STP REFUSES; tau_f>tau_d regime; no theta/gamma read as K (static) | run against a monotone-K tau_f sweep (positive control) |
| Ignition battery | dual-derived gate agreement; published bands; five failure modes distinguishable | shuffled-label control within noise = artifact red; recurrence ablation kills peak 2 |
| Metacognition floors | refuses calibration below derived label-entropy floor; double dissociation | relay construction must FAIL it |
| Evaluator integrity | hash-pins on evaluator/invariants/surface/envelope; harness-only evidence | candidate-emitted score ignored; evaluator-byte diff routed to owner |
| Cutover barrier | quiesce/join/watermark before exit | kill mid-batch → clean watermark recovery |

Self-disposition: these gates are census-covered by `policy_manifest` once registered; none lands bare (a tool wired in bare cannot redden the build).

# P8 Invention Register (specs/022)

Every method invented under P8 for the unified brain arc lives here with its full debt. Each
entry owes SIX things (P8 clauses 1-5 + 6, specs/022): a success criterion with a REACHABLE
RED (constructed failing input; no reachable red = deleted); an EXTERNAL CONTROL that is not
the method itself; PER-ITEM ATTRIBUTION; UNPROMPTED DISCLOSURE of contamination or vacuous
preconditions; DELETION ON A NULL (recorded); and ANTI-RIGGING — every neuter enumerates ALL
live readers, headlines the isolated single-consumer counterfactual, and states whether the
neutered information is recoverable elsewhere. Evidence lands in
`specs/022-unified-brain-arc/evidence/`; this file is the index, never the measurement.

## 1. Activation-driven paging of a living substrate (L2, T066)
Neurons as lake records; residency decided by spike traffic, never by OS eviction.
RED: a spike arriving at a paged-out neuron must not be silently dropped (constructed before
the mechanism exists). CONTROL: forced-total residency at a derived comparison width under the
host-RAM ceiling. ATTRIBUTION: per-tick rounds counter vs its integer budget. DISCLOSURE: the
comparison width is NOT the deployment width (>RAM), stated wherever the win is claimed.
NULL: paging loses to forced residency at the comparison width → the layer is DELETED.
MECHANIZED 2026-09-08 (T066, `substrate/neuron_pager.py`): both reds in-tree and counted red
before the mechanism; a calm paged run is byte-identical to the resident twin; S-R3 stays T068's.
DELETED ON ITS NULL 2026-09-09 — evidence/s5-paging-null.md §3.4/§3.5/§4. The NULL clause above
fired as written: memory claim met, cost bound missed by an order of magnitude. The row STAYS (a
method deleted on its measurement is this register working); the files its MECHANIZED line names
are gone, last carried at tag `t068-last-with-pager`.

## 2. JoinPolicy + mechanical charge measurement (L3, T020-T022)
Declared fan-in/fan-out policies per port (silent elementwise summation abolished — both
major neuromorphic IRs default to it and it is the measured one-scalar collapse mechanism),
plus charge instruments: pinned-payload neuters, cleanup margins, replay A/B.
RED: undeclared-join manifest red; a registered-but-uncharged region counts UNWIRED.
CONTROL: the charge probe's own constructed positive. ATTRIBUTION: per-port. DISCLOSURE:
registration is never charge. NULL: a port no consumer's output depends on is retired.
LANDED (declaration + enforcement halves): T018/T019 2026-08-20 — the undeclared-join red
fires in `engine/wiring_manifest.join_check` and `gates/manifest_gate`; T020
2026-08-20 — enforcement in `engine/signal_ring.JoinConsumer` (all five joins; arbitrate
carries the {winner, margin_to_runner_up} cleanup record; a policy-stripped port REFUSES via
`UndeclaredJoin`) and `DeliveringProducer` (graded REFUSES unbound via `GradedUnbound`;
incumbent source `reentrant_workspace._assoc`, final binding = T030). Charge instruments =
T022. Float-sum determinism: producer-sorted float64 accumulation, disclosed in-module.

## 3. Multi-rule composition arbitration (L4, T051a-c)
N local learning rules sharing one synapse under typed read/write sets, with standing
per-rule neuter attribution. RED: two composed rules whose naive sum is measurably wrong vs
either alone. CONTROL: each rule's single-rule arm. ATTRIBUTION: the point of the invention —
per-rule contribution ledger. DISCLOSURE (corrected at T051a): arbitration order is bare add()
insertion order — NOT learned, NOT recorded in the payload; the learned/recorded form is owed,
undated. ANTI-RIGGING (T051b-iii): first reader = rule_charge, the per-rule A/B consuming
last_attribution(); TICK-LIVE since T053-ii: recorded bindings, CHARGED.
NULL: a nothing-contributing rule is retired (P3) — INERT is its constructed case.
MECHANIZED: `plasticity/rule_composer.py` — T051a kernel; T051b-i jepa LPL declaration; T051b-ii
the T048 pair at component grain; T051b-iii check_producers (FR-019, production manifest) +
rule_charge; T051b-iv the stack-sandwich MERGE expressed (gain two-writer + base weights
deposit-then-trim vs spied production; both stacks remain production — a declaration layer,
P2 merge-must-reduce NOT discharged: two incumbents before and after); T053-i/-ii the FIRST
TICK-CHARGED COMPOSED RULE (InhibitorySTDP.step applies through it; neuter max|w|
0.791 vs 1.030; dimensionality prediction NULL). No tissue: T054.

## 4. Growth stopping criterion (L5, T047)
|LP|+charge-derived per-region growth gates with an executive-independent arm (a gate routed
through the executive would let the guard's caller constrain it). RED: a fixed-step overgrowth
run must trip it. CONTROL: a fixed growth schedule. ATTRIBUTION: per-region gate decisions.
DISCLOSURE: both NDP source papers name stopping as unsolved; this is invention, not adoption.
NULL: if gated growth never beats fixed-schedule growth, the criterion is deleted.
MECHANIZED: `growth/growth_stop.py` (T047, 2026-08-26) — STICKY stop (the panel measured the
first draft re-opening on its own frozen rows, a 1-in-7 duty cycle; now 0 post-stop growth
refits in 10x100); charge margin = the null SE sigma*sqrt(2/PATIENCE); |LP| arm = edge-
triggered order-statistic excursion (false-fire 0.83% on the live zero-inflated shape; the
draft's bare cur>median was a 51.9% coin flip); NULL = the real two-arm engine A/B, asserts
armed: stops 6/6 seeds, unit deltas sum −27, acc median −0.017 — measured, not tripped;
evidence tail persisted with the substrate (restart is not amnesia).

## 5. Brain-change evaluator cascade (L6, T073)
Staged fitness for INTERNAL-ONLY proposals (FR-025: no external model) on harness-only
evidence with resource terms. RED: a candidate-fabricated score must be ignored (the DGM
failure reproduced as a test). CONTROL: the held-out twin evaluator the proposer never sees.
ATTRIBUTION: per-stage rejection reasons, archived. DISCLOSURE: evaluator/invariant/surface/
envelope bytes are hash-pinned outside the mutation surface; touching them routes to the owner
(OD-2). NULL: if archived proposals never beat their parents, the loop reports it and halts.

## 6. Consciousness dissociation battery (L7, T033 + T077-T081)
MECHANIZED (first instrument): `instruments/ignition_instrument.py` (T033, 2026-08-21) — the
ignition scorer, dual-derived (last-spike-trough window re-derived from rasters; antimode +
logistic-inflection routes must agree at instrument resolution), driven through the T032
actuator under sweep hold, with the shuffled-label + recurrence-ablation controls run in the
same call and all five failure modes (SATURATED/SILENT/UNIMODAL/DISAGREE/CONTROL_FAIL)
reached on constructed inputs in its `--self-check`; baseline in `tools/ignition_baseline.json`.
Access-consciousness instruments with dissociations, never markers: ignition (dual-derived
gate, actuator-whole, published-band checks — 100% ignition = saturated detector), relay-proof
metacognition (lesion second-order → first-order preserved, calibration collapses), integration
(perturb-and-compress, never called Phi), agency (efference-copy mismatch), binding
(membership-matched primary null), report-order (perturbed intermediate). RED: each instrument
fires its constructed red before any positive claim (SC-009). CONTROLS: shuffled-label,
recurrence-ablation, capacity-matched shams — external per instrument. DISCLOSURE: phenomenal
experience is never claimed from a passing criterion (P8 sentience clause). NULL: an instrument
whose red never fires is not an instrument and is deleted.
MECHANIZED (second, T077, 2026-09-15): `metacognitive_monitor.track_error` (auroc2, permutation-null readability,
the in-vivo dissociation; reds R0-R5 fired first) READ NOT MET -- `evidence/t077-metacognition.md` 3.3-3.4; nothing deleted.
MECHANIZED (third, T079, 2026-09-15): `agency_monitor` per-joint efference copy + cause-blind twin + in-vivo dissociation
(reds R1-R6 fired first) READ NOT MET -- `evidence/t079-agency.md` 3.3-3.4; nothing deleted.
MECHANIZED (fourth, T080, 2026-09-16): `instruments/binding_instrument` reads the live cross-modal binder against a
membership-matched null at matched K (reds R1-R6 fired first) READ PASS, t +13.54 -- `evidence/t080-temporal-binding.md`
3.3-3.4; the C5 row charged at 0.2309; nothing deleted.

## 7. Two-stage two-regime arealization composition (L5, T040-T042)
Adopted parts: positional-information organizer fields, repression-dominant GRN readout,
prosomeric (neuromere, zone) label grid, one-shared-program locality. INVENTED composition:
stage-1 region emission at birth, stage-2 cell-type deferred; sharp compartments for
hindbrain/cerebellum vs graded-then-postmitotic for cortex (importing one regime into the
other is a published category error). RED: region-coverage over the pinned growable set reads
below floor TODAY (cortex-only placement); boundary-width vs a gradient-only control in the
same command. MECHANIZED: gates/growth_coverage_gate.py (T046, 2026-08-26). CONTROLS: scrambled-field spatial null; adjusted-Rand vs an out-of-model
partition (Allen / published prosomeric map). ATTRIBUTION: per-region emission counts.
DISCLOSURE: no computational multi-region growth policy exists to adopt — three searches, none
found. NULL: if the GRN readout never beats gradient-only labels, the composition is deleted.

## 8. Precision-from-attention-schema third factor (L7, T078)
Precision computed FROM a self-model of attention (the ACS vector) routed as a learning third
factor. Preconditions G3/G4 carried with stop/void wording. RED-1: constant-pin the factor —
matching learning curves mean it is decorative (run BEFORE any positive claim). RED-2: swap
schema-residual for raw-residual precision — equal performance means adopt plain
precision-weighting and report the null. RED-3: double ablation under redundancy control.
CONTROL: capacity-matched sham carrying shuffled schema content — three arms, one command.
DISCLOSURE: the ADOPTED half (ACS structure, train-without protocol) is Graziano's; only the
third-factor routing is invented. NULL: RED-2 equality → the invention is deleted, the
adoption stays.

## 9. Two-chain world-red instrument (Phase 1 W3, PLAN-FINAL-2026-09-02)
Adopted parts: the in-tree distinct-values liveness idiom (`autonomous_loop.py`'s
`len(set(sal_live))` with its pinned-constant control), Shannon entropy, rank 2c's two archived
probes. INVENTED composition: the two curiosity chains scored SEPARATELY and never averaged
(`learning_progress.py` records the spiking drive as MEASURED INVALID as an |LP| basis, so a merged
statistic would lie in both directions); a liveness verdict keyed on the 4 dp ROUNDED distinct count,
the raw count reported beside it; the refit count DERIVED as scheduled (`ticks // refit_every`) minus
effective (`len(trajectory)`), the difference counting refits that early-returned on the data guard.
RED: 11 criteria, every one the catching assert for at least one of 22 mutants (22 red, 0
survived; the unmutated check is the sensitivity control, run before and after each) -- a pinned constant
must read AT FLOOR, a varying one LIVE, an unfed channel missing rather than 0.0, a channel varying only
below 1e-4 dead despite 200 raw-distinct values, a NaN channel dead rather than maximally live, and
`summarise` must slice the RIGHT rows into each window. CONTROL: the unmutated check as sensitivity; a ramp
beside the pinned constant, one command. ATTRIBUTION: per-channel and per-seed, never a seed average --
s2c singled out seed 1 as the one seed not yet converged, and an average hides exactly the seed a treatment
passes for free on. DISCLOSURE: at landing this row read "7 criteria, 8 mutants" and the arc's panel showed
criterion 7 was VACUOUS for the property it named -- four slice mutants passed it green, including one that
would have relocated every reported window; it was rebuilt, and three further criteria were added for
defects no criterion could see. Also disclosed: neither refit derivation survives a lake restore, so every
mode runs `persist=False` under a throwaway home and refuses the live lake home; and the refit gap is a
CONSEQUENCE of a single-category window, not independent corroboration of one. NULL: an instrument whose
criteria stop being reachable under mutation is deleted, not weakened (P8 clause 5) -- the W1/W2 revert
below is the EXPERIMENT's null, not this invention's. If no channel separates W1 from its time-shuffled
control, W-R1 fails and W1/W2 revert.

## 10. Authoritative-label provenance for the chat intent learner (G0(a), T108)
Adopted parts: the incumbent R-STDP learner and its sleep-replay ring (`SpikingInteractionLearner`). INVENTED: a
person's correction is AUTHORITATIVE for its normalized utterance; a later self-label on it is SUPERSEDED, and the
ring's older entries for it are REWRITTEN. RED: each part's neutered twin (B1: 192 post-correction wake self-labels;
B2: 295 post-correction replayed ones). CONTROL: the uncorrected incumbent (C) and a one-shot correction (B0).
ATTRIBUTION: supersession carries routing (B1 0.750 vs A 1.0); the rewrite guards only the teaching record (B2 routes
1.0). NULL: a part whose red stops being reachable is deleted. `evidence/g0a-chat-correction.md`.

## 11. The learned route consumed as a confirmed clarification (G0(b), T108)
Adopted parts: the corrected spiking intent learner and incumbent grounded replies. INVENTED: read the learned
route before teaching, propose one interpretation, and answer only after the person's next-turn confirmation.
RED: no paired useful-answer gain against either the incumbent or shuffled proposals; any seed with excess
misfires. CONTROL: consumption off (N), deranged routes (S), discarded branch (D), and direct answers (U).
ATTRIBUTION: full reply text scored against the frozen key, alongside confirmed-route labels; unnecessary
questions reported separately. NULL: any campaign RED deletes the branch, never weakens the bar. SCOPE: the
frozen corpus, seeds 0-7, the no-background-chat schedule, and a cooperative truthful person only. Confirmation
does not teach authoritatively. PASS 2026-09-24: A beat S (t 16.9) and N (t 26.0), no excess misfires.
`evidence/g0b-reply-consumes-route.md`, section 3.

## 12. The chat learner's restart: a validated whole-tree checkpoint and lake-sourced authority (G0(c), T108)
Adopted parts: the incumbent named-state record (`_T_STATE`), the episode tag (`_T_EPISODE`) and G0(a)'s ring rewrite.
INVENTED: one codec for the learner's whole object tree that validates class and attribute sets and stages everything
before it writes; a checkpoint at the end of each night and on close; authoritative labels appended at once as typed
teaching events, with the authority rebuilt from them and the restored ring reconciled. RED: a restart that differs
from an uninterrupted twin on the tree, routes, post-step weights or replay draws, through either production hook; a
stale ring after a post-checkpoint correction. CONTROL: no restore (N), one omission per carrier, and a skipped
reconciliation (O_rec). SCOPE: this corpus, seeds 0-7, a same-code restart. PASS 2026-09-24.
`evidence/g0c-learner-persists.md`, section 3.

## 13. The live grower replayed from one recording, its gate fed a causally lagged signal (G1(a), T109)
Adopted parts: the grower (`NeurogenesisSelfImprove`), its growth gate (`GrowthStop`, now fed through one shared
function) and the L-rows' feed idiom. INVENTED: record the live stream once (acquisition attempts, the gate's |LP| at
every delivery and sleep tick, per-refit fingerprints) and replay arms offline through channel-restricted feature maps;
the gate reads the |LP| recorded at the last delivery, because the live value after a delivery carries the next
frame's surprise; one decision per event, cached before its target. RED: a replay that differs from the in-brain
grower event for event; a suffix mutation that changes a prediction at or before its cut; a missing or disconnected
|LP| read that is not refused. CONTROL: the in-brain run itself, a leaking predictor, a forbidden-channel map, an
unseeded-noise arm. SCOPE: pilot seeds 100-103, 2,400 ticks. PASS (harness validation) 2026-09-24.
`evidence/g1a-causal-contract.md`, section 3.

## 14. The growth trigger's exact lift certificate (G3 P1, T111)
INVENTED: on the taught learner's full current code, contradictions -> DATA/STATE; at k = 0, CAPACITY only on a
Farkas witness over the (synapse, rank) lift that an LP finds and an exact Fraction check accepts; SEPARABLE only
through the real `route_text`; else UNRESOLVED. RED: a doubled multiplier; twin labels through the XOR witness; a
capable grown unit. CONTROL: the LP status; the separable twin. NULL: the gated loop does not beat the literal loop
-> the gate leaves the loop. `evidence/g3-growth-loop.md` 2.4, 2.9 C2.
SYMBOLS: invention `capacity_certificate.diagnose` / `farkas_witness` / `check_farkas` / `separating_witness`. REDS:
`test_doubled_multiplier_fails_the_exact_check`, `test_twin_labels_pushed_through_the_xor_witness_fail`,
`test_a_capable_grown_unit_stops_the_certificate_on_the_augmented_code`. CONTROLS:
`test_diagnose_twin_is_separable`, `test_an_incapable_grown_unit_keeps_the_certificate_firing`. ATTRIBUTION: per-item
Farkas multipliers (`farkas_witness`). DISCLOSURE: no lift witness is conservative, not proof of separability
(UNRESOLVED branch). NOT YET RUN: the gated-vs-literal NULL contrast (needs the grown-segment port, owner #2).

## 15. The trace-trained candidate source (G3 P2, T111)
INVENTED: a candidate row trained on the multiclass residual by its own eligibility trace and a FIXED random
feedback vector (never a transposed weight). RED: a W_out.T variant is flagged (C7); shuffled training labels must
lower the held-out score (C10). CONTROL: presized-selected, random growth. NULL: targeted beats neither -> DELETED;
the random source stays. `evidence/g3-growth-loop.md` 2.5, 2.8.
SYMBOLS: invention `growth_loop.trace_train` (declared LOCAL, never a gradient claim). REDS:
`test_a_transposed_readout_in_the_trace_rule_is_flagged` (C7, eprop_transport_gate). CONTROLS:
`test_the_trace_rule_carries_no_weight_transport`; `test_trace_training_moves_the_row` (eta 0 does not).
NOT YET BUILT/RUN: C10's shuffled-label red and the presized-selected / random contrasts
(`growth_campaign.primary` reads them once a runner exists; no runner does, owner #2). DISCLOSURE: p_c is
softmax(v/T) at T = 1, fixed, a session call (2.5 D4).

## 16. The problem-keyed budget pause (G3 P3, T111)
INVENTED: grow a whole batch, attempt, repeat; GrowthStop's problem key pauses when the next batch exceeds beta x
median(positive-spend solves) (first: 52); a 3-attempt no-growth cap; an IMMORTAL seq-checked ledger. RED: an
unsolvable cell pauses UNSOLVED, never solved (C8); a frozen learner ends LEARNING_RULE_DEFICIT (C18); a dropped row
refuses the rebuild (C13). CONTROL: a solvable twin. `evidence/g3-growth-loop.md` 2.6.
SYMBOLS: invention `GrowthLoop` + `Ledger` + `GrowthStop.observe_problem/should_grow_problem`. REDS:
`test_unsolvable_problem_pauses_at_the_budget_and_stays_open`, `test_literal_loop_with_the_diagnosis_disabled_still_pauses`
(C8), `test_separable_problem_that_never_solves_hits_the_attempt_cap` and
`test_unresolved_problem_hits_the_attempt_cap` (C18), the seq-gap fold refusal (`test_growth_loop.py` ledger tests,
C13). CONTROLS: `test_capacity_problem_grows_until_solved_and_emits_one_solve`, `test_a_fresh_ledger_rebuilds_byte_equal`.
ATTRIBUTION: `test_budget_baseline_counts_only_positive_spend_solves`. DISCLOSURE: beta is a [PROJECTION]
(`taught_problems.BETA` is None); 0x0B cumulative growth is claim-with-a-check (decision 13).
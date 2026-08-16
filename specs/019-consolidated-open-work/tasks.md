# Tasks: Consolidated Open Work

**One list replacing fourteen.** Generated from the verified inventory, never hand-transcribed:
178 items hand-copied is 178 chances to introduce the citation defect this repo keeps recording.
Sources: 8 `specs/*/tasks.md` files with unchecked tasks, plus the 66 open rows of
`specs/BACKLOG_RANKED.md`. See `spec.md` for why this exists and how it was verified.

**173 items.** 120 open, 53 blocked, 0 closed at HEAD but kept here because a live document still asserts them.

> # START HERE: BAND 2. Band 1 is CLOSED and MERGED TO MAIN.
>
> **`origin/main` = `4db316e7`** (was `f362cb52`). The 018 lane -- 18 commits, eight lanes -- is
> merged; every Band 1 fix is on main, verified per artifact rather than inferred. The next open
> band is **Band 2 (P0, the purpose, which outranks P1-P8 explicitly)**, 15 items.
>
> **STANDING RULE, owner 2026-08-16: EVERY BUILD MUST REACH `main`, not merely the governed tree.**
> P4c says a build lands inside the governed tree from the start; this adds the other half. A lane
> that ends with its work correct, gated and sitting on an unmerged branch has not delivered it --
> anyone cutting from `main` gets none of it. This rule exists because it happened: 18 commits sat
> unmerged while `main` was missing TWO safety-path fixes (a fabricated fear association on every
> idle sensor sample; a 3x units-contract gap), a DETERMINISM breach fix (the same seed grew a
> different brain on a GPU box than on CI), and the only mechanization the e-prop invariant has ever
> had. Every one was green, committed and pushed. None of it was on main. See `docs/POLICIES.md`
> P4c for the amendment and its gate specification.
>
> **BAND 1 CLOSED 2026-08-15 -- all 6.** Status is hand-applied here; the item list below is still
> the GENERATED artifact (FR-001) and was not re-rendered, so treat this block as the authority on
> Band 1 and the entries below as the original inventory.
>
> | item | disposition |
> |---|---|
> | `[B1.01]` | DONE, and UNCOMMITTABLE -- `CLAUDE.md` is gitignored (`.gitignore:135`), the documented P4c exception, so no commit or gate can carry it |
> | `[B1.02]` | ADJUDICATED `326c8f68` (transport CONFIRMED, NOT a violation -- `MLP` is a CONTROL) + control exemption GRANTED by the owner `d880a7b0`. **ARC promotion cluster unblocked** |
> | `[B1.03]` | FIXED `2f9eba7b` -- idle sentinel is a MISSING reading, not contact |
> | `[B1.04]` | PINNED `4a2ae912` -- **and this task's stated DIRECTION was wrong, see the correction on its entry below** |
> | `[B1.05]` | PINNED `6e506235` -- values AND meanings, with a demonstrated layout-shift red |
> | `[B1.06]` | FIXED `0afafd40` -- grown weights drawn from the CPU stream |
>
> **Open work these created, recorded rather than left implicit:** (a) `eprop_transport_gate` has a
> SECOND disclosed blind spot -- it keys on an array-library `.T`, so `reasoning/eprop.py`'s
> pure-loop hand-rolled BPTT is invisible to it entirely, and a module spelling backprop in Python
> loops passes silently on promotion; widening the rule is an owner decision and the wider W5 rule
> was already built and rejected at 59 findings. (b) Three module-level helpers in `ng_snn.py`
> (`_toy`'s dataset draw, the random-beta control, the poisson encoder) still build device-bound
> generators; they seed the SELF-CHECK's data rather than the brain's weights, so they were out of
> `[B1.06]`'s scope, but they carry the same cross-hardware hazard for the self-check's numbers.

Effort: 70 medium, 66 small, 36 large, 1 unknown

---

## Banding rule

`docs/POLICIES.md:3` -- *"P0 is the purpose; P1-P8 are how it is pursued without lying to
ourselves"* -- plus the cross-cutting invariants block, which is never relaxed by any policy.

- **[M.01] META - COLLAPSE RESULT: 38 source rows carried 17 duplicated defects; 226 input rows -> 141 ranked items + 2 new nodes the auditors missed**
  - Cross-source merges: gf-MLP backprop 3->1, arc cluster 4->1, latentq method-gate entry 3->1, cite-rot 4->1, and 13 further 2->1 (exempt-method TODOs, bare consolidate, microcircuit false claim, esp32 exemption, winner-key, GrowthCoordinator, curiosity flags, ledger re-run, safe_d, fear disposition, growth floor, graphify, quickstart-L7); backlog-40 alone appeared 5x under 4 keys plus 3 precondition rows under 3 keys.
- **[M.02] META - BANDING RULE: B1=invariant breach, B2=P0, B3=P1, B4=P2, B5=P3, B6=P4/P4b/P4c, B7=P5, B8=P6, B9=P7 (empty), B10=P8, B11=hygiene**
  - Re-derived the authority at HEAD: POLICIES.md:1542 'Cross-cutting invariants (never relaxed by any policy)' names exactly safety envelope / e-prop-not-backprop / determinism / verify-first; multi-policy items are banded at their LOWEST policy number (so P7+P2 -> B4), which is why B9 is empty; items the auditors tagged 'invariant' that constrain UNBUILT code are criteria-owed (P8 clause 1), not breaches, and sit in B10.
- **[M.03] META - CRITICAL PATH: by depth = sensor-feed -> charge-gate -> reflex-ch8 -> caution-matrix (4 deep, 12 dependents); by rank = selfbuild-proposer -> godel-attempt -> coverage-band0 (P0, root node MISSING from the inventory)**
  - Nine roots gate 62 of 141 items: sensor-feed-module-missing (12), blimp-exposure-lexicon-disjoint (8, widest, and its cheapest discharge is a DECISION not a build), cortical-prior-module-absent (8), per-module-task-metrics-missing (4), 014-baselines-unrecorded (3), selfbuild-proposer-empty (2, absent from inventory), cortical-microcircuit-no-input-port-false (2), gf-mlp-handwritten-backprop (1 large), lake-census-missing (1, misfiled as closed).

Within a band, order is (a) a mechanized gate over an unmechanized one, (b) what BLOCKS other
work, (c) cost. Per P3's honesty guard and P8 clause 5, **removing a false claim outranks adding
a capability** wherever the false claim is load-bearing on prioritisation -- which is why several
one-line documentation strikes sit above large builds.

## Top blockers

| blocks | dedupe_key |
|---:|---|
| 8 | `sensor-feed-module-missing` |
| 6 | `cortical-prior-module-absent` |
| 5 | `blimp-exposure-lexicon-disjoint` |
| 3 | `per-module-task-metrics-missing` |
| 3 | `neurogenesis-units-unplaced` |
| 3 | `014-baselines-unrecorded` |
| 2 | `selfbuild-proposer-empty` |
| 2 | `growth-ab-harness-absent` |
| 2 | `schema-induction-art-clusterer` |
| 2 | `language-organ-unbuilt` |

---

## Band 1 -- OPEN BREACH of a never-relaxed invariant

6 item(s).

### [B1.01] Strike CLAUDE.md's cross-cutting-invariants paragraph: it calls the e-prop rule 'all review-enforced' and names mlp_q + predictive_coding UNADJUDICATED, both false since the 015/018 lanes

- **status** DONE (uncommittable -- CLAUDE.md is gitignored) &nbsp;|&nbsp; **policy** invariant: e-prop &nbsp;|&nbsp; **effort** small
- **key** `claude-md-eprop-invariant-para-stale`
- **source** NEW - not in inventory; CLAUDE.md:50 vs docs/POLICIES.md:1548,:1635,:1686
- **evidence** Re-derived at HEAD myself: CLAUDE.md:50 still reads 'Two more are UNADJUDICATED ... (1) the dh step in basal_ganglia/mlp_q.py::MlpQ.update (2) cortex/predictive_coding.py:200-201' while POLICIES.md:1635 says RESOLVED 2026-08-14 and :1686 ADJUDICATED 2026-08-14 NOT a violation, and POLICIES.md:1548 now reads 'PARTIALLY mechanized since 2026-08-15' - a false claim about the one thing no lane may relax, and it is the first text every session reads.

### [B1.02] Adjudicate or port reasoning/gf.py::MLP's hand-written backprop - the tree's one ungoverned weight-transport finding, and the gate that reports it landed today

- **status** DONE (`326c8f68` adjudicated + `d880a7b0` exemption granted) &nbsp;|&nbsp; **policy** invariant: e-prop &nbsp;|&nbsp; **effort** medium
- **key** `gf-mlp-handwritten-backprop`
- **source** specs/BACKLOG_RANKED.md rows 40-precondition x2 + 40-PRE; src/snn_data_lake/substrate/eprop_transport_gate.py
- **evidence** COLLAPSED 3 rows; three auditors independently ran eprop_transport_gate at HEAD -> GOVERNED 0 / UNGOVERNED 1 = reasoning.gf.train:1795, and POLICIES.md:1581-1584 (read at HEAD) records the same one report plus 'it becomes a governed violation the day gf.py is promoted'; MLP is a baseline with one construction site (gf.py:1814) and zero refs outside the file.

### [B1.03] Guard the idle sentinel in the consumer: a negative distance is a MISSING reading, and today it binds a fear association as real contact

- **status** DONE (`2f9eba7b`) &nbsp;|&nbsp; **policy** invariant: safety envelope &nbsp;|&nbsp; **effort** small
- **key** `idle-sentinel-minus-one`
- **source** specs/014-live-sensor-transduction/tasks.md T029,T030,T031
- **evidence** RE-DERIVED LIVE at HEAD (and the auditor's blocked_by is wrong - see F.03): substrate/esp32_encoder_path.py:78 clamps distance_m with no sentinel while :81 special-cases temp_k<=0, and brainstem/autonomous_loop.py:1473 `distance < CONTACT` reaches `am.pair(pat)` at :1479 for a finite -1.0; the file is in substrate/, not cortex/ as several cites imply.

### [B1.04] RE-BANDED to [B10.42] -- it was never an invariant breach

- **status** MOVED 2026-08-15, work DONE (`4a2ae912`) &nbsp;|&nbsp; **key** `raw-metres-vs-normalized-distance`
- **why** The row entered Band 1 on the word "loosens", and measurement refuted that word. Banding is
  DERIVED from policy, so a row whose evidence changes must be RE-DERIVED, not annotated in place --
  the earlier pass flagged it here and left it, which meant the highest-priority band carried an item
  that no invariant covers. Corrected evidence, corrected band. The full entry, its correction and
  its commit now live at **[B10.42]** in Band 10; the stub stays so the id resolves and so the
  mis-banding is part of the record rather than erased.

### [B1.05] Pin DEFAULT_RULES[6]/[7] values AND meanings in a test with a demonstrated layout-shift red - nothing in the tree pins them today

- **status** DONE (`6e506235`) &nbsp;|&nbsp; **policy** invariant: safety envelope &nbsp;|&nbsp; **effort** small
- **key** `reflex-channel-6-7-unpinned`
- **source** specs/014-live-sensor-transduction/tasks.md T049,T050
- **evidence** VERIFIED myself at HEAD: brainstem/reflex.py:22-25 defines ch6 (-50,800,'STOP') and ch7 (-300,300,'WITHDRAW'), and `grep -c DEFAULT_RULES tests/test_reflex.py` = 0 against a positive control of a file that does exercise Reflex - a channel renumber changes what STOP means with every gate green.

### [B1.06] Draw grown weights from a CPU generator - same seed must give the same brain on any hardware; the breach is masked, not fixed

- **status** DONE (`0afafd40`) -- three module-level `ng_snn` helpers remain device-bound, disclosed and out of scope &nbsp;|&nbsp; **policy** invariant: determinism &nbsp;|&nbsp; **effort** medium
- **key** `device-bound-rng-nondeterminism`
- **source** specs/010-policy-compliance-audit/tasks.md T048 (C24)
- **evidence** Cite RE-DERIVED exact: hippocampus/ng_snn.py:169 `torch.Generator(device=self.device).manual_seed(seed)` consumed by randperm at :320; auditor filed it P5 but CLAUDE.md's determinism block makes it an invariant, and neurogenesis_engine.py:198-199's select() routing flips on its own as width crosses 262,144.

## Band 2 -- P0, the purpose (outranks P1-P8 explicitly)

15 item(s).

### [B2.01] Correct the two P0 rows still calling growth retention unmeasured - they point P0 effort at work that shipped

- **status** DONE -- both rows corrected; the ENGINE restart is proven, the BRAIN restart is not (that is [B2.09]) &nbsp;|&nbsp; **policy** P0 + P4b honesty &nbsp;|&nbsp; **effort** small
- **key** `growth-retention-denial-vs-shipped-proof`
- **source** specs/BACKLOG_RANKED.md row 9
- **evidence** VERIFIED in the CLAUDE.md text at HEAD: 'no cross-process restart has been demonstrated ... That measurement is the open P0 work here', with POLICIES.md:117 carrying the twin, against neurogenesis_engine.py:691/:700 which assert exactly that restart; a false claim sitting on the P0 build list outranks every capability item in this band.

### [B2.02] Retire the false 'no input port' reason at all 4 sites - it justifies a permanent EXEMPT and sizes a P0 avenue against a port that has existed since 2026-08-04

- **status** DONE (`3107dcec`) -- corrected at all 4 sites; the port EXISTS, the true reason is that nothing drives it &nbsp;|&nbsp; **policy** P0 avenue + P3 &nbsp;|&nbsp; **effort** small
- **key** `cortical-microcircuit-no-input-port-false`
- **source** specs/BACKLOG_RANKED.md row 8; specs/011-h01-neurogenesis-prior/tasks.md T041
- **evidence** COLLAPSED 2 rows; VERIFIED in CLAUDE.md at HEAD ('.simulate(t_ms) take no input signal anywhere ... EXEMPT-permanent at dormancy_gate.py:100-104 because it has no input port'), plus POLICIES.md:118, dormancy_gate.py:222-226 and specs/011 tasks.md:124, while the port is real at cortical_microcircuit.py:241 and consumed at :287 - and the CLAUDE.md cite :100-104 has itself rotted onto gwt_deep.

### [B2.03] Fill the self-build proposer - the route fires and proposes ZERO candidates, and CLAUDE.md itself says this is the open P0 work, not a call-site wire

- **status** OPEN &nbsp;|&nbsp; **policy** P0 &nbsp;|&nbsp; **effort** large
- **key** `selfbuild-proposer-empty`
- **source** NEW - named as a blocker by backlog-13 but absent from the inventory; CLAUDE.md P0 block, tests/test_selfbuild_route.py
- **evidence** Read at HEAD in CLAUDE.md's P0 row: 190 ticks under NORD_SELFMOD=1 leave _selfbuild_last {'night':8,'proposed':0,'kept':0} so SelfModHarness is never constructed; every auditor cited this key as backlog-13's blocker and no auditor filed the node, leaving the P0 critical path headless.

### [B2.04] Couple growth to the learning path (e-prop only) so grown units stop emitting zero - and strike the stale BLOCKED annotation that hides it

- **status** OPEN &nbsp;|&nbsp; **policy** P0 &nbsp;|&nbsp; **effort** large
- **key** `grown-units-emit-zero`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T014
- **evidence** tasks.md:205-206 still marks this 'NOT to be attempted as written' behind T013, a blocker spec.md amendment 3 and amendment 4 withdrew ('growth was never dormant. Withdrawn for the second and final time') and tasks.md never propagated - the annotation is the reason this P0 item has sat untouched.

### [B2.05] Feed a real distance to the always-safe argument at the LIVE site - the caller constrains its own guard, so threat() can never fire

- **status** OPEN &nbsp;|&nbsp; **policy** P0 &nbsp;|&nbsp; **effort** medium
- **key** `live-caller-passes-safe-d`
- **source** specs/014-live-sensor-transduction/tasks.md T036 + its Notes block
- **evidence** COLLAPSED 2 rows; RE-DERIVED myself: cortex/brain.py:1067 `sd = SAFE_DISTANCE` builds all three _explore bands, consumed at brain.py:1316, via live_brain.py:90 - while the task's named site autonomous_loop.py:2269 is inside main(), the module self-check, so the lane's central premise cites a self-check.

### [B2.06] Re-resolve the ~13 drifted cites in CLAUDE.md's P0 block - the P0 block is the BUILD LIST, so a rotted cite mis-sizes the work

- **status** OPEN &nbsp;|&nbsp; **policy** P0 &nbsp;|&nbsp; **effort** small
- **key** `claude-md-p0-block-cite-rot`
- **source** specs/BACKLOG_RANKED.md row 15
- **evidence** 11 of ~13 stale and some drifted further (neurogenesis_engine.py:211/:250/:279/:321, brain.py:3767/:3780/:2722/:1075, home.py:68/:231/:280, dormancy_gate.py:100-104 now gwt_deep); the prescribed replacements DO resolve, and citation_guard.py:122's regex needs a path separator and discards the :NNN suffix, so the guard cannot catch these.

### [B2.07] Restore self-recalibration to CLAUDE.md's P0 avenue list and make 'future avenues' falsifiable

- **status** OPEN &nbsp;|&nbsp; **policy** P0 &nbsp;|&nbsp; **effort** large
- **key** `avenue-list-diverged-claude-vs-policies`
- **source** specs/BACKLOG_RANKED.md row 17
- **evidence** POLICIES.md:119-120 carry self-recalibration -> substrate/signal_calibrator.py and 'open by construction' while CLAUDE.md:26-30 lists four avenues and none of them; 0 of 30 tools/*.py contain 'avenue' against a positive control on 'def ', so no gate can see the divergence.

### [B2.08] Drive cortical_microcircuit.thalamic_drive from a live path

- **status** UNBLOCKED; drive MEASURED, consumer deliberately NOT wired &nbsp;|&nbsp; **policy** P0 avenue &nbsp;|&nbsp; **effort** medium
- **key** `cortical-microcircuit-unwired-port` &nbsp;|&nbsp; ~~blocked by `cortical-microcircuit-no-input-port-false`~~ -- discharged `3107dcec`
- **MEASURED 2026-08-16** Paired A/B, 12 seeds, driven vs undriven at the SAME seed, with a NULL control (`driven(0.0)` vs undriven = exactly 0.0, so the pairing is real). At the paper's 15 Hz: **L4e t=3.1 (10/12 seeds), L4i t=3.0 (9/12)** -- the direct thalamo-recipients -- **plus L2/3i t=3.0**, which has ZERO thalamic in-degree and can only have responded through the INTRACORTICAL route. At 50 Hz all four recipients respond (L4e t=8.8, 12/12) plus L5e. Dose-response monotone: L4e +1.1 -> +2.0 -> +6.1 across 1/15/50 Hz. **The drive charges the circuit.** COST: the port is 0.125 ms, but `simulate(t_ms=5)` is 10.5 ms against a ~27.8 ms tick, so the circuit cannot run per-wake-tick and belongs where the brain already does expensive work.
- **a FIRST harness produced a FALSE NULL, recorded rather than deleted.** It compared a WITHIN-seed delta against the BETWEEN-seed range and reported 0/8 beyond noise at EVERY rate. Those are different quantities: at 771 neurons some populations hold ~10 cells, so that band is 6-23 Hz wide and swallows a real effect. Caught only because every delta was positive, which noise does not do. Had it shipped, this item would have been DELETED under P8 clause 5 on a measurement error.
- **source** specs/BACKLOG_RANKED.md row 10
- **evidence** Every construction is inside its own file (:460,:468,:518,:546,:551) and thalamic_drive is called only at :287/:546/:551; __init__.py:102 is a REGIONS name-string and run_selfchecks.py:69 a registration - zero external drivers, and the backlog says fix row 8 first so the lane is sized against a true reason.

### [B2.09] Prove growth survives a BRAIN restart, not only an engine restart

- **status** OPEN &nbsp;|&nbsp; **policy** P0 + P4b &nbsp;|&nbsp; **effort** medium
- **key** `growth-restart-proven-only-at-engine-level`
- **source** specs/BACKLOG_RANKED.md row 11
- **evidence** neurogenesis_engine.py has zero occurrences of 'Brain(' or 'make_brain' across 756 lines; _cross_process_restart :428 driven at :607, live paths brain.py:3101 save_growth and :1176 restore both resolve, and brain.py:4193 def close() has no growth flush.

### [B2.10] Promote reasoning/godel_machine.py + reasoning/self_improve.py into the governed tier - and fix the gate that punishes the fix

- **status** OPEN &nbsp;|&nbsp; **policy** P4c blocking P0 &nbsp;|&nbsp; **effort** large
- **key** `godel-selfimprove-ungoverned`
- **source** specs/BACKLOG_RANKED.md row 12
- **evidence** Precondition confirmed by running the gate: import_manifest prints '6 ungoverned modules / 8 call sites' and substrate/import_manifest.py:309 asserts len(man) >= 5, so promoting both leaves 4 and REDS it - the same shape as running a gate only against regressions; pyproject omit is at :323 not :322.

### [B2.11] Wire godel_machine.attempt_on_repo to a caller - after the proposer proposes something

- **status** BLOCKED &nbsp;|&nbsp; **policy** P0 &nbsp;|&nbsp; **effort** medium
- **key** `godel-attempt-on-repo-never-invoked` &nbsp;|&nbsp; **blocked by** `selfbuild-proposer-empty`
- **source** specs/BACKLOG_RANKED.md row 13
- **evidence** Zero call sites tree-wide (only godel_machine.py:172 def plus docstrings); POLICIES.md:115 forbids the force-wire while the route proposes ZERO, and the sub-cite self_mod_harness.py:94 is STALE (that line is `import shutil`, the import is :102).

### [B2.12] Wire or retire Brain.self_improve - no gate can see it because 'improve' is not a learning verb

- **status** OPEN &nbsp;|&nbsp; **policy** P0 &nbsp;|&nbsp; **effort** medium
- **key** `brain-self-improve-dead-entrypoint`
- **source** specs/BACKLOG_RANKED.md row 14
- **evidence** Scripted scan of src/tests/tools for `.self_improve(` returns 0 call sites against brain.py:4173's lone def; method_dormancy_gate.py:48 LEARN_VERBS omits 'improve', so this method is invisible to the per-method gate by construction.

### [B2.13] Fix the vacuous --rerun comparison FIRST, then re-run the 13 stale ledger artifacts and disclose non-reproducers

- **status** OPEN &nbsp;|&nbsp; **policy** P0 &nbsp;|&nbsp; **effort** medium
- **key** `ledger-rerun-vacuous-and-undone` &nbsp;|&nbsp; **blocked by** `exempt-method-standing-todos`
- **source** specs/001-charge-the-dormant/tasks.md T027; specs/BACKLOG_RANKED.md row 66
- **evidence** COLLAPSED 2 rows; `--rerun time_cells_ab_eprop` printed 'reproduces' comparing recorded observables () to now () - a negative the harness cannot produce - while `score --check` warns 13 rows measured at 8b6aff77 (now 230 commits back) and exits 0; coverage_score.py:394 confirms an honest re-record reddens the gate, which is why it has not happened.

### [B2.14] Record the SC-006 gate result (all four green, none modified to make them so)

- **status** OPEN &nbsp;|&nbsp; **policy** P0 &nbsp;|&nbsp; **effort** small
- **key** `spec001-sc006-record-missing`
- **source** specs/001-charge-the-dormant/tasks.md T031
- **evidence** Auditor measured all four at HEAD 39c412f9 (selfchecks 301/301, pytest 91.27%, ruff clean, coverage_score --check exit 0) and grep for 'SC-006\|none was modified' over specs/001 and docs/coverage returns only the spec text - the work is done and the record is not.

### [B2.15] Move the 12 coverage rows scoring band 0 - they are the P0 build list and are ~194 commits old

- **status** OPEN &nbsp;|&nbsp; **policy** P0 build list &nbsp;|&nbsp; **effort** large
- **key** `coverage-band0-rows-stale` &nbsp;|&nbsp; **blocked by** `selfbuild-proposer-empty`
- **source** specs/BACKLOG_RANKED.md row 16
- **evidence** All 12 cites re-derived exact and still band 0 / measured_at 2026-07-30 (Affect:18, Executive:21, Gf:27-28, Imagination:16-17, Language:19, Perception:40-41, Social:17, WholeBrain:27-28), and WholeBrain.md:28 godel_machine_selfbuild=0 corroborates the P0 chain root above.

## Band 3 -- P1 brain-native / no-hardcode

12 item(s).

### [B3.01] Replace the two fixed 0.5 decision thresholds with the running-baseline forms already in the same file

- **status** OPEN &nbsp;|&nbsp; **policy** P1 &nbsp;|&nbsp; **effort** medium
- **key** `fixed-half-threshold-vs-running-baseline`
- **source** specs/010-policy-compliance-audit/tasks.md T043 (C17) + T044 (C18)
- **evidence** COLLAPSED 2 rows; prefrontal_executive.py:87/:100/:262 gate_threshold=0.5 resolves exactly, brain.py's `> 0.5` novelty test is at :2147 NOT the cited :2040 (ROTTEN), and the remedy idiom sits 260 lines above it at brain.py:1886 `> self._nov_base` with the _margin_base EMA at :2102-2105.

### [B3.02] Stop quantising the float elapsed estimate into _learned_latency, then re-measure

- **status** OPEN &nbsp;|&nbsp; **policy** P1 &nbsp;|&nbsp; **effort** small
- **key** `learned-latency-int-quantiser` &nbsp;|&nbsp; **blocked by** `growth-ab-harness-absent`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T009 + T010
- **evidence** COLLAPSED 2 rows (T010 verifies T009); defect LIVE as `max(1, int(round(el)))` but at cortex/brain.py:1754, NOT the cited :1603 which is a comment inside the elig dict - the cite is ROTTEN.

### [B3.03] Supply real per-tract conduction delays for the 83 connectome edges - every projection is a uniform 1 tick

- **status** OPEN &nbsp;|&nbsp; **policy** P1 &nbsp;|&nbsp; **effort** large
- **key** `connectome-delays-uniform-1`
- **source** specs/BACKLOG_RANKED.md row 7
- **evidence** Re-derived: EDGES=83 with delay histogram Counter({1: 83}); connectome.py:96-99's own obligation text, brain.py:920 ConnectomeRouter ctor and :2288 router.step(drive) all resolve exactly, and tools/ab_connectome_delay.py already exists to score it.

### [B3.04] Build degradable task metrics for the 9 of 12 connectivity owner classes that have none - it gates three other 009 items

- **status** OPEN &nbsp;|&nbsp; **policy** P1 (auditor said invariant; it is an unmet measurement, not a breach) &nbsp;|&nbsp; **effort** large
- **key** `per-module-task-metrics-missing`
- **source** specs/009-sparse-connectivity/tasks.md T019
- **evidence** research.md:14-38 establishes the pattern for EpisodeConsolidator alone (retained() + two nulls at 0.0000) and no per-module metric registry or equivalent exists for any other owner class in src/ or tests/; without it 'thin it' has no before/after.

### [B3.05] Derive and A/B a resolution floor for substrate/, hippocampus/, neuromodulation/, plasticity/, cerebellum/, reasoning/

- **status** OPEN &nbsp;|&nbsp; **policy** P1 &nbsp;|&nbsp; **effort** large
- **key** `per-region-growth-floor-underived`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T026-T031 + T035
- **evidence** COLLAPSED 7 rows (6 per-region + T035 which shares the key); `grep -rl 'resolution floor' docs/ specs/ src/ tools/` returns ONLY specs/007's own four documents - zero floors derived anywhere, and 'no derivable floor' was never recorded as an outcome either.

### [B3.06] Implement top-k-per-row wiring and confirm it holds F* 96-128 under proportional load

- **status** BLOCKED &nbsp;|&nbsp; **policy** P1 &nbsp;|&nbsp; **effort** medium
- **key** `topk-per-row-unimplemented` &nbsp;|&nbsp; **blocked by** `episode-rate-instrument-dormant`
- **source** specs/009-sparse-connectivity/tasks.md T016
- **evidence** episode_consolidation.py:207-230 `_wire()` implements uniform random k-out only and a tree-wide grep for top-k-per-row returns nothing, so the topology research.md:93 says the design rests on does not exist.

### [B3.07] Re-derive F* at N=4,096 and N=16,384 with fanout_bisect and verify 4x beyond the tuned width

- **status** OPEN &nbsp;|&nbsp; **policy** P1 &nbsp;|&nbsp; **effort** medium
- **key** `fstar-not-re-derived-with-tool`
- **source** specs/009-sparse-connectivity/tasks.md T014 + T015
- **evidence** COLLAPSED 2 rows; tools/fanout_bisect.py exists and sweep() is generic but its only entrypoint _brain_config() (:127-160) pins the brain's derived dim=256, no 4,096/16,384 run is recorded in specs/009, and research.md:158-164 still labels every rung above 10^4 EXTRAPOLATED.

### [B3.08] Thin predictive_coding, jepa_world_model and criticality_regulator to bounded fanout

- **status** BLOCKED &nbsp;|&nbsp; **policy** P1 &nbsp;|&nbsp; **effort** large
- **key** `modules-not-thinned` &nbsp;|&nbsp; **blocked by** `per-module-task-metrics-missing`
- **source** specs/009-sparse-connectivity/tasks.md T021,T022,T023
- **evidence** COLLAPSED 3 rows (three sites, one defect); a tree-wide fanout grep over src/ returns only r_stdp, neuronal_avalanches, episode_consolidation, function_preserving_growth, growable_readout, growable - none of these three carries any wiring bound.

### [B3.09] Record every module that genuinely needs dense connectivity as a DOCUMENTED EXCEPTION (FR-003)

- **status** BLOCKED &nbsp;|&nbsp; **policy** P1 &nbsp;|&nbsp; **effort** medium
- **key** `documented-exceptions-empty` &nbsp;|&nbsp; **blocked by** `per-module-task-metrics-missing`
- **source** specs/009-sparse-connectivity/tasks.md T025
- **evidence** research.md:170 still reads 'None confirmed yet' at HEAD, and with 11 of 12 owner classes unmeasured the exception set cannot be populated, so SC-007's K is undefined.

### [B3.10] Thin sleep_replay AttractorNet after resolving its contaminated (oracle) evidence

- **status** BLOCKED &nbsp;|&nbsp; **policy** P1 + P8 &nbsp;|&nbsp; **effort** medium
- **key** `attractornet-oracle-metric` &nbsp;|&nbsp; **blocked by** `per-module-task-metrics-missing`
- **source** specs/009-sparse-connectivity/tasks.md T020
- **evidence** research.md:172-174 and POLICIES.md:1458 both still record the assembly-derived mask as an ORACLE scoring 1.000 untrained, and hippocampus/sleep_replay.py carries no fanout/mask parameter, so no thinning has been attempted.

### [B3.11] Read arm_config.CONTACT as the simulator danger limit; define no new safety constant

- **status** BLOCKED &nbsp;|&nbsp; **policy** P1 &nbsp;|&nbsp; **effort** small
- **key** `danger-limit-reads-arm-config-contact` &nbsp;|&nbsp; **blocked by** `sensor-feed-module-missing`
- **source** specs/014-live-sensor-transduction/tasks.md T015
- **evidence** Cite RE-DERIVED: brainstem/arm_config.py:27 CONTACT = 0.04 exists and imports cleanly; nothing reads it from a simulator because no simulator exists yet.

### [B3.12] Seed the grown population from cortical_prior once at construction and score SC-005's three arms

- **status** BLOCKED &nbsp;|&nbsp; **policy** P1 &nbsp;|&nbsp; **effort** medium
- **key** `prior-seeding-absent` &nbsp;|&nbsp; **blocked by** `cortical-prior-module-absent`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T030 + T031
- **evidence** COLLAPSED 2 rows; no seeding path exists because cortical_prior.py is absent, and T030's read-once-into-mutable-state is the P1 movability mechanism the entire feature rests on.

## Band 4 -- P2 adopt-better or merge, then retire

26 item(s).

### [B4.01] Supply a grounded exposure source whose sentences contain function words, or accept spec 005 as permanently terminal - the widest single blocker at 8 items

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `blimp-exposure-lexicon-disjoint`
- **source** specs/005-language-organ/tasks.md T011 stop (gates T013-T070)
- **evidence** Re-ran tools/lang_coverage.py at HEAD: exit 2, 0 of 67 paradigms, 0 of 67000 pairs, 605 triples / 508-word lexicon, 39 of 2637 BLiMP word types - byte-identical to the 95c3f806 record 13 days on; the terminal-record branch is SMALL and discharges 8 blocked rows, so rank it first in this band on unblocking-per-cost.

### [B4.02] Retire the dense wiring path and wire sparse connectivity into the live brain - the P1 module runs 100% all-to-all today

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `sparse-wiring-not-live-in-brain`
- **source** specs/009-sparse-connectivity/tasks.md T026
- **evidence** Re-derived with a positive control in the same read: episode_consolidation.py:157/197-198 carry fanout=None -> _mask=None and the ONLY live construction (brain.py:961-964) passes capacity+seed and no fanout, while fanout_bisect.py:148 does pass it; SC-006 = 2 mechanisms.

### [B4.03] Select the single surviving growth mechanism (SC-007 = exactly 1) with the A/B that justifies it - the count is 7+

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `multiple-growth-mechanisms-p2`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T025
- **evidence** At least 7 live at HEAD: time_cells.grow:310/grow_if_earned:434, growable_readout.grow:227/_grow_one:266/:557, developmental_loop_brain.grow:191, laminar_local_readout.grow:201, growing_learner.grow_dim:101, multimodal_encoder.add_modality:102, plus neurogenesis_engine.

### [B4.04] Run the M25 decision gate on growable.py's two seams and retire GrowthCoordinator/register_seam - the result IS 0, so it is decidable now

- **status** OPEN &nbsp;|&nbsp; **policy** P2 + P3 shape 7 &nbsp;|&nbsp; **effort** medium
- **key** `growable-growthcoordinator-zero-refs`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T023+T024; specs/BACKLOG_RANKED.md row 45
- **evidence** COLLAPSED 2 rows; all 11 GrowthCoordinator/register_seam hits are inside substrate/growable.py itself, the one importer growing_learner.py:37 takes only grow_vector, and M25 is still live at AUDIT_BACKLOG.md:103,:283 - one cite is ROTTEN, multimodal_encoder.py:198 is an assert inside main(), not a growth seam.

### [B4.05] Retire the superseded curiosity drive and collapse the three action-value substrates - a flag that selects between mechanisms is additive with extra steps

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `duplicate-substrates-behind-flags`
- **source** specs/010-policy-compliance-audit/tasks.md T046 (C20) + T047 (C21); specs/BACKLOG_RANKED.md row 47
- **evidence** COLLAPSED 3 rows; autonomous_loop.py:368-372 (SpikingCuriosityDrive elif CuriosityDrive) and :1570-1586 (latent_q / mlp_q / tabular) both resolve, and the two learning_progress defs sit at intrinsic_motivation.py:122 and spiking_curiosity.py:244 - but row 47's 'overclaiming docstring' sub-claim is STALE, :8-16 is already an honest-framing block.

### [B4.06] Retire one of the two live Wang-2002 spiking decision attractors

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `wang-attractor-duplicated`
- **source** specs/010-policy-compliance-audit/tasks.md T045 (C19)
- **evidence** Two independent implementations: basal_ganglia/evidence_accumulation.py:311 and cortex/prefrontal_executive.py:117-131, the second hand-rolling its own pool sizes, weights and threshold and importing nothing but random (:73) - it does not delegate.

### [B4.07] Resolve the two Global Workspaces that both run every tick - and strike the row's own correction, which is the stale claim

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `two-global-workspaces-additive`
- **source** specs/BACKLOG_RANKED.md row 46
- **evidence** Debt is self-declared at brain.py:1113-1117 and the cite resolves, but the row's 'really 13 vs 1' does NOT re-derive: explicit-import files at HEAD are GlobalWorkspace 8 / ReentrantWorkspace 2, so POLICIES' 8-vs-3 reproduces; the POLICIES cite has drifted :290-293 -> ~:293-295.

### [B4.08] Promote or triage the ARC / fluid-reasoning cluster - gf.py, its named 'pure leaf', converts an ungoverned report into a governed red

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4c + invariant: e-prop &nbsp;|&nbsp; **effort** large
- **key** `arc-fluid-cluster-promotion` &nbsp;|&nbsp; **blocked by** `gf-mlp-handwritten-backprop`
- **source** specs/BACKLOG_RANKED.md row 40 (4 duplicate rows under 4 keys)
- **evidence** COLLAPSED 4 rows; gf.py 92,598 B and reasoning/arc_agi 6 modules both exact, zero governed importers across all 11 cluster modules, p4c_ratchet reasoning live=108 baseline=108 delta=+0 - the honest disposition is adjudication or port, never a silent promotion.

### [B4.09] Delete the schema_induction ART clusterer and adopt the groupby, with a max_prototypes cap

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** medium
- **key** `schema-induction-art-clusterer`
- **source** specs/BACKLOG_RANKED.md row 31
- **evidence** All three cites exact (:199 vigilance=0.7, :270 _art_cluster, :455 _selfcheck) and the caveat holds - max_prototypes=256 is enforced at :297, so the bare groupby still owes a cap; the 8-store neuter A/B was NOT re-run by the auditor.

### [B4.10] Discharge M21 - the duplicated LinearReadout head and the per-image _cache_features loop

- **status** BLOCKED &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** medium
- **key** `m21-duplicated-linear-readout` &nbsp;|&nbsp; **blocked by** `promote-developmental-loop-family`
- **source** specs/BACKLOG_RANKED.md row 39
- **evidence** developmental_loop_brain.py:150 _LinearReadout and _loop_common.py:35 LinearReadout both exact with the M21 note at :155-160 confirming the block is structural; autapse_neuron.py:112 is correctly excluded and _cache_features has DRIFTED :207 -> :218.

### [B4.11] Settle the research/ vs reasoning/ duplicate subpackages - 15 same-named twins, none byte-identical

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `research-reasoning-duplicate-subpackages`
- **source** specs/BACKLOG_RANKED.md row 41
- **evidence** Re-derived by sha1: research/ng_snn 9 twins of 11 + research/arc_agi 6 of 6 = 15, byte-identical 0; the governed copy reasoning/arc_agi/run_arc.py:19 still tells the reader to run `python research/arc_agi/run_arc.py`.

### [B4.12] Extract the shared ng_snn nmnist clone set (MERGE, not ADOPT - the 2-conv/3-conv difference is deliberate)

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** medium
- **key** `ngsnn-nmnist-clone-set`
- **source** specs/BACKLOG_RANKED.md row 42
- **evidence** AST body-hash: 13 identical top-level defs between nmnist_spatiotemporal.py and _3conv.py, 6 three-way with nmnist_pipeline.py, mirrored in research/; _3conv.py:4-8 is the deliberate WHY block and all three fail identically on `from softhebb_mnist import ...`.

### [B4.13] Extract the shared _HJepa from the hjepa torch clones - 17 identical-body def groups

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** medium
- **key** `hjepa-torch-clone-17-groups`
- **source** specs/BACKLOG_RANKED.md row 49
- **evidence** AST body-hash re-derived: exactly 17 identical groups between hjepa_arm_torch.py and hjepa_torch.py, with vicreg_terms bit-identical three-way against jepa_scaffold_torch.py:153; only drift is AUDIT_BACKLOG.md:244 now reading 'Four kinds'.

### [B4.14] Merge gf.py's five inducers - the file count fell and the mechanism count did not

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `gf-five-inducers-unmerged`
- **source** specs/BACKLOG_RANKED.md row 48
- **evidence** Every cite exact: gf.py:1 'COMBINED (not merged) into one file', :10-11 'the five distinct implementations coexist', three constant sets at :50/:324/:1329 and three evaluators at :65/:334/:1334; spike_chat_lake.py imports four families at :330,:542,:586-588,:653.

### [B4.15] Collapse the duplicated _World harness (+_clamp01/_cosine/_bucket_key) - cheapest P2 in the band

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** small
- **key** `world-harness-duplicate-metacog-wholistic`
- **source** specs/BACKLOG_RANKED.md row 50
- **evidence** All four cites exact: class _World at metacognitive_monitor.py:442 and wholistic_loop.py:268, constructed at wholistic_loop.py:323, and brain.py:426/:427 build WholisticLoop twice on the birth path; wholistic_loop.py:80 already imports from that module, so the collapse adds no dependency and no cycle.

### [B4.16] Import TwoLayerNet instead of copying it, and dedup _env/_envf in the same file

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** small
- **key** `twolynet-copied-not-imported`
- **source** specs/BACKLOG_RANKED.md row 51
- **evidence** class TwoLayerNet at deep_stdp_latency.py:282 AND plasticity/deep_stdp.py:240, while deep_stdp_latency.py:97 already imports four other names from that exact module; second copies at :110/:115 where plasticity/deep_stdp.py now has neither.

### [B4.17] Deduplicate _Rng across two CORE cortex modules

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** small
- **key** `rng-duplicate-two-core-cortex`
- **source** specs/BACKLOG_RANKED.md row 52
- **evidence** class _Rng at spiking_lm_serve.py:85 and trained_brain.py:39, both CORE-registered at run_selfchecks.py:81 and :88 - every cite in this row survived the drift.

### [B4.18] Deduplicate _datasets across the two training MNIST loaders

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** small
- **key** `datasets-duplicate-mnist-loaders`
- **source** specs/BACKLOG_RANKED.md row 53
- **evidence** def _datasets(root, download=True) at training/mnist.py:111 and training/mnist_genn.py:99; the blocker re-derives too - mnist.py:34 imports snntorch at module top and mnist_genn.py:57-66 is the ml_genn block, so the shared module must import torchvision only.

### [B4.19] Scope SC-010 before adding it - a second layered-population mechanism already exists

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** small
- **key** `second-layered-mechanism-laminar-readout`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T027
- **evidence** cortex/laminar_local_readout.py:37-46 defines a named L4 -> L2/3 -> L5 -> L6 motif with per-layer populations and its own grow(), so 'exactly one layered-population mechanism exists' is red on arrival unless the criterion says what it counts.

### [B4.20] Name the adopt-or-merge direction against the hippocampal recruit rules

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** medium
- **key** `schema-induction-recruit-rule-direction` &nbsp;|&nbsp; **blocked by** `schema-induction-art-clusterer`
- **source** specs/BACKLOG_RANKED.md row 54
- **evidence** schema_induction.py:32-41's 'WHY ART-1, NOT spiking_som.SOM' block resolves but names only the ungoverned SOM, never the CORE growing_learner comparison; the row's run_selfchecks.py:141 cite has DRIFTED to :148.

### [B4.21] Adopt pillow in arm_view.encode_png and delete the hand-rolled PNG encoder

- **status** OPEN &nbsp;|&nbsp; **policy** P2 + P7 &nbsp;|&nbsp; **effort** medium
- **key** `pillow-handrolled-png-encoder`
- **source** specs/BACKLOG_RANKED.md row 68
- **evidence** arm_view.py:59 encode_png and :87 the raw PNG magic both exact with zero `from PIL` imports under src/; POLICIES.md:1313 still calls pillow 'a real incident', and dependency_ledger.EXEMPT['pillow'] is at :150-163 not the cited :484-498.

### [B4.22] Build cortex/language_organ.py (Area/cap-k/derivations/expose/acceptability/save-load) plus its 9-test suite and registration

- **status** BLOCKED &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `language-organ-unbuilt` &nbsp;|&nbsp; **blocked by** `blimp-exposure-lexicon-disjoint`
- **source** specs/005-language-organ/tasks.md T019-T034
- **evidence** COLLAPSED 16 rows (one absent file); neither src/snn_data_lake/cortex/language_organ.py nor tests/test_language_organ.py exists at HEAD and Phase 4 is downstream of the T011 stop the spec states in advance.

### [B4.23] Build tools/lang_organ_ab.py: exposure split, four arms, ADOPT conjunction, density block, latency, verdict

- **status** BLOCKED &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `lang-organ-ab-harness-absent` &nbsp;|&nbsp; **blocked by** `blimp-exposure-lexicon-disjoint`
- **source** specs/005-language-organ/tasks.md T035-T037,T039-T047,T064
- **evidence** COLLAPSED 13 rows (one absent file); tools/ holds lang_coverage, lang_diag, lang_probe, lang_scale and no lang_organ_ab.py, and T039's first act loads pair_ids that coverage.md records as NONE.

### [B4.24] ADOPT-only retire of cortex/language_acquisition.py: port expose/persistence, swap _lm_serve, DELETE, verify the reply changes

- **status** BLOCKED &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `language-acquisition-not-retired` &nbsp;|&nbsp; **blocked by** `blimp-exposure-lexicon-disjoint`
- **source** specs/005-language-organ/tasks.md T051,T052,T054,T055,T056
- **evidence** COLLAPSED 5 rows; Phase 6 is explicitly ADOPT-ONLY, no A/B ran, and language_acquisition.py is live at brain.py:4062 and TORCH_GATED at run_selfchecks.py:414 - deleting it without the A/B is what P2's honesty guard forbids.

### [B4.25] Pluggable PHON encoder surface with the snnlp/poisson/graphemefold/rbfpop arms

- **status** BLOCKED &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** large
- **key** `phon-encoder-arms-absent` &nbsp;|&nbsp; **blocked by** `language-organ-unbuilt`
- **source** specs/005-language-organ/tasks.md T061,T062,T063,T065
- **evidence** COLLAPSED 4 rows; all four target the absent language_organ.py, though both reuse targets resolve at HEAD (language_acquisition.py:83 _encode_ctx, population_encoding.py:25 PopulationEncoder).

### [B4.26] Route readings through the EXISTING parse->encode->spikes chain and verify no second parser appeared - a constraint on future work, green and vacuous today

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** medium
- **key** `single-transduction-mechanism` &nbsp;|&nbsp; **blocked by** `sensor-feed-module-missing`
- **source** specs/014-live-sensor-transduction/tasks.md T035 + T067
- **evidence** COLLAPSED 2 rows; all four cites re-derive correct (esp32_encoder_path.py:33/:66/:91/:120) and `git diff --name-only main` on that file is empty, so T067 asserts nothing until the feed lands - carry it as a commit-time rule, not a task.

### ADDED 2026-08-15 -- the benchmarks the BLOCKED half of P2 is waiting on

> Hand-authored, not generated (FR-001 covers the 173 inventory items above). Added after the P2
> merge program split its work in two by measurement: LITERAL DUPLICATES are decidable now and two
> are already landed (`[B4.17]` `_Rng` 2->1 at `53ffa064`; `[B4.15]`'s `_clamp01` 8->1 at
> `8f3f93f3`), while COMPETING MECHANISMS -- `[B4.03]`, `[B4.05]`, `[B4.06]`, `[B4.07]` -- are
> blocked by P2's own bar: *"better requires a measurement that exists"*, and for most of them it
> does not. **The answer POLICIES.md gives is BUILD THE BENCHMARK FIRST, never pick the more
> fashionable or the more biologically faithful one** -- fidelity is ruled non-evidence at
> `docs/POLICIES.md` P2, and the Thousand Brains proposal to replace the Global Workspace with
> columnar voting was REFUSED on exactly this ground.
>
> Every criterion below is an INVENTED measurement and therefore owes P8's burden in full: a
> reachable RED constructed and recorded, a control that is not the method itself, per-item
> attribution, unprompted disclosure, and DELETION ON A NULL. Two repo-specific traps are named at
> each site because this tree has already been caught by both: `not_an_oracle`
> (`specs/009/data-model.md` -- assembly-derived wiring scored 1.000 untrained because the
> co-activity mask IS the ground-truth partition) and `can_degrade` (a metric that improves
> monotonically with parameter count is arithmetic, not function).

### [B4.27] Build the consensus benchmark that settles the two Global Workspaces - POLICIES specifies its shape and has carried it as open debt since 2026-07-28

- **status** OPEN &nbsp;|&nbsp; **policy** P2 + P8 &nbsp;|&nbsp; **effort** large &nbsp;|&nbsp; **blocks** `two-global-workspaces-additive`
- **key** `consensus-benchmark-absent`
- **source** docs/POLICIES.md P2 ("The benchmark that would settle it"); specs/019 [B4.07]
- **evidence** Re-derived at HEAD: POLICIES.md:305-307 states the design verbatim -- *"a set of ambiguous multi-source inputs with a known correct winner, scored on which workspace's ignited content matches it. Until that exists, both stay and this entry is the honest record of why."* Both workspaces are live and fed every tick (72 submits each over 40 wake ticks); importer counts re-derive at GlobalWorkspace 8 / ReentrantWorkspace 2.
- **owed** (a) the ambiguous multi-source set with a KNOWN winner, built so the winner is not recoverable from the input encoding -- the `not_an_oracle` trap; (b) a scorer reading IGNITED CONTENT from each workspace on the identical input, not a proxy; (c) a permutation/shuffled-ground-truth null, since a scorer that cannot score chance cannot score either workspace; (d) a constructed RED -- an input set on which the benchmark must report NO separation; (e) DELETE the criterion if the two workspaces are indistinguishable on it, and record that both survive with the reason. A null here is a real result: it means the P2 debt stays open honestly rather than being resolved by preference.

### [B4.28] Build the decision benchmark that discriminates the two Wang-2002 attractors - accuracy AND reaction time under graded evidence, or it discriminates nothing

- **status** OPEN &nbsp;|&nbsp; **policy** P2 + P8 &nbsp;|&nbsp; **effort** large &nbsp;|&nbsp; **blocks** `wang-attractor-duplicated`
- **key** `decision-benchmark-absent`
- **source** specs/019 [B4.06]; specs/010-policy-compliance-audit T045 (C19)
- **evidence** Two independent implementations at HEAD -- `basal_ganglia/evidence_accumulation.py` and `cortex/prefrontal_executive.py`, the second hand-rolling its own pool sizes, weights and threshold and importing only `random`, so it does not delegate. Neither has ever been scored against the other.
- **owed** Both driven from the IDENTICAL evidence stream at several coherence levels, scored on the two observables the Wang-2002 model exists to produce -- choice accuracy and decision TIME. Accuracy alone cannot separate two attractors that both eventually decide correctly; the speed-accuracy relationship is the discriminating shape. Control: at ZERO coherence both must fall to chance, and a benchmark whose zero-coherence arm is above chance is leaking the answer. Per-item attribution: report which of pool size / weights / threshold accounts for any gap, by neutering each, rather than declaring one implementation better wholesale.

### [B4.29] Build the growth A/B that SC-007 already demands - 7+ mechanisms live, and the criterion says exactly ONE survives

- **status** OPEN &nbsp;|&nbsp; **policy** P2 + P8 &nbsp;|&nbsp; **effort** large &nbsp;|&nbsp; **blocks** `multiple-growth-mechanisms-p2`
- **key** `growth-benchmark-absent`
- **source** specs/019 [B4.03]; specs/007-whole-brain-neurogenesis T025
- **evidence** At least SEVEN live at HEAD (time_cells, growable_readout, developmental_loop_brain, laminar_local_readout, growing_learner, multimodal_encoder, neurogenesis_engine). SC-007 requires exactly 1 survivor and no A/B has ever been run, so the count is the finding and the selection is unjustified either way.
- **owed** A task where growth MATTERS (capacity must be the binding constraint, or every mechanism ties and the benchmark is vacuous), scored on capacity-vs-data monotonicity AND retention across a restart -- P4b's guard, since a grower whose units die with the process has not grown. `hippocampus/ng_snn.py`'s earn-your-place stop is the template (capacity monotone in data, 4,512 units at 24,000 rows). `can_degrade` applies directly here: a score that rises with unit count is measuring the manipulation, not the function. Expect several to be RETIRED rather than compared -- P2's honesty guard requires porting any unique capability first.

### [B4.30] RE-SCOPE [B4.05]: its benchmark ALREADY EXISTS, so it is not blocked - and the measurement says MERGE, not adopt

- **status** OPEN &nbsp;|&nbsp; **policy** P2 &nbsp;|&nbsp; **effort** medium
- **key** `action-value-benchmark-exists`
- **source** specs/019 [B4.05]; src/snn_data_lake/basal_ganglia/mlp_q.py
- **evidence** RE-DERIVED at HEAD, and this corrects the assumption that every competing-mechanism item waits on a benchmark. `mlp_q`'s module docstring carries a per-arm held-out table that `main()` RE-DERIVES on every gate run: XOR (non-linear) **MlpQ 0.993 vs LatentQ 0.467 (~chance)**; linearly-separable **MlpQ 0.983 vs LatentQ 0.993**. So P2's bar -- *"better requires a measurement that exists"* -- is already MET for the action-value pair.
- **owed** What the measurement says is that **NEITHER DOMINATES**: MlpQ wins decisively where the rule is non-linear and loses slightly where a linear map is already exact. P2's two routes are co-equal and this is the MERGE case, not the ADOPT case -- build the ONE mechanism superseding both, then retire both originals. The `latent_q` / `mlp_q` / `tabular` selection flag at `brainstem/autonomous_loop.py` is precisely the *"flag that selects between mechanisms is additive with extra steps"* failure, and a merge that keeps it has not merged. The curiosity-drive half of [B4.05] is a separate question and does NOT inherit this evidence.

## Band 5 -- P3 no dormancy

23 item(s).

### [B5.01] Fix import_manifest._targets()'s ImportFrom blindness - the P3 gate prints 'all 6 cross-tier targets import' while one of them raises

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** medium
- **key** `import-manifest-importfrom-blindness`
- **source** specs/BACKLOG_RANKED.md row 19
- **evidence** import_manifest.py:157-158 still does `names = [n.module]` and discards n.names, and running the self-check at HEAD prints 'self-check OK: all 6 cross-tier targets import' WHILE B5.02's ImportError is live - a mechanized gate emitting a false green outranks every capability item in this band.

### [B5.02] Repair the live broken import at plasticity/snn_mnist_stdp_genn.py:113

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** small
- **key** `broken-import-snn-mnist-stdp-genn`
- **source** specs/BACKLOG_RANKED.md row 18
- **evidence** RUNTIME RED re-derived: :113 imports snn_data_lake.reasoning.snn_mnist_stdp, which does not exist (the module lives at plasticity/), so oracle() RAISES; the swallow at :603-607 and the pygenn canary at :612 both resolve exactly.

### [B5.03] Narrow esp32_encoder_path's EXEMPT reason to read_serial_lines - today's dormancy_gate change makes it a red, not a nit

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** small
- **key** `esp32-exemption-overbroad`
- **source** specs/014-live-sensor-transduction/tasks.md T059; specs/BACKLOG_RANKED.md row 23
- **evidence** COLLAPSED 2 rows; dormancy_gate.py:255-258 still blames 'pyserial absent and arm_config SAFETY_CALIBRATED=False' while those two strings appear ZERO times in esp32_encoder_path.py (positive control: its 6 defs match) and only read_serial_lines (:128) imports pyserial lazily at :145-147; HEAD 39c412f9 added dormancy_gate.py:551 `assert not cov['stale_exempt']`, so making the module reachable now REDS the gate.

### [B5.04] Re-key EXEMPT_METHOD['consolidate'] to class-qualified form - the bare key suppresses 8 sibling findings and SC-003 fails at HEAD

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** small
- **key** `exempt-method-bare-consolidate-key`
- **source** specs/001-charge-the-dormant/tasks.md T032; specs/BACKLOG_RANKED.md row 27
- **evidence** COLLAPSED 2 rows; I re-derived method_dormancy_gate.py:75 as a bare `"consolidate": (` and the lookup accepts a bare name at :302, while `def consolidate` under src/ finds 8 others; the gate's own :290 comment calls the bare form 'far too blunt for a real disposition', and its reason cite affective_memory.py:173 has rotted to :175.

### [B5.05] Fix the LatentQ.update entry, which is false twice: the comment denies an exemption that exists, and the exemption's reason denies a live call site

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** small
- **key** `method-gate-latentq-entry-false-twice`
- **source** specs/BACKLOG_RANKED.md row 26; src/snn_data_lake/substrate/method_dormancy_gate.py (2 rows the auditors filed as closed_or_stale)
- **evidence** COLLAPSED 3 rows; :146-150 asserts 'It is NOT exempted here' while EXEMPT_METHOD holds the key at :113 with disposition 'wire', and that entry's 'never called on a live path' is refuted by autonomous_loop.py:1792 `qsub.update(z_lq, action, r)` - off the tick only because latent_q defaults False at :176, the honest framing CLAUDE.md already uses for mlp_q.

### [B5.06] Retire the false 'the reachability walk does not traverse function-level imports' claim in dormancy_gate

- **status** OPEN &nbsp;|&nbsp; **policy** P3 gate mechanics &nbsp;|&nbsp; **effort** small
- **key** `dormancy-gate-false-reachability-walk-claim`
- **source** specs/BACKLOG_RANKED.md row 20
- **evidence** Claim intact at dormancy_gate.py:206-207 while _graph() DOES carry developmental_loop_brain -> competence_executive whose three import sites are all inside functions (:135,:251,:391 - the row's :240/:380 ROTTED), and reachable() (now :441, not :434-445) never references EXEMPT.

### [B5.07] Re-adjudicate competitive_learning - its 'recognizer_bank is itself not live' clause is false

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** medium
- **key** `competitive-learning-exemption-recognizer-bank-not-live-false`
- **source** specs/BACKLOG_RANKED.md row 22
- **evidence** dormancy_gate.py:230-235 intact, but dormancy_gate.reachable() at HEAD returns 199 modules (positive control) and cortex.recognizer_bank IS in that set.

### [B5.08] Correct deep_stdp_latency's 'nothing governed reaches it yet' - it conflates governed tier with reachable

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** small
- **key** `deep-stdp-latency-exemption-nothing-governed-reaches-false`
- **source** specs/BACKLOG_RANKED.md row 24
- **evidence** dormancy_gate.py:186-193 intact while developmental_loop_brain.py:109 imports ConvLatencyStdp at module top and constructs it at :263 and :445 (the row's :252/:434 ROTTED), and hippocampus/ is inside the governed tree.

### [B5.09] Re-adjudicate the lake_memory exemption - its reason names a consumer set that does not exist

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** medium
- **key** `lake-memory-exemption-names-nonexistent-consumer`
- **source** specs/BACKLOG_RANKED.md row 21
- **evidence** dormancy_gate.py:302 ('used by reasoning tools/apps') intact while a tree-wide *.py search returns 15 lake_memory mentions and ZERO import statements against a positive control; the orphaned branch has drifted to live_brain.py:426-429.

### [B5.10] Correct deep_stdp's parenthetical calling an already-promoted module reasoning-tier

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** small
- **key** `deep-stdp-parenthetical-stale-tier`
- **source** specs/BACKLOG_RANKED.md row 25
- **evidence** dormancy_gate.py:198 still reads '(deep_stdp_latency, deep_stdp_2stage -- reasoning-tier until their own rungs)' while the sibling entry eight lines above files deep_stdp_latency as cortex/ TRAIN-TIER; one line, one file, no second copy.

### [B5.11] Strike the stale 'NOT re-adjudicated' warning and re-derive the P3 gate counts in POLICIES.md

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** small
- **key** `policies-p3-section-stale-counts`
- **source** specs/001-charge-the-dormant/tasks.md T030
- **evidence** POLICIES.md:455 records the adjudication as done while :421-423 still says the set 'has NOT been re-adjudicated' and the mechanization block still publishes 56/0/16/5 against a measured 66/0/20/10 + 4 open TODOs at HEAD; a member of the B10.01 sweep, listed separately because it is a P3 gate figure.

### [B5.12] Amend the fear_conditioning DISPOSITION note - and drop the half of T061 that targets a file with no such entry

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** small
- **key** `fear-conditioning-disposition-note`
- **source** specs/014-live-sensor-transduction/tasks.md T041 + T061
- **evidence** COLLAPSED 2 rows; amygdala/fear_conditioning.py:28-33 still reads 'exempt -- awaiting a prerequisite (a real proximity sensor and a real body), NOT wire it', while grep for amygdala\|fear in dormancy_gate.py returns 0 and method_dormancy_gate has no entry either - the docstring is the only disposition record in the tree.

### [B5.13] Build substrate/sensor_feed.py: SensorReading + bounded SensorQueue, registered, self-checked, tested - it gates 12 items, the deepest chain in the inventory

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** medium
- **key** `sensor-feed-module-missing`
- **source** specs/014-live-sensor-transduction/tasks.md T007,T008,T010,T011,T012
- **evidence** COLLAPSED 5 rows; src/snn_data_lake/substrate/ has no sensor_feed.py, grep -c sensor_feed in __init__.py and run_selfchecks.py = 0/0, no tests/test_sensor_feed.py.

### [B5.14] Measure whether stored load scales with population - the gating measurement, and its instrument has zero callers

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** medium
- **key** `episode-rate-instrument-dormant`
- **source** specs/009-sparse-connectivity/tasks.md T013
- **evidence** episode_rate_observed() (episode_consolidation.py:232) has ZERO callers tree-wide (matches specs/010/findings.md:930), while `dim = DIM_PER_EPISODE * capacity` (:112,:117) makes load structurally proportional to width - the case research.md:80-93 says kills the uniform bound.

### [B5.15] Add PROVENANCE entries for access_index, region_coherence, ignitions, reentrant_availability, reentrant_strength

- **status** OPEN &nbsp;|&nbsp; **policy** P3 + P4c &nbsp;|&nbsp; **effort** small
- **key** `provenance-missing-headline-keys`
- **source** specs/010-policy-compliance-audit/tasks.md T033
- **evidence** Controlled grep of the PROVENANCE dict body (brain_telemetry.py:140-183): control coverage_reachable = 1 hit, the five targets = 0; cockpit.py:871 still draws access_index as the untagged headline (the task's :798 cite is STALE).

### [B5.16] Build tools/growth_census.py (producer/consumer per region, FLOOR-not-census disclosure, BOTH count)

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** medium
- **key** `growth-census-tool-absent`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T018,T019,T039
- **evidence** COLLAPSED 3 rows; growth_census appears in no tracked file outside specs/007, and both premises re-verified live - REGIONS has 17 keys with pons+olfactory empty, and GrowingLearner still recruits inline at growing_learner.py:89 inside observe(), not in a grow* method.

### [B5.17] Discharge the 4 non-permanent EXEMPT_METHOD standing TODOs - and correct the DevelopmentalEnvelope rationale, which is refuted

- **status** BLOCKED &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** large
- **key** `exempt-method-standing-todos` &nbsp;|&nbsp; **blocked by** `multiple-growth-mechanisms-p2`
- **source** specs/001-charge-the-dormant/tasks.md T024; specs/BACKLOG_RANKED.md row 28
- **evidence** COLLAPSED 2 rows; the gate at HEAD prints 10 entries with exactly 4 non-permanent TODOs (3 wire, 1 retire), COVERAGE_TO_100.md:164 records 'NOTHING was wired or retired' so there is no A/B to record, and the 'never called on any live path' rationale is refuted by autonomous_loop.py:1937 `self._dev.update(self._comp_ema)`.

### [B5.18] Remove cortical_microcircuit from dormancy_gate EXEMPT and keep the gate green

- **status** BLOCKED &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** small
- **key** `microcircuit-exempt-entry-removal` &nbsp;|&nbsp; **blocked by** `cortical-microcircuit-unwired-port`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T009
- **evidence** Recursive grep for cortical_microcircuit across **/*.py returns 4 files (the gate, run_selfchecks, __init__, itself), so deleting the EXEMPT entry at :222-226 moves it into `dormant` and REDS the gate until B2.08 lands; the auditor's blocker key 'microcircuit-port-undriven' does not exist - remapped here.

### [B5.19] Run the P3 neuter-A/B on placement (SC-006) - a wiring job once placement lands, not a new build

- **status** BLOCKED &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** medium
- **key** `placement-charge-unproven` &nbsp;|&nbsp; **blocked by** `neurogenesis-units-unplaced`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T032
- **evidence** No placement to neuter, but the instrument exists (tools/charge_probe.py, carrying the 47/60 external positive control).

### [B5.20] Verify the probe harness is sensitive, run the two-path neuter-A/B, write the GO/NO-GO verdict - the 014 decision point

- **status** BLOCKED &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** medium
- **key** `charge-gate-unrun` &nbsp;|&nbsp; **blocked by** `sensor-feed-module-missing`
- **source** specs/014-live-sensor-transduction/tasks.md T043,T044,T045,T046
- **evidence** COLLAPSED 4 rows; tools/charge_probe.py exists (12.3K) so the harness is available, but nothing consumes a simulated reading at HEAD, so there is no feed to neuter.

### [B5.21] Measure Brain.respond latency and the live _language hit rate at >=3 seeds - runnable now, needs no pair set

- **status** OPEN &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** small
- **key** `language-hitrate-single-seed-only`
- **source** specs/005-language-organ/tasks.md T016
- **evidence** baseline.txt heads its Phase-3 block 'T013 / T014 / T015 / T016 / T017 -- NOT RUN' and the only hit-rate figure in-tree is a single-seed 9-of-10 at docs/coverage/Language.md measured_at 2026-07-30; the one 005 item the T011 stop does not block.

### [B5.22] Give the 9 entry-point-less out-of-gate modules a __main__, and register the 9 that only need a line

- **status** BLOCKED &nbsp;|&nbsp; **policy** P3 &nbsp;|&nbsp; **effort** medium
- **key** `selfcheck-entrypoint-registration` &nbsp;|&nbsp; **blocked by** `promotion-sweep-regex-and-has-main`
- **source** specs/BACKLOG_RANKED.md row 30
- **evidence** All 9 named modules exist and contain ZERO __main__ (verified per-file including 'integrations/robot arm/sensor_reading.py'), but every registration cite has drifted (run_selfchecks.py :531 -> :583, and :354/:369/:370/:378/:379/:380/:411) because that file is modified in the working tree today.

### [B5.23] Retire or repair reasoning/arc_agi/ - two diverged copies and the promoted one is not runnable

- **status** OPEN &nbsp;|&nbsp; **policy** P3 + P4c &nbsp;|&nbsp; **effort** medium
- **key** `arc-agi-diverged-broken-copies`
- **source** specs/BACKLOG_RANKED.md row 69
- **evidence** All 6 shared filenames differ by SHA256; the governed play.py:36 binds the first-party package then calls arc_agi.Arcade() at :58 (row said :51 - DRIFTED) while research/arc_agi/play.py:28 imports correctly; file count is 7 src + 6 research, not 12.

## Band 6 -- P4 / P4b / P4c one brain, one lake, every build

31 item(s).

### [B6.01] Re-derive the 248-ungoverned threshold - it is stale by 46 and would pass a large regression through the ratchet

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** small
- **key** `p4c-baseline-248-stale`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T040 (auditor filed it closed_or_stale)
- **evidence** Auditor ran p4c_ratchet: TOTAL ungoverned 202 (reasoning 108) against tasks.md's 248/150, and run_selfchecks MODULES = 301 against a header asserting 280/280; a ratchet asserting against a loose baseline is the one failure mode a ratchet has, so it leads this band.

### [B6.02] Rename the episode_retained telemetry key and its comment - it is labelled durable and measures the volatile store

- **status** OPEN &nbsp;|&nbsp; **policy** P4b + P4c &nbsp;|&nbsp; **effort** small
- **key** `episode-retained-mislabelled-durable`
- **source** specs/010-policy-compliance-audit/tasks.md T034
- **evidence** brain.py:3404 emits 'episode_retained' with the comment 'durable-episode recall fraction' while its source EpisodeConsolidator.retained() (:469) iterates self.codes, the in-memory attractor, and brain.py:3714 renders it inside '(durable: ... held verbatim in my consolidated store)' - P4c binds hardest on what the cockpit reports.

### [B6.03] Correct CLAUDE.md's P4c paragraph - 161 modules, 48/29.8% unimportable, 12 promoted, 107/29/16/9

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** small
- **key** `claude-md-p4c-stale-numbers`
- **source** specs/BACKLOG_RANKED.md row 35
- **evidence** CLAUDE.md:38 still carries all four figures verbatim (I re-read it at HEAD) against a measured `src/snn_data_lake/reasoning 108` (TOTAL ungoverned 202) from tools/p4c_ratchet.py - the numbers are the P4c build list, so they mis-size every promotion lane below.

### [B6.04] Refresh promotion-plans.json's ready[] bucket and P4C_BACKLOG.md's blocked count

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** small
- **key** `promotion-plans-ready-bucket-stale`
- **source** specs/BACKLOG_RANKED.md row 34
- **evidence** ready=4 / blocked=2 / settled=11 at HEAD with all 4 ready[] sources Test-Path False and all 4 cortex/ targets True, so READY is truly 0; P4C_BACKLOG.md:268 says '5 BLOCKED' while naming 2, and the 'TWO blocked entries remain' sub-cite drifted six lines.

### [B6.05] Fix promotion_sweep.py's LOAD-BEARING bare-stem regex and its has_main mislabel - it feeds two other lanes

- **status** OPEN &nbsp;|&nbsp; **policy** P4c instrument &nbsp;|&nbsp; **effort** small
- **key** `promotion-sweep-regex-and-has-main`
- **source** specs/BACKLOG_RANKED.md row 29
- **evidence** All 4 cites exact (:147-148 bare-stem re.compile, :202-205 n.name=='main', :22-23, :249) and the live sweep says LOAD-BEARING 9 / no-main 25 with both named false positives re-derived (eprop_shd.py:413 is a torch DataLoader import credited to ng_snn/data.py; snn_mnist_stdp_genn.py:148 is a docstring), so the corrected count 7 holds.

### [B6.06] Delete the stale gcp//infra/ prose from POLICIES.md and close the 36 small-zone files

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** medium
- **key** `p4c-small-zones-and-stale-gcp-infra-prose`
- **source** specs/BACKLOG_RANKED.md row 43
- **evidence** p4c_ratchet reproduces every zone exactly (docs 16, integrations 8, src/training 6, training 4, lakehouse 2 = 36, all delta +0) while `ls gcp infra` -> No such file or directory and POLICIES.md:553 and :623 both still name them as live P4c debt - a 2-line false-claim fix.

### [B6.07] Record the per-module DISPOSITION mapping behind the 161 classification

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** medium
- **key** `p4c-per-module-disposition-missing` &nbsp;|&nbsp; **blocked by** `claude-md-p4c-stale-numbers`
- **source** specs/BACKLOG_RANKED.md row 36
- **evidence** P4C_BACKLOG.md:3-10 states the defect verbatim ('aggregate counts were written into CLAUDE.md, but the per-module enumeration was never committed') and p4c_ungoverned_baseline.json carries total 202 paths across 10 zones with NO module->verdict field, so the remainder is uncomputable.

### [B6.08] Create, register and populate cortex/cortical_prior.py with its four-attribute constants and self-check red - it gates 8 items

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** medium
- **key** `cortical-prior-module-absent` &nbsp;|&nbsp; **blocked by** `spec011-decisions-unmade`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T015,T016,T017
- **evidence** COLLAPSED 3 rows; recursive search for cortical_prior* returns nothing tree-wide and importing run_selfchecks shows it absent from MODULES (301 entries); gates T018-T020, T023-T025, T030, T034.

### [B6.09] Extend the Phase 0 census to the lake - re-scoped from 4 record types to the 7 that exist

- **status** OPEN &nbsp;|&nbsp; **policy** P4b &nbsp;|&nbsp; **effort** medium
- **key** `lake-census-missing`
- **source** specs/009-sparse-connectivity/tasks.md T034 (auditor filed it closed_or_stale)
- **evidence** home.py:73-178 defines SEVEN tags at HEAD (_T_FACT/_T_SPIKE/_T_GROWTH/_T_EPISODE/_T_SELFBUILD/_T_CALIB-retired/_T_STATE) against the task's stated 4, partial census data is scattered in tag comments, and no consolidated writers/readers/counts report is filed - it is OPEN work, and it blocks B6.12.

### [B6.10] Consume the durable lake-recall path in a live consumer, or exempt it with a reason

- **status** OPEN &nbsp;|&nbsp; **policy** P4b &nbsp;|&nbsp; **effort** medium
- **key** `durable-recall-unconsumed`
- **source** specs/010-policy-compliance-audit/tasks.md T038 (C02) + T041 (C23)
- **evidence** COLLAPSED 2 rows (one defect); Brain.recall_episode_durable (brain.py:2932) has zero callers under src/ (only tests/test_episode_durability.py:142-143) while the one live consumer at brain.py:3698-3716 calls the working-memory attractor and never the lake.

### [B6.11] Record the (seed, dim, fanout) a store was built with and report loudly when a restored brain rebuilds a different network

- **status** OPEN &nbsp;|&nbsp; **policy** P4b &nbsp;|&nbsp; **effort** medium
- **key** `retention-change-silently-rewires`
- **source** specs/009-sparse-connectivity/tasks.md T031
- **evidence** brain.py:83 EPISODE_RETENTION_DAYS=53 feeds capacity_for -> required_dim (episode_consolidation.py:117, DIM_PER_EPISODE=4) -> dim -> _wire() mask, and no wiring provenance is among the seven tags in home.py:73-178, so a retention tweak silently rewires the store over the same episodes.

### [B6.12] Decide whether the durable lake should carry structure, and record the answer either way

- **status** OPEN &nbsp;|&nbsp; **policy** P4b &nbsp;|&nbsp; **effort** medium
- **key** `lake-carries-no-structure`
- **source** specs/009-sparse-connectivity/tasks.md T033
- **evidence** save_growth(blob, path=None) at home.py:662 still takes a flat bytes blob and the newer generic carrier save_named_state(name, blob) (:517) is also raw bytes keyed by name, so the lake offers no structure/mask/adjacency and no decision is recorded.

### [B6.13] Re-scope SC-007's denominator from connectivity-bearing modules to the whole brain including the lake

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4b &nbsp;|&nbsp; **effort** small
- **key** `sc007-denominator-modules-only` &nbsp;|&nbsp; **blocked by** `lake-census-missing`
- **source** specs/009-sparse-connectivity/tasks.md T035
- **evidence** spec.md:224 SC-007 still reads 'N of M sparse, K documented exceptions' over modules only, unchanged since 2026-08-04; the Phase 7 preamble's P4b re-scope was never applied to the spec text.

### [B6.14] Re-diagnose why the live episodic-consolidation consumer goes inert once the store fills - the capacity premise was REFUTED

- **status** OPEN &nbsp;|&nbsp; **policy** P4b &nbsp;|&nbsp; **effort** large
- **key** `episodic-consolidation-inert-undiagnosed`
- **source** specs/010-policy-compliance-audit/tasks.md T039
- **evidence** grep for C03 / 'inert once' / 'ring evict' across episode_consolidation.py and brain.py returns 1 unrelated hit (brain.py:201), so the verifier's Control A refutation lives only in tasks.md and no re-diagnosis exists in code.

### [B6.15] Give the canonical stack's action+reward layer durable state

- **status** OPEN &nbsp;|&nbsp; **policy** P4b &nbsp;|&nbsp; **effort** medium
- **key** `arm-layer-no-durable-state`
- **source** specs/010-policy-compliance-audit/tasks.md T040 (C05)
- **evidence** AutonomousArm.__init__ builds GrowingLearner() (autonomous_loop.py:195) and self.q = {} (:197, cite resolves exactly) fresh every construction; the only disk read in the file is load_bundle (:1394) for trained skills and self.q is written only in memory at :1561/:1796.

### [B6.16] Address the 1 Hz telemetry poll writing brain state from the HTTP thread

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** medium
- **key** `state-writes-from-http-thread`
- **source** specs/010-policy-compliance-audit/tasks.md T035
- **evidence** Brain.state() (brain.py:3227, class Brain at :172) assigns self._emotion = self._affect_core.recognize(...), and cockpit.py:437/:1042 serve snapshot(brain) -> brain.state() from the HTTP thread while _wake_tick calls the same method from the tick thread; the verifier's 'already called once per tick' is true and does not remove the race.

### [B6.17] Record the cockpit-open vs cockpit-closed tick-rate baseline for SC-008

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** small
- **key** `sc008-tick-baseline-missing`
- **source** specs/010-policy-compliance-audit/tasks.md T003
- **evidence** `ls specs/010-policy-compliance-audit/baseline-tick.txt` -> No such file, while baseline-envelope.txt and baseline-gates.txt DO exist in the same directory (positive control).

### [B6.18] Re-run quickstart Scenario 5 (observing must not become the load) against the T003 baseline

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** small
- **key** `sc008-rerun-blocked` &nbsp;|&nbsp; **blocked by** `sc008-tick-baseline-missing`
- **source** specs/010-policy-compliance-audit/tasks.md T036
- **evidence** tools/charge_probe.py exists but Scenario 5 is a COMPARISON against a baseline-tick.txt that does not exist, so no red is constructible.

### [B6.19] SIMULATED marker: required-never-defaulted at construction, validated at the endpoint, carried through state report / lake record / cockpit render with a red per hop

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** medium
- **key** `simulated-marker-provenance` &nbsp;|&nbsp; **blocked by** `sensor-feed-module-missing`
- **source** specs/014-live-sensor-transduction/tasks.md T009,T022,T037,T038,T039,T040
- **evidence** COLLAPSED 6 rows (one defect: a simulated reading must be undeniably simulated at every hop); `grep -ci simulated` = 0 in both cockpit.py and learn_daemon.py.

### [B6.20] Add POST /api/sensor to cockpit do_POST (404/400/202), thread the queue through serve(), test the four reds

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** medium
- **key** `cockpit-sensor-endpoint-missing` &nbsp;|&nbsp; **blocked by** `sensor-feed-module-missing`
- **source** specs/014-live-sensor-transduction/tasks.md T020,T021,T023,T024,T025,T026,T027
- **evidence** COLLAPSED 7 rows; cites RE-DERIVED - do_POST at cockpit.py:358 with the /api/act block at :360-374, `def serve(brain, port=8137, turns=None, block=True, acts=None)` at :396 with no sensor param, learn_daemon.py:265 correct.

### [B6.21] Drain the sensor queue on the tick thread beside the existing acts drain

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** small
- **key** `tick-thread-drain-missing` &nbsp;|&nbsp; **blocked by** `sensor-feed-module-missing`
- **source** specs/014-live-sensor-transduction/tasks.md T034
- **evidence** Work open and the cite ROTTED - the acts drain is `acts.pump(b, budget=1)` at learn_daemon.py:320, not the '~:292' the task names (:292 is an `except Exception`).

### [B6.22] Confirm no second durable store: anything persisted lands in brain_lake.bin under its own record tag

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4b &nbsp;|&nbsp; **effort** small
- **key** `no-second-durable-store` &nbsp;|&nbsp; **blocked by** `simulated-marker-provenance`
- **source** specs/014-live-sensor-transduction/tasks.md T069
- **evidence** Nothing is persisted yet (T038 unwritten), so this check is unrun rather than passing.

### [B6.23] Measure p4c_ratchet and p6_gate counts against baseline rather than asserting they are unchanged

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** small
- **key** `ratchets-unchanged-unmeasured` &nbsp;|&nbsp; **blocked by** `014-baselines-unrecorded`
- **source** specs/014-live-sensor-transduction/tasks.md T064
- **evidence** Both tools exist (tools/p4c_ratchet.py 22.9K, tools/p6_gate.py 82.9K); the T004 baseline they compare against does not.

### [B6.24] Add a PlacementState record tag to home.py and prove it across a real process boundary

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4b &nbsp;|&nbsp; **effort** medium
- **key** `placement-state-not-persisted` &nbsp;|&nbsp; **blocked by** `neurogenesis-units-unplaced`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T028 + T029
- **evidence** COLLAPSED 2 rows; home.py carries 7 tags (_T_FACT :73 through _T_STATE :178) and none is placement, and T029's method already exists and must be reused not rebuilt - neurogenesis_engine._cross_process_restart (:428, subprocess :480, claim-5 assert :700).

### [B6.25] Clear the four spike_chat_lake promotion blockers and promote the 136 KB hub

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** large
- **key** `promote-spike-chat-lake` &nbsp;|&nbsp; **blocked by** `import-manifest-importfrom-blindness`
- **source** specs/BACKLOG_RANKED.md row 37
- **evidence** 136,208 B exact; blocker 1 exact (p6_gate.py:279/:292 hardcode the reasoning/ path), blocker 2 confirmed (ruff --isolated I001 fires at :43:1, masked by project config), blocker 4 measured 34.7 s warm and the module sits in RESEARCH so tests/test_selfchecks.py:36 would newly swallow it; blocker 3's :1407 cite is STALE - _targets() is at import_manifest.py:145.

### [B6.26] Promote the developmental-loop / autonomous-brain family as one 10-module lane

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** large
- **key** `promote-developmental-loop-family`
- **source** specs/BACKLOG_RANKED.md row 38
- **evidence** concept_reflection.py:34 exact (module-top TrainOfThought import) and developmental_loop_brain.py:155-160 exact ('M21 is owed by whichever lane promotes _loop_common + fast_features'); the e-prop cite DRIFTED - LinearReadout is at _loop_common.py:35 with torch.optim.Adam at :51, not the cited :43/:49/:50.

### [B6.27] Promote reasoning/schema_induction.py (44,330 B, load-bearing from brain.py)

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** medium
- **key** `promote-schema-induction` &nbsp;|&nbsp; **blocked by** `schema-induction-art-clusterer`
- **source** specs/BACKLOG_RANKED.md row 32
- **evidence** Size exact and TORCH_GATED registration confirmed by import, but the cite drifted :411 -> :418, and the prose blocker is FIVE kb_ranking sites (:63,:87,:108,:309,:321) not six - the cited :155 contains no kb_ranking at all.

### [B6.28] Fix the 8 bare sibling imports in reasoning/ng_snn and reasoning/arc_agi

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** small
- **key** `reasoning-bare-sibling-imports`
- **source** specs/BACKLOG_RANKED.md row 33
- **evidence** Exactly 8 present at HEAD: `from softhebb_mnist import ...` in ng_snn/{extend_3conv:17, nmnist_pipeline:40, nmnist_spatiotemporal:57, _3conv:65, train_softhebb:24} and `import env as arc_env` in arc_agi/{brain_player:54, rule_inducer:93, run_arc:33}.

### [B6.29] tools/ zone: 30 ungoverned files that can never become governed in place

- **status** OPEN &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** large
- **key** `tools-zone-ungoverned-30`
- **source** specs/BACKLOG_RANKED.md row 44
- **evidence** p4c_ratchet prints `tools 30 30 +0` and p4c_ratchet.py:51-56's GOVERNED-is-DERIVED block maps REGIONS -> src/snn_data_lake/<region>/<module>.py, so a tools/ file is structurally ineligible; all six ci.yml cites resolve exactly.

### [B6.30] Settle the [lake] extra - lakehouse/ is invisible to ruff and pyarrow's exemption stands on it

- **status** OPEN &nbsp;|&nbsp; **policy** P4c + P7 &nbsp;|&nbsp; **effort** medium
- **key** `lakehouse-gitignored-invisible-to-ruff`
- **source** specs/BACKLOG_RANKED.md row 70
- **evidence** .gitignore:62 is `lakehouse/` exact; `ruff check . --show-files` lists 482 files with ZERO under lakehouse/ (positive control: brain.py 7x) though all three modules are git-tracked; polars/pyarrow/deltalake all False in .venv-lake (control numpy True); dependency_ledger.py:148 rests the pyarrow exemption on 'the tier talks to polars'.

### [B6.31] Map the source's seven regions onto the circuit's populations and add an inhibitory-only L1

- **status** BLOCKED &nbsp;|&nbsp; **policy** P4c &nbsp;|&nbsp; **effort** medium
- **key** `l1-inhibitory-population-absent` &nbsp;|&nbsp; **blocked by** `h01-source-figures-unverified`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T026
- **evidence** cortical_microcircuit.py is still the 8-population Potjans-Diesmann set (POP_NAMES 0-7, L2/3e..L6i) with no L1 population and no white-matter exclusion counter.

## Band 7 -- P5 use the GPU where it helps

3 item(s).

### [B7.01] Re-measure the GPU crossover under bounded fanout and record it (or the deferral) in POLICIES P5 - the published table uses a declared round number

- **status** OPEN &nbsp;|&nbsp; **policy** P5 &nbsp;|&nbsp; **effort** medium
- **key** `gpu-crossover-not-remeasured-under-fanout`
- **source** specs/009-sparse-connectivity/tasks.md T018 + T029
- **evidence** COLLAPSED 2 rows; POLICIES.md:785 carries a sparse-memory table at a DECLARED 'fanout 100' against research.md:70-75's measured F*=128-176, its tick-crossover table is fanout-independent elementwise LIF, and the deferral reason lives only in plan.md:75.

### [B7.02] Source or replace the ms/tick budget before scoring anything against it - the yardstick exists in no tracked file

- **status** BLOCKED &nbsp;|&nbsp; **policy** P5 &nbsp;|&nbsp; **effort** small
- **key** `tick-cost-budget-unsourced` &nbsp;|&nbsp; **blocked by** `per-region-growth-floor-underived`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T034 + T035
- **evidence** COLLAPSED 2 rows (T035 shares the blocker); `ms/tick = 56.60 + 0.0804 * N_episodes` occurs in NO tracked file under docs/, specs/, src/ or tools/, and no region has been admitted, so both the scale check and the crossover check are vacuous.

### [B7.03] Route cortex/brain.py's cuda-by-availability defaults through the measured device router

- **status** OPEN &nbsp;|&nbsp; **policy** P5 &nbsp;|&nbsp; **effort** medium
- **key** `cuda-by-availability-unrouted`
- **source** specs/010-policy-compliance-audit/tasks.md T050 (C26)
- **evidence** The benchmark exists (substrate/device.py::crossover/select/decisions) but controlled grep shows brain.py has ZERO references to substrate.device or resolve_device (control: neurogenesis_engine.py has 1) and still carries device='cuda' at :186 with bare torch.cuda.is_available() at :618 and :1154 - device.py's own docstring names these sites.

## Band 8 -- P6 mutual reach and use

8 item(s).

### [B8.01] Make the re-route consumer take the winner KEY, not just the value - it gates two other P6 items

- **status** OPEN &nbsp;|&nbsp; **policy** P6 &nbsp;|&nbsp; **effort** medium
- **key** `reroute-consumer-discards-winner-key`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T041; specs/BACKLOG_RANKED.md row 55
- **evidence** COLLAPSED 2 rows; brain.py:1775 `self._reroute_bias = round(float(max(rb.values())), 4)` re-derives verbatim (NOT the :1623 both the task and docs/P6_INTERCONNECT.md:1032 cite - ROTTEN), consumed at :1776 and surfaced at :3353, while the {winner: bias} producer INVENTION.md calls deleted is alive at reentrant_workspace.py:380.

### [B8.02] Re-derive docs/P6_INTERCONNECT.md against HEAD - it says 'No wire built' and publishes 42/9 against a measured 82/65

- **status** OPEN &nbsp;|&nbsp; **policy** P6 &nbsp;|&nbsp; **effort** small
- **key** `p6-interconnect-doc-stale`
- **source** specs/BACKLOG_RANKED.md row 61
- **evidence** P6_INTERCONNECT.md:3 'Status: MEASURE AND DESIGN. No wire built.', :17 '42 across 10 modules / 9 across 5' against the gate's 82 writes / 65 reads, :35 names cortex/competence_executive.py which is absent from the gate's PER-MODULE TOPOLOGY, and :292 lists a half the gate reports BUILT.

### [B8.03] Wire the 3-4 faculty callers that never hand the shared workspace to the constructor

- **status** OPEN &nbsp;|&nbsp; **policy** P6 &nbsp;|&nbsp; **effort** medium
- **key** `p6-faculty-callers-dont-inject-shared-bus`
- **source** specs/BACKLOG_RANKED.md row 56
- **evidence** All 7 cites re-derive exact: train_of_thought.py:106, rpm_solver.py:370, access_consciousness.py:288, brain.py:790 (no injection param), and callers concept_reflection.py:107 / model_builder.py:160 / raven_bench.py:233 all pass no bus; p6_gate.py:66-73 confirms SITE count was REJECTED as a criterion.

### [B8.04] Drain or retire the never-drained 'language' subscriber

- **status** BLOCKED &nbsp;|&nbsp; **policy** P6 &nbsp;|&nbsp; **effort** medium
- **key** `p6-language-subscriber-never-drained` &nbsp;|&nbsp; **blocked by** `reroute-consumer-discards-winner-key`
- **source** specs/BACKLOG_RANKED.md row 60
- **evidence** brain.py:808-809 subscribes both 'action' and 'language' while :1774 broadcast_to('action') is the only live drain, and brain.py:802-807 now carries an in-code note stating this defect and its ordering dependency on the winner-key fix.

### [B8.05] Reduce the writer modules whose workspace nobody reads - the real residue is 4, not 6

- **status** BLOCKED &nbsp;|&nbsp; **policy** P6 &nbsp;|&nbsp; **effort** large
- **key** `p6-unread-writer-modules` &nbsp;|&nbsp; **blocked by** `p6-faculty-callers-dont-inject-shared-bus`
- **source** specs/BACKLOG_RANKED.md row 58
- **evidence** p6_gate prints 6 of 9 unread writers, and both exclusions hold - brain.py:2626 injects ws=self.workspace at runtime (defeating the static :288 fallback) and the gate itself labels spike_chat_lake UNATTRIBUTABLE, 'the residual false-red channel of this ceiling'.

### [B8.06] Close P6's headline: 0 faculty-to-faculty workspace reads

- **status** BLOCKED &nbsp;|&nbsp; **policy** P6 &nbsp;|&nbsp; **effort** large
- **key** `p6-zero-faculty-to-faculty-reads` &nbsp;|&nbsp; **blocked by** `reroute-consumer-discards-winner-key`
- **source** specs/BACKLOG_RANKED.md row 57
- **evidence** p6_gate at HEAD: cross_module_reads=1 and its sole edge is live_brain.py:313 (role ACTION) into autonomous_loop.py:259's ACTION bus - the exact move P6 calls the defect - so faculty-to-faculty is still 0 and explicitly NOT ASSERTED; the 6-verb allowlist at :214-221 discloses its own weakness at :211-213.

### [B8.07] Reconcile the connectome - 11 undeclared code paths, 8 with brainstem as an endpoint

- **status** OPEN &nbsp;|&nbsp; **policy** P6 &nbsp;|&nbsp; **effort** large
- **key** `p6-connectome-undeclared-region-imports`
- **source** specs/BACKLOG_RANKED.md row 59
- **evidence** p6_gate FINDING 2 = 11 of 27 with 8 brainstem endpoints and FINDING 1 = 21 of 37, both exact; autonomous_loop.py imports cerebellum :38-40, hippocampus :44-48, neuromodulation :50-55, and ConnectomeRouter is constructed brain.py:920 and driven :2288 - so FINDING 1's 'provably carries nothing' wording is the part needing correction.

### [B8.08] Add region-suppression mode to the A/B harness and run it per producer/consumer pair as ConsumptionRecords

- **status** BLOCKED &nbsp;|&nbsp; **policy** P6 &nbsp;|&nbsp; **effort** medium
- **key** `region-suppression-mode-absent` &nbsp;|&nbsp; **blocked by** `growth-ab-harness-absent`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T037 + T038
- **evidence** COLLAPSED 2 rows (build + run); deep_fingerprint.py has only the --demo A/A path and tools/charge_probe.py neuters a named DRIVE SOURCE key in _drive_w (:55-76, :102), not a region's contribution.

## Band 10 -- P8 invent the method (criteria owed, reachable reds, deletion on a null)

41 item(s).

### [B10.01] Sweep the remaining doc-consistency findings, verifying each before it drives an edit - two live falsities re-derived at HEAD

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** large
- **key** `docs-consistency-sweep-unrun`
- **source** specs/010-policy-compliance-audit/tasks.md T031; specs/009-sparse-connectivity/quickstart.md:7 (DUPLICATE, collapsed here)
- **evidence** COLLAPSED 2 rows; no sweep record exists in specs/010, and two instances re-derive now: CLAUDE.md:96 still claims a NORD_TORCH=0 stdlib CI matrix job that ci.yml:14-18 says was DELETED 2026-07-29, and POLICIES.md:36 publishes '400 files scanned, 13 criteria; READS_SYSTEM 13' while the live gate prints 375/12/12; quickstart.md:7 ('fanout_bisect.py does not exist yet') is the same defect class and is already logged at findings.md:972.

### [B10.02] Strike CLAUDE.md:46's P8 'live defect' sentence - false four ways

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `claude-md-p8-live-defect-sentence-false`
- **source** specs/BACKLOG_RANKED.md row 63
- **evidence** The sentence is intact (I re-read it at HEAD) while spike_chat_lake.py:925 is a docstring about _attended() (routing is :1228 -> :1513), consciousness_probe.py:171 is `def probe_access(brain=None)` so 'zero arguments' is false, 1 MARKER token survives, and red_reachability --assert reports NO_RED: 0 with both probes in READS_SYSTEM; the row's own AUDIT_BACKLOG.md:220 cite has DRIFTED to :222 and says something else there.

### [B10.03] POLICIES.md:156 still asserts '4 of 4 markers with no reachable red' in the present tense

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `policies-156-4of4-markers-present-tense`
- **source** specs/BACKLOG_RANKED.md row 64
- **evidence** POLICIES.md:156-157 resolves exact and is contradicted at HEAD by red_reachability --assert (NO_RED: 0, READS_SYSTEM: 12 including probe_access(brain) and probe_integration(seeds), rc=0) with the module now holding 2 probe defs and 1 MARKER token; :1456 already carries the same thing CLOSED with a strikethrough.

### [B10.04] Re-derive the superseded feedback-alignment figures in POLICIES.md - the module already retracted them

- **status** OPEN &nbsp;|&nbsp; **policy** P8 honesty &nbsp;|&nbsp; **effort** small
- **key** `policies-stale-feedback-alignment-figures`
- **source** specs/BACKLOG_RANKED.md row 65
- **evidence** CITE DRIFTED :1571-1578 -> :1605-1611 (I re-read that block at HEAD: 0.657/0.654, 'FA won on only 2 of 5 seeds', ~34%, cosine 0.971) against the module's own table at jepa_world_model.py:448-454 (0.627/0.620, cosine 0.990/0.991, ~37%) and its explicit retraction at :459-462 ('under the stated Bz it wins on 4 of 5').

### [B10.05] Correct the false claim the neurogenesis coverage row rests on, then update docs/coverage/

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `wholebrain-coverage-row-stale`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T043
- **evidence** docs/coverage/WholeBrain.md:286 states 'neurogenesis_engine.py contains no persistence path of any kind', FALSE at HEAD (growth_state_bytes:278, restore_growth_state:307, substrate_is_retained:349, consumed at brain.py:3094), and the row's own cites are rotten - brain.py:2720 is access-hold EMA code and :2358 for self_model.observe(grown_units=) is _learned_evc.learn (real call :2419).

### [B10.06] Repair the 25 W6 coverage citations pointing at comment/blank lines

- **status** OPEN &nbsp;|&nbsp; **policy** P8 / coverage ledger &nbsp;|&nbsp; **effort** medium
- **key** `coverage-w6-citations-on-comment-lines`
- **source** specs/BACKLOG_RANKED.md row 67
- **evidence** coverage_score.py at HEAD emits 45 warnings of which 25 are W6 and the per-category split matches exactly (Perception 6, WholeBrain 4, Executive 3, Memory 3, Consciousness 3, Social 2, +4 singletons); brain.py:1300 and brain.py:988 are each flagged 'points at a comment line'.

### [B10.07] Verify the 61 carried-unverified findings and record each deferral with its blocker named

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** large
- **key** `carried-findings-unverified`
- **source** specs/010-policy-compliance-audit/tasks.md T054 (FR-011) + T056 (FR-012)
- **evidence** COLLAPSED 2 rows (one record-keeping gap on one table); findings.md:914 '## Carried unverified -- 61 findings, verify before acting' is a bare \|dim\|sev\|policy\|title\|site\| table with no verification, disposition or blocker column, and no deferral record file exists in specs/010.

### [B10.08] Rewrite quickstart Scenarios 1-7 before re-running them - they describe a tree and an env that no longer exist

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `quickstart-scenarios-stale`
- **source** specs/010-policy-compliance-audit/tasks.md T059 (auditor filed it closed_or_stale)
- **evidence** quickstart.md:14 and :123 require .venv-cu128, retired 2026-08-05 and absent from disk, and Scenario 1's 'this must FAIL' precondition is unreproducible: dependency_ledger --assert exits 0 (UNDECLARED 0 / UNUSED 0 / MISDESCRIBED 0) and red_reachability prints 'gate OK'.

### [B10.09] Record the six Phase-1 baselines (gate, coverage, ratchets, threat, envelope diff) under evidence/ - it gates 3 items

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `014-baselines-unrecorded`
- **source** specs/014-live-sensor-transduction/tasks.md T001-T006
- **evidence** COLLAPSED 6 rows into one absent directory; `ls specs/014-live-sensor-transduction/evidence/` -> No such file or directory.

### [B10.10] Make the three Phase-2 decisions (scored task, wrong-prior definition, draw-vs-sample) and open decisions.md - T003 gates all of Phase B

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `spec011-decisions-unmade`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T003,T004,T005,T042
- **evidence** COLLAPSED 4 rows that all write the same absent file; decisions.md does not exist in specs/011 and plan.md still carries D1-D4 as open questions at :94,:117,:130,:141.

### [B10.11] Re-derive the four H01 source figures and record byte-count + SHA-256 of the five source files

- **status** DONE (`7b61f9ba`) &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **result** All five files fetched from `github.com/ashapsoncoe/h01` at commit `db223b8d769d8ae046f0026f12b251103f1dbab0`, 13,202,522 bytes, each recorded with byte count AND sha256 in `cortex/cortical_prior.py::SOURCES`. All four figures RE-DERIVED and every one reproduced the Phase 0 reading exactly: 49,377 soma records, 45,994/45,994 = 100% join coverage on `agglo_seg`, 7 cortical layer values, 6 layer circles (11 cell types too). Raw sources deliberately NOT vendored -- regenerating is an offline step per specs/011 quickstart, and 13 MB of upstream data does not belong in this tree. Also closes specs/011 US1 (`cortical_prior.py` built, registered in REGIONS + run_selfchecks + dormancy_gate EXEMPT). **GP_STRENGTH_E/I remain UNDERIVED and are named as such** -- they need the AIS-exclusion caveat in provenance, and a half-derived distribution with a missing caveat is worse than an absent one ([B10.19]).
- **key** `h01-source-figures-unverified`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T000b + T002
- **evidence** COLLAPSED 2 rows (both need the same external fetch); no baseline.md exists and no SHA-256 or byte count for agglo_20200916c3_cell_data.json appears anywhere under specs/011 - every H01 figure in spec.md is quoted, never checked.

### [B10.12] Run the BLOCKING Phase-0 preflight and record the re-derived gate baselines

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `spec011-preflight-not-run`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T000 + T001
- **evidence** COLLAPSED 2 rows; specs/011 holds no preflight.md and no baseline.md, and its stated baselines are already measurably wrong (see B6.01).

### [B10.13] Build the DriftRecord + three-arm comparison harness with both SC-005 reds - check P2 first, a drift mechanism already exists

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 + P2 &nbsp;|&nbsp; **effort** large
- **key** `seeded-drift-instrument-absent` &nbsp;|&nbsp; **blocked by** `spec011-decisions-unmade`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T010-T014
- **evidence** COLLAPSED 5 rows; grep for DriftRecord\|drift_record across cortex/ and substrate/ returns zero, while substrate/drift_monitor.py exists as a PSI distribution monitor - a different mechanism, so building a second one without an ADOPT/MERGE decision is the P2 failure.

### [B10.14] Wire placement into neurogenesis_engine at creation, plus the growth report and SC-001 - it gates 3 items

- **status** BLOCKED &nbsp;|&nbsp; **policy** P0 + P8 &nbsp;|&nbsp; **effort** medium
- **key** `neurogenesis-units-unplaced` &nbsp;|&nbsp; **blocked by** `cortical-prior-module-absent`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T021 + T022
- **evidence** COLLAPSED 2 rows; case-insensitive grep for layer\|position\|sign\|excitatory\|inhibitory in neurogenesis_engine.py returns 5 hits, all the prose phrase 'persistence layer' (positive control layer\|growth_state_bytes\|_model returns 17) - the engine has no spatial or laminar concept at all.

### [B10.15] Implement radial layer assignment + the scale assertion, and run SC-002's units-mismatch RED

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `layer-assignment-absent` &nbsp;|&nbsp; **blocked by** `cortical-prior-module-absent`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T018,T019,T020
- **evidence** COLLAPSED 3 rows; no layer-assignment code exists anywhere, and T020's red is the one that stops a nm-for-micron error shipping as a valid assignment because the fall-through layer is not an error state.

### [B10.16] Assign immutable per-cell sign, make L1+excitatory unrepresentable, add SC-003/SC-004 with three reds

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `cell-sign-unassigned` &nbsp;|&nbsp; **blocked by** `cortical-prior-module-absent`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T023,T024,T025
- **evidence** COLLAPSED 3 rows on one absent concept; neurogenesis_engine.py carries no sign field, and SC-004's second red (a no-magnitude-change run must also fail) is the anti-vacuity half with no code to attach to.

### [B10.17] Add the SC-009 determinism criterion over >=20 runs hashing floats - with zlib.crc32, never builtin hash

- **status** BLOCKED &nbsp;|&nbsp; **policy** invariant (criterion owed, not breached) &nbsp;|&nbsp; **effort** small
- **key** `placement-determinism-unproven` &nbsp;|&nbsp; **blocked by** `neurogenesis-units-unplaced`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T033
- **evidence** Nothing to hash: no placement, sign or seeded value exists at HEAD; a criterion owed on unbuilt code, not a live determinism breach.

### [B10.18] Add tests/test_cortical_prior.py, test_placement.py, test_drift.py pinning the reds as regressions

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `spec011-tests-absent` &nbsp;|&nbsp; **blocked by** `cortical-prior-module-absent`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T037,T038,T039
- **evidence** COLLAPSED 3 rows; tests/ contains no file matching prior\|placement\|drift except test_drift_monitor.py, which imports a different mechanism entirely.

### [B10.19] Phase D: seed connection strengths from GP_STRENGTH_E/I with the three caveats and the inhibitory-tail red - lowest priority in 011

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** large
- **key** `gp-strength-seeding-absent` &nbsp;|&nbsp; **blocked by** `cortical-prior-module-absent`
- **source** specs/011-h01-neurogenesis-prior/tasks.md T034,T035,T036
- **evidence** COLLAPSED 3 rows; no GP_STRENGTH constants exist and tasks.md itself marks this phase 'Ships last, and may never ship' - the feature does not depend on it.

### [B10.20] Build the growth ON/OFF A/B driver - deep_fingerprint already supersedes four of the five sub-tasks

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `growth-ab-harness-absent`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T003,T004,T005,T006,T007
- **evidence** COLLAPSED 5 rows; growth_charge_ab appears in no tracked file outside specs/007, and tools/deep_fingerprint.py already supplies the A/A control (:148-167), unrounded walk (:49), TAUTOLOGICAL tuple (:36-42) and persist=False (:157) - only the growth-on/off driver is genuinely unbuilt.

### [B10.21] Create tests/test_neurogenesis_charge.py with the SC-001, FR-004, R1/SC-004, R4, FR-005 and FR-003 asserts

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `test-neurogenesis-charge-absent`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T015,T016,T032,T033,T040,T042
- **evidence** COLLAPSED 6 rows (one file); tests/test_neurogenesis_charge.py does not exist and the only tests carrying getstate()/ripple are test_instrument_guard.py and test_time_cells.py, neither of which asserts any of the four criteria.

### [B10.22] Create specs/007's baseline.md with the three gate results and per-tick cost - it is the recording target for five tasks

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `spec007-baseline-md-absent`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T001
- **evidence** `ls specs/007-whole-brain-neurogenesis/` returns checklists/ contracts/ data-model.md plan.md quickstart.md research.md spec.md tasks.md and no baseline.md, the target T002/T008/T017/T034/T036 all write to.

### [B10.23] Make widest_log_gap() select among RESOLVABLE gaps so it cannot pick a site the floor refuses

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `widest-log-gap-ignores-floor`
- **source** specs/007-whole-brain-neurogenesis/tasks.md T011
- **evidence** cortex/time_cells.py:297-308 still returns max(range(n_cells-1), key=log-gap) with no resolvability filter, while resolvable() is a separate method the caller consults only afterwards at :337 and :455.

### [B10.24] Re-run the US1 sparse-vs-dense comparison through deep_fingerprint with a matched A/A control

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `sparse-null-not-fingerprinted`
- **source** specs/009-sparse-connectivity/tasks.md T011
- **evidence** tools/deep_fingerprint.py exists (6,467 B) but specs/009 references it only as an intention (plan.md:45, tasks.md:76/127); no fingerprint run or A/A reading zero is recorded, so FR-012's 'no downstream change' half rests on retained() alone.

### [B10.25] Whole-brain A/B sparse vs dense through deep_fingerprint with a matched A/A control at 0

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** large
- **key** `whole-brain-sparse-ab` &nbsp;|&nbsp; **blocked by** `sparse-wiring-not-live-in-brain`
- **source** specs/009-sparse-connectivity/tasks.md T027
- **evidence** Cannot run: brain.py:961 never passes fanout, so 'sparse whole brain' is not a constructible state at HEAD.

### [B10.26] Record SpikingGridCAN as OUT OF SCOPE with its measurement, not with an argument

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `gridcan-exclusion-unmeasured`
- **source** specs/009-sparse-connectivity/tasks.md T024
- **evidence** The exclusion is recorded twice (plan.md:139, research.md:170-171) but with a structural claim only - 'already O(N) in a real torus-phase metric' carries no number - and hippocampus/spiking_grid_can.py:121 has no connectivity census attached.

### [B10.27] SimulatedEnvironment with per-instance random.Random(seed) + forced hook, determinism test, no module-scope seeding

- **status** BLOCKED &nbsp;|&nbsp; **policy** invariant (criterion owed, not breached) &nbsp;|&nbsp; **effort** medium
- **key** `simulated-env-determinism` &nbsp;|&nbsp; **blocked by** `sensor-feed-module-missing`
- **source** specs/014-live-sensor-transduction/tasks.md T013,T014,T017,T068
- **evidence** COLLAPSED 4 rows (T068's polish grep is the verification half of T013); no sensor_feed.py exists so there is no producer and no seed to grep - a determinism criterion owed on unbuilt code, not a live breach.

### [B10.28] Wire the producer into learn_daemon default-OFF and assert byte-identical behaviour when disabled

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `simulator-default-off` &nbsp;|&nbsp; **blocked by** `sensor-feed-module-missing`
- **source** specs/014-live-sensor-transduction/tasks.md T016 + T018
- **evidence** COLLAPSED 2 rows; cite RE-DERIVED CORRECT (learn_daemon.py:263 `acts = Acts()`) with no producer construction anywhere near it.

### [B10.29] Add reflex channel 8 strictly above 6 and 7 in DEFAULT_RULES - gated by the Phase 6 GO/NO-GO by design

- **status** BLOCKED &nbsp;|&nbsp; **policy** invariant: safety envelope (approval-gated edit) &nbsp;|&nbsp; **effort** small
- **key** `reflex-channel-8-absent` &nbsp;|&nbsp; **blocked by** `charge-gate-unrun`
- **source** specs/014-live-sensor-transduction/tasks.md T048
- **evidence** Cite RE-DERIVED and I confirmed it myself: reflex.py:22-25 defines only channels 6 and 7; this is a safety-envelope EDIT, so it needs explicit approval as well as the charge verdict.

### [B10.30] Measure the 8-row caution matrix (sweep ch8, never-later/never-less-severe, non-finite row inherited)

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** large
- **key** `caution-matrix-unmeasured` &nbsp;|&nbsp; **blocked by** `reflex-channel-8-absent`
- **source** specs/014-live-sensor-transduction/tasks.md T051,T052,T053,T054,T055
- **evidence** COLLAPSED 5 rows (one table); no tests/test_reflex_caution.py exists and no channel 8 exists to compare against - the deepest node on the critical path.

### [B10.31] Three-path charge attribution + a re-measured threat() + the unprompted 'hurt only in simulation' disclosure

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `charge-attribution-three-path` &nbsp;|&nbsp; **blocked by** `reflex-channel-8-absent`
- **source** specs/014-live-sensor-transduction/tasks.md T058,T062,T063
- **evidence** COLLAPSED 3 rows; the third path needs channel 8, and the baseline re-derives to a 2026-07-28 in-file claim (fear_conditioning.py:4 'threat() is called 38 times per 80 ticks'), not a HEAD measurement - and B2.05 explains why that 38 was uninformative.

### [B10.32] Execute FR-012's revert of C1-C3 if both paths are null - deletion on a null is the licence working

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** unknown
- **key** `fr012-revert-on-null` &nbsp;|&nbsp; **blocked by** `charge-gate-unrun`
- **source** specs/014-live-sensor-transduction/tasks.md T047
- **evidence** Contingent on the charge verdict; there are no C1-C3 commits to revert (HEAD's f362cb52 merged 014 as SPEC ONLY).

### [B10.33] Diff docs/coverage/ against the T003 baseline and assert no band rose

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `coverage-band-must-not-rise` &nbsp;|&nbsp; **blocked by** `014-baselines-unrecorded`
- **source** specs/014-live-sensor-transduction/tasks.md T060
- **evidence** Target row exists and is currently 0 (docs/coverage/Affect.md:18 amygdala_fear_conditioning \| 0) but the baseline file it diffs against does not exist yet.

### [B10.34] Enumerate every new criterion and confirm each has a recorded red

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `every-criterion-needs-recorded-red` &nbsp;|&nbsp; **blocked by** `014-baselines-unrecorded`
- **source** specs/014-live-sensor-transduction/tasks.md T066
- **evidence** Zero criteria and zero reds exist at HEAD (no evidence/ dir, no new tests), so the enumeration is currently empty rather than satisfied - a vacuous green is exactly what P8 clause 1 forbids.

### [B10.35] Final full gate batch from a clean checkout plus a last envelope diff before push

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `final-gate-batch`
- **source** specs/014-live-sensor-transduction/tasks.md T071; specs/010 T060 (envelope diff, folded)
- **evidence** All ten commands resolve at HEAD (run_selfchecks, p4c_ratchet, p6_gate, citation_guard, red_reachability, dependency_ledger, import_manifest) and the envelope diff is currently CLEAN (five brainstem SHA256s byte-identical to specs/010's baseline-envelope.txt, arm_config.py:37 SAFETY_CALIBRATED=False, serial_arm.py:59-65 still raises) - carry it as the ship rule, not a task.

### [B10.36] Run the full-set three-scorer blimp_baseline table - its stated anchors are already retired

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `blimp-51pct-anchor-stale`
- **source** specs/005-language-organ/tasks.md T013
- **evidence** tools/blimp_baseline.py exists and needs no frozen set, but DEPTH_CEILINGS.md:636 records 'the 51.0% incumbent is stale, and the live number is below' and the cited :118-127 now lands on the specs/006 fixed-point table (anchors moved to :174-180).

### [B10.37] Score incumbent and char-bigram control on the frozen pair_ids

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `frozen-pair-set-empty` &nbsp;|&nbsp; **blocked by** `blimp-exposure-lexicon-disjoint`
- **source** specs/005-language-organ/tasks.md T014 + T015
- **evidence** COLLAPSED 2 rows (same missing input); coverage.md records 'pair_ids: NONE -- no set was frozen' and the HEAD re-run emits the same line.

### [B10.38] Run all four gates after Phases 3/4/5

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `gate-rerun-no-delta` &nbsp;|&nbsp; **blocked by** `blimp-exposure-lexicon-disjoint`
- **source** specs/005-language-organ/tasks.md T018,T038,T048
- **evidence** COLLAPSED 3 rows (identical gate rerun); each gates a phase that produced no tree change - git status shows zero files under src/snn_data_lake/cortex/ modified.

### [B10.39] Record the predictive_coding.py consumer count as N of M in verdict.md - the measurement is already made

- **status** OPEN &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** small
- **key** `predictive-coding-consumer-count`
- **source** specs/005-language-organ/tasks.md T053
- **evidence** The auditor ran it: 13 files reference predictive_coding but only 2 import it at module top (language_acquisition.py:40, live_brain.py:23), so retiring the incumbent orphans nothing; only the record is missing.

### [B10.40] SST-2 loader, licence stop-gate, sentiment block below the verdict, and the cannot-rescue-a-NULL test

- **status** BLOCKED &nbsp;|&nbsp; **policy** P8 &nbsp;|&nbsp; **effort** medium
- **key** `sst2-sentiment-block-absent` &nbsp;|&nbsp; **blocked by** `language-organ-unbuilt`
- **source** specs/005-language-organ/tasks.md T067,T068,T069,T070
- **evidence** COLLAPSED 4 rows; src/snn_data_lake/cortex/sst2_loader.py does not exist, T068's stop-gate never fired, and T070 depends on T042 and T069, neither of which has a harness to assert against.

### [B10.41] Re-file the two P4b durability tests as 'symptom gone, cause unknown' rather than closing them as understood

- **status** OPEN &nbsp;|&nbsp; **policy** P4b + P8 &nbsp;|&nbsp; **effort** medium
- **key** `p4b-durability-suite-order-failure`
- **source** specs/010-policy-compliance-audit/tasks.md T061 (listed twice; auditor filed it closed_or_stale)
- **evidence** COLLAPSED 2 duplicate rows; the symptom does not reproduce at HEAD (isolation 6/6, the full alphabetical prefix of 248 tests in one process 100% green, full pytest -x reached the coverage report with no FAILURES) but no diagnosis was ever recorded, so the order-dependence may still be latent.

### [B10.42] Assert the safety path receives RAW METRES and the spike path the normalized feats - substituting one moves the trip point 3x

- **status** DONE (`4a2ae912`) &nbsp;|&nbsp; **policy** P8 (a criterion with a constructed reachable red) &nbsp;|&nbsp; **effort** small
- **key** `raw-metres-vs-normalized-distance`
- **source** specs/014-live-sensor-transduction/tasks.md T032,T033; RE-BANDED from [B1.04] 2026-08-15
- **evidence** Arithmetic re-derived: `arm_config.CONTACT` = 0.04 m against `esp32_encoder_path.normalize`'s `distance_m / 3.0`, so a normalized value fed to the metres threshold moves the trip point to 0.12 m. Same two files as [B1.03] and equally not blocked on the unbuilt feed.
- **WHY IT MOVED OUT OF BAND 1, and the title was wrong before it was re-banded.** It was filed as *"loosens the trip point 3x"*. The 3x is exact and reproduces; "loosens" does not. `normalized < CONTACT` is `raw < 0.12 m` -- a WIDER stop radius, so the arm trips at three times the distance and stops MORE often. Swept 0.005-2.9 m at runtime through `AutonomousArm.step`: **there is no reading where the raw path trips and the normalized path does not.** The substitution is strictly MORE conservative, so it is a CAPABILITY defect (a 3x false-stop radius) and **the safety envelope is not breached by it**. Band 1 holds only genuine open breaches of a never-relaxed invariant; this is not one. It lands in Band 10 because what shipped is a criterion carrying a constructed reachable red, which is what P8 governs.
- **still worth having.** A silent 3x move in a safety threshold is a defect whichever way it points, the units contract was asserted nowhere, and the direction is now pinned by its own test so the record cannot drift back to "loosens".

## Band 11 -- repo hygiene (not a policy)

8 item(s).

### [B11.01] Make cite re-derivation a per-commit discipline and mechanize what citation_guard can catch - 4 lanes each carry rotted cites in their own task lists

- **status** OPEN &nbsp;|&nbsp; **policy** hygiene &nbsp;|&nbsp; **effort** small
- **key** `cite-rederivation-per-commit`
- **source** specs/014 tasks.md T019,T028,T042,T057,T065; specs/001 tasks.md (8 rotted cites); specs/005 tasks.md (6 rotted cites); specs/015 tasks.md T012/T013/T014/T031
- **evidence** COLLAPSED 4 rows; specs/001 misses 8 of 8 (affective_memory :173->:175, autonomous_loop :30->:31, :41->:44, :1548->:1911, :1570->:1933, time_cells :146->:232, intrinsic_plasticity :31->:37, home :152->:361), specs/005 misses 6 of 6, specs/014 carries :292 and :2269, and specs/015's criterion-identity/permutation/red/post-adoption-gates evidence files never existed under those names; citation_guard.py:122's regex needs a path separator and discards :NNN, so this class is only half-mechanizable today.

### [B11.02] Fix the frozen rule string still naming reasoning/world_kb_stream.py - a dead path is being emitted into a machine artifact

- **status** OPEN &nbsp;|&nbsp; **policy** P4c + hygiene &nbsp;|&nbsp; **effort** small
- **key** `world-kb-stream-path-moved`
- **source** tools/lang_coverage.py:8,:68; specs/005-language-organ/tasks.md:151
- **evidence** world_kb_stream.py and rose_language.py were promoted to cortex/ and lang_coverage.py:54 imports the new path, but its docstring :8 and the machine-emitted lexicon_source at :68 still print 'reasoning/world_kb_stream.py::load_kb()' - the HEAD re-run reproduced that dead path verbatim into the artifact.

### [B11.03] Run the three gates plus citation_guard and record the exact counts in specs/009

- **status** OPEN &nbsp;|&nbsp; **policy** hygiene &nbsp;|&nbsp; **effort** small
- **key** `gates-not-recorded-in-009`
- **source** specs/009-sparse-connectivity/tasks.md T012 + T030
- **evidence** COLLAPSED 2 rows (same task); citation_guard --assert exits 0 at HEAD with STRICT 0 unexempted / 1 exemption / BROAD 717 report-only, but no gate output or count is recorded anywhere under specs/009 - the deliverable is the record, and it is absent even where the gate is green.

### [B11.04] Update docs/coverage/ with the measured sparse-connectivity result, citing file:line

- **status** OPEN &nbsp;|&nbsp; **policy** hygiene &nbsp;|&nbsp; **effort** small
- **key** `009-result-not-in-coverage-docs`
- **source** specs/009-sparse-connectivity/tasks.md T028
- **evidence** A grep for '009\|sparse connectivity\|fanout' across docs/ returns zero hits in docs/coverage's 12 .md files (only Perception.md matched an unrelated line-number token) - the US1 result is recorded nowhere.

### [B11.05] Strike the stale '(BLOCKED - see above)' marker on the Phase 7 header

- **status** OPEN &nbsp;|&nbsp; **policy** hygiene &nbsp;|&nbsp; **effort** small
- **key** `015-phase7-header-stale-blocked`
- **source** specs/015-resolve-weight-transport/tasks.md:188
- **evidence** Line 188 still reads 'Phase 7 ... *(BLOCKED - see above)*' while T040-T042 are all [X], line 159 says RESOLVED, and both phases' output is live in POLICIES.md (:1635 mlp_q RESOLVED, :1686 predictive_coding ADJUDICATED - I re-derived both).

### [B11.06] Strike three task groups that measurement refuted - leaving them unchecked reads as unfinished work

- **status** OPEN &nbsp;|&nbsp; **policy** hygiene &nbsp;|&nbsp; **effort** small
- **key** `strike-refuted-task-rows`
- **source** specs/007 tasks.md T008 + T020; specs/015 tasks.md T036-T039
- **evidence** COLLAPSED 3 rows: specs/007 T008 would reproduce a baseline its own spec.md retracted twice (amendments 3 and 4), T020 deletes reasoning/readout_ensemble.py which does not exist (promoted to developmental_loop_brain.py, and T021 recorded the opposite disposition), and specs/015 T036-T039 are the NULL branch of a mutually-exclusive decision the evidence resolved GO (evidence/null.md correctly does not exist).

### [B11.07] Add timeout-minutes to the four CI job keys

- **status** OPEN &nbsp;|&nbsp; **policy** ops hygiene &nbsp;|&nbsp; **effort** small
- **key** `ci-jobs-no-timeout-minutes`
- **source** specs/BACKLOG_RANKED.md row 71
- **evidence** `grep -c timeout .github/workflows/ci.yml` returns 0 with a positive control `runs-on` = 4 in the same file, and ci.yml is the only workflow, so all four jobs fall back to the 360-minute default.

### [B11.08] Adopt the pin-to-a-commit rule for the consolidated spec - and do not copy row 72's pin, which rotted in one day

- **status** OPEN &nbsp;|&nbsp; **policy** invariant: verify-first &nbsp;|&nbsp; **effort** small
- **key** `pin-followup-to-commit-not-head`
- **source** specs/BACKLOG_RANKED.md row 72
- **evidence** The premise re-derives (HEAD 39c412f9 != origin/main, 4 worktrees, 16 local branches) but its prescription is already wrong: origin/main advanced 462da205 -> f362cb52, so 'Pin to 462da205' is the defect it names; the consolidated spec should pin to 39c412f9 and say so.

---

## CLOSED at HEAD -- do not resurrect

These are not work. They are here because the three lanes that landed on 2026-08-14/15 closed
them, and a consolidated list that omits them invites the next lane to re-open work that is
already done. Each was verified at HEAD, not inherited.

### [X.01] DO NOT RESURRECT - the 018 lane CLOSED these and CHANGED the ground under three others

- **source** HEAD 39c412f9 + ebf79ad2 (018 lane, landed today)
- **evidence** CLOSED: e-prop transport is now PARTIALLY mechanized (substrate/eprop_transport_gate.py, CORE, GOVERNED 0 / UNGOVERNED 1) per POLICIES.md:1561-1584 which I read at HEAD; instrument_guard extended to src/ + CONSTRUCTOR-SEED rule; dormancy_gate now mechanizes the e-prop binding constraint. CHANGED: dormancy_gate.py:551 `assert not cov['stale_exempt']` makes B5.03 a red rather than a nit; the gate CREATED B1.02 (gf.py) as a new finding; and it made CLAUDE.md:50 false (B1.01).

### [X.02] DO NOT RESURRECT - 015 closed mlp_q weight transport, the predictive_coding adjudication, the NULL branch, and its own final gate batch

- **source** specs/015-resolve-weight-transport (merged 17548c6f, in origin/main)
- **evidence** POLICIES.md:1635 RESOLVED 2026-08-14 (mlp_q -> Kolen-Pollack) and :1686 ADJUDICATED 2026-08-14 NOT a violation (predictive_coding w1.T feeds an ACTIVATION) both re-derived by me; spec-010 T051 (jepa) and T052 (predictive-coding) are closed with STALE cites; T036-T039 can never execute; T055's record lives in the 17548c6f commit body (11/11 exit codes, pytest 91.00%), never in tasks.md.

### [X.03] DO NOT RESURRECT - spec-010 items verified CLOSED at HEAD: T026, T037, T042, T049, T053, T055, T058, T060, T062

- **source** specs/010-policy-compliance-audit/tasks.md
- **evidence** P0 self_improve row corrected (POLICIES.md:115); _episodic fed from the lake (brain.py:975-976, cite :909-910 STALE); the 0.6 salience hardcode replaced and pinned by its own red (autonomous_loop.py:1233/:1544/:2243-2277); substrate/device.py::select consumed at neurogenesis_engine.py:199 (the task's training/device.py path never existed); import-time reseeds 0 hits against a 529-hit positive control; REFUTED section exists at findings.md:862; P8 first-run output at POLICIES.md:1515-1519; envelope SHA256 diff clean; arcengine discharged (dependency_ledger --assert exits 0).

### [X.04] DO NOT RESURRECT - 007 T002/T017/T036/T044/T045/T046/T047/T048, 009 T017/T032, 011 T006/T007/T008, 005 T017/T049/T050/T058/T066

- **source** specs/007, specs/009, specs/011, specs/005 tasks.md
- **evidence** 007: safety-seam refusal recorded at spec.md:451-473, three gates green at HEAD, specs/003 record already carries both halves at bounding.md:30-45, LIF-growth misnomer purged, no new module to register, citation_guard exit 0. 009: memory ceiling recorded with its binding constraint at research.md:133-164, _T_SPIKE retired with a reason at home.py:60-72. 011: the input port landed 2026-08-04 (cortical_microcircuit.py:242/:277/:582) two days BEFORE the spec was written, with SC-007's red already constructed and rewritten at :305-312. 005: baseline figures reproduce, the NULL-path substance holds, Language.md:278-292 was updated, and no 2401.17911 energy figure leaked.

### [X.05] DO NOT RESURRECT - the graph is at HEAD; the two auditors disagreed and the one calling it stale is wrong

- **source** specs/014 tasks.md T070; specs/015 tasks.md T054
- **evidence** COLLAPSED 2 rows and re-derived myself: graphify-out/graph.json carries built_at_commit 39c412f95c1443b06987f0dccb50d4f060fed769 == HEAD with 14174 nodes; the only absentee is the untracked eprop_transport_gate.py, so `graphify update .` belongs in the ship rule (B10.35), not the backlog.

---

## Corrections the synthesis made to the auditors

Recorded rather than silently applied: a consolidation that quietly overrides its own sources is
indistinguishable from one that fabricated them.

### [F.01] AUDITOR WRONG - 'graphify-out/graph.json predates this lane' is false at HEAD

I loaded the file: built_at_commit == HEAD 39c412f9, 14174 nodes; the auditor inferred staleness from a `graphify query` for sensor_feed returning nothing, but sensor_feed.py does not exist, so an empty result is the correct answer, not evidence of a stale graph - a negative from a command that cannot produce a positive.

### [F.02] AUDITOR WRONG - lake-census-missing was filed under closed_or_stale while an OPEN item names it as its blocker

The auditor's own evidence says the census 'is not filed' and only its DENOMINATOR is stale (4 tags stated, 7 exist), which is a re-scope, not a closure; spec-009-T035 lists blocked_by: lake-census-missing, so filing it closed orphans that dependency - re-opened here as B6.09.

### [F.03] AUDITOR WRONG - the idle-sentinel and raw-metres defects are marked blocked_by sensor-feed, but both are LIVE at HEAD in code that already exists

I re-derived both without any new module: substrate/esp32_encoder_path.py:78 has no distance sentinel while :81 has the temperature one, and autonomous_loop.py:1473-1479 binds a fear association for a finite -1.0; blocking them on an unbuilt feed defers two safety-path fixes that are landable today (promoted to B1.03/B1.04).

### [F.04] AUDITOR GAP - blocked_by 'selfbuild-proposer-empty' names a node no auditor filed, leaving the P0 critical path headless

No inventory row carries that dedupe_key; CLAUDE.md's P0 block states the work plainly ('Filling the proposer is the open P0 work; a call-site wire is not', proposed: 0 over 190 ticks), and it also gates coverage row godel_machine_selfbuild - filed as B2.03.

### [F.05] AUDITOR WRONG - blocked_by 'microcircuit-port-undriven' matches no dedupe_key in the inventory

The intended blocker is backlog-10's cortical-microcircuit-unwired-port; remapped in B5.18, otherwise the item is unschedulable.

### [F.06] AUDITOR OVERLAP - one backlog row and its one precondition were filed 8 times under 7 distinct dedupe_keys

Keys arc-fluid-cluster-triage / -unclassified / -promotion / gf-py-ungoverned-weight-transport-blocks-arc-promotion all describe row 40, and gf-mlp-hand-written-backprop / -handwritten-backprop / -ungoverned-weight-transport all describe the same gf.py:1795 finding; the evidence is consistent across all eight, so this is a keying failure, not a disagreement - collapsed to B1.02 + B4.08.

### [F.07] AUDITOR CLASSIFICATION ERROR - 'STALE_CLAIM' was used for two opposite things, and 8 rows of real open work were filed as closed

Some STALE_CLAIM rows mean 'the task is refuted, strike it' (007 T008/T020, 015 T036-T039) and others mean 'a live false claim needs removing' (001 T032-comment, 001 T025-claim, 001 cites, 005 CITES, 005 PATHS, 009 T034, 010 T059, 011 T040) - the second group is OPEN work and is re-filed here as B5.05, B6.01, B6.09, B10.08, B11.01, B11.02; only the first group belongs in B11.06.

### [F.08] AUDITOR OVER-LABELLING - 'invariant' was applied to unmet measurements and to criteria on unbuilt code, which would have put 9 non-breaches in Band 1

POLICIES.md:1542-1584 lists exactly safety envelope / e-prop / determinism / verify-first; an unrun F* bisect or an absent per-module metric is a measurement debt (banded P1) and a determinism test for a module that does not exist is a criterion owed (banded P8) - only 6 items are genuine open breaches, and B1 is sized accordingly.

---

## Band 9 is empty, and that is a result

No item bands to P7. Multi-policy items band at their LOWEST policy number, so a P7+P2 item
sits in Band 4. P7 removed one argument from the table -- *"it would be another dependency"* --
and nothing in the open work turns on it.


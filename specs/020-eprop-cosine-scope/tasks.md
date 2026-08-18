# Tasks: Scope the e-prop cosine clause, and prove the criterion can discriminate before it binds

**Input**: Design documents from `/specs/020-eprop-cosine-scope/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: REQUIRED, not optional. Standing Invariant 5 ("verify the verifier"), FR-004, FR-010a,
FR-015a, SC-007, SC-011 and SC-017 all demand a CONSTRUCTED failure rather than a happy path.

**Regenerated 2026-08-17** against the spec as amended by the second `/speckit-clarify` (35 FRs / 19
SCs). The prior 67-task list predated five decisions and did not cover any of them.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: no logical dependency on another incomplete task. **This repo commits ONE change at a time
  with the full gate green**, so `[P]` never means "commit together".
- **[Story]**: US1–US4 map to the four user stories in spec.md

## Names and homes, settled by measurement before generation

| thing | value | basis |
|---|---|---|
| checker module | `src/snn_data_lake/substrate/gradient_claim_gate.py` | `tools/` is at ZERO p4c headroom (30/30) for **`.py`** files (research.md R2). `graph-exists` found no duplicate |
| baseline artifact | `tools/gradient_claim_baseline.json` | p4c corpus is `*.py` only (`p4c_ratchet.py:114`, pathspec `:171`), proven by `p4c_ungoverned_baseline.json` living in that zone while it reads 30/30 green |
| declaration constant | `GRADIENT_CLAIM` | spec FR-002c/FR-002d, plan.md and contracts/cli.md; measured LEARN_VERBS-irrelevant (`method_dormancy_gate` inspects methods on classes only) |
| on-demand half | `--measure` flag of the SAME module | one module, not a second file (contracts/cli.md) |

## Verification cadence — the FAST LOOP, and when the full gate is actually required

**MEASURED 2026-08-17, on this machine, each command timed separately:**

| loop | cost | catches |
|---|---|---|
| `ruff check .` + `ruff format --check .` | **0.2s** | style, import order |
| `gradient_claim_gate --assert` + `--self-check` | **15.0s** | all five bounds, all six constructed reds |
| `pytest tests/test_gradient_claim_gate.py --no-cov` | **56.7s** | all 11 tests |
| **FAST LOOP total** | **72s** | |
| `pytest` (full, with coverage) | **1195.6s** | cross-module regressions, the 70% floor |
| `run_selfchecks.py` | **600s+** | every module's `__main__` |
| **FULL GATE total** | **~1795s (30 min)** | |

**Run the FAST LOOP per task. Run the FULL GATE per PHASE, and before any commit that touches
`src/`.** The fast loop is **25x cheaper** and covers everything a single task can plausibly break;
what it cannot see is a cross-module regression, which is a phase-boundary risk rather than a
per-task one.

Why this is a requirement and not a preference: the repo has a recorded CI-minutes wall due around
mid-September at ~250 min/day, `cancel-in-progress: true` means each push kills the previous run
(seven consecutive cancellations already happened once), and a 30-minute local cycle per task across
53 remaining tasks is ~26 hours of gate time. **Batch the pushes too** — wait for green between
pushes that touch `src/`.

**`pytest -q` is effectively `-qq` here** (addopts already carries `-q`), so the `N passed` line can
vanish. Count with `-v` and reconcile against `grep -c '^def test_'` — two independent counts, which
is how the 11 above were confirmed.

## Hard orderings that are requirements, not preferences

1. **T038 before any task in Phase 8.** FR-007/SC-005 — counts published before any governing text
   changes, verifiable from commit order. Three prior policy proposals died at this bar.
2. **Phase 7 (the `EXEMPT` merge) after Phase 3.** Port before retire; A/B before deleting.
3. **T065 before T070.** `stale_exempt`'s honesty property is ported before the dict it makes
   trustworthy is removed.
4. **T050 and T051 in the SAME commit.** Two copies of the 90.7% figure.
5. **T072 and T073 in the SAME commit.** FR-008a — a one-surface amendment IS the failure mode.
6. **T060 before T074.** Do not publish a provenance number naming the encoders before running them.
7. **Tests before their implementation** within each story.

---

## Phase 1: Setup

- [X] T001 Create `src/snn_data_lake/substrate/gradient_claim_gate.py` with a module docstring stating the FOURTH question ("does this rule OWE a cosine at all?") and `# distinct-from:` blocks naming `eprop_transport_gate`, `dormancy_gate`, `method_dormancy_gate`, per the incumbent's QUESTION-not-TECHNIQUE precedent at `eprop_transport_gate.py:63-77`
- [X] T002 Register the module in `REGIONS['substrate']` in `src/snn_data_lake/__init__.py` — **not bookkeeping**: `src/snn_data_lake/substrate` is ABSENT from `tools/p4c_ungoverned_baseline.json`'s zones (verified, `reasoning` as positive control) because registered modules are skipped as governed, and `compare()` treats an absent zone as allowing 0. Unregistered, this file manufactures a new zone at implicit baseline 0 and reddens p4c on its own commit
- [X] T003 Register the module's self-check in `run_selfchecks.py` in the NON-torch tier, so the static half runs in the stdlib gate
- [X] T004 Establish the baseline artifact at `tools/gradient_claim_baseline.json`, resolving the path by **REUSING `substrate/self_mod_harness._repo_root`** (anchored on `run_selfchecks.py` + `src`), with a `root=None` parameter for test injection, and record in the module docstring that it returns None under an installed wheel. **This task originally said to copy `doc_budget_gate.py:49-50`'s four-level COUNTED climb; that was wrong** — `laminar_local_readout._repo_root` records that a counted depth encodes the module's current nesting and silently resolves to the wrong directory when the file moves. The P2 hook blocked the hand-rolled version; the "heavy imports" objection to reuse was measured FALSE (47 ms / 111 modules, torch not loaded)
- [X] T005 [P] Replace the five `<module>` placeholders in `specs/020-eprop-cosine-scope/quickstart.md` (lines 22, 37, 40, 55, 70) with `gradient_claim_gate` — every command in the validation document is currently uncopyable
- [X] T006 [P] Replace the `<module>` placeholder in the report-mode heading of `specs/020-eprop-cosine-scope/contracts/cli.md:15`
- [X] T007 [P] Correct the Project Structure tree in `specs/020-eprop-cosine-scope/plan.md:130,140,143`: `<new classifier>` is decided, and `<on-demand instrument>` under `tools/` is now WRONG — the on-demand half is a `--flag` of the substrate module

**Checkpoint**: module exists, registered, gates green, and every design document is copy-pasteable.

---

## Phase 2: Foundational (Blocking Prerequisites)

**⚠️ No user story work can begin until this phase completes.** Every story reads the census.

- [X] T008 Port the overlay-capable walk `walk(root, overlay=None)` from `tools/p6_gate.py` into `src/snn_data_lake/substrate/gradient_claim_gate.py` — the only gate whose walk accepts an in-memory overlay, which is what lets `--self-check` construct a RED against the REAL tree touching nothing on disk
- [X] T009 [P] Implement `Disposition` in `src/snn_data_lake/substrate/gradient_claim_gate.py` — two values are insufficient because `reasoning/gf.py::MLP` is an owner-EXEMPT CONTROL and a two-value classifier condemns it. **Shipped with FIVE, not the four this task specified**: the first real declaration pass (T031) measured four insufficient — a diagnostic, a search, two symbolic-grammar operations and a combinator could not be described truthfully by any of them — so `NOT_A_RULE` was added under FR-002e with the drawer guards in FR-002f
- [X] T010 [P] Implement `ClassificationState` with six values including `MISSING_DECLARATION` and `UNDETERMINABLE` in `src/snn_data_lake/substrate/gradient_claim_gate.py`, per data-model.md
- [X] T011 Implement the `LearningRule` record in `src/snn_data_lake/substrate/gradient_claim_gate.py`, keyed per QUALIFIED symbol (`<module>::<Class>.<method>`) and carrying a `governed` flag — per-module keying is impossible because `cortex/eprop_kernel.py` holds a rule, its BPTT control and a surrogate `autograd.Function` in one file
- [X] T012 Implement the learning-rule ENUMERATION over the **WHOLE tree, governed and ungoverned alike** (FR-001), in `src/snn_data_lake/substrate/gradient_claim_gate.py`, and record in-file that the enumeration is itself a heuristic of unproven recall — research.md R1 measured a hand-named control of 32 canonical rules hitting 28
- [X] T013 Implement the governed/ungoverned PARTITION in `src/snn_data_lake/substrate/gradient_claim_gate.py` (FR-001a), deriving `governed` from REGIONS/TORCH_TIER/TRAIN_TIER membership the same way the sibling gates do — the census walks everything, the bounds assert on governed only
- [X] T014 Implement the declaration reader in `src/snn_data_lake/substrate/gradient_claim_gate.py`: locate the module-scope `GRADIENT_CLAIM` assign via `ast` and read it with `ast.literal_eval`, NEVER importing the analysed module — ~25.7% of the subtree does not import in the gate environment
- [X] T015 Implement the structural-inference recall net in `src/snn_data_lake/substrate/gradient_claim_gate.py` that can return only `CLAIM` or `UNKNOWN` and is structurally incapable of returning `NO_CLAIM` (FR-002b) — inference measured ~31% error in both directions with false negatives on `mlp_q`, `jepa_world_model`, `gf.MLP`, `predictive_coding`
- [X] T016 Implement the intra-module call-graph upgrade to the inference in `src/snn_data_lake/substrate/gradient_claim_gate.py` — in every claimant found, the gradient is computed in one function and applied in another (`grad_eprop`/`grad_bptt` → `Net.apply`), so neither a mutation-site scan nor a claim-site scan alone sees a rule
- [X] T017 Implement resolution to a single `ClassificationState` in `src/snn_data_lake/substrate/gradient_claim_gate.py`, declaration AUTHORITATIVE and inference unable to overrule it; no declaration plus `inferred == CLAIM` MUST resolve to `MISSING_DECLARATION`
- [X] T018 [P] Implement parse-failure accounting in `src/snn_data_lake/substrate/gradient_claim_gate.py` — 378/378 parse today, so a new failure is a silent census hole and the bound is ABSOLUTE, not a report

**Checkpoint**: the census computes and partitions. Nothing asserts yet.

---

## Phase 3: User Story 1 - Tell me which rules the bar applies to, and count them (Priority: P1) 🎯 MVP

**Goal**: a re-derived census classifying every learning rule in the whole tree, asserting on the
governed tier in CI, without a GPU and without importing the analysed modules.

**Independent Test**: run the census and confirm a known STDP variant classifies `CLASSIFIED_LOCAL`
and the e-prop reference classifies `CLASSIFIED_REFERENCE` — both derived from the tree, not a list.

### Tests for User Story 1 (write FIRST, confirm they FAIL)

- [X] T019 [P] [US1] Write `tests/test_gradient_claim_gate.py::test_a_local_rule_owes_nothing_and_the_reference_is_not_a_claimant` — a known STDP variant resolves `CLASSIFIED_LOCAL`, `cortex/eprop_kernel` resolves `CLASSIFIED_REFERENCE`
- [X] T020 [P] [US1] Write `tests/test_gradient_claim_gate.py::test_a_non_importing_module_is_still_classified` naming a module known not to import in `.venv-lake` (SC-010)
- [X] T021 [P] [US1] Write `tests/test_gradient_claim_gate.py::test_update_baseline_refuses_a_rise` constructing an attempt to record a worse baseline and asserting exit 1 (SC-011) — demonstrated by construction, never asserted in prose
- [X] T022 [P] [US1] Write `tests/test_gradient_claim_gate.py::test_a_dead_walk_fails_loudly` driving `rules_seen` to zero via the overlay and asserting the liveness floor fires (SC-007, FR-006)
- [X] T023 [P] [US1] Write `tests/test_gradient_claim_gate.py::test_an_ungoverned_finding_is_reported_but_never_asserted` (FR-001a) — an ungoverned rule with no declaration must appear in the census and must NOT redden `--assert`
- [X] T024 [P] [US1] Write `tests/test_gradient_claim_gate.py::test_a_ratchet_at_its_floor_reads_SATISFIED_not_merely_green` (SC-018, FR-017a) — with a control at a non-floor count reading GREEN, so the two states are shown distinguishable

### Implementation for User Story 1

- [X] T025 [US1] Implement report mode (default, exit 0 regardless of findings) in `src/snn_data_lake/substrate/gradient_claim_gate.py` printing the census block from `contracts/cli.md` with **governed and ungoverned subtotals reported separately** (SC-001)
- [X] T026 [US1] Implement `--assert` in `src/snn_data_lake/substrate/gradient_claim_gate.py` with five bounds scoped to the **GOVERNED tier** (FR-001a): `unclassified == 0` ABSOLUTE, `missing_declaration <= baseline` RATCHET, `undeterminable <= baseline` RATCHET, `rules_seen >= floor` LIVENESS, `parse_failures == 0` ABSOLUTE
- [X] T027 [US1] Implement SATISFIED-vs-GREEN reporting for BOTH ratchets in `src/snn_data_lake/substrate/gradient_claim_gate.py` (FR-017a) — SATISFIED when a count rests at its floor and the bound can never fire again, GREEN when it is above the floor and still live; and make the output say a green ratchet means "no worse", never "the scope is correct" (FR-017)
- [X] T028 [US1] Implement wall-clock cost measurement and printing in `src/snn_data_lake/substrate/gradient_claim_gate.py` (FR-013a) — MEASURED and PRINTED, never asserted, because wall-clock on shared runners is not deterministic and a fixed second-count would be a P1 magic constant
- [X] T029 [US1] Implement `--update-baseline` in `src/snn_data_lake/substrate/gradient_claim_gate.py` writing `tools/gradient_claim_baseline.json`, machine-written only, REFUSING a worse value and exiting 1 naming the offending counts (FR-018) — **reuse `tools/p4c_ratchet.py:441-449`** and its `_load_baseline` at `:255-268`, which reads `utf-8-sig` deliberately so a hand-edit's BOM cannot crash the gate
- [X] T030 [US1] Implement `--self-check` in `src/snn_data_lake/substrate/gradient_claim_gate.py` constructing all six reds from `contracts/cli.md` via the overlay, including red 5 (a GREEN control proving the repaired shape is NOT flagged) and red 6 (determinism across two runs, SC-009)
- [X] T031 [US1] Add `GRADIENT_CLAIM` declarations to the claimant and reference modules — `cortex/eprop_kernel.py`, `cortex/eprop_shd.py`, `cortex/eprop_nmnist.py`, `reasoning/eprop.py`, `basal_ganglia/mlp_q.py`, `cerebellum/jepa_world_model.py`, `cortex/predictive_coding.py` — keyed per qualified symbol with a one-line reason each
- [X] T032 [US1] Add `GRADIENT_CLAIM` declarations dispositioning the ~22 local rules as `LOCAL`, principally under `src/snn_data_lake/plasticity/`, taking exact membership from T038's census rather than from any prose list. **The "~22" prose estimate was low by 3.3x** — the census found **72** governed `LOCAL` symbols, which is exactly why the task said to take membership from the census and not from a list
- [X] T033 [US1] Wire `--assert` and `--self-check` into the CI lint job in `.github/workflows/ci.yml`, following the sibling gates' two-step pattern — a tool wired in BARE cannot redden the build
- [X] T034 [US1] Verify SC-019: the cost figure is printed on every run and cannot influence the EXIT CODE -- `elapsed` appears in no return expression, no branch condition, and nowhere inside `evaluate()`. **Restated during execution.** As written this task said "no assertion reads it", and that check was VACUOUS: the module gates by `return 1` from `main()` and contains zero `assert` statements, so it passed for a reason unrelated to the cost. Its own positive control caught that by reporting 0 asserts found. Controls now used: `findings` and `baseline` must be found in the same search, proving it can detect gating
- [X] T035 [US1] Verify SC-009 by running `--assert` twice on an unchanged tree and diffing the output byte-for-byte
- [X] T036 [US1] Verify FR-013's no-import guarantee by grepping the module for `importlib`/`__import__`/`exec` with a positive control — a classifier that imports what it classifies silently skips ~25.7% of the subtree
- [X] T037 [US1] Verify SC-001: the per-state counts sum to `rules seen`, and the governed and ungoverned subtotals sum to the whole
- [X] T038 [US1] **RUN `python -m snn_data_lake.substrate.gradient_claim_gate` against the real tree and COMMIT its counts** into `specs/020-eprop-cosine-scope/census-baseline.md`, recording the mis-scoped count (FR-007, SC-005). **No governing document may be edited before this commit lands.** — **RUN and WRITTEN; the COMMIT is held** at the owner's standing instruction (no commits until specs/020 finishes). Mis-scoped count: **79 of 97 real governed learning rules (81%) owe no cosine**; 18 owe one. Phase 8 stays blocked by this task's own gating clause until the commit lands, which is the intended behaviour, not a stall

- [X] T032a [US1] Disposition the governed `UNDETERMINABLE` residue -- 99 symbols with no declaration that the inference could not decide, which T032 never covered because it worked the `MISSING_DECLARATION` list. FR-016a ratchets this state DOWN precisely so it does not become a drawer, and 99 of 150 governed rules is a drawer. Each resolves to `NOT_A_RULE` (the recall-biased walk over-capturing) or to a real disposition. **The split is the deliverable**: every `NOT_A_RULE` is a measurement of ENUMERATION IMPRECISION, which is what lets the walk be tightened on evidence instead of the by-hand `step`/`grad` tuning done earlier. Known false positives to expect, one predicted by name in research.md R1: `cortex/adex_neuron::AdEx.step` (caught on `w`, the AdEx ADAPTATION CURRENT, not a synapse) and `brainstem/serial_arm::assert_calibrated` (caught on the `calibrate` substring, in the safety envelope). **DELIVERED: 99 -> 0.** The split is **52 `NOT_A_RULE` / 47 real dispositions**, and both of R1's named predictions resolved as predicted — `AdEx.step` is declared `NOT_A_RULE` (KEPT in the walk: its `w` augmented-assign is a real signal that happens to name an adaptation current), `assert_calibrated` was dropped outright by T032b. Governed totals now **149 symbols, 149 declared, 0 undeclared**; the control is `--self-check` red 1, which constructs an undeclared claimant and proves the count can be nonzero. Negative control on the coverage figure: **181 governed modules enumerate ZERO symbols**, so "87 modules carry a declaration" is a real subset and not a tautology

- [X] T032b [US1] **Tighten the enumeration on the evidence T032a produced, then ratchet the baseline down.** T032a's 52 `NOT_A_RULE` labels are a precision measurement of the walk, which is what the split was for. Scored against all 157 hand-reviewed labels: verb-only **57%**, weight-AugAssign-only **73%**, both signals together **94%**; per verb, `calibrate` was **0/5** and `plast` **0/3** — pure false-positive generators. Dropped both via `_DROPPED_VERBS` in `src/snn_data_lake/substrate/gradient_claim_gate.py`. **Measured BEFORE landing, because tightening already destroyed recall once in this feature** (dropping `grad` made `cortex/eprop_kernel` — the reference gradient — vanish from the census entirely while every gate stayed green): the drop removes **1 symbol and 0 real rules**, `rules_seen` 150 -> 149. Baseline re-derived to `MISSING_DECLARATION: 0, UNDETERMINABLE: 0, rules_seen_floor: 149` with the reason stored in the file. Both ratchets now read SATISFIED rather than merely NO WORSE

**Checkpoint**: US1 is independently shippable. The scope question is answerable from a re-derived
census, and it delivers value even if the invariant's wording never changes.

---

## Phase 4: User Story 2 - Refuse to let a criterion condemn a rule it could never pass (Priority: P2)

**Goal**: before the bar binds on a rule, the criterion is shown able to DISCRIMINATE for it. Where it
cannot, the INSTRUMENT is reported falsified and the rule is left UNJUDGED.

**Independent Test**: reconstruct the `mlp_q` feedback-alignment configuration and confirm the check
reports `INSTRUMENT_FALSIFIED`, not `RULE_FAILS` — a historical case with a known correct answer.

### Tests for User Story 2 (write FIRST, confirm they FAIL)

- [X] T039 [P] [US2] Write `tests/test_gradient_claim_measure.py::test_the_feedback_alignment_case_falsifies_the_instrument_not_the_rule` (SC-003). **REWRITTEN 2026-08-17 -- as originally worded this task would have planted a stale number and a vacuous assert.** (a) It said to pin `[-0.472, +0.279]`. MEASURED: shipped `python -m snn_data_lake.basal_ganglia.mlp_q` prints `[-0.4366, +0.1844]` today. The interval has DRIFTED and is stale in THREE places -- `mlp_q.py:92`, `mlp_q.py:178` and `docs/POLICIES.md:1694` -- against the original `[-0.4722, +0.2785]` in `specs/015/evidence/baseline-mlpq.txt`. The test MUST assert the stable PROPERTY (`|cos| > 0.5` in 0 of 8 draws, and the transport identity reading 1.000) and MUST NOT pin the interval. (b) An `assert sum(1 for c in cos if abs(c) > BAR) == 0` passes on an EMPTY list, so the test MUST first assert the sweep is non-empty and of the expected length. (c) The positive control MUST NOT be `_cos(f(W), f(W))` -- that is an algebraic identity that reads 1.0 with the hidden layer dead. Use the SHIPPED Kolen-Pollack path (+0.9031, clears the bar) as the control: it is a different configuration, not the same one twice
- [X] T039a [P] [US2] File the drift T039 found: record in `specs/020-eprop-cosine-scope/census-baseline.md` that three sites carry `[-0.472, +0.279]` while the module prints `[-0.4366, +0.1844]`. **Do NOT edit `docs/POLICIES.md`** -- T038's commit gates governing-document edits, and the correction belongs in the same amendment as T072/T074
- [X] T040 [P] [US2] Write `tests/test_gradient_claim_measure.py::test_unavailable_is_not_a_failure` asserting "no cosine available in this environment" is distinguishable in output from "cosine measured and below the bar" (FR-005, SC-002). **The obvious form is unconditionally true**: `replace(ok, aligned=0.21)` does not change `verdict`, so asserting `"UNAVAILABLE" not in rendered` holds no matter what the renderer does. The test MUST construct an actual `UNAVAILABLE` result and an actual below-bar `RULE_JUDGED` result and assert the two RENDER differently, with both directions checked
- [X] T040a [P] [US2] Write `tests/test_gradient_claim_measure.py::test_the_gate_imports_no_rule_module_in_any_mode` (FR-013b, SC-023) as a SUBPROCESS test carrying its own positive control in the same test -- an in-process `assert "torch" not in sys.modules` is red for an unrelated reason, because `tests/conftest.py` records that the suite deliberately imports torch. Cover BOTH CI modes, `--assert` and `--self-check`, not just one. Reachable red: hoisting `import torch` to module scope in the gate turns it red while the control stays green

### Implementation for User Story 2

- [X] T041 [US2] Implement `DiscriminationResult` per data-model.md in `src/snn_data_lake/substrate/gradient_claim_gate.py`, always carrying a PAIR -- no rule is ever judged on one number (FR-003) -- and carrying `coherence` alongside `null_sample` (FR-009d). Enforce the pair with an explicit `raise`, NOT a bare `assert`: `python -O` strips asserts, so "enforced structurally" would be conditional on the optimisation level
- [X] T041a [US2] Implement `SeparationReport` with `bar_sd_above_null` and `aligned_sd_above_null` as SEPARATE fields plus pinned `draws`/`seed` (FR-009e). MEASURED on the anchor fixture: `0.5` sits at +3.06 sd while the aligned cosine sits at +5.93 sd -- one field feeding both readings is right in the printed line and wrong in the JSON
- [X] T041b [US2] Implement the FR-009d coherence guard in `src/snn_data_lake/substrate/gradient_claim_gate.py`: compute `|mean|/rms` for candidate and reference, and return `INSTRUMENT_FALSIFIED` above the ceiling rather than reporting a separation. MEASURED justification, kept with the code: at coherence 0.99 the mandated null has mean +0.980, P(null>0.5) = 1.0000 and the aligned cosine sits 1.75 sd BELOW it -- the statistic inverts, blessing a shuffled rule and condemning an aligned one
- [X] T042 [US2] Implement per-fixture gradient extraction in `src/snn_data_lake/substrate/gradient_claim_gate.py` running each fixture in a SUBPROCESS (FR-013b), so the gate imports no rule-carrying module in any mode and the 89 `never imported by it` stamps stay true. The aligned pair is the rule's own gradient against an INDEPENDENT reference; the non-aligned input is the FR-009a permutation of the candidate against the UNCHANGED reference
- [X] T043 [US2] Implement the `INSTRUMENT_FALSIFIED` verdict in `src/snn_data_lake/substrate/gradient_claim_gate.py` (FR-004 = P8 clause 1 applied reflexively), and prove it REACHABLE by the FR-009d coherence guard on a constructed coherent pair (SC-022). **A verdict with no real instance is a verdict nobody has ever seen fire**; the guard is what gives it one, in the same run that returns `RULE_JUDGED` for the real fixture
- [X] T044 [US2] Implement the `UNAVAILABLE` verdict and exit code 2 in `src/snn_data_lake/substrate/gradient_claim_gate.py`, structurally distinct from a failed measurement. Under FR-013b this is NATURAL rather than special-cased: a fixture whose subprocess fails to import is unavailable. Exit 2 MUST have a reachable red -- state precisely which condition triggers it and construct that condition in a test
- [X] T045 [US2] Implement `--measure` mode in `src/snn_data_lake/substrate/gradient_claim_gate.py` printing the DISCRIMINATING POWER block from `contracts/cli.md` including the `null=/draws=/seed=` header and the `coherence cand/ref` line, and confirm it does NOT assert in CI (FR-014)
- [X] T045a [US2] Write the FR-015 artifact from `--measure` and verify SC-024 by running it twice on an unchanged tree and diffing byte-for-byte. Unpinned, the null moves `0.5` between +2.97 and +3.27 sd across seeds, so the artifact would differ on an unchanged tree and its own staleness signal would be noise

**Checkpoint**: no rule can be condemned by a criterion that could never pass it.

---

## Phase 5: User Story 3 - Measure the rule that actually ships (Priority: P3)

**Goal**: reconcile the reference gradient with the kappa-filtered factorization the shipped encoders
run, and prove by measurement that the criterion is attached to the rule that ships.

**P3 is a BUILD-ORDER position, not optional.** US2's discriminating-power check is expected to FLAG
this mismatch on its own, and fixing an instrument while holding the detector that catches the defect
beats editing it first and hoping. **Do not reorder ahead of US2.**

### Tests for User Story 3 (write FIRST, confirm they FAIL)

- [X] T046 [P] [US3] Write `tests/test_gradient_claim_measure.py::test_perturbing_a_shipped_encoder_moves_the_criterion` (FR-012) — two factorizations can look equivalent and score differently, so inspection cannot substitute
- [X] T047 [P] [US3] Write a truncation-regime case in `tests/test_gradient_claim_measure.py` at `g_rec=50` where the two forms DO separate verdicts (0.7376 GREEN vs 0.4105 RED) — at initialisation healthy cells sit 0.26–0.50 above the bar, so a test there would be vacuous

### Implementation for User Story 3

- [X] T048 [US3] Reconcile `_TinyNet.eprop_grads` in `src/snn_data_lake/cortex/eprop_kernel.py` to the kappa-filtered form both shipped encoders run (~13 insertions / 7 deletions in one function) — **this is FR-010's single named exception**
- [X] T049 [US3] Verify `tests/test_eprop_train_tier_unreachable.py::test_the_reference_gradient_is_not_a_hazard` stays green — it asserts `cortex.eprop_kernel` is in BRAIN_MODULES, reachable, and NOT a hazard (one function calling both `.backward()` and `.step()`); the reconciliation reds this if it puts an optimizer step alongside `_fwd_bwd`'s `.backward()`
- [X] T050 [US3] Re-derive the seven recorded numbers inside `src/snn_data_lake/cortex/eprop_kernel.py` comments: cos_hidden 0.9714→0.9993, cos_all 0.8236→0.9993, RED cosine −0.2540→−0.1782, the "90.7% of squared norm" figure→81.2%, and three seeded-defect rows 0.5859/0.7465/0.7398→0.6504/0.9214/0.9155
- [X] T051 [US3] Co-edit the SECOND copy of the 90.7% figure in `src/snn_data_lake/basal_ganglia/mlp_q.py` in the SAME commit as T050 — a claim living in two places gets corrected in one
- [X] T052 [US3] Correct the `_self_check` claim in `src/snn_data_lake/cortex/eprop_kernel.py` that the readout gradients are "IDENTICAL between e-prop and BPTT by construction" — measurably false for the present form (cos_readout 0.986830, relative L2 7.423e-01) and exactly true under the filtered form (cos 1.000000); this is what the reconciliation actually buys
- [X] T053 [US3] Report every before/after cosine movement caused by the reconciliation, with both values, in `specs/020-eprop-cosine-scope/census-baseline.md` and in `eprop_kernel.py`'s `_self_check` output (SC-008) — zero silently-absorbed movements
- [X] T054 [US3] Record in `src/snn_data_lake/cortex/eprop_kernel.py` that this fixes an inline TEST FIXTURE, not the guard protecting the shipped encoders — all three mechanizations already run in the CI torch job and each encoder guards itself on its own form (research.md R0)

**Checkpoint**: the criterion measures the factorization that ships, proven by a moved number.

---

## Phase 6: User Story 4 - Say plainly what the number is, without moving it (Priority: P4)

**Goal**: measure the aligned-vs-null separation, publish where `0.5` sits, and make the record's
staleness visible. **The bar does not move.**

- [X] T055 [P] [US4] Implement the within-matrix permutation null in `src/snn_data_lake/substrate/gradient_claim_gate.py` — permute `W_in` coordinates among themselves and off-diagonal `W_rec` among themselves, then cosine against the UNCHANGED reference; precedent at `tests/probe_015/criterion.py::permutation_null`
- [X] T056 [US4] Implement the three REPORTED-ONLY nulls (isotropic matched-norm, within-UNIT permutation, random relabel) in `src/snn_data_lake/substrate/gradient_claim_gate.py`, asserting on NONE (FR-009c); record that the untrained twin is REJECTED because it moves the net, the forward path and the reference at once
- [X] T057 [US4] Implement `SeparationReport` per data-model.md in `src/snn_data_lake/substrate/gradient_claim_gate.py`, reporting sd-above-null and P(null > 0.5)
- [X] T058 [US4] Implement the Provenance record in `src/snn_data_lake/substrate/gradient_claim_gate.py` (FR-015): the recorded `--measure` output PLUS a CONTENT DIGEST of the measured inputs, deriving STALE when the digest no longer matches the tree and CURRENT when it does
- [X] T059 [P] [US4] Write `tests/test_gradient_claim_measure.py::test_changing_one_measured_input_flips_the_record_to_stale` (SC-017, FR-015a) — a digest never recomputed reads identical to one that always matches, so the staleness signal owes its own reachable red
- [X] T060 [US4] RUN both encoders' actual smoke paths and compare to the `_TinyNet` transcription on matched geometry BEFORE publishing any provenance number naming those modules — the encoders were never EXECUTED for these figures, only transcribed (research.md R4 risk 1)
- [X] T061 [US4] Record in the `--measure` output block of `src/snn_data_lake/substrate/gradient_claim_gate.py` that every separation measurement is at INITIALISATION, and that e-prop truncation error grows as recurrence strengthens, so a trained net could read materially lower
- [X] T062 [US4] Verify by `git diff` on `docs/POLICIES.md` and `.specify/memory/constitution.md` that the bar `0.5` is UNCHANGED (SC-013, FR-009b); if the measurement shows it is poorly placed, record that in `specs/020-eprop-cosine-scope/census-baseline.md` as a finding for a separate decision and do NOT move it

**Checkpoint**: `0.5` has a published provenance, an unchanged value, and visible staleness.

---

## Phase 7: The P2 merge - retire `eprop_transport_gate.EXEMPT` (lands LAST)

**Purpose**: without this the feature ships a P2 violation of its own making — two hand-maintained
records of one symbol's disposition. **Port before retire.**

- [X] T063 [P] Write `tests/test_gradient_claim_gate.py::test_the_transport_gate_is_byte_identical_after_adopting_the_registry` capturing the incumbent's findings and census before and after (FR-019b, SC-015)
- [X] T064 [P] Write `tests/test_gradient_claim_gate.py::test_an_exemption_blessing_nothing_still_fails_loudly` — the ported honesty property needs its own red, not an assertion that it was ported
- [X] T065 Port `stale_exempt`'s honesty property into the registry path in `src/snn_data_lake/substrate/gradient_claim_gate.py` FIRST (FR-019a) — every exemption must still match a REAL finding, so a blessing that blesses nothing fails loudly; that property is what makes the incumbent's dict trustworthy
- [X] T066 Declare `MLP.train` as `CONTROL` in `src/snn_data_lake/reasoning/gf.py` carrying the owner's full adjudication text, so it lives beside the code it describes rather than in a central list that drifts
- [X] T067 Make `src/snn_data_lake/substrate/eprop_transport_gate.py` resolve a finding's disposition by reading the owning module's `GRADIENT_CLAIM` instead of its own `EXEMPT` dict — update all use sites at `:323-326`, `:345-350`, `:354`, `:393-395`
- [X] T068 Resolve the keying hazard in `src/snn_data_lake/substrate/eprop_transport_gate.py`: `_exempt_key(rel, fn)` at `:142` builds its key from the ENCLOSING FUNCTION name (`reasoning.gf.train`, a bare `train`) while the registry is keyed per QUALIFIED symbol (`MLP.train`) — a naive match silently misses
- [X] T069 Run `python -m snn_data_lake.substrate.eprop_transport_gate` before and after the swap and confirm findings and census are BYTE-IDENTICAL (SC-015)
- [X] T070 Delete `EXEMPT` from `src/snn_data_lake/substrate/eprop_transport_gate.py:128` and repair its self-check reds at `:519-540`, which plant a key into `EXEMPT` directly and must now plant into the registry path
- [X] T071 Verify SC-014 by grepping the tree with a positive control: exactly ONE hand-maintained record of any symbol's gradient disposition exists, and `EXEMPT` is absent from `eprop_transport_gate.py`

**Checkpoint**: one record of one symbol's disposition. The P2 debt this feature would have created is closed.

---

## Phase 8: Governing documents (AFTER T038, per FR-007/SC-005)

**⚠️ Do not start until T038 has landed.** Counts published before wording changes, verifiable from
commit order.

- [X] T072 Amend the e-prop invariant's SCOPE in `docs/POLICIES.md` in **~250 words** — state that the clause is a PIVOT rule governing work that was about to use backprop, not a requirement on every learning rule, and that a rule which cannot pass the criterion by construction has falsified the INSTRUMENT rather than itself (FR-008)
- [X] T073 Amend Standing Invariant 8 in `.specify/memory/constitution.md` in the SAME commit as T072 (FR-008a) — it restates the clause verbatim, and the constitution's own Governance section makes the constitution "the defect" where the two disagree
- [X] T074a Settle FR-002h in `docs/POLICIES.md` in the SAME amendment as T072: state LIVE-PATH REACHABILITY as the general criterion deciding whether an autograd trainer is inside the cosine clause's scope, promoting it from the per-module note `developmental_loop_brain` carries verbatim from specs/010 to a clause the census can cite — and say what happens when reachability CHANGES, since that constraint's own wording admits nothing currently announces the flip. Then bring the TWO genuine autograd copies (`developmental_loop_brain::_LinearReadout.train_step`, `reasoning/_loop_common::LinearReadout.train_step`, the M21 duplication) to the same disposition and verify SC-021. NOT three sites: `cortex/autapse_neuron::_LinearReadout.train` is a stdlib delta rule sharing only the name, measured and declared LOCAL
- [X] T074 Record the numeric bar's provenance in `docs/POLICIES.md` (FR-009): the measured aligned-vs-null separation, the named null, whether `0.5` falls inside the separating region, and the P1 case-(a) justification for a fixed point on a normalized dimensionless output
- [X] T075 Run `python -m snn_data_lake.substrate.doc_budget_gate --assert` — headroom is **541 words** and real policy blocks in that file run 548–747, so either the amendment lands near 250 words or the ceiling is raised in the SAME commit with its reason stated there

**Checkpoint**: both governing surfaces agree, and neither asserts anything the checker has not counted.

---

## Phase 9: Polish & Cross-Cutting Concerns

- [X] T076 [P] Write `tests/test_gradient_claim_gate.py::test_the_fr010_check_fires_on_a_constructed_violation` — plant a change to a `brainstem/` path in a synthetic diff and assert the check reports it (FR-010a); a check never shown to produce a finding cannot evidence its absence
- [X] T077 Implement the FR-010 diff verification in `src/snn_data_lake/substrate/gradient_claim_gate.py` and confirm SC-016: the feature's diff touches ZERO safety-envelope files and zero learning-rule bodies outside FR-010's single named exception (`_TinyNet.eprop_grads` plus the recorded-number corrections in `eprop_kernel.py` and `mlp_q.py`)
- [X] T078 [P] Record the `--measure` output plus its content digest to `tools/gradient_claim_baseline.json`'s sibling artifact so "nobody has run it lately" is visible rather than indistinguishable from "it passed" (FR-015)
- [X] T079 Verify SC-006 by diffing `python -m snn_data_lake.substrate.eprop_transport_gate` output against its pre-feature baseline: ZERO learning rules newly reported as violations — a rule compliant before and failing after means the scoping is wrong, not the rule
- [X] T080 Verify SC-012 from the census output: the disagreement count falls to zero, or every remaining disagreement is NAMED in `specs/020-eprop-cosine-scope/census-baseline.md` with the reason it is unresolved — a number with no accompanying list is not an acceptable end state
- [X] T081 Run the FULL GATE capturing EACH exit code SEPARATELY with `.venv-lake\Scripts\python.exe` — `ruff check .`, `ruff format --check .`, `run_selfchecks.py`, `pytest`. A chained run reports only the last, and the `python` on PATH is the hermes venv with no ruff. **~30 min measured; this is a PHASE-boundary task, not a per-task one — see the Verification cadence section above for the 72s fast loop that belongs between tasks**
- [X] T082 Walk every scenario in `specs/020-eprop-cosine-scope/quickstart.md` and confirm each behaves as written (now possible, given T005)
- [X] T083 Run `graphify update .` to keep the knowledge graph current after the new module lands

---

## Dependencies & Execution Order

- **Phase 1 Setup**: no dependencies
- **Phase 2 Foundational**: depends on Phase 1 — **BLOCKS all user stories**
- **Phase 3 US1 (P1)**: depends on Phase 2. **MVP.**
- **Phase 4 US2 (P2)**: depends on Phase 2; uses US1's census for scoping only
- **Phase 5 US3 (P3)**: depends on Phase 4 — US2's check is the detector expected to FLAG the mismatch US3 fixes. **Do not reorder.**
- **Phase 6 US4 (P4)**: depends on Phase 4 (needs `--measure`); independent of US3
- **Phase 7 merge**: depends on Phase 3 (registry must exist and its census be counted)
- **Phase 8 docs**: depends on **T038** specifically, not merely on Phase 3
- **Phase 9 polish**: depends on everything it verifies; T077 depends on the whole diff existing

### Parallel Opportunities

- T005, T006, T007 — three different design documents
- T009, T010, T018 — distinct concerns in the new file; parallel once it is sectioned
- T019–T024 — six independent test functions
- T039, T040 · T046, T047 · T055 · T059 · T063, T064 · T076, T078

---

## Parallel Example: User Story 1 tests

```bash
# Six independent test functions, writable together:
Task: "test_a_local_rule_owes_nothing_and_the_reference_is_not_a_claimant"
Task: "test_a_non_importing_module_is_still_classified"
Task: "test_update_baseline_refuses_a_rise"
Task: "test_a_dead_walk_fails_loudly"
Task: "test_an_ungoverned_finding_is_reported_but_never_asserted"
Task: "test_a_ratchet_at_its_floor_reads_SATISFIED_not_merely_green"
```

---

## Implementation Strategy

### MVP: Phases 1–3, through T038

At T038 the feature has already delivered its headline value: "does this rule owe a cosine?" is
answerable from a re-derived census, the governed half asserts in CI, and the project's own
count-first invariant is satisfied. Nothing after T038 is required for that claim to hold.

**STOP and VALIDATE** at that point against quickstart Scenarios 1–3.

### Incremental delivery

- +US2 → no rule can be condemned by a criterion it could never pass (Scenario 4)
- +US3 → the criterion measures the rule that ships (Scenario 6)
- +US4 → `0.5` has a published provenance, unchanged, with visible staleness (Scenario 5)
- +Phase 7 → the P2 debt is closed, one record per symbol
- +Phase 8 → both governing surfaces state the scope, after the counts

### The risk this plan accepts, stated rather than discovered

SC-008 accepts that reconciling the reference gradient MAY move the reported cosine for the shipped
encoders. If a rule that measured above the bar now measures below it, that is a real finding about
the rule and is reported as such — not absorbed, and not a reason to abandon the reconciliation. This
is also why the BAR is frozen (FR-009b): moving it in the same feature would leave any flipped verdict
with two candidate causes and no way to attribute it.

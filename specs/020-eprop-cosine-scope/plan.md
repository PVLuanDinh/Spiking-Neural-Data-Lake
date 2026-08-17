# Implementation Plan: Scope the e-prop cosine clause, and prove the criterion can discriminate

**Branch**: `020-eprop-cosine-scope` | **Date**: 2026-08-17 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/020-eprop-cosine-scope/spec.md`

## Summary

The e-prop invariant's `cosine > 0.5` admission bar has an unwritten SCOPE, a documented history of
failing to discriminate, and an instrument that measures a factorization neither shipped encoder
runs. This feature makes the scope explicit and checkable, and does it in the order this project
requires: **the checker is built and its output counted BEFORE any governing text changes.**

Two halves, split by cost (clarified 2026-08-17):

- a **static classifier** that asserts in CI -- which learning rules even CLAIM gradient
  approximation, determined from two independent signals (declaration + structural inference) that
  are cross-checked, with disagreement reported rather than resolved;
- an **on-demand instrument** that measures the cosine, proves the criterion can discriminate for the
  rule under test, and measures where `0.5` sits between aligned and null gradients.

The bar's VALUE is frozen for this feature. Only its scope, its instrument, and its provenance move.

## Technical Context

**Language/Version**: Python 3.14.6 free-threading (`requires-python = ">=3.14"`, runtime guard
refuses GIL builds).

**Primary Dependencies**: stdlib `ast` for the static half (no import of the analysed module -- see
Constraints). `torch` 2.11.0+cu128 locally / CPU torch in CI, plus `numpy`, for the numeric half.
**No new dependency is proposed.** Per the constitution, "no new dependency" is no longer evidence of
anything on its own, so stated positively: nothing here needs a library the tree lacks, and the
static half deliberately uses `ast` rather than an import-based reflection so it can classify modules
that do not load.

**Storage**: one machine-generated ratchet baseline artifact, and a recorded output for the on-demand
instrument. Neither is brain state. Precedent: `tools/p4c_ungoverned_baseline.json` is exactly this
shape, so no new storage pattern is introduced and P4b is not engaged (see Constitution Check).

**Testing**: `pytest` (>=70% line coverage floor), each new module's own `__main__` self-check
registered in `run_selfchecks.py`, and `ruff check` + `ruff format --check`.

**Target Platform**: Windows 11 local with an RTX 5070; CI on {ubuntu, windows} with **no GPU** and a
CPU-only torch build.

**Project Type**: single project -- a governed Python package (`src/snn_data_lake/`) plus a gate/tool
tier. No frontend, no service.

**Performance Goals**: the CI half must be comparable to the existing AST gates (`dormancy_gate`
walks the package in seconds); it runs on every CI job so it cannot be expensive. The on-demand half
has no latency budget -- it is invoked deliberately.

**Constraints**:
- CI has no GPU, and CI runs a **different torch build** from the local gate, so a build-sensitive
  numeric assertion can be green in one and red in the other. This is the documented reason the
  numeric half does not assert in CI.
- The static half MUST classify modules that **do not import** in the gate environment (~30% of the
  ungoverned subtree). This forbids an import-based or reflection-based classifier.
- `docs/POLICIES.md` sits near its enforced word ceiling (28,559 / 29,100 measured 2026-08-17, 1.9%
  headroom), so the amendment's wording budget is tight and may require offsetting cuts.
- The five `brainstem/` safety-envelope files are out of scope entirely.

**Scale/Scope**: ~481 Python files / ~150k lines across `src`, `tests`, `tools`. Roughly 22 local
learning rules that owe nothing, plus a small claimant set (exact membership is Phase 0 research).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment |
|---|---|
| **P0** autonomous self-improvement | **PASS, and directly serves it.** P0's honesty guard states "a measurement that cannot fail is not a measurement". This feature mechanizes exactly that guard for the one criterion the e-prop invariant rests on. It adds no avenue and claims none. |
| **P1** brain-native / no-hardcode | **PASS with the exception cited.** `0.5` stays fixed under documented exception (a) -- a fixed point on a NORMALIZED output; cosine is bounded in [-1,1] and dimensionless, matching the exception's own worked example. FR-009 requires this justification be recorded rather than implied. **Watch item:** the structural-inference half must not smuggle in its own magic threshold; any constant it needs is a P1 event in its own right and must be declared or derived. |
| **P2** adopt-better or merge | **AT RISK -- see Complexity Tracking.** `substrate/eprop_transport_gate.py` already exists and already scans the tree on e-prop concerns. A second gate over the same invariant is the additive shape P2 forbids. Resolution is a Phase 0/1 deliverable: either extend the incumbent, or record a structural argument for why classification and transport-detection are genuinely different jobs. Not deferrable. |
| **P3** no dormancy | **PASS with an obligation.** Any new governed module must be registered in `__init__.py` REGIONS and `run_selfchecks.py`, and be reachable from a live path or carry an EXEMPT reason. Where the new checker lives (`substrate/` vs `tools/`) changes which obligation applies; that is Phase 0 research. |
| **P4** one brain | **PASS.** No `Brain` is constructed anywhere in this feature; no stack class is re-implemented. |
| **P4b** the durable lake is the brain | **PASS, and worth stating rather than assuming.** The ratchet baseline is gate metadata, not something the brain learned, so it does not belong in `brain_lake.bin` and writing it beside the repo is not a "second durable store". Precedent: `p4c_ungoverned_baseline.json`. If any part of this feature ever persisted something the brain LEARNED, P4b would bind and the answer would be the lake. |
| **P5** GPU where measured | **PASS.** The numeric half is CPU-reproducible by requirement; the static half needs no accelerator. The suite stays runnable without a GPU. |
| **P6** faculty-to-faculty | **NOT ENGAGED.** A gate is not a faculty; it consumes no faculty's output and changes no faculty's behaviour. Recorded explicitly so a later reader does not read the silence as an oversight. |
| **P7** any dependency accepted | **PASS.** No dependency proposed. Stated per the constitution's instruction to say what a dependency would BUY rather than certifying its absence: nothing here is blocked by a missing library. |
| **P8** invent the method | **PASS, and it is the binding principle.** The checker is itself an invented instrument and owes what P8 demands: a REACHABLE RED (a constructed input that makes each of its checks fail), a control that is not the method itself, and deletion on a null. FR-004 is P8 clause 1 applied reflexively -- an instrument that cannot fail has falsified itself. |

**Standing Invariants:**

| # | Invariant | Assessment |
|---|---|---|
| 1 | Count a gate's output before any rule leans on it | **This is the feature's spine.** FR-007 and SC-005 make the ordering verifiable from commit order. |
| 2 | Never silence a check to make it pass | The ratchet is the risk surface. FR-018 answers it: the baseline is machine-written and the writer REFUSES a rise. |
| 3 | The negative verdict is the dangerous one | FR-004 is precisely this -- where the criterion cannot discriminate, the rule is reported UNJUDGED, never failed. |
| 4 | Absence of a warning is not evidence of absence | FR-006: an empty census fails loudly. |
| 5 | Verify the verifier | The new checker needs a self-check that constructs its own failures, not one that runs the happy path. Non-negotiable deliverable. |
| 6 | Runtime evidence outranks static reading | **TENSION, stated rather than smoothed.** The CI half is deliberately STATIC. The justification is that a *claim* is a static property of the code -- it is what the rule asserts about itself -- whereas whether the claim HOLDS is a runtime question, and that is precisely what the on-demand numeric half measures. The split honours the invariant rather than evading it: no static reading is ever allowed to conclude that a rule's gradient claim is TRUE. |
| 7 | Safety envelope untouched | FR-010. No `brainstem/` file is read for classification or edited. |
| 8 | E-prop, not backprop | **This feature amends the STATEMENT of invariant 8, not the prohibition.** The ban on backprop, BPTT and weight transport is unchanged and explicitly out of scope. FR-008a requires both surfaces carrying this invariant to be amended together, since the constitution restates it verbatim and Governance makes the constitution "the defect" where the two disagree. |

**Gate result: PASS with one AT-RISK item (P2), tracked below and required to be resolved by Phase 1
rather than carried into implementation.**

## Project Structure

### Documentation (this feature)

```text
specs/020-eprop-cosine-scope/
├── spec.md              # /speckit-specify + /speckit-clarify output
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (CLI contracts for the two checkers)
├── checklists/
│   └── requirements.md  # spec quality checklist
└── tasks.md             # /speckit-tasks output — NOT created here
```

### Source Code (repository root)

```text
src/snn_data_lake/
├── substrate/
│   ├── eprop_transport_gate.py     # INCUMBENT — P2 adjudication target, may absorb the new work
│   ├── dormancy_gate.py            # AST-walk template: classifies without importing
│   ├── method_dormancy_gate.py     # per-METHOD AST precedent
│   └── <new classifier>            # IF a new module is justified over extending the incumbent
├── cortex/
│   ├── eprop_kernel.py             # the reference gradient — reconciliation target (US3)
│   ├── eprop_shd.py                # shipped encoder
│   └── eprop_nmnist.py             # shipped encoder
└── <learning-rule modules>         # gain an explicit gradient-claim declaration where they claim one

tools/
├── p4c_ratchet.py                  # machine-written baseline + refuse-a-rise precedent
├── p6_gate.py                      # ANTI-pattern: source-dict baseline, raisable by one-line edit
└── <on-demand instrument>          # cosine + discriminating-power + null-separation

tests/
└── test_<checker>.py               # constructed reds per rule, per Standing Invariant 5
```

**Structure Decision**: single project, matching the existing tree. The two halves land in the two
homes this repo already distinguishes -- asserting gates live in `src/snn_data_lake/substrate/`
(where `dormancy_gate` and `eprop_transport_gate` are), on-demand instruments live in `tools/` (where
`charge_probe` and the ratchets are). Whether the static half is a NEW substrate module or an
extension of `eprop_transport_gate` is the open P2 question below; the directory is settled either
way.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| **P2 — a second gate over the e-prop invariant** (`eprop_transport_gate` already scans the tree on e-prop concerns) | UNRESOLVED at Phase 0 entry. The two jobs may be genuinely distinct: the incumbent detects weight TRANSPORT (a forbidden mechanism), while this feature CLASSIFIES which rules owe a cosine (a scope question) and measures whether the criterion discriminates. Different inputs, different verdicts, different failure modes. | **Not yet rejected — that is the point.** Extending the incumbent is the P2-preferred route and has not been ruled out. Phase 0 research must report whether the incumbent's walk and predicate can carry the classification, and Phase 1 must either adopt that or record the structural argument for separation. Shipping a second module without that adjudication would be the additive failure P2 names. |

## Open questions — RESOLVED by Phase 0 (see [research.md](./research.md))

1. **Statically determinable?** SPLIT. Parsing is sound (378/378, 0 failures) so coverage is not the
   risk; but structural INFERENCE measured **~31% error both directions** and cannot adjudicate.
   Resolved by making the declaration authoritative and demoting inference to a CLAIM-or-UNKNOWN
   recall net, plus a first-class `UNDETERMINABLE` state. Spec amended (FR-002a–d, FR-016/016a).
2. **Does `eprop_transport_gate` absorb this work?** **STILL OPEN** — see Complexity Tracking. Phase 0
   settled the HOME (`substrate/`, registered) and the SHAPE (one module, `--flag` for the on-demand
   half), but did not adjudicate extend-vs-new. Must be decided before tasks are written.
3. **What does the reconciliation MOVE?** Measured: no verdict moves — but that is a statement about
   SLACK (healthy cells sit 0.26–0.50 above the bar), not about the change being unimportant. In the
   truncation regime the forms DO separate verdicts (0.7376 GREEN vs 0.4105 RED). **The spec's premise
   for US3 was also measurably wrong and has been corrected in place** (research.md R0).
4. **Is `0.5` well-placed?** Depends entirely on the null, which the spec had left ambiguous —
   permutation and shuffled-label give OPPOSITE verdicts. Null now named exactly (FR-009a).
5. **Wording budget?** **541 words** of headroom; real policy blocks in that file run 548/623/747
   words. The amendment must land near **250 words** or raise the ceiling in the same commit.

## Constitution Re-Check (post-design)

| Item | Post-design assessment |
|---|---|
| **P2** | **STILL AT RISK, unchanged.** Phase 0 removed the directory question but not the adjudication. Carried into Complexity Tracking as a blocking item for `/speckit-tasks`. |
| **P1** | **HOLDS, and the risk named pre-design materialised.** The inference half would have needed its own tuned thresholds; demoting it to a recall net removes that surface rather than tuning it. `0.5` stays fixed under exception (a), now with a measured placement to record. |
| **P3** | **HOLDS with a measured cost of zero.** A declaration constant moves none of the three P3 gates (+0/+0/+0), each verified against a positive control that DID move. |
| **P4c** | **NEW CONSTRAINT, decisive.** `tools/` is at zero ratchet headroom (30/30), so a tool-tier file reddens CI on its own commit. The Structure Decision is now forced, not preferred. |
| **P8** | **HOLDS and is sharpened.** `INSTRUMENT_FALSIFIED` and the constructed-red self-check are P8 clauses 1–2 applied to the checker itself. |
| **Invariant 3** | **Was violated by FR-016 as originally written**; found by measurement, fixed by defining "classified" to include `UNDETERMINABLE`. |
| **Invariant 5** | **Sharpened by evidence**: an existing gate launders a regression past both `--assert` and `--self-check`. FR-018 is load-bearing, not polish. |

**Gate result post-design: PASS, with P2 unresolved and blocking `/speckit-tasks`.**

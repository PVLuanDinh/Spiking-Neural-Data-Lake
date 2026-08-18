# Specification Quality Checklist: Scope the e-prop cosine clause

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-17
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- Items marked incomplete require spec updates before `/speckit-clarify` or `/speckit-plan`

### Validation round 1 — 2026-08-17

**One item fails: FR-011 carries the single remaining [NEEDS CLARIFICATION] marker.** It is a genuine
scope fork with materially different effort, not a detail with a safe default, which is why it was
left marked rather than guessed:

- Reconciling the reference gradient with the shipped encoders' factorization is a code change to the
  one instrument the e-prop invariant depends on. Doing it inside this feature makes the feature
  larger and touches the criterion that certifies other work.
- Disclosing the divergence and tracking reconciliation separately keeps this feature to scoping +
  measurement, at the cost of shipping a scoped bar whose instrument is known-suspect for a while.

Both are defensible. The choice changes what "done" means for User Story 3, so it is an owner
decision and is being asked rather than assumed.

### Validation round 2 — 2026-08-17, RESOLVED

**Owner chose: reconcile inside this feature.** FR-011 now REQUIRES the reference gradient to match
the shipped encoders' factorization; a disclosure alone is no longer sufficient. Consequences folded
into the spec rather than left implicit:

- FR-012 added — the reconciliation must be shown by MEASUREMENT (perturb a shipped rule, the
  criterion must move), because two factorizations can look equivalent on inspection and score
  differently.
- User Story 3 re-framed from optional-or-disclose to required-for-done, with its P3 position
  explicitly labelled a BUILD-ORDER position rather than an optionality signal, and a note not to
  reorder it ahead of Story 2 — Story 2's check is the detector expected to catch this mismatch, and
  fixing the instrument while holding the detector beats editing it blind.
- SC-008 added — reconciling may MOVE the measured cosine for a shipped encoder. If a rule that
  cleared the bar now falls below it, that is a real finding to report, not a reason to abandon the
  reconciliation. The spec deliberately does not predict the direction.

All checklist items now pass. Spec is ready for `/speckit-plan`.

### Validation round 3 — 2026-08-17, after `/speckit-clarify`

Four clarifications asked and integrated; all 16 items re-checked against the updated spec and all
16 still pass (16/16 → 16/16, no regressions). Requirements grew 12 → 23, success criteria 7 → 13.

What the clarification session changed, beyond recording answers:

- **Claim detection** is now dual-signal (declaration AND structural inference, cross-checked) with
  disagreement as a reportable third state — chosen because a missing marker and a genuine
  no-claim are otherwise indistinguishable, which is the silent under-count FR-006 exists to stop.
- **Run location** split by cost: the static census asserts in CI, the numeric cosine work is an
  on-demand instrument. Driven by the project's own recorded hazard that CI and the local gate run
  different torch builds, so a build-sensitive numeric assert can be green in one and red in the
  other. This also forced FR-013's requirement that the CI half must NOT need the module to import.
- **Enforcement** split: absolute on "zero unclassified", ratchet on disagreements — plus FR-018,
  which requires the baseline be machine-written and refuse a rise, because one existing gate in
  this tree keeps its baseline as a source-level dict that can be raised by a one-line edit.
- **The bar is explicitly frozen** for this feature (FR-009b). Reconciling the instrument may already
  move measured cosines; moving the threshold at the same time would leave any flipped verdict with
  two candidate causes and no way to attribute it.

**One gap found by reading the constitution, not by a question**: Standing Invariant 8 restates the
`cosine > 0.5` clause verbatim, so the amendment has TWO surfaces. FR-008a now requires both to move
together — a one-surface amendment would create exactly the divergence the constitution's Governance
section names, where the constitution becomes "the defect".

**Note on two items that were reviewed carefully rather than ticked by default:**

- *"No implementation details"* — the spec names specific modules (`cortex/eprop_kernel`,
  `cortex/eprop_shd`, `basal_ganglia/mlp_q`). These appear only as EVIDENCE for measured findings, not
  as prescriptions of how to build. Marked pass on that basis; a planner is free to choose a different
  structure entirely.
- *"Written for non-technical stakeholders"* — this is a governance feature about an admission
  criterion, so some domain vocabulary (gradient, cosine, local learning rule) is irreducible. The
  WHY of each story is stated in plain language and the Success Criteria avoid jargon. Marked pass
  with that qualification recorded here rather than silently.

### Validation round 4 — 2026-08-17, second `/speckit-clarify` (run POST-plan and POST-tasks)

Five clarifications asked and integrated; all 16 items re-checked and all 16 still pass
(**16/16 → 16/16, no regressions, no newly-passing items**). Requirements grew 30 → 35, success
criteria 15 → 19.

Unusual and worth stating: this round ran AFTER `/speckit-plan` and `/speckit-tasks`, not before.
Three of the five findings were only visible because plan.md and tasks.md existed — a scope
contradiction, a requirement cited by zero tasks, and two properties stated without a mechanism.

- **Census scope was contradictory** and is now split (FR-001a): the census walks the WHOLE tree, the
  bounds assert on the GOVERNED tier only, ungoverned is reported. FR-001 had said "the governed
  tree" while FR-013, SC-010 and the ungoverned assumption all required otherwise. Measured stake:
  `reasoning/` alone is ~108 modules against ~22 governed local rules.
- **FR-010 contradicted FR-011** — a blanket "change no learning rule" against a required edit to the
  reference gradient — and was cited by ZERO tasks. FR-011's target is now the SINGLE named
  exception; FR-010a requires the prohibition be VERIFIED from the diff with a positive control,
  which is the only thing separating "we did not intend to touch the safety envelope" from "we did
  not touch it".
- **FR-015 stated a property with no mechanism.** Staleness is now a content digest of the measured
  inputs (FR-015), owing a reachable red (FR-015a). A commit-SHA comparison and a timestamp threshold
  were both rejected with reasons recorded.
- **The CI cost was unquantified** ("cannot be expensive"). FR-013a now requires it MEASURED and
  PRINTED and explicitly NOT asserted — wall-clock on shared runners is not deterministic, and a
  fixed second-count would be a P1 magic constant.
- **A ratchet at its floor was indistinguishable from one that never ran.** FR-017a requires SATISFIED
  vs GREEN reported distinctly, for both ratchets. Retiring the ratchets for an absolute `== 0` was
  offered and DECLINED by the owner: it would redden CI on any new undeclared rule, tightening the
  gate against unrelated work rather than describing the state honestly.

Two items re-examined rather than ticked by default, both holding on round 3's recorded basis:
*"no implementation details"* — the new FR-001a and FR-010 name `substrate/eprop_transport_gate` and
`cortex/eprop_kernel.py::_TinyNet.eprop_grads`, but as the SCOPE of a prohibition's exception and as
the precedent for a verdict contract, not as build prescriptions. *"success criteria are
measurable"* — SC-004's "under one minute without reading source" remains human-timed and is still
the weakest criterion here; it has no verifying task and is recorded as a known gap rather than
tightened, because changing it was outside what any of the five questions asked.

**Note on this file's own convention.** The skill instructs that only changed checkbox markers be
edited, to keep diffs quiet. This round changed no marker, so strictly nothing needed writing — but
rounds 1–3 are all recorded here, and a fourth round that left no trace would make the checklist's own
history misleading. Recorded deliberately, and the reasoning stated so the departure is visible.

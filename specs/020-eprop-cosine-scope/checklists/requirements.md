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

**Note on two items that were reviewed carefully rather than ticked by default:**

- *"No implementation details"* — the spec names specific modules (`cortex/eprop_kernel`,
  `cortex/eprop_shd`, `basal_ganglia/mlp_q`). These appear only as EVIDENCE for measured findings, not
  as prescriptions of how to build. Marked pass on that basis; a planner is free to choose a different
  structure entirely.
- *"Written for non-technical stakeholders"* — this is a governance feature about an admission
  criterion, so some domain vocabulary (gradient, cosine, local learning rule) is irreducible. The
  WHY of each story is stated in plain language and the Success Criteria avoid jargon. Marked pass
  with that qualification recorded here rather than silently.

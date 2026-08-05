# Specification Quality Checklist: Policy Compliance (P0-P8)

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-04
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

## Feature-specific gate (P8 applied to this spec)

The subject matter of this feature is criteria that cannot fail, so the spec is held to its own
standard:

- [x] **Every success criterion states its RED** — the input under which it reports FAIL. All of
      SC-001 through SC-011 carry an explicit RED clause.
- [x] **Criteria are not of the form "the audit passes"** — each names an observable outcome and a
      falsifying input, not the completion of a process.
- [x] **Criteria that fail TODAY are marked as such**, so the starting state is recorded and the
      change is demonstrable rather than asserted. Seven of eleven are marked failing today.
- [x] **No criterion is satisfied by the act of writing this spec.**

## Validation notes

**Iteration 1 findings, resolved before this checklist was marked:**

1. *Implementation detail leakage.* First draft named `consciousness_probe.py`, `pyproject.toml`,
   `scikit-learn` and specific line numbers throughout the scenarios and criteria. Rewritten to
   describe observable outcomes ("a package the project never declares", "a headline value that
   cannot vary"). The concrete file-and-line evidence is planning input and belongs in `plan.md`
   and `tasks.md`, not here.
2. *A criterion that could not fail.* An early SC read "the full-tree audit completes against all
   nine policies" — process completion, not an outcome, and unfalsifiable in exactly the way this
   feature exists to remove. Deleted rather than reworded, per F11.
3. *Missing edge case.* The case where the honest remedy is to correct a CLAIM rather than change
   code was absent; without it, a reader could conclude every finding implies a code change, which
   would invert the defect. Added.
4. *Unbounded scope.* "All files" was clarified to the tracked tree, with the safety envelope
   explicitly report-only.

**Deliberately accepted:**

- **User Stories 1 and 2 are both P1.** The template implies a single top priority, but these are
  independently shippable and neither dominates: Story 1 is the only finding that reaches a human,
  Story 2 is the only one that breaks a clean install. Ranking them against each other would be
  arbitrary, and both are genuinely MVP-viable alone.
- **SC-008 uses "measurably slow" rather than a fixed threshold.** A hard number would be invented
  rather than measured; the baseline is established during planning and the criterion is the
  comparison, not a constant. Recorded here so the vagueness is deliberate and visible.

## Notes

- Items marked incomplete require spec updates before `/speckit-clarify` or `/speckit-plan`.
- No [NEEDS CLARIFICATION] markers were raised. The feature description supplied verified evidence
  and explicit scope, and the three areas that would normally need asking — scope boundary,
  deletion-vs-repair preference, and whether severing the self-report retreats from the sentience
  goal — were all answered directly in the input and are recorded under Assumptions.

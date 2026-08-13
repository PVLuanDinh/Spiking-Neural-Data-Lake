# Specification Quality Checklist: E-prop Kernel MERGE + Surrogate Perception Tier Retirement

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-12
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

Two template criteria are applied in the spirit rather than the letter, and the deviation is
recorded here rather than silently:

- **"No implementation details" / "technology-agnostic"**: this feature's SUBJECT is a source-tree
  structure — which definitions are duplicated, at which lines, and which gates assert on them.
  File paths and line numbers are the domain vocabulary here, not leaked implementation. They are
  retained deliberately; a version of this spec without them would be unverifiable.
- **"Written for non-technical stakeholders"**: the stakeholder is the project's own policy set
  (P2/P3/P4c and the e-prop invariant). The spec is written to be checkable against those policies.

Provenance flags:

- Claims WITHOUT a marker were re-derived independently by the main agent at HEAD `47555a74`.
- Claims marked **[F]** come from the measurement pass and were NOT independently re-run. Every
  **[F]** claim that gates a commit is restated as a pre-commit measurement in the plan, not
  trusted as-is.

**Amendment pass against `docs/POLICIES.md` (2026-08-12, on owner instruction "follow POLICIES.md").**
Reading the policy text rather than the CLAUDE.md summary added four requirements the original draft
had missed, one of which changes a headline ruling:

- **FR-018** — P2's honesty guard says "any unique capability of the retired mechanism ported
  first". The A/B does not discharge this: an A/B compares the shared job, not the unshared
  capability. 19 surrogate-only defs now need dispositions, and the DELETE ruling is conditional on
  that clearing. This is the material change.
- **FR-019** — P8 guard (1) requires a reachable RED for any new criterion. The kernel self-check
  needed one and did not have one specified.
- **FR-020** — the cross-cutting invariant on citation rot; this lane rewrites both cited files.
- **NG-005** — `reasoning/eprop.py` measured and ruled OUT of scope (different substrate, not a
  third copy), with its own P4c debt flagged rather than folded in.

Open items that the plan (not the spec) must resolve:

- The n=256 A/B is carried as sufficient-for-P2 but insufficient-for-an-accuracy-claim. If the plan
  wants a stronger statement it must schedule a larger run and budget a `tonic` install.
- FR-017's bit-exact pin needs a concrete harness; the spec states the requirement, the plan owes
  the mechanism.

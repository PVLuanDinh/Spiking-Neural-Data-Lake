# Specification Quality Checklist: Resolve the e-prop Weight Transport at `mlp_q`

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-14
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

**Two template criteria are applied in spirit, and the deviation is recorded rather than hidden** —
the same deviation specs/013 and specs/014 recorded, for the same reason:

- **"No implementation details"** — this feature's SUBJECT is a credit-assignment mechanism and the
  criterion that judges it. Naming `cos(W2, B)` is not leaked implementation; it is the defect. A
  version of FR-001 that did not name the quantity would be unverifiable and would omit the single
  thing this feature exists to fix. Note what the spec still avoids: it does not name a target
  implementation, a data structure, or an API, and FR-008 is written as "exactly one reachable path"
  rather than as a design.
- **"Written for non-technical stakeholders"** — the stakeholder is the project's own policy set
  (the e-prop cross-cutting invariant, P1, P2, P8). The spec is written to be checkable against it.

**Deliberately pre-committed, because these are the decisions that go soft under a result:**

- FR-003 (prediction written before the measurement) and FR-007 (deletion on a null) are both
  recorded now, before any number exists. R-002 states why.
- FR-002 requires the criterion be shown to separate good from bad using a reference chosen
  INDEPENDENTLY of the candidate rule, so the repair cannot be tuned toward the wanted answer
  (R-001).
- SC-004 accepts "it could not distinguish them" as a complete result.

**Verified before writing, not assumed:**

- Both owner-supplied citations were retrieved and checked. arXiv 2403.08804 resolves to the claimed
  SFDFA paper; the bioRxiv preprint resolves to the claimed Predictive E-prop paper.
- **One correction was applied and is load-bearing**: SFDFA estimates its feedback weights rather
  than fixing them randomly, which is why it can satisfy the project's existing invariant. FR-017
  requires that correction survive wherever the mechanism is described.
- **One limit was found and is recorded**: the Predictive E-prop preprint does not address weight
  transport at all (0 full-text hits for transport/symmetry/alignment terms against 30 for BPTT), so
  it cannot resolve `predictive_coding.py`. FR-012 forbids resting that adjudication on its silence.

**Open for `/speckit-plan`, not for the spec:**

- HOW the criterion is repaired — whether the existing cosine is rescoped or replaced. The spec
  requires only that it discriminate (FR-002).
- HOW converging feedback is realised in a stdlib hand-written module, and whether it is feasible
  there at all. The spec makes infeasibility a reportable finding rather than a failure.
- The order of the commits, and where the `predictive_coding` adjudication lands relative to the
  `mlp_q` work.

# Specification Quality Checklist: Order neurogenesis by cortical architecture, from a prior that must be beatable

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-06
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

## Project-specific gates (this repo deletes criteria that cannot fail)

- [x] **Every success criterion states the input that makes it FAIL.** All eleven do; SC-005 states
      two, one per direction.
- [x] **The headline criterion can fail in both directions.** SC-005 fails on a pinned seed (drift
      zero) and on a harmful prior (underperforms the random control). A criterion that can only fail
      one way would not be measuring what this feature claims.
- [x] **No criterion measures the defect and scores it as the feature.** Checked explicitly against
      the five instances this project has already caught. The nearest risk here is SC-006: "placement
      changes behaviour" could be satisfied by placement making things *worse*, so SC-005's control
      comparison is what keeps it honest, and the two are deliberately separate criteria.
- [x] **No criterion can pass vacuously.** SC-004 additionally fails if *no* magnitude changed, so it
      cannot pass on a brain that never learned. SC-007's red confirms it measures the input port
      rather than something incidental.
- [x] **The silent-failure mode has its own criterion and its own run red.** SC-002 exists solely
      because a length-unit mismatch assigns every unit to the catch-all layer and produces a result
      that looks valid.
- [x] **Non-additive.** FR-016 and SC-010 require exactly one layered-population mechanism to survive.
- [x] **Relationship to the adjacent feature is stated.** 007 owns which regions grow; 011 owns how
      what grows is ordered. The spec states that needing a second growth mechanism means 011 is wrong.
- [x] **Externally sourced constants are replaceable in one edit** (FR-006) and never presented as
      measurements of this brain (FR-007).
- [x] **What the source cannot support is refused rather than approximated.** FR-019 excludes delays
      outright; the Assumptions section states why as a result rather than as a gap.

## Notes

**Validation run 1 — three issues found and fixed before this checklist was marked complete:**

1. *SC-005 originally read "drift is non-zero", which is satisfiable by a brain that drifts randomly
   and performs worse.* Added the control comparison and the second red. This was the single most
   important fix in the pass — without it the headline criterion rewards movement rather than
   improvement.
2. *SC-004 could pass vacuously on a brain that never learned* (no sign changed because nothing
   changed). Added the requirement that magnitudes must have changed.
3. *"The prior" was used loosely across FR-006 to FR-010 to mean both the source data and the seeded
   values.* Separated in Key Entities: **Prior** is the external statistics; **Drift record** is the
   per-statistic comparison.

**Deliberate departures from the template, each justified:**

- Two sections precede the mandatory ones (*Why this feature exists*, *The one constraint everything
  else follows from*). The central constraint is not a requirement but the premise every requirement
  depends on, and burying it under FR-006 would lose it.
- Three user stories share P1. They are genuinely co-equal: placement without sign is anatomy without
  biology, sign without the drift measurement is hardcoding with a citation, and the drift measurement
  without the other two has nothing to measure. Each remains independently testable, which is the
  template's actual requirement.

**Carried into planning, not resolved here** (design decisions, correctly deferred):

- The mapping between the source's seven regions and the circuit's population scheme. The spec
  requires it be recorded as a decision (Assumptions); the plan chooses it.
- The placement rule for units grown beyond the measured volume. The spec requires an explicit stated
  rule and a reported count (FR-005); the plan chooses the rule.
- What the prior means once the grown population exceeds the measured one (Edge Cases). This is the
  sharpest open design question in the feature and it belongs to the plan.

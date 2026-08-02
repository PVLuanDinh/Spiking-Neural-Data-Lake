# Specification Quality Checklist: The Brain-Native Assembly-Calculus Language Organ

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-02
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
      *Qualified pass.* This spec names existing file paths, function names, and measured figures
      throughout. That is deliberate and matches `specs/003-grow-the-live-populations`, which does the
      same. In this repo a requirement that does not cite `file:line` is unfalsifiable, and the coverage
      rubric's evidence standard is explicit: *"Every score cites `file:line`. A score without a citation
      is not a score."* What the spec does NOT do is prescribe the organ's internal implementation - no
      class layout, no data structures, no algorithm listing. Those are in `plan.md` and
      `contracts/language_organ.md`.
- [x] Focused on user value and business needs
      The user is the operator; the value is a language faculty that is not at chance, and an answer that
      can come back "no".
- [ ] Written for non-technical stakeholders
      **Fails, deliberately and permanently.** The only stakeholder is an engineer operating a spiking
      brain. Rewriting the measured findings for a non-technical reader would remove the evidence that
      makes each requirement checkable. Recorded as an accepted deviation, not an oversight.
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
      Three genuine unknowns exist. Per the task's instruction they are recorded verbatim in
      `## Open questions` (Q1-Q3) with a stated assumption each, rather than left as inline markers that
      would block planning.
- [x] Requirements are testable and unambiguous
      Each FR names an observable. FR-005 (null is a valid end state), FR-010 (deletion, not flag-gating)
      and FR-014 (ties count as wrong) were each rewritten once to remove a hedge that would have made
      them unfalsifiable.
- [x] Success criteria are measurable
      SC-001 through SC-013 each name a number, a file, or an exit code.
- [ ] Success criteria are technology-agnostic (no implementation details)
      **Fails by design, same reason as above.** SC-006 names an exit code and SC-007 names a file that
      must not exist. Both are the only unambiguous forms of those criteria in this repo.
- [x] All acceptance scenarios are defined
      Five user stories, 21 acceptance scenarios.
- [x] Edge cases are identified
      Twelve, including the two that can end the feature: lexicon coverage, and no derivable comparable
      score.
- [x] Scope is clearly bounded
      `## Out of Scope` names six exclusions, including the 862 MB dead model file that is explicitly
      NOT this feature's to remove.
- [x] Dependencies and assumptions identified
      Ten assumptions, eight dependencies, and the concurrent-agent / moving-HEAD hazard stated in the
      header.

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
      US1 instrument honesty, US2 baseline, US3 acquisition, US4 verdict, US5 retire.
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] The four honesty requirements from the directive are first-class
      Baseline before implementation: FR-003, SC-002, US2. Metric admission control: FR-002, SC-001, US1.
      Null/control arm: FR-004, SC-004, US3 scenarios 2-3. Null is a valid end state: FR-005, SC-008,
      US4 scenario 2. `N of M`: FR-019.
- [x] The P2 obligation is unambiguous
      FR-010 requires deletion and names flag-gating as non-compliance; US5's independent test is a
      survivor count.

## Notes

- Two items are marked failing and both are accepted deviations with a stated reason rather than debt.
  `specs/003-grow-the-live-populations` carries the same two and for the same reason.
- Q1 is the question most likely to end the feature. It is scheduled into Phase 0 of `plan.md` and is a
  hard gate before any organ code is written, so the failure mode "discover mid-implementation that the
  benchmark cannot express the organ" is closed.
- No item required a spec rewrite loop. One pass, then the three hedge removals noted above.

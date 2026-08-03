# Specification Quality Checklist: Whole-brain neurogenesis -- every region a producer AND a consumer

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-03
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

## Validation notes -- what was fixed to reach PASS

Three iterations were needed. The failures are recorded because they are the same failures the
feature itself is about.

**Iteration 1 -- success criteria were unfalsifiable.** SC-002 originally read "most regions both
grow and are consumed". "Most" is not measurable and gives the implementer the denominator. Fixed by
stating the measured baseline (2 of 17 defining regions) and requiring the denominator be reported.

**Iteration 2 -- an implementation detail leaked into a requirement.** An early FR named a specific
shared base class as the mechanism every region should implement. That is a plan decision, not a
requirement: it presumes inheritance is the right shape before anyone has compared it against a
registry or a protocol. Rewritten as FR-006 ("each region MUST be able to grow its own live
population"), which states the outcome and leaves the shape to the plan.

**Iteration 3 -- the acceptance test could be passed without doing the work.** FR-001 originally
said growth must "change the brain's behaviour", which the existing tautological test already claims
to satisfy: it changes the growth counters, and those are part of the brain's state. Fixed by adding
FR-002, which names the forbidden observables outright. This is the single most important line in
the document, because the worked example of this exact failure is what motivated the feature.

## Deliberate deviations from the template

- The spec opens with a section explaining why the obvious reading of the request is wrong, before
  the user stories. The template does not have this section. It is included because the literal
  request ("add growth to every region") would produce a measurably worse brain, and a spec that
  buries that reasoning in Assumptions invites the wrong build.
- Success criteria cite measured baselines rather than target values alone. The project constitution
  makes an unmeasured capability claim a defect, so a criterion with no baseline cannot be scored.

## Notes

- Items marked incomplete require spec updates before `/speckit-clarify` or `/speckit-plan`.
- All items pass. The spec is ready for `/speckit-plan`.
- One judgement call is worth a second opinion at plan time rather than a clarification now: whether
  "every part of the brain" should include the 2 regions that contain no modules at all. This spec
  excludes them as vacuous. If the intent is that those regions should be POPULATED as part of this
  work, the scope is substantially larger and the plan must say so.

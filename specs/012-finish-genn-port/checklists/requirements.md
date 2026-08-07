# Specification Quality Checklist: Finish the GeNN Port of the Burst+x_tar MNIST Learner

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-07
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs) — GeNN/pygenn/CUDA named only as
      the feature's SUBJECT (the port target and its measured environment), not as design choices
      this spec makes; no code structure, no API shapes prescribed
- [x] Focused on user value and business needs — framed on what the project (the owner mandate,
      the policies) needs and why
- [x] Written for non-technical stakeholders — as far as the domain allows; each requirement
      states WHY in policy terms
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain — the ledger row + owner mandate resolve scope;
      remaining choices (target region) are explicitly deferred to plan with a named default
- [x] Requirements are testable and unambiguous — each FR names its measurement
- [x] Success criteria are measurable — accuracy bands, control bars, gate exit codes, recorded
      wall-clock
- [x] Success criteria are technology-agnostic — stated as outcomes (verified toolchain, parity,
      control, determinism, records); the named tools are the feature's subject matter
- [x] All acceptance scenarios are defined — Given/When/Then per story, including the STOP path
- [x] Edge cases are identified — ABI/wheel risk, CUDA-version risk, compiler discovery, missing
      theta expressiveness, numeric drift, CI-no-GPU, the P2 sibling question
- [x] Scope is clearly bounded — pre-flight STOP semantics; CPU reference explicitly out of
      scope; sibling GeNN modules explicitly out of scope
- [x] Dependencies and assumptions identified — Assumptions section, 7 entries

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows — pre-flight, port+parity, payoff, governance
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- The domain is inherently technical: GeNN, CUDA and the gate env are the feature's subject, so
  they appear by name. The spec still prescribes no code structure, API, or file layout; the one
  structural choice (target region) is deferred to plan with `plasticity/` as the named default.
- FR-003's 3 pp parity band and FR-004's 20% chance bar are justified in Assumptions (ablation
  distance 7.2 pp; untrained-drift allowance) rather than left as bare numbers.

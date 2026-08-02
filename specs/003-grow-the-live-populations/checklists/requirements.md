# Specification Quality Checklist: Grow the Live Populations (Autonomous LIF Growth)

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-07-31
**Revised**: 2026-07-31, after adversarial verification (see Note 5)
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs) — see Note 1, deliberate deviation
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders — see Note 1
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain — Q1 resolved by owner 2026-07-31
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
- [x] No implementation details leak into specification — see Note 1

## Notes

**Note 1 — the "no implementation details / non-technical stakeholders" items are deliberately deviated
from, under constitutional authority.** The evidence discipline requires that "every score cites
`file:line`; a score without a citation is not a score", and P0's honesty guard makes an unmeasured
claim fake. Problem and Findings therefore cite source locations as *evidence the problem exists*,
exactly as `specs/002-capability-and-bounding` does — the house style, which wins over the stock item.

The distinction held to: **no FR and no SC names a mechanism, data structure, function, or call site.**
Every FR states a property that must hold; how it is achieved is left to `/speckit-plan`.

**Note 2 — five of the nine success criteria can be satisfied by a negative result**, by design. SC-005
accepts a recorded null; SC-006 accepts "documented as not-yet-persistable". Intentional under P0's guard
that a capability reported without measurement is fake; a spec whose only success is a positive outcome
would pressure the measurement.

**Note 3 — two dependencies do not exist yet, and one existing check is worse than absent.** FR-002
requires held-out evidence; the nearest existing measurement
(`test_learn_interval_reduces_reproduction_error`, `tests/test_time_cells.py:32–43`) measures
`|threshold/drive − target|` — a parameter distance, not a timing error — and never ticks the clock. A
spec treating it as the earn-your-place instrument would manufacture a positive result.

**Note 4 — the one scope fork was resolved by the owner, not defaulted.** Q1 asked how slice 2
establishes a population-dependent consumed quantity. Decided 2026-07-31: **repair `decode()`**, rather
than substituting `sd` or measuring only the saturated regime. That makes slice 2 a defect fix owed
regardless of growth, and the first slice of this feature that changes live behaviour — so it carries its
own before/after evidence and its own risk, which `/speckit-plan` must account for.

**Note 5 — the first draft of this spec was materially wrong, and adversarial verification caught it.**
A 37-agent verification pass over the same source produced three corrections that changed the spec's
shape:

1. **The blocking finding was missing entirely.** Below saturation `decode()` collapses to
   `estimate = base + (v − base) = v`, so the threshold vector is causally inert on `_elapsed_time`, the
   only quantity the brain consumes. Verified directly: max `|estimate − v|` over 63 unsaturated ticks is
   **0.0**; perturbing `thresholds[30]` by `+0.4` leaves the estimate sequence bit-identical; a grown and
   an ungrown clock are bit-identical for 60 unsaturated ticks and differ only past saturation
   (`64.0` clamp → `68.367`). On the live brain 92.5–95.8% of clock-advancing ticks are unsaturated. The
   original slice 2 was therefore unbuildable — its A/B would have returned exactly 0.0 by algebra. This
   became F9 and a new user story.
2. **A headline number was refuted.** The draft claimed a naive regrow displaces cell 32 by 0.206, "more
   than everything the population has ever learned (0.132)". At 600 live ticks the max learned
   displacement is **0.2472**, so the claim was false and getting worse with runtime. The argument now
   runs on the top cell (64.000 → 60.918, a shift of 3.082), which cannot be overtaken.
3. **Two invariants the draft assumed were already broken.** Ascending threshold order is not guaranteed
   today — the minimum inter-threshold gap is 0.0682 while a single live update reached 0.1811 (F10) —
   and nothing in the brain can observe population width at all (F11). Both became requirements.

Two further corrections were folded in: the spec is governed by `docs/POLICIES.md`, not by
`.specify/memory/constitution.md`, which subordinates itself to it (`constitution.md:36–38`); and the
handoff's "tolerance is DERIVED" is inaccurate — what is derived is `l_max`, while `r`, `patience`,
`eps_tol` and others are plain literals (F5).

**Note 6 — one earlier self-correction, before the verification pass.** An initial reading concluded
`TimeCellClock` was never constructed in `src/`; that came from a truncated search result, not the code.
It is constructed at `cortex/brain.py:744`. The Standing Invariant "absence of a warning is not evidence
of absence" caught it.

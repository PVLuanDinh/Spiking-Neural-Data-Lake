# Specification Quality Checklist: MuJoCo SO-101 Becomes The Arm

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-01
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs) — see Note 1, deliberate deviation
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders — see Note 1
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

**Note 1 — the "no implementation details / non-technical stakeholders" items are deliberately
deviated from, under constitutional authority**, exactly as in `specs/003`. The constitution's evidence
discipline requires `file:line` citations, and P0's guard makes an unmeasured claim fake. Problem and
Findings therefore cite source locations as evidence the problem exists. The line held: **no FR and no
SC names a mechanism, library, function, or call site.** Every FR states a property; how it is achieved
is left to `/speckit-plan`. "MuJoCo" appears in the title and Problem as the owner's named decision,
never inside a requirement.

**Note 2 — every owner authorisation is quoted in the header rather than assumed.** This feature
modifies a module CLAUDE.md gates behind explicit approval, and puts a new component in the path of
every motor command. The authorisations are recorded so a later reader can see the approval was
obtained, not inferred.

**Note 3 — three requirements exist to preserve safety properties, not to add them.** FR-002 (E-STOP
latch), FR-003 (calibration no-bypass) and FR-004 (wire validation independent of the mirror) are each
a property the current system already has. Authorisation to reimplement them is not authorisation to
weaken them, and the spec treats any loss as a defect.

**Note 4 — the hardest problem is named rather than deferred.** F4: the existing arm reports collision
without preventing it, by design and asserted under test, while MuJoCo's solver prevents penetration by
construction. FR-008 forces this to be settled explicitly — either the contract is preserved and
demonstrated, or it changes and every consumer moves in the same change. It must not be discovered
during implementation.

**Note 5 — P2 eligibility is a precondition, and the spec sequences it first.** `docs/POLICIES.md`
requires a benchmark that exists before "better" can be claimed, and none does. US1 builds it. This also
means the feature's first deliverable answers the build notes' actual question — whether STS3215 servos
hold 500 g at 300 mm — which is the return on the risk being taken.

**Note 6 — one assumption is load-bearing and worth challenging in planning.** The spec assumes the
SO-101 MJCF's actuator properties faithfully represent real STS3215 servos, on the strength of the
±3.350 N·m force range matching the notes' ~30 kg·cm at 12 V. Every torque conclusion this feature
produces inherits that assumption. It is stated in Assumptions rather than buried.

**Note 7 — findings F1–F17 were verified before being written**, F1 and F14 by running the model on
this machine (load, step, offscreen render, a `qfrc_bias` torque readout, and 2000 steps in 0.038 s),
the rest against source with `file:line`. An adversarial pass produced the corrections that shaped the
spec: `ArmSim` is not a physics simulator at all, and the daemon's environment is not the environment
either physics backend was installed into.

**Note 8 — one finding in the first draft was wrong and is corrected in place.** F9 originally claimed
the cockpit "has no binary content type". It has no binary *route*, but `_send` already accepts bytes
with an arbitrary content type (`cockpit.py:137-144`), so an image path is a smaller change than the
draft implied. The claim came from a summary rather than from the send function itself.

**Note 9 — the alternative scope is recorded in Status, not silently dropped.** A recon pass
recommended a read-only mirror over full replacement, on safety and P2-eligibility grounds. The owner
chose full replacement with that trade-off in hand. Recording the rejected option keeps the decision
legible and names the fallback if slice 1's benchmark refuses the swap.

**Note 12 — a clarification session (2026-08-01) resolved all five open questions and GREW the
feature.** Recorded in the spec's Clarifications section. The consequential one: the planner is now a
**full six-joint replacement**, not a port and not retired debt — which moved motion planning from Out
of Scope into scope and added FR-021 plus two tasks. Also settled: reach is the maximum horizontal TCP
radius (0.4795 m), and the 300 mm build figure is the *shoulder lever* — the two were never the same
measurement, so SC-006 stopped asserting they were equal; the gate clamps against the range that binds
a position command, not the physical stop, so `clamped` means exactly "the command was altered"; the
spatial-novelty cell becomes 3-D because most of the new joints move the tool vertically; and the
unsatisfiable-command verdict must penalise action selection, not merely appear on a dashboard —
telemetry alone was explicitly rejected as not discharging P3.

One item is worth watching: FR-021 says the planner "plans by sampling candidate configurations",
which is closer to a mechanism than the other requirements get. It is kept because the requirement that
matters is the *exclusion* — exhaustive enumeration is infeasible at six joints — and stating only "must
scale" would have been the vague-adjective failure the checklist exists to catch.

**Note 11 — a cross-artifact analysis found ONE CRITICAL gap, and it was an inversion of the usual
kind.** The highest-severity work item in the whole feature — keeping the arm inside the set of files
the brain may not rewrite — existed in `plan.md` and `tasks.md` but **nowhere in this spec**. Tasks
without a requirement look harmless; they are not, because regenerating tasks from the spec alone would
have silently deleted the protection. Closed by F18 and **FR-020**. Also corrected: F6 said the command
vocabulary lives in three places when it lives in **eight** (the undercount was the finding); Out of
Scope committed to "the planner is migrated" while the plan permits retiring it as recorded debt; the
dependency facts were stale after mujoco landed in `.venv-cu128`; the `body` contract was fixed in
plan/data-model with no trace here; and retiring the `OBSTACLE` verdict was a contract change with no
requirement covering it (now in FR-008).

**Note 10 — three requirements were added after the first validation pass**, all from findings that
arrived late: FR-012a (the "sim is a toy" assertions must be rewritten, never removed — they are the
anti-overclaim mechanism and a real body needs them more), FR-012b (a missing render context must
degrade to "viewer unavailable" rather than kill the daemon), and a licence clause on FR-012.

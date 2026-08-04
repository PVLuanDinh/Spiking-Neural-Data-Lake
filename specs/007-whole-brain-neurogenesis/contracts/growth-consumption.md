# Contract: growth and consumption

**Feature**: specs/007-whole-brain-neurogenesis | **Phase**: 1 | **Date**: 2026-08-03

This is the interface contract a region must satisfy to be reported as complete. It is written as
obligations and prohibitions rather than signatures, because the plan has not chosen between a
protocol, a base class and a registry (research.md Decision 1) and choosing here would prejudge it.

---

## C1. Producer contract

A region that grows MUST:

1. **Derive its floor.** Provide a halting condition computed from its own measured dispersion. A
   declared constant is a P1 violation.
2. **Refuse before it grows.** `grow_if_earned`-shaped: evaluate the floor first and return without
   evaluating anything more expensive when it refuses.
3. **Count refusals.** A run in which growth occurred and `refused == 0` MUST NOT be reported as
   earning. It means the gate was never tested, not that it passed.
4. **Consume zero random numbers in the growth DECISION** (R1). `rng.getstate()` MUST be
   bit-identical across attempts, whether they grow or refuse. This is what decouples "did it grow"
   from the noise stream and is measured today.
5. **Grow only at a state reached from its own seeded dynamics** (R2) -- never on a wall clock, tick
   count, or external schedule.
6. **Preserve function.** Existing outputs MUST be unchanged at the moment of widening.
7. **Refuse on the safety envelope.** Any seam matching arm / dof / joint / motor / actuator /
   effector / estop must refuse unconditionally.

   **CORRECTED 2026-08-03 -- this is a REQUIREMENT, not a description of the tree.** As written it
   implied the guard is enforced today. It is not. `is_safety_seam` is called from exactly 1 site
   (`growable.py:139`, inside `register_seam`), and nothing in the tree ever calls `register_seam`,
   so the check is structurally unreachable. It is also a substring match on the name, which makes
   the same object safety-critical or not depending on the label passed:
   `is_safety_seam('arm.learner') = True` but `is_safety_seam('autonomous_loop.learner') = False`.
   Any mechanism admitted under this contract must ENFORCE the refusal at the growth call site,
   and must be verified by attempting a real growth on a safety-named seam and observing the
   refusal -- never by calling the predicate directly, which returns True from a function nobody
   invokes. (Arm safety itself is unaffected: it comes from `ArmSim.apply()`'s gate and the reflex
   E-STOP, both live and tested.)

A region that cannot satisfy C1.1 is `admitted = False` and does not grow. **That is a passing
outcome**, recorded as "no derivable floor", not a failure to be worked around.

---

## C2. Consumer contract

A consumer that counts toward this feature MUST:

1. **Live in a different module from the producer.** Cortex-internal pairs are disqualified even
   across files, because `REGIONS["cortex"]` holds `time_cells`, `brain` and `executive_control`.
2. **Branch on the value**, not merely sum it. A blind add into a common-mode scalar --
   `autonomous_loop.py:696`'s `[td[i] + ext[i] for i in ...]` -- is transport, not consumption.
3. **Preserve resolution.** MUST NOT quantise away the refinement it is being handed.
   `brain.py:1603`'s `max(1, int(round(el)))` destroys 100% of the timing refinement
   (`_learned_latency` differs on 0 of 300 ticks). Any consumer that rounds its input to a coarser
   grid than the producer's improvement is not a consumer.
4. **Depend on identity where identity is carried** (FR-005). A consumer that is
   permutation-invariant can never satisfy this -- `executive_control.conflict()` documents itself
   as such and asserts it in its own self-check.
5. **Degrade gracefully.** MUST function at the ungrown width. Growth may improve a consumer; it may
   never be required by one.

---

## C3. Evidence contract

This is the contract that decides whether a claim counts. It is the strictest of the three because
every prior failure in this area was an evidence failure, not an engineering one.

**Every acceptance claim MUST ship:**

| Requirement | Reason |
|---|---|
| A neuter-and-measure A/B | Reachability is not charge. |
| A matched A/A control reading **0** | Two false positives in Phase 0 were caught only by this. A nonzero A/A means the harness has run nondeterminism and the A/B is unreadable. |
| Hermetic isolation (`persist=False`, isolated `NORD_HOME`) | The same A/B with `persist=True` yields 18 of 102 differing leaves -- almost all of it cross-run contamination. |
| Raw unrounded internals alongside `state()` leaves | The real cortex-to-brainstem effect is ~1e-4 and invisible at `state()`'s 3-decimal rounding. A null at the leaf is not a null in the brain. |
| The exact command that produced each number | A number without its command is not evidence. |

**FORBIDDEN as evidence** (FR-002), because the neuter gates the only thing that changes them:

- `time_cells_grown`, `time_cells_refused`, `time_cells_width`, and any per-region equivalent
- `elapsed_time`, `timing_sd` -- the return value of the very clock being manipulated
- any field on the mechanism's own side of the neuter

An A/B drawing only on these cannot return an answer other than "it changed", which is why the
existing specs/003 charge test reported success while 88 of 93 downstream fields were identical.

---

## C4. Reporting contract

1. Counts written `N of M`, never `N/M`.
2. Region counts always `"N of 15, 2 regions empty and excluded"`. The denominator is never silently
   shrunk.
3. The `reasoning` package reported separately, since it is outside the region map.
4. A census MUST state that it reports a FLOOR, not a census -- name-based scanning undercounts
   inline recruitment such as `GrowingLearner.observe`.
5. A region that grows but is not consumed is reported as **producer-only**, never as complete.
6. ASCII only.

---

## C5. What this contract deliberately does not specify

- **The shared mechanism's shape.** Protocol, base class, registry, or none -- deferred to the M25
  decision gate, which is an A/B rather than an opinion.
- **Which mechanism survives.** P2 requires exactly one, and requires A/B before deleting. The A/B
  has not run.
- **How units get charged after growth.** Growth couples to the learning path, but the coupling is a
  Story P1 implementation decision, constrained here only by C2 and the e-prop rule.

Specifying these now would be deciding by assertion what the plan requires be decided by
measurement.


> **BASELINE CORRECTED 2026-08-03.** Any `0 of 88` / `88 of 93` / `5 of 93` figure in this
> document is a SEED-0 measurement and is NOT representative. Ten-seed sweep: growth moves a
> non-tautological field on **7 of 10 seeds** at the shipped width (mean 1.10) and **9 of 10**
> at init 32 (mean 1.30). SC-001 already passes on most seeds. See the third AMENDMENT in
> spec.md.

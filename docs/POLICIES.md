# Development Policies (P1–P4)

Four standing policies govern how this brain is built. They are **mechanized** where possible (a CI
self-check that fails on violation) and otherwise enforced by review. Each has an **honesty guard** —
the rule that stops the policy from being satisfied by theater.

---

## P1 — Brain-native / no-hardcode

Every behaviour-arbitrating value must be a **learned weight** or a **running-baseline-relative** form
(three-factor R-STDP / e-prop; running EMA or `SignalCalibrator` quantile), not a
magic constant. Corollary: **never keep a constant fixed** — when a hardcode resists replacement, the
blocking constraint is usually the bug, not the number.

**Honesty guard (defeasible only by gate evidence):** a constant may stay fixed in exactly three proven
cases — (a) it is a fixed point on a **normalized** output (e.g. a threshold on a `tanh` confidence in
[-1,1] is dimensionless, like `p ≥ 0.3`); (b) a running-baseline replacement is **not reproducible from
the persisted policy** (e.g. a durable-reload contract persists weights but not the running baseline);
(c) the constant is **inert** (default-off, zero live callers) so a learned form would never charge.
Reuse anchors: `basal_ganglia/r_stdp.py` (`w += lr*d*e`), `substrate/signal_calibrator.py`, the
`_*_base` running-EMA idiom, `cortex/brain.py::_drive_w`.

## P2 — Adopt-better + retire the inferior (non-additive)

When two mechanisms do the same job, make the **better** one THE path and **delete** the other. Do not
leave both live "additively" — that dilutes the signal and grows bloat. Merge when it yields high value.

**Honesty guard:** prove the swap is behaviour-equivalent (or superior) **before** deleting — an A/B on
identical inputs, gate-verified no regression, and any unique capability of the retired mechanism ported
first. Reject the swap with evidence if it regresses (verify-first works when it says *stop*).

## P3 — No dormancy

**Every part of the brain must never remain dormant; all present AND future builds must be wired and
utilized appropriately.** A faculty is not "done" until it is one of three dispositions:

1. **wired + charged** — imported by a live path AND its output routes into behaviour (decision / value
   / learning / reply), proven to change an outcome;
2. **retired** — superseded / no honest consumer → deleted (this is P2 applied to dormancy);
3. **exempt** — legitimately not on the live tick, listed with a reason (train-tier trainer, benchmark
   harness, entrypoint-only tool, or awaiting a prerequisite such as a real sensor).

**Mechanized:** `substrate/dormancy_gate.py` (in `run_selfchecks` CORE) builds the static import graph
and asserts every REGIONS/TORCH_TIER module is **reachable from a live root/entrypoint OR in the EXEMPT
allowlist** — a new build that no live path imports and that is not exempted FAILS CI.

**Honesty guard (the crux):** reachability proves a module is *imported*, NOT that its output changes
behaviour. "Utilized **appropriately**" (does the wire actually *charge*) is a second, per-faculty tier,
proven by the runtime **neuter-A/B probe** (`scratchpad/smoke_drive_charge.py` → "BEHAVIOURAL CHARGE OK"
vs "CONSUMED-BUT-NOT-DECISIVE") during a wiring wave — too expensive for CI. So "never dormant" must NOT
become "force-wire everything": a wire that does not charge is **reverted and retired/exempted**, never
faked with a dead consumer. (History: a coverage-audit once flagged PFC / amygdala / cortical-column /
synfire as dormant; every one was REFUTED on runtime A/B. Static grep proves import-absence, not
behavioural inertness — mark unproven cases UNCERTAIN, not dormant.)

### P3 amendment — dormancy is per-METHOD, not per-module

**If a module on a live path computes a prediction error, free energy, or any other learning signal,
the update rule that consumes it must also be called from a live path.** A learning-verb method on a
reachable module with no live-path call site is dormant, and must be wired, retired, or listed in
`EXEMPT_METHOD` with a disposition and a reason.

**Why module granularity is not enough.** `dormancy_gate.py:71` keys on the module basename, so a
module passes P3 clean with half its capability dead. That is the live state of
`cortex/predictive_coding.py`: the brain settles its 3-level Rao-Ballard net on **every wake tick**
(`cortex/live_brain.py:161`), and the error-neuron architecture is implemented faithfully — but
`PredictiveCodingNetwork.learn()` (`:244`) has exactly **one** call site in the tree, inside the
module's own `__main__` self-check (`:332`). The live path calls `reset()` + `infer()` only, against
weights that are seeded random at construction and never change (verified: `w1` is `array_equal`
before and after 40 further wake ticks). We built predictive coding's inference and left its learning
uncalled, and the module-level gate cannot see that.

**Mechanization: SPECIFIED, NOT YET BUILT.** A pure static AST pass (~1 s, no runtime cost). Two
details are load-bearing, both found by implementing the naive version first and watching it fail:

1. **Import-scoped resolution.** Matching bare `ast.Attribute` call names across all reachable modules
   scores 0 failures / 47 UNCERTAIN — the name `learn` resolves in 20 unrelated modules, so it cannot
   even catch its own motivating case. A call site counts only if its module imports the defining
   module.
2. **Same-module callers COUNT, except under `if __name__ == "__main__"`.** Excluding the defining
   module produces a real false positive (`plasticity/meta_plasticity.py:261` legitimately drives
   `_Unit.learn` at `:131`). The `__main__` guard is precisely what isolates the genuine case.

Under that rule the current candidate set is 5: `cortex/predictive_coding.py:244`,
`hippocampus/affective_memory.py:173`, `neuromodulation/drive_reduction.py:196`,
`neuromodulation/empowerment.py:86`, `plasticity/meta_plasticity.py:131`.

**Honesty guard.** (a) P3's neuter-A/B tier applies unchanged — a call site proves a *caller*, not a
*charge*; adding a dead consumer to silence the gate is the failure this policy already forbids.
(b) An unresolved name is UNCERTAIN and hand-adjudicated, never auto-dormant. (c) `EXEMPT_METHOD`
entries tagged `wire` that outlive their wave are reported as a standing TODO, so exempting everything
is visible rather than silent.

**Stated honestly: this amendment would have caught no past incident.** The audited history is
*consumption* gaps (a producer with callers whose output nothing reads) and an unsupplied constructor
argument — a call-site check passes all of them. It earns its place on the one live case it finds
today, not on precedent.

## P4 — One brain / no replication

**There is only 1 brain.** A build must **utilize** the canonical brain functionality, never
**re-implement** it — that causes bloat. The canonical brain is a clean 3-layer **composition**:
`cortex.brain.Brain` (cognition + circadian clock) → composes `cortex.live_brain.LiveBrain` (perception
+ workspace) → composes `brainstem.autonomous_loop.AutonomousArm` (action + reward + arm + SAFETY).

**Composition is NOT replication** (important): `Brain` owning an inner `LiveBrain` + wake/sleep loop is
the *correct* design — it delegates, it does not re-implement. The bloat P4 targets is a **second**
definition of a stack class (a parallel "brain"), a consumer standing up its **own** brain + drive-loop
+ reply engine, or duplicate world-model / Q-learner families.

**Mechanized:** `substrate/replication_guard.py` (in `run_selfchecks` CORE) asserts each canonical stack
class (`Brain` / `LiveBrain` / `AutonomousArm`) is defined **exactly once**, in its canonical module,
unless a known duplicate is on the `PENDING_RETIREMENT` allowlist (with a reason). The single sanctioned
construction path is **`cortex.brain.make_brain()`** — consumers obtain the brain through it rather than
calling `Brain(...)` directly. "One brain" = one canonical class + one construction path, not a
process-wide singleton (the daemon, a chat session, and a test each legitimately own an instance).

**Honesty guard:** before retiring a replica, port its unique capability into the canonical path and
verify nothing is lost (P2 discipline).

---

## Cross-cutting invariants (never relaxed by any policy)

- **Safety envelope untouched:** the five brainstem files — `reflex.py`, `serial_arm.py`, `arm_sim.py`,
  `arm_config.py`, `interpreter.py` — are modified only with explicit approval; the reflex E-STOP and
  cortisol/safety paths are never relaxed. Every wired drive enters selection **after** the E-STOP
  early-return, so it is safe by construction (proven: max drive + distance 0 → EMERGENCY_STOP).
- **E-prop, not backprop (review-enforced, NO GATE).** Any backprop consideration pivots to **e-prop**
  (Bellec 2020) — a local eligibility trace times a broadcast learning signal; no BPTT, no weight
  transport — verified **cosine > 0.5** against an independent reverse-mode BPTT gradient. Verification
  anchor: `reasoning/eprop.py:38-40`, with the independent gradient at `:191-192`.
  This rule governed the project before it was written down (it is why v3.79 pivoted all learning to
  e-prop on measured evidence: 73.14% vs the repo's backprop SSM at ~32%), which was the gap — a rule
  enforced only by whoever remembered it.
  **It is NOT mechanized, and a `.backward()` tripwire would be theatre:** it passes green today while
  `cerebellum/jepa_world_model.py:272` performs explicit weight transport on every wake tick (the same
  `Wz` used forward and transposed backward, in hand-written numpy) and `hippocampus/ng_snn.py:137`
  runs a closed-form batch solve over its whole buffer each sleep tick. Neither uses autograd; both are
  non-local. Absence of `.backward()` is not evidence of locality.
- **Verify-first + honest framing:** probe → act → prove (A/B / gate) → commit; report the true number
  including a regression; reject theater with evidence; never fake a capability, a metric, or a consumer.
- **Gated + reversible:** one change per commit, full gate green (`run_selfchecks` + `pytest` + `ruff`),
  commit + tag locally; a regression is isolated to its own change.

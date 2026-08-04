# Quickstart: validating whole-brain neurogenesis

**Feature**: specs/007-whole-brain-neurogenesis | **Phase**: 1 | **Date**: 2026-08-03

How to run the acceptance measurements and read their output honestly.

**Read this first.** The five measurements below do not have runnable tools yet -- building them is
the first task of Phase 2, and Measurement 1's harness is a prerequisite for every other claim in
this feature. Each is therefore specified as *what the harness must do and what it must print*,
never as a command you can paste today. Writing a paste-ready command for a tool that does not exist
is the exact defect `citation_guard` catches, and it caught this document's first draft.

The baselines quoted are real -- they were produced during Phase 0 research by scratch probes -- and
they are what the feature must move.

## Prerequisites

```bash
cd D:/AI-project/spiking-neural-data-lake
```

Interpreter is the deterministic CPU gate env `.venv-lake\Scripts\python.exe` with `PYTHONPATH=src`.
Do not use the GPU env for acceptance runs: the gate must be reproducible and CI has no GPU.

## The three gates, which DO run today

```bash
.venv-lake/Scripts/python.exe run_selfchecks.py
```

```bash
.venv-lake/Scripts/python.exe -m pytest -q
```

```bash
.venv-lake/Scripts/python.exe -m ruff check .
```

Current: 278 of 278 self-checks, pytest exit 0 at 92.85% coverage, ruff clean. All three must be
green before and after.

---

## Measurement 1: the charge A/B (SC-001) -- BUILD THIS FIRST

**What the harness must do**: run the brain twice from an identical seed for an equal tick count,
once with growth enabled and once disabled, and compare every scalar field of `Brain.state()`.

**Required properties, all load-bearing:**

- **Hermetic**: `persist=False` and an isolated home directory. The same A/B with persistence
  enabled yields **18 of 102** differing leaves, nearly all cross-run contamination rather than
  growth.
- **A matched A/A control**, growth-ON against growth-ON, which MUST read `0 of N`. An uncontrolled
  version of this harness reported 5 non-tautological movers in Phase 0 -- `access_index`,
  `grown_units`, `predict_accuracy`, `reliability`, `self_surprise_gen` -- every one of which
  differs ON-vs-ON as well. Without the control those would have been reported as the feature
  working.
- **Raw unrounded internals** captured alongside `state()` leaves, because `state()` rounds to 3
  decimals and the real effect is ~1e-4.

**Current baseline, which is what must move:**

```
time-cell width  grow ON: 69   grow OFF: 64
fields that DIFFER      : 5 of 93
identical               : 88 of 93
  elapsed_time          <- the manipulated clock's own output   FORBIDDEN as evidence
  timing_sd             <- the manipulated clock's own output   FORBIDDEN as evidence
  time_cells_grown      <- the mechanism's own counter          FORBIDDEN as evidence
  time_cells_refused    <- the mechanism's own counter          FORBIDDEN as evidence
  time_cells_width      <- the mechanism's own counter          FORBIDDEN as evidence
```

**Reading it**: all 5 movers are forbidden as evidence under contract C3, so the honest score today
is **0 of 88 non-tautological fields**. SC-001 requires at least 1.

## Measurement 2: the sub-resolution check

**What it must do**: the same A/B, but reading the brainstem's raw internals rather than rounded
state leaves.

**Current baseline:**

```
A/A  arm._ext_drive  0 of 300   (max |delta| 0.000000000)
A/B  arm._ext_drive  211 of 300 (max |delta| 0.000099808)
A/B  arm._conflict    23 of 300 (max |delta| 0.000100000)
```

**Reading it**: the channel is not dead, it is three orders of magnitude too small to survive
rounding and is then absorbed into a summation. This is why the plan fixes the CONSUMER rather than
growing harder. A null at a rounded leaf is not a null in the brain.

## Measurement 3: does the earn gate discriminate (SC-005)

**What it must do**: run three arms -- both gate halves live; the resolution floor forced always-
true; the held-out benefit test forced always-negative.

**Current baseline:**

```
ARM A  both arms live                n_cells  68  grown   4  refused 196
ARM B  resolvable() always True      n_cells 264  grown 200  refused   0
ARM C  would_earn() always -1.0      n_cells  68  grown   4  refused 196
```

**Reading it**: ARM C is byte-identical to ARM A, so the held-out benefit arm contributes **0 of 196
refusals**. Every refusal comes from the resolution floor. Any future gate change must move ARM C
away from ARM A, or it has changed nothing.

## Measurement 4: determinism (SC-004)

**What it must do**: construct the brain repeatedly at a fixed seed and compare grown widths and a
state hash. MUST report `20 of 20 identical`.

Growth decisions consume zero random numbers (R1, measured: `rng.getstate()` bit-identical across 50
attempts), so grown-vs-ungrown leaves the internal ramp bit-identical while the decoded output
differs. If this trial count ever regresses, R1 has been violated somewhere.

## Measurement 5: the region census (SC-002)

**What it must do**: inventory every region as producer and consumer, distinguishing OWN growth code
from a BORROWED mechanism the region merely embeds.

**Required output shape:**

```
producers      : N of 15, 2 regions empty and excluded
consumers      : N of 15, 2 regions empty and excluded
BOTH           : N of 15      <- the feature's headline
reasoning pkg  : reported separately (outside the region map)
NOTE: name-based scanning reports a FLOOR, not a census.
```

**Current baseline**: 2 of 15 regions grow with their own code (cortex, hippocampus); 4 more grow
only by embedding another region's learner; 8 have no growth path at all. **BOTH is 0 of 15**,
because nothing is consumed.

---

## What "done" looks like

| Criterion | Today | Required |
|---|---|---|
| SC-001 non-tautological movers | 0 of 88 | at least 1, with the A/A control at 0 |
| SC-002 regions that BOTH produce and consume | 0 of 15 | above baseline, denominator stated |
| SC-004 determinism | not yet measured | 20 of 20 |
| SC-005 gate discriminates | refusals exist, but 0 of 196 from the benefit arm | ARM C must diverge from ARM A |
| SC-007 surviving mechanisms | 5 growth paths | exactly 1 |

## The failure mode to watch for

If SC-001 starts passing, check FIRST that the consumer was not invented to make it pass. A branch
added to a module purely so a wider population changes its output satisfies the letter of the
criterion and violates P3's honesty guard and Standing Invariant 2.

The test: **would this consumer exist, and read this quantity, if the feature had never been
written?** If not, report P1 blocked and amend the spec. Do not add an artificial branch so the
number moves.


> **BASELINE CORRECTED 2026-08-03.** Any `0 of 88` / `88 of 93` / `5 of 93` figure in this
> document is a SEED-0 measurement and is NOT representative. Ten-seed sweep: growth moves a
> non-tautological field on **7 of 10 seeds** at the shipped width (mean 1.10) and **9 of 10**
> at init 32 (mean 1.30). SC-001 already passes on most seeds. See the third AMENDMENT in
> spec.md.

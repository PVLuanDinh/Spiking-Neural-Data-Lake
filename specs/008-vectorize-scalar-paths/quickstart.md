# Quickstart: validating the vectorization work

**Feature**: specs/008-vectorize-scalar-paths | **Phase**: 1 | **Date**: 2026-08-04

How to run the measurements and read them honestly.

**Read this first.** `tools/convert_gate.py` does not exist yet -- building it is the first task of
Phase 2. Measurements that need it are specified as *what the harness must do and what it must
print*, never as a paste-ready command. Writing a runnable command for a tool that does not exist is
the defect `citation_guard` catches, and it caught the specs/007 quickstart's first draft.

`tools/deep_fingerprint.py` DOES exist and its commands below are real.

## Prerequisites

```bash
cd D:/AI-project/spiking-neural-data-lake
```

CPU gate env `.venv-lake\Scripts\python.exe` with `PYTHONPATH=src`. GPU tier `.venv-cu128`, run from
PowerShell. The CPU gate is authoritative; CI has no GPU.

## The gates, which run today

```bash
.venv-lake/Scripts/python.exe run_selfchecks.py
```

```bash
.venv-lake/Scripts/python.exe -m pytest -q
```

```bash
.venv-lake/Scripts/python.exe -m ruff check .
```

Current: 278 of 278, pytest exit 0 at 92.85%, ruff clean.

## The instrument, which runs today

```bash
.venv-lake/Scripts/python.exe tools/deep_fingerprint.py
```

Self-check: 4 of 4 properties (leaf walk, diff, cycle guard, tautology filter).

```bash
.venv-lake/Scripts/python.exe tools/deep_fingerprint.py --demo
```

A/A control on a default brain. **MUST print 0 differing.** A nonzero A/A means run nondeterminism
and every A/B through this harness is unreadable.

---

## Measurement 1: Gate Zero -- the Amdahl ceiling (BUILD FIRST)

**What it must do**: neuter a site to a constant-returning stub, rebinding EVERY consumer module,
and time the tick with and without.

**Baseline, measured:**

```
live tick                     : 74.26 ms
_mathutils.dot, neutered      : 16.02 percent of tick  ->  Amdahl ceiling 1.191x
same site via cProfile        :  7.6 percent, ranked SECOND
cProfile's first-ranked site  : crossmodal_binder.py:117 at 4,966,400 calls
```

**Reading it**: attribution is not time. A site cannot deliver more than its ceiling no matter what
its microbenchmark says, and the profiler disagreed with the neuter by 2x on magnitude and by rank.

## Measurement 2: the live width histogram

**What it must do**: count the widths a site actually sees over 200+ live ticks.

**Baseline, measured:**

```
dot calls over 200 ticks : 437,268
   width 16 : 424,416  (97.06%)
   width 64 :  11,192  ( 2.56%)
```

**Reading it**: the wide case does not exist. Any speedup quoted at width 512 describes 0 percent of
this brain's traffic.

## Measurement 3: drop-in versus data format -- the decision that shapes the feature

**What it must do**: time the scalar loop, the array op on pre-converted data, and the array op
paying per-call conversion.

**Baseline, measured, 20,000 reps per cell:**

```
width  scalar us   numpy PRE-array   numpy +convert   drop-in verdict
   16      0.909            0.494            1.663    0.55x  SLOWER
   64      2.821            0.496            4.051    0.70x  SLOWER
  512     20.058            0.533           26.763    0.75x  SLOWER
```

**Reading it**: the function swap LOSES at every width; the format change WINS by 1.84x to 37.6x.
Any conversion that leaves data as Python lists has made the brain slower. This is the single most
important number in the feature.

## Measurement 4: equivalence, against the right instrument

**What it must do**: A/B the converted site with a matched A/A control, through
`tools/deep_fingerprint.py`, with the swap proven charged.

**The instrument comparison that motivates this:**

```
deep fingerprint   A/A 0 of 98,320    B/B 0 of 98,320    A/B 1,713 of 98,320
via Brain.state()  A/A 0 of 149       B/B 0 of 149       A/B     0 of 149
```

**Reading it**: `state()` reports a clean null where 1,713 leaves moved. **A null measured through
`state()` is not a null.**

## Measurement 5: GPU determinism and the CPU/GPU gap

**Baseline, measured on RTX 5070 / torch 2.11.0+cu128:**

```
same op, 100 identical repeats -> DISTINCT results:
  sum, elementwise LIF, sparse matvec at 1M and 16M   : 1 distinct  (bit-stable)
  atomics-based spike accumulate                      : NOT stable
CPU vs GPU, identical input:
  N=1,000 and 100,000  absdiff 0.000e+00
  N=1,000,000          rel 1.504e-06
  N=16,000,000         rel 1.464e-04
```

**Reading it**: determinism is a KERNEL CHOICE, not a device property -- avoid atomics and the GPU
path is reproducible. But the two tiers do not agree with each other and the gap grows with N, so
each needs its own gate and neither proves the other.

---

## What "done" looks like

| Criterion | Today | Required |
|---|---|---|
| Perimeter established on one subsystem | none | 1, measured faster END TO END |
| Internal conversions inside a perimeter | n/a | 0 |
| Sites with a verdict (converted or DECLINED with a number) | 0 | all hot sites |
| Nulls measured through `Brain.state()` | historical | 0 |
| Connectivity law | UNMEASURED | measured and fitted |
| GPU path atomics on the live path | n/a | 0 |

## The failure mode to watch for

**A half-converted subsystem measures slower than either endpoint.** The drop-in penalty is paid at
every boundary, so a perimeter that is not closed is not a partial win -- it is a loss. If P1's one
subsystem cannot be made faster end to end, the approach is wrong and the feature should STOP rather
than widen. That is a legitimate outcome and it is cheaper to discover at one subsystem than at ten.

---

## DATED SUCCESSION 2026-09-09 (specs/022 S6 panel fold)

Nothing above is edited: it is the record of what was planned and measured in 2026-08. What has since
changed, in one place:

- **Gate Zero LANDED** as `src/snn_data_lake/substrate/amdahl_gate.py` (`b8413c1`, 2026-09-09), NOT as
  `tools/convert_gate.py` -- that path is refused by the p4c ratchet, which counts `tools/` files as
  ungoverned (`tasks.md`'s Discipline note). Measurement 1's "(BUILD FIRST)" is therefore satisfied and
  every `tools/convert_gate.py` reference above is a planned location, not a live one.
- **The T0-T3-ladder half of that planned tool is still open**, as `tasks.md` T004.
- **This file's 74.26 ms baseline is 2026-08's** and reads stale against the 108.663 ms wake p50 recorded
  at `../022-unified-brain-arc/evidence/s6-kernel-wire.md:61`. `spec.md` SC-001 still pins 74.26; re-recording
  that baseline is its own task with its own measurement, never a doc annotation, so it is NOT done here.
- **Run the gate with this file's own pinned env** -- bare `python` on the build host resolves to a 3.11
  interpreter that the free-threading import guard refuses:
  `PYTHONUTF8=1 PYTHONPATH=src .venv-lake\Scripts\python.exe -m snn_data_lake.substrate.amdahl_gate --self-check`

The same line is appended to `plan.md`, whose `tools/convert_gate.py` references carry the identical
superseded assertion. The S6 disposition's own pre-registered scope was "specs/008 gets its `tasks.md`",
so this is residual doc drift recorded forward, not a missed obligation of that arc.

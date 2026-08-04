# Quickstart: validating sparse connectivity

**Feature**: specs/009-sparse-connectivity | **Phase**: 1 | **Date**: 2026-08-04

How to run the measurements and read them honestly.

**Read this first.** `tools/fanout_bisect.py` does not exist yet -- building it is a Phase 2 task.
Measurements needing it are specified as *what the harness must do and print*, never as a paste-ready
command. Writing a runnable command for a tool that does not exist is the defect `citation_guard`
catches. Everything else below runs today.

## Prerequisites

```bash
cd D:/AI-project/spiking-neural-data-lake
```

Interpreter `.venv-lake\Scripts\python.exe` with `PYTHONPATH=src`.

## The gates

```bash
.venv-lake/Scripts/python.exe run_selfchecks.py
```

```bash
.venv-lake/Scripts/python.exe -m pytest tests/test_sparse_connectivity.py -q
```

```bash
.venv-lake/Scripts/python.exe -m ruff check .
```

The sparse suite is 7 tests: function first, savings second, determinism, symmetry, mask
containment, dense-unchanged.

## The module's own self-check, which must stay green

```bash
.venv-lake/Scripts/python.exe -m snn_data_lake.hippocampus.episode_consolidation
```

Reports 64 real episodes stored, all completing from a 50 percent partial cue, earliest surviving
later stores and replay, verbatim episode read-out, and the negative control (unbalanced blank-noise
replay) collapsing to ~0.72. **If thinning ever breaks the negative control's collapse, the metric
has stopped discriminating and every number below is worthless.**

---

## Measurement 1: function survives thinning (SC-002) -- the gating one

**Baseline, measured, load 40, seeds fixed:**

```
     N        F   mean degree   density   retained   verdict
   256    dense         255.0     99.6%      1.000   baseline
   256      160         219.4     85.7%      1.000   PASS
   256      128         191.6     74.8%      1.000   PASS
  1024    dense        1023.0     99.9%      1.000   baseline
  1024      160         294.8     28.8%      1.000   PASS
  1024      128         240.0     23.4%      1.000   PASS
```

**Reading it**: at N=1,024 and F=128 the store keeps 23.4 percent of its connections and loses
nothing measurable. Note `mean degree` is 240, not 128 -- symmetrisation raises realised degree
above the request, and the reported number is the measured one.

**The trap**: an earlier run of this comparison stored raw random vectors instead of the module's
own `encode()` output. The dense baseline read 0.150 rather than 1.000, and the sparse comparison
was meaningless -- 95 percent of a broken baseline is not evidence. **Always build through
`encode()`.**

## Measurement 2: the knee, bracketed (SC-001) -- BUILD FIRST

**What the harness must do**: bisect F against the module's own metric at a fixed width and load,
with the pre-registered tolerance encoded, and REFUSE to report an unbracketed F\*.

**Required output shape:**

```
N=<width>  load=<L>  seeds=0..4
  tolerance : mean 0.02, min 0.05 vs dense at the same width and load
  dense     : <baseline>
  F=<...>   : mean <...>  min <...>   PASS/FAIL
  F* = <value>   BRACKETED by F=<lower> which failed
  (or: F* CENSORED at F >= <grid floor> -- no failing F below it)
```

**Baseline from research:**

```
     N     F*   knee density   bracketed by
   256    176        68.8%     160 -> 0.965 fails
 1,024    160        15.6%     144 -> 0.975
 4,096    160         3.9%     128 -> 0.975
16,384    128         0.78%    112 -> 0.335  (a CLIFF, not a knee)
```

## Measurement 3: does the saving grow with population (SC-003)

**What it must do**: report connection memory against population at fixed F, and fit.

**The number that decides the feature** -- whether stored load scales with the population:

```
FIXED load        : F* falls 176 -> 128 over a 64x width range   -> memory O(N*F), LINEAR
PROPORTIONAL load : uniform F* grows 160 -> 1,024 as N^0.68      -> memory O(N^1.68)
                    top-k-per-row F* holds 96 -> 128             -> the survivable path
```

**Reading it**: this is not settled for THIS brain. Whether its stored load scales with its
population is open question 1 and the whole feature turns on it.

## Measurement 4: determinism (SC-005)

```bash
.venv-lake/Scripts/python.exe -m pytest tests/test_sparse_connectivity.py -q -k deterministic
```

Same seed must give identical wiring; different seeds must not.

---

## What "done" looks like

| Criterion | Today | Required |
|---|---|---|
| SC-001 fanout does not grow with population | verified at fixed load only | across a 16x range at the brain's real load |
| SC-002 function within tolerance | 1 module, 2 widths | every module reported SPARSE |
| SC-003 memory ~linear | measured at 10^4 | fitted across the ladder |
| SC-004 ceiling raised 10x | 312x at fixed load | held at the real load |
| SC-005 determinism | PASS | 20 of 20 |
| SC-006 exactly 1 mechanism | **2 -- dense and sparse coexist** | 1 (closes the P2 deviation) |
| SC-007 "N of M sparse, K exceptions" | 1 sparse, 0 exceptions, 1 out of scope, 10 unmeasured, of 12 | all four numbers, always |

## The failure mode to watch for

**Every number in this feature except function improves as connections are deleted.** Memory,
density, fanout, ceiling -- all maximised by an empty brain. If a report leads with a saving and
mentions function second or not at all, it is measuring the benefit and calling it the result.
Function is the constraint; the saving is what you get for respecting it.

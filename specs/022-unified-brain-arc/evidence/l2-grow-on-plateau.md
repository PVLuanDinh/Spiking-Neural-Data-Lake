# L2 — grow on plateau, prune in sleep (FR-023, FR-024)

*Pre-registration written BEFORE any code. Red L-R3 (Part F). The row's trigger was RESTATED in the plan on
2026-09-05 when L1b was deleted on its null, and this file carries the consequence of that restatement.*

## 1. Territory (read 2026-09-06 at `a96730d`)

**The row as written names a mechanism that no longer exists.** L2's red says *"neutering L1b raises the
growth rate"*. L1b — a local rule on the grown units' INPUT weights — was measured and DELETED on its null
(`evidence/l1b-synaptic-online.md`, P8 clause 5). There is no L1b to neuter. The plan anticipated exactly
this and restated the trigger: **L2 reads the online READOUT's residual plateau if L1c lands.** L1c landed
(`evidence/l1c-online-readout.md`, the delta rule on `NGSNN.beta`), so that is the branch this row takes.

What exists today:

- **`cortex/growth_stop.GrowthStop` (T047)** decides, per region, whether the NEXT refit may grow. Two
  evidence streams, both its own: the refit trajectory (`observe_refit(region, n_units, heldout_acc)`) and
  the |LP| facade through an injected reader. The stop is STICKY and re-opens only on charging evidence. It
  accepts no tuning input at decision time — the executive-independence contract — and `should_grow(region)`
  is consulted on the sleep path from `cortex/brain.py`.
- **`hippocampus/online_readout.OnlineReadout`** learns `beta` on every wake percept. Its `learn()` already
  computes `p = softmax(s @ beta)` on the production path, so the readout's own prediction error is one
  subtraction away and costs nothing extra.
- **`cortex/neurogenesis_engine.NeurogenesisSelfImprove`** refits every 20 sleep ticks and grows on top.

**What is MISSING is the trigger, not the stop.** T047 answers *"has this region grown past its optimum?"*
Nothing anywhere answers *"has learning stalled, so that a new unit is the remaining move?"* A refit today
may grow while the readout is still improving on the units it already has, which is growth substituting for
learning — the pathology FR-023 names.

## 2. Pre-registration

### 2.1 Exactly one mechanism (P2)

The plateau condition lands as a **third evidence stream inside `GrowthStop`**, not as a second gate. A
separate plateau gate beside the stop gate would be two mechanisms deciding one question, and the caller
would then compose them — which is precisely the executive-constrains-its-own-guard shape T047 was built to
avoid. Concretely:

- `GrowthStop.observe_residual(region, residual)` accumulates the readout's own error, the same shape
  `observe_refit` already has.
- `should_grow(region)` returns True only if the existing criterion says OPEN **and** the residual has
  plateaued. It still takes the region name and nothing else.

### 2.2 The residual, and why this one

`OnlineReadout.learn` computes `p = softmax(s @ beta)` before its update. The residual is
`1 - p[cat]` — the probability mass the readout did NOT put on the category that actually followed. It is
the readout's own error on the transition it is about to learn from, measured BEFORE the update, on the
production path, with no second forward pass. It is scored on the same pair the live instrument scores, so
it inherits that instrument's honesty and its limits.

### 2.3 The plateau test, self-relative

The same null discipline the rest of the gate uses, and for the same reason (a fixed threshold is calibrated
on the wrong distribution): a region has PLATEAUED when the mean residual over the last `PATIENCE` readings
is not below the mean of the `PATIENCE` before them by more than the series' own noise scale — median
absolute step, converted to the null SE of a difference of two `PATIENCE`-means exactly as `_decide`
already does. No new constant is introduced; `PATIENCE` and the noise conversion are reused, which is also
what stops this arm from being tuned independently of the one beside it.

**An unobserved residual permits growth.** The gate cannot gate on what it has never measured — the same
rule `should_grow` already applies to an unobserved region. Stated because the alternative is a silent
global growth stop the first time a caller forgets to feed the stream.

### 2.4 Red and controls (L-R3)

- **RED — the trigger is real.** On a stream whose readout error is still FALLING, the gate must refuse
  growth; on the same stream once the error flattens, it must permit it. A gate that permits either way has
  no trigger, and a gate that refuses either way has stopped growth rather than gated it. Both arms in one
  command.
- **ATTRIBUTION — the restated form of the row's own red.** NEUTER the online readout (`mu = 0`, so `beta`
  never moves and the residual cannot fall) and the growth rate must CHANGE against the live readout. The
  direction is predicted here, before the run: neutering should RAISE it, because a constant residual is
  trivially flat and every refit then reads as plateaued. If the rate does not change, the arm is not
  attributable to the readout and the row deletes it.
- **CONTROL (not the method) — the gate with the plateau arm OFF** is the incumbent T047 behaviour, and it
  must reproduce the incumbent growth rate. This is the OFF arm of a real A/B, not a list of literals.
- **CONTROL 2 — the stop still stops.** T047's own overgrowth red must still trip and hold with the plateau
  arm present. An arm that suppresses growth so thoroughly that the stop never has anything to stop would
  pass the red above and destroy the gate beside it.

### 2.5 Disposition, fixed in advance

| Outcome | Disposition |
|---|---|
| The red fires both ways AND neutering changes the growth rate | The arm lands. Record the direction and the size. |
| The red fires but neutering does NOT change the growth rate | The plateau arm is not attributable to the readout. DELETE it (P8 clause 5) and record that the residual carried no growth-relevant signal at this cadence. |
| The red cannot fire in one direction | The criterion has no reachable red in that direction and is DELETED rather than weakened. |
| The plateau arm suppresses growth entirely on the live stream | Refuse the landing and record it: a trigger that never fires is a growth stop wearing a trigger's name. |

### 2.6 What this row does NOT do

It does not build a second grower — the P2 adjudication already recorded in the plan stands: `GrowingLearner`
recruits PROTOTYPE vectors, `NGSNN` admits LIF units, they are composed and no second LIF grower is built. It
does not change what a refit does once permitted. It does not touch pruning, which stays in sleep as it is.
It does not claim the readout's residual measures learning in general — it measures this readout's error on
this stream, which is what the trigger needs and no more.

## 3. Measured

### 3.0 The first run was a NULL, and the null was the instrument

The first measurement showed the plateau arm changing **nothing**: 8 of 8 seeds identical across the live
arm, the neutered arm and the control. Before that was written anywhere it was checked, and the cause was in
my own code: the caller's cursor was `len(self._res_hist[region])`, and the history trims to `_WINDOW`. Once
the gate held 36 rows the slice `residuals[36:]` was permanently empty and **the gate never received another
residual.**

`GrowthStop.observed()` carries a docstring that names this exact defect — *"a monotone counter, deliberately
NOT len(history): the evidence windows trim to _WINDOW rows and a cursor that shrank with them would re-feed
old rows"* — and I wrote the broken cursor after reading it. The cursor was then **removed rather than
fixed**: the residual window is the producer's own trailing buffer, is not persisted, and starts empty after
a relaunch, so `observe_residuals(region, series)` replaces the gate's copy. The class of bug is gone rather
than corrected. Recorded here because a null and a dead detector are indistinguishable in the output, and
this one would have read as a clean deletion licence.

### 3.1 The red, both directions, on the criterion itself

| residual series | verdict | reason |
|---|---|---|
| still falling (0.90 → 0.24) | **REFUSES growth** | residual fell +0.1800 > null scale 0.0514 over the last 6 transitions |
| the same, flattened | **PERMITS growth** | residual fell −0.0027 ≤ null scale 0.0068 |
| unobserved / shorter than 2·PATIENCE | **PERMITS growth** | no stall evidence |

Both directions fire, so the criterion has a reachable red in each. A fourth red is structural and is the one
the design turns on: a plateau refusal must not become the sticky stop. Constructed and run — refuse on a
falling residual, feed a flat one, and the same refit history grows again. Had the refusal reached the stop
log, *"not yet"* would have become *"past the optimum"* and only charging evidence could have left it.

### 3.2 Attribution, 8 seeds × 1200 transitions, refit every 20

| seed | live units / acc | plateau arm OFF | readout NEUTERED (mu=0) | plateau refusals |
|---|---|---|---|---|
| 0 | 44 / 0.017 | 38 / 0.017 | 38 / 0.017 | 3 |
| 1 | 4 / 0.033 | 4 / 0.033 | 4 / 0.033 | 1 |
| 2 | **3 / 0.283** | 27 / 0.067 | 27 / 0.067 | 3 |
| 3 | 3 / 0.217 | 3 / 0.217 | 3 / 0.217 | 3 |
| 4 | 16 / 0.017 | 16 / 0.017 | 16 / 0.017 | 2 |
| 5 | **2 / 0.483** | 38 / 0.000 | 38 / 0.000 | 2 |
| 6 | **3 / 0.700** | 2 / 0.483 | 2 / 0.483 | 12 |
| 7 | 16 / 0.067 | 14 / 0.067 | 14 / 0.067 | 5 |
| mean | 11.38 / 0.2271 | 17.75 / 0.1125 | 17.75 / 0.1125 | 31 total |

**The attribution is clean, and it is an identity rather than a test.** With the readout neutered the arm is
inert: `neutered == arm-off` on **units and accuracy, bit-for-bit, on all 8 seeds**. A frozen readout's
residual cannot fall, so every window reads as plateaued and the trigger never refuses. Any live difference
is therefore the readout's learning, not the arm's presence. The live arm differs from both on 5 of 8 seeds.

### 3.3 The size does NOT clear the plan's bar, in either direction

| comparison | mean | t (bar 1.895) | sign-flip p (bar 0.05) | ties | verdict |
|---|---|---|---|---|---|
| units, live − arm-off | −6.375 | −1.197 | 0.500 | 3 | **does not pass** |
| accuracy, live − arm-off | +0.1146 | +1.818 | 0.250 | 5 | **does not pass** |
| units, neutered − arm-off | 0.0000 | 0.000 | 1.000 | 8 | the control's own null, armed and silent |
| accuracy, neutered − arm-off | 0.0000 | 0.000 | 1.000 | 8 | the control's own null, armed and silent |

The accuracy comparison has **5 ties, so only 3 non-tied seeds** — DEGENERATE by this project's own rule
(below 5 non-tied, a sign test decides nothing) — and it is recorded as degenerate rather than reported as a
trend. The unit comparison's mean is carried by two seeds (2: 27→3, 5: 38→2) against three that moved the
other way (0: 38→44, 6: 2→3, 7: 14→16).

**What is worth stating without a statistic:** accuracy was **never lower** on any seed, and strictly higher
on three — including seed 5, where the incumbent grew 38 units for an accuracy of **0.000** and the live arm
grew 2 for 0.483. That is the pathology FR-023 names, seen once. It is one seed and it is reported as one
seed.

### 3.4 Control 2: the stop still stops

T047's entire harness passes unchanged with the plateau arm present — overgrowth red 10/10 with the stop
holding, both regimes, the |LP| stationary/excursion/ablation arms, the evidence re-open, idempotence, and
the armed real-engine two-arm null which did not trip.

**This is control-2 and NOT evidence about the trigger, and the harness now says so in its own output.** That
harness constructs its engine with no online readout, so `residual_trace` is empty, the plateau arm permits
every verdict, and the T047 numbers are unchanged BY CONSTRUCTION.

## 4. Disposition applied

The pre-registered branch is the first row of §2.5 — **the red fires both ways and neutering changes the
growth rate** — so the arm LANDS. The attribution is stronger than the row asked for: the neutered arm is not
merely different from the live one, it is bit-identical to the incumbent on every seed and every metric.

**What is claimed:** growth is now admitted only when the online readout's error has stopped falling faster
than its own noise. The trigger is real, it is reachable in both directions, it is attributable to the
readout, and it does not disturb the stop beside it.

**What is NOT claimed, and the numbers that refuse it:** that the arm improves accuracy, or that it reduces
unit count, by any margin this run established. Both fail the plan's paired bar (t −1.197 / +1.818 against
1.895; sign-flip p 0.500 / 0.250 against 0.05), and the accuracy comparison is degenerate at 3 non-tied
seeds. A future run at more seeds may settle either; this one does not, and the bar is not moved after the
fact to rescue it.

**Deletion licence, still open.** P8's clause 5 stands: if a later measurement shows the trigger changes no
outcome the readout does not already change, the arm deletes. The identity `neutered ≡ incumbent` is what
makes that test cheap to run again.

### 4.1 What this row did not do

No second grower was built — the P2 adjudication in the plan stands. Pruning is untouched and stays in sleep.
The row does not claim the residual measures learning in general: it measures this readout's error on this
stream, scored on the same pair the live instrument scores, and inherits that instrument's limits.

### 4.2 One interface change, disclosed

The growth-gate callable took ONE argument by contract, and `tests/test_growth_stop_wire.py` asserted that
contract by name. It now takes two — the refit trajectory and the readout's residual window — with every
implementer taking `residuals=()`. The alternative considered and rejected was for the brain's adapter to
read the residual off the engine directly; that was rejected because the adapter's own docstring records that
it was written to avoid reading the engine, and evidence arriving through a parameter is what keeps the gate
from reaching into its consumer.

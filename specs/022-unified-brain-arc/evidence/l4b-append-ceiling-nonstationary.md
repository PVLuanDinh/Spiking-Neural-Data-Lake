# L4b — the append ceiling is not stale, it is NON-STATIONARY

*Opened 2026-09-06 by the live daemon, minutes after L4 wired `AppendBudget` to its first caller. Measured
before any number was touched. Red L-R8 (Part F, added by this row).*

## 1. What the live brain reported

L4's §8.5 disclosed that an ordinary tick reaches 79–83 % of the `0x07` flood ceiling and was still climbing.
The daemon relaunched on that arc's push and, at `nights_elapsed = 8`, reported:

```
"append_floods": 7
"append_flood_last": "FLOOD: tag 0x07 (IMMORTAL) appended 160470 B this tick > 8.0x its recorded p95 (12381 B)"
```

The stated fix was "re-record the p95 on the pinned host". **That fix is wrong, and this row exists because
measuring it first is what showed why.**

## 2. Measured: the quantity a fixed ceiling bounds does not settle

One persisting brain, 1200 ticks, throwaway home, per-tick `0x07` bytes broken down by state name (the live
lake was never opened). Of 1200 ticks, 67 wrote `_T_STATE` at all.

p95 over successive windows of the writing ticks:

| writes | p95 | median | max |
|---|---|---|---|
| 0–10 | 52,763 B | 1,440 B | 52,763 B |
| 11–21 | 59,701 B | 55,778 B | 59,701 B |
| 22–32 | 66,386 B | 65,080 B | 66,386 B |
| 33–43 | 73,015 B | 70,802 B | 73,015 B |
| 44–54 | 75,156 B | 73,119 B | 75,156 B |
| 55–65 | **77,998 B** | 77,244 B | 77,998 B |

**Monotone, every window, with no sign of settling at 1200 ticks.** The live daemon's 160,470 B is roughly
twice this run's endpoint, which is what a brain that has lived far longer should read.

### 2.1 What carries it: whole-frame blobs that grow with the brain

| state name | total over the run | first write | last write | growth |
|---|---|---|---|---|
| `growing_learners.json` | 2,560,272 B | 26,561 B | 61,389 B | **×2.3** |
| `reasoning_curriculum.json` | 517,588 B | 5,514 B | 11,014 B | ×2.0 |
| `nightly_skills.json` | 93,456 B | 865 B | 2,692 B | ×3.1 |
| `coverage_cells` | 60,614 B | 165 B | 3,087 B | **×18.7** |
| `time_cells.json` | 28,800 B | 1,343 B | 1,690 B | ×1.3 |

Each is a WHOLE FRAME rewritten in full on every persist tick, and each frame grows as the brain grows. The
per-tick byte count is therefore a function of the brain's size, not a property of its behaviour.

### 2.2 The recorded baseline is stale twice over

Re-running the ORIGINAL method — the baseline file records "BinLog.append+extend wrap over
make_brain(seed=0) x120 ticks, NORD_HOME scratch; p95 over nonzero per-tick byte sums" — gives **52,763 B**
on today's tree, against the recorded **12,381 B**. So the number is stale for two independent reasons: the
tree changed under it, and the brain grows past it. Only the second is interesting.

## 3. The consequence, stated plainly

**A fixed per-tick byte ceiling on `0x07` is stale by construction.** Re-recording it buys months and then
fires forever again. Seven floods in eight nights is not detection — it is a stuck alarm, and a stuck alarm
is worse than none because it trains its reader to ignore the one that matters.

This is also what P1 forbids: *learned-weight or running-baseline-relative forms; never keep a magic constant
fixed*. The recorded p95 is a magic constant that was measured once, and the class it bounds is one whose
honest per-tick size is a moving target.

## 4. Two pathologies, and the instrument only ever addressed one

The confusion this row resolves is that `AppendBudget` was asked to catch two different things with one
number:

- **A FLOOD** — a sudden spike against what this class normally writes. That is a rate anomaly, and it is
  correctly measured against the class's OWN RECENT behaviour.
- **A RAMP** — a class growing without bound over the brain's life. That is the pathology the module
  docstring names (`_T_GROWTH` reaching 97.64 % of the lake), and a fixed p95 never detected it: it fires
  once, forever, and says nothing about the slope.

A fixed ceiling answers neither well. It false-positives on ordinary growth (measured above) and it cannot
report a slope.

## 5. Pre-registration

### 5.1 The change

`AppendBudget` keeps a RUNNING baseline per tag — a trailing window of that tag's own nonzero per-tick byte
sums — and flags a flood when a tick exceeds `FLOOD_FACTOR ×` the window's own p95. The recorded file becomes
the SEED for a tag that has not yet accumulated a window, not a permanent ceiling. Exactly one mechanism
survives (P2): the fixed comparison is replaced, not kept alongside.

`FLOOD_FACTOR` is NOT touched. Moving it would be tuning a constant to silence a measurement, which is the
move this row exists to refuse.

### 5.2 Red and controls (L-R8)

- **RED — a real flood is still caught.** A constructed spike far above the class's own recent rate must
  flag, after the window has filled. A budget that adapts to everything catches nothing.
- **CONTROL 1 (not the method) — ordinary growth must NOT flag.** Replay the measured series above (52 KB →
  78 KB over 67 writes, a ×1.5 monotone climb) and require ZERO flags. The current fixed ceiling flags it,
  which is the defect.
- **CONTROL 2 — a cold tag still uses the seed.** Before a window exists, behaviour must match today's, so
  the change cannot silently disarm a tag that has not written yet.
- **CONTROL 3 — the adaptive bar cannot be walked up by the flood itself.** A sustained flood must not
  become the new normal within the window and go quiet. Measure how many consecutive over-bar ticks it
  takes to stop flagging, and state it.

### 5.3 Disposition, fixed in advance

| Outcome | Disposition |
|---|---|
| The red flags and all three controls hold | The change lands, and the recorded file is re-documented as a seed. |
| The red flags but control 1 still flags on ordinary growth | The window is too short to absorb the growth rate; report the measured rate and REFUSE rather than lengthen the window until it passes. |
| The red does not flag after the window fills | The adaptive bar has no reachable red and is DELETED, leaving the fixed one with its staleness disclosed. |
| Control 3 shows a flood goes quiet within a few ticks | Report the number. A bar that a flood can walk up is disclosed as such or the row deletes. |

### 5.4 What this row does NOT do

It does not address the RAMP. Detecting a class growing without bound over a brain's life needs a slope
instrument, and this row explicitly does not pretend a flood assert is one — that is the confusion it exists
to end.

It does not change what `_T_STATE` writes. **The whole-frame rewrite is the deeper finding and it is named,
not fixed here:** rewriting a 61 KB `growing_learners.json` in full on every persist tick to change part of
it is the reason these bytes scale at all, and delta-encoding named state is its own row with its own
reconstruction proof. Recorded so the next reader does not mistake a bounded alarm for a bounded lake.

## 6. Measured

All four arms live in `lake_engine`'s own self-check and in `tests/test_append_budget_wire.py`, so they run
in the battery rather than in a scratch script.

| arm | result |
|---|---|
| **RED** — a real spike, after the window is all-real | **flags**: `appended 1014336 B this tick > 8.0x its own recent rate (62650 B)` |
| **CONTROL 1** — an INTERPOLATION of the measured window-p95 trajectory (52,763 → 77,998 B over 67 writes) | **0 flags**, seeded AND unseeded |
| CONTROL 1, honestly | its largest step is 1.006x against an 8x bar and its max sits under the 99,048 B seed floor, so the SEEDED arm cannot fail on two counts; the UNSEEDED arm beside it is the one that can |
| **CONTROL 2** — day-one protection, a cold tag with a seed | **flags immediately**: the existing calm/flood pair still passes, reported as `a window 31/32 seeded` |
| **CONTROL 3** — a stale seed against the live rate | **3 floods, then quiet** |
| a tag with neither seed nor history | never trips, reported `unbaselined` |

### 6.1 A correction the control itself forced

The first draft of CONTROL 1 asserted that the retired fixed ceiling *would have flagged this series* — the
sentence that motivates the whole change. **It is false, and the test caught it.** The 1200-tick series tops
out at 77,998 B, still UNDER the retired bar of 12,381 × 8 = 99,048 B. What crosses that bar is the LIVE
daemon at 160,470 B/tick, a far older brain further along the same monotone curve.

So the precise statement, which is what both the test and the self-check comment now carry: the 1200-tick
series is the **no-false-alarm control**, and the live rate is where the two comparisons actually diverge.
The overstatement is recorded rather than quietly fixed, because the assertion that caught it is the reason
to write controls that can fail.

### 6.2 The seed migrates instead of standing

The recorded numbers now PRE-FILL each tag's window rather than standing as a permanent ceiling. Day-one
protection is preserved — a tag is bounded from its first tick — and a stale recording self-corrects: against
the live daemon's 160,470 B/tick the stale 12,381 B seed costs **exactly 3 floods** before the window is real,
against *forever* under the retired comparison.

## 7. Disposition applied

§5.3's first row: **the red flags and all three controls hold, so the change lands.** The bar is the tag's own
recent rate; `tools/lake_append_baseline.json` is now the SEED for a tag's opening window, and its other
consumer — L5's `immortal_retention.budget_coverage()` census of which immortal tags have ever been measured —
is unchanged. `FLOOD_FACTOR` was not touched.

### 7.1 The cost, stated rather than buried

**A window that adapts is a window a sustained writer can walk up, and the first version of this section
stated that cost at its BEST rather than its worst.** A panel measured the real shape: `_p95_of` over a
32-window returns the 30th sorted value, which on a rising series is the value from THREE writing ticks back,
so the no-flag condition is r**3 <= FLOOD_FACTOR -- **any per-write growth ratio at or under 2.0 never flags,
at ANY magnitude.** Doubling every write for 67 writes reaches ~9e23 B/tick and flags ZERO times. The
"3 ticks" below is true only of a STEP to a new constant rate; a flood that arrives over three or more
writing ticks costs nothing at all. Both numbers are now printed by the self-check and pinned by tests.

The same mechanism that ends a stale seed's stuck alarm after 3 ticks means a sustained STEP flood goes quiet
after 3 ticks. That is the trade
of any self-relative bar, it is asserted in both the self-check and a test so it cannot drift, and it is why
§5.4 says this row does not address the RAMP: detecting a class growing without bound over a brain's life
needs a slope instrument, and a flood assert has never been one.

What changed is which failure this instrument has. Before: it cried wolf forever on ordinary growth, which
teaches its reader to ignore it — and the live daemon had already reached that state within minutes. After:
it is silent on ordinary growth, loud on a spike, and blind to a slow ramp it was never able to see anyway.

### 7.2 Still open, named, and not fixed here

- **The RAMP has no instrument.** Named in §5.4 and now twice in code.
- **The whole-frame rewrite is the deeper finding.** `growing_learners.json` is rewritten in full — 61 KB by
  tick 1200 — on every persist tick, and `coverage_cells` grew ×18.7 over the same run. That is why these
  bytes scale at all. Delta-encoding named state is its own row with its own reconstruction proof.
- **The window does not survive a relaunch.** The daemon restarts on every push and re-seeds, so each
  relaunch pays the 3-flood migration again. Persisting the window belongs to L4's re-opened persistence
  half, which now has a second named carrier to go with ignition.

---

## 8. Confirmed on the live daemon

The daemon relaunched on this arc's push. Against the same lake and the same workload that produced §1's
seven-and-climbing stuck alarm:

```
"nights_elapsed": 14
"append_floods": 3
"append_flood_last": "FLOOD: tag 0x07 (IMMORTAL) appended 160380 B this tick
                      > 8.0x a window 30/32 seeded (12381 B)"
```

**Three floods, and the message says the window is migrating** — `30/32 seeded`, so the stale recording is
being pushed out by real ticks exactly as designed. This is the migration cost §6 measured in the abstract
(3 floods) and §7.1 disclosed, now observed end to end on the production brain rather than in a harness.

Two things are confirmed at once: the wire still fires on the real lake, and the alarm now **stops**. Before
this arc the same brain reported floods on every consolidation night without limit; the honest failure mode
of the instrument has moved from *cries wolf forever* to *blind to a slow ramp*, which §4 and §7.1 say in
those words.

The 160,380 B figure also confirms §2's non-stationarity from the other end: the live brain writes roughly
twice the 77,998 B this row measured at 1200 ticks, on the same monotone curve.

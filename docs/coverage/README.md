# The coverage ledger — status, and how to read the number in it

`tools/coverage_score.py` re-derives the brain-faculty percentage from the eleven category files in
this directory. Run it:

```bash
python tools/coverage_score.py                 # re-derive the total, or refuse and say why
python tools/coverage_score.py --rerun drive_60  # re-execute a recorded artifact and compare
python tools/coverage_score.py record --kind ab --name drive_60 -- tools/charge_probe.py drive 60
```

## Why this exists

`COVERAGE_RUBRIC.md:8-9` says *"a percentage nobody can re-derive is not a measurement"* — and then the
baseline it governs was assembled entirely by hand. No `.py` in the tree read either document. The
rubric ended the problem for the **standard** and left it open for the **score**. This is the score
half.

The scorer has **no opinion about what band a faculty deserves.** An auditor writes that in a category
file. What the scorer does is *refuse* to total up claims the rubric's own evidence standard does not
support: band > 40 needs an A/B artifact whose **recorded exit code** says the A/B passed, band > 80
needs held-out evidence naming its metric and baseline, every scored row cites a `file:line` that
resolves, and all eleven categories must be present so the denominator cannot quietly shrink.

The anti-forgery core is that a band ceiling comes from the **recorded exit code**, not from the claim.
`tools/charge_probe.py place 60` exits 2 (NOT-ELIGIBLE); a row claiming band 65 against that artifact is
refused, and editing the row cannot change the exit code. It is not airtight — a determined forger can
hand-write a transcript, which survives `score` and dies on `--rerun` — but it removes the *accidental*
inflation this project keeps hitting.

### `record`/`rerun` now isolate `NORD_HOME` themselves, and two artifacts cannot be reproduced at all

Established 2026-07-30 by re-executing **all 82 artifacts** after the `_fingerprint` regex was widened
(see the comment above `_fingerprint` in `tools/coverage_score.py`). Three results worth having in one
place:

1. **70 of 82 reproduced with the operator's live `learn_daemon.py` running; 80 of 82 reproduce once
   `NORD_HOME` points at a scratch directory.** The ten-artifact gap is entirely the daemon: it writes
   `~/.snn_data_lake` continuously, most probes read it, and `consciousness_refute_probe.py` is
   well-behaved enough to *detect* the resulting noise and exit 2 with
   `UNKNOWN: arm A is not reproducible at seed=0 -- no verdict from a noisy harness` rather than emit a
   verdict. **FIXED 2026-07-30: `record` and `rerun` now run every probe against a fresh empty
   `NORD_HOME`** (a per-invocation temp dir), so the ledger no longer depends on what the operator's
   brain happened to be doing. An explicit `NORD_HOME` in the caller's environment still wins, which
   keeps a frozen-lake snapshot possible without editing the instrument. Confirmed by re-running all 82
   under isolation: **80 reproduce, 2 fail** (the pair in point 2), and **not one artifact that
   reproduced ambiently broke under isolation** — the risk worth checking first was that a cold lake
   would turn real evidence into null evidence, and it did not, so nothing was re-recorded to reach 80.
2. **`cons_refute_strength_iso_120` and `exec_refute_marg_taskset_given_pfcbias` are not reproducible in
   principle**, and neither was re-recorded. Both are deterministic *given* a lake, and both report a
   `metacog reliability` counter that reads that mutable durable state: `48/120` recorded, `48/120` on an
   empty lake but with the reply count moving `0/30 -> 1/30`, and `50/120` against a frozen copy of the
   live lake; the Executive one reads `3/60`, `5/60` and `22/60` across the same three conditions.
   Re-recording would capture one more snapshot of a moving number and call it evidence, which is the
   failure this instrument exists to prevent. Stated as a known limit instead.
3. **The regex widening caused none of this.** Every artifact was scored under both the old and the new
   regex from the same execution: the same 70 pass and the same 12 fail either way, so no recorded
   artifact was invalidated by the change. It did add one catch — under an isolated `NORD_HOME`,
   `cons_refute_strength_iso_120`'s only divergence is a `N/M turns` line, which the old regex could not
   see and would have passed as reproducing.

## The state of this ledger, stated plainly

**It is a first pass, not the adversarially-verified sweep the rubric specifies.** The rubric's
procedure is one independent auditor per category plus a refuter for every score above 40
(`COVERAGE_RUBRIC.md:85-95`). What is here was populated in a single pass while building the
instrument. Treat the per-faculty bands as **provisional and re-adjudicable**; treat the *refusals* as
binding, because those are mechanical.

The roster is also incomplete: 30 faculty rows across eleven categories, 10 of them `UNSCORED`. Several
categories have a single row. A category with one row is not a measured category.

## `37.5%` is WITHDRAWN, and it is not replaced by this number

`COVERAGE_BASELINE.md` records an equal-weighted total of 37.5%. That figure is withdrawn as
**unreproducible**, for a reason that does not depend on anybody's judgment about bands:

> **There was not one recorded A/B transcript in the repository.** All 602 tracked files were searched:
> `BEHAVIOURAL CHARGE` appeared in three (this rubric, POLICIES.md, and `charge_probe.py` itself) and
> `act/hold decision changed` in one (the probe). The rubric forbids any band above 40 without a
> runtime A/B. **So the maximum re-derivable total was 40%** — and 37.5% sits at 94% of that ceiling,
> which requires nearly every faculty in the brain to sit at the very top of "live, not consumed" while
> the baseline's own text documents multiple rule-4 zeros (amygdala `threat()` constant,
> `region_wiring`'s empty registry, the reentrant-workspace key mismatch).

A category score is *defined* as the mean over its constituent faculties, and no per-faculty band table
was ever committed for any category, so **0 of the 11 recorded scores can be re-derived at all**. The
rubric asked each auditor for exactly that table (`:85-91`); the deliverable was produced and discarded.

**The number this ledger currently prints is therefore a floor on a partial roster, not a re-measurement
of the brain.** Do not compare it to 37.5% — the two are not on the same scale, which is precisely the
mistake `COVERAGE_RUBRIC.md:14-16` warns about for the earlier 22.8% and ~31% figures. The next
adversarial sweep populates these files properly and *that* becomes the first comparable baseline.

## The hole the instrument cannot close

**Roster padding.** The category score is the mean over rows the auditor chooses, so adding ten
trivially-live band-40 sub-faculties lifts a category from 20 to ~35 with no new capability. The scorer
cannot refuse this because the rubric never fixes the faculty roster. Closing it needs a **tracked,
fixed roster per category, changed only with a stated reason** — a rubric-level change, so it is flagged
here rather than quietly implemented.

Lesser known holes, all real: a chosen tick count (`drive 4` exits 1, `drive 60` exits 0 — same faculty,
same seed, opposite verdicts; the artifact records `n`, so it is visible but not refused); a long run
until one flip appears, since `charge_probe`'s contract is `flips > 0` (surfaced as W4); and a citation
that resolves to a line saying something else, which no cheap check can catch.

## What the ceiling looks like, so nobody plans past it

The rubric pre-declares that wiring into `cortex/brain.py::_route_percept_to_action` moves a three-way
arm decision and **cannot** raise Language, Imagination, Gf, Consciousness or WholeBrain. That is
**five categories excluded — 45% of the weight** at 9.09 points each — which move only on capability.

**Corrected 2026-07-29, twice, and both errors were mine.** This paragraph said "only 4 of the 11
categories are reachable", which does not add up: 11 − 5 excluded leaves **six**. The rubric explicitly
names four that wiring can raise (Executive, Affect, SpatialTemporal, action-relevant Perception) and
explicitly excludes five; **Memory and Social are simply unstated**, and quietly filing them under
"unreachable" made the arithmetic wrong in the direction that flattered the plan.

It also claimed a **48.5%** wiring ceiling. That number is **withdrawn**: it was computed from the
*retired* 37.5% per-category roster, and this file forbids comparing anything to 37.5% two sections
above. Quoting a ceiling derived from a withdrawn baseline is the same category error, one level up.

Re-derived from the evidence this instrument actually reports (Perception 23.3, Memory 20.0, Executive
26.2, Consciousness 5.0, Social 12.5, the rest 0.0): taking all **six** non-excluded categories to band
70 yields **38.6%**; taking only the four named yields **25.9%**. Both figures are **provisional on a
30-row partial roster** and will move once the ledger is populated properly — which is exactly why the
number belongs here, next to the evidence, rather than in a plan document as a headline.

The shape of the conclusion is unchanged and is the part that matters: **wiring cannot reach 100%, and
the five excluded categories need capability.** That is why `COVERAGE_TO_100.md` puts capability last
and largest, not first and cheapest.

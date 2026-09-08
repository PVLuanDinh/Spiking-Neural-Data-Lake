# S2 / T065 — the binding width, from measured state-per-unit (2026-09-07)

The row (plan Phase 3): *"**S2 = T065** the binding width from measured state-per-unit, recorded by
the EXISTING `substrate/perf_baseline.py --record/--assert` (T065's `tools/perf_baseline.py` wording
is stale: the p4c ratchet refused `tools/` at T001); 16k stays the ingest milestone only."*

T064's `UnitBlocks` deliberately refused to pick a block size, on the grounds that the right block
follows from state-per-unit and **this row is the one that measures it**. This is that measurement.

## 1. The naive instrument is CONFOUNDED, twice, and both were measured before anything was built

The obvious figure is `blob_bytes / units`. It is wrong in two independent ways, and neither is
visible from the ratio alone:

**(a) With width fixed, the ratio tracks the TRAJECTORY, not the units.** Ticking a scratch brain
200 → 1200 ticks:

| ticks | units | blob B | "B/unit" | trajectory rows |
|---|---|---|---|---|
| 200 | 2 | 5550 | 2775 | 5 |
| 400 | 2 | 5670 | 2835 | 10 |
| 800 | 2 | 5958 | 2979 | 20 |
| 1200 | 2 | 6214 | 3107 | 30 |

**The unit count never leaves 2.** The "per-unit cost" rises 2775 → 3107 B/unit while not one unit is
added — every byte of that movement is the trajectory tail lengthening. A per-unit figure that moves
when no unit is added is measuring something else and wearing the wrong name.

**(b) With width varying, the ratio charges the FIXED OVERHEAD to the units.** Holding ticks at 600
and forcing width with the admission cap (`_l_max`, with `_eps_tol` driven to zero so the cap
actually binds — the inertness measured in `evidence/sc-012a-width.md`):

| cap | units | blob B | "B/unit" |
|---|---|---|---|
| 1 | 1 | 5503 | **5503** |
| 2 | 2 | 5806 | 2903 |
| 4 | 4 | 6396 | 1599 |
| 8 | 8 | 7592 | 949 |
| 16 | 11 | 8533 | **776** |

A **seven-fold** spread in "bytes per unit" on one machine, one seed, one tick count — all of it
intercept being divided by different denominators.

## 2. The separating instrument: a slope, not a ratio

Hold ticks fixed, vary width, regress blob bytes on units:

- **MARGINAL cost: 302.01 bytes per unit** (the slope — what one more unit actually costs)
- **FIXED overhead: 5195.57 bytes** (the intercept — what a ratio blames on the units)
- **max |residual|: 19.62 B** on 5.5–8.5 KB fits — the relation is linear to well under a percent

The binding width then follows by division, and the BUDGET is reported as an input rather than
picked here (P1 — a budget is a property of the deployment, not of the brain):

| budget | binding width |
|---|---|
| 64 MB | 222,193 units |
| 256 MB | 888,825 units |
| 1024 MB | 3,555,352 units |

**The plan's 16k is confirmed as an ingest milestone and nothing more:** at 302 B/unit, 16,384 units
costs about **5.0 MB** — three orders of magnitude below the 64 MB binding width. Nothing in this
row's measurement makes 16k a constraint.

## 3. How it is recorded, and what was deliberately NOT disturbed

**Reused, not rebuilt (P2):** the existing `substrate/perf_baseline.py` carries it — the module the
row names — via `_measure_state_per_unit` and a `_score_state_per_unit` ratchet, with the metric
stored in the existing `tools/perf_baseline.json`.

**The first measurement establishes ONLY its own block.** A full `--record` would re-measure ticks
and iops and silently reset the reference the tick ratchet has asserted against since `S0-kernel` —
a side effect nobody asked for, and exactly the quiet re-baseline this module's `--stage` refusal
exists to prevent. So `--state-per-unit` writes `state_per_unit` alone; verified after the run:
`stage` still `S0-kernel`, `ticks.wake_ratio.p50` still `183.517`.

**The ratchet is NOT in the pre-push gate, and that is disclosed rather than implied.** The slope
needs five brains × 600 ticks — **332.7 s** measured. Folding that into `--assert` would put minutes
onto every push, against the standing invariant that anything observing the brain must be cheap
enough not to become the load. It binds when `--state-per-unit` is run; its absence from `--assert`
is stated here and in the module.

## 4. The red is REACHABLE, and shown to be

The ratchet's direction is that a per-unit cost may not RISE — a growing per-unit cost shrinks the
binding width at every budget at once. Scored against the recorded 302.01 B/unit baseline:

| arm | slope | verdict |
|---|---|---|
| lean (0.5×) | 151.0 | OK |
| at baseline | 302.0 | OK |
| within tolerance (1.10×) | 332.2 | OK |
| **FAT (2×)** | **604.0** | **REGRESSION** ← the red fires |
| absent | — | reported, never asserted |

The absent arm matters as much as the fat one: a baseline recorded before this metric existed, or a
run without the growth engine, is a MISSING MEASUREMENT and not a regression — the standing rule that
a zero is a claim about the instrument first. The bare `run_selfchecks` path exercises all three
behaviours on synthetic numbers, so a ratchet that stopped discriminating would red there rather
than pass quietly.

## 5. What this row does NOT claim

It does not claim the brain can reach any of these widths — growth on today's stream stops at 2 units
(measured above, and Phase 2's exit records that width past ~8 buys no accuracy). The binding width
is what the SUBSTRATE would cost, not what the brain has grown, and Phase 3 running at a constructed
width remains capacity ahead of consumption exactly as the plan's O5(f) discloses. It also does not
re-baseline tick performance, and does not put the new ratchet in the pre-push path.

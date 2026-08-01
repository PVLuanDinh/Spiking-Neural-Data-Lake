# Phase 0 Research: Grow the Live Populations

**Feature**: `specs/003-grow-the-live-populations` | **Date**: 2026-08-01
**Baseline commit**: `c6ca8c8d`. Every figure below was measured against that commit with a clean
working tree (`git diff --quiet HEAD -- src/snn_data_lake/cortex/time_cells.py brain.py` → clean).

> **Provenance rule adopted for this document.** An earlier research pass measured an *uncommitted*
> `decode()` rewrite sitting in the working tree while labelling it "today's decode". Every number
> here has been re-derived at `c6ca8c8d` with the tree verified clean first. Any figure that cannot
> name its commit is not carried forward.

---

## R1. The readout: which quantity can judge growth?

**Question**: FR-005 requires the judging quantity to depend on the population. F9 established that
today's `decode()` collapses to the ramp `v`. What replaces it?

### The two criteria a candidate must satisfy

A candidate readout is admissible only if it satisfies **both**:

- **C1 — degenerate-ladder control.** Run the module self-check (`assert final_err < 5.0`,
  `time_cells.py:250`) with `thresholds = [1.0 + 1e-6*i for i in range(64)]`, a ladder whose 64
  entries span 6.3e-5 and therefore carry no timing information. The candidate **must FAIL**. A
  candidate that passes is not reading its population, and Standing Invariant 5 — *"a green gate that
  structurally cannot see the thing it checks is worse than no gate"* — disqualifies it.
- **C2 — width monotonicity.** RMSE against true elapsed must fall as cells are added. A flat curve
  means growth is unmeasurable by construction.

C1 is the single most valuable artefact of this research. It is cheap, deterministic, and it
disqualified the most attractive-looking candidate.

### Decision table (MEASURED, 16 seeds, horizon 60, `c6ca8c8d`)

| readout | C1 degenerate | C2 width 64→512 | C3 RMSE @ n=64 |
|---|---|---|---|
| HEAD (ramp) | FAILS ✓ | 0.7276 → 0.7276 **flat** | 0.7276 |
| spike-tick anchor | **PASSES ✗** | 0.5152 → 0.2475 falls | 0.5152 |
| floor `thresholds[k-1]/drive` | FAILS ✓ | 1.5090 → 0.7924 falls | 1.5090 |
| **geometric-mid** | **FAILS ✓** | **1.0536 → 0.7586 falls** | **1.0536** |

### Decision

**Adopt the geometric-mid population readout.**

```
k == 0            -> 0.0
k >= n_cells      -> thresholds[-1] / drive
otherwise         -> sqrt(thresholds[k-1] * thresholds[k]) / drive
```

Given only the rank-order code — "cells `0..k-1` have fired, cell `k` has not" — the estimand is
bracketed by the two neighbouring nominal latencies, and the geometric mean is the natural centre of
a bracket on a geometrically tiled axis. It introduces no constant: every term is existing state.

**Rationale.** It is the only candidate satisfying C1 and C2. It reads the population and nothing
else, so growth and learning both move it.

**Cost, stated plainly.** +45% RMSE at the live width (1.0536 vs 0.7276), closing to +4% at n=512.
This is a real degradation of a live-consumed quantity and it is the price of making growth
measurable at all. It is reversible via the FR-016 off-switch.

### Alternatives considered

| Option | Rejected because |
|---|---|
| Keep HEAD's ramp readout | Fails C2 absolutely — RMSE is **identical** at 64, 128, 256 and 512. Growth is not merely unmeasured but provably worthless. Its only threshold-dependence is the saturation clamp, which is the FR-006 fake-capability channel. |
| Spike-tick anchor `spike_tick[k-1] + (v - thresholds[k-1])/drive` | **Fails C1**, decisively. MEASURED: on the degenerate ladder it scores err 0.995 where HEAD scores 49.000 — it passes a control it must fail. Cause: `spike_tick[i]` is assigned `self.t` (`time_cells.py:124`), which *is* ticks-since-reset, the estimand. The readout emits ground truth as its level and extrapolates. Its superb RMSE (0.4965 at t=50 vs HEAD's 0.6353) is an artefact of measuring against the simulator's own tick counter. **This candidate looked like a free win and is the reason C1 exists.** |
| Floor readout `thresholds[k-1]/drive` | Satisfies both criteria but is strictly dominated by geometric-mid on C3 (1.5090 vs 1.0536) — it sits at the bottom of the bracket rather than its centre, for no compensating benefit. |
| Population-mean over all fired cells | Threshold-sensitive but **flat in width** (0.8310 at n=8 → 0.8186 at n=512), so it fails C2. Geometric tiling packs most cells at low thresholds, so the mean is dominated by long-past crossings. |

### Two defect fixes that survive independently of the readout choice

Both are owed regardless and should be treated as such rather than bundled into the growth argument:

1. **`decode()` never divides by `drive`, while `learn_interval` does** (`:168`, `:182`). MEASURED at
   `drive=0.5`, true elapsed 40: HEAD reports **19.340**. The clock reports ramp units as if they were
   ticks. Latent today only because `brain.py:744` takes the `drive=1.0` default.
2. **The saturation clamp `estimate = base` (`:147`)** freezes the readout at the top threshold.
   MEASURED at n=64, t=600: RMSE **292.8993**. Live intervals reach 73 ticks against a top threshold
   of 64, so this regime is entered in normal operation. Note geometric-mid **also** clamps when
   saturated; range extension (R2) is what actually addresses it.

### Open

- Whether *any* readout can attribute a point-estimate improvement to how well the population **tiles**
  the interval, rather than to how recently *any* cell fired, is unresolved. Geometric-mid is admissible
  because it reads only the population — not because it is more accurate. This should be stated in any
  claim rather than smoothed over.

---

## R2. The growth rule

**Question**: FR-001 (bit-identical existing state), FR-003 (derived, no new constant), FR-012
(ascending order, asserted).

### The composition problem that forces the answer

**MEASURED, and it is the central constraint of this plan: append-at-top growth is unmeasurable below
saturation under *every* readout tested** — HEAD's, the spike-tick anchor, and the population
readouts. Paired-seed A/B at horizons 55 and 60: **ties 20 of 20, max |Δ| = 0.000000**.

The mechanism is structural, not incidental. A cell appended above `thresholds[-1]` never enters the
fired set for any ramp below the top threshold, so `k`, the fired set, and every function of them are
bit-identical between the grown and ungrown arms. No change confined to `decode()` can fix this: a
cell that has not fired is causally inert on any readout of the fired set.

This contradicts the spec's Assumptions (*"growth is assumed to be an append-at-the-top operation"*),
and the spec is wrong on that point.

### Decision

**Interior insertion, attempted only at a reset boundary.**

- **Interior**, because only a cell inside the traversed range can fire and therefore contribute.
- **At a reset boundary** (`n_fired == 0`, `v == 0.0`, `spike_tick` all `None` — the state `reset()`
  establishes at `:107-115`), because that is exactly where interior insertion's problems become
  vacuous. Away from a boundary, inserting at `j <= n_fired` requires either leaving `spike_tick[j] =
  None` — MEASURED to raise `TypeError` at `:154` inside `brain.py:1934`'s bare `except Exception:
  pass`, silently freezing the clock — or fabricating a crossing time, which MEASURED changes the
  live-consumed `sd` by +0.005256 at the instant of growth with zero ticks elapsed. At a boundary the
  fired set is empty, so neither arises.
- The brain already resets on salient novelty (`brain.py:1932`), so boundaries occur naturally.

**Placement**: the widest gap in log-space, split at its geometric midpoint —
`sqrt(thresholds[j] * thresholds[j+1])`. Derived from existing state; no constant.

**A second, separate mode — extend-at-top — is retained only to address saturation**, and any benefit
it produces MUST be reported as clamp-widening under FR-006, never as earning. Its threshold is
`float(n_cells + 1)`, the constructor's own top for the next width.

### Alternatives considered

| Option | Rejected because |
|---|---|
| Append at the population's own top ratio, `t[-1] * (t[-1]/t[-2])` | Recommended by research, **refuted by arithmetic**. MEASURED: it equals `64**(64/63)` = 68.36746821212175 — not `64**(65/63)` as claimed. The constructor puts width-65's top at **64.99999999999999**, so the rule displaces the top cell by **+3.367**, *larger* than the −3.082 the spec cites to condemn recompute-at-n+1 as destructive. Worse under repetition, which US4 requires: 64 successive growths give a top of **4375.52** against the constructor's **128.00** at that width, falsifying the class's own claim to tile `[1, n_cells]` (`time_cells.py:76`). Also seed-dependent to the point of unusability — on the live brain the appended value varies 67.94 / 68.37 / 74.83 across seeds 11 / 6 / 0. |
| Recompute the vector at n+1 | The spec's F2: destroys every learned threshold. |
| Interior insertion anywhere, any time | The `spike_tick` problem above. |

### Mandatory co-requisite: make `learn_interval` order-preserving

FR-012 is **not satisfiable by the growth path alone**, because the existing learning rule already
breaks ascending order on a live-shaped path. MEASURED: saturating no-reset regime, 4000 ticks, 3 of 6
seeds end with `t[63] < t[62]` (seed 3: `t[62]=75.505`, `t[63]=52.401`); reset-driven regime, 9 of 12
seeds inverted somewhere. This is the spec's F10 confirmed as *actual*, not merely reachable.

Fix by **refusing** an order-breaking update, not clamping it:

```
proposed = max(0.5, thresholds[i] + delta)
lo = 0.5 if i == 0 else thresholds[i-1]
hi = inf if i+1 >= n_cells else thresholds[i+1]
if lo < proposed < hi:  thresholds[i] = proposed     # else drop the update
```

Clamping was measured and rejected: it collapses gaps to **2.220e-16 (1 ULP)** versus **5.298e-03**
under refusal, with identical learning coverage (49.8/64 thresholds moved either way). A 1-ULP gap is
a denominator in any bracket readout. With refusal: 0 of 12 seeds inverted at 200/600/1200/4000 ticks.

This is a **bug fix owed regardless of growth** and should be sequenced first.

---

## R3. The held-out instrument

**Decision**: one tracked probe, `tools/timing_heldout.py`, readout-agnostic (takes `R(clock, tau)`
as a parameter), gated on self-checks it runs before it may report.

- **Interval universe derived from the population's own ladder**, not an absolute grid: for each of the
  `n_cells - 1` gaps place 10 points at `lo + (hi-lo)*(p+0.5)/10`; order gap-major; take a deterministic
  positional 1-in-5 stride (`index % 5 == 4` → TEST). MEASURED at n=64: 630 intervals → 504 teach / 126
  test, the 20% ratio ported from `ng_snn.py:178`, no RNG (so determinism semantics survive).
- **Which split semantics transfer (FR-009)**: the *discipline* transfers verbatim — teach on one
  portion, judge on a disjoint remainder, refuse when either is too small, never prune to nothing
  (`ng_snn.py:178-181`, `:200-201`). The *position* does not transfer onto the raw interval axis.
  MEASURED: a literal trailing-20% split leaves only **4 of 63** gaps able to move the metric; a stride
  over the raw integer axis gives 15 of 63; the gap-major grid gives **63 of 63**. Porting the position
  rather than the discipline would have produced a split that is real and blind simultaneously.
- **Metric**: held-out RMSE of `R(clock, tau)` against `tau`. Per-seed effect
  `d_s = RMSE(grown) - RMSE(ungrown)`; negative means the cell earned.
- **Self-test (FR-005 applied to the instrument)**: perturb one mid threshold and require `R`'s output
  to change; if it does not, print `NOT-ELIGIBLE` and exit 2. Under HEAD's decode this fails by
  construction, so the instrument refuses a verdict until R1 lands. That is correct behaviour.
- **Degenerate-ladder control (C1)** is run as an admission precondition on `R` itself.
- **Seeds: 12**, derived not conventional. MEASURED over 30 seeds: `mean_d = -0.019762`, `sd =
  0.022569`, so `(3.0*sd/|m|)^2` = 12 seeds for `|mean/se| >= 3.0`, 6 for the repo's ported 2.0 bar.
  Expected `|mean/se|`: 3 seeds → 1.52, 12 → 3.03, 30 → 4.80. **Three seeds is below even the 2.0 bar**,
  which independently confirms the spec's F12 criticism of the recorded 3-seed grow-on-top result.
- **Mandatory null-control arm (FR-006)**: insert a cell at the *lowest-teach-residual* gap. MEASURED
  mean +0.000000, improves 0 of 20. Its measured zero is what licenses reading the main arm as
  contribution rather than as a benefit available to any added cell.
- **Statistic**: port the arithmetic of `scratchpad/grow_on_top_nonstationary.py:130-137` (it is
  population-agnostic) but not the file — it is gitignored and `tools/citation_guard.py:320` forbids a
  tracked file citing an untracked `.py`. Two changes: print tie counts **unconditionally** (MEASURED 10
  of 30 paired differences are exactly 0.0; a `mean/se` hiding 33% zero-inflation overstates power), and
  print every count as `N of M`, never `N/M`, because the artifact fingerprint harvests bare ratios
  positionally.
- **Environment traps**: probes run with `PYTHONPATH=<root>/src` only, which drops the brain to a blind
  stdlib perception tier, and `NORD_HOME` is forced to a fresh temp dir, so a probe assuming accumulated
  durable state sees an empty lake. The harness must insert the repo root itself.
- **Ceiling, worth knowing before the work starts**: `record` writes no `metric:`/`baseline:` keys, so
  F10 refuses any band > 80 for a freshly recorded `kind: heldout` artifact. The honest deliverable of
  this feature may be a correct instrument, a real number, and a ledger row that moves from 35 to 40.

---

## R4. Persistence

**Decision**: **do not put the clock's state in the lake.** Persist `{fmt, n_cells, seed, drive,
noise_sd, thresholds}` via the existing `home.save_state`/`load_state` atomic JSON blob.

**Rationale.** The lake's own owning file states the rule — `home.py:41-42`: *"Weights/calibration stay
separate STATE (save_state), never in the memory lake."* A threshold vector **is** the learned weight
set (F3). Three independent lines converge:

- **Extending `_T_GROWTH` is a measured live regression.** After appending a clock-shaped npz under that
  tag, `load_growth()` returns the clock payload and `restore_growth_state()` returns **False**, swallowed
  by `brain.py:984 except Exception: pass` — the NG-SNN silently starts ungrown. Both writers sit in the
  same `if self._persist:` block; whichever appends last wins.
- **A second npz key set is strictly worse**: `growth_state_bytes()` returns `None` when `_model is None`
  (`neurogenesis_engine.py:255-256`), so on the `NORD_TORCH=0` CI job the clock's state would have no
  carrier at all.
- **Bounding (FR-014).** MEASURED on the real 187 MB lake: **180.0 of 187.5 MB (96.0%)** is `_T_GROWTH`
  — 25,502 records averaging 7,054 B — against 7.47 MB of facts. FR-014 forbids continuous growth writing
  to a store with no prune policy, and that policy (`specs/002` FR-002) is unbuilt. With `save_state`,
  **overwrite is the prune policy**, so slice 4 stops depending on unbuilt work and no retention constant
  is introduced (P1).

**Restore cost nobody had priced**: `load_growth()` constructs a `BinLog`, which materializes every
payload into a dedup set (`binlog.py:45-47`). MEASURED on the real lake: **0.400 s and 187.5 MB resident,
on every Brain construction**. `load_state` of a small JSON blob is **0.110 ms** — 3,600× cheaper and
O(1) in lake size.

**Reconciliation rule**: **adopt persisted, wholesale** — rebuild the object at the persisted width and
assign the persisted vector. Never mutate `n_cells` in place (`reset()` sizes `fired`/`spike_tick` from
it), never merge element-wise, never truncate or pad. `IGNORE` is rejected because the constructed 64 is
exactly the magic constant this feature exists to remove; `TRUNCATE/PAD` is rejected because F3 leaves no
way to reconstruct learned state; `raise` is rejected because `brain.py:749`/`:1934` would swallow it.

**REFUSE and record the refusal** (FR-011, Standing Invariant 3) on: fmt mismatch, `len(thresholds) !=
n_cells`, seed/drive/noise_sd mismatch, `n_cells < 1`, or a non-ascending vector. Recording
seed/drive/noise_sd costs 24 bytes and is what makes refusal possible at all: `learn_interval` sets
`t_pre` from `spike_tick` under *that* RNG's noise, so a vector learned under another seed is another
lineage's state.

**Hook site**: at the Brain level immediately after the construction block, in its **own** `try` that does
not share the `brain.py:736-752` handler — that one nulls five populations (F6). Never inside
`TimeCellClock.__init__`, or `cortex.time_cells`'s own self-check reads the previous gate run's state.

**Correction to the brief**: `_persist` is **TRUE** in both gates at HEAD, so the restore does run there;
isolation, not the persist flag, is what keeps them deterministic.

---

## R5. Observability

**Decision**: export `time_cells_width`, `time_cells_grown`, `time_cells_refused` from `state()`,
following existing naming.

**Rationale.** FR-011 and SC-007 are unsatisfiable by any instrument in the tree today: `n_cells` appears
exactly once in `brain.py` (at `:744`), and `state()` exports no width, no thresholds, no growth event.
`ab_method.py` reads `state()` only, so the canonical neuter-A/B is structurally blind to growth — a
verdict from it would be uninformative rather than negative. The precedent is already in the tree:
`neurogenesis_engine.py:189` writes `self._pruned` and nothing anywhere reads it.

A refusal counter is not optional decoration: it is what makes SC-007's halt *observable* rather than
inferred from absence, which is the difference between a null and silence.

**Risk to check during implementation**: adding keys to `state()` changes any digest or golden comparison
built over it, including `ab_method.py`'s behavioural digest. Enumerate consumers before adding.

---

## Cross-cutting: execution order must differ from the spec's priority order

The spec numbers its slices US1 (growth) → US2 (readout) → US3 (measure) → US4 (autonomous). **The
readout must be built first.**

MEASURED at HEAD's decode, append-at-top, 20 paired seeds:

| horizon | ties | max &#124;Δ elapsed&#124; | changed ticks |
|---|---|---|---|
| 55 | 20 of 20 | 0.000000 | 0 |
| 63 | 15 of 20 | 1.497230 | 2 |
| 70 | **0 of 20** | **4.367468** | 9 |
| 120 | 0 of 20 | 4.367468 | 59 of 120 |

`decode()` reads population **width** directly at `:143` (`if k < self.n_cells`), so appending flips a
saturated read out of the clamp branch into interpolation — *firing is not required*. That value feeds
`brain.py:1561` and the `elig["timing"]` drive-weight update at `:1574`. So a slice-1 demonstration on a
live brain produces up to 4.37 ticks of behaviour change that is **pure clamp-widening — exactly the
FR-006 artefact**. And slice 2 changes its magnitude (4.367 → 1.633), so slice-1 evidence recorded before
slice 2 is invalidated by slice 2.

Since the readout work is a defect fix owed regardless, sequencing it first costs nothing.

**Execution order**: R2's `learn_interval` order fix → R1 readout → growth → instrument → persistence and
observability → autonomous growth.

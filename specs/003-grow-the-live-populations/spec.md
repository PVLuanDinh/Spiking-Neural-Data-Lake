# Feature Specification: Grow the Live Populations (Autonomous LIF Growth)

**Feature Directory**: `specs/003-grow-the-live-populations`

**Created**: 2026-07-31

**Source**: Owner directive, 2026-07-31. Governed by `docs/POLICIES.md` (P0–P6), which is the
authoritative policy text; `.specify/memory/constitution.md` is its governance surface and defers to it
by its own rule — *"Where the two ever disagree, `docs/POLICIES.md` wins and this file is the defect"*
(`constitution.md:36–38`). The Standing Invariants (`constitution.md:101–114`) apply throughout.
Follows `specs/002-capability-and-bounding`. That feature's unbuilt spiking prune policy was originally a
dependency of this one's final slice; Phase 0 removed it by persisting outside the lake — see Dependencies.

**Status**: Draft — ready for `/speckit-plan`. Q1 resolved 2026-07-31 (see Status).

## Problem

The brain grows in exactly one place, and it is not the live tick.

`NeurogenesisSelfImprove` (`cortex/neurogenesis_engine.py:21`) grows an NG-SNN — a genuinely
contribution-grown LIF network — retains its substrate across restarts through a `_T_GROWTH` record in
the one lake (`substrate/home.py:62`), and restores it at brain construction (`cortex/brain.py:979–985`).
That path works.

Every population that actually runs each tick is a fixed width someone typed:

| Population | Width | Site |
|---|---|---|
| `TimeCellClock` | `n_cells=64` | `cortex/brain.py:744` |
| `learned_evc` | `n_cells=40` | live tick |
| `metacognitive_monitor` | `pop_size=24` | live tick |
| `second_order_access` | `pop_size=24` | live tick |
| `neuronal_avalanches` | `n=1000` | live tick |

Two policy failures at once: **P1**, a magic constant kept fixed where a computed value belongs, and
**P0**, an available avenue of self-improvement not taken.

**The reason this has not been done is not that growing is hard.** A paired grow-on-top A/B was a
clean null: more units, no accuracy benefit. Capacity added where nothing rewards capacity is dormant
capacity, and P0's own guard says a grown unit that changes no outcome is not improvement.

**And for the chosen first target the situation is worse than "nothing rewards capacity": nothing can
even register it.** Below saturation `TimeCellClock`'s readout is algebraically independent of its own
threshold vector (F9). A 65th cell cannot change the brain's timing estimate on 92–96% of the ticks the
clock runs. So this feature's first real question is not how to grow, and not whether growth pays —
it is **whether the population is wired so that growth could pay at all.**

## Findings already in hand (verified against source at `c6ca8c8d`, 2026-07-31)

**F1 — `TimeCellClock` is genuinely live, and has three consumers.** Constructed at `cortex/brain.py:744`
(`TimeCellClock(n_cells=64, seed=self._seed)`, function-local import at 741), ticked every live tick at
`brain.py:1933`, taught on reward onset at `brain.py:1562` (`learn_interval(self._learned_latency, da)`).
Output read by the timed Go window (`brain.py:1559`), telemetry (`brain.py:2797`), and the chat
elapsed-time answer (`brain.py:3197`).

**F2 — Growth is DESTRUCTIVE as the population is currently constructed.** `cortex/time_cells.py:100–103`:

```python
span = math.log(float(n_cells)) if n_cells > 1 else 0.0
self.thresholds = [
    math.exp(span * (i / (n_cells - 1))) if n_cells > 1 else 1.0 for i in range(n_cells)
]
```

so `thresholds[i] == n_cells ** (i / (n_cells - 1))`. **Both the span and the exponent's denominator
depend on `n_cells`**, so rebuilding the vector at width 65 moves every existing threshold:

| Cell | Width 64 | Width 65 | Shift |
|---|---|---|---|
| 32 | 8.268 | 8.062 | −0.206 |
| 63 (top) | 64.000 | 60.918 | **−3.082** |

The top cell's displacement, 3.082, is **more than twelve times the largest learned displacement
observed on a live brain** (0.2472 over 600 ticks, 86 `learn_interval` calls), and unlike the mid-cell
figure it cannot be overtaken by longer running — a learned delta is bounded per call at
`|delta| <= lr * |reward|` with `lr = 0.1`. Growth implemented as "recompute the vector at n+1" does
not add a cell; it erases the population and adds a cell.

**F3 — The threshold vector IS the learned state.** `learn_interval` writes in place:
`self.thresholds[i] = max(0.5, self.thresholds[i] + delta)` (`time_cells.py:193`). No separate
learned-delta array, no baseline copy, no way to reconstruct what was learned once the vector is
recomputed. F2 and F3 together are why FR-001 exists.

**F4 — `TimeCellClock`'s learned state is not persisted at all.** Rebuilt from `seed` at `brain.py:744`
on every construction. Nothing writes `thresholds` to the one lake. The daemon relaunches on every new
commit — **~20 times in a working session** by the project's own count (`brain.py:965`) — so today every
threshold the clock learns is discarded within the hour.

Nor is there a slot to put it in. The growth record's payload is an uncompressed `.npz` under format
string `ngsnn.growth.v1` with exactly nine keys — `fmt, W, b, beta, d_in, m_out, n_units, acc, ticks`
(`cortex/neurogenesis_engine.py:248`) — shaped entirely around the NG-SNN's grown weights and ridge
readout. A threshold vector has nowhere to go in it.

**F5 — The existing contribution criterion cannot be called on `TimeCellClock`, and it is less
"derived" than the handoff recorded.** Admission (`ng_snn.py:344–350`) scores a candidate by the squared
correlation between its firing-rate response and the current per-class residual `E = Y[:n_tr] - S[:n_tr]
@ beta` (`ng_snn.py:375`) — genuine error reduction, but it presupposes targets `Y` and a solved readout
`beta`. Retention is `prune_unearned(X, y, tol=0.0)` (`ng_snn.py:146`). Both need a supervised row set
and a re-solvable ridge readout; `TimeCellClock` has a ramp, a threshold vector, and `decode()`.
**What transfers is the standard, not the function.**

Correcting the record: what is *derived* is `l_max`, which defaults to `n_tr`, the number of rows the
solve actually fits (`ng_snn.py:325`, `:65`). The prune tolerance is a default parameter, `tol=0.0`.
Many constants in that path are plain literals — `r=0.9`, `patience=48`, `eps_tol` (`0.28` at the
engine, `neurogenesis_engine.py:104`), `reg_eps=1e-6`, `t_cand=40`, `lambdas=(0.1,1.0,5.0)`, the
`0.2`/`8` split fractions, a `1e-4` improvement epsilon and a `1e12` rank bonus. "Tolerance is derived"
should not be repeated.

**F6 — The construction path fails silently.** `brain.py:749–752` catches every exception and sets
`_time_clock` **and four other populations** to `None`; `brain.py:1934` is a bare `except Exception:
pass` around the tick. A growth path that throws would not fail loudly — it would silently delete five
populations and keep running.

**F7 — The lake half this would write to is the unbounded half.** `_T_GROWTH` records are append-only
and *"the newest `_T_GROWTH` record IS the current substrate"* (`substrate/home.py:196–205`), with no
prune path, and `load_growth` is a full linear scan of the lake on every construction
(`home.py:214–218`). `specs/002-capability-and-bounding` FR-002 specifies that prune policy and it is
**not built**.

**F8 — No held-out timing metric exists, and both things that look like one are in-sample.** The
module self-check takes `final_err = abs(estimates[-1] - 50)` on the single trial that produced the
estimates (`time_cells.py:199–205`). The pytest test
`test_learn_interval_reduces_reproduction_error` (`tests/test_time_cells.py:32–43`) is worse than
in-sample: it measures `|thresholds[i]/drive − target|`, a **parameter-space distance, not a timing
error**, and the clock is never ticked between the 30 `learn_interval` calls — nothing about interval
reproduction is measured at all. A repo-wide search for `held.?out` finds no hit in `cortex/time_cells.py`,
and only 2 of the 85 recorded artifacts carry `kind: heldout`. Honest null: **the instrument must be
built.**

This project has already been burned by exactly this shape. Before NG-SNN's held-out split was made
real, its stop scored rows the solve had already fit, so it never fired: *"it grew 600 units to fit 600
RANDOM labels and reported that as earned"* (`ng_snn.py:226–231`), with the regression control now at
`tools/capacity_curve.py:51–54`. Standing Invariant 5 — a gate that structurally cannot see what it
checks manufactures confidence — is the governing rule for slice 2.

**F9 — THE BLOCKING FINDING: below saturation, the elapsed estimate is algebraically independent of the
threshold vector.** `time_cells.py:142–147`:

```python
base = self.thresholds[k - 1]
if k < self.n_cells:
    nxt_frac = max(0.0, min(1.0, (self.v - base) / (self.thresholds[k] - base)))
    estimate = base + nxt_frac * (self.thresholds[k] - base)
```

The latch loop at `:122` guarantees `thresholds[k-1] <= v < thresholds[k]` whenever `0 < k < n_cells`,
so the clamp is inert and the expression collapses to `estimate = base + (v − base) = v`. At `k == 0`
it is `v / thresholds[0]`, and `thresholds[0]` is exactly `1.0` by construction — also `v`.

Measured directly: over 63 unsaturated ticks at `n_cells=64, seed=0`, **max |estimate − v| = 0.0**.
Perturbing `thresholds[30]` by `+0.4` (the size of several learned updates) leaves the estimate sequence
**bit-identical**; only `sd` moves (0.833165 → 0.845130). Appending a 65th cell above the top threshold
leaves 60 unsaturated ticks **bit-identical**, and differs only past saturation, where the ungrown clock
clamps at exactly `64.0` and the grown one reads `68.367`.

On the live brain (600 ticks, seeds 0/1/2), **92.5–95.8% of clock-advancing ticks are unsaturated.**

So the population is degenerate in both directions:

- **Unsaturated (>92%)**: a held-out timing A/B on the elapsed estimate must return exactly `0.0`, every
  seed, *by algebra*. That is not a null about growth; it is an instrument that cannot move.
- **Saturated (4–8%)**: the readout is a clipping ceiling frozen at the top threshold. *Any* added cell
  reduces error on long intervals for the same reason widening any clamp does — no contribution, no
  earning, no competition. A "benefit" verdict from this regime is the exact shape of a fake capability
  under P0.

`sd` (`time_cells.py:154–155`) *is* threshold-dependent and is the one honest population-sensitive
channel the class currently exposes.

This also re-explains the recorded null in `docs/coverage/artifacts/time_cells_ab_eprop.txt`
("CHANGED NOTHING") as **structural**, not merely stale.

**F10 — Ascending threshold order is not an invariant today; learning alone can break it.** The latch
loop (`:122`) walks upward from `n_fired` and stops at the first uncrossed threshold, so an inversion
makes every higher cell permanently unreachable, silently. The only clamp on a learned threshold is a
floor (`:193`). Measured: the minimum inter-threshold gap is **0.0682** (cells 0–1), while a single live
update reached **|delta| = 0.1811** — 2.7× that gap — and `|delta| <= 0.1 * |reward|` with `reward` an
unbounded RPE (max observed 1.9936). Worse, `decode()` divides by `(thresholds[k] − base)` at `:144`, so
a zero or inverted gap raises `ZeroDivisionError` inside `brain.py:1934`'s bare `except Exception: pass`
— the clock would freeze at its last estimate rather than fail. This is a **pre-existing defect the
feature inherits**, not one growth introduces.

**F11 — Nothing in the brain can observe the population's width.** `n_cells` appears exactly once in
`cortex/brain.py` — at `:744`, the constructor. `state()` exports `learned_latency` (`:2785`),
`elapsed_time` (`:2796`) and `timing_sd` (`:2799`) — no width, no thresholds, no growth event. The
canonical neuter-A/B harness `docs/coverage/artifacts/ab_method.py` reads `state()` only, so it is
**structurally blind to growth**; a verdict from it would be uninformative rather than negative. The
repo has already burned here: `neurogenesis_engine.py:189` writes `self._pruned` and nothing in `src/`,
`tools/`, `tests/` or `learn_daemon.py` ever reads it.

**F12 — The prior grow-on-top result is weaker evidence than the handoff implies.** The only paired
harness is `scratchpad/grow_on_top_nonstationary.py`; its statistic is a hand-rolled paired mean/sd/se
with a `|mean/se| >= 2.0` rule of thumb, explicitly not a p-value, computed at two measurement points.
The 10-seed run and its cited `|mean/se| = 0.44` are **not recorded anywhere in the repo** — what is
recorded is an earlier **3-seed** result, in the commit body of `f459c40a`. The harness also nulls
`_model` before each refit, which isolates seeding and **not** pruning, and its own closing line states
that unit cost is unmeasured. Additionally, `scratchpad/` is gitignored and `tools/citation_guard.py` is
an asserting CI gate: a tracked file may not cite an untracked `.py` as a runnable reproduction path, so
this harness **cannot be cited by any ledger row**.

**F13 — The existing TimeCellClock artifact is stale and its ledger write-up is wrong at HEAD.**
`time_cells_ab_eprop.txt` was recorded at `6acb892f`, **9 commits before HEAD**, predating the three
commits that made `learn_interval` able to move at all (`df2874d9`, `1747d357`, `c6ca8c8d`).
`docs/coverage/README.md:239` still quotes the retired integer ladder `[1.0, 2.0, 3.0, 4.0]`, which no
longer exists. Separately, the `interval_timing_time_cells` ledger row is banded 35 and explicitly rests
on the interval *timer*, not on `learn_interval` — a growth result would have to move that row on its
own evidence.

## User Scenarios & Testing

Four slices. Each is independently testable, and slices 2 and 3 are genuine stopping points — either may
return a result that forbids what follows, and that is success, not failure.

### User Story 1 — One population can gain a cell without losing what it learned (Priority: P1)

The operator grows `TimeCellClock` from 64 to 65 cells on a brain that has already learned, and every
threshold it had learned is still there afterwards. Nothing about the *decision* to grow is in scope —
only that growing is survivable.

**Why this priority**: F2 shows the obvious implementation destroys the population, and F9 shows no
measurement can run until a growth mechanism exists at all (`TimeCellClock` has no add/prune method
today). This slice is the precondition for everything else.

Building it before slice 3 has returned a verdict is explicitly permitted: *"Scaffolding a capability
the project does not have yet is PERMITTED, and is not a fake capability — owner decision, 2026-07-30.
This guard forbids reporting a capability without a measurement. It does not forbid building the channel
one will need. … Build the channel, then fill it, then measure it — and do not credit it until the third
step"* (`docs/POLICIES.md:60–71`). Slice 1 is the channel. It must be labelled infrastructure and claim
no capability.

**Independent Test**: Teach a clock, snapshot its thresholds, grow it by one, compare. No growth policy
and no measurement rig required.

**Acceptance Scenarios**:

1. **Given** a clock at width 64 whose thresholds have been moved by `learn_interval`, **When** one cell
   is added, **Then** all 64 pre-existing thresholds are bit-identical to their pre-growth values.
2. **Given** a clock at width 64, **When** one cell is added, **Then** the resulting threshold vector is
   strictly ascending, and this is asserted rather than assumed — per F10 sortedness is not currently
   guaranteed even without growth.
3. **Given** a clock mid-interval with a partially risen ramp, **When** growth is attempted, **Then** it
   is **refused** and no state changes. Away from a reset boundary an inserted cell below the ramp would
   have to carry a fabricated crossing time — measured to shift the live-consumed `sd` by +0.005256 with
   zero ticks elapsed — or leave it unset and raise inside a bare `except`. Refusal is the design, not a
   limitation.
4. **Given** the new cell's threshold, **When** its value is traced to its inputs, **Then** it is a
   function of the existing thresholds alone and introduces no new tuned constant (P1).
5. **Given** growth has occurred, **When** the brain's exported state is read, **Then** the population's
   current width is visible in it — per F11 nothing can observe width today.

---

### User Story 2 — The population's own output can register the population (Priority: P2)

The operator can point at a quantity the brain actually consumes and show that changing the threshold
vector changes it. Today they cannot: per F9 the elapsed estimate is the ramp, and 92–96% of live ticks
sit in the regime where the entire population is causally inert on it.

**Why this priority**: This is the blocking slice, and it was invisible until measured. Slice 3 cannot
return a meaningful verdict while the measured quantity is independent of the thing being grown — the
A/B would return exactly 0.0 by algebra, and the only regime where it moves rewards clamp-widening
rather than contribution. Standing Invariant 5 forbids proceeding on an instrument that structurally
cannot see its subject.

Per the resolved Q1 this slice repairs `decode()` rather than substituting a different measured quantity,
which makes it a **defect fix owed regardless of growth**: a 64-cell population whose readout discards 63
of its cells is computed-but-not-consumed. It is therefore independently shippable and is the first slice
that changes live behaviour — so it carries its own before/after evidence and its own risk.

**Independent Test**: Perturb a mid threshold and show the consumed quantity moves. Delivers a
population-sensitive readout independent of any growth.

**Acceptance Scenarios**:

1. **Given** a clock in the unsaturated regime, **When** one threshold is perturbed, **Then** the
   quantity the brain consumes changes — the current behaviour, where the estimate sequence is
   bit-identical, is the defect this slice closes.
2. **Given** the live brain, **When** the fraction of clock-advancing ticks in each regime is measured,
   **Then** it is reported, so any later claim states which regime produced it.
3. **Given** a proposed population-sensitive quantity, **When** it is adopted, **Then** it is shown not
   to be a clipping-ceiling artefact — a benefit available to any added cell regardless of contribution
   is disqualified.
4. **Given** the change, **When** the gates run, **Then** the module's Weber/scalar-timing self-check and
   the same-seed determinism test still pass without being loosened.

---

### User Story 3 — The added cell is measured, and may be found worthless (Priority: P3)

The operator asks whether cell 65 earned its place against the population-sensitive quantity from slice
2, on intervals the clock was not taught on. The answer is a number with a control, and "no" is a
permitted answer recorded as a null.

**Why this priority**: This is the point of the feature, and P0's guard — a capability reported without
measurement is fake — makes it the gate rather than a follow-up. It is third because slices 1 and 2 are
both preconditions.

**Independent Test**: Paired-seed A/B of grown vs ungrown clock on held-out intervals.

**Acceptance Scenarios**:

1. **Given** a grown and an ungrown clock from the same seed given the same teaching intervals, **When**
   both are evaluated on intervals withheld from teaching, **Then** an effect and its dispersion across
   seeds are reported.
2. **Given** the paired A/B, **When** the effect is smaller than its own dispersion, **Then** it is
   recorded as an explicit null and slice 4 does not proceed.
3. **Given** any reported improvement, **When** it is published, **Then** it names the held-out split,
   the control it beat, and the saturation regime it was measured in.
4. **Given** a unit that fails the criterion, **When** the decision is applied, **Then** the unit is
   refused or removed and the population returns to its pre-growth width with its learned state intact.
5. **Given** the held-out split, **When** it is sized, **Then** it is shown to contain enough data for
   the criterion to function — the prior harness recorded a failed first run where ~5 held-out rows per
   class made the earn-your-place stop unable to see any gain, which is a precondition, not tuning.

---

### User Story 4 — The brain grows itself, and can be told no (Priority: P4)

The brain adds cells on its own during operation, without an operator asking, stops when added cells stop
earning their place, and its growth survives a daemon restart.

**Why this priority**: The owner's actual goal, and last because it is unsafe before the rest. Growth
that cannot be refused is unbounded accumulation — phase 7's problem with a new source (F7). Gated on
slice 3 returning a positive result.

**Independent Test**: Run the daemon across a restart; confirm grown width and learned thresholds both
persist, and that a sustained no-benefit regime stops growth.

**Acceptance Scenarios**:

1. **Given** a brain that has autonomously grown its clock, **When** the daemon is restarted, **Then**
   the population returns at its grown width with its learned thresholds, not at the seeded default.
2. **Given** a regime where added cells stop earning their place, **When** growth is attempted, **Then**
   growth halts and the halt is observable in exported state rather than inferred from absence.
3. **Given** continuous operation, **When** the durable footprint is measured, **Then** a growth rate per
   unit time is reported and stated as bounded or unbounded.
4. **Given** growth is disabled, **When** the brain runs, **Then** live behaviour is indistinguishable
   from today's.

### Edge Cases

- ~~**A cell is added while the ramp is mid-interval.**~~ **RESOLVED BY DESIGN.** `n_fired` is an index
  into a sorted vector, so a cell admitted below the ramp would corrupt it. Growth is therefore attempted
  only at a reset boundary, where the fired set is empty and the problem is vacuous; mid-interval growth
  is refused. See [research.md](./research.md) R2.
- **Learning inverts two thresholds.** Per F10 this is reachable today: a single update can exceed the
  smallest inter-threshold gap, which silently strands every higher cell and can raise
  `ZeroDivisionError` inside a bare `except`.
- **Growth is attempted at width 1.** The constructor special-cases `n_cells == 1`; the growth rule must
  be defined there rather than dividing by `n_cells - 1`.
- **Cell 0 remains unlearnable.** Its threshold is exactly `1.0` by construction, so a target of 1 tick
  yields a zero error term. Growth must not be reported as fixing this.
- **Growth code raises.** Per F6 the enclosing `try` would silently null five populations.
- **A restored width disagrees with the constructed width.** Restore must reconcile without silently
  truncating learned state or discarding the record.
- **Two brains, one lake.** Restore reads a file a live daemon is concurrently appending to; this already
  made brain construction non-deterministic in ~42% of trials once (`brain.py:971–976`).
- **The population's own self-check is a CI gate.** `cortex.time_cells` is registered in
  `run_selfchecks.py:105`, and its `main()` asserts Weber/scalar timing. A growth rule must keep that
  true rather than be exempted, and the assertion must not be loosened (Standing Invariant: never silence
  a check to make it pass).
- **Determinism.** `tests/test_time_cells.py:46` asserts two same-seed clocks produce identical output.
- **The recording harness must be tracked, not in `scratchpad/`.** `scratchpad/` is gitignored and
  `tools/citation_guard.py` asserts that no tracked file cites an untracked `.py` as a runnable
  reproduction path (F12).
- **Probe environment differs from the daemon's.** Recorded probes run with `PYTHONPATH=<root>/src`
  only, which drops the brain to a blind stdlib perception tier, and `NORD_HOME` is forced to a fresh
  temp dir — so a probe assuming accumulated durable state sees an empty lake.
- **Artifact fingerprinting is positional.** The reproduction check harvests every bare `N/M` ratio in a
  probe's stdout; adding one ratio to a shared harness changes every artifact's fingerprint. Report
  counts as `N of M`.

## Requirements

- **FR-001**: Growth MUST NOT alter any already-learned parameter of the population it grows. Adding a
  unit MUST leave every pre-existing unit's learned state bit-identical.
- **FR-002**: A unit MUST be admitted or kept only on evidence that it reduces error on data the
  population was not taught on. Activity, movement, or magnitude MUST NOT be substituted as a proxy for
  contribution.
- **FR-003**: Every parameter of a newly added unit MUST be derived from the population's existing state.
  No new tuned constant may be introduced by the growth path (P1). A derived initial value satisfies P1,
  which forbids a magic constant sitting where a computed value belongs (`docs/POLICIES.md:101–106`); it
  is only the unit's starting point and becomes a learned one once the population's rule moves it.
- **FR-004**: Growth MUST be refusable. Every admission path MUST have a rejection path returning the
  population to its prior width with its learned state intact, and MUST NOT be able to prune to nothing.
  P0's guard is explicit that this is part of growing: *"a grown unit that is not KEPT is not even
  growth"* (`docs/POLICIES.md:72–75`).
- **FR-005**: The quantity used to judge contribution MUST be demonstrably dependent on the population
  being grown, in the regime where the measurement is taken. A measurement whose value is independent of
  the grown parameters MUST NOT be used to issue any verdict, positive or null.
- **FR-006**: A benefit that would accrue to any added unit regardless of its contribution MUST NOT be
  reported as earning. Where the only available effect is of that kind, the result MUST be reported as
  an artefact of the readout, not as growth.
- **FR-007**: Grown state MUST survive process restart, or growth MUST NOT be enabled. A population that
  grows but resets on restart MUST be reported as not-yet-growable rather than as grown.
- **FR-008**: Continuous, brain-initiated growth MUST NOT be enabled until a single added unit has
  demonstrably earned its place under FR-002. A null result MUST block it.
- **FR-009**: Exactly ONE earn-your-place standard may exist in the repo, and it MUST be ported
  accurately. The existing standard solves the readout on one portion of the rows and judges on a
  disjoint remainder, refuses to act when either portion is too small to be meaningful, and never prunes
  to nothing (`ng_snn.py:178–200`); its split is **positional**, which for interval data is a temporal
  holdout with different statistics from a random one. The plan MUST state which split semantics
  transfer. A second competing standard MUST NOT be introduced (P2).
- **FR-010**: The measurement MUST be able to return a null, and a null MUST be recordable as the
  feature's outcome. Reporting growth as a capability without a held-out number is forbidden by P0.
- **FR-011**: The population's current width, and each growth or refusal event, MUST be observable in the
  brain's exported state. A quantity that is computed and never read does not satisfy this.
- **FR-012**: The growth path MUST maintain ascending threshold order and MUST assert it, including
  against updates made by the population's existing learning rule.
- **FR-013**: A failure anywhere in the growth path MUST NOT silently degrade unrelated live populations,
  and MUST NOT be swallowed into a state indistinguishable from growth having been declined.
- **FR-014**: Any durable record written by growth MUST have its growth rate measured and stated, and MUST
  be declared bounded or unbounded. Growth MUST NOT be made continuous while writing to a store with no
  prune policy.
- **FR-015**: Any experiment supporting a claim MUST be reproducible from a tracked path, and MUST record
  the saturation regime and held-out split it was measured under.
- **FR-016**: The feature MUST have an off-switch under which live behaviour is unchanged from the
  pre-feature baseline.

### Key Entities

- **Live population**: a set of spiking units instantiated at brain construction and advanced every tick,
  whose width is currently a literal. `TimeCellClock` is the first instance in scope.
- **Learned state**: the parameters a live population modifies in response to reward. For
  `TimeCellClock` this is the threshold vector itself, which is also its structural definition — the
  reason growth and learning collide here.
- **Consumed quantity**: the population output the brain actually reads and acts on. Distinct from any
  quantity the population computes; per F9 the two currently differ.
- **Admission evidence**: a held-out error measurement on a consumed, population-dependent quantity,
  paired against a control, deciding whether a unit is kept.
- **Growth record**: the durable entry letting a grown population return at its grown width after restart.

## Success Criteria

- **SC-001**: After adding a unit to a population that has already learned, 100% of pre-existing learned
  parameters are unchanged, verified by direct comparison against exported state rather than by inspection.
- **SC-002**: Perturbing a single threshold produces a measurable change in the quantity the brain
  consumes, in the regime covering the majority of live ticks. Today that change is exactly zero.
- **SC-003**: The proportion of live clock-advancing ticks in each saturation regime is measured and
  reported, and every subsequent claim names the regime it was measured in.
- **SC-004**: A single added unit has a held-out effect on a population-dependent quantity, reported with
  its dispersion across a pre-declared number of paired seeds, and a stated verdict of benefit or null.
- **SC-005**: Where the verdict is benefit, it is stated against a named control and a named held-out
  split, and shown not to be available to an arbitrary added unit; where it is null, the null is recorded
  with the same rigour and the feature stops.
- **SC-006**: A grown population returns at its grown width, with its learned parameters, across a process
  restart — or growth is documented as not-yet-persistable and remains disabled.
- **SC-007**: Population width and growth events are readable from exported state, and a halt in growth is
  visible there rather than inferred from absence.
- **SC-008**: The durable footprint attributable to growth has a measured rate per unit time and a stated
  bounded-or-unbounded verdict.
- **SC-009**: With growth disabled, live behaviour is unchanged from the pre-feature baseline, and all
  three gate commands pass without any assertion being loosened.

## Assumptions

- `TimeCellClock` is the right first target because it is live, taught, and multiply consumed (F1) — not
  because it is the most valuable population. The other four hardcoded widths stay fixed until the pattern
  is proven on one.
- The readout degeneracy in F9 is assumed specific to this population's decode, not general to the other
  four. That assumption is untested and should be checked before generalizing.
- "Held-out timing error" is a measurement that must be **built**; per F8 the only existing checks are
  in-sample, and one measures parameter distance rather than timing at all.
- ~~Growth is assumed to be an append-at-the-top operation on a rank-ordered population. Interior
  insertion has different invariants and is not assumed to work.~~ **REFUTED by Phase 0 measurement,
  2026-08-01.** Append-at-top is inert below saturation under *every* readout tested — 20 of 20 paired
  ties, max |Δ| exactly 0.000000 — because a cell above the ramp never fires and is therefore causally
  inert on any readout of the fired set. Growth is **interior insertion at a reset boundary**; see
  [research.md](./research.md) R2. Extend-at-top is retained only to address saturation and its effect
  must be reported as clamp-widening under FR-006, never as earning.
- The prior grow-on-top null is treated as directional evidence only: its 10-seed run is unrecorded, only
  a 3-seed result is in the tree, and its design isolates seeding rather than pruning (F12).
- The one-lake, one-brain composition is unchanged; this feature adds no store and no second brain.

## Out of Scope

- **The other four hardcoded widths.** `learned_evc`, `metacognitive_monitor`, `second_order_access`,
  `neuronal_avalanches` stay fixed. Generalizing needs slice 3's answer.
- **Building the spiking-half prune policy.** Specified in `specs/002-capability-and-bounding` FR-002 and
  remains that feature's work. This spec only forbids depending on its absence (FR-014).
- **Pressure engineering.** Routing, multiple bus readers, anything creating differential usage so added
  capacity specializes, is the long-term answer to the grow-on-top null and is a separate line of work.
- **Repairing the stale ledger row and artifact.** `time_cells_ab_eprop.txt` is 9 commits stale and
  `docs/coverage/README.md:239` quotes a retired ladder (F13). Both should be corrected, but re-recording
  a moving number is its own decision and is not this feature's work.
- **Fixing cell 0's structural zero error term**, and **removing the silent `except Exception` blocks** at
  `brain.py:749` and `:1934`. FR-013 requires the growth path not to hide behind them; repairing the
  existing handlers is its own change.
- **Implementation.** No mechanism, data structure, or call site is chosen here.

## Dependencies

- ~~`specs/002-capability-and-bounding` FR-002 (spiking-half prune policy) gates slice 4 under FR-014.~~
  **REMOVED 2026-08-01.** Phase 0 chose `home.save_state` over a lake record, where **overwrite is the
  prune policy** — so FR-014 is satisfied without the lake's unbuilt prune, and this feature no longer
  depends on `specs/002`. The decision also avoids a measured 0.400 s / 187.5 MB-resident lake scan per
  Brain construction. See [research.md](./research.md) R4.
- The measurement instrument required by FR-002 does not exist and must be built (F8).
- A population-dependent consumed quantity does not currently exist for this population and must be
  established before slice 3 can return a verdict (F9).

## Status

**Slice 1 is unblocked and is the near-term work.** F2 makes it concrete: the destructive rescale is a
real, quantified defect and non-destructive growth is a well-posed problem with a testable invariant.
`TimeCellClock` has no growth API at all today, so this slice also creates the B arm every later A/B
needs.

**Slice 2 is the blocking discovery of this spec.** Growth of this population cannot be measured on the
quantity the brain consumes, because that quantity is the ramp and not the population (F9, verified by
direct measurement). This was not visible from the handoff and changes the feature's shape: the first
real obstacle is a readout that ignores its own population, not the absence of a growth mechanism.

**Slice 3 is blocked on slices 1 and 2**, and on an instrument that does not exist (F8).

**Slice 4 is blocked on one thing, not two** — slice 3 returning a benefit. The bounding blocker is gone:
persisting via `save_state` makes overwrite the prune policy, so FR-014 is satisfiable today and this
feature no longer waits on `specs/002-capability-and-bounding`. The honest current statement is that the
brain cannot yet be allowed to grow this population continuously, and the remaining reason is
**measurability alone**.

### Resolved decision

**Q1 — how slice 2 establishes a population-dependent consumed quantity. DECIDED 2026-07-31 (owner):
repair `decode()` so the population contributes to the elapsed estimate.**

The alternatives considered and rejected: adopting `sd` as the measured quantity would have judged growth
on a figure the action path does not consume; measuring only in the saturated regime would have rewarded
clamp-widening and could not have supported a capability claim under FR-006.

The consequence is that slice 2 is a **defect fix, not an instrument choice**. A 64-cell population whose
readout discards 63 of its cells is the computed-but-not-consumed shape P3 exists to catch, and the fix
is owed whether or not anything is ever grown. This makes slice 2 valuable on its own terms and
independently shippable — the first slice of this feature that changes live behaviour.

It also raises the stakes: `decode()` is on the timed Go window path (`brain.py:1559`), so slice 2 must
carry its own before/after evidence that live behaviour changed in the intended direction, and must keep
the module's Weber/scalar-timing self-check and the same-seed determinism test green without loosening
either.

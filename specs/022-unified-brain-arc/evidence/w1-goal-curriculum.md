# W1: goals drive the arm — the premise probe, pre-registered (2026-09-03)

*The plan of record, Phase 1: "**W1 — goals drive the arm inside the existing world (this IS T076's
mechanism, pulled forward under O5(c)).** … Build the module T076 names, `basal_ganglia/curriculum.py`,
with `propose()/observe()` and the |LP| bandit over goal-space regions, by PORTING
`cortex/autotelic_goals.py`'s goal sampling (competence-progress babbling over achieved outcomes, HER)
from its toy grid onto the arm's state space … as `GoalStore`'s (`_T_GOAL`) first live caller, under
FR-027's JOIN … **W1's commit carries T076's red**: an objective with `coverage_reward` neutered must
measurably change goal selection."*

T076, verbatim: *"Curriculum as the curiosity consumer: `propose()/observe()` interface, |LP| bandit
over goal-space regions with the exploration floor DERIVED from measured |LP| signal-to-noise on the
goal store (P1 — 20% is the literature reference, never the value)."*
FR-027, verbatim: *"The incumbent objective `coverage_reward` carries its two measured properties
(saturation curve reported; `_covered` lake-persisted at S0) and is joined, never silently replaced,
by |LP| goals."*

## Why this file opens with a probe and not a build

Part C of the plan states the rule this follows: **"The world before the wire. No growth or learning
task starts until its stimulus has been shown, on the exact signal the task names, to carry structure
its red can detect (rank 2c's in-process probe is the template: probe, then build; a roadmap number is
a claim about an instrument — re-measure it first)."** Part I step 4 says the same operationally:
*"Pre-register the task's protocol in its evidence file; probe the premise on the named signal first."*

A six-lens orientation pass over the incumbents found that **four of W1's written premises do not
survive contact with the tree.** Each is recorded here before any code is written, because three of
them change what W1 should build and the fourth decides whether it should be built at all.

**P1 — the port source does not contain the mechanism the plan says to port.**
`cortex/autotelic_goals.py::AchievedOutcomeBuffer.sample` draws **uniformly** from the achieved-outcome
buffer intersected with an allowed pool (`pool = [s for s in self._seen if s in allowed]; …
return rng.choice(pool)`). Competence-progress never touches goal drawing. It appears only in
`sagg_riac_practice`, which is an **offline replay harness**: it never runs the environment, it indexes
a pre-measured competence curve by practice count (`k = min(practice[r], len(curves[r]) - 1)`). What IS
portable from that module: HER relabelling (`collect_labels(hindsight=True)`, the standard future
strategy over an episode trajectory) and the achieved-outcome buffer itself. What is not portable is an
online competence-progress sampler, because there is not one.

**P2 — the |LP| bandit already exists, so building one is a P2 violation, not the task.**
`most_curious()` has two implementations and 12+ live call sites and IS "an |LP| bandit over regions".
Under P2 exactly one mechanism survives; W1 must ADOPT it (or the `LearningProgress` facade over it) and
must not add a third. Recorded here because the plan's wording ("build the |LP| bandit") reads as a
licence to write one.

**P3 — the drive `autotelic_goals` uses was already refused as an |LP| basis.** It builds on
`SpikingCuriosityDrive`, whose readout is rectified `max(0, slow - fast)`.
`basal_ganglia/learning_progress.py` records that its own four-regime check **measured that drive
INVALID as an |LP| basis** ("its pools are calibrated at one reference error (0.9), and on a FLAT stream
at other levels the rates do not converge") and adopted the rate `CuriosityDrive`'s
`learning_progress_abs` = |slow − fast| instead. A port that copies the source module's kernel choice
re-introduces a defect this tree has already dispositioned.

**P4 — T076's red, as written, is ALREADY SATISFIED WITH NO W1 AT ALL.** Measured on an identical
seeded stream, 60 ticks, LiveBrain-shaped arm: neutering `coverage_reward`'s return value changes
**18 of 60 actions** and takes `_z_goal` from 60/60 ticks to 0/60. So *"an objective with
`coverage_reward` neutered must measurably change goal selection"* passes today, on the incumbent path,
before W1 exists. **A red that can be satisfied without the mechanism it exists to test is not a red for
that mechanism** (P8 clause 1). The corrected form is pre-registered below.

## The premise, stated as something that can fail

**W1's mechanism is an |LP| bandit that picks WHICH goal-space region to pursue. That mechanism can
only work if |LP| DISCRIMINATES among regions at the scale the live arm produces.** The orientation
measured two facts that put this in doubt, and neither was known when the plan was written:

- the live |LP| channel runs about **55× smaller** than the harness scale `learning_progress.py` was
  validated on (live ≈ 0.0035 against ≈ 0.19), with **≈51% exact zeros** over 400 production percepts;
- the `_covered` lattice is lake-persisted and restored at boot, so on the live daemon the arm
  **almost never earns the fresh-cell reward** (max |r_ext| = 0.10 over 194 live steps, zero new cells).
  In-process under `throwaway_home()` the ledger starts empty, so this probe does NOT inherit that
  saturation — which is itself a disclosure: the mechanism will behave differently on the live daemon
  than it does here, and that gap is reported, not smoothed.

If |LP| cannot separate regions at this scale, an |LP| bandit over them is dormant by construction
(P3: computed, not consumed), and no amount of wiring changes that.

## Protocol — PRE-REGISTERED before the first run (this section is written first; results follow)

**This run decides whether W1 is built.** It is a premise probe, not a treatment. There is no
treatment arm, because the treatment is the thing whose construction this run authorises or refuses.

**Arm.** One: today's arm, unchanged. `make_brain(seed, persist=False)` inside
`substrate.home.throwaway_home()`, driven by `coverage_reward` exactly as the live wake path does. No
new environment, no scene change, no widening of the 64-wide context.

**The goal space, named in advance.** The **coverage lattice the objective already maintains**: the
integer triple `(round(ee_x/c), round(ee_y/c), round(ee_z/c))` with `c = _cover_cell()`, measured
0.023852924940072198 m (20 cells per reach). It is chosen over end-effector floats or joint vectors for
three measured reasons: it is already the objective's own unit of novelty, so a goal in it is not a
fabricated target (`coverage_reward`'s docstring refuses an invented target as *"the same class of
theatre as a panel with no signal behind it"*); it is **hashable and integral**, so
`AchievedOutcomeBuffer`'s exact-equality dedup works, where on continuous floats it silently fails
(measured: 3 near-identical floats dedup to 3, 3 equal ints to 1); and it is populated — 19 distinct
cells in 40 ticks from HOME.

**Seeds.** 8 (0–7), the plan's S for every seed-paired statistic. Reported **per seed, never
averaged** — the W-R2 correction generalises past entropy to every statistic in this phase.

**Ticks.** 3000 per seed, the checkpoint W3's baselines already use.

**M1 — is the goal space viable?** Per seed: distinct cells visited, visits per cell, the fraction of
ticks whose cell is new, and the cell-visit distribution over the last 1000 wake rows. Window named:
the FIRST 1000 wake rows for the baseline figure, the LAST 1000 reported beside it, with the overlap
stated (W3's §4 correction).

**M2 — does |LP| discriminate among regions? THE DECIDING MEASUREMENT.** Per seed, per cell, the
`LearningProgress` facade's |LP| (the rate-drive `learning_progress_abs`, NOT the refuted spiking
drive), and from it the **signal-to-noise T076 requires**: between-cell variance of mean |LP| over
within-cell variance, with the count of exact zeros reported beside it. **The exploration floor T076
demands is DERIVED from this number and from nothing else** — P1 forbids the literature's 20% as a
value.

**M3 — does the incumbent bandit discriminate at that scale?** `most_curious()` over the same cells,
per seed: how many distinct cells it selects across the run, and whether its choice is invariant to the
|LP| it is fed. A bandit that returns the same region regardless of |LP| is the P3 "constant wearing a
signal's clothes" shape, and it is measured here rather than assumed.

**THE BAR, fixed in advance.** The premise HOLDS if, per seed, **|LP| separates cells above its own
within-cell noise** — between-cell variance exceeds within-cell variance on at least 6 of 8 seeds AND
the exact-zero fraction is below 0.9 — **and** `most_curious()` selects more than one distinct region.
Both halves are required: a signal that separates but a bandit that cannot read it is equally fatal to
the mechanism.

**Controls, each in the same command as the measurement it guards.**
- *Positive control for the |LP| instrument*: the same statistic computed on the module's own validated
  ramp, which must separate. If it does not, the instrument is broken and the live zero means nothing —
  this is the project's standing rule that a zero is a claim about the instrument first.
- *Null for the separation statistic*: cell labels shuffled marginal-preserving. Between-cell variance
  on the shuffle is the noise floor the live number must beat; without it, "variance between cells" is
  just variance.
- *Liveness control for the goal space*: a constant-context arm must collapse to one cell. If the
  lattice reports many cells on a frozen arm, it is measuring noise.

**Disposition on a null, pre-registered.** If the premise fails, **W1 is REFUSED ON MEASUREMENT and no
`basal_ganglia/curriculum.py` is written** — the rank 2c precedent, where the corrected finding was
worth more than the code. The recorded finding would be that |LP| over this body's reachable workspace
does not carry region structure at live scale, which re-targets T076's mechanism rather than deleting
it, and Phase 2's L2 inherits the same warning. A refusal here is the licence working (P8 clause 5).

## Protocol AMENDMENT, made BEFORE the deciding run and disclosed as smoke-motivated

A 300-tick smoke run (one seed, 1/10 scale, to check the instrument) returned a fact the protocol
above did not anticipate: at the objective's own resolution the arm visits **72 distinct cells in 146
wake ticks, median 1 visit per cell**. |LP| is a comparison of a slow and a fast EMA of the error
observed IN a region; a region entered once has no such comparison. So a null at that partition would
not have been a fact about the SIGNAL, it would have been a fact about the PARTITION — and the
protocol as written could not have told those apart.

**A second arm is therefore added before the deciding run: the same measurement on a COARSER
partition, cells 4x wider per axis.** Three things about this amendment, said plainly because
amending a pre-registration after seeing data is the exact move pre-registration exists to prevent:

1. **The original bar is UNCHANGED.** The fine-partition arm keeps its bar and is still reported.
2. **The new arm's bar is stated here, in advance, and is the same bar** — |LP| must separate cells
   above its own within-cell noise on at least 6 of 8 seeds with the exact-zero fraction below 0.9,
   and `most_curious()` must select more than one distinct region.
3. **The smoke run was a 1/10-scale instrument check, not the deciding run**, and it moved no bar; it
   added an arm that can also fail. The deciding run had not been executed when this was written.

Two supporting changes, same commit: the |LP| snapshot cadence tightens from every 50 wake rows to
every 25 (at 3000 ticks the original gave ~29 samples per cell, too few for a variance ratio), and a
cell must now contribute at least 3 snapshots before it enters the F-ratio at all — at 2 the within-cell
variance of a constant pair is exactly 0 and the ratio reports Infinity, which is an artifact of
sampling rather than a separation.

**The coarse partition is computed by dividing the measured cell size in the probe, never by mutating
`_COVER_CELLS_PER_REACH`.** That is deliberate: `_cover_cell()` is `lru_cache`d, and a prior coverage
red was confounded exactly by moving the constant while the cached grid did not follow.

## The corrected W-R3, pre-registered

Part F's row reads *"goal selection unchanged with `coverage_reward` neutered | the un-neutered run |
W1"*. That is **vacuous as written**, measured above: the incumbent already moves 18/60 actions under
that neuter. The corrected red, which W1's commit carries if W1 is built:

- **The observable is W1's OWN PROPOSAL** — the goal cell `propose()` returns — not the action and not
  `_z_goal`. The incumbent cannot produce that observable at all, so the red cannot be satisfied
  without the mechanism. That is what makes it a red for W1 rather than a red for the arm.
- **Treatment**: with the curriculum wired, neutering `coverage_reward`'s RETURN VALUE (side effects
  kept, since the measured causal channel is the returned scalar only — pinning the return and
  neutering the whole function give byte-identical action sequences) must change the proposed-goal
  sequence, paired per seed, S = 8, paired one-sided t > 1.895 at 7 d.f., ties counted, permutation
  false-positive rate printed beside the verdict.
- **SPECIFICITY CONTROL, and it is the half the written red lacked**: neutering an unrelated scalar of
  comparable magnitude on the same path must NOT change the proposed-goal sequence. Without it, a
  proposal that is merely noisy passes the treatment arm for free.
- **Attribution**: per-item, using the tree's own existing mechanism rather than a new one —
  `_td_flipped` is already computed in both `_select` and `_bg_select` by comparing the top-drive
  action with and without the top-down vote. W1 reuses that comparison for its own channel instead of
  inventing a second attribution path.

## Traps this protocol commits to avoiding, each measured by someone before it cost a run

- **`_cover_cell()` is `lru_cache`d.** A prior coverage red was confounded exactly this way: the
  constant moved and the cached grid did not (107 cells measured against a ceiling of 33). Any arm that
  changes the cell size must clear the cache and prove the grid actually changed.
- **A neuter that leaves `coverage_reward`'s persisted state readable is not a neuter** — `_covered` is
  lake-persisted with a restore path, and `coverage_rate` / `coverage_restored` ride on PROVENANCE.
- **Neutering `coverage_reward` does not disarm exploration**: `_paid` is set by
  `reward_fn is not None`, never by its value, so a neutered arm still explores with full count-based
  optimism. "The arm stopped moving" would be the wrong evidence.
- **`step()` has two return shapes** — the reflex/E-STOP early return carries no `goal`, no `percept`,
  no `grew`. A consumer reading `out["goal"]` unconditionally KeyErrors on the first contact tick.
- **`ee` is reported in millimetres** (`state["ee_mm"]`) while every coverage consumer works in metres
  through the single `_ee_m` helper, which exists to prevent exactly one double conversion.
- **`GoalStore` is single-writer by contract** and `home._log` caches BinLogs, so two concurrently open
  views share a BinLog and previously collided on goal ids and silently destroyed a goal. Exactly one
  store may be open per lake. It also **refuses a zero-norm embedding**.

## Result

*(To be appended after the run. This section is deliberately empty in the commit that pre-registers the
protocol above; the probe had not been executed when the protocol was written.)*

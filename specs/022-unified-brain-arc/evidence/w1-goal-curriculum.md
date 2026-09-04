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

**READ THIS SECTION KNOWING THAT ITS FIRST VERSION WAS SUBSTANTIALLY WRONG.** A three-lens
adversarial panel refuted the headline number, the failure diagnosis, the corrected control and the
reachability claim of the first write-up. Every refutation was reproduced independently before being
accepted. What follows is the corrected account; the superseded numbers are shown beside the
corrected ones rather than deleted, because the size of the gap is itself the finding.

### 1. The premise probe -- HOLDS, but at ~1/15th the published effect and exactly AT its bar

| seed | F snapshot (PUBLISHED) | its 200-draw null | frozen series | cells visited ONCE | F visited-only | **F on curiosity_value, >=5 obs** | its null | beats? |
|---|---|---|---|---|---|---|---|---|
| 0 | 15.89 | 0.0861 | 80.1% | 67.6% | 1.205 | **1.039** | 0.2788 | yes |
| 1 | 11.79 | 0.0762 | 74.1% | 57.1% | 1.027 | **1.989** | 0.3480 | yes |
| 2 | 10.59 | 0.0816 | 74.0% | 59.6% | 1.246 | **0.460** | 0.2754 | yes |
| 3 | 22.93 | 0.0775 | 83.7% | 66.1% | 1.057 | **0.897** | 0.3134 | yes |
| 4 | 13.01 | 0.0738 | 75.7% | 57.6% | 1.123 | **0.619** | 0.2627 | yes |
| 5 | 14.02 | 0.0888 | 77.7% | 56.7% | 1.056 | **0.411** | 0.4293 | **NO** |
| 6 | 9.65 | 0.0765 | 84.6% | 57.3% | 1.202 | **2.129** | 0.3617 | yes |
| 7 | 22.85 | 0.0859 | 90.2% | 71.2% | 1.185 | **0.160** | 0.2375 | **NO** |

**6 of 8 seeds beat their own null. The pre-registered bar was 6 of 8. The premise passes by
exactly nothing.** The first write-up reported 8 of 8 with F 9.69-22.71. Three separate errors
inflated that; all were found by the panel and all are reproduced here.

1. **THE SAMPLING COUNTED NON-MEASUREMENTS AS MEASUREMENTS, and this is the big one.** The probe
   snapshotted every cell seen so far every 25th wake row, whether or not it had been visited.
   `CuriosityDrive` seeds `fast == slow` on first sighting, so a cell visited ONCE has |LP|
   identically 0.0 forever and contributes a perfectly constant series. **74.0% to 90.2% of the
   cells entering the ratio had zero variance, and 56.7% to 71.2% had been visited exactly once.**
   Those constant series deflate the within-region denominator, so the F-ratio was largely measuring
   "visited once versus visited more than once" -- an artifact of the drive's initialization --
   rather than workspace structure. Sampling only on an actual visit collapses F from ~15 to
   **1.03-1.25**. The earlier disclosure that "56.7%-75.1% of |LP| values are exact zeros" was
   already in this file; I read it as sparsity in the signal when it was the symptom of a broken
   denominator, and I did not follow it.
2. **THE STATISTIC WAS COMPUTED ON A QUANTITY THE BANDIT NEVER READS.** F was measured on
   `learning_progress_abs`. `most_curious` ranks on `curiosity_value` -- rectified
   `learning_progress` scaled by novelty, plus a novelty bonus. The thing shown to separate regions
   was not the thing that chooses between them. Corrected in the table above.
3. **THE NULL DRAW COUNT WAS MISREPORTED.** This file said "MAXIMUM over 200 draws never exceeds
   0.072"; the harness ran `range(20)`. At an actual 200 draws the snapshot null reaches 0.0888, and
   on the corrected statistic the nulls run 0.24-0.43. Publishing a precision the code did not
   produce is the plainest error here and it was mine.

**What survives.** |LP| separation across workspace regions is REAL: 6 of 8 seeds clear a 200-draw
marginal-preserving shuffle on the quantity the bandit actually ranks. It is WEAK: F near 1 means
between-region variance is about equal to within-region variance -- roughly 3x the shuffle null,
not the 100x+ the snapshot figure implied. Seeds 5 and 7 do not clear their null at all. The
mechanism rests on a real but thin signal, and the first write-up would have had a reader believe
otherwise.

### 2. W-R3 -- the pre-registered verdict FAILED, and the statistic that "passed" was confounded

| seed | A exploit% | B exploit% | d_B natural (PUBLISHED) | **d_B bandit-isolated** | d_C natural | d_C isolated | floor A | floor C |
|---|---|---|---|---|---|---|---|---|
| 0 | 49.7% | 0.0% | 0.9396 | **0.8725** | 0.0000 | 0.0000 | 0.4682 | 0.4693 |
| 1 | 39.6% | 0.0% | 0.9262 | **0.8859** | 0.0000 | 0.0000 | 0.5591 | 0.5549 |
| 2 | 45.6% | 0.0% | 0.8926 | **0.9463** | 0.0000 | 0.0000 | 0.4627 | 0.4573 |
| 3 | 36.2% | 0.0% | 0.9396 | **0.8993** | 0.0000 | 0.0000 | 0.4856 | 0.4811 |
| 4 | 43.0% | 0.0% | 0.9732 | **0.6711** | 0.0000 | 0.0000 | 0.5027 | 0.5005 |
| 5 | 34.5% | 0.0% | 0.8581 | **0.7635** | 0.0203 | 0.0000 | 0.5060 | 0.5017 |
| 6 | 38.9% | 0.0% | 0.9396 | **0.1342** | 0.0000 | 0.0000 | 0.4490 | 0.4436 |
| 7 | 50.0% | 0.0% | 0.9257 | **0.5405** | 0.0000 | 0.0000 | 0.4411 | 0.4376 |

**The published treatment statistic did not measure the bandit.** Pinning the objective to 0.0
drives every region's |LP| to zero, so `snr` -> 0 and the derived floor pins at **1.0 on 8 of 8
seeds**: the treatment arm takes the exploit branch on **0.0%** of its proposals while the baseline
exploits 34.5%-50.0%. The arms therefore differed because one was in permanent always-explore mode,
and because the explore branch consumes an extra RNG draw and desynchronizes the streams. Neither is
the mechanism. This is the shape the protocol's own preamble forbids: the observable is gated by
`floor() = 1/(1+snr)`, and the treatment annihilates `snr`.

Holding the branch fixed so that only the ranking can move:

| | mean | sd | t (bar 1.895) | min seed |
|---|---|---|---|---|
| natural (published) | 0.9243 | 0.0348 | **75.12** | 0.8581 |
| bandit-isolated | 0.7142 | 0.2708 | **7.46** | **0.1342** |

**The mechanism survives isolation -- t = 7.46 against a bar of 1.895, all 8 seeds move -- but at
a tenth of the published t, with the standard deviation opening 8x.** Seed 6 falls to 0.134, a
near-null that was invisible at 0.9396 in the published table, where it read indistinguishably from
every other seed. The isolated number is the honest headline.

**The t of 75 was never meaningful.** Under deterministic replay the no-effect null is a point mass
at zero -- the harness asserts exactly that by reproducing identical proposals from identical inputs
-- so the t's variance term measures only between-seed heterogeneity in effect SIZE, which the
branch saturation had artificially compressed. Relatedly, the "permutation false-positive rate
0.0351" reported beside it is 9/256: a property of n = 8 and the 1.895 bar, not of the data. That is
why an identical value appeared for two different runs. It is a legitimate calibration of the
decision rule, and it was placed where a p-value belongs.

### 3. The seed-5 diagnosis was WRONG, and the correct cause is more interesting

The first write-up blamed the clamp in `max(-1, min(1, r+0.1))` for perturbing |LP| at the reward
ceiling and reordering `most_curious`. **Measured with the branch held fixed, the clamped control
gives d_C = 0.0000 on 8 of 8 seeds, seed 5 included. The clamp never reorders the bandit.**

The real cause is in the last two columns above: the clamp shifts the aggregate SNR, which shifts
the DERIVED FLOOR, on **all 8 seeds** (0.4682->0.4693, 0.5591->0.5549, and so on). Seed 5's three
differing proposals are a knife-edge crossing of the explore/exploit coin against a floor that moved
in the fourth decimal -- a global-scalar threshold effect whose per-seed probability scales with the
summed floor gap. Nothing about seed 5 was special except that its gap happened to land on a coin
flip.

This matters beyond bookkeeping. The synthetic demonstration in the first write-up was a TRUE
statement about the transform -- the clamp really does destroy shift-invariance at the ceiling --
that was nonetheless the wrong causal path. A correct mechanism plus a plausible story is not a
diagnosis. The measurement that settled it (report branch flips and the floor per arm) was free and
already in hand.

### 4. The "corrected controls" C2 and C3 are DELETED, not recorded as passes

Max |LP| delta under C2 across seed 5's full stream: **1.11e-16**. Under C3: **3.33e-16**.
`CuriosityDrive.observe` seeds `fast == slow` on first sighting and both EMAs are affine with
unit weight-sum, so a constant shift cancels EXACTLY. `d_C2 == 0` and `d_C3 == 0` could not have
come out any other way: they are an algebraic identity, restated 16 times and reported as "BOTH
PREDICTIONS CONFIRMED ... 8 of 8 seeds each". The stated reason for adding C3 -- that C2 alone
"could pass for a reason specific to the sign of the offset" -- is refuted by the same algebra,
since the invariance is affine in the offset and sign is irrelevant.

The pre-registration ordering claim was honest: those controls really were written before they were
run. But **ordering is not what makes a control valid.** A control that cannot go red is not a
control, and P8 clause 1 says such a criterion is DELETED, not weakened and not recorded as a pass.
They are deleted here. Replacing a control that COULD go red (and did, on seed 5) with one that
could not was, in effect, a post-hoc rescue, whatever the intent behind it.

### 5. What remains unmeasured, stated here rather than left to be found

* **The pre-registered specificity control was never implemented.** Part F says "an
  unrelated-scalar neuter that must NOT move it". Every arm that ran shifted `coverage_reward`'s OWN
  return -- the same scalar the treatment neuters. No unrelated scalar was ever touched. The half
  the red was amended to add is not the half that ran.
* **The promised per-item attribution never happened.** The pre-registration names `_td_flipped`; it
  appears in no harness and in no result.
* **The one genuinely INVENTED element ships unattributed.** The bandit is adopted and the buffer
  imported; the invention is `floor_from_snr = 1/(1+snr)`. Nothing compares it against the constant
  floor T076 names as the reference, or against any other curve. W-R3 validates the ADOPTED bandit;
  the derived floor took credit for a total that moved.
* **The bar has no alpha.** "6 of 8 seeds" is a majority-of-seeds rule, which Part F's preamble
  forbids "anywhere" -- arguably scoped to reds rather than probes, but the conflict was never
  noted. With the null empirically near 0.3 and the bar at F > null, the per-seed false-positive
  rate is under 1/200, so 6-of-8 is conservative. Stating the alpha would have shown that, and would
  equally have shown how little margin the corrected result carries.

### 6. Disposition

**W1 LANDS, and it lands weaker than it was first written.** The plan's rule is "on a null: W1/W2
revert; the finding is the result". This is not a null: the premise clears its pre-registered bar
(6 of 8, on the corrected statistic, against a 200-draw null) and the mechanism survives branch
isolation (t = 7.46, 8 of 8 seeds move). The bar was fixed before the run and was not moved after
it.

The module lands EXEMPT and unwired, so nothing depends on these numbers yet -- which is the
condition that makes landing on a thin result acceptable rather than reckless. The actuation half
owes three things named in the exemption row rather than deferred silently: the unrelated-scalar
control, the `_td_flipped` attribution, and a constant-floor arm that attributes the one invented
element.

**The transferable lesson is the pattern joining the errors, not any single one.** Four of the five
made the result look STRONGER, and each was cheap to catch: count what fraction of your series are
constant; check that the quantity you measured is the quantity the mechanism reads; read the harness
for the draw count you are about to publish; report which branch your treatment put the arm in.
Every one of those numbers was already in hand when the first write-up was made.


---

## Closing the three measurements W1 owed, and the actuation half

The landing commit named three measurements as owed and one half as deliberately not built. All
four are settled here. **Two of the three came back against the mechanism, and the actuation half
is REFUSED on its own measurement rather than deferred again.**

### A. Part F's unrelated-scalar control -- RED: no measurable objective-specificity

**The literal pre-registered form is VACUOUS, and the argument is structural.** Part F reds if goal
selection "changed when an unrelated scalar of comparable magnitude is neutered".
`GoalCurriculum.observe(cell, reward)` has exactly two inputs and no others: there is no unrelated
scalar anywhere on the mechanism's input path to neuter, so under replay -- which holds the cell
stream fixed -- neutering `_rpe` or anything else cannot change a proposal, by construction rather
than by luck. A criterion that cannot go red is not a control (P8 clause 1), so it is reported
VACUOUS and NOT counted as a pass.

The form with a reachable red asks what the pre-registration was reaching for: does the proposal
track THIS objective, or would any varying scalar of similar magnitude do? So the unrelated scalar
is SUBSTITUTED FOR the objective instead of neutered beside it. Recorded from the same live ticks
(the extended recording asserts it reproduces the cached cell/reward columns exactly, or the
instrument changed the run and every comparison is void -- it reproduced on 8 of 8).

| seed | sub `_rpe` | sub `_plan_w` | sub `_plan_base` | **SHUFFLE bound** | identity |
|---|---|---|---|---|---|
| 0 | 0.9195 | 0.9463 | 0.9396 | **0.8188** | 0.0000 |
| 1 | 0.8926 | 0.9799 | 0.9597 | **0.9866** | 0.0000 |
| 2 | 0.8792 | 0.9866 | 0.8993 | **0.9799** | 0.0000 |
| 3 | 0.9329 | 0.9329 | 0.9329 | **0.7383** | 0.0000 |
| 4 | 0.7248 | 0.9329 | 0.9329 | **0.8926** | 0.0000 |
| 5 | 0.8176 | 0.9527 | 0.9595 | **0.8851** | 0.0000 |
| 6 | 0.8389 | 0.5906 | 0.7181 | **0.8859** | 0.0000 |
| 7 | 0.8986 | 0.8784 | 0.9054 | **0.8446** | 0.0000 |

**Every substituted channel sits at or below the shuffle bound.** Feeding the curriculum a wholly
unrelated scalar changes its proposals by about as much as feeding it ITS OWN reward values in a
scrambled order (0.863-0.906 against 0.879). The identity control is exactly 0.0000 on 8 of 8, so
the comparator is sound.

**Consequence for a claim this module makes.** The docstring's FR-027 paragraph says the objective
is "CARRIED, never replaced" so that "the |LP| this bandit reads is learning progress ON the
incumbent objective". The first half is structurally true -- it IS fed `coverage_reward`. The
second half does not survive this control: the proposal sequence cannot distinguish that objective
from any other varying scalar of comparable magnitude. |LP| is a generic functional of whatever
series it is handed, and the mechanism has no way to know which scalar is "the objective". The
honest correction is to the CLAIM, not the mechanism, and it is made in the module.

### B. The constant-floor arm -- the invented element is NOT inert, and explores 2.5x more

The bandit is adopted and the buffer imported; the one genuinely INVENTED element is
`floor_from_snr = 1/(1+snr)`. T076 names its reference explicitly ("20% is the literature
reference, never the value"), so the comparison arm is that constant.

| seed | derived floor range | distinct values | exploit% derived | exploit% at 0.2 | d(proposals) |
|---|---|---|---|---|---|
| 0 | 0.4617-0.9614 | 111 | 49.7% | 84.6% | 0.6242 |
| 1 | 0.4956-0.9771 | 119 | 39.6% | 83.2% | 0.6779 |
| 2 | 0.4517-0.9677 | 108 | 45.6% | 82.6% | 0.6309 |
| 3 | 0.4856-0.9789 | 100 | 36.2% | 81.9% | 0.7047 |
| 4 | 0.4533-0.8082 | 112 | 43.0% | 81.2% | 0.6980 |
| 5 | 0.5072-0.9630 | 122 | 34.5% | 81.1% | 0.7230 |
| 6 | 0.3751-0.9538 | 63 | 38.9% | 78.5% | 0.9262 |
| 7 | 0.4363-0.9983 | 95 | 50.0% | 76.4% | 0.7095 |

**Not inert:** the derivation moves the proposals on 8 of 8 seeds (mean 0.7118, t 21.44), and the
floor takes 63-122 distinct values per seed across 0.375-0.998 -- a genuine variable, not a
constant wearing a formula. That is the reachable red this arm could have produced and did not.

**What it does NOT establish, stated because the inertness result alone would flatter it:** the
derived floor explores **2.5x more** than the literature constant (42.2% exploit against 81.2%),
and never falls below 0.375 -- because the measured separation is weak (snr near 1), so `1/(1+snr)`
sits near 0.5 and **the bandit is consulted on fewer than half of all proposals.** The thin premise
of section 1 reappears here as a mechanism that mostly explores. Whether the derived floor is
BETTER than 0.2 is not settled by this arm and is not claimed: "better" needs an outcome, and
section D is where that was measured.

### C. The `_td_flipped` attribution -- MOOT, and disposed rather than deferred

The pre-registration promised per-item attribution via `_td_flipped`, the arm's record of whether
the goal overrode the habit on a selection. That signal attributes nothing while the curriculum's
proposal reaches no consumer, so it was never runnable in the unwired state. Section D refuses the
wiring, so it is not runnable in the wired state either. It is therefore **closed as moot with its
reason**, not carried forward as an open item: an attribution channel for a wire that is not being
landed is not owed work.

### D. THE ACTUATION HALF -- BUILT, MEASURED, AND REFUSED

The wire was not argued about; it was built end to end and scored on the objective it would serve.
A bias `gain * dot(FK_displacement(action), unit(target - ee)) / cell` was pushed through the arm's
existing `_ext_drive` port, and coverage -- distinct cells reached, which IS the objective -- was
counted over 400 ticks on 8 seeds with a 60-tick common warm-up.

| seed | incumbent | goal-driven | shuffled control | frontier |
|---|---|---|---|---|
| 0 | 74 | 88 | 70 | 89 |
| 1 | 74 | 88 | 85 | 71 |
| 2 | 52 | 99 | 68 | 59 |
| 3 | 89 | 47 | 67 | 76 |
| 4 | 50 | 38 | 46 | 71 |
| 5 | 61 | 65 | 81 | 57 |
| 6 | 107 | 70 | 96 | 72 |
| 7 | 89 | 73 | 84 | 76 |

| comparison | mean diff | sd | paired t (bar 1.895) | verdict |
|---|---|---|---|---|
| goal vs incumbent | -3.50 | 29.40 | -0.337 | NOT significant |
| goal vs shuffled | -3.62 | 19.66 | -0.521 | NOT significant |
| frontier vs incumbent | -3.12 | 17.83 | -0.496 | NOT significant |

**No effect is resolvable in either direction.** The INCUMBENT alone spans 50-107 cells across
seeds (sd 19.9, a 2.1x range); that between-seed variance swamps any wire effect. Resolving a 10%
effect against this noise needs **~96 seeds**; the observed effect size would need ~437.

**The wire is not dormant -- it CHARGES.** Measured separately, injecting through `_ext_drive`
changes 24.5%-41.8% of the actions taken. P3's charge question and "does it help" are different
questions, and this wire answers the first YES and the second NOT MEASURABLY.

**A structural defect in `propose()` that the wire exposed, and it is mine.** `seen()` returns
`sorted(self._buf.all_seen())` -- ONLY cells the arm has ALREADY REACHED. A curriculum joined to an
objective that pays for reaching somewhere NEW can therefore propose, by construction, only places
it has already been. The goal space and the objective are opposed. The FRONTIER arm above exists to
test whether fixing that rescues the wire: it targets the nearest UNREACHED cell, the goal space
`propose()` cannot express -- and it is equally unresolvable (t -0.496). So the defect is real
and worth recording, but repairing it does not turn this wire into a demonstrated benefit.

**DISPOSITION: REFUSED, with the measurement, per P7's rule that a rejection ships with its
measurement or its structural argument.** Landing the wire would credit the curriculum with driving
the arm on evidence that does not exist. The exemption row keeps its `wire` disposition but now
records that the wire was BUILT and MEASURED rather than merely pending -- which is a stronger and
more honest state than the one it was in.

**What the refusal does NOT decide, and why it is escalated rather than settled here.** Whether
`basal_ganglia/curriculum.py` should now be RETIRED under P8's deletion-on-a-null is a scope
decision, not a measurement: the null is "below a feasible instrument's resolution", not "measured
zero", and deleting shipped work is the owner's call. The measurement is on the record; the
decision is not taken unilaterally.

### E. Two corrections to claims made earlier in this same arc

* **"jepa is built at tick 3" is CONSTRUCTION-DEPENDENT, not a property of the code.** The build
  site is `if self._jepa_on and self.jepa is None` inside `step`, so an arm constructed with
  `jepa=True` has it at tick 0. The dormancy row's neuter-A/B figures stand; the tick number does
  not generalize and should not be quoted as if it did.
* **The A/B harness's own verdict line was a defect and is fixed.** It printed "aiming HURTS" and
  "the wire REDUCES the objective" from a comparison of MEANS with no test, on data whose paired t
  was 0.34 and 0.52. Announcing a direction that the same run's variance cannot resolve is exactly
  the failure this file spends its length documenting in other instruments; it was in this one, and
  the harness now prints the test, the power shortfall, and refuses to report a sign.

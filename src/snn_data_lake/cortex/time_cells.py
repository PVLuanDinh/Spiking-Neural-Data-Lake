"""
Spiking population clock -- ramping-to-threshold "time cells" that tile an interval.

Fixes: interval timing elsewhere in the brain has been a hand-coded constant. Here a bank of n_cells
LIF-style accumulators is triggered by reset() (the "start" event) and each ramps toward its own
STAGGERED threshold: cell i's nominal threshold is i+1, so cell 0 crosses first, cell 1 next, and so
on -- the population TILES elapsed time with one preferred latency per cell (hippocampal "time
cells": MacDonald, Lepage, Eden & Eichenbaum 2011, Neuron, "Hippocampal time cells bridge the gap in
memory for discontiguous events"). decode() reads elapsed time off this population with a rank-order
code and NOTHING ELSE: "cells 0..k-1 have crossed, cell k has not" brackets the answer between two
neighbouring nominal latencies, and the estimate is the geometric centre of that bracket. It used to
refine the rank with the shared ramp's partial activation, which algebraically collapsed to the ramp
itself -- so the readout was independent of the population and RMSE was bit-identical at widths 64
through 512. See decode() for the measurement and the cost of the replacement.

Every tick adds the same drift (`drive`) plus ONE shared Gaussian noise draw to a single internal
ramp, and each cell's threshold is just a read-out level on that ramp (a driftdiffusion accumulator
racing toward n_cells staggered boundaries at once) -- sharing the noise draw across cells is what
keeps the population's firing order strictly tied to threshold order (a higher threshold can only be
crossed after every lower one, since crossing it implies the lower ones already hold), so the rank-
order code stays well-defined even though the ramp itself is noisy. For a Wiener process with drift mu
and per-step noise variance sigma^2 hitting a boundary at level theta, the classic first-passage-time
result gives Var[T] = theta * sigma^2 / mu^3 -- LINEAR in the threshold theta, hence linear in the
cell's mean latency. Because later cells sit at larger thresholds, their firing ticks scatter further
from their nominal (noise-free) latency than early cells do; decode() reports this SPREAD directly (an
RMS of actual-vs-nominal spike tick over the cells that have fired so far), so its returned sd GROWS
with elapsed time as an EMERGENT property of the accumulator's own noise -- not a fixed multiplier
baked into the formula. This is the standard scalar-timing / Weber's-law account of interval timing
(constant relative, i.e. growing absolute, variability -- Gibbon 1977; Buhusi & Meck 2005, Nat Rev
Neurosci, "What makes us tick? Functional and neural mechanisms of interval timing").

learn_interval(target_steps, reward) retunes ONE cell -- the one whose current nominal latency
(threshold / drive) is nearest target_steps -- toward reproducing that interval. It mirrors
plasticity/r_stdp.py's reward-modulated STDP exactly: treat the external target_steps as a "post"
(teacher) event and the cell's own nominal latency as its "pre" event, form dt = t_post - t_pre,
turn that into an STDP eligibility (LTP, e = a_plus*exp(-dt/tau), if the cell fires too EARLY, dt>0;
LTD, e = -a_minus*exp(dt/tau), if it fires too LATE, dt<0), then gate the eligibility into the cell's
threshold with the caller-supplied reward: threshold += lr*reward*e. A positive reward on a too-early
cell raises its threshold (delays it toward the target); on a too-late cell it lowers the threshold
(hastens it). Repeated calls with a consistent reward sign converge the cell's threshold on the
target interval -- this REPLACES the hand-coded interval constant with a learned one.

DISTINCT from existing modules (read them):
- hippocampus/legendre_memory.py (LMU) holds a CONTINUOUS delay-line window of RECENT INPUT on a
  Legendre-polynomial basis (any lag in [0, theta] is reconstructable). TimeCellClock has no input
  window to reconstruct at all -- it is a pure internal population CLOCK: reset() marks a start
  event, and elapsed time since that event is read off which cells have crossed threshold. A window
  over an input signal vs. a population code for "how long has it been".
- hippocampus/ring_attractor.py integrates ONE continuously-decoded scalar (an angle) from a velocity
  input, and that scalar can move in either direction (persist, advance, reverse). TimeCellClock
  integrates ELAPSED TIME since reset from a constant internal drive (no external velocity, no
  reverse) and its output is a POPULATION rank-order/graded code across many cells, not one
  continuously-updated attractor state.
- hippocampus/htm_sequence_memory.py predicts the next discrete SDR from sequence CONTEXT (columns x
  cells, distal segments) -- it is about WHAT comes next. TimeCellClock has no context and predicts
  nothing about content; it only measures WHEN, via one preferred latency per cell.
- plasticity/r_stdp.py is the general-purpose reward-modulated STDP synapse (arbitrary pre/post
  spike trains). learn_interval() here is a direct, narrower APPLICATION of that same eligibility/
  gating math to one specific plastic parameter (a time cell's threshold) so the population can learn
  a target interval, instead of the general many-synapse rule.

Ceiling -- a small (n_cells<=~100) discretized drift-diffusion accumulator bank with one plastic
threshold per cell; not a trained hierarchical timing network, not the full striatal-beat-frequency
or ramping-clock circuitry, no working-memory gating of the count. Deterministic given a seed (RNG is
per-instance, random.Random(seed), never module-level). Not cognition.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.time_cells
"""

import bisect as _bisect
import math
import random

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "TimeCellClock.learn_interval": (
        "LOCAL",
        "reward-modulated STDP: the target interval plays the role of a teacher 'post' event and the "
        "nearest cell is retuned by spike timing. No gradient is formed anywhere",
    ),
}

# The held-out interval grid, and the split ratio ported from `growth/ng_snn.py`'s
# `NGSNN.prune_unearned` (`n_val = max(8, int(0.2 * N))`) -- the cite read `reasoning/ng_snn.py:178`,
# a path with no file behind it since the promotion into hippocampus/. The
# DISCIPLINE transfers (teach on one portion, judge on a disjoint remainder, refuse when either is
# too small), the POSITION does not. MEASURED at n=64: a literal trailing-20% split leaves only 4 of
# 63 gaps able to move the metric and a stride over the raw integer axis gives 15 of 63, while this
# gap-major grid gives 63 of 63. Lives here rather than in the probe because it is a property of the
# population's own ladder, and one implementation is the rule (P2).
GRID_PER_GAP = 10
GRID_TEST_STRIDE = 5

# The persisted lineage record. Bumping `STATE_FMT` is how an incompatible change refuses every blob
# written before it, instead of misreading one.
STATE_FMT = "time_cells/v1"
STATE_NAME = "time_cells.json"


def interval_grid(thresholds):
    """(teach, test) over a ladder's own gaps: GRID_PER_GAP points per gap at
    `lo + (hi-lo)*(p+0.5)/GRID_PER_GAP`, ordered gap-major, deterministic positional
    1-in-GRID_TEST_STRIDE stride (`index % stride == stride-1` -> TEST). No RNG, so determinism
    semantics survive; disjoint by construction; every gap is represented on both sides."""
    universe = []
    for j in range(len(thresholds) - 1):
        lo, hi = thresholds[j], thresholds[j + 1]
        for p in range(GRID_PER_GAP):
            universe.append(lo + (hi - lo) * (p + 0.5) / GRID_PER_GAP)
    teach = [t for i, t in enumerate(universe) if i % GRID_TEST_STRIDE != GRID_TEST_STRIDE - 1]
    test = [t for i, t in enumerate(universe) if i % GRID_TEST_STRIDE == GRID_TEST_STRIDE - 1]
    return teach, test


class TimeCellClock:
    """A population of ramping-to-threshold time cells tiling elapsed time since the last reset().

    Cell i has nominal threshold i+1 (so, at drive=1, its noise-free firing tick is i+1) -- the
    population of n_cells thresholds TILES the interval [1, n_cells]. A single shared ramp `v`
    integrates drive + noise each tick (a drift-diffusion accumulator); each cell fires once, the
    first time `v` reaches ITS threshold, and then latches. Because all cells read the same ramp,
    crossing a higher threshold always implies every lower one was already crossed, so cells fire in
    strictly non-decreasing threshold order even though the ramp itself is noisy."""

    def __init__(self, n_cells=64, seed=0, drive=1.0, noise_sd=0.15):
        if n_cells < 1:
            raise ValueError("n_cells must be >= 1")
        if float(drive) <= 0.0:
            # decode() divides every nominal latency by `drive`; a non-positive drive makes the ramp
            # never advance (or advance backwards) and turns that division into a sign flip or a
            # ZeroDivisionError deep inside the live tick, where the bare `except Exception` that
            # wraps the clock in `Brain._c1_tick` would swallow it and freeze the clock. Refuse at
            # the boundary instead. (Cited as `brain.py:1934` until 2026-08-04; that line is now
            # `self._pc_surprise = round(pcs, 4)`, which is not a handler at all.)
            raise ValueError("drive must be > 0")
        self.n_cells = n_cells
        self.seed = seed
        self.drive = float(drive)
        self.noise_sd = float(noise_sd)
        # GEOMETRIC tiling of 1..n_cells, not the integer ladder. With `float(i + 1)` and drive 1.0
        # every nominal latency was exactly an integer tick, so cell i crossed at tick i+1 and the
        # ACTUAL spike time equalled the NOMINAL one -- which made learn_interval's error term
        # identically zero and the threshold unlearnable in principle, not merely in practice.
        # Measured before this change: 32 learn calls, dt == 0.0 on every one, thresholds byte-identical
        # after 120 ticks.
        #
        # Log spacing is what this module already claims to implement: its own self-check asserts
        # Weber/scalar timing (sd growing with duration), and interval-timing populations tile time
        # logarithmically. The span is still exactly 1..n_cells and the only input is n_cells, so no
        # tuned constant is introduced (P1) -- the spacing is derived from the population size.
        span = math.log(float(n_cells)) if n_cells > 1 else 0.0
        self.thresholds = [
            math.exp(span * (i / (n_cells - 1))) if n_cells > 1 else 1.0 for i in range(n_cells)
        ]
        self.rng = random.Random(seed)
        # Growth outcomes are counted, not inferred. `refused` is what makes a HALT observable rather
        # than read off an absence of events -- the difference between an honest null and silence --
        # and `failed` is kept separate from it because an exception on the growth path is a defect
        # while a refusal is a normal outcome, and collapsing the two hides the defect inside the
        # normal case (FR-011, FR-013).
        self.grown = 0
        self.refused = 0
        self.failed = 0
        self.restore_refused = 0
        self.reset()

    def reset(self):
        """Mark a new 'start' event: clear the ramp, spike record, and the elapsed-tick counter."""
        self.v = 0.0
        self.fired = [False] * self.n_cells
        self.spike_tick = [None] * self.n_cells
        self.n_fired = 0  # == number of cells with threshold <= v (the rank-order code)
        self.t = 0
        self._last_estimate = 0.0
        return self

    def tick(self):
        """Advance one step: the shared ramp integrates drive + noise; latch any newly-crossed cells
        in threshold order. Returns decode()'s (elapsed_estimate, sd) after the step."""
        self.t += 1
        self.v += self.drive + self.rng.gauss(0.0, self.noise_sd)
        while self.n_fired < self.n_cells and self.v >= self.thresholds[self.n_fired]:
            self.fired[self.n_fired] = True
            self.spike_tick[self.n_fired] = self.t
            self.n_fired += 1
        return self.decode()

    def decode(self):
        """GEOMETRIC-MID POPULATION READOUT: (elapsed_estimate, sd).

        The rank-order code is all this population actually knows: cells 0..k-1 have crossed, cell k
        has not. That brackets elapsed time between two neighbouring NOMINAL LATENCIES, and on a
        geometrically tiled axis the geometric mean is the centre of that bracket:

            k == 0        -> 0.0                                        (nothing has fired yet)
            k >= n_cells  -> thresholds[-1] / drive                     (saturated: the range ceiling)
            otherwise     -> sqrt(t[k-1] * t[k]) / drive

        Every term is existing state, so no constant is introduced. Latencies are in TICKS -- divided
        by `drive`, which the previous readout never did while learn_interval always has (`:168`,
        `:182`); MEASURED at drive=0.5, true elapsed 40, the old code reported 19.340 ramp units as if
        they were ticks.

        WHY NOT THE RAMP. The readout this replaces interpolated the shared accumulator `v` between
        two thresholds, which algebraically collapses to `v` itself -- so it read the accumulator, not
        the population. MEASURED consequence: RMSE against true elapsed was BIT-IDENTICAL at widths
        64, 128, 256 and 512 (0.7276, 16 seeds, horizon 60). A population whose output cannot register
        its own size cannot be grown on purpose; every growth experiment against it is unfalsifiable.
        The cost of the swap is stated rather than hidden: +45% RMSE at the live width (1.0536 vs
        0.7276), closing to +4% by n=512. That is the price of measurability.

        WHAT IS STILL WRONG. The saturated branch is a ceiling, not an estimate -- once every cell has
        fired the population has no information left, and RMSE against true elapsed diverges (292.8993
        at t=600, n=64). This readout does not fix that and must not be credited with fixing it; only
        extending the population's RANGE does, which is `grow("extend")` and is reported as
        clamp-widening, never as earning.

        `sd` is unchanged, byte for byte: the RMS of (actual spike tick - nominal noise-free latency)
        over the cells that have fired, an empirical measure of accumulator jitter that grows with
        elapsed time (Weber/scalar timing)."""
        k = self.n_fired
        if k == 0:
            estimate = 0.0
        elif k >= self.n_cells:
            estimate = self.thresholds[-1] / self.drive
        else:
            estimate = math.sqrt(self.thresholds[k - 1] * self.thresholds[k]) / self.drive
        # Fired cells latch, so the estimate may not regress WITHIN an interval -- learn_interval can
        # lower a threshold mid-trial, and a clock that ran backwards would break the monotonicity
        # contract three live readers depend on. reset() clears this.
        estimate = max(estimate, self._last_estimate)
        self._last_estimate = estimate

        if k == 0:
            sd = 0.0
        else:
            deviations = [self.spike_tick[i] - self.thresholds[i] / self.drive for i in range(k)]
            sd = math.sqrt(sum(d * d for d in deviations) / k)
        return estimate, sd

    def learn_interval(self, target_steps, reward, lr=0.1, a_plus=1.0, a_minus=1.0, tau=10.0):
        """Reward-modulated STDP: retune the cell nearest target_steps to reproduce that interval.

        target_steps plays the role of a teacher ('post') event; the nearest cell's current nominal
        latency (threshold / drive) plays the 'pre' event. dt = target_steps - t_pre sets an STDP
        eligibility (LTP if the cell fires too early, dt>0; LTD if too late, dt<0), and the caller's
        reward gates that eligibility into the cell's threshold. Returns (cell_index, delta)."""
        target_steps = float(target_steps)
        i = min(
            range(self.n_cells),
            key=lambda idx: abs(self.thresholds[idx] / self.drive - target_steps),
        )
        # 'pre' is when the cell ACTUALLY FIRED, not its nominal latency. Using the nominal made this
        # rule self-referential and structurally unable to learn: `i` is selected as the cell nearest
        # `target_steps` BY `thresholds[i]/drive`, so re-deriving t_pre from that same quantity gives
        # dt == 0 whenever any cell sits on the target -- always, since the ladder is integers and the
        # targets are integers. Measured on the live path before this fix: 32 calls, dt == 0.0 on all
        # 32, delta == 0.0 on all 32, thresholds byte-identical after 120 ticks. The rule could only
        # ever move when NO cell matched, i.e. exactly when it was worst placed.
        #
        # `spike_tick[i]` is the tick the cell crossed threshold under the noisy ramp (noise_sd=0.15),
        # which is what "fired too early / too late" in the docstring above actually means, and what
        # decode() already differences as its `deviations`. Unfired cells have no observed event, so
        # they keep the nominal -- there is nothing measured to correct yet.
        t_pre = float(self.spike_tick[i]) if self.fired[i] else self.thresholds[i] / self.drive
        dt = target_steps - t_pre
        if dt > 0:
            e = a_plus * math.exp(-dt / tau)  # fired too early -> LTP -> raise threshold (delay it)
        elif dt < 0:
            e = -a_minus * math.exp(
                dt / tau
            )  # fired too late -> LTD -> lower threshold (hasten it)
        else:
            e = 0.0
        delta = lr * reward * e
        # ORDER PRESERVATION BY REFUSAL, NOT BY CLAMPING. tick() walks `thresholds` in index order and
        # stops at the first uncrossed level, so an inversion strands every higher cell permanently.
        # This rule already produced them: MEASURED over 4000 ticks, 9 of 12 seeds inverted somewhere
        # in the reset-driven regime (3 of 6 in the saturating one; seed 3 ended t[62]=75.505 above
        # t[63]=52.401). Clamping the proposal into the neighbours' bracket was measured and rejected:
        # it collapses the gap to 1 ULP (2.220e-16) against 5.298e-03 under refusal, with identical
        # learning coverage (49.8 of 64 thresholds still move either way) -- and a 1-ULP gap is a
        # denominator in the bracket readout decode() now uses. Dropping the update keeps the vector
        # usable; the cell simply does not learn from an update it cannot legally apply.
        proposed = max(0.5, self.thresholds[i] + delta)
        lo = 0.5 if i == 0 else self.thresholds[i - 1]
        hi = math.inf if i + 1 >= self.n_cells else self.thresholds[i + 1]
        if lo < proposed < hi:
            self.thresholds[i] = proposed
        else:
            delta = 0.0  # the update was refused; report no movement rather than a phantom one
        return i, delta

    # ---------------------------------------------------------------- growth
    def at_reset_boundary(self):
        """The only state in which this population may gain a cell.

        `n_fired` carries a dual role -- a count AND a sorted index into `thresholds` -- so inserting
        below the ramp mid-interval either leaves `spike_tick[j] = None` (MEASURED: TypeError in
        decode()'s deviations loop, swallowed by the bare `except Exception` around the clock in
        `Brain._c1_tick`, which silently freezes
        the clock) or fabricates a crossing time (MEASURED: shifts the live-consumed `sd` by +0.005256
        with zero ticks elapsed). At a boundary the fired set is empty and neither can arise."""
        return self.n_fired == 0 and self.v == 0.0 and all(s is None for s in self.spike_tick)

    def widest_log_gap(self):
        """Index j maximising log(t[j+1]) - log(t[j]) -- where this ladder resolves time worst.

        Log space, because the ladder is geometrically tiled: the widest ABSOLUTE gap is always the
        top one by construction, which would make 'widest' a synonym for 'append near the top' and
        the placement rule vacuous."""
        if self.n_cells < 2:
            return None
        return max(
            range(self.n_cells - 1),
            key=lambda j: math.log(self.thresholds[j + 1]) - math.log(self.thresholds[j]),
        )

    def grow(self, mode="interior", gap=None):
        """Add one cell. Returns its threshold, or None if the attempt was REFUSED.

        Two modes, and they are not interchangeable:

        * ``interior`` -- split a gap at its geometric midpoint, `sqrt(t[j] * t[j+1])`. Defaults to
          the widest log-gap; `gap` names one explicitly, which the held-out instrument's null arm
          needs. This is the ONLY mode that can earn: a cell inside the traversed range can fire, so
          it can change the readout for reasons that are about tiling.
        * ``extend`` -- append at `n_cells + 1`, the constructor's own top for the next width. This
          addresses SATURATION and nothing else, and any benefit it produces MUST be reported as
          clamp-widening under FR-006. MEASURED: below saturation, appending at the top is inert
          under every readout tested -- ties 20 of 20, max |delta| 0.000000 -- because a cell above
          the ramp never enters the fired set, so `k` and every function of it are bit-identical.

        Refusals (return None, `refused` increments, NO state changes): away from a reset boundary,
        on an already non-ascending vector, on a degenerate gap whose midpoint would not sit strictly
        between its neighbours, or on an unknown mode. A refusal is data, not an error.

        Draws no random numbers: `rng.getstate()` is identical across a growth, which is what keeps
        same-seed determinism green. `n_cells` is mutated LAST, so every intermediate state stays
        in-bounds for a concurrent read."""
        if not self._ascending() or not self.at_reset_boundary():
            self.refused += 1
            return None
        try:
            if mode == "interior":
                j = self.widest_log_gap() if gap is None else int(gap)
                if j is None or not (0 <= j < self.n_cells - 1):
                    self.refused += 1
                    return None
                value = math.sqrt(self.thresholds[j] * self.thresholds[j + 1])
                index = j + 1
            elif mode == "extend":
                value = float(self.n_cells + 1)
                index = self.n_cells
            else:
                self.refused += 1
                return None
            lo = self.thresholds[index - 1]
            hi = self.thresholds[index] if index < self.n_cells else math.inf
            if not lo < value < hi:
                # A gap already collapsed to (or below) one ULP has no interior to insert into.
                self.refused += 1
                return None

            self.thresholds.insert(index, value)
            self.fired.insert(index, False)
            self.spike_tick.insert(index, None)
            self.n_cells += 1  # last: bounds tick()'s loop and sizes reset()'s arrays

            assert self._ascending(), "grow() broke the ascending-threshold invariant"
            assert (
                len(self.thresholds) == len(self.fired) == len(self.spike_tick) == self.n_cells
            ), "grow() broke length coherence"
        except Exception:
            # An exception here is a DEFECT, and must not be laundered into the refusal count -- a
            # refusal is a normal outcome and burying a defect inside one hides it (FR-013).
            self.failed += 1
            raise
        self.grown += 1
        return value

    def _ascending(self):
        return all(
            self.thresholds[i] < self.thresholds[i + 1] for i in range(len(self.thresholds) - 1)
        )

    def nominal_estimate(self, tau):
        """What this population would report for a true interval of `tau` ticks, NOISE-FREE.

        Same branch structure as decode(), evaluated against a supplied elapsed time instead of the
        live ramp. Noise-free on purpose: it asks how well the LADDER TILES time, which is the only
        thing growth can change, without the accumulator's per-trial jitter on top."""
        v = tau * self.drive
        k = _bisect.bisect_right(self.thresholds, v)
        if k == 0:
            return 0.0
        if k >= self.n_cells:
            return self.thresholds[-1] / self.drive
        return math.sqrt(self.thresholds[k - 1] * self.thresholds[k]) / self.drive

    def heldout_rmse(self, intervals):
        if not intervals:
            raise ValueError("refusing to compute an RMSE over an empty interval set")
        total = sum((self.nominal_estimate(tau) - tau) ** 2 for tau in intervals)
        return math.sqrt(total / len(intervals))

    def would_earn(self, mode="interior", gap=None, intervals=None):
        """The change in held-out error this growth WOULD cause, without performing it.

        Negative means the added cell earns its place. Returns None if the growth would be refused.
        Evaluated on a copy, so a candidate that does not earn costs nothing and leaves no trace --
        which is what lets growth be gated on benefit without needing a prune to undo a bad one.
        (There is no prune, deliberately: a population that can shrink can shrink to zero.)"""
        if intervals is None:
            _teach, intervals = interval_grid(self.thresholds)
        if not intervals:
            return None
        candidate = TimeCellClock(
            n_cells=self.n_cells, seed=self.seed, drive=self.drive, noise_sd=self.noise_sd
        )
        candidate.thresholds = list(self.thresholds)
        before = candidate.heldout_rmse(intervals)
        if candidate.grow(mode, gap=gap) is None:
            return None
        return candidate.heldout_rmse(intervals) - before

    def first_passage_sd(self, theta):
        """The accumulator's OWN dispersion at level `theta`, in ticks.

        For a Wiener process with drift mu and per-step variance sigma^2 hitting a boundary at theta,
        Var[T] = theta * sigma^2 / mu^3 -- the result this module's header already cites and whose
        consequence decode()'s `sd` already reports empirically. Used here as the RESOLUTION FLOOR."""
        if self.noise_sd <= 0.0:
            return 0.0
        return math.sqrt(theta * self.noise_sd**2 / self.drive**3)

    def resolvable(self, j):
        """Can a cell inserted in gap `j` be resolved at all, or is the gap already finer than the
        accumulator's own jitter there?"""
        lo, hi = self.thresholds[j], self.thresholds[j + 1]
        return (hi - lo) / self.drive > self.first_passage_sd(hi)

    def grow_if_earned(self, mode="interior", gap=None, intervals=None):
        """Grow only while the added cell still earns its place (FR-008), and REFUSE otherwise.

        This is the halt condition, and it is a refusal rather than a silence: a regime in which
        nothing further can be earned shows up as `refused` climbing while `grown` stands still,
        which is observable in `state()`. Inferring a halt from an absence of growth events cannot
        distinguish 'stopped because it stopped helping' from 'never ran'.

        TWO conditions, and the second one is the load-bearing one. Held-out improvement ALONE is a
        gate that cannot go red: on a noise-free tiling metric, splitting the widest gap ALWAYS
        lowers held-out error, so `delta < 0` is satisfied by every insertion forever. MEASURED
        before this was noticed: 600 live ticks grew the population 64 -> 104 with 0 refusals, which
        is not a working earn-your-place rule but an unbounded one wearing its clothes -- exactly the
        Standing-Invariant-5 failure this project keeps finding in its own instruments.

        The real bar is the accumulator's own noise. A bracket narrower than the first-passage
        dispersion at that level cannot be resolved by this population no matter how finely it is
        tiled, so a cell that only subdivides below that floor buys precision the substrate cannot
        deliver. `resolvable()` is derived entirely from existing state -- thresholds, `drive`,
        `noise_sd` -- so no benefit threshold is introduced (P1)."""
        if mode == "interior":
            j = self.widest_log_gap() if gap is None else int(gap)
            if j is None or not (0 <= j < self.n_cells - 1) or not self.resolvable(j):
                self.refused += 1
                return None
        delta = self.would_earn(mode, gap=gap, intervals=intervals)
        if delta is None or delta >= 0.0:
            self.refused += 1
            return None
        return self.grow(mode, gap=gap)

    # ---------------------------------------------------------------- persistence
    def save(self, name=STATE_NAME):
        """Persist the LEARNED state -- the threshold vector and the lineage keys that make it
        interpretable -- INTO the one durable lake. Returns True if written, False if unchanged.

        P4b VIOLATION CLOSED 2026-08-05. This paragraph twice argued the sidecar was correct, on the
        authority of `substrate/home.py`'s record-tag header -- *"weights/calibration stay separate
        STATE, never in the memory lake"* -- and that sentence was RETRACTED in `home.py` as "the
        exact arrangement P4b forbids". A threshold vector IS the learned weight set and
        `home.save_state` WAS the `json.dump` P4b names. It is no longer: `save_state` now writes a
        `_T_STATE` record into `brain_lake.bin` under this same `name`, and imports the existing
        `<home>/time_cells.json` (2,031 B on the operator's home, left on disk inert) the first time
        it is read. Not one line here changed, which is what closing a SEVEN-caller family by fixing
        the mechanism looks like from the caller's side.

        THE THREE MEASUREMENTS THAT USED TO SIT HERE were arguments against reusing `_T_GROWTH`, and
        the fix honours all three by NOT reusing it: appending a clock-shaped npz under `_T_GROWTH`
        makes `restore_growth_state()` return False and the NG-SNN silently start ungrown, swallowed
        by the bare `except Exception` around `load_growth()` in `Brain._make_neuro`;
        `growth_state_bytes()` returns None when `_model is None`, so with torch absent the clock's
        state would have no carrier; and `_T_GROWTH` dominates the lake (RE-MEASURED 2026-08-05 by a
        read-only frame scan of the live 516,810,138 B lake: 73,081 growth records / 505,367,813 B,
        97.8%, against 21,033 FACT and 29,005 EPISODE -- the "180.0 of 187.5 MB (96.0%)" that stood
        here was measured on a smaller lake and is superseded, not contradicted; re-scanned
        independently hours later at 519,646,815 B / 124,536 records the daemon had moved every
        count -- 73,490 growth / 508,082,042 B / 97.77%, 21,040 FACT, 30,005 EPISODE -- so the RATIO
        is the durable claim here and the absolute counts are a timestamp). `_T_STATE` is
        clock-sized, keyed by name, and shares nothing with the growth blob but the file.

        WHAT IT NOW COSTS, stated because it is a real regression and the alternative was staying
        non-compliant: a load is a replay of the one lake. One full pass over that 516.8 MB lake
        measured 349 / 376 / 378 ms, and 250 / 271 / 279 ms on a re-scan of the (larger) lake hours
        later -- cache- and load-dependent, so a quarter to four tenths of a second is the honest
        statement -- against the ~0.1 ms this used to pay for a 2 KB file read.
        `home._state_index` pays it ONCE per BinLog instance, so the growth path here -- which calls
        `save()` on every tick the clock earns a cell (`engine/brain.py`, in the wake tick's
        `grow_if_earned` branch) -- pays it once, not per call.

        Overwrite IS the prune policy for the clock's own state, so FR-014 is satisfied without a
        retention constant to tune (P1). At the lake level the record is append-only, and an
        UNCHANGED vector is not re-appended at all.

        Per-trial state (`v`, `fired`, `spike_tick`, `n_fired`, `t`) is deliberately NOT persisted:
        it describes an interval in progress, that interval ended when the process died, and
        restoring it against a fresh RNG would be a fabrication."""
        from snn_data_lake.substrate import home

        return home.save_state(
            name,
            {
                "fmt": STATE_FMT,
                "n_cells": int(self.n_cells),
                "seed": self.seed,
                "drive": float(self.drive),
                "noise_sd": float(self.noise_sd),
                "thresholds": [float(t) for t in self.thresholds],
            },
        )

    def restored(self, name=STATE_NAME):
        """Return the clock this one SHOULD be, given what was persisted. Never mutates `self`.

        Returns `(clock, reason)`. On success `clock` is a NEW TimeCellClock rebuilt at the persisted
        width with the persisted vector assigned -- adopt wholesale, which is the only reconciliation
        that is not a lie. Never mutate `n_cells` in place (`reset()` sizes `fired`/`spike_tick` from
        it), never merge element-wise, never truncate or pad: F3 leaves no way to reconstruct a
        learned threshold that was dropped.

        On refusal `clock` is `self` with `restore_refused` incremented and `reason` set. Refuses on
        fmt mismatch, `len(thresholds) != n_cells`, `n_cells < 1`, a seed/drive/noise_sd lineage
        mismatch, or a non-ascending vector. The lineage keys cost 24 bytes and are what makes
        refusal possible at all: `learn_interval` takes `t_pre` from `spike_tick` under THAT RNG's
        noise, so a vector learned under another seed is another lineage's state, not this one's."""
        from snn_data_lake.substrate import home

        blob = home.load_state(name, default=None)
        if not isinstance(blob, dict):
            return self, "nothing persisted"
        why = self._restore_refusal(blob)
        if why:
            self.restore_refused += 1
            return self, why
        fresh = TimeCellClock(
            n_cells=int(blob["n_cells"]),
            seed=self.seed,
            drive=self.drive,
            noise_sd=self.noise_sd,
        )
        fresh.thresholds = [float(t) for t in blob["thresholds"]]
        fresh.grown = self.grown
        fresh.refused = self.refused
        fresh.failed = self.failed
        fresh.restore_refused = self.restore_refused
        return fresh, ""

    def _restore_refusal(self, blob):
        """The reason this blob may not be adopted, or '' if it may. Split out so the refusal
        conditions are one readable list rather than a nest inside the happy path."""
        if blob.get("fmt") != STATE_FMT:
            return f"fmt mismatch: {blob.get('fmt')!r} is not {STATE_FMT!r}"
        n = blob.get("n_cells")
        if not isinstance(n, int) or n < 1:
            return f"persisted n_cells is not a positive int: {n!r}"
        vec = blob.get("thresholds")
        if not isinstance(vec, list) or len(vec) != n:
            return f"length mismatch: {len(vec) if isinstance(vec, list) else vec!r} != {n}"
        if blob.get("seed") != self.seed:
            return f"lineage mismatch: seed {blob.get('seed')!r} is not {self.seed!r}"
        if float(blob.get("drive", -1.0)) != self.drive:
            return f"lineage mismatch: drive {blob.get('drive')!r} is not {self.drive!r}"
        if float(blob.get("noise_sd", -1.0)) != self.noise_sd:
            return f"lineage mismatch: noise_sd {blob.get('noise_sd')!r} is not {self.noise_sd!r}"
        if any(vec[i] >= vec[i + 1] for i in range(len(vec) - 1)):
            return "persisted vector is not strictly ascending"
        return ""


def main():
    # ---- 1. decode() tracks true elapsed steps, monotonically, as the population fires in order ----
    clock = TimeCellClock(n_cells=64, seed=0)
    estimates = []
    for _ in range(50):
        est, _sd = clock.tick()
        estimates.append(est)
    monotonic = all(estimates[i + 1] >= estimates[i] for i in range(len(estimates) - 1))
    final_err = abs(estimates[-1] - 50)

    # ---- 2. Weber/scalar timing: sd grows with elapsed time, read off ONE trial at two points ----
    clock2 = TimeCellClock(n_cells=64, seed=0)
    sd_short = sd_long = 0.0
    for step in range(1, 51):
        _est, sd = clock2.tick()
        if step == 10:
            sd_short = sd
        if step == 50:
            sd_long = sd

    # ---- 3. learn_interval improves interval reproduction: retune toward an OFF-GRID target ----
    learner = TimeCellClock(n_cells=64, seed=1)
    target = 25.5  # deliberately between two integer thresholds
    i_before = min(
        range(learner.n_cells),
        key=lambda idx: abs(learner.thresholds[idx] / learner.drive - target),
    )
    error_pre = abs(learner.thresholds[i_before] / learner.drive - target)
    for _ in range(30):
        i_cell, _delta = learner.learn_interval(target, reward=1.0)
    error_post = abs(learner.thresholds[i_cell] / learner.drive - target)

    print("=" * 76)
    print(
        "TIME CELL CLOCK -- ramping-to-threshold population timing (geometric-mid population readout)"
    )
    print("=" * 76)
    print(
        f"1 tracking   : 50 ticks -> decode {estimates[-1]:.2f} (true 50, err {final_err:.2f}); "
        f"monotonic={monotonic}"
    )
    print(
        f"2 weber law  : sd@10 ticks={sd_short:.4f}  sd@50 ticks={sd_long:.4f}  "
        f"(ratio {sd_long / sd_short if sd_short else float('inf'):.2f}x)"
    )
    print(
        f"3 learning   : target={target}  cell#{i_before}->#{i_cell}  "
        f"error_pre={error_pre:.4f} -> error_post={error_post:.4f}"
    )
    print("=" * 76)

    # ---- self-check: deterministic asserts, ASCII ----
    assert monotonic, "decode() must track elapsed time monotonically (fired cells never un-fire)"
    assert final_err < 5.0, f"decode() must track true elapsed steps ({estimates[-1]:.2f} vs 50)"
    assert sd_long > sd_short, (
        f"Weber's law: sd must grow with elapsed time (sd@10={sd_short:.4f}, sd@50={sd_long:.4f})"
    )
    assert error_post < error_pre, (
        f"learn_interval must reduce interval-reproduction error ({error_pre:.4f} -> {error_post:.4f})"
    )

    # ---- 4. P4b: the learned thresholds live IN the one lake, proven ACROSS A PROCESS BOUNDARY ----
    # There was no persistence check here at all before 2026-08-05, while `save()` carried three
    # paragraphs arguing about WHERE to persist. A store nothing exercises is a store nothing can
    # certify, and P4b is explicit that an in-process re-read certifies nothing.
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    from snn_data_lake.substrate import home

    src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prior_home = os.environ.get("NORD_HOME")  # restored, never popped (in-process gate runs)
    work = tempfile.mkdtemp(prefix="time_cells_home_")
    try:
        os.environ["NORD_HOME"] = work
        grown = TimeCellClock(n_cells=64, seed=0)
        grown.grow("interior", gap=10)  # a learned threshold no fresh clock has
        assert grown.n_cells == 65, grown.n_cells
        assert grown.save() is True, "the first save of a grown ladder must write a lake record"
        assert grown.save() is False, "an unchanged ladder must not be re-appended to the one lake"
        assert not os.path.exists(home.state_path(STATE_NAME)), (
            "learned thresholds must go IN brain_lake.bin -- no time_cells.json sidecar"
        )
        child = (
            f"import sys; sys.path.insert(0, {src_root!r})\n"
            f"import os; os.environ['NORD_HOME'] = {work!r}\n"
            "from snn_data_lake.cortex.time_cells import TimeCellClock\n"
            "back, why = TimeCellClock(n_cells=64, seed=0).restored()\n"
            "assert why == '', why\n"
            f"assert back.thresholds == {grown.thresholds!r}, back.thresholds\n"
            "print('  4 durable   : child rebuilt %d thresholds from brain_lake.bin alone'"
            " % back.n_cells)\n"
        )
        rc = subprocess.run([sys.executable, "-c", child]).returncode
        assert rc == 0, f"the learned ladder did not survive into a SEPARATE process (rc={rc})"
    finally:
        if prior_home is None:
            os.environ.pop("NORD_HOME", None)
        else:
            os.environ["NORD_HOME"] = prior_home
        shutil.rmtree(work, ignore_errors=True)

    print(
        f"self-check OK: decode tracks elapsed monotonically (err {final_err:.2f}/50, monotonic={monotonic}); "
        f"sd grows with duration (sd@10={sd_short:.4f} -> sd@50={sd_long:.4f}, Weber law); learn_interval "
        f"cuts reproduction error {error_pre:.4f} -> {error_post:.4f}; deterministic"
    )


if __name__ == "__main__":
    main()

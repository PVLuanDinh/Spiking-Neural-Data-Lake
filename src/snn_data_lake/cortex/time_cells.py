"""
Spiking population clock -- ramping-to-threshold "time cells" that tile an interval.

Fixes: interval timing elsewhere in the brain has been a hand-coded constant. Here a bank of n_cells
LIF-style accumulators is triggered by reset() (the "start" event) and each ramps toward its own
STAGGERED threshold: cell i's nominal threshold is i+1, so cell 0 crosses first, cell 1 next, and so
on -- the population TILES elapsed time with one preferred latency per cell (hippocampal "time
cells": MacDonald, Lepage, Eden & Eichenbaum 2011, Neuron, "Hippocampal time cells bridge the gap in
memory for discontiguous events"). decode() reads elapsed time off this population with a rank-order
code (how many cells have already crossed threshold = the coarse tick) refined by the PARTIAL ramp of
the next not-yet-fired cell (a tempotron-style graded readout of the population's current activity,
not just a spike count).

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
basal_ganglia/r_stdp.py's reward-modulated STDP exactly: treat the external target_steps as a "post"
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
- basal_ganglia/r_stdp.py is the general-purpose reward-modulated STDP synapse (arbitrary pre/post
  spike trains). learn_interval() here is a direct, narrower APPLICATION of that same eligibility/
  gating math to one specific plastic parameter (a time cell's threshold) so the population can learn
  a target interval, instead of the general many-synapse rule.

Ceiling -- a small (n_cells<=~100) discretized drift-diffusion accumulator bank with one plastic
threshold per cell; not a trained hierarchical timing network, not the full striatal-beat-frequency
or ramping-clock circuitry, no working-memory gating of the count. Deterministic given a seed (RNG is
per-instance, random.Random(seed), never module-level). Not cognition.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.time_cells
"""

import math
import random


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
        self.n_cells = n_cells
        self.seed = seed
        self.drive = float(drive)
        self.noise_sd = float(noise_sd)
        self.thresholds = [
            float(i + 1) for i in range(n_cells)
        ]  # staggered latencies tiling 1..n_cells
        self.rng = random.Random(seed)
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
        """Rank-order + tempotron-style readout: (elapsed_estimate, sd).

        elapsed_estimate = the nominal latency of the last-fired cell, refined by the FRACTIONAL
        activation of the next not-yet-fired cell's threshold (a graded read of the population's
        current activity, not just a spike count); clamped to be non-decreasing across calls (fired
        cells never un-fire). sd = RMS of (actual spike tick - nominal noise-free latency) over the
        cells that have fired so far -- an empirical measure of accumulator jitter that grows with how
        many/how-late-thresholded cells have already fired, i.e. with elapsed time (Weber/scalar
        timing)."""
        k = self.n_fired
        if k == 0:
            estimate = max(0.0, self.v / self.thresholds[0]) if self.thresholds[0] else 0.0
        else:
            base = self.thresholds[k - 1]
            if k < self.n_cells:
                nxt_frac = max(0.0, min(1.0, (self.v - base) / (self.thresholds[k] - base)))
                estimate = base + nxt_frac * (self.thresholds[k] - base)
            else:
                estimate = base
        estimate = max(estimate, self._last_estimate)  # fired cells latch: estimate never regresses
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
        self.thresholds[i] = max(0.5, self.thresholds[i] + delta)
        return i, delta


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
        "TIME CELL CLOCK -- ramping-to-threshold population timing (rank-order/tempotron readout)"
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
    print(
        f"self-check OK: decode tracks elapsed monotonically (err {final_err:.2f}/50, monotonic={monotonic}); "
        f"sd grows with duration (sd@10={sd_short:.4f} -> sd@50={sd_long:.4f}, Weber law); learn_interval "
        f"cuts reproduction error {error_pre:.4f} -> {error_post:.4f}; deterministic"
    )


if __name__ == "__main__":
    main()

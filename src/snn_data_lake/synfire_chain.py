"""
Synfire chain -- stable feedforward propagation of a synchronous spike volley (pure stdlib).

A feedforward chain of L layers ("pools") of w coincidence neurons each, fully connected pool to
pool. A synchronous PULSE PACKET -- `a` spikes with a temporal spread (jitter) `sigma` -- is injected
into layer 1. Every layer-(k+1) neuron is a leaky coincidence unit: it sums the (jittered) layer-k
spikes through a membrane with leak time-constant `window`, and the first time its potential crosses
its own threshold it emits ONE output spike. Because the membrane leaks, spikes that arrive spread out
(large sigma) decay away before they can sum, while near-coincident spikes (small sigma) pile up and
fire the neuron. The packet then either

  * PROPAGATES  -- its (a, sigma) converges to a stable ATTRACTOR (near-full activation with SHARPENED
    timing, small stable sigma), or
  * EXTINGUISHES -- a -> 0 within a few layers.

A SEPARATRIX in the (a, sigma) plane splits survival from extinction: a strong, synchronous packet
survives; a weak OR temporally-dispersed one dies. Same chain, opposite fate depending only on the
injected packet. This is the landmark result that cortex can transmit precisely-timed synchronous
activity (Abeles, Corticonics 1991; Diesmann, Gewaltig & Aertsen, Nature 402:529, 1999).

Realization: an EXPLICIT LIF-pool coincidence simulation (option a). The state carried between pools is
the actual list of output spike TIMES of the previous pool's firing neurons (not a redrawn Gaussian).
Each pool integrates those times through a leaky membrane (the leak IS the coincidence window); each
neuron carries a small fixed threshold heterogeneity (`jitter_gain`, drawn once) so a graded FRACTION
of the pool fires and so output timing jitter = jitter_gain / (membrane slope at crossing) -- a steep
(coincident) rise sharpens timing, a shallow (dispersed) rise smears it. Propagation is therefore a
genuine consequence of leaky coincidence + threshold + jitter, never a hardcoded outcome.

DISTINCTNESS (kuramoto_sync / neuronal_avalanches / delay_learning / polychronization read first):
- `kuramoto_sync.py` synchronizes N continuously-COUPLED PHASE oscillators; its signature is the order
  parameter r rising through a coupling threshold (a coherence phase transition of coupled phases).
  THIS propagates a synchronous SPIKE PACKET FORWARD through a feedforward chain of pools; its signature
  is survival-vs-extinction plus jitter SHARPENING across layers -- not an order parameter, no coupling.
- `neuronal_avalanches.py` is a STOCHASTIC branching cascade whose signature is a scale-free POWER-LAW
  size distribution. A synfire volley is the OPPOSITE: a stable, near-deterministic packet locked to an
  attractor, not a scale-free cascade.
- research `delay_learning.py` / `polychronization.py` are DELAY-based coincidence groups (spikes
  converge because of matched conduction DELAYS). A synfire chain is FEEDFORWARD SYNCHRONY -- a zero-lag
  volley propagating pool to pool with no delay lines. Not `htm_sequence_memory.py` (SDR sequence
  prediction), not `theta_gamma_code.py` (discrete phase slots).

ponytail: a REDUCED feedforward synfire model -- an LIF-pool coincidence chain on a HOMOGENEOUS chain
(no recurrence, no plasticity, no real cortical connectivity), the packet-propagation LEVEL of
synchrony, not spiking biophysics; not cognition.

  python -m snn_data_lake.synfire_chain        # (also run by run_selfchecks.py)
"""
import math
import random


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    """Population standard deviation (temporal jitter of a spike volley). 0 for < 2 spikes."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


class SynfireChain:
    """Feedforward chain of L coincidence-neuron pools; propagates a synchronous pulse packet.

    Each pool has `width` LIF coincidence neurons with a small fixed threshold spread (jitter_gain).
    `propagate(a0, sigma0)` injects an `a0`-spike packet with temporal jitter `sigma0` into layer 1 and
    returns the per-layer (a, sigma) trace as the packet crosses the chain. Deterministic given seed."""

    def __init__(self, n_layers=12, width=100, threshold=30.0, window=4.0,
                 weight=1.0, jitter_gain=2.2, dt=0.25, seed=0):
        self.n_layers = n_layers
        self.width = width
        self.threshold = threshold        # base membrane firing threshold
        self.window = window              # membrane leak time-constant = the coincidence window (tau_m)
        self.weight = weight              # per-synapse EPSP amplitude J (instantaneous current per spike)
        self.jitter_gain = jitter_gain    # std of per-neuron threshold heterogeneity; with the membrane
        self.dt = dt                      #   slope it sets output jitter = jitter_gain / slope ("gain")
        self.seed = seed
        self.decay = math.exp(-dt / window)
        # fixed per-neuron threshold heterogeneity, drawn once (balanced excitability spread) -> a graded
        # fraction of the pool crosses threshold rather than all-or-none.
        rng = random.Random(seed)
        self._thresholds = [threshold + rng.gauss(0.0, jitter_gain) for _ in range(width)]

    def _inject(self, a0, sigma0):
        """Build the injected pulse packet: a0 spike times with Gaussian temporal spread sigma0."""
        n = int(round(a0))
        rng = random.Random(self.seed + 10007)     # fixed derived stream -> propagate() is idempotent
        return [rng.gauss(0.0, sigma0) for _ in range(n)]

    def _step(self, spikes):
        """One pool-to-pool transfer: incoming spike times -> outgoing spike times (recentered).

        The whole pool shares one leaky-membrane drive V(t) (homogeneous full connectivity); neurons
        differ only in threshold, so each fires at the first interpolated crossing of its own threshold.
        The leak is the coincidence mechanism: dispersed spikes decay before they sum."""
        if not spikes:
            return []
        dt, decay, w = self.dt, self.decay, self.weight
        pad = 3.0 * self.window
        t0 = min(spikes) - pad
        t1 = max(spikes) + pad
        n_steps = int((t1 - t0) / dt) + 1

        current = [0.0] * n_steps                       # bin the input spikes into per-step current
        for t in spikes:
            s = int((t - t0) / dt)
            if 0 <= s < n_steps:
                current[s] += w

        v = 0.0                                         # leaky integration of the common pool drive
        vtrace = [0.0] * n_steps
        for s in range(n_steps):
            v = v * decay + current[s]
            vtrace[s] = v

        out = []
        for theta in self._thresholds:                  # first interpolated threshold crossing per neuron
            fired = -1
            for s in range(n_steps):
                if vtrace[s] >= theta:
                    fired = s
                    break
            if fired < 0:
                continue
            if fired == 0:
                out.append(t0)
            else:
                v_lo, v_hi = vtrace[fired - 1], vtrace[fired]
                frac = (theta - v_lo) / (v_hi - v_lo) if v_hi > v_lo else 0.0
                out.append(t0 + dt * ((fired - 1) + frac))

        m = _mean(out)                                  # recenter so the packet stays near 0 (bounds grid;
        return [t - m for t in out]                     #   does not change a or sigma)

    def propagate(self, a0, sigma0):
        """Inject a packet and return the per-layer [(a, sigma), ...] trace across the whole chain.

        trace[0] is the injected packet as it enters layer 1; trace[k] is the packet after k transfers.
        Length is always n_layers (extinguished layers are padded with (0, 0.0))."""
        spikes = self._inject(a0, sigma0)
        trace = [(len(spikes), _std(spikes))]
        for _ in range(self.n_layers - 1):
            spikes = self._step(spikes)
            trace.append((len(spikes), _std(spikes)))
            if not spikes:
                break
        while len(trace) < self.n_layers:
            trace.append((0, 0.0))
        return trace

    def final_state(self, a0, sigma0):
        """(a, sigma) of the packet at the last layer of the chain."""
        return self.propagate(a0, sigma0)[-1]

    def survives(self, a0, sigma0, floor_frac=0.3):
        """True if the packet reaches the last layer with activity above floor_frac * width."""
        a_final, _ = self.final_state(a0, sigma0)
        return a_final >= floor_frac * self.width

    def sharpening(self, a0, sigma0):
        """(initial sigma, final sigma) -- how much the surviving volley's timing jitter sharpens."""
        trace = self.propagate(a0, sigma0)
        return trace[0][1], trace[-1][1]


def _fmt(trace):
    """One-line ASCII rendering of a per-layer (a, sigma) trace."""
    return " ".join(f"L{i}:({a},{sig:.2f})" for i, (a, sig) in enumerate(trace))


def main():
    chain = SynfireChain(n_layers=12, width=100, seed=0)          # tuned homogeneous chain
    w = chain.width

    # ---- 1. survival vs extinction separatrix (same chain, opposite fate) ----
    strong = chain.propagate(a0=90, sigma0=3.0)                   # high a, tight -> survives
    weak = chain.propagate(a0=35, sigma0=3.0)                     # low a, same tightness -> dies
    strong_a = strong[-1][0]
    weak_a = weak[-1][0]

    # ---- 2. self-sharpening synchrony + activity attractor (for the surviving packet) ----
    sig0, sigF = chain.sharpening(a0=90, sigma0=3.0)
    tail_a = [a for a, _ in strong[-4:]]                          # last few layers' activity (attractor?)

    # ---- 3. dispersion kills propagation: same strong count, smeared timing ----
    dispersed = chain.propagate(a0=90, sigma0=9.0)                # high a but desynchronized -> dies
    dispersed_a = dispersed[-1][0]

    # ---- 4. determinism: same params/seed -> identical per-layer trace across fresh instances ----
    det_a = SynfireChain(seed=0).propagate(a0=90, sigma0=3.0)
    det_b = SynfireChain(seed=0).propagate(a0=90, sigma0=3.0)

    print("=" * 68)
    print("SYNFIRE CHAIN -- feedforward propagation of a synchronous spike volley")
    print("=" * 68)
    print(f"chain: {chain.n_layers} pools x {w} coincidence neurons, window(tau_m)={chain.window}, "
          f"threshold={chain.threshold}, seed={chain.seed}")
    print("per-layer (a, sigma) traces  [a = active neurons, sigma = spike-time jitter]:")
    print(f"  STRONG  (a0=90, sigma0=3.0): {_fmt(strong)}")
    print(f"  WEAK    (a0=35, sigma0=3.0): {_fmt(weak)}")
    print(f"  SMEARED (a0=90, sigma0=9.0): {_fmt(dispersed)}")
    print(f"survival separatrix : strong final a={strong_a}/{w} (survives) vs "
          f"weak final a={weak_a}/{w} (extinct)")
    print(f"self-sharpening     : sigma {sig0:.2f} (injected) -> {sigF:.2f} (locked); "
          f"activity attractor tail a={tail_a}")
    print(f"dispersion kills    : a0=90 but sigma0=9.0 -> final a={dispersed_a}/{w} "
          f"(synchrony, not count, carries the packet)")
    print("=" * 68)

    # ---- self-check (ponytail: deterministic asserts, zero-dep, ASCII-only) ----
    assert strong_a > 0.7 * w, f"strong synchronous packet must propagate near saturation ({strong_a}/{w})"
    assert weak_a == 0, f"weak packet must extinguish to zero activity ({weak_a})"
    assert strong_a > 0.7 * w and weak_a == 0, "same chain must give opposite fates for strong vs weak"
    assert sigF < sig0 - 0.5, f"surviving packet's timing must sharpen ({sig0:.2f} -> {sigF:.2f})"
    assert max(tail_a) - min(tail_a) <= 2, f"activity must settle to an attractor (tail {tail_a})"
    assert dispersed_a == 0, f"a temporally-dispersed strong packet must fail to propagate ({dispersed_a})"
    assert det_a == det_b, "same params/seed must give an identical per-layer trace"
    print(f"self-check OK: separatrix strong a={strong_a}(survives)/weak a={weak_a}(extinct); "
          f"sharpening sigma {sig0:.2f}->{sigF:.2f}, attractor tail {tail_a}; dispersion kills "
          f"(a0=90/sigma0=9 -> a={dispersed_a}); deterministic")


if __name__ == "__main__":
    main()

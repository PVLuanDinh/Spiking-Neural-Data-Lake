"""
Neuronal avalanches / critical branching -- scale-free cascade statistics with the branching ratio
as the explicit control parameter (pure stdlib).

A homogeneous network of N units. An AVALANCHE is a cascade seeded by ONE spontaneously-active unit:
at each generation every currently-active unit independently activates some downstream units, tuned so
the EXPECTED descendants per active unit equals the branching ratio sigma -- THE control parameter.
Each active unit spawns Binomial(fanout, sigma/fanout) descendants (mean exactly sigma, finite nonzero
variance); the next generation's active count is that sum, capped at the available N so a cascade can
never exceed the network. The avalanche runs until no unit is active; its SIZE = total activations and
its DURATION = number of generations. This is the reduced Galton-Watson / mean-field reading of the
Beggs & Plenz cortical-avalanche result (J Neurosci 23:11167, 2003; branching interpretation after
Harris's Galton-Watson process). sigma is the knob:

    sigma < 1 (subcritical)  : cascades die out fast -> small avalanches, an approximately exponential
                               size tail;
    sigma = 1 (critical)     : scale-free -> the avalanche-size distribution is an approximate POWER
                               LAW P(s)~s^-1.5 (the mean-field Galton-Watson exponent) spanning many
                               orders of magnitude;
    sigma > 1 (supercritical): cascades tend to explode -> giant avalanches bounded only by N and a
                               runaway size cutoff (a heavy / bimodal upper tail).

DISTINCTNESS (read wilson_cowan / kuramoto_sync before writing this):
- `wilson_cowan.py` is a DETERMINISTIC two-population rate ODE integrated with forward-Euler whose
  signature is ATTRACTOR structure -- a stable fixed point, a limit cycle, OR bistable hysteresis under
  hand-picked parameters. THIS is a STOCHASTIC branching process whose signature is the AVALANCHE-SIZE
  DISTRIBUTION and its POWER LAW at criticality, with the branching ratio sigma as an explicit control
  parameter driving a subcritical -> critical -> supercritical transition. A different object entirely:
  critical statistics of random cascades, not fixed points of a rate field. No ODE and no attractor
  here; no branching ratio, no avalanche statistics and no power law there.
- `kuramoto_sync.py` is phase-oscillator SYNCHRONIZATION (order parameter r) -- a coherence transition
  of N coupled phases. Not that: not phases, not an order parameter, not a single-neuron model. This is
  network-level CRITICALITY / scale-free cascade statistics (Beggs-Plenz), the regime that maximizes
  dynamic range -- absent from the core until now.

A REDUCED Galton-Watson / mean-field branching process on a HOMOGENEOUS network -- no spatial
structure, no real synaptic weights, no refractory dynamics -- the avalanche-STATISTICS level of
criticality, not spiking biophysics. Small enough scale (modest N and avalanche counts) that the
fixed-seed statistics are reproducible; a runaway size cutoff terminates supercritical cascades. Not
cognition.

  python -m snn_data_lake.cortex.neuronal_avalanches        # (also run by run_selfchecks.py)
"""
import math
import random


def _lstsq_slope(xs, ys):
    """Least-squares slope of ys vs xs (both already log-transformed). Needs >= 2 points."""
    m = len(xs)
    mx = sum(xs) / m
    my = sum(ys) / m
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


class BranchingNetwork:
    """Galton-Watson branching network: cascades of unit activations with mean offspring = sigma.

    avalanche() runs ONE cascade from a single seed unit and returns (size, duration). run(k) runs k
    cascades, returns their (size, duration) records, and stores a per-generation (ancestors,
    descendants) trace in `self.branching_trace` for measured_branching_ratio(). Each active unit
    spawns Binomial(fanout, sigma/fanout) descendants (mean exactly sigma); the next generation's
    active count is that sum capped at N. A cascade ends on extinction OR at the runaway size cutoff
    `max_size` (which bounds supercritical giants). Deterministic given seed: a private
    random.Random(seed) drives every Bernoulli trial."""

    def __init__(self, n=1000, sigma=1.0, fanout=4, seed=0, max_size=4000):
        if fanout < 1:
            raise ValueError("fanout must be >= 1")
        if not 0.0 <= sigma <= fanout:
            raise ValueError("need 0 <= sigma <= fanout so p = sigma/fanout is a probability")
        self.n = n
        self.sigma = sigma
        self.fanout = fanout
        self.p = sigma / fanout          # per-target activation prob; Binomial(fanout, p) has mean sigma
        self.seed = seed
        self.max_size = max_size         # runaway cutoff: truncate a cascade once it reaches this size
        self.rng = random.Random(seed)
        self.branching_trace = []        # list of (ancestors, descendants) generation pairs from run()

    def _one(self):
        """One cascade. Returns (gen_active, gen_raw): per-generation active counts (post-cap; first
        entry = 1, the seed) and the RAW descendants those units spawned (pre-cap). Ends on extinction
        (active -> 0) or the runaway size cutoff. Raw (pre-cap) descendants are what makes the measured
        branching ratio an unbiased estimate of sigma even where the per-generation cap N binds."""
        rnd = self.rng.random
        p, fanout, cap, max_size = self.p, self.fanout, self.n, self.max_size
        gen_active, gen_raw = [], []
        active = 1
        size = 0
        while active > 0:
            gen_active.append(active)
            size += active
            raw = 0
            for _ in range(active * fanout):     # active units x fanout Bernoulli(p) = Binomial(., p)
                if rnd() < p:
                    raw += 1
            gen_raw.append(raw)
            if size >= max_size:                 # runaway (supercritical): truncate the giant cascade
                break
            active = raw if raw <= cap else cap   # cap the active set at the available network size N
        return gen_active, gen_raw

    def avalanche(self):
        """Run one cascade from a single seed unit; return (size, duration)."""
        gen_active, _ = self._one()
        return sum(gen_active), len(gen_active)

    def run(self, n_avalanches, trace=True):
        """Run n_avalanches cascades; return a list of (size, duration). When trace is True, store the
        flat per-generation (ancestors, descendants) pairs in self.branching_trace."""
        records = []
        tr = []
        for _ in range(n_avalanches):
            gen_active, gen_raw = self._one()
            records.append((sum(gen_active), len(gen_active)))
            if trace:
                tr.extend(zip(gen_active, gen_raw))
        self.branching_trace = tr
        return records

    @staticmethod
    def measured_branching_ratio(trace):
        """Empirical branching ratio = sum(descendants) / sum(ancestors) over a (ancestors,
        descendants) per-generation trace (as recorded in self.branching_trace). Approaches sigma."""
        tot_a = 0
        tot_d = 0
        for a, d in trace:
            tot_a += a
            tot_d += d
        return tot_d / tot_a if tot_a else 0.0

    @staticmethod
    def size_distribution(sizes, normalize=False):
        """Histogram of avalanche sizes -> {size: count}, or {size: probability} when normalize."""
        dist = {}
        for s in sizes:
            dist[s] = dist.get(s, 0) + 1
        if normalize:
            total = sum(dist.values())
            return {s: c / total for s, c in dist.items()}
        return dist

    @staticmethod
    def powerlaw_slope(sizes, s_min, s_max, n_bins=12):
        """Least-squares log-log slope of the avalanche-size DENSITY over the scaling range
        [s_min, s_max]. Sizes are collected into n_bins logarithmically-spaced bins; each bin's density
        is count / bin_width, and the slope is a least-squares fit of log(density) vs log(geometric bin
        center) over the bins that have support. Dividing by bin width is what makes a true
        P(s)~s^exponent recover a fitted slope ~ exponent (raw log-binned counts would be off by +1).
        Returns (slope, n_bins_with_support)."""
        dist = BranchingNetwork.size_distribution(sizes)          # {size: count}
        lo, hi = math.log(s_min), math.log(s_max)
        edges = [math.exp(lo + (hi - lo) * i / n_bins) for i in range(n_bins + 1)]
        xs, ys = [], []
        for i in range(n_bins):
            a, b = edges[i], edges[i + 1]
            count = sum(c for s, c in dist.items() if a <= s < b)
            if count > 0:
                xs.append(math.log(math.sqrt(a * b)))             # geometric bin center
                ys.append(math.log(count / (b - a)))              # density = count / bin width
        if len(xs) < 2:
            return 0.0, len(xs)
        return _lstsq_slope(xs, ys), len(xs)


def main():
    n, fanout, seed, max_size = 1000, 4, 0, 4000
    n_test1, n_scaling = 4000, 20000       # avalanches per sigma (test 1); critical/subcritical run
    fit_min, fit_max = 4, 256              # power-law scaling range for the log-log slope fit

    # ---- 1. branching ratio is the real control parameter: measured sigma matches, mean size grows ----
    rows = []                              # (sigma, measured_sigma, mean_size, max_size)
    for sigma in (0.8, 1.0, 1.2):
        net = BranchingNetwork(n=n, sigma=sigma, fanout=fanout, seed=seed, max_size=max_size)
        records = net.run(n_test1)
        meas = BranchingNetwork.measured_branching_ratio(net.branching_trace)
        sizes = [s for s, _ in records]
        rows.append((sigma, meas, sum(sizes) / len(sizes), max(sizes)))
    sub_mean, crit_mean, super_mean = rows[0][2], rows[1][2], rows[2][2]

    # ---- 2. scale-free avalanches at criticality: power-law slope + critical tail >> subcritical ----
    crit = BranchingNetwork(n=n, sigma=1.0, fanout=fanout, seed=seed, max_size=max_size)
    crit_sizes = [s for s, _ in crit.run(n_scaling, trace=False)]
    slope, n_bins = BranchingNetwork.powerlaw_slope(crit_sizes, fit_min, fit_max)
    crit_max = max(crit_sizes)
    sub = BranchingNetwork(n=n, sigma=0.8, fanout=fanout, seed=seed, max_size=max_size)
    sub_sizes = [s for s, _ in sub.run(n_scaling, trace=False)]
    sub_max = max(sub_sizes)
    tail_ratio = crit_max / sub_max

    # ---- 3. determinism: same n/sigma/seed -> identical records across fresh instances ----
    det_a = BranchingNetwork(n=n, sigma=1.0, fanout=fanout, seed=seed, max_size=max_size).run(200)
    det_b = BranchingNetwork(n=n, sigma=1.0, fanout=fanout, seed=seed, max_size=max_size).run(200)
    identical = det_a == det_b

    # one-cascade API demo: 20 single avalanche() draws at criticality, report the largest
    sampler = BranchingNetwork(n=n, sigma=1.0, fanout=fanout, seed=seed, max_size=max_size)
    biggest = max((sampler.avalanche() for _ in range(20)), key=lambda sd: sd[0])

    print("=" * 60)
    print("NEURONAL AVALANCHES -- critical branching process (Beggs-Plenz / Galton-Watson)")
    print("=" * 60)
    print(f"network N={n}, fanout={fanout}, offspring ~ Binomial(fanout, sigma/fanout) [mean=sigma], "
          f"seed={seed}")
    print(f"runaway size cutoff={max_size}; test-1 avalanches/sigma={n_test1}; "
          f"critical/subcritical run={n_scaling}")
    print("")
    print(" sigma  regime          measured-sigma   mean size    max size")
    regime = {0.8: "subcritical", 1.0: "critical", 1.2: "supercritical"}
    for sg, meas, mean_s, max_s in rows:
        print(f" {sg:4.2f}  {regime[sg]:<13}      {meas:6.3f}       {mean_s:8.2f}    {max_s:8d}")
    print("")
    print("critical size distribution: approx power law P(s)~s^-1.5")
    print(f"  log-log slope over s in [{fit_min}, {fit_max}] = {slope:.3f} "
          f"({n_bins} bins, mean-field exponent -1.5)")
    print(f"critical max avalanche {crit_max} vs subcritical max {sub_max} -> "
          f"{tail_ratio:.1f}x larger tail")
    print(f"one-cascade API: largest of 20 avalanche() draws (size, duration) = {biggest}")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep, ASCII-only ----
    for sg, meas, _mean_s, _max_s in rows:
        assert abs(meas - sg) < 0.05, f"measured sigma must match set sigma ({meas:.3f} vs {sg})"
    assert sub_mean < crit_mean < super_mean, \
        f"mean size must grow with sigma ({sub_mean:.2f} -> {crit_mean:.2f} -> {super_mean:.2f})"
    assert crit_mean > 5.0 * sub_mean, \
        f"critical mean size must dwarf subcritical ({crit_mean:.2f} vs {sub_mean:.2f})"
    assert super_mean > 3.0 * crit_mean, \
        f"supercritical mean size must dwarf critical ({super_mean:.2f} vs {crit_mean:.2f})"
    assert -2.1 < slope < -1.1, \
        f"critical size distribution must be an approx power law near -1.5 (slope {slope:.3f})"
    assert n_bins >= 6, f"power-law fit must span a real scaling range ({n_bins} bins)"
    assert crit_max > 20 * sub_max, \
        f"critical tail must be orders of magnitude above subcritical ({crit_max} vs {sub_max})"
    assert identical, "same n/sigma/seed must give identical records"
    print(f"self-check OK: measured-sigma matches set sigma (err<0.05); mean size "
          f"{sub_mean:.1f}<<{crit_mean:.1f}<<{super_mean:.1f}; critical P(s)~s^{slope:.2f} over "
          f"[{fit_min},{fit_max}]; critical tail {tail_ratio:.0f}x subcritical; deterministic")


if __name__ == "__main__":
    main()

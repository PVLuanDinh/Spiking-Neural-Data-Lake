"""
Kuramoto coupled-oscillator synchronization -- emergent population phase-locking (pure stdlib).

A population of N phase oscillators, each with its own natural frequency omega_i (sampled from a
fixed seeded Gaussian, centered at 0), all-to-all sine-coupled with strength K:

    dtheta_i/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i)

The COLLECTIVE state is read off the order parameter  r * e^{i*psi} = (1/N) * sum_j e^{i*theta_j}:
r in [0,1] measures synchrony -- r ~ 0 = incoherent (every oscillator runs at its own frequency,
phases scattered), r -> 1 = phase-locked. The point is emergent: above a CRITICAL coupling K_c the
population spontaneously synchronizes (r rises from the finite-N floor ~1/sqrt(N)); below K_c it stays
near that floor. That is a phase transition, not a designed rhythm (Kuramoto 1975; Strogatz, Physica D
143:1, 2000). This module integrates the exact MEAN-FIELD form of the all-to-all sum,

    dtheta_i/dt = omega_i + K * r * sin(psi - theta_i)

(an exact identity for all-to-all coupling, not an approximation -- since (1/N)sum_j sin(theta_j-theta_i)
= r*sin(psi-theta_i)), with a fixed Euler step. This is O(N) per step.

DISTINCTNESS (read cpg_oscillator / ring_attractor / theta_gamma_code before writing this):
- `cpg_oscillator.py` is a Matsuoka HALF-CENTER oscillator: two mutually-inhibiting adapting neurons
  producing ONE anti-phase rhythm to DRIVE the arm -- a designed 2-unit rhythm GENERATOR. THIS is a
  POPULATION of N coupled phase oscillators whose COLLECTIVE synchronization EMERGES above a coupling
  threshold, measured by the order parameter r -- a different phenomenon (emergent phase transition,
  not a hand-built 2-cell rhythm).
- Not `ring_attractor.py` (a single 1D spatial bump), not `theta_gamma_code.py` (discrete phase SLOTS
  for working memory). This is continuous coupled-oscillator dynamics plus the synchronization order
  parameter -- neural-oscillation coherence / communication-through-coherence.

Mean-field / all-to-all Kuramoto phase oscillators with a fixed Euler step on synthetic
(seeded Gaussian) natural frequencies -- a reduced synchronization model, NOT biophysical spiking
neurons, no conduction delays, no network topology beyond all-to-all; not cognition.

  python -m snn_data_lake.cerebellum.kuramoto_sync        # (also run by run_selfchecks.py)
"""
import math
import random

from snn_data_lake.substrate._mathutils import mean


def critical_coupling(freq_spread):
    """Analytic K_c for a Gaussian natural-frequency distribution (std = freq_spread).

    General mean-field result K_c = 2 / (pi * g(0)); for a Gaussian g(0) = 1/(sigma*sqrt(2*pi)),
    so K_c = 2 * sigma * sqrt(2/pi) ~= 1.5958 * sigma. Below K_c the population stays incoherent;
    above it r grows from the finite-N floor. (For a Lorentzian of half-width gamma the exact
    result is the cleaner K_c = 2*gamma; we use Gaussian frequencies here.)"""
    return 2.0 * freq_spread * math.sqrt(2.0 / math.pi)


def _stdev(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


class KuramotoNetwork:
    """N all-to-all sine-coupled phase oscillators; holds phases theta, reports the order parameter r.

    omega_i are drawn ONCE from a seeded Gaussian centered at 0 (std = freq_spread); initial phases are
    seeded-uniform on [0, 2*pi) (scattered -> genuine incoherent start). step() advances one exact
    mean-field Euler update and returns the current r; run() returns the r time series."""

    def __init__(self, n=256, coupling=1.0, freq_spread=1.0, seed=0, dt=0.05):
        self.n = n
        self.coupling = coupling
        self.freq_spread = freq_spread
        self.seed = seed
        self.dt = dt
        rng = random.Random(seed)
        # symmetric unimodal frequencies centered at 0 -> analytic K_c; Gaussian has no heavy tails,
        # so above K_c the population locks crisply.
        self.omega = [rng.gauss(0.0, freq_spread) for _ in range(n)]
        self._init_phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(n)]
        self.theta = list(self._init_phases)
        self._series = []

    @property
    def phases(self):
        return self.theta

    @property
    def frequencies(self):
        return self.omega

    def reset(self):
        """Restore the seeded initial phases (same omega); clears the stored r series."""
        self.theta = list(self._init_phases)
        self._series = []

    def order_parameter(self):
        """Return (r, psi) with r*e^{i*psi} = (1/N) sum_j e^{i*theta_j}. r in [0,1] = coherence."""
        c = sum(math.cos(t) for t in self.theta)
        s = sum(math.sin(t) for t in self.theta)
        return math.sqrt(c * c + s * s) / self.n, math.atan2(s, c)

    def instantaneous_frequencies(self):
        """dtheta_i/dt at the current state. Locked oscillators share a common value (entrainment)."""
        r, psi = self.order_parameter()
        kr = self.coupling * r
        return [self.omega[i] + kr * math.sin(psi - self.theta[i]) for i in range(self.n)]

    def step(self, dt=None):
        """One exact mean-field Euler update. Returns the order parameter r that drove this step."""
        dt = self.dt if dt is None else dt
        r, psi = self.order_parameter()
        kr = self.coupling * r
        self.theta = [self.theta[i] + dt * (self.omega[i] + kr * math.sin(psi - self.theta[i]))
                      for i in range(self.n)]
        return r

    def run(self, steps, dt=None):
        """Advance `steps` Euler updates; store and return the r time series (r before each step)."""
        dt = self.dt if dt is None else dt
        self._series = [self.step(dt) for _ in range(steps)]
        return self._series

    def mean_r(self, last_frac=0.25):
        """Steady-state r: mean of the last `last_frac` of the most recent run's r series."""
        if not self._series:
            raise RuntimeError("call run() before mean_r()")
        tail = max(1, int(len(self._series) * last_frac))
        return mean(self._series[-tail:])


def _steady_r(n, coupling, freq_spread, seed, steps, dt, last_frac=0.25):
    """Build a fresh network at this coupling and return its steady-state (tail-mean) r."""
    net = KuramotoNetwork(n=n, coupling=coupling, freq_spread=freq_spread, seed=seed, dt=dt)
    net.run(steps, dt)
    return net.mean_r(last_frac)


def main():
    n, seed, dt, steps = 256, 0, 0.05, 1200
    spread = 1.0
    k_c = critical_coupling(spread)                       # ~= 1.596 for spread = 1.0
    floor = 1.0 / math.sqrt(n)                            # finite-N incoherent order-parameter floor

    # ---- 1. coupling-threshold transition: same seeded frequencies, low K vs high K ----
    k_low, k_high = 0.40, 5.00                            # ~0.25*K_c  and  ~3.1*K_c
    r_low = _steady_r(n, k_low, spread, seed, steps, dt)
    net_high = KuramotoNetwork(n=n, coupling=k_high, freq_spread=spread, seed=seed, dt=dt)
    net_high.run(steps, dt)
    r_high = net_high.mean_r()

    # ---- 2. monotone-in-coupling sweep spanning K_c ----
    sweep_k = [0.0, 0.8, 1.6, 3.2, 6.4]
    sweep_r = [_steady_r(n, k, spread, seed, steps, dt) for k in sweep_k]

    # ---- 3. identical oscillators (zero spread) fully synchronize for any K > 0 ----
    ident = KuramotoNetwork(n=n, coupling=2.0, freq_spread=0.0, seed=seed, dt=dt)
    ident.run(steps, dt)
    r_ident, _ = ident.order_parameter()

    # ---- 4. frequency entrainment: instantaneous-frequency spread collapses at high K ----
    spread_k0 = _stdev(net_high.omega)                    # at K=0 the inst. freq IS omega
    spread_locked = _stdev(net_high.instantaneous_frequencies())
    collapse = spread_locked / spread_k0

    # ---- 5. determinism: same n/seed/coupling -> identical r series across fresh instances ----
    det_a = KuramotoNetwork(n=n, coupling=k_high, freq_spread=spread, seed=seed, dt=dt).run(200, dt)
    det_b = KuramotoNetwork(n=n, coupling=k_high, freq_spread=spread, seed=seed, dt=dt).run(200, dt)

    print("=" * 60)
    print("KURAMOTO SYNC -- emergent phase-locking of N coupled oscillators (order parameter r)")
    print("=" * 60)
    print(f"N={n} oscillators, Gaussian omega spread={spread:.2f}, K_c(analytic)={k_c:.2f}")
    print("coupling K   ->   steady-state r")
    for k, r in zip(sweep_k, sweep_r):
        print(f"   {k:5.2f}     ->   {r:.3f}")
    print(f"transition : mean_r(low K={k_low:.2f})={r_low:.3f}  vs  "
          f"mean_r(high K={k_high:.2f})={r_high:.3f}   (incoherent floor 1/sqrt(N)={floor:.3f})")
    print(f"identical oscillators (spread=0), K=2.0 : r={r_ident:.3f} (full phase-lock)")
    print(f"entrainment at high K : inst-freq stdev {spread_k0:.3f} (K=0) -> "
          f"{spread_locked:.3f} (locked), collapse x{collapse:.3f}")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert r_low < 0.30, f"below K_c the population must stay near the incoherent floor ({r_low:.3f})"
    assert r_high > 0.80, f"above K_c the population must synchronize ({r_high:.3f})"
    assert r_high > 3.0 * r_low, f"high-K r must be far above low-K r ({r_high:.3f} vs {r_low:.3f})"
    for a, b in zip(sweep_r, sweep_r[1:]):
        assert b >= a - 0.08, f"steady-state r must be (weakly) increasing in K ({a:.3f} -> {b:.3f})"
    assert sweep_r[-1] > sweep_r[0], f"r must rise across the sweep ({sweep_r[0]:.3f} -> {sweep_r[-1]:.3f})"
    assert r_ident > 0.99, f"identical oscillators must fully phase-lock ({r_ident:.3f})"
    assert collapse < 0.25, f"high-K must entrain instantaneous frequencies ({collapse:.3f})"
    assert det_a == det_b, "same n/seed/coupling must give an identical r series"
    print(f"self-check OK: transition r {r_low:.3f}(low)->{r_high:.3f}(high) above floor {floor:.3f}; "
          f"monotone in K; identical-osc r={r_ident:.3f}; freq-spread collapse x{collapse:.3f}; "
          f"deterministic")


if __name__ == "__main__":
    main()

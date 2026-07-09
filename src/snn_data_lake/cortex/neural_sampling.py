"""
Neural sampling -- a recurrent net of stochastic spiking neurons whose spikes are MCMC samples
from a Boltzmann distribution (probabilistic inference in spikes; pure stdlib).

The neural-sampling hypothesis (Buesing, Bill, Nessler & Maass, "Neural Dynamics as Sampling",
PLoS Comput Biol 7:e1002211, 2011): a network of N stochastic binary neurons z in {0,1}^N with
SYMMETRIC weights W (W_ij = W_ji, W_ii = 0) and biases b samples, in its stationary distribution,
the Boltzmann joint
    p(z) proportional to exp( sum_i b_i z_i + (1/2) sum_ij W_ij z_i z_j ).
The reduced dynamics = single-site (Gibbs) updates on stochastic sigmoid neurons: to update neuron
i, form its membrane potential u_i = b_i + sum_j W_ij z_j and spike (z_i := 1) with probability
sigmoid(u_i), else z_i := 0. Because sigmoid(u_i) is EXACTLY the Boltzmann conditional
p(z_i = 1 | z_rest), asynchronous single-site updates leave p(z) invariant, so the spike train is
an exact-in-the-limit sample from the joint. Clamping a subset of units to evidence turns the same
chain into posterior sampling p(z_free | z_clamped) -- Bayesian inference computed by spiking.

Capability: probabilistic inference in spikes -- a NEW class for the lake (posterior sampling from a
defined generative joint), distinct from every existing module:
  - research/bayesian_snn.py does MC-dropout-style CALIBRATED UNCERTAINTY: it runs a discriminative
    readout several times under random Bernoulli weight masks and reports the output disagreement as
    an uncertainty bar. It never defines or samples a joint distribution. THIS module is the opposite
    object: the recurrent stochastic dynamics ARE an MCMC chain whose stationary distribution is a
    Boltzmann p(z); it returns posterior samples, not a variance heuristic. Different object (a
    defined joint p(z) vs ensemble variance), different capability (posterior sampling vs error bars).
  - Not valence_stdp / plasticity (no learning rule here -- W and b are given).
  - Not htm_sequence_memory (sequence prediction), not ring_attractor / population coding.

The ceiling is the abstract discrete-time stochastic-binary-neuron neural-sampling chain
(single-site Gibbs updates with sigmoid firing) -- the reduced model of Buesing 2011, NOT the full
continuous-time absolute-refractory spiking dynamics; no biophysics, no learning. Networks are kept
small so the exact Boltzmann distribution can be enumerated (2^N states) for the correctness test.

  python -m snn_data_lake.cortex.neural_sampling        # (also run by run_selfchecks.py)
"""
import math
import random

from snn_data_lake.substrate._mathutils import sigmoid


class BoltzmannSampler:
    """Stochastic spiking network that samples p(z) prop exp(sum b_i z_i + 1/2 sum W_ij z_i z_j).

    Asynchronous single-site Gibbs: each step picks one free neuron and spikes it with probability
    sigmoid(u_i), the exact Boltzmann conditional -- so the stationary distribution is the Boltzmann
    joint. `clamp` fixes units to evidence, turning the chain into a posterior sampler.
    """

    def __init__(self, weights, biases, seed=0):
        self.n = len(biases)
        if any(len(row) != self.n for row in weights):
            raise ValueError("weights must be an N x N matrix matching biases")
        for i in range(self.n):
            if weights[i][i] != 0:
                raise ValueError("weights must have zero diagonal (W_ii = 0)")
            for j in range(i + 1, self.n):
                if weights[i][j] != weights[j][i]:
                    raise ValueError("weights must be symmetric (W_ij = W_ji)")
        self.W = [list(map(float, row)) for row in weights]
        self.b = [float(x) for x in biases]
        self.rng = random.Random(seed)
        self._z = [0] * self.n            # deterministic all-zeros start
        self._clamped = {}                # index -> fixed value
        self._free = list(range(self.n))

    # ---- state ---------------------------------------------------------------------------------
    @property
    def state(self):
        """Current binary state z as a tuple."""
        return tuple(self._z)

    def clamp(self, indices, values):
        """Fix `indices` to `values` (evidence); free units are then sampled given the clamp."""
        for i, v in zip(indices, values):
            self._clamped[i] = int(v)
            self._z[i] = int(v)
        self._free = [i for i in range(self.n) if i not in self._clamped]

    def unclamp(self):
        """Release all clamps."""
        self._clamped = {}
        self._free = list(range(self.n))

    # ---- dynamics ------------------------------------------------------------------------------
    def _potential(self, i):
        return self.b[i] + sum(self.W[i][j] * self._z[j] for j in range(self.n))

    def step(self):
        """One asynchronous single-site update; returns the neuron index if it spiked, else None."""
        if not self._free:
            return None
        i = self._free[self.rng.randrange(len(self._free))]
        self._z[i] = 1 if self.rng.random() < sigmoid(self._potential(i)) else 0
        return i if self._z[i] == 1 else None

    def run(self, n_steps):
        """Advance the chain n_steps single-site updates; returns the final state tuple."""
        for _ in range(n_steps):
            self.step()
        return self.state

    def sample(self, n_samples, burn_in=2000, thin=None):
        """Burn in, then collect n_samples states, one every `thin` steps (default one sweep = N)."""
        if thin is None:
            thin = max(1, self.n)
        for _ in range(burn_in):
            self.step()
        out = []
        for _ in range(n_samples):
            for _ in range(thin):
                self.step()
            out.append(self.state)
        return out

    # ---- analytic Boltzmann (enumeration; small N only) ----------------------------------------
    def _score(self, z):
        """The exponent sum_i b_i z_i + 1/2 sum_ij W_ij z_i z_j (= sum_i<j W_ij z_i z_j)."""
        s = 0.0
        for i in range(self.n):
            if z[i]:
                s += self.b[i]
                for j in range(i + 1, self.n):
                    if z[j]:
                        s += self.W[i][j]
        return s

    def energy(self, z):
        """Boltzmann/Ising energy E(z) = -score(z); p(z) prop exp(-E(z))."""
        return -self._score(z)

    def _states(self, clamp):
        """Yield every state (as a tuple) consistent with the clamp dict (2^N enumeration)."""
        for code in range(1 << self.n):
            z = tuple((code >> i) & 1 for i in range(self.n))
            if all(z[k] == v for k, v in clamp.items()):
                yield z

    def analytic_distribution(self, clamp=None):
        """Exact normalized Boltzmann distribution over states consistent with `clamp` (dict)."""
        clamp = clamp or {}
        zs = list(self._states(clamp))
        ws = [math.exp(self._score(z)) for z in zs]
        total = sum(ws)
        return {z: w / total for z, w in zip(zs, ws)}

    def prob(self, z):
        """Exact normalized Boltzmann probability of a single full state z."""
        return self.analytic_distribution()[tuple(z)]


# ---- empirical estimators (module-level helpers) -----------------------------------------------
def empirical_distribution(samples):
    """Sampled states -> {state: frequency} empirical distribution."""
    total = len(samples)
    counts = {}
    for z in samples:
        counts[z] = counts.get(z, 0) + 1
    return {z: c / total for z, c in counts.items()}


def marginals(samples):
    """Sampled states -> per-neuron empirical marginal P(z_i = 1)."""
    n = len(samples[0])
    total = len(samples)
    return [sum(z[i] for z in samples) / total for i in range(n)]


def _tv(p, q):
    """Total-variation distance between two distributions over states."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _analytic_marginals(dist):
    """Per-neuron P(z_i = 1) from an enumerated distribution dict."""
    n = len(next(iter(dist)))
    return [sum(pr for z, pr in dist.items() if z[i]) for i in range(n)]


def _corr(dist, i, j):
    """E[z_i z_j] under an enumerated distribution dict."""
    return sum(pr for z, pr in dist.items() if z[i] and z[j])


def _emp_corr(samples, i, j):
    """Empirical E[z_i z_j]."""
    return sum(z[i] and z[j] for z in samples) / len(samples)


def main():
    # A small non-trivial 3-neuron network: correlated units, no symmetry in the marginals.
    weights = [
        [0.0, 1.0, -0.5],
        [1.0, 0.0, 0.8],
        [-0.5, 0.8, 0.0],
    ]
    biases = [0.5, -0.2, 0.1]

    sampler = BoltzmannSampler(weights, biases, seed=0)
    analytic = sampler.analytic_distribution()                 # exact p(z) over all 2^3 states
    samples = sampler.sample(40000, burn_in=3000, thin=3)      # spikes = MCMC samples of p(z)
    empirical = empirical_distribution(samples)
    tv = _tv(empirical, analytic)

    a_marg = _analytic_marginals(analytic)
    e_marg = marginals(samples)
    marg_err = max(abs(a - e) for a, e in zip(a_marg, e_marg))

    a_corr = _corr(analytic, 0, 1)                             # E[z0 z1] analytic
    e_corr = _emp_corr(samples, 0, 1)                          # E[z0 z1] empirical
    corr_err = abs(a_corr - e_corr)

    # Conditional inference: clamp z2 = 1 (evidence) -> posterior p(z0, z1 | z2 = 1).
    cond = BoltzmannSampler(weights, biases, seed=1)
    cond.clamp([2], [1])
    cond_analytic = cond.analytic_distribution({2: 1})
    cond_samples = cond.sample(40000, burn_in=3000, thin=3)
    cond_empirical = empirical_distribution(cond_samples)
    cond_tv = _tv(cond_empirical, cond_analytic)
    ca_marg = _analytic_marginals(cond_analytic)               # includes z2 == 1 exactly
    ce_marg = marginals(cond_samples)
    cond_marg_err = max(abs(ca_marg[i] - ce_marg[i]) for i in (0, 1))

    print("=" * 66)
    print("NEURAL SAMPLING -- spikes are MCMC samples of a Boltzmann p(z)")
    print("=" * 66)
    print("state      analytic p(z)   empirical   energy")
    for code in range(1 << sampler.n):
        z = tuple((code >> i) & 1 for i in range(sampler.n))
        print(f"  {''.join(map(str, z))}         {analytic[z]:.4f}        "
              f"{empirical.get(z, 0.0):.4f}     {sampler.energy(z):+.3f}")
    print("-" * 66)
    print(f"TV(empirical, Boltzmann)        = {tv:.4f}   (< 0.03 target)")
    print(f"marginals P(z_i=1) analytic     = [{a_marg[0]:.3f}, {a_marg[1]:.3f}, {a_marg[2]:.3f}]")
    print(f"marginals P(z_i=1) empirical    = [{e_marg[0]:.3f}, {e_marg[1]:.3f}, {e_marg[2]:.3f}]")
    print(f"max marginal error              = {marg_err:.4f}")
    print(f"pairwise E[z0 z1] analytic/emp  = {a_corr:.4f} / {e_corr:.4f}  (err {corr_err:.4f})")
    print("-" * 66)
    print("clamp z2=1  ->  posterior p(z0,z1 | z2=1):")
    print(f"  P(z0=1|z2=1) analytic/emp     = {ca_marg[0]:.4f} / {ce_marg[0]:.4f}")
    print(f"  P(z1=1|z2=1) analytic/emp     = {ca_marg[1]:.4f} / {ce_marg[1]:.4f}")
    print(f"  conditional TV                = {cond_tv:.4f}   (max marg err {cond_marg_err:.4f})")
    print("=" * 66)

    # ---- self-check: deterministic asserts, zero-dep --------------------------------
    # 1. stationary distribution IS the Boltzmann distribution (the headline correctness test).
    assert tv < 0.03, f"empirical must match Boltzmann p(z): TV {tv:.4f}"
    # 2. marginals and one pairwise correlation match the analytic values.
    assert marg_err < 0.02, f"marginals must match analytic: max err {marg_err:.4f}"
    assert corr_err < 0.02, f"pairwise E[z0 z1] must match analytic: err {corr_err:.4f}"
    # 3. conditional inference by clamping recovers the exact posterior p(z_free | z_clamped).
    assert all(z[2] == 1 for z in cond_samples), "clamped unit must stay fixed to its evidence"
    assert cond_tv < 0.03, f"clamped conditional must match posterior: TV {cond_tv:.4f}"
    assert cond_marg_err < 0.02, f"conditional marginals must match: max err {cond_marg_err:.4f}"
    # 4. determinism: same weights/biases/seed -> identical chain and identical estimates.
    twin = BoltzmannSampler(weights, biases, seed=0)
    assert twin.sample(40000, burn_in=3000, thin=3) == samples, "same seed must give identical chain"

    print(f"self-check OK: spikes sample Boltzmann p(z) (TV {tv:.4f}); marginals err {marg_err:.4f}, "
          f"E[z0 z1] err {corr_err:.4f}; clamp z2=1 posterior TV {cond_tv:.4f}; deterministic")


if __name__ == "__main__":
    main()

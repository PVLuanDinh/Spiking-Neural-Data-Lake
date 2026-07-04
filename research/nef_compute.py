"""
NEF compute — representing and transforming values in a spiking population (pure Python, no tensors).

The Neural Engineering Framework (Eliasmith & Anderson) is how you COMPUTE with spiking neurons: a scalar (or
vector) x is REPRESENTED by the activity of a heterogeneous population -- each neuron is a LIF cell with its own
gain, bias and preferred direction, giving a distinct tuning curve over x. The value is read back by a linear
decode of the population activity, with decoders solved once by LEAST SQUARES to minimise reconstruction error.
The second principle, TRANSFORMATION, is the payoff: solving the decoders against a target function f(x) instead
of x yields a population that computes f -- the SAME spikes, different read-out weights, so a downstream synapse
can read x, x^2, or 2x for free. This is the spiking-computation substrate the whole stack sits on (Nengo / Loihi
use it). Pure standard library, deterministic. On-thesis with population_encoding / spike_preprocessing.

ponytail: rate-mode NEF (steady-state LIF tuning curves + least-squares decoders) -- the representation and
transformation principles, minus the temporal spike-filtering and the dynamics principle. Enough to show a
population represent a value and compute a nonlinear function of it; not full Nengo.

  python -m research.nef_compute        # (also run by run_selfchecks.py, root on PYTHONPATH)
"""
import math
import random

T_REF, TAU_RC = 0.002, 0.02                 # LIF refractory + membrane time constants (Nengo defaults)


def lif_rate(current):
    """Steady-state LIF firing rate for a constant input current (0 below the rheobase current J=1)."""
    return 1.0 / (T_REF - TAU_RC * math.log(1.0 - 1.0 / current)) if current > 1.0 else 0.0


def _solve(a, b):
    """Solve the linear system a x = b (Gaussian elimination with partial pivoting). a is n x n (mutated)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        d = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= d
        for r in range(n):
            if r != col and m[r][col]:
                f = m[r][col]
                for j in range(col, n + 1):
                    m[r][j] -= f * m[col][j]
    return [m[i][n] for i in range(n)]


class NEFEnsemble:
    """A population of LIF neurons that represents a scalar x in [-1, 1] and decodes functions of it."""

    def __init__(self, n=60, seed=0):
        rng = random.Random(seed)
        self.n = n
        self.enc = [rng.choice((-1.0, 1.0)) for _ in range(n)]     # preferred direction
        self.alpha, self.bias = [], []
        for _ in range(n):
            x_int = rng.uniform(-0.95, 0.95)                       # x where the neuron starts firing
            r_max = rng.uniform(100.0, 300.0)                      # firing rate at the preferred extreme
            j_max = 1.0 / (1.0 - math.exp((T_REF - 1.0 / r_max) / TAU_RC))
            gain = (j_max - 1.0) / (1.0 - x_int)                   # current = 1 at x_int, j_max at x=+1 (in enc dir)
            self.alpha.append(gain)
            self.bias.append(1.0 - gain * x_int)

    def rates(self, x):
        return [lif_rate(self.alpha[i] * self.enc[i] * x + self.bias[i]) for i in range(self.n)]

    def solve_decoders(self, func, samples=80, reg=0.1):
        xs = [-1.0 + 2.0 * j / (samples - 1) for j in range(samples)]
        act = [self.rates(x) for x in xs]                          # M x N activity matrix
        gamma = [[sum(act[m][i] * act[m][j] for m in range(samples)) for j in range(self.n)]
                 for i in range(self.n)]
        ridge = (reg * max(gamma[i][i] for i in range(self.n)))    # regularize -> smooth decoders (noise model)
        for i in range(self.n):
            gamma[i][i] += ridge
        upsilon = [sum(act[m][i] * func(xs[m]) for m in range(samples)) for i in range(self.n)]
        return _solve(gamma, upsilon)

    def decode(self, decoders, x):
        r = self.rates(x)
        return sum(decoders[i] * r[i] for i in range(self.n))


def _rmse(ens, decoders, func, pts):
    return (sum((ens.decode(decoders, x) - func(x)) ** 2 for x in pts) / len(pts)) ** 0.5


def main():
    ens = NEFEnsemble(n=80, seed=0)
    test = [-1.0 + 2.0 * j / 40 for j in range(41)]               # test points (off the training grid)

    d_id = ens.solve_decoders(lambda x: x)                        # representation: decode x
    d_sq = ens.solve_decoders(lambda x: x * x)                    # transformation: decode x^2
    rmse_id = _rmse(ens, d_id, lambda x: x, test)
    rmse_sq = _rmse(ens, d_sq, lambda x: x * x, test)

    # a smaller population represents the value less accurately (the NEF error ~ 1/N scaling)
    small = NEFEnsemble(n=12, seed=0)
    rmse_small = _rmse(small, small.solve_decoders(lambda x: x), lambda x: x, test)

    print("=" * 60)
    print("NEF COMPUTE — represent + transform a value in a spiking population")
    print("=" * 60)
    print(f"population: {ens.n} LIF neurons over x in [-1, 1]")
    print(f"representation  decode(x)   RMSE {rmse_id:.4f}")
    print(f"transformation  decode(x^2) RMSE {rmse_sq:.4f}")
    print(f"small pop ({small.n})  decode(x)   RMSE {rmse_small:.4f} (fewer neurons -> worse)")
    print(f"spot: x=0.5 -> decode(x)={ens.decode(d_id, 0.5):.3f}, decode(x^2)={ens.decode(d_sq, 0.5):.3f} (want 0.25)")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert rmse_id < 0.05, f"the population must represent x accurately ({rmse_id:.4f})"
    assert rmse_sq < 0.05, f"the population must compute x^2 by decoding ({rmse_sq:.4f})"
    assert rmse_small > rmse_id, f"fewer neurons must decode worse (NEF error scaling) ({rmse_small:.4f} vs {rmse_id:.4f})"
    assert abs(ens.decode(d_sq, 0.5) - 0.25) < 0.05, f"decode(0.5^2) must be ~0.25 ({ens.decode(d_sq, 0.5):.3f})"
    assert ens.solve_decoders(lambda x: x) == d_id, "same ensemble -> identical decoders (deterministic)"
    print(f"self-check OK: represent x (RMSE {rmse_id:.3f}), compute x^2 (RMSE {rmse_sq:.3f}) from the same "
          f"spikes; {small.n}-neuron pop worse ({rmse_small:.3f}); deterministic")


if __name__ == "__main__":
    main()

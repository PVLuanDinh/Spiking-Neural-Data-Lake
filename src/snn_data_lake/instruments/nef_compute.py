"""
NEF compute -- representing and transforming values in a spiking population.

The Neural Engineering Framework (Eliasmith & Anderson) is how you COMPUTE with spiking neurons: a scalar (or
vector) x is REPRESENTED by the activity of a heterogeneous population -- each neuron is a LIF cell with its own
gain, bias and preferred direction, giving a distinct tuning curve over x. The value is read back by a linear
decode of the population activity, with decoders solved once by LEAST SQUARES to minimise reconstruction error.
The second principle, TRANSFORMATION, is the payoff: solving the decoders against a target function f(x) instead
of x yields a population that computes f -- the SAME spikes, different read-out weights, so a downstream synapse
can read x, x^2, or 2x for free. This is the spiking-computation substrate the whole stack sits on (Nengo / Loihi
use it). Deterministic (per-instance random.Random(seed); no import-time seeding).

Rate-mode NEF (steady-state LIF tuning curves + least-squares decoders) -- the representation and
transformation principles, minus the temporal spike-filtering and the dynamics principle. Enough to show a
population represent a value and compute a nonlinear function of it; not full Nengo.

P7 (2026-08-04) -- what the retired zero-dependency rule cost here, now restored. The docstring used to boast
"pure Python, no tensors / Pure standard library". That was not a stylistic flourish: it bought a hand-rolled
Gaussian-elimination solver (`_solve`, deleted) and an O(N^2 * M) Python triple-comprehension for the Gram
matrix. A/B'd on this tree at the widths that actually occur (mean of repeated calls, .venv-lake, no GPU):
`solve_decoders` n=80 51.3 ms -> 0.15 ms (334x), n=100 (nef_integrator) 82.2 ms -> 2.9 ms (28x), n=150
(the width `substrate/nengo_bridge.py` builds for its cross-engine ground-truth check) 197.8 ms -> 9.2 ms
(22x). The tuning curves, the Gram matrix and the ridge solve are now numpy/LAPACK; `np.linalg.solve` is
additionally better conditioned than a hand-rolled Gauss-Jordan on the normal equations. Numerics moved only
at float64 round-off: rates are BIT-IDENTICAL, decoders differ by <= 3.1e-13 relative, and every consumer
number (both nengo_bridge zero-dep RMSEs, the whole nef_integrator trajectory) by <= 5.2e-13 relative --
measured at full precision, not to four decimals. The scalar `lif_rate` entry point was DELETED at the
2026-08-12 promotion: it had ZERO call sites tree-wide (measured, importlib/string dispatch included) and
its body was one of the two statements the self-check never executed -- a wrapper with no consumer is
presence, not participation (P3), and the P2 one-definition property it claimed to protect lives in
`_lif_rates` itself, which every real caller already uses.

RE-MEASURED 2026-08-05 (.venv-lake, median of 15 calls), because a timing in a comment is a claim and this
file argues that everywhere else. Two of the three post-fix numbers reproduce and ONE DOES NOT:
`solve_decoders` n=80 0.15 -> 0.12 ms, n=100 2.9 -> 2.69 ms, **n=150 9.2 -> 4.32 ms (2.1x apart)**. The gap
is in the conservative direction -- the real speedup at n=150 is LARGER than the 22x recorded, not smaller --
but median-of-15 is not noise at that size, so the honest reading is that the n=150 figure was taken under
different machine load and should not be quoted as-is. The correctness numbers all still hold exactly: the
n=12 control reads 0.0704 and the permutation null 11.57x (recorded as 11.6x).

P2 -- this does NOT duplicate `sensorimotor/population_encoding.py`, despite CHANGELOG v1.5 asserting that it
"covers encode/decode". PopulationEncoder tiles a range with fixed Gaussian receptive fields and reads the
value back as an activation-weighted centroid of the centres: it can represent x and it cannot compute f(x).
There is no decoder solve, so there is no TRANSFORMATION principle in it -- the whole point of this module.
Neither is retired by the other.

E-PROP -- nothing here learns by gradient descent. The decoders are a closed-form ridge least-squares fit
solved ONCE (`np.linalg.solve` on the normal equations); no `.backward()`, no BPTT, no eligibility trace, no
weight update on any tick. The e-prop invariant is not engaged by this module in either direction.

The run hint is DERIVED from the module's own identity (see `_run_hint`) so moving this file cannot leave it
stale -- see that docstring for the hazard it closes.
"""

import math
import os
import random

import numpy as np

# P1 -- MODEL-DEFINING PHYSICAL CONSTANTS, fixed on purpose and labelled rather than carried silently across
# the tier boundary. These are the LIF refractory period and membrane time constant, in SECONDS. They are not
# behaviour thresholds a running baseline could replace: they ARE the neuron model, and this module's job is
# to be the ground truth a second engine is checked against. `substrate/nengo_bridge.py:56-57` pins the
# identical pair (they are also Nengo's own defaults) and `:570` asserts that Nengo's decode error is
# COMPARABLE to the numbers produced here. Letting these drift would not make the model more brain-native; it
# would delete the comparison. Closest documented exception: the fourth (a raw physical value that stays fixed
# BECAUSE an invariant depends on it) -- applied here to a model-fidelity contract, not to the safety envelope.
T_REF, TAU_RC = 0.002, 0.02

# P1 exception (a) -- a fixed point on a NORMALISED output. x is represented over [-1, 1], so an RMSE
# tolerance of 0.05 is 2.5% of the represented range: dimensionless, not a unit-bearing magic number. It is
# named once here because `substrate/nengo_bridge.py:569` asserts the same 0.05 against these decoders, and a
# single name is what stops the two drifting apart.
RMSE_TOL = 0.05

# P1 exception (a) -- a dimensionless discrimination RATIO for the permutation null, not an error bound. The
# measured ratio on this tree is 11.6x (see main()); 5x is the band, well inside the observed separation.
NULL_MARGIN = 5.0

# The reference tuning-curve distribution: intercepts (the x at which a neuron starts firing) and maximum
# rates. Not fixed -- they are constructor parameters, so any caller may choose another population. These
# DEFAULTS are a contract: `substrate/nengo_bridge.py:85` documents that its Nengo ensemble is configured to
# match "intercepts in [-0.95, 0.95]", so changing the defaults silently would break a cross-engine check.
X_INT_RANGE = (-0.95, 0.95)
R_MAX_RANGE = (100.0, 300.0)


def _lif_rates(current):
    """Steady-state LIF rate for constant input current(s); 0 at or below the rheobase current J=1.

    Vectorised over a numpy array of any shape; this is the ONE definition of the rate law in this module.
    """
    j = np.asarray(current, dtype=float)
    above = j > 1.0
    safe = np.where(above, j, 2.0)  # keeps log() in-domain where the result is masked out anyway
    return np.where(above, 1.0 / (T_REF - TAU_RC * np.log(1.0 - 1.0 / safe)), 0.0)


class NEFEnsemble:
    """A population of LIF neurons that represents a scalar x in [-1, 1] and decodes functions of it."""

    def __init__(self, n=60, seed=0, x_int_range=X_INT_RANGE, r_max_range=R_MAX_RANGE):
        rng = random.Random(seed)  # per-instance seeding (project determinism invariant)
        self.n = n
        enc = [rng.choice((-1.0, 1.0)) for _ in range(n)]  # preferred direction
        alpha, bias = [], []
        for _ in range(n):
            x_int = rng.uniform(*x_int_range)  # x where the neuron starts firing
            r_max = rng.uniform(*r_max_range)  # firing rate at the preferred extreme
            j_max = 1.0 / (1.0 - math.exp((T_REF - 1.0 / r_max) / TAU_RC))
            gain = (j_max - 1.0) / (1.0 - x_int)  # current = 1 at x_int, j_max at x=+1 (in enc dir)
            alpha.append(gain)
            bias.append(1.0 - gain * x_int)
        self.enc = np.array(enc)
        self.alpha = np.array(alpha)
        self.bias = np.array(bias)
        self._slope = self.alpha * self.enc  # tuning-curve slope along the encoder direction

    def rates(self, x):
        """Firing rate of every neuron at the scalar x (vector over the population)."""
        return _lif_rates(self._slope * x + self.bias)

    def activity(self, xs):
        """M x N activity matrix over a vector of sample points (one matmul-shaped build, not M loops)."""
        return _lif_rates(np.outer(np.asarray(xs, dtype=float), self._slope) + self.bias)

    def solve_decoders(self, func, samples=80, reg=0.1):
        """Ridge least-squares decoders for func(x) over [-1, 1]. func has a SCALAR contract."""
        xs = -1.0 + 2.0 * np.arange(samples) / (samples - 1)
        act = self.activity(xs)  # M x N activity matrix
        gamma = act.T @ act
        # P1: `reg` is already a RUNNING-BASELINE-RELATIVE form, not an absolute magic number -- the ridge is
        # a fraction of the population's OWN Gram diagonal, so it rescales with that ensemble's activity.
        gamma[np.diag_indices(self.n)] += (
            reg * gamma.diagonal().max()
        )  # smooth decoders (noise model)
        upsilon = act.T @ np.array([func(float(x)) for x in xs])
        return np.linalg.solve(gamma, upsilon)

    def decode(self, decoders, x):
        """Linear read-out of the population at x. Returns a plain float (the consumer type contract)."""
        return float(np.asarray(decoders) @ self.rates(x))


def _rmse(ens, decoders, func, pts):
    return (sum((ens.decode(decoders, x) - func(x)) ** 2 for x in pts) / len(pts)) ** 0.5


def _run_hint():
    """The command that runs this self-check, DERIVED from the module's own identity.

    Closes a promotion hazard this file actually carried: the hint was a hardcoded string naming the
    retired `research.` package, already wrong since the reasoning fold and about to go wrong again
    on the move into cortex -- a stale self-path that still prints green. `__spec__.name` is the
    true dotted name under `python -m <pkg.mod>` AND under the in-process runner
    (`run_selfchecks.py`'s `run_one_in_process` calls `runpy.run_module(..., run_name="__main__")`,
    which sets `__spec__` to the real module spec), so it cannot go stale wherever the file moves.

    TWO CORRECTIONS, 2026-08-04, and the second is a defect this audit INTRODUCED and then caught:
      * the cite was `run_selfchecks.py:507`, which resolves to the bare `import runpy` inside that
        function, not to the `run_module` call this sentence is about -- a line number drifted off
        its target. Now named by SYMBOL, the fix this module argues for everywhere else, because
        `tools/citation_guard.py` resolves FILES and never line numbers.
      * a sweep that rewrote dead `research/` paths across this tier rewrote the HISTORICAL string
        quoted in this very paragraph, turning "the hint was `python -m research.nef_compute`,
        already wrong" into a sentence quoting the CORRECT path and calling it wrong. A mechanical
        path fix cannot tell a live citation from a quoted dead one. The quotation is paraphrased
        now so no future sweep can invert its meaning again.
    """
    spec = globals().get("__spec__")
    if spec is not None and getattr(spec, "name", None):
        return f"python -m {spec.name}"
    return f"python {os.path.basename(__file__)}"


def main():
    ens = NEFEnsemble(n=80, seed=0)
    test = [-1.0 + 2.0 * j / 40 for j in range(41)]  # test points (off the training grid)

    d_id = ens.solve_decoders(lambda x: x)  # representation: decode x
    d_sq = ens.solve_decoders(lambda x: x * x)  # transformation: decode x^2
    rmse_id = _rmse(ens, d_id, lambda x: x, test)
    rmse_sq = _rmse(ens, d_sq, lambda x: x * x, test)

    # CONTROL 1 -- a smaller population represents the value less accurately (the NEF error ~ 1/N scaling).
    small = NEFEnsemble(n=12, seed=0)
    rmse_small = _rmse(small, small.solve_decoders(lambda x: x), lambda x: x, test)

    # CONTROL 2 (permutation null) -- the SAME decoder values reassigned to the WRONG neurons. Separates
    # "least squares matched each weight to its neuron's tuning curve" from "any weight vector of this
    # magnitude decodes x^2". A control that is not the method itself.
    perm = list(range(ens.n))
    random.Random(1).shuffle(perm)
    rmse_shuf = _rmse(ens, np.asarray(d_sq)[perm], lambda x: x * x, test)

    # CONTROL 3 -- a different seed must build a different population (the seed is actually honoured).
    other = NEFEnsemble(n=80, seed=1)

    print("=" * 60)
    print("NEF COMPUTE -- represent + transform a value in a spiking population")
    print("=" * 60)
    print(f"population: {ens.n} LIF neurons over x in [-1, 1]")
    print(f"representation  decode(x)   RMSE {rmse_id:.4f}")
    print(f"transformation  decode(x^2) RMSE {rmse_sq:.4f}")
    print(f"small pop ({small.n})  decode(x)   RMSE {rmse_small:.4f} (fewer neurons -> worse)")
    print(
        f"CONTROL perm null decode(x^2) RMSE {rmse_shuf:.4f} ({rmse_shuf / rmse_sq:.1f}x the fitted decoders)"
    )
    print(
        f"spot: x=0.5 -> decode(x)={ens.decode(d_id, 0.5):.3f}, decode(x^2)={ens.decode(d_sq, 0.5):.3f} (want 0.25)"
    )
    print(f"run: {_run_hint()}")
    print("=" * 60)

    # ---- self-check. P8: every criterion below has a REACHABLE RED, named on its own line, and for the two
    # tolerance asserts the failing input is MEASURED IN THIS SAME RUN (the n=12 control reads 0.0704 > 0.05).
    assert rmse_id < RMSE_TOL, f"the population must represent x accurately ({rmse_id:.4f})"
    # RED: n=12 -- rmse_small below is that same criterion evaluated on the failing input.
    assert rmse_sq < RMSE_TOL, f"the population must compute x^2 by decoding ({rmse_sq:.4f})"
    # RED: n=12, or the permutation null.
    assert rmse_small > rmse_id, (
        f"fewer neurons must decode worse (NEF error scaling) ({rmse_small:.4f} vs {rmse_id:.4f})"
    )
    # RED: build `small` at the same n as `ens` -- identical ensemble, equal RMSE, `>` fails.
    assert abs(ens.decode(d_sq, 0.5) - 0.25) < RMSE_TOL, (
        f"decode(0.5^2) must be ~0.25 ({ens.decode(d_sq, 0.5):.3f})"
    )
    # RED: decode with `d_id` instead of `d_sq` -- reads 0.503, off by 0.25.
    assert rmse_shuf > NULL_MARGIN * rmse_sq, (
        f"PERMUTATION NULL: shuffled decoders must decode far worse ({rmse_shuf:.4f} vs {rmse_sq:.4f})"
    )
    # RED: collapse the population to one tuning curve -- NEFEnsemble(n=80, seed=0, x_int_range=(0.0, 0.0),
    # r_max_range=(200.0, 200.0)) makes every neuron interchangeable, so the permutation is a no-op.
    assert not np.array_equal(other.alpha, ens.alpha), (
        "the seed must build a different population (per-instance random.Random, not module-level state)"
    )
    # RED: construct `other` at seed=0 -- identical draws, arrays equal, assert fails.
    #
    # DELETED 2026-08-04 (P8, no reachable red): `assert ens.solve_decoders(lambda x: x) == d_id,
    # "same ensemble -> identical decoders (deterministic)"`. `solve_decoders` is a pure function of the
    # ensemble's already-fixed parameters, so calling it twice on the same object with the same target
    # cannot return anything different -- for ANY n, seed, sample count or target function. No constructible
    # input reddens it, so per P8 it is deleted rather than weakened. What it claimed to guard (determinism)
    # is guarded for real by CONTROL 3 above, whose red IS constructible. Note it would also have crashed
    # under the numpy rewrite (elementwise `==` on two arrays has no truth value), which is a second reason
    # not to keep it: it was a shape that could only ever pass or explode, never fail informatively.
    print(
        f"self-check OK: represent x (RMSE {rmse_id:.3f}), compute x^2 (RMSE {rmse_sq:.3f}) from the same "
        f"spikes; {small.n}-neuron pop worse ({rmse_small:.3f}); permutation null {rmse_shuf / rmse_sq:.0f}x worse"
    )


if __name__ == "__main__":
    main()

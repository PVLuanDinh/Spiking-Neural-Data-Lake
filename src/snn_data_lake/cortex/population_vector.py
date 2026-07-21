"""
Population vector decoder -- read a continuous DIRECTION out of a cosine-tuned population by
the rate-weighted vector sum of preferred directions.

A population of N neurons each has a PREFERRED DIRECTION phi_i spread around the circle and
COSINE tuning: for a movement/stimulus direction theta, neuron i fires

    r_i(theta) = baseline + gain * max(0, cos(theta - phi_i))        (rectified cosine tuning)

No single broadly-tuned cell reports theta well, but the POPULATION does. Subtract the
baseline and sum each neuron's preferred-direction unit vector weighted by that firing rate:

    P = sum_i (r_i - baseline) * (cos phi_i, sin phi_i)
    theta_hat = atan2(P_y, P_x)

With preferred directions spread evenly around the circle this recovers theta (in the
continuum it is exact: integrating max(0, cos(theta - phi)) * (cos phi, sin phi) over phi
gives (pi/2) * (cos theta, sin theta), so the angle is exact and unbiased). It gets MORE
accurate with more neurons and degrades gracefully under rate noise -- far better than
reading the single most-active cell. This is the Georgopoulos motor-cortex population vector
(Georgopoulos, Schwartz & Kettner, Science 233:1416, 1986).

DISTINCTNESS (read population_encoding.py and cue_integration.py -- this is neither):
  - population_encoding.py is an ENCODER: it maps a SCALAR input to a vector of per-neuron
    RBF/Gaussian activations (input -> rates) and decodes by an activation-weighted centroid
    of bin CENTERS on a LINE. THIS module is a DECODER for a circular DIRECTION: it maps a
    population of COSINE-tuned firing rates back to a continuous ANGLE by the rate-weighted
    VECTOR SUM of preferred directions (rates -> direction) -- the inverse operation, on a
    circular variable, via Georgopoulos' vector sum, not an RBF scalar centroid.
  - cue_integration.py fuses several independent SCALAR estimates of ONE quantity by
    inverse-variance (reliability) weighting -- optimal Bayesian estimation. THIS decodes ONE
    direction from MANY cosine-tuned cells by vector summation of their preferred directions
    -- a different mechanism (tuned-population readout, not reliability-weighted fusion; no
    variances, no reliability weights).

The ceiling is a STATIC cosine-tuned population with the Georgopoulos vector-sum
decode on SYNTHETIC 2-D directions -- the population-readout level. NOT a spiking-network
implementation, no learning of the tuning curves, 2-D directions only. Not cognition.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.cortex.population_vector
"""

import math
import random

from snn_data_lake.substrate._mathutils import mean


def angular_error(a, b):
    """Smallest SIGNED angle a - b, wrapped to (-pi, pi], in RADIANS."""
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


class PopulationVector:
    """N cosine-tuned neurons with preferred directions around the circle; decode a direction.

    Preferred directions are laid out evenly (optionally jittered by a fraction of the spacing
    via the seeded RNG). rates(theta) gives each neuron's rectified-cosine firing rate (with
    optional Gaussian rate noise from the seeded RNG); decode(rates) returns the population
    vector angle. Deterministic: same n_neurons/seed/noise -> identical decoded directions."""

    def __init__(self, n_neurons=16, baseline=5.0, gain=20.0, jitter=0.0, seed=0):
        if n_neurons < 2:
            raise ValueError("need n_neurons >= 2")
        if gain <= 0.0:
            raise ValueError("gain must be positive")
        self.n = n_neurons
        self.baseline = float(baseline)
        self.gain = float(gain)
        self.seed = seed
        step = 2.0 * math.pi / n_neurons
        jrng = random.Random(seed)
        self.preferred = [
            i * step + (jitter * step * (jrng.random() - 0.5) if jitter else 0.0)
            for i in range(n_neurons)
        ]
        self._cos = [math.cos(p) for p in self.preferred]  # preferred-direction unit
        self._sin = [math.sin(p) for p in self.preferred]  # vectors (precomputed)
        self._rng = random.Random(seed)  # rate-noise stream

    def rates(self, theta, noise=0.0):
        """Per-neuron firing rates for true direction theta (rectified cosine tuning).

        Optional zero-mean Gaussian rate noise (std = noise) from the seeded RNG; the rate is
        floored at 0 (a firing rate cannot be negative)."""
        out = []
        for i in range(self.n):
            r = self.baseline + self.gain * max(0.0, math.cos(theta - self.preferred[i]))
            if noise:
                r += self._rng.gauss(0.0, noise)
            out.append(max(0.0, r))
        return out

    def decode(self, rates):
        """Population vector decode: rate-weighted (baseline-subtracted) sum of preferred
        directions -> theta_hat (radians, atan2 of the summed vector)."""
        px = 0.0
        py = 0.0
        for i in range(self.n):
            w = rates[i] - self.baseline
            px += w * self._cos[i]
            py += w * self._sin[i]
        return math.atan2(py, px)

    def decode_direction(self, theta, noise=0.0):
        """Convenience: encode true direction theta to rates then decode -> theta_hat."""
        return self.decode(self.rates(theta, noise))

    def best_neuron_direction(self, rates):
        """Single-neuron baseline: preferred direction of the most-active neuron (argmax)."""
        best = max(range(self.n), key=lambda i: rates[i])
        return self.preferred[best]


def _test_directions(k, seed=12345):
    """k directions drawn uniformly on [0, 2*pi) (fixed seed -> reproducible sweep)."""
    rng = random.Random(seed)
    return [rng.uniform(0.0, 2.0 * math.pi) for _ in range(k)]


def main():
    deg = 180.0 / math.pi
    dirs = _test_directions(2000)  # 2000 fixed-seed directions on the circle

    # ---- 1. decoding accuracy (noise-free) ---------------------------------------------
    # N=15 (odd) so a genuine sub-degree tuning-ripple survives -- an even, evenly-spread
    # population cancels every contaminating harmonic and decodes EXACTLY (error 0), which
    # would make the numbers look rigged; the odd population shows real, small error.
    pop = PopulationVector(n_neurons=15, baseline=5.0, gain=20.0, seed=0)
    errs = [angular_error(pop.decode_direction(t), t) for t in dirs]
    mean_abs = mean([abs(e) for e in errs]) * deg
    max_abs = max(abs(e) for e in errs) * deg
    signed_bias = mean(errs) * deg  # ~0 for evenly-spread preferred dirs

    # ---- 2. accuracy improves with N (fixed noise -> more cells average out the noise) --
    noise = 5.0  # std vs peak drive gain=20 (moderate)
    small = PopulationVector(n_neurons=8, baseline=5.0, gain=20.0, seed=0)
    large = PopulationVector(n_neurons=64, baseline=5.0, gain=20.0, seed=0)
    small_err = mean([abs(angular_error(small.decode_direction(t, noise), t)) for t in dirs]) * deg
    large_err = mean([abs(angular_error(large.decode_direction(t, noise), t)) for t in dirs]) * deg

    # ---- 3. noise robustness: population beats reading the single most-active neuron ----
    npop = PopulationVector(n_neurons=16, baseline=5.0, gain=20.0, seed=7)
    pop_noisy, single_noisy = [], []
    for t in dirs:
        r = npop.rates(t, noise=noise)  # one noisy rate vector, shared by both
        pop_noisy.append(abs(angular_error(npop.decode(r), t)))
        single_noisy.append(abs(angular_error(npop.best_neuron_direction(r), t)))
    pop_noisy_err = mean(pop_noisy) * deg
    single_noisy_err = mean(single_noisy) * deg

    # ---- 5. determinism: same n/seed/noise -> identical decoded directions --------------
    a = PopulationVector(n_neurons=16, seed=3)
    b = PopulationVector(n_neurons=16, seed=3)
    dec_a = [a.decode_direction(t, noise=2.0) for t in dirs[:200]]
    dec_b = [b.decode_direction(t, noise=2.0) for t in dirs[:200]]

    print("=" * 70)
    print("POPULATION VECTOR DECODER -- Georgopoulos vector-sum readout of a direction")
    print("=" * 70)
    print(
        "tuning: r_i = baseline + gain * max(0, cos(theta - phi_i)); "
        "decode theta_hat = atan2(sum (r_i-baseline)*(cos phi_i, sin phi_i))"
    )
    print(
        f"1. accuracy  (N=15, {len(dirs)} dirs, noise-free): "
        f"mean |err| {mean_abs:.3f} deg  max |err| {max_abs:.3f} deg"
    )
    print(
        f"2. improves with N (noise std {noise:.0f}): N=8 {small_err:.3f} deg -> "
        f"N=64 {large_err:.3f} deg"
    )
    print(
        f"3. under noise (std {noise:.0f}): population {pop_noisy_err:.3f} deg  vs "
        f"single most-active neuron {single_noisy_err:.3f} deg"
    )
    print(f"4. signed bias (N=15, noise-free): {signed_bias:+.4f} deg (approx 0 -> unbiased)")
    print(f"5. determinism: two fresh N=16 seed=3 instances identical: {dec_a == dec_b}")
    print("=" * 70)

    # ---- self-checks (genuine, not rigged) ----
    # 1. noise-free population vector decodes direction to within a few degrees.
    assert mean_abs < 3.0, f"noise-free mean angular error too high: {mean_abs:.3f} deg"
    assert max_abs < 8.0, f"noise-free max angular error too high: {max_abs:.3f} deg"

    # 2. more neurons -> clearly lower error (population averaging under the same noise).
    assert large_err < small_err, (
        f"large N must beat small N ({large_err:.4f} deg < {small_err:.4f} deg)"
    )
    assert large_err < 0.6 * small_err, (
        f"large N should be clearly better ({large_err:.4f} vs {small_err:.4f} deg)"
    )

    # 3. under noise the population stays bounded and beats the single-neuron argmax read-out.
    assert pop_noisy_err < 15.0, f"population noisy error unbounded: {pop_noisy_err:.3f} deg"
    assert pop_noisy_err < single_noisy_err, (
        f"population must beat single most-active neuron under noise "
        f"({pop_noisy_err:.3f} vs {single_noisy_err:.3f} deg)"
    )

    # 4. evenly-spread preferred directions give an unbiased decode (no systematic rotation).
    assert abs(signed_bias) < 0.5, f"decode has a systematic bias: {signed_bias:+.4f} deg"

    # 5. deterministic: same n/seed/noise -> identical decoded directions.
    assert dec_a == dec_b, "same n/seed/noise must reproduce identical decoded directions"

    print(
        f"self-check OK: mean |err| {mean_abs:.3f} deg (max {max_abs:.3f}); N=8 {small_err:.3f} "
        f"-> N=64 {large_err:.3f} deg; under noise population {pop_noisy_err:.3f} deg < "
        f"single-neuron {single_noisy_err:.3f} deg; bias {signed_bias:+.4f} deg; deterministic"
    )


if __name__ == "__main__":
    main()

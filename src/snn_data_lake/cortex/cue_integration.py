"""
Bayesian cue integration -- reliability-weighted (inverse-variance) optimal fusion of
several noisy estimates of ONE quantity (pure stdlib, zero deps).

Given N independent Gaussian estimates x_i +/- sigma_i of the SAME latent value, the
maximum-likelihood (Bayesian-optimal) combination is a closed form:

    x_hat   = ( sum_i x_i / sigma_i^2 ) / ( sum_i 1 / sigma_i^2 )
    var_hat = 1 / ( sum_i 1 / sigma_i^2 )
    w_i     = (1 / sigma_i^2) / ( sum_j 1 / sigma_j^2 )      (weight = reliability = 1/var)

Two consequences ARE the point:
  1. the more RELIABLE (lower-variance) cue DOMINATES the fused estimate (larger weight);
  2. the fused estimate is MORE PRECISE than EVERY single cue: var_hat < min_i(sigma_i^2).
This is the textbook optimal multisensory integration result (Ernst & Banks, Nature
415:429, 2002, visual-haptic; Alais & Burr, Curr Biol 14:257, 2004, audio-visual).

DISTINCTNESS (this is reliability-weighted ESTIMATION, not encoding or time-integration):
  - multimodal_encoder.py BANDS four sensor modalities (visual/temperature/audio/distance)
    into the N=64 spike substrate -- an ENCODER that CONCATENATES modalities into disjoint
    feature bands. It applies NO reliability weighting and makes NO precision/variance
    claim. THIS module fuses several noisy estimates of ONE quantity by inverse-variance
    (reliability) weighting into a single estimate provably MORE precise than any single
    cue -- a different operation (optimal Bayesian estimation, not modality concatenation).
  - evidence_accumulation.py integrates ONE noisy stream over TIME to a decision bound
    (a choice + a reaction time). THIS fuses SEVERAL simultaneous estimates by reliability
    in closed form -- no time, no bound, no decision.
  - not population_encoding (an RBF scalar encoder), not neural_sampling / predictive_coding
    (different inference machinery).

The ceiling is the closed-form Gaussian MLE cue combination on SYNTHETIC noisy
cues -- the optimal-integration level, NOT a spiking population implementation and not a
full Bayesian network. Small fixed cue counts and a fixed-seed empirical variance keep the
demonstration reproducible. Not cognition.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.cortex.cue_integration
"""
import math
import random


def _check(estimates, variances):
    """Validate paired estimate/variance inputs (non-empty, aligned, positive variance)."""
    if len(estimates) != len(variances):
        raise ValueError("estimates and variances must have the same length")
    if not variances:
        raise ValueError("need at least one cue")
    if any(v <= 0.0 for v in variances):
        raise ValueError("variances must be positive")


def weights(variances):
    """Reliability weights (proportional to 1/variance), normalized to sum to 1.

    The lower a cue's variance, the higher its weight -- the reliable cue dominates."""
    if not variances:
        raise ValueError("need at least one cue")
    if any(v <= 0.0 for v in variances):
        raise ValueError("variances must be positive")
    inv = [1.0 / v for v in variances]
    denom = sum(inv)
    return [k / denom for k in inv]


def fuse(estimates, variances):
    """Closed-form MLE fusion of N Gaussian cues -> (x_hat, var_hat).

    x_hat is the inverse-variance-weighted mean; var_hat = 1/sum(1/var_i) is its variance,
    which is strictly smaller than the smallest single-cue variance (more precise than any
    single cue). Deterministic (no RNG)."""
    _check(estimates, variances)
    inv = [1.0 / v for v in variances]
    denom = sum(inv)
    x_hat = sum(x * k for x, k in zip(estimates, inv)) / denom
    var_hat = 1.0 / denom
    return x_hat, var_hat


class CueIntegrator:
    """A known true value observed through N Gaussian cues with fixed sigmas.

    sample() draws one noisy reading per cue; fuse_sample() draws one such reading set and
    fuses it by inverse-variance weighting. The RNG is seeded once at construction, so two
    fresh instances with the same true/sigmas/seed reproduce identical trials."""

    def __init__(self, true_value, sigmas, seed=0):
        if not sigmas:
            raise ValueError("need at least one cue")
        if any(s <= 0.0 for s in sigmas):
            raise ValueError("sigmas must be positive")
        self.true_value = float(true_value)
        self.sigmas = [float(s) for s in sigmas]
        self.variances = [s * s for s in self.sigmas]
        self._rng = random.Random(seed)

    def sample(self):
        """One independent Gaussian reading per cue about the true value."""
        return [self._rng.gauss(self.true_value, s) for s in self.sigmas]

    def fuse_sample(self):
        """Draw one reading per cue and fuse them -> (readings, x_hat, var_hat)."""
        readings = self.sample()
        x_hat, var_hat = fuse(readings, self.variances)
        return readings, x_hat, var_hat


def _variance(values):
    """Population variance of values about their own mean."""
    n = len(values)
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def _rmse(values, target):
    """Root-mean-square error of values about a target."""
    return math.sqrt(sum((v - target) ** 2 for v in values) / len(values))


def main():
    # ---- 1. reliability weighting: reliable cue dominates; equal cues share evenly ----
    s_reliable, s_unreliable = 1.0, 3.0
    w_reliable = weights([s_reliable ** 2, s_unreliable ** 2])       # weight by 1/sigma^2
    w_equal = weights([2.0 ** 2, 2.0 ** 2])

    # ---- 2 + 3. optimality: empirical fused variance vs analytic; fused beats each cue ----
    true_value = 10.0
    s1, s2 = 1.0, 2.0
    n_trials = 40000                                                 # fixed-seed, stable stats
    integ = CueIntegrator(true_value, [s1, s2], seed=0)

    readings1, readings2, fused = [], [], []
    for _ in range(n_trials):
        r, x_hat, _var = integ.fuse_sample()
        readings1.append(r[0])
        readings2.append(r[1])
        fused.append(x_hat)

    _x, var_hat = fuse([true_value, true_value], [s1 ** 2, s2 ** 2])  # analytic var_hat
    emp_var = _variance(fused)
    single_var1, single_var2 = s1 ** 2, s2 ** 2
    best_single_var = min(single_var1, single_var2)

    rmse1 = _rmse(readings1, true_value)
    rmse2 = _rmse(readings2, true_value)
    rmse_fused = _rmse(fused, true_value)

    # ---- 4. analytic identity on a hand example ----
    hx, hv = fuse([10.0, 12.0], [1.0, 4.0])     # weights 0.8/0.2 -> x_hat 10.4, var_hat 0.8

    print("=" * 70)
    print("BAYESIAN CUE INTEGRATION -- inverse-variance optimal fusion of noisy cues")
    print("=" * 70)
    print(f"reliability weights (sigma {s_reliable} vs {s_unreliable}): "
          f"reliable {w_reliable[0]:.3f}  unreliable {w_reliable[1]:.3f} -> reliable dominates")
    print(f"reliability weights (equal sigma 2.0 / 2.0)  : "
          f"{w_equal[0]:.3f} / {w_equal[1]:.3f} -> equal")
    print(f"fused var    : empirical {emp_var:.4f}  vs analytic var_hat {var_hat:.4f}")
    print(f"single-cue var: cue1 {single_var1:.4f}  cue2 {single_var2:.4f}  "
          f"-> fused {var_hat:.4f} < best single {best_single_var:.4f}")
    print(f"RMSE to true : cue1 {rmse1:.4f}  cue2 {rmse2:.4f}  fused {rmse_fused:.4f}")
    print(f"hand example : fuse([10,12],[1,4]) -> x_hat {hx:.3f} (between, closer to cue1), "
          f"var_hat {hv:.3f}")
    print("=" * 70)

    # ---- self-checks (genuine, not rigged) ----
    # 1. reliability weighting: reliable cue dominates; equal sigmas -> equal weights.
    assert w_reliable[0] > 4 * w_reliable[1], (
        f"reliable cue must dominate ({w_reliable[0]:.3f} vs {w_reliable[1]:.3f})")
    assert abs(w_equal[0] - 0.5) < 1e-9 and abs(w_equal[1] - 0.5) < 1e-9, (
        f"equal sigmas must give equal weights ({w_equal})")

    # 2. optimality: empirical fused variance matches analytic var_hat AND beats the best cue.
    assert abs(emp_var - var_hat) / var_hat < 0.05, (
        f"empirical fused variance must match analytic var_hat "
        f"({emp_var:.4f} vs {var_hat:.4f})")
    assert var_hat < best_single_var, (
        f"analytic fused variance must beat the best single cue "
        f"({var_hat:.4f} < {best_single_var:.4f})")
    assert emp_var < best_single_var, (
        f"empirical fused variance must beat the best single cue "
        f"({emp_var:.4f} < {best_single_var:.4f})")

    # 3. fused beats EACH single cue in RMSE to the true value.
    assert rmse_fused < rmse1 and rmse_fused < rmse2, (
        f"fused RMSE must beat each cue ({rmse_fused:.4f} vs {rmse1:.4f}, {rmse2:.4f})")

    # 4. analytic identity: closed form exact on the hand example.
    assert abs(hx - 10.4) < 1e-12 and abs(hv - 0.8) < 1e-12, (
        f"hand example must reproduce the closed form (x_hat {hx}, var_hat {hv})")
    assert 10.0 < hx < 12.0 and hv < single_var1 and hv < single_var2, (
        "x_hat must lie between the cues and var_hat below both single-cue variances")

    # 5. determinism: same true/sigmas/seed -> identical sampled + fused trials.
    a = CueIntegrator(7.0, [1.0, 2.5], seed=3)
    b = CueIntegrator(7.0, [1.0, 2.5], seed=3)
    assert [a.fuse_sample() for _ in range(100)] == [b.fuse_sample() for _ in range(100)], (
        "same seed must reproduce identical fused trials")

    print(f"self-check OK: weights reliable {w_reliable[0]:.3f}/{w_reliable[1]:.3f} "
          f"equal {w_equal[0]:.2f}/{w_equal[1]:.2f}; fused var empirical {emp_var:.4f} "
          f"~ analytic {var_hat:.4f} < best cue {best_single_var:.4f}; RMSE fused "
          f"{rmse_fused:.4f} < cue1 {rmse1:.4f}, cue2 {rmse2:.4f}; deterministic")


if __name__ == "__main__":
    main()

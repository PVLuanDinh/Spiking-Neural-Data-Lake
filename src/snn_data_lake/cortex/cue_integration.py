"""
Spiking probabilistic population code (PPC) cue integration -- reliability-weighted
(inverse-variance) optimal fusion of several noisy estimates of ONE quantity, done with
GENUINE SPIKES instead of a closed-form average (pure stdlib, zero deps).

Ma, Beck, Latham & Pouget (Nat Neurosci 9:1432, 2006) showed the Bayesian-optimal way to
combine cues is NOT a formula the brain must evaluate -- it falls out of spiking itself.
Encode each cue as a population of Poisson neurons with Gaussian tuning curves over the
stimulus axis. With Poisson variability the population's GAIN (its total spike count) is
proportional to the cue's RELIABILITY 1/sigma^2, and the log-likelihood the population
carries is linear in its spikes. So the neurally-optimal fusion of two cues is simply to
ADD their spike trains neuron-by-neuron: gains add, hence reliabilities add, and the summed
population is itself a PPC whose decoded estimate is the inverse-variance-weighted mean --
the closed-form MLE, reproduced by spike addition with no divide anywhere in the loop.

    encode:  cue (x_i, sigma_i^2)  ->  Poisson spikes k_j ~ Pois( gain_i * bump_j(x_i) ),
             gain_i = GAIN_UNIT / sigma_i^2            (reliable cue -> more spikes)
    fuse:    r_fused[j] = sum_i r_i[j]                 (per-neuron spike-train SUMMATION)
    decode:  x_hat  = sum_j r_fused[j] * c_j / sum_j r_fused[j]   (spike centroid)
             var_hat = GAIN_UNIT / sum_j r_fused[j]              (reliability = total gain)

Because gain_i = GAIN_UNIT/sigma_i^2, decoding var = GAIN_UNIT/gain recovers sigma_i^2 for a
single cue and 1/(sum_i 1/sigma_i^2) for the sum -- the analytic var_hat -- while the spike
centroid recovers the reliability-weighted mean. Two consequences ARE the point:
  1. the more RELIABLE (lower-variance) cue fires MORE spikes, so it DOMINATES the centroid;
  2. the fused population has the most spikes of all, so var_hat < min_i(sigma_i^2): fusion
     is provably MORE precise than every single cue.
This is the textbook optimal multisensory integration result (Ernst & Banks, Nature
415:429, 2002; Alais & Burr, Curr Biol 14:257, 2004), here realized as pure spiking.

HONEST CEILING: Poisson spikes are noisy, so the spiking estimate only APPROACHES the
closed-form MLE -- it does not equal it. Averaging GAIN_UNIT-scaled populations over a
fixed-seed batch of trials drives the gap small but never to zero. Measured on
fuse([10,12],[1,4]): x within ~0.002 of the analytic 10.4 and var within ~0.2% of 0.8; the
self-check asserts a generous 0.05 / 3% tolerance. That residual noise is the genuine cost
of doing the inference with discrete spikes rather than a formula -- reported, not hidden.

DISTINCTNESS (this is reliability-weighted spiking ESTIMATION, not encoding or accumulation):
  - multimodal_encoder.py BANDS four sensor modalities into disjoint feature bands of the
    N=64 substrate -- an ENCODER that CONCATENATES modalities, no reliability weighting, no
    precision claim. THIS fuses several noisy estimates of ONE quantity by spiking gain into
    a single estimate provably MORE precise than any single cue.
  - population_encoding.py is an RBF encoder of ONE scalar into a rate/count code; it does
    not weight by reliability and makes no fusion or variance claim. THIS uses a population
    per CUE and fuses ACROSS cues by spike addition.
  - evidence_accumulation.py integrates ONE noisy stream over TIME to a decision bound. THIS
    fuses SEVERAL simultaneous cues at one instant by summing spikes -- no time, no bound.
  - not neural_sampling / predictive_coding (different inference machinery).

Small fixed cue counts and a fixed-seed batch keep the demonstration reproducible. Not cognition.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.cortex.cue_integration
"""
import math
import random
import sys

# ---- population constants (a fixed property of the spiking substrate) ----
_GAIN_UNIT = 400.0   # expected spikes emitted per unit of reliability (1/variance); sets gain
_WIDTH = 1.0         # Gaussian tuning-curve width (stimulus units); sets decode resolution
_FUSE_TRIALS = 400   # Poisson trials averaged inside fuse() to tame spike noise (deterministic)
_FUSE_SEED = 20240607


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

    These are exactly the fraction of the fused population's spikes contributed by each cue:
    the lower a cue's variance, the higher its gain, the more it dominates the sum."""
    if not variances:
        raise ValueError("need at least one cue")
    if any(v <= 0.0 for v in variances):
        raise ValueError("variances must be positive")
    inv = [1.0 / v for v in variances]
    denom = sum(inv)
    return [k / denom for k in inv]


def _mle_fuse(estimates, variances):
    """Closed-form inverse-variance MLE -> (x_hat, var_hat). The analytic reference the
    spiking fusion approaches (no RNG); used to score the PPC, not on the spiking path."""
    _check(estimates, variances)
    inv = [1.0 / v for v in variances]
    denom = sum(inv)
    return sum(x * k for x, k in zip(estimates, inv)) / denom, 1.0 / denom


def _poisson(rng, lam):
    """One Poisson spike count with mean lam, drawn from rng (discrete, non-negative int).

    Knuth's product algorithm for small lam (exact); a clamped Gaussian for large lam where
    Knuth would loop lam times (the Poisson -> Normal(lam, lam) limit)."""
    if lam <= 0.0:
        return 0
    if lam < 30.0:
        target = math.exp(-lam)
        k = 0
        prod = 1.0
        while True:
            k += 1
            prod *= rng.random()
            if prod <= target:
                return k - 1
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


def make_basis(estimates, variances, pad_sigmas=4.0, spacing_frac=0.5):
    """Tuning-curve centers shared by all cue populations (the common PPC axis).

    A dense, evenly spaced grid wide enough to hold every cue's bump plus several sigma of
    reading spread, so no population is truncated (truncation would bias the spike centroid)."""
    _check(estimates, variances)
    sig = [math.sqrt(v) for v in variances]
    lo = min(e - pad_sigmas * s for e, s in zip(estimates, sig)) - 3.0 * _WIDTH
    hi = max(e + pad_sigmas * s for e, s in zip(estimates, sig)) + 3.0 * _WIDTH
    dx = _WIDTH * spacing_frac
    n = int(math.ceil((hi - lo) / dx)) + 1
    return [lo + j * dx for j in range(n)]


def encode_cue(rng, estimate, variance, centers):
    """Encode one cue as a Poisson-spiking population over centers -> list of spike counts.

    Neuron j's mean is gain * bump_j, where bump is the (normalized) Gaussian tuning profile
    peaked at the cue estimate and gain = GAIN_UNIT/variance. So total expected spikes ==
    gain scales with reliability: a low-variance (reliable) cue emits MORE spikes."""
    if variance <= 0.0:
        raise ValueError("variance must be positive")
    gain = _GAIN_UNIT / variance
    s2 = 2.0 * _WIDTH * _WIDTH
    bump = [math.exp(-((c - estimate) ** 2) / s2) for c in centers]
    z = sum(bump)
    return [_poisson(rng, gain * b / z) for b in bump]


def sum_populations(populations):
    """The fusion primitive: add spike trains neuron-by-neuron -> one summed population.

    This IS the neurally-optimal cue combination -- no weighting or division, just spikes
    landing on the same neurons. Gains add, so reliabilities add."""
    return [sum(column) for column in zip(*populations)]


def decode_population(spikes, centers):
    """Decode a population's spike counts -> (x_hat, var_hat, gain) or None if silent.

    x_hat is the spike-weighted centroid of the centers (the population estimate); var_hat =
    GAIN_UNIT/gain reads the reliability straight off the total spike count; gain is that
    total. On the encoding calibration these invert the gain law exactly in expectation."""
    total = sum(spikes)
    if total == 0:
        return None
    x_hat = sum(k * c for k, c in zip(spikes, centers)) / total
    return x_hat, _GAIN_UNIT / total, total


def fuse(estimates, variances):
    """Spiking PPC fusion of N Gaussian cues -> (x_hat, var_hat).

    Encodes each cue as a Poisson population on a shared axis, SUMS the spike trains, and
    decodes the summed population; averages GAIN_UNIT-scaled spikes over a fixed-seed batch
    of trials so the Poisson noise is controlled and the result is deterministic. x_hat is
    the reliability-weighted mean and var_hat = 1/sum(1/var_i) -- the closed-form MLE,
    reproduced by spike addition, up to a small measured Poisson residual (see module docs)."""
    _check(estimates, variances)
    centers = make_basis(estimates, variances)
    rng = random.Random(_FUSE_SEED)
    sx = 0.0
    sv = 0.0
    for _ in range(_FUSE_TRIALS):
        pops = [encode_cue(rng, x, v, centers) for x, v in zip(estimates, variances)]
        decoded = decode_population(sum_populations(pops), centers)
        if decoded is None:                                  # silent batch (never at these gains)
            continue
        x_hat, var_hat, _gain = decoded
        sx += x_hat
        sv += var_hat
    return sx / _FUSE_TRIALS, sv / _FUSE_TRIALS


class CueIntegrator:
    """A known true value observed through N Gaussian cues with fixed sigmas.

    sample() draws one noisy reading per cue; fuse_sample() draws one reading set, encodes
    each reading as a Poisson population on a shared tuning axis, SUMS the spike trains, and
    decodes the fused estimate. The RNG is seeded once at construction and drives BOTH the
    readings and the spikes, so two fresh instances with the same true/sigmas/seed reproduce
    identical spiking trials."""

    def __init__(self, true_value, sigmas, seed=0):
        if not sigmas:
            raise ValueError("need at least one cue")
        if any(s <= 0.0 for s in sigmas):
            raise ValueError("sigmas must be positive")
        self.true_value = float(true_value)
        self.sigmas = [float(s) for s in sigmas]
        self.variances = [s * s for s in self.sigmas]
        self._rng = random.Random(seed)
        # shared PPC axis wide enough for readings scattered around the true value
        self._centers = make_basis([self.true_value] * len(self.variances), self.variances)

    def sample(self):
        """One independent Gaussian reading per cue about the true value."""
        return [self._rng.gauss(self.true_value, s) for s in self.sigmas]

    def fuse_sample(self):
        """Draw one reading per cue, fuse by spiking population addition -> (readings, x_hat, var_hat).

        One genuine spiking trial: each reading is Poisson-encoded (gain = reliability), the
        populations are summed neuron-by-neuron, and the summed spikes are decoded. Single-trial
        Poisson noise makes this noisier than fuse(); averaging over many calls converges."""
        readings = self.sample()
        pops = [encode_cue(self._rng, r, v, self._centers)
                for r, v in zip(readings, self.variances)]
        decoded = decode_population(sum_populations(pops), self._centers)
        if decoded is None:                                  # silent trial (never at these gains)
            x_hat, var_hat = _mle_fuse(readings, self.variances)
        else:
            x_hat, var_hat, _gain = decoded
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
    w_reliable = weights([s_reliable ** 2, s_unreliable ** 2])       # weight by 1/sigma^2 = gain
    w_equal = weights([2.0 ** 2, 2.0 ** 2])

    # ---- 2. GENUINELY SPIKING: encode two cues, show gain scales with reliability ----
    centers = make_basis([10.0, 10.0], [1.0, 4.0])
    g_rng = random.Random(5)
    pop_reliable = encode_cue(g_rng, 10.0, 1.0, centers)            # variance 1.0 -> high gain
    pop_unreliable = encode_cue(g_rng, 10.0, 4.0, centers)          # variance 4.0 -> low gain
    gain_reliable, gain_unreliable = sum(pop_reliable), sum(pop_unreliable)
    fused_pop = sum_populations([pop_reliable, pop_unreliable])     # spike-train SUMMATION

    # ---- 3. spiking fuse() vs the closed-form MLE on a hand example ----
    hx, hv = fuse([10.0, 12.0], [1.0, 4.0])                         # analytic: 10.4, var 0.8
    mle_x, mle_v = _mle_fuse([10.0, 12.0], [1.0, 4.0])

    # ---- 4. empirical: many single-trial spiking fusions beat every single cue ----
    true_value = 10.0
    s1, s2 = 1.0, 2.0
    n_trials = 3000
    integ = CueIntegrator(true_value, [s1, s2], seed=0)
    readings1, readings2, fused_x, fused_var = [], [], [], []
    for _ in range(n_trials):
        r, x_hat, var_hat = integ.fuse_sample()
        readings1.append(r[0])
        readings2.append(r[1])
        fused_x.append(x_hat)
        fused_var.append(var_hat)

    _ax, analytic_var = _mle_fuse([true_value, true_value], [s1 ** 2, s2 ** 2])
    emp_var = _variance(fused_x)
    mean_gain_var = sum(fused_var) / n_trials
    single_var1, single_var2 = s1 ** 2, s2 ** 2
    best_single_var = min(single_var1, single_var2)
    rmse1 = _rmse(readings1, true_value)
    rmse2 = _rmse(readings2, true_value)
    rmse_fused = _rmse(fused_x, true_value)

    print("=" * 70)
    print("SPIKING PPC CUE INTEGRATION -- reliability-weighted fusion by spike addition")
    print("=" * 70)
    print(f"reliability weights (sigma {s_reliable} vs {s_unreliable}): "
          f"reliable {w_reliable[0]:.3f}  unreliable {w_reliable[1]:.3f} -> reliable dominates")
    print(f"reliability weights (equal sigma 2.0 / 2.0)  : "
          f"{w_equal[0]:.3f} / {w_equal[1]:.3f} -> equal")
    print(f"spiking gain : reliable cue {gain_reliable} spikes  vs unreliable {gain_unreliable} "
          f"-> more reliable = more spikes (gain ~ 1/variance)")
    print(f"fusion       : summed population = {sum(fused_pop)} spikes "
          f"(= {gain_reliable} + {gain_unreliable}, per-neuron spike addition)")
    print(f"spiking fuse : fuse([10,12],[1,4]) -> x_hat {hx:.4f} (MLE {mle_x:.1f})  "
          f"var_hat {hv:.4f} (MLE {mle_v:.1f})")
    print(f"fused var    : gain-decoded {mean_gain_var:.4f}  empirical {emp_var:.4f}  "
          f"vs analytic {analytic_var:.4f}")
    print(f"single-cue var: cue1 {single_var1:.4f}  cue2 {single_var2:.4f}  "
          f"-> fused {mean_gain_var:.4f} < best single {best_single_var:.4f}")
    print(f"RMSE to true : cue1 {rmse1:.4f}  cue2 {rmse2:.4f}  fused {rmse_fused:.4f}")
    print("=" * 70)

    # ---- self-checks (genuine, not rigged) ----
    # (a1) reliability weighting: reliable cue dominates; equal sigmas -> equal weights.
    assert w_reliable[0] > 4 * w_reliable[1], (
        f"reliable cue must dominate ({w_reliable[0]:.3f} vs {w_reliable[1]:.3f})")
    assert abs(w_equal[0] - 0.5) < 1e-9 and abs(w_equal[1] - 0.5) < 1e-9, (
        f"equal sigmas must give equal weights ({w_equal})")

    # (b) GENUINELY SPIKING: discrete spikes, gain scales with reliability, fusion is summation.
    assert all(isinstance(k, int) and k >= 0 for k in pop_reliable + pop_unreliable), (
        "populations must emit discrete non-negative integer spike counts")
    assert gain_reliable > 0 and gain_unreliable > 0, "both populations must actually spike"
    assert gain_reliable > 2 * gain_unreliable, (
        f"gain (total spikes) must scale with reliability: reliable {gain_reliable} "
        f"should dwarf unreliable {gain_unreliable} (variance 1 vs 4 -> ~4x gain)")
    assert all(fused_pop[j] == pop_reliable[j] + pop_unreliable[j] for j in range(len(fused_pop))), (
        "fusion must be exact per-neuron spike-train summation")
    assert "numpy" not in sys.modules and "torch" not in sys.modules, (
        "must be pure stdlib: numpy/torch must not be imported")

    # (a2) FUNCTION: spiking fuse() approaches the closed-form MLE within a real tolerance.
    assert abs(hx - mle_x) < 0.05, (
        f"spiking fused estimate must approach the MLE x_hat ({hx:.4f} vs {mle_x:.1f})")
    assert abs(hv - mle_v) / mle_v < 0.03, (
        f"spiking fused variance must approach the analytic var_hat ({hv:.4f} vs {mle_v:.1f})")
    assert hv < single_var1 and hv < single_var2, (
        f"fused var_hat must beat every single cue ({hv:.4f} < {single_var1}, {single_var2})")
    assert abs(hx - 10.0) < abs(hx - 12.0), (
        f"the more reliable cue (10, var 1) must dominate the fused estimate ({hx:.4f})")

    # (a3) FUNCTION (empirical): single-trial spiking fusion beats every single cue.
    assert abs(mean_gain_var - analytic_var) / analytic_var < 0.03, (
        f"gain-decoded fused variance must track analytic var_hat "
        f"({mean_gain_var:.4f} vs {analytic_var:.4f})")
    assert emp_var < best_single_var, (
        f"empirical fused-estimate variance must beat the best single cue "
        f"({emp_var:.4f} < {best_single_var:.4f})")
    assert rmse_fused < rmse1 and rmse_fused < rmse2, (
        f"fused RMSE must beat each cue ({rmse_fused:.4f} vs {rmse1:.4f}, {rmse2:.4f})")

    # (c) DETERMINISM: same true/sigmas/seed -> identical spiking trials; fuse() is stable.
    a = CueIntegrator(7.0, [1.0, 2.5], seed=3)
    b = CueIntegrator(7.0, [1.0, 2.5], seed=3)
    assert [a.fuse_sample() for _ in range(100)] == [b.fuse_sample() for _ in range(100)], (
        "same seed must reproduce identical spiking fused trials")
    assert fuse([10.0, 12.0], [1.0, 4.0]) == (hx, hv), (
        "fuse() must be deterministic (fixed internal seed)")

    print(f"self-check OK: gain reliable {gain_reliable} > unreliable {gain_unreliable} "
          f"(spikes ~ 1/var); fusion = spike sum -> {sum(fused_pop)}; spiking fuse x_hat "
          f"{hx:.4f} ~ MLE {mle_x:.1f}, var {hv:.4f} ~ {mle_v:.1f} < best cue "
          f"{best_single_var:.1f}; RMSE fused {rmse_fused:.4f} < cue1 {rmse1:.4f}; deterministic")


if __name__ == "__main__":
    main()

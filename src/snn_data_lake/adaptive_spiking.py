"""
v2.68 — adaptive sigma-delta spike coding  (stdlib-only, zero deps).

An ADAPTIVE SPIKING NEURON (ASN) encoder: a *predictive* / sigma-delta coder that turns a
continuous analog signal s(t) into a sparse train of signed spikes. It is a CORE peer to the
`encode_latency` / `encode_poisson` encoders in `spike_preprocessing.py` — but where those emit a
FIXED-rate code (spike count tracks feature MAGNITUDE, no reconstruction feedback), this one is a
closed-loop coder whose spike count tracks signal VARIATION, not magnitude or duration.

How it works (Zambrano & Bohte, refs below):

  * The neuron carries an internal RECONSTRUCTION `s_hat` (the running sum of the impulses it has
    already emitted — `reconstruct` replays exactly this) and an ADAPTIVE threshold `theta`.
  * At each sample it looks at the residual r = s - s_hat. While |r| >= theta it emits a signed
    impulse of size kappa*theta toward s (predictive sigma-delta: it only spikes for the part of
    the signal it has NOT yet explained), moving s_hat closer.
  * After every spike theta ADAPTS UP MULTIPLICATIVELY (theta *= bump); every step it relaxes back
    toward the baseline theta0 (theta = theta0 + (theta - theta0)*decay).

The multiplicative bump is the whole point. On a fast transient or a big jump, consecutive spikes
drive theta up geometrically, so the step size grows and a LARGE excursion is covered by a handful
of ever-bigger spikes (logarithmic, not linear, in the jump size). On smooth or constant stretches
theta relaxes to theta0 and the coder tracks finely with few spikes. A constant hold, once caught,
costs ZERO spikes no matter how long it lasts.

The efficiency thesis, stated as a FAIR claim (see `main`): compared to a NON-ADAPTIVE control — the
SAME coder with theta frozen at a fixed value, its threshold BISECTION-TUNED to the same
reconstruction RMSE (its best fixed step for that fidelity, not a crippled one) — the adaptive coder
reaches equal reconstruction error with FEWER spikes. Measured on the mixed-dynamics test signal:
241 adaptive spikes vs 411 fixed at RMSE ~= 0.061 (matched to 0.2%), corr 0.993 — a 1.71x saving.

Deterministic by construction (no RNG anywhere): identical input -> identical spikes.

Run:  python -m snn_data_lake.adaptive_spiking

Refs:
  Zambrano, D. & Bohte, S.M. (2016), "Fast and Efficient Asynchronous Neural Computation with
    Adapting Spiking Neural Networks", arXiv:1609.02053.
  Bohte, S.M. (2012), "Efficient Spike-Coding with Multiplicative Adaptation in a Spike Response
    Model", NeurIPS 2012.
"""
import math

from snn_data_lake._mathutils import pearson

THETA0 = 0.05       # baseline / resting threshold (fine tracking resolution)
BUMP = 1.4          # multiplicative theta bump applied AFTER each spike (> 1)
DECAY = 0.5         # per-step relaxation of (theta - theta0) toward the baseline (0..1)
KAPPA = 1.0         # impulse size as a fraction of the current threshold
THETA_MAX = 8.0     # cap on theta so a runaway transient cannot diverge


# ---- ASN encoder + reconstruction -------------------------------------------
def encode(signal, theta0=THETA0, bump=BUMP, decay=DECAY, kappa=KAPPA,
           s_hat0=0.0, adapt=True, theta_max=THETA_MAX):
    """Adaptive sigma-delta encode: continuous `signal` -> [(t, signed_impulse), ...].

    Walks the signal once holding a reconstruction s_hat and an adaptive threshold theta. At each
    sample, WHILE the residual |s - s_hat| >= theta it emits a signed impulse of kappa*theta toward
    s (asynchronous sigma-delta: as many events as the residual needs) and, when adapting, bumps
    theta UP multiplicatively; every step theta relaxes toward theta0. `adapt=False` freezes theta at
    theta0 -> the NON-ADAPTIVE control (plain fixed-step delta modulation, SAME code path). No RNG:
    the same input always yields the same spikes. `s_hat0` seeds the reconstruction (0.0 = start
    from silence)."""
    spikes = []
    s_hat = s_hat0
    theta = theta0
    for t, s in enumerate(signal):
        while True:
            r = s - s_hat
            if abs(r) < theta:
                break
            inc = (kappa * theta) if r > 0.0 else (-kappa * theta)
            s_hat += inc
            spikes.append((t, inc))
            if adapt:
                theta *= bump
                if theta > theta_max:
                    theta = theta_max
        if adapt:
            theta = theta0 + (theta - theta0) * decay
    return spikes


def reconstruct(spikes, length, s_hat0=0.0):
    """Rebuild s_hat(t) for t in range(length) from the signed-impulse train: a right-continuous
    staircase, s_hat(t) = s_hat0 + sum of impulses at times <= t. This is exactly the s_hat the
    encoder tracked, so decode is a plain running sum. (Each impulse magnitude is the adaptation gain
    kappa*theta at emission -- a deterministic function of the spike TIMING and fixed params -- so the
    spike COUNT is the sole efficiency-relevant quantity, and the adaptive and control coders share
    this identical representation: no information is smuggled into the amplitudes.)"""
    out = [0.0] * length
    level = s_hat0
    j = 0
    n = len(spikes)
    for t in range(length):
        while j < n and spikes[j][0] == t:
            level += spikes[j][1]
            j += 1
        out[t] = level
    return out


# ---- fidelity metrics + fair-matching of the non-adaptive control -----------
def rmse(a, b):
    """Root-mean-square reconstruction error between signal `a` and reconstruction `b`."""
    n = len(a)
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)) / n)


def correlation(a, b):
    """Pearson correlation of signal `a` and reconstruction `b` (fidelity of shape, 1.0 = perfect)."""
    return pearson(a, b)


def fixed_control_at_error(signal, target_rmse, s_hat0=0.0, iters=90):
    """Tune the NON-ADAPTIVE control (theta frozen) so its reconstruction RMSE matches `target_rmse`.

    The asynchronous coder's RMSE is monotone INCREASING in the fixed threshold (bigger step ->
    coarser -> more error), so a bisection on theta finds the fixed step that reaches the target
    fidelity. This is the FAIR baseline for the efficiency claim: the SAME coder given its BEST fixed
    threshold for the desired error (never crippled), so any spike-count gap is due to ADAPTATION
    alone -- not to a mismatched threshold. Returns (theta_fixed, spikes, achieved_rmse)."""
    length = len(signal)

    def rmse_of(theta):
        sp = encode(signal, theta0=theta, adapt=False, s_hat0=s_hat0)
        return rmse(signal, reconstruct(sp, length, s_hat0)), sp

    lo, hi = 1e-4, 0.05
    while rmse_of(hi)[0] < target_rmse and hi < 1000.0:   # expand until the step is coarse enough
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rmse_of(mid)[0] > target_rmse:                 # too coarse -> shrink the step
            hi = mid
        else:
            lo = mid
    theta = 0.5 * (lo + hi)
    achieved, sp = rmse_of(theta)
    return theta, sp, achieved


# ---- test signal with MIXED dynamics ----------------------------------------
def make_signal(n=600):
    """A mixed-dynamics test signal in [-1, 1]: mostly slow/flat with short fast bursts -- exactly
    where an adaptive coder earns its keep.
        [0,   200)  slow sinusoid   (smooth, low-frequency)
        [200, 320)  constant hold   (zero variation)
        [320, 380)  fast burst      (high-frequency, high-amplitude)
        [380, 440)  square jumps    (big +/-1.0 steps -> large excursions)
        [440, n)    slow sinusoid   (smooth again)"""
    sig = []
    for t in range(n):
        if t < 200:
            v = 0.5 * math.sin(2 * math.pi * t / 150.0)
        elif t < 320:
            v = 0.4
        elif t < 380:
            v = 0.9 * math.sin(2 * math.pi * (t - 320) / 9.0)
        elif t < 440:
            v = -1.0 if (t // 10) % 2 else 1.0
        else:
            v = 0.5 * math.sin(2 * math.pi * (t - 440) / 150.0)
        sig.append(v)
    return sig


# ---- demo + measure + self-checks -------------------------------------------
def main():
    sig = make_signal()
    length = len(sig)

    # adaptive coder
    sp_a = encode(sig)
    rec_a = reconstruct(sp_a, length)
    n_a = len(sp_a)
    rmse_a = rmse(sig, rec_a)
    corr_a = correlation(sig, rec_a)

    # non-adaptive control, threshold tuned to the SAME reconstruction error (fair match)
    theta_fix, sp_f, rmse_f = fixed_control_at_error(sig, rmse_a)
    n_f = len(sp_f)
    match_rel = abs(rmse_f - rmse_a) / rmse_a

    # variation scaling: equal-length CONSTANT vs FAST segment (count tracks variation, not duration)
    seg = 120
    const = [0.4] * seg
    fast = [0.9 * math.sin(2 * math.pi * i / 8.0) for i in range(seg)]
    n_const = len(encode(const, s_hat0=const[0]))     # seeded at its own level -> pure-variation test
    n_fast = len(encode(fast, s_hat0=fast[0]))

    # determinism
    deterministic = (encode(sig) == encode(sig))

    print("=" * 64)
    print("ADAPTIVE SIGMA-DELTA SPIKE CODING  (ASN, Zambrano-Bohte)")
    print("=" * 64)
    print(f"signal            : {length} samples, range [{min(sig):.2f}, {max(sig):.2f}], "
          f"mixed slow/hold/burst/jumps")
    print("efficiency @ MATCHED reconstruction error (RMSE, not matched threshold):")
    print(f"  adaptive        : {n_a:4d} spikes   RMSE {rmse_a:.4f}   corr {corr_a:.4f}")
    print(f"  non-adaptive    : {n_f:4d} spikes   RMSE {rmse_f:.4f}   corr "
          f"{correlation(sig, reconstruct(sp_f, length)):.4f}   (theta_fix {theta_fix:.4f})")
    print(f"  -> matched to {match_rel*100:.2f}% error; adaptive uses {n_f - n_a} FEWER spikes "
          f"({n_f / n_a:.2f}x) at equal fidelity")
    print(f"variation scaling : equal-length ({seg}) segments -> "
          f"constant {n_const} spikes vs fast {n_fast} spikes")
    print(f"determinism       : {deterministic}  (identical spikes on a repeat encode)")
    print("=" * 64)

    # ---- self-checks --------------------------------------------------------
    # (a) FIDELITY: the adaptive coder reconstructs the signal well.
    assert rmse_a < 0.10, f"adaptive reconstruction RMSE too high: {rmse_a:.4f}"
    assert corr_a > 0.98, f"adaptive reconstruction correlation too low: {corr_a:.4f}"

    # (b) EFFICIENCY (the defining property): at MATCHED reconstruction error the adaptive coder
    #     emits clearly FEWER spikes than the best-fixed-threshold control (which is NOT crippled --
    #     its threshold is bisection-tuned to the same RMSE). Compared on error, not on threshold.
    assert match_rel <= 0.05, f"control not matched on error: rel diff {match_rel:.4f}"
    assert n_a < n_f, f"adaptive did not beat the control: adaptive {n_a} vs fixed {n_f}"
    assert n_f >= 1.3 * n_a, f"efficiency margin too thin: fixed {n_f} vs adaptive {n_a}"

    # (c) VARIATION SCALING: a constant segment costs far fewer spikes than a fast segment of the
    #     SAME length -- spike count follows variation, not magnitude or duration.
    assert n_const < n_fast, f"constant segment not cheaper than fast: {n_const} vs {n_fast}"
    assert n_const <= 2, f"constant segment should be ~free, got {n_const} spikes"
    assert n_fast >= 50, f"fast segment should be spike-dense, got {n_fast} spikes"

    # (d) DETERMINISM: identical output on a repeat run.
    assert deterministic, "encode is not deterministic"

    # reconstruction is a faithful running sum of the impulses the encoder tracked
    assert reconstruct(sp_a, length) == rec_a, "reconstruct is not a pure function of the spikes"

    print("self-check OK: high-fidelity reconstruction, FEWER spikes than the fixed-threshold "
          "control at matched error, spike count scales with variation not duration, deterministic")


if __name__ == "__main__":
    main()

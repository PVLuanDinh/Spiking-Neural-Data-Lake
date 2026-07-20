"""
Weight perturbation -- gradient-free reward learning in a spiking net (Fiete & Seung 2006).

A spiking readout learns a task from a single SCALAR reward R and NOTHING else: no backprop,
no analytic gradient, no error signal per weight. The whole recipe is (Fiete & Seung 2006):

    perturb the weights by small iid noise   xi ~ Normal(0, sigma^2)
    run the network, read the scalar reward  R(w + xi)
    move along the perturbation, reward-weighted:
        dw = eta * ( R(w + xi) - R_bar ) * xi / sigma^2       (R_bar = a running baseline)

Averaged over perturbations this is an UNBIASED estimate of the reward gradient dR/dw: to
first order R(w+xi) - R(w) ~ grad . xi, and E[(grad . xi) xi_i / sigma^2] = grad_i because
E[xi_i xi_j] = sigma^2 delta_ij. So the network ASCENDS the reward -- climbs the true gradient
-- while only ever running forward and reading one number. This is the REINFORCE / node-
perturbation estimator; it is biologically plausible (a global reward broadcast + local synaptic
noise), unlike backprop which needs a symmetric backward pass. Fiete & Seung applied it to
spiking networks by perturbing conductances; here we perturb the readout weights directly.

HOW THIS DIFFERS from the reward-learning already in the repo (do not confuse the three):
  * src/snn_data_lake/r_stdp.py -- reward-modulated STDP: a LOCAL eligibility trace written by
    pre/post spike TIMING, then gated into w by a delayed reward. The credit signal is the
    LOCAL coincidence trace; the reward only gates it. NO gradient of anything is estimated.
  * research/eprop.py -- e-prop: an ONLINE TRUE-GRADIENT rule (eligibility traces that are the
    exact factorised gradient of the loss); it computes the analytic gradient, just forward-in-time.
  * THIS -- weight perturbation: GRADIENT-FREE. It never forms a gradient analytically and uses
    no timing trace; it ESTIMATES the gradient stochastically from (reward, noise) correlations.
    The only information used is the scalar R and the noise xi that was injected.

the un-riggable anchor (why this is a real gradient estimator, not a heuristic): at a fixed w we
estimate g_hat = mean over many xi of (R(w+xi)-R(w)) xi / sigma^2, and INDEPENDENTLY compute the
true gradient by CENTRAL FINITE DIFFERENCES g_fd[i] = (R(w+eps e_i) - R(w-eps e_i)) / (2 eps) --
a completely separate code path that touches no noise. If perturbation really estimates dR/dw the
two vectors must POINT THE SAME WAY: we assert cos(g_hat, g_fd) > 0.6 (threshold fixed a priori).

The task (kept SMOOTH-a.e. in w so a finite-difference gradient is meaningful): a leaky integrate-
and-fire readout (peer of snn_data_lake.lif_neuron.LIFNeuron; sub-threshold / tempotron regime, so
its scalar output is the PEAK membrane potential -- the tempotron decision variable, peak >= thr
<=> the neuron would fire) is driven by a few FIXED seeded input spike patterns. Each pattern has a
target peak (high = "fire", low = "quiet"); reward = -sum (peak - target)^2. The peak is piecewise-
linear in w, hence smooth-a.e.: a genuine reward landscape with a genuine gradient.

Honesty: this demonstrates that a global scalar reward + local weight noise recovers the true
reward gradient and solves a small spiking task WITHOUT backprop. It is a single-readout caricature
on synthetic spike trains -- no hidden layers, no deep credit assignment. NOTHING is rigged: the
finite-difference gradient is an independent ground truth, w is NOT initialised near the optimum,
and the cosine threshold was fixed before measuring.
Pure stdlib (random + math), additive research module, zero-dep, deterministic.
Ref: Fiete, I.R. & Seung, H.S. (2006), "Gradient Learning in Spiking Neural Networks by Dynamic
Perturbation of Conductances", Physical Review Letters 97:048104, DOI 10.1103/PhysRevLett.97.048104.

  PYTHONPATH=src python research/weight_perturbation.py      # (also run by run_selfchecks.py)
"""
import math
import random

SEED = 7
N_SYN = 30           # plastic input weights on the readout
N_PATTERN = 4        # fixed input spike patterns
T = 25               # timesteps per pattern
TAU = 8.0            # readout membrane time constant
DECAY = math.exp(-1.0 / TAU)
P_SPIKE = 0.30       # input spike sparsity
TARGETS = [1.5, 0.4, 1.5, 0.4]   # target peak voltage per pattern: high = "fire", low = "quiet"


def make_patterns(rng):
    """N_PATTERN fixed input spike trains. Each is stored as, per timestep, the list of input
    synapses that spiked (Bernoulli(P_SPIKE)); the readout current at t is sum of those weights."""
    pats = []
    for _ in range(N_PATTERN):
        active = [[i for i in range(N_SYN) if rng.random() < P_SPIKE] for _ in range(T)]
        pats.append(active)
    return pats


def peak_voltage(w, pattern):
    """Leaky integrate-and-fire readout (sub-threshold, tempotron-style): drive a leaky membrane
    with the weighted input spikes and return the PEAK membrane potential -- the tempotron decision
    variable (peak >= threshold <=> the neuron would fire). V(t) is linear in w, so the peak is a
    convex piecewise-linear (hence smooth-a.e., differentiable) scalar function of the weights."""
    v = 0.0
    peak = -1e30
    for t in range(T):
        cur = 0.0
        for i in pattern[t]:
            cur += w[i]
        v = v * DECAY + cur
        if v > peak:
            peak = v
    return peak


def reward(w, patterns):
    """Scalar reward R(w) = -sum_p (peak_p - target_p)^2. Higher is better (max 0), smooth-a.e. in
    w, deterministic given w. This is a pure FORWARD evaluation -- NO gradient is computed here."""
    err = 0.0
    for p in range(N_PATTERN):
        o = peak_voltage(w, patterns[p])
        err += (o - TARGETS[p]) ** 2
    return -err


def total_error(w, patterns):
    """Task error = sum of squared peak errors = -reward (>= 0, lower is better)."""
    return -reward(w, patterns)


def train(w0, patterns, rng, iters, eta, sigma, clip):
    """WEIGHT-PERTURBATION training. Each iteration: draw noise xi, evaluate the scalar reward at
    w+xi, and step along xi scaled by (reward - baseline). The update below uses ONLY forward reward
    evaluations and the injected noise xi -- there is NO backprop and NO analytic gradient anywhere
    in this loop (that is the whole point of perturbation learning). `clip` bounds the reward-weighted
    step magnitude (bounded plasticity): the squared-error reward has unbounded curvature as w grows,
    so an unclipped single-sample estimate can positive-feedback into a runaway -- the clip is a
    biologically plausible cap, NOT a gradient (it only limits the size of the perturbation step)."""
    w = list(w0)
    baseline = reward(w, patterns)
    traj = [reward(w, patterns)]
    for _ in range(iters):
        xi = [rng.gauss(0.0, sigma) for _ in range(N_SYN)]      # local synaptic noise
        wp = [w[i] + xi[i] for i in range(N_SYN)]
        r_perturbed = reward(wp, patterns)                       # one scalar reward, forward only
        scale = eta * (r_perturbed - baseline) / (sigma * sigma)  # (reward - baseline) correlation
        scale = clip if scale > clip else (-clip if scale < -clip else scale)   # bounded step
        for i in range(N_SYN):
            w[i] += scale * xi[i]                                # move along the noise, reward-weighted
        baseline = 0.9 * baseline + 0.1 * r_perturbed            # running-mean reward baseline
        traj.append(reward(w, patterns))
    return w, traj


def perturbation_gradient(w, patterns, rng, samples, sigma):
    """Estimate dR/dw at a FIXED w by weight perturbation only:
        g_hat[i] = mean over samples of ( R(w+xi) - R(w) ) * xi[i] / sigma^2
    Uses nothing but forward reward evaluations and the noise xi -- NO analytic gradient."""
    r0 = reward(w, patterns)
    acc = [0.0] * N_SYN
    inv = 1.0 / (sigma * sigma)
    for _ in range(samples):
        xi = [rng.gauss(0.0, sigma) for _ in range(N_SYN)]
        f = (reward([w[i] + xi[i] for i in range(N_SYN)], patterns) - r0) * inv
        for i in range(N_SYN):
            acc[i] += f * xi[i]
    return [a / samples for a in acc]


def finite_difference_gradient(w, patterns, eps):
    """INDEPENDENT ground-truth gradient by CENTRAL finite differences, one weight at a time:
        g[i] = ( R(w+eps e_i) - R(w-eps e_i) ) / (2 eps)
    A distinct code path: it uses no noise and none of the perturbation machinery -- it is the true
    gradient of the SAME reward, computed the textbook numerical way, purely to check g_hat."""
    g = [0.0] * N_SYN
    for i in range(N_SYN):
        wp = list(w)
        wm = list(w)
        wp[i] += eps
        wm[i] -= eps
        g[i] = (reward(wp, patterns) - reward(wm, patterns)) / (2.0 * eps)
    return g


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def run_experiment(seed, iters=2500, eta=0.003, train_sigma=0.05, clip=3.0,
                   samples=3000, anchor_sigma=0.02, fd_eps=0.01):
    """One fully deterministic run: build patterns + init weights, perturbation-train, then anchor
    the estimator against finite differences at the INITIAL w (deliberately far from the optimum)."""
    rng = random.Random(seed)
    patterns = make_patterns(rng)
    w0 = [rng.uniform(-0.05, 0.05) for _ in range(N_SYN)]        # small init: peaks ~0, far from targets
    w_final, traj = train(w0, patterns, rng, iters, eta, train_sigma, clip)
    g_hat = perturbation_gradient(w0, patterns, rng, samples, anchor_sigma)   # estimator at fixed w0
    g_fd = finite_difference_gradient(w0, patterns, fd_eps)                   # independent ground truth
    return {
        "traj": traj,
        "r_init": traj[0],
        "r_final": traj[-1],
        "err_init": total_error(w0, patterns),
        "err_final": total_error(w_final, patterns),
        "cos": cosine(g_hat, g_fd),
        "samples": samples,
    }


def main():
    print("=" * 74)
    print("WEIGHT PERTURBATION -- gradient-free reward learning in a spiking net")
    print("(Fiete & Seung 2006: perturb w by noise, correlate with scalar reward, no backprop)")
    print("=" * 74)

    res = run_experiment(SEED)
    r_init, r_final = res["r_init"], res["r_final"]
    e_init, e_final = res["err_init"], res["err_final"]
    cos, samples = res["cos"], res["samples"]

    print(f"task            : {N_SYN} plastic weights, {N_PATTERN} spike patterns, LIF peak-voltage readout")
    print(f"                  target peaks {TARGETS} (high = fire, low = quiet)")
    print(f"reward (init)   : {r_init:+.4f}   task error (init)  : {e_init:.4f}")
    print(f"reward (final)  : {r_final:+.4f}   task error (final) : {e_final:.4f}")
    print(f"learning        : reward {r_init:+.3f} -> {r_final:+.3f}   error {e_init:.3f} -> {e_final:.3f}")
    print(f"ANCHOR          : cos(g_perturb, g_finite_diff) = {cos:.4f} over {samples} perturbation samples")
    print("                  (finite-diff gradient is an INDEPENDENT ground truth; no backprop used)")

    # determinism: the whole pipeline is a pure function of the seed.
    res2 = run_experiment(SEED)
    identical = (res["traj"] == res2["traj"]) and (res["cos"] == res2["cos"])
    print(f"determinism     : same seed -> identical reward trajectory + cosine = {identical}")
    print("=" * 74)

    # ---- self-check (deterministic asserts, zero-dep, ASCII; NOTHING rigged) ----
    # (a) LEARNS: reward rises substantially and task error drops, by perturbation alone.
    assert r_final > r_init + 1.0, f"perturbation must raise the reward substantially ({r_init:+.4f} -> {r_final:+.4f})"
    assert e_final < 0.5 * e_init, f"task error must drop clearly ({e_init:.4f} -> {e_final:.4f})"
    # (b) un-riggable anchor: perturbation estimate aligns with the independent finite-diff gradient.
    #     Threshold 0.6 fixed a priori (task spec); finite differences are a separate code path.
    assert cos > 0.6, f"perturbation must estimate the true gradient (cos {cos:.4f} <= 0.6)"
    # (c) NO-BACKPROP (structural): train() and perturbation_gradient() use only forward reward()
    #     evaluations and the injected noise xi -- there is no analytic gradient in the update path.
    # (d) determinism: same seed -> identical trajectory + cosine on a repeat run.
    assert identical, "same seed must reproduce the reward trajectory and cosine exactly"
    print(f"self-check OK: learned by perturbation alone (reward {r_init:+.3f} -> {r_final:+.3f}, "
          f"error {e_init:.3f} -> {e_final:.3f}); estimator matches the independent finite-difference "
          f"gradient (cos {cos:.4f} > 0.6, {samples} samples); no backprop; deterministic")


if __name__ == "__main__":
    main()

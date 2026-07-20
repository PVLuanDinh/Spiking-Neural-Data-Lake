"""
STDP-WTA Bayesian computation -- unsupervised online EM emerges from spike-timing plasticity.

Nessler, Pfeiffer, Buesing & Maass (2013) proved that a generic winner-take-all (WTA) cortical
microcircuit, wired with a soft-max lateral inhibition and a simple STDP rule, performs UNSUPERVISED
online expectation-maximization for a generative MIXTURE model -- Bayesian inference is not programmed
in, it EMERGES from the plasticity. Over a binary input x, neuron k's membrane is u_k = b_k + sum_i
w_ki*y_i (with y a complementary/population code of x); the WTA soft-max p(k|x) = softmax(u) IS the
posterior over hidden causes; and the STDP-like update on the winner drives exp(w_ki) -> P(y_i=1|k) and
exp(b_k) -> P(k). After learning, each WTA neuron has become ONE mixture component and the winning neuron
identifies the cause that generated the input. The canonical winner update on a spike is
dw_ki ~ eta*(y_i*exp(-w_ki) - 1), whose fixed point is exp(w_ki) = <y_i | k wins> = P(y_i=1|k).

Faithful to the paper, each binary feature x_i is POPULATION-CODED by two input lines -- an "on" line
(fires when x_i=1) and an "off" line (fires when x_i=0) -- so the membrane u_k is a proper log-likelihood
log P(x|k) + log P(k), not just the x_i=1 half. exp(w_on_ki) converges to P(x_i=1|k) (the template) and
exp(w_off_ki) to P(x_i=0|k). Weights start at the UNINFORMATIVE log(1/2) (equidistant from both template
levels -- not the answer) plus tiny noise for symmetry breaking.

DISTINCT from the other Bayesian/sampling research here:
  - bayesian_snn.py     -- MC-dropout-style UNCERTAINTY CALIBRATION on a fixed recall (masked passes,
                           disagreement rate). No learning of a generative model; this module LEARNS one.
  - neural_sampling.py  -- MCMC (samples a fixed distribution). This is online EM that FITS the model.
  - restricted_boltzmann.py -- a trained RBM density model. Here the density (a mixture) is discovered
                           unsupervised by local STDP in a WTA, not by contrastive-divergence gradients.

This demo plants a K-component Bernoulli mixture (distinct prototype templates + a non-uniform prior),
streams samples through the WTA with the Nessler STDP rule -- UNSUPERVISED, the true component label is
NEVER used during training -- then, only AFTER training, uses the labels to score: each neuron is matched
to the component it best represents and we measure (a) clustering accuracy on held-out data, (b) the
correlation between the learned exp(w_on) and the true templates (the proof the weights became log-probs),
(c) that the posterior is a proper distribution concentrating on the right cause. Weights start
uninformative (no planted answer); training is label-free; everything is measured, nothing is rigged.

Ref: Nessler, B., Pfeiffer, M., Buesing, L. & Maass, W. (2013), "Bayesian Computation Emerges in Generic
Cortical Microcircuits through Spike-Timing-Dependent Plasticity", PLoS Comput Biol 9(4):e1003037,
DOI 10.1371/journal.pcbi.1003037.

  python research/stdp_wta_bayes.py        # (also run by run_selfchecks.py)
"""
import itertools
import math
import random

from snn_data_lake._mathutils import softmax

K = 4              # WTA neurons == planted mixture components
D = 25             # binary features (a 5x5 field)
N_TRAIN = 8000     # online stream length (single pass)
N_TEST = 1500      # held-out samples for scoring
P_HIGH = 0.85      # foreground Bernoulli prob in a component template
P_LOW = 0.15       # background Bernoulli prob
SEED = 7
LOG_HALF = math.log(0.5)   # uninformative init: exp(w)=1/2, equidistant from P_LOW and P_HIGH
W_LO, W_HI = -6.0, 0.5     # weight clamp (exp(w) is a probability -> w <= ~0)


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def plant_mixture(rng):
    """K distinct binary prototype masks -> per-feature Bernoulli templates + a non-uniform prior.

    Masks are drawn from the rng and re-drawn until pairwise well-separated, so the mixture is genuinely
    separable (not an accident of a lucky seed) -- but nothing about the ANSWER is handed to the learner.
    """
    masks = []
    for _ in range(K):
        while True:
            m = [1 if rng.random() < 0.4 else 0 for _ in range(D)]
            if sum(m) >= 5 and all(_hamming(m, prev) >= D // 4 for prev in masks):
                break
        masks.append(m)
    templates = [[P_HIGH if bit else P_LOW for bit in m] for m in masks]
    raw = [rng.random() + 0.3 for _ in range(K)]        # non-uniform prior over causes
    total = sum(raw)
    prior = [r / total for r in raw]
    return templates, prior


def _cumulative(p):
    out, run = [], 0.0
    for v in p:
        run += v
        out.append(run)
    return out


def _pick(cum, u):
    for i, c in enumerate(cum):
        if u <= c:
            return i
    return len(cum) - 1


def sample_dataset(templates, prior, n, rng):
    """Pick a component by the prior, then sample each feature Bernoulli(template[c][i])."""
    cum = _cumulative(prior)
    data = []
    for _ in range(n):
        c = _pick(cum, rng.random())
        x = [1 if rng.random() < templates[c][i] else 0 for i in range(D)]
        data.append((x, c))
    return data


def membrane(x, w_on, w_off, b):
    """u_k = b_k + sum_i (w_on_ki if x_i else w_off_ki) -- a proper log-likelihood via complementary code."""
    return [b[k] + sum(w_on[k][i] if x[i] else w_off[k][i] for i in range(D)) for k in range(K)]


def posterior(x, w_on, w_off, b):
    """p(k|x) = softmax(u) -- the WTA soft-max IS the posterior over hidden causes."""
    return softmax(membrane(x, w_on, w_off, b))


def _clamp(v):
    return W_LO if v < W_LO else W_HI if v > W_HI else v


def train(data, rng, eta0=0.05, tau=2000.0):
    """Unsupervised online STDP-WTA (Nessler rule). The component label is NEVER read here.

    Per sample: form the posterior, STOCHASTICALLY select a winner z ~ p(k|x) (the WTA spike), then apply
    the winner update dw = eta*(y*exp(-w) - 1) to every synapse [y=1 if that input line fired] -- fixed
    point exp(w) = P(line fired | z) -- and the homeostatic bias db_k = eta*(1{k==z}*exp(-b_k) - 1),
    fixed point exp(b_k) = P(k).
    """
    w_on = [[LOG_HALF + rng.uniform(-0.05, 0.05) for _ in range(D)] for _ in range(K)]   # uninformative
    w_off = [[LOG_HALF + rng.uniform(-0.05, 0.05) for _ in range(D)] for _ in range(K)]  # + tiny noise
    b = [0.0 for _ in range(K)]
    for t, (x, _label) in enumerate(data):        # _label is deliberately ignored -> unsupervised
        eta = eta0 * tau / (tau + t)              # decaying rate -> convergence
        p = posterior(x, w_on, w_off, b)
        z = _pick(_cumulative(p), rng.random())   # stochastic winner (sampled from the posterior)
        won, woff = w_on[z], w_off[z]
        for i in range(D):
            on = x[i]                             # exactly one of the two lines fires
            won[i] = _clamp(won[i] + eta * (on * math.exp(-won[i]) - 1.0))
            woff[i] = _clamp(woff[i] + eta * ((1 - on) * math.exp(-woff[i]) - 1.0))
        for k in range(K):
            hit = 1.0 if k == z else 0.0
            b[k] = _clamp(b[k] + eta * (hit * math.exp(-b[k]) - 1.0))
    return w_on, w_off, b


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0.0 or vb == 0.0:
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(va * vb)


def best_assignment(w_on, templates):
    """Brute-force the neuron<->component permutation maximizing summed corr(exp(w_on_k), template_c).

    Uses ONLY learned weights + planted templates (no test labels). Returns neuron_of[c] = neuron k."""
    expw = [[math.exp(wk[i]) for i in range(D)] for wk in w_on]
    best, best_score = None, -1e9
    for perm in itertools.permutations(range(K)):      # perm[c] = neuron for component c
        score = sum(pearson(expw[perm[c]], templates[c]) for c in range(K))
        if score > best_score:
            best_score, best = score, perm
    return best


def run_experiment(seed):
    """Full deterministic experiment from one seeded rng -> a metrics dict. No global state."""
    rng = random.Random(seed)
    templates, prior = plant_mixture(rng)
    train_set = sample_dataset(templates, prior, N_TRAIN, rng)
    test_set = sample_dataset(templates, prior, N_TEST, rng)
    w_on, w_off, b = train(train_set, rng)

    neuron_of = best_assignment(w_on, templates)        # component -> neuron
    comp_of = {neuron_of[c]: c for c in range(K)}        # neuron -> component

    # confusion on held-out data (labels used ONLY here, to score -- never in train())
    conf = [[0] * K for _ in range(K)]                   # conf[true_c][winner_neuron]
    for x, c in test_set:
        u = membrane(x, w_on, w_off, b)
        winner = max(range(K), key=lambda k: u[k])       # hard MAP inference
        conf[c][winner] += 1

    recall = [conf[c][neuron_of[c]] / sum(conf[c]) for c in range(K)]
    accuracy = sum(conf[c][neuron_of[c]] for c in range(K)) / N_TEST

    expw = [[math.exp(w_on[k][i]) for i in range(D)] for k in range(K)]
    matched_corr = [pearson(expw[neuron_of[c]], templates[c]) for c in range(K)]
    mismatch_corr = []
    for c in range(K):
        others = [pearson(expw[neuron_of[c]], templates[cc]) for cc in range(K) if cc != c]
        mismatch_corr.append(sum(others) / len(others))

    # clean-sample sharpening: the MAP prototype of each component should fire its matched neuron hard
    clean_ok, clean_mass, sums_ok = True, [], True
    for c in range(K):
        x = [1 if templates[c][i] >= 0.5 else 0 for i in range(D)]
        p = posterior(x, w_on, w_off, b)
        sums_ok = sums_ok and abs(sum(p) - 1.0) < 1e-9
        if max(range(K), key=lambda k: p[k]) != neuron_of[c]:
            clean_ok = False
        clean_mass.append(p[neuron_of[c]])

    # bias recovery: exp(b_k) -> P(k) (an extra, honest read on the prior-learning claim)
    learned_prior = softmax(b)
    prior_true = [prior[c] for c in range(K)]
    prior_learned = [learned_prior[neuron_of[c]] for c in range(K)]
    prior_corr = pearson(prior_true, prior_learned)

    return {
        "neuron_of": neuron_of, "comp_of": comp_of, "recall": recall, "accuracy": accuracy,
        "matched_corr": matched_corr, "mismatch_corr": mismatch_corr,
        "clean_ok": clean_ok, "clean_mass": clean_mass, "sums_ok": sums_ok,
        "prior_corr": prior_corr, "prior_true": prior_true, "prior_learned": prior_learned,
    }


def main():
    r = run_experiment(SEED)
    chance = 1.0 / K
    mean_matched = sum(r["matched_corr"]) / K
    mean_mismatch = sum(r["mismatch_corr"]) / K

    print("=" * 72)
    print("STDP-WTA BAYESIAN COMPUTATION (Nessler 2013) -- unsupervised online EM")
    print("=" * 72)
    print(f"planted mixture: K={K} components, D={D} binary features, "
          f"templates {P_LOW:.2f}/{P_HIGH:.2f}, non-uniform prior")
    print(f"training: {N_TRAIN} samples, single online pass, STDP-WTA, LABEL-FREE (unsupervised)")
    print("-" * 72)
    print("component | matched neuron | clustering acc | corr(exp(w), true template)")
    print("-" * 72)
    for c in range(K):
        print(f"    {c}     |       {r['neuron_of'][c]}        |     {r['recall'][c]:.3f}      "
              f"|            {r['matched_corr'][c]:.3f}")
    print("-" * 72)
    print(f"overall clustering accuracy : {r['accuracy']:.3f}   (chance 1/K = {chance:.3f})")
    print(f"mean corr(exp(w), template) : matched {mean_matched:.3f}  vs  mismatched {mean_mismatch:.3f}")
    print(f"clean-prototype posterior mass on matched neuron: "
          f"{[round(m, 3) for m in r['clean_mass']]}")
    print(f"bias recovery corr(exp(b) -> P(k)) : {r['prior_corr']:.3f}")
    print("=" * 72)

    # ---- self-check: deterministic asserts; unsupervised, nothing rigged ----
    # (a) specialization: unsupervised clustering well above chance
    assert r["accuracy"] > 0.8, f"WTA must specialize (acc {r['accuracy']:.3f} vs chance {chance:.3f})"
    assert all(rec > 0.5 for rec in r["recall"]), f"every component must be covered: {r['recall']}"
    # (b) the Bayesian claim: exp(w) converged to the conditional probabilities (log-prob weights)
    assert mean_matched > 0.8, f"learned exp(w) must track its template (matched corr {mean_matched:.3f})"
    assert mean_matched > mean_mismatch + 0.3, (
        f"matched >> mismatched required: {mean_matched:.3f} vs {mean_mismatch:.3f}")
    assert all(mc > mm for mc, mm in zip(r["matched_corr"], r["mismatch_corr"])), \
        "each neuron must correlate with its OWN component more than any other"
    # (c) the posterior is a proper distribution that concentrates on the right cause
    assert r["sums_ok"], "posterior p(k|x) must sum to 1"
    assert r["clean_ok"], "a clean prototype must fire its matched neuron as the MAP winner"
    assert all(m > 0.5 for m in r["clean_mass"]), f"posterior must concentrate on the cause: {r['clean_mass']}"
    # (d) determinism: same seed -> byte-identical metrics
    r2 = run_experiment(SEED)
    assert r2["accuracy"] == r["accuracy"], "same seed must reproduce identical clustering accuracy"
    assert r2["matched_corr"] == r["matched_corr"], "same seed must reproduce identical weight correlations"

    print(f"self-check OK: unsupervised STDP-WTA recovered the planted mixture at "
          f"{r['accuracy']:.0%} clustering accuracy (chance {chance:.0%}); learned exp(w) tracks the true "
          f"templates at corr {mean_matched:.2f} (matched) vs {mean_mismatch:.2f} (mismatched) -- the "
          f"weights became log P(x_i|k); Bayesian computation emerged from plasticity, deterministic.")


if __name__ == "__main__":
    main()

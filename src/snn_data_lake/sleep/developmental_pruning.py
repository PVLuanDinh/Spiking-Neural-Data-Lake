"""Developmental synaptic pruning for a spiking linear readout.

Brain-inspired sparsifier, after BrainCog's developmental-plasticity pruning
(Zeng et al. 2023, PMC10435966): the brain starts massively over-connected and
prunes synapses over development, converging to a sparse, energy-efficient net.
Fewer synapses means fewer SynOps -- the neuromorphic-efficiency angle here.

Model (same argmax-readout spirit as snn_classifier / trained_brain): a rate-
coded spiking linear readout. Input x in [0, 1] is a per-neuron firing rate; the
class score is the expected accumulated current s[c] = sum_i mask[c][i] * W[c][i]
* x[i]; predict argmax_c s[c]. W is trained with an online softmax delta rule
(Widrow-Hoff / perceptron style) on a TRAIN split; held-out TEST is reported.

Developmental pruning: over several rounds we prune a fraction of the SURVIVING
synapses, then keep training the survivors. Two strategies, compared at MATCHED
final sparsity (identical per-round cut counts -- only WHICH synapses differ):
  * magnitude -- the real mechanism: cut the surviving synapses with the
    SMALLEST |W| (the least-used connections), per the developmental story.
  * random    -- the fair control: cut a RANDOM subset at the same counts.

Honest claim (the acceptance bar): at the SAME final sparsity, magnitude
(keep-strong) pruning achieves HIGHER held-out accuracy than random pruning,
because the small-|W| synapses it removes are the non-discriminative ones. The
dense (no-prune) accuracy is the ceiling; the achieved mask density confirms the
comparison is truly matched. Params are fixed, not task/param-shopped: the task
starts over-connected (only a minority of input dims carry class info) and we
prune hard, so which synapses survive genuinely matters.

Note: BrainCog's DPAP also regrows a few dead synapses each round. We omit
regrowth deliberately so both strategies stay at EXACTLY matched sparsity -- the
core magnitude-vs-random comparison is what must stay honest.

  python -m snn_data_lake.sleep.developmental_pruning
"""

import math
import random

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "_train_epoch": (
        "LOCAL",
        "one online softmax delta-rule pass touching only surviving synapses. A single-layer delta "
        "rule: the error is the output error itself, not a signal carried back across a layer",
    ),
}

# ---- fixed hyperparameters (chosen once; not tuned per seed) ----------------
K = 5  # classes
D = 64  # input dimension (firing-rate vector)
N_SIGNAL = 16  # discriminative dims; the other 48 are shared distractors (no class info)
PER_TRAIN = 30  # train samples per class
PER_TEST = 20  # held-out test samples per class
SIGMA = 0.18  # gaussian sample noise (rate units)
LR = 0.2  # delta-rule learning rate
WARMUP = 15  # dense training epochs before pruning begins
ROUNDS = 6  # developmental pruning rounds
RETRAIN = 6  # survivor-training epochs after each prune round
PRUNE_FRAC = 0.30  # fraction of SURVIVORS cut each round
TARGET_DENSITY = (1.0 - PRUNE_FRAC) ** ROUNDS  # ~0.12 surviving after ROUNDS rounds


def _clip01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def _scores(w, act, x):
    """Class scores over the surviving synapses (act[c] = live input indices for class c)."""
    return [sum(w[c][i] * x[i] for i in act[c]) for c in range(len(w))]


def _predict(w, act, x):
    s = _scores(w, act, x)
    return max(range(len(s)), key=lambda c: s[c])


def _accuracy(w, act, data):
    return sum(1 for x, y in data if _predict(w, act, x) == y) / len(data)


def _train_epoch(w, act, data, lr):
    """One online softmax delta-rule pass; only surviving synapses are touched."""
    for x, y in data:
        probs = _softmax(_scores(w, act, x))
        for c in range(len(w)):
            g = lr * ((1.0 if c == y else 0.0) - probs[c])
            wc = w[c]
            for i in act[c]:
                wc[i] += g * x[i]


# ---- pruning ----------------------------------------------------------------
def _survivors(act):
    """Deterministically ordered (class, index) pairs for synapses still alive."""
    return [(c, i) for c in range(len(act)) for i in act[c]]


def _live_count(act):
    return sum(len(a) for a in act)


def _drop(act, victims):
    """Return new act lists with the victim (class, index) pairs removed."""
    dead = set(victims)
    return [[i for i in act[c] if (c, i) not in dead] for c in range(len(act))]


def _prune_magnitude(w, act, n_remove):
    """Cut the n_remove surviving synapses with the smallest |W| (least-used)."""
    ranked = sorted(_survivors(act), key=lambda ci: (abs(w[ci[0]][ci[1]]), ci[0], ci[1]))
    return _drop(act, ranked[:n_remove])


def _prune_random(act, n_remove, rng):
    """Fair control: cut n_remove surviving synapses chosen at random (same count)."""
    return _drop(act, rng.sample(_survivors(act), n_remove))


def _develop(train, test, strategy, rng=None):
    """Warm up dense, then prune PRUNE_FRAC of survivors each round and retrain.

    Both strategies follow the same schedule, so they reach identical survivor
    counts -- only which synapses get cut differs. Returns (test_acc, density).
    """
    w = [[0.0] * D for _ in range(K)]
    act = [list(range(D)) for _ in range(K)]
    for _ in range(WARMUP):
        _train_epoch(w, act, train, LR)
    for _ in range(ROUNDS):
        n_remove = int(PRUNE_FRAC * _live_count(act))
        if strategy == "magnitude":
            act = _prune_magnitude(w, act, n_remove)
        else:
            act = _prune_random(act, n_remove, rng)
        for _ in range(RETRAIN):
            _train_epoch(w, act, train, LR)
    return _accuracy(w, act, test), _live_count(act) / (K * D)


def _dense(train, test):
    """No-prune ceiling, trained for the SAME total epoch budget as the pruned runs."""
    w = [[0.0] * D for _ in range(K)]
    act = [list(range(D)) for _ in range(K)]
    for _ in range(WARMUP + ROUNDS * RETRAIN):
        _train_epoch(w, act, train, LR)
    return _accuracy(w, act, test)


# ---- synthetic task ---------------------------------------------------------
def _make_data(rng):
    """K class prototypes over D rate dims. Only N_SIGNAL dims are discriminative;
    the rest share one baseline across all classes (pure distractors). Samples =
    prototype + gaussian noise, clipped to [0, 1]. Learnable but non-trivial."""
    base = [rng.random() for _ in range(D)]
    signal = set(rng.sample(range(D), N_SIGNAL))
    protos = []
    for _ in range(K):
        p = base[:]
        for i in signal:
            p[i] = rng.random()  # class-specific value on discriminative dims
        protos.append(p)

    def sample(c):
        return [_clip01(protos[c][i] + rng.gauss(0.0, SIGMA)) for i in range(D)]

    train = [(sample(c), c) for c in range(K) for _ in range(PER_TRAIN)]
    test = [(sample(c), c) for c in range(K) for _ in range(PER_TEST)]
    rng.shuffle(train)
    return train, test


def run(seed=0):
    """Run dense / magnitude / random from the SAME data + init. Deterministic in seed."""
    train, test = _make_data(random.Random(seed))
    dense_acc = _dense(train, test)
    mag_acc, mag_den = _develop(train, test, "magnitude")
    rnd_acc, rnd_den = _develop(train, test, "random", rng=random.Random(seed + 1))
    return {
        "dense_acc": dense_acc,
        "mag_acc": mag_acc,
        "mag_den": mag_den,
        "rnd_acc": rnd_acc,
        "rnd_den": rnd_den,
    }


def main():
    seed = 0
    r = run(seed)
    margin = r["mag_acc"] - r["rnd_acc"]
    chance = 1.0 / K

    print("=" * 60)
    print("DEVELOPMENTAL SYNAPTIC PRUNING  (BrainCog-inspired)")
    print("=" * 60)
    print(
        f"task     : K={K} classes, dim={D} ({N_SIGNAL} signal + {D - N_SIGNAL} distractor), "
        f"noise sigma={SIGMA}"
    )
    print(
        f"schedule : warmup {WARMUP} ep, then {ROUNDS} rounds x prune {PRUNE_FRAC:.0%} of "
        f"survivors, +{RETRAIN} ep each"
    )
    print("-" * 60)
    print(f"dense (no-prune) accuracy : {r['dense_acc']:6.1%}   (ceiling; chance {chance:.0%})")
    print(f"magnitude-pruned accuracy : {r['mag_acc']:6.1%}   @ density {r['mag_den']:.1%}")
    print(
        f"random-pruned    accuracy : {r['rnd_acc']:6.1%}   @ density {r['rnd_den']:.1%}   "
        f"(fair control, matched sparsity)"
    )
    print(f"magnitude - random margin : {margin:+.1%}")
    print(
        f"achieved mask density     : {r['mag_den']:.1%}   (target {TARGET_DENSITY:.1%}, "
        f"pruned {1 - r['mag_den']:.0%})"
    )
    print("=" * 60)

    # ---- self-check: deterministic, honest asserts ----
    # 1. magnitude-pruned held-out accuracy is well above chance at the target sparsity.
    assert r["mag_acc"] >= chance + 0.30, (
        f"magnitude-pruned accuracy {r['mag_acc']:.2f} not well above chance {chance:.2f}"
    )
    # 2. THE mechanism anchor: keep-strong beats random at matched final sparsity.
    assert margin >= 0.05, (
        f"magnitude ({r['mag_acc']:.2f}) must beat random ({r['rnd_acc']:.2f}) by a real margin"
    )
    # 3. matched sparsity: both strategies reach the same density.
    assert r["mag_den"] == r["rnd_den"], "magnitude and random must reach identical density"
    # 4. pruning actually happened -- achieved density near the schedule target.
    assert abs(r["mag_den"] - TARGET_DENSITY) <= 0.05, (
        f"achieved density {r['mag_den']:.3f} off target {TARGET_DENSITY:.3f}"
    )
    # 5. determinism: same seed -> identical numbers.
    assert run(seed) == r, "run(seed) must be deterministic"

    print(
        f"self-check OK: magnitude beats random by {margin:+.1%} at {r['mag_den']:.0%} density "
        f"(dense ceiling {r['dense_acc']:.0%}); deterministic, matched sparsity"
    )


if __name__ == "__main__":
    main()

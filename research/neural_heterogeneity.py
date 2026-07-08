"""
Neural heterogeneity promotes robust learning (Perez-Nieves et al. 2021) -- zero-dep.

Perez-Nieves, Leung, Dragotti & Goodman, "Neural heterogeneity promotes robust learning", Nature
Communications 12:5791 (2021), DOI 10.1038/s41467-021-26022-3: spiking populations whose neurons carry
a DISTRIBUTION of membrane time constants learn temporal tasks better and more robustly than homogeneous
populations -- because a spread of tau TILES the temporal input (short-tau neurons resolve WHEN energy
arrives, long-tau neurons integrate a sustained level), giving a richer, more linearly-separable representation.

Every neuron model in this lake (lif_neuron, izhikevich_neuron, adex_neuron, ...) uses a SINGLE tau, and the
temporal-coding demos (temporal_coding_storage, legendre_memory, spiking_ssm) never vary it ACROSS a
population. This is the missing control: a HETEROGENEOUS-tau LIF population vs a HOMOGENEOUS one with the SAME
size N and the SAME MEAN tau, on a task that needs TWO timescales at once. Both populations use unit input and
the same readout; the neurons differ ONLY in tau, and the ONLY difference between the two populations is the
SPREAD of tau -- so any accuracy gap is caused by heterogeneity, nothing else.

Task (two orthogonal timescales, seeded). Each stimulus is a 1-D current with two INDEPENDENT bits:
  CONCENTRATION bit -- a fixed packet of 5 sub-threshold pulses is either CONCENTRATED into 5 consecutive
    early steps or DIFFUSED across the early window. The total energy is IDENTICAL either way, so it is legible
    only to a SHORT-tau neuron -- whose narrow memory lets the packed pulses SUMMATE to threshold while the
    spread ones decay between arrivals. A long-tau neuron integrates the same total energy either way.
  LEVEL bit -- a sustained sub-threshold offset, HIGH or LOW, over the whole window. A short-tau neuron leaks
    it away between steps and never fires on it; only a LONG-tau neuron ACCUMULATES it, so only long tau reads it.
The 4 classes are the (concentration x level) grid. The key: at a SINGLE tau the response to a concentrated
packet and the response to a higher sustained level are nearly the SAME number -- one filter cannot tell "energy
packed in time" from "energy held at a higher level" -- so the homogeneous population CONFUSES the
concentrated-low class with the diffuse-high class and stalls well below ceiling. A DIVERSITY of taus supplies
both a short-tau sub-population (reads concentration) and a long-tau sub-population (reads level), so the two
classes separate. The readout is a plain spike-COUNT linear classifier (no hand-coded timing features), so the
temporal discrimination has to live in the neuron dynamics == in tau.

Measured (seed 0, deterministic, held-out): HETEROGENEOUS 100.0% vs HOMOGENEOUS-at-mean-tau 70.2% (margin
+29.8%); and NO single tau matches the diverse population -- tau=min alone 56.0%, tau=max alone 81.0%, mean 70.2%.

  Run: python neural_heterogeneity.py       # (also run by run_selfchecks.py)
"""
import math
import random

from snn_data_lake.lif_neuron import LIFNeuron  # canonical zero-dep LIF (PYTHONPATH=src under run_selfchecks)

SEED = 0
N = 50                     # neurons per population (het and homo are the SAME size)
T = 60                     # timesteps per stimulus
PER = 70                   # stimuli per class (4 classes)
K = 4                      # (concentration bit) x (level bit)
A = 0.5                    # pulse height -- sub-threshold alone (needs summation to fire a short-tau neuron)
CONC_STEPS = (2, 3, 4, 5, 6)              # the 5-pulse packet, CONCENTRATED
DIFF_STEPS = (2, 8, 14, 20, 26)           # the SAME 5 pulses, DIFFUSED (identical total energy)
O_HI, O_LO = 0.14, 0.03    # sustained level bit: sub-threshold offset over the whole window
TAU_MIN, TAU_MAX = 2.0, 40.0   # heterogeneous tau range (log-uniform)
V_TH = 1.0
REFRACTORY = 2             # caps the max firing rate so counts don't run away
NOISE = 0.02               # per-step current jitter
EPOCHS = 60
LR = 0.1


def log_uniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def make_taus(rng):
    """Heterogeneous tau draws + the homogeneous control. homo_tau = the arithmetic MEAN of the het draws,
    so the control has EXACTLY the same mean tau and the same N -- only the spread differs."""
    het = [log_uniform(rng, TAU_MIN, TAU_MAX) for _ in range(N)]
    homo_tau = sum(het) / N
    return het, [homo_tau] * N, homo_tau


def prototype(conc, level):
    """Noise-free 1-D current = sustained offset (level bit) + a 5-pulse packet (concentrated or diffused)."""
    seq = [O_HI if level else O_LO] * T
    for t in (CONC_STEPS if conc else DIFF_STEPS):
        seq[t] += A
    return seq


def stimulus(rng, conc, level):
    return [c + rng.uniform(-NOISE, NOISE) for c in prototype(conc, level)]


def dataset(rng):
    data = [(stimulus(rng, conc, level), conc * 2 + level)
            for conc in (0, 1) for level in (0, 1) for _ in range(PER)]
    rng.shuffle(data)
    return data


def count(seq, tau):
    """Spike count of one LIF neuron over the stimulus -- the only feature (rate, no timing hand-coding)."""
    n = LIFNeuron(tau_m=tau, v_threshold=V_TH, refractory=REFRACTORY)
    return sum(n.step(c) for c in seq)


def features(data, taus):
    return [([count(seq, taus[j]) for j in range(N)], y) for seq, y in data]


def standardize(train, test):
    """Per-feature z-score using TRAIN statistics only (applied identically to both populations)."""
    mean = [sum(f[i] for f, _ in train) / len(train) for i in range(N)]
    std = [math.sqrt(sum((f[i] - mean[i]) ** 2 for f, _ in train) / len(train)) or 1.0 for i in range(N)]
    z = lambda rows: [([(f[i] - mean[i]) / std[i] for i in range(N)] + [1.0], y) for f, y in rows]  # +bias
    return z(train), z(test)


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def train_readout(train):
    """Delta-rule softmax linear readout over the spike-count features (the ONLY trained part)."""
    nf = len(train[0][0])
    W = [[0.0] * nf for _ in range(K)]
    for _ in range(EPOCHS):
        for f, y in train:
            p = softmax([sum(W[c][i] * f[i] for i in range(nf)) for c in range(K)])
            for c in range(K):
                err = (1.0 if c == y else 0.0) - p[c]
                wc = W[c]
                for i in range(nf):
                    wc[i] += LR * err * f[i]
    return W


def accuracy(W, test):
    nf = len(test[0][0])
    ok = sum(max(range(K), key=lambda c: sum(W[c][i] * f[i] for i in range(nf))) == y for f, y in test)
    return ok / len(test)


def selectivity_spread(taus):
    """Response diversity: spread (stdev) of a per-neuron selectivity index that is +1 for a neuron driven
    only by the concentration probe and -1 for one driven only by the level probe. Diverse tau -> some neurons
    are concentration-selective and others level-selective -> wide spread; a single tau -> everyone alike."""
    conc_probe = prototype(conc=1, level=0)   # fast concentrated energy, minimal sustained level
    level_probe = prototype(conc=0, level=1)  # sustained high level, no concentration
    idx = []
    for tau in taus:
        c = count(conc_probe, tau)
        l = count(level_probe, tau)
        idx.append((c - l) / (c + l + 1))
    mu = sum(idx) / N
    return math.sqrt(sum((v - mu) ** 2 for v in idx) / N)


def run_experiment(seed):
    rng = random.Random(seed)
    het_taus, homo_taus, homo_tau = make_taus(rng)
    data = dataset(rng)                      # SAME data for every population
    cut = int(0.7 * len(data))

    def evaluate(taus):
        train, test = features(data[:cut], taus), features(data[cut:], taus)
        train, test = standardize(train, test)
        return accuracy(train_readout(train), test)

    return {
        "het": evaluate(het_taus),
        "homo": evaluate(homo_taus),
        "short": evaluate([TAU_MIN] * N),        # single-tau reference: everyone fast
        "long": evaluate([TAU_MAX] * N),         # single-tau reference: everyone slow
        "mean_tau": homo_tau,
        "het_spread": selectivity_spread(het_taus),
        "homo_spread": selectivity_spread(homo_taus),
    }


def main():
    r = run_experiment(SEED)
    het, homo, short, long, mean_tau = r["het"], r["homo"], r["short"], r["long"], r["mean_tau"]
    chance = 1.0 / K

    print("=" * 64)
    print("NEURAL HETEROGENEITY -- diverse tau beats homogeneous on a temporal task")
    print("=" * 64)
    print(f"N={N} neurons/pop  T={T}  classes=(concentration x level)={K}  per-class={PER}  chance={chance:.0%}")
    print(f"heterogeneous tau : log-uniform [{TAU_MIN:.0f}, {TAU_MAX:.0f}]  (mean {mean_tau:.2f})")
    print(f"homogeneous tau   : every neuron = {mean_tau:.2f}  (SAME N, SAME unit input, SAME mean tau)")
    print()
    print("population              | mean tau | held-out accuracy")
    print("-" * 54)
    print(f"HETEROGENEOUS (spread)  |  {mean_tau:6.2f}  |   {het:6.1%}")
    print(f"homogeneous (mean tau)  |  {mean_tau:6.2f}  |   {homo:6.1%}")
    print(f"  ref: homogeneous fast |  {TAU_MIN:6.2f}  |   {short:6.1%}")
    print(f"  ref: homogeneous slow |  {TAU_MAX:6.2f}  |   {long:6.1%}")
    print("-" * 54)
    print(f"heterogeneity margin : +{het - homo:.1%}  (het - homo, same mean tau)")
    print(f"selectivity spread   : het {r['het_spread']:.2f}  vs  homo {r['homo_spread']:.2f}  (wider = more timescales tiled)")
    print("no single tau matches the diverse population -- diversity is what wins.")
    print("=" * 64)

    # ---- self-check (deterministic; the paper's fair same-mean-tau control) ----
    assert het > homo + 0.12, f"heterogeneity must clearly beat the fair homogeneous control: {het:.3f} vs {homo:.3f}"
    assert het > chance + 0.2, f"heterogeneous population must be well above chance: {het:.3f}"
    assert homo > chance, f"homogeneous control must be above chance (not crippled): {homo:.3f} <= {chance:.3f}"
    assert het > long + 0.1, f"the diverse pop must beat even the best single tau: het {het:.3f} vs long {long:.3f}"
    assert r["het_spread"] > r["homo_spread"], f"heterogeneous pop must tile more timescales: {r['het_spread']:.3f} <= {r['homo_spread']:.3f}"
    repeat = run_experiment(SEED)
    assert (repeat["het"], repeat["homo"]) == (het, homo), "same seed must give identical accuracies (determinism)"
    print(f"self-check OK: heterogeneous {het:.1%} > homogeneous {homo:.1%} (+{het - homo:.1%}) and > any single tau, "
          f"both > chance {chance:.0%}, wider timescale tiling, deterministic")


if __name__ == "__main__":
    main()

"""
Spiking state-space sequence model — a zero-dep Liquid State Machine (spiking reservoir).

The streaming/telemetry side of this lake is long event SEQUENCES; a state-space model is the right
primitive. Deep trained spiking SSMs are SOTA on long-range sequences (SpikingSSMs, AAAI 2025;
P-SpikeSSM, ICLR 2025; FLAMES, arXiv:2504.01257) but need BPTT + torch. The zero-dep counterpart that
keeps this in the CI-able stdlib core is the **Liquid State Machine** (Maass 2002): a FIXED random
recurrent spiking reservoir provides the state-space memory (h_t = leak·h_{t-1} + W_in·x_t + W_rec·s_{t-1},
spike + reset), and only a linear READOUT is trained — no backprop through time. So a sequence's temporal
structure is separated by the reservoir's fading memory and read out by a cheap delta-rule classifier.

This demonstrates the mechanism the repo cares about — temporal memory, event-driven SPARSITY, and
SynOps (piped through energy.py for a Joules estimate) — trained with pure stdlib. The trained deep
SpikingSSM/P-SpikeSSM on SHD/SSC is the accuracy upgrade (torch, own venv), referenced not built here.

  Run: python spiking_ssm.py       # (also run by run_selfchecks.py)
"""
import random

import stats_bounds as sb  # research/ on sys.path when run as a script

from snn_data_lake.energy import compare  # PYTHONPATH=src under run_selfchecks

random.seed(0)

N = 100           # reservoir neurons
FANIN = 8         # sparse recurrent connections per neuron
T = 16            # sequence length (time steps)
LEAK = 0.7        # membrane leak (fading memory = the state-space part)
THRESH = 1.0      # spike threshold
CLASSES = 3       # temporal pattern classes
PER = 24          # sequences per class
EPOCHS = 40       # delta-rule readout epochs
NOISE = 0.10      # per-step input jitter
SEEDS = range(5)


def build_reservoir(seed):
    """Fixed random sparse recurrent spiking reservoir + input weights (never trained)."""
    rng = random.Random(seed)
    w_in = [rng.uniform(-1, 1) for _ in range(N)]
    rec = [[(rng.randrange(N), rng.uniform(-0.5, 0.5)) for _ in range(FANIN)] for _ in range(N)]
    return w_in, rec


def run_reservoir(seq, w_in, rec):
    """Drive the reservoir with a scalar input sequence; return (mean spike rate per neuron, synops)."""
    h = [0.0] * N
    s = [0] * N
    counts = [0] * N
    synops = 0
    for x in seq:
        drive = [LEAK * h[i] + w_in[i] * x for i in range(N)]
        for i in range(N):
            acc = 0.0
            for j, w in rec[i]:
                if s[j]:
                    acc += w
                    synops += 1            # a synaptic op only on a presynaptic spike (event-driven)
            drive[i] += acc
        s = [1 if drive[i] >= THRESH else 0 for i in range(N)]
        h = [0.0 if s[i] else drive[i] for i in range(N)]   # reset on spike
        for i in range(N):
            counts[i] += s[i]
    return [c / T for c in counts], synops


def make_sequences(rng):
    """CLASSES temporal signatures (rising / falling / mid-pulse) + noise — separable only by the
    time course, so the reservoir's memory matters. Returns [(features, synops, label)]."""
    protos = [
        [t / (T - 1) for t in range(T)],                    # 0: rising ramp
        [1.0 - t / (T - 1) for t in range(T)],              # 1: falling ramp
        [1.0 if 6 <= t <= 9 else 0.0 for t in range(T)],    # 2: mid pulse
    ]
    return protos


def dataset(seed):
    rng = random.Random(seed)
    w_in, rec = build_reservoir(seed)
    protos = make_sequences(rng)
    data, total_syn = [], 0
    for label in range(CLASSES):
        for _ in range(PER):
            seq = [min(1.0, max(0.0, protos[label][t] + rng.uniform(-NOISE, NOISE))) for t in range(T)]
            feat, syn = run_reservoir(seq, w_in, rec)
            data.append((feat + [1.0], label))          # +bias term for the readout
            total_syn += syn
    rng.shuffle(data)
    return data, total_syn / len(data)


def train_readout(train, rng):
    """Delta-rule linear readout over reservoir features (the only trained part — no BPTT)."""
    nf = len(train[0][0])
    w = [[0.0] * nf for _ in range(CLASSES)]
    for _ in range(EPOCHS):
        for feat, label in train:
            scores = [sum(w[c][i] * feat[i] for i in range(nf)) for c in range(CLASSES)]
            for c in range(CLASSES):
                err = (1.0 if c == label else 0.0) - scores[c]
                for i in range(nf):
                    w[c][i] += 0.05 * err * feat[i]
    return w


def accuracy(w, test):
    nf = len(test[0][0])
    ok = 0
    for feat, label in test:
        scores = [sum(w[c][i] * feat[i] for i in range(nf)) for c in range(CLASSES)]
        if max(range(CLASSES), key=lambda c: scores[c]) == label:
            ok += 1
    return ok / len(test)


def run_once(seed):
    data, mean_syn = dataset(seed)
    cut = int(0.7 * len(data))
    train, test = data[:cut], data[cut:]
    w = train_readout(train, random.Random(seed))
    return accuracy(w, test), mean_syn


def main():
    accs, syns = [], []
    for s in SEEDS:
        a, syn = run_once(s)
        accs.append(a)
        syns.append(syn)
    acc_m, acc_s = sb.mean_std(accs)
    syn_mean = sum(syns) / len(syns)
    # sparsity: fraction of reservoir neuron-steps that spiked = synops / (fanin-weighted step budget)
    # report event-driven SynOps -> energy via the repo's helper (dense readout MACs as the baseline)
    dense_macs = N * CLASSES * T                      # a dense per-step readout over the full state
    e = compare(int(syn_mean), dense_macs)

    print("=" * 62)
    print("SPIKING STATE-SPACE SEQUENCE MODEL — Liquid State Machine (zero-dep)")
    print("=" * 62)
    print(f"reservoir N={N} (fan-in {FANIN}, leak {LEAK})  T={T}  classes={CLASSES}  seeds={len(accs)}")
    print(f"classification accuracy : {acc_m:.1%} ± {acc_s:.1%}  (chance {1 / CLASSES:.0%})")
    print(f"reservoir SynOps/seq    : {syn_mean:,.0f}  (event-driven; only spiking synapses count)")
    print(f"energy (vs dense readout): {e['snn_nj']:.1f} nJ vs {e['ann_nj']:.1f} nJ "
          f"-> {e['energy_ratio']:.1f}x (see energy.py)")
    print("upgrade path: trained deep SpikingSSM / P-SpikeSSM on SHD/SSC (torch, own venv) for SOTA"
          " long-range accuracy; this fixed-reservoir LSM is the zero-dep, CI-able mechanism.")
    print("=" * 62)

    # ---- self-check (ponytail: one runnable assert-based check; deterministic) ----
    assert acc_m > 1.0 / CLASSES + 0.2, f"reservoir readout no better than chance: {acc_m:.2f}"
    assert acc_m >= 0.8, f"temporal classes should be well separated by the reservoir: {acc_m:.2f}"
    assert syn_mean > 0, "reservoir produced no synaptic operations"
    assert e["snn_nj"] < e["ann_nj"], "event-driven SynOps should cost less than the dense readout"
    print(f"self-check OK: LSM reservoir classifies temporal sequences at {acc_m:.0%} "
          "(zero-dep, trained readout only), event-driven SynOps -> energy via energy.py")


if __name__ == "__main__":
    main()

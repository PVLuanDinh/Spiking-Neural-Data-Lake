"""
IS-SNN forward pass — intrinsically-stable multi-layer LIF forward dynamics (pure Python, no tensors).

A custom spiking forward pass over a stack of LIF layers (reused `lif_neuron.LIFNeuron`) with a hard-coded
INTRINSICALLY-STABLE init: sparse excitatory weights scaled to the LIF's self-sustaining rate fixed point
(mean synaptic current ~ v_threshold at the operating rate) so the per-layer firing rate is DEPTH-INVARIANT
-- it neither dies nor saturates as signal propagates, WITHOUT any dynamic BatchNorm. A deliberately-weak
init is the control that suffers catastrophic firing-rate decay. The connectivity is SPARSE (FlyHash-style,
like flyhash_index / sdm in this repo): a dense positive projection would low-pass every neuron toward the
same mean drive and collapse the code, so each unit samples a random subset of inputs -> the first hidden
layer is a discriminative random projection whose spike-rate vector is a class-separable embedding a
Van Rossum / Victor-Purpura index can store. Pure standard library, deterministic. The zero-dep counterpart
to the torch IS-SNN in research/neuromorphic_torch.

  python -m snn_data_lake.cortex.is_snn_forward        # (also run by run_selfchecks.py)
"""
import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron

STABLE_GAIN = 2.5   # excitatory scale = STABLE_GAIN*v_threshold/fan_in_sparse -> rate flat across depth
WEAK_GAIN = 1.0     # too weak -> firing rate decays layer to layer (the bad-init control)
SPARSITY = 0.4      # fraction of inputs each unit samples (sparse random projection keeps the code separable)


def _tau_schedule(n_layers, tau_min, tau_max, mode="geometric"):
    """MONOTONICALLY INCREASING per-layer membrane time constants (short/fast early -> long/slow deep).

    Geometric (log-linear) interpolation so tau_l = exp( lerp(log(tau_min), log(tau_max), l/(n-1)) ):
    early layers get a SHORT tau (track fast input structure), deep layers get a LONG tau (integrate over
    long horizons). This temporal hierarchy is the heterogeneous/multi-timescale prior of the SNN
    timescale-hierarchy benchmark (arXiv:2407.18838) -- a stack spanning many timescales solves a task with
    fast AND slow structure that no single (flat) timescale resolves. stdlib math only, deterministic."""
    if mode != "geometric":
        raise ValueError(f"unknown tau schedule mode: {mode!r} (only 'geometric')")
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if n_layers == 1:
        return [float(tau_min)]
    lo, hi = math.log(tau_min), math.log(tau_max)
    return [math.exp(lo + (hi - lo) * (l / (n_layers - 1))) for l in range(n_layers)]


class ISSNNForward:
    """Stacked LIF layers scaled to the firing-rate fixed point; forward() -> (embedding, per-layer rates)."""

    def __init__(self, sizes, tau_m=8.0, v_threshold=1.0, seed=0, stable=True):
        self.sizes = sizes
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        n_layers = len(sizes) - 1
        if isinstance(tau_m, (list, tuple)):                       # per-layer temporal hierarchy (short->long)
            if len(tau_m) != n_layers:
                raise ValueError(f"tau_m list must have one entry per layer ({n_layers}), got {len(tau_m)}")
            self._layer_tau = [float(t) for t in tau_m]
        else:
            self._layer_tau = [tau_m] * n_layers                   # scalar -> flat schedule (byte-identical back-compat)
        rng = random.Random(seed)
        gain = STABLE_GAIN if stable else WEAK_GAIN
        self.w = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            k = max(1, int(SPARSITY * fan_in))
            scale = gain * v_threshold / k                     # scale sparse fan-in to the LIF rate fixed point
            layer = []
            for _ in range(fan_out):
                row = [0.0] * fan_in
                for i in rng.sample(range(fan_in), k):         # sparse random projection (FlyHash-style)
                    row[i] = rng.uniform(0.5, 1.0) * scale
                layer.append(row)
            self.w.append(layer)

    def _run(self, spike_input):
        """Simulate the LIF stack; each layer li uses its OWN membrane tau (self._layer_tau[li]).

        Returns (per-layer spike counts, t_steps). Shared by forward() and layer_embeddings()."""
        t_steps = len(spike_input)
        layers = [[LIFNeuron(self._layer_tau[l], self.v_threshold) for _ in range(b)]
                  for l, b in enumerate(self.sizes[1:])]
        counts = [[0] * b for b in self.sizes[1:]]
        for t in range(t_steps):
            x = spike_input[t]
            for li, (wl, neurons) in enumerate(zip(self.w, layers)):
                y = []
                for j, nrn in enumerate(neurons):
                    current = sum(wl[j][i] * x[i] for i in range(len(x)))
                    s = nrn.step(current)
                    y.append(s)
                    counts[li][j] += s
                x = y
        return counts, t_steps

    def forward(self, spike_input):
        """spike_input: (T, n_in) list of 0/1 vectors. Returns (embedding rates[hidden1], per-layer mean rate)."""
        counts, t_steps = self._run(spike_input)
        layer_rates = [sum(c) / (len(c) * t_steps) for c in counts]
        embedding = [c / t_steps for c in counts[0]]           # first hidden = discriminative sparse projection
        return embedding, layer_rates

    def layer_embeddings(self, spike_input):
        """Per-neuron firing-rate vector for EVERY layer (one list per layer); the DEEP (terminal) layer is
        the multi-timescale readout used by the temporal-hierarchy self-check in main()."""
        counts, t_steps = self._run(spike_input)
        return [[c / t_steps for c in lc] for lc in counts]


def _spike_batch(rng, rates, t_steps):
    return [[1 if rng.random() < r else 0 for r in rates] for _ in range(t_steps)]


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


_MT_FAST = (0, 1, 2)   # input channels carrying the FAST (short-timescale) stream
_MT_SLOW = (3, 4, 5)   # input channels carrying the SLOW (long-timescale) stream


def _mt_sample(rng, n_in, t_steps, fast_bit, slow_bit, n_pair=4, far_gap=16, block=18):
    """One MULTI-TIME-SCALE input (arXiv:2407.18838 idea): a FAST stream + a SLOW stream, each MEAN-MATCHED
    so only TEMPORAL STRUCTURE (not spike count) carries the label -- a flat rate code cannot separate them.

    fast_bit=1 -> the spike PAIRS are adjacent (gap 1): a fast coincidence that a SHORT tau resolves and a
    LONG tau integrates away.   fast_bit=0 -> the same pairs sit `far_gap` apart.
    slow_bit=1 -> a single contiguous BLOCK: long temporal summation only a LONG tau accumulates to threshold.
    slow_bit=0 -> the same number of spikes scattered.   Same per-channel spike count in every case."""
    v = [[0] * n_in for _ in range(t_steps)]
    for c in _MT_FAST:
        used, placed, gap = set(), 0, (1 if fast_bit else far_gap)
        while placed < n_pair:
            t0 = rng.randrange(0, t_steps - gap)
            if t0 in used or (t0 + gap) in used:
                continue
            v[t0][c] = 1
            v[t0 + gap][c] = 1
            used.add(t0)
            used.add(t0 + gap)
            placed += 1
    for c in _MT_SLOW:
        if slow_bit:
            t0 = rng.randrange(0, t_steps - block + 1)
            for t in range(t0, t0 + block):
                v[t][c] = 1
        else:
            for t in rng.sample(range(t_steps), block):
                v[t][c] = 1
    return v


def _mt_accuracy(net, n_in, t_steps, seed, reps=8, trials=40):
    """Nearest-centroid accuracy on the 4-way (fast_bit x slow_bit) task, reading the DEEP (terminal) layer.

    Deterministic (seeded). Chance = 25%. A flat-tau stack resolves at most one timescale -> ~one axis; a
    temporal hierarchy resolves both fast and slow structure at once -> higher accuracy at EQUAL params."""
    rng = random.Random(seed)
    classes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    centroids = []
    for fast_bit, slow_bit in classes:
        acc = None
        for _ in range(reps):
            deep = net.layer_embeddings(_mt_sample(rng, n_in, t_steps, fast_bit, slow_bit))[-1]
            acc = deep if acc is None else [a + b for a, b in zip(acc, deep)]
        centroids.append([a / reps for a in acc])
    hits = 0
    for i in range(trials):
        fast_bit, slow_bit = classes[i % 4]
        deep = net.layer_embeddings(_mt_sample(rng, n_in, t_steps, fast_bit, slow_bit))[-1]
        pred = min(range(4), key=lambda c: _dist2(deep, centroids[c]))
        hits += pred == (i % 4)
    return hits / trials


def main():
    rng = random.Random(1)
    n_in, K, T = 16, 3, 30
    class_rates = [[rng.random() * 0.5 + 0.1 for _ in range(n_in)] for _ in range(K)]

    net = ISSNNForward([n_in, 32, 32, 16], seed=0, stable=True)
    bad = ISSNNForward([n_in, 32, 32, 16], seed=0, stable=False)

    # per-class centroid embeddings (train) then classify fresh jittered samples (nearest centroid)
    centroids = [net.forward(_spike_batch(rng, class_rates[k], T))[0] for k in range(K)]
    hits, trials = 0, 60
    stable_rates = None
    for i in range(trials):
        k = i % K
        emb, stable_rates = net.forward(_spike_batch(rng, class_rates[k], T))
        pred = min(range(K), key=lambda c: _dist2(emb, centroids[c]))
        hits += pred == k
    acc = hits / trials
    _, bad_rates = bad.forward(_spike_batch(rng, class_rates[0], T))

    print("=" * 60)
    print("IS-SNN FORWARD — intrinsically-stable multi-layer LIF forward pass")
    print("=" * 60)
    print(f"net {net.sizes}; sparse init gain {STABLE_GAIN}/fan_in @ p={SPARSITY} (rate fixed point), no BatchNorm")
    print(f"stable layer rates : {[round(r, 3) for r in stable_rates]} (alive + bounded across depth)")
    print(f"bad-init rates     : {[round(r, 3) for r in bad_rates]} (catastrophic decay)")
    print(f"embedding classification: {acc:.0%} nearest-centroid over {K} classes")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert min(stable_rates) > 0.2 and max(stable_rates) < 0.75, \
        f"stable init must keep the rate ALIVE and BOUNDED across depth (no death, no saturation): {stable_rates}"
    assert stable_rates[-1] > 0.7 * stable_rates[0], f"stable rate must not decay across depth: {stable_rates}"
    assert bad_rates[-1] < 0.5 * stable_rates[-1], f"weak init must decay vs stable ({bad_rates} vs {stable_rates})"
    assert acc >= 0.8, f"the sparse forward embedding must separate classes ({acc:.2f})"
    emb1 = net.forward(_spike_batch(random.Random(9), class_rates[0], T))[0]
    emb2 = net.forward(_spike_batch(random.Random(9), class_rates[0], T))[0]
    assert emb1 == emb2, "same seed+input -> identical forward pass (deterministic)"

    # ---- back-compat: a scalar tau_m is BYTE-IDENTICAL to the flat per-layer schedule [tau]*n ----
    scal = ISSNNForward([n_in, 32, 32, 16], tau_m=8.0, seed=0, stable=True)
    flat_list = ISSNNForward([n_in, 32, 32, 16], tau_m=[8.0, 8.0, 8.0], seed=0, stable=True)
    probe = _spike_batch(random.Random(5), class_rates[0], T)
    assert scal._layer_tau == [8.0, 8.0, 8.0], "scalar tau_m must expand to a flat per-layer schedule"
    assert scal.forward(probe) == flat_list.forward(probe), \
        "scalar tau_m must be byte-identical to the flat [tau]*n schedule (determinism preserved)"

    # ---- multi-time-scale task: a hierarchical per-layer tau BEATS a flat tau at EQUAL parameter count ----
    # ---- (same seed -> identical weights; ONLY the tau schedule differs).                                ----
    mt_sizes, mt_T, mt_seed = [12, 24, 24, 24], 48, 1
    mt_layers = len(mt_sizes) - 1
    schedule = _tau_schedule(mt_layers, 2.0, 45.0)                       # short/fast early -> long/slow deep
    geo_tau = math.exp(sum(math.log(t) for t in schedule) / mt_layers)  # flat control at the SAME mean timescale
    hier_net = ISSNNForward(mt_sizes, tau_m=schedule, seed=0, stable=True)
    flat_net = ISSNNForward(mt_sizes, tau_m=geo_tau, seed=0, stable=True)
    hier_acc = _mt_accuracy(hier_net, mt_sizes[0], mt_T, mt_seed)
    flat_acc = _mt_accuracy(flat_net, mt_sizes[0], mt_T, mt_seed)
    short_acc = _mt_accuracy(ISSNNForward(mt_sizes, tau_m=schedule[0], seed=0, stable=True), mt_sizes[0], mt_T, mt_seed)
    long_acc = _mt_accuracy(ISSNNForward(mt_sizes, tau_m=schedule[-1], seed=0, stable=True), mt_sizes[0], mt_T, mt_seed)
    best_flat = max(flat_acc, short_acc, long_acc)
    print(f"multi-timescale (fast+slow, 4-way, deep readout): hier tau {[round(t, 1) for t in schedule]} -> "
          f"{hier_acc:.0%}  vs flat tau {geo_tau:.1f}/{schedule[0]:.0f}/{schedule[-1]:.0f} -> "
          f"{flat_acc:.0%}/{short_acc:.0%}/{long_acc:.0%}  (equal params)")
    assert hier_net.w == flat_net.w, "hier vs flat must share weights -> EQUAL parameter count (only tau differs)"
    assert hier_acc >= 0.5, f"hierarchical-tau net must SOLVE the multi-timescale task ({hier_acc:.2f} of 4-way)"
    assert hier_acc >= best_flat + 0.1, \
        f"temporal hierarchy must BEAT every flat tau at equal params (hier {hier_acc:.2f} vs best flat {best_flat:.2f})"

    print(f"self-check OK: stable rates {[round(r,2) for r in stable_rates]} (alive+bounded) vs bad "
          f"{[round(r,2) for r in bad_rates]} (decays); embedding acc {acc:.0%}; "
          f"multi-timescale hier {hier_acc:.0%} > flat {best_flat:.0%} (equal params); deterministic")


if __name__ == "__main__":
    main()

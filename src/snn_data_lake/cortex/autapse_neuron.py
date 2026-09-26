"""
Time-delayed autapse LIF (TDA-LIF) -- a single neuron with the capacity of a reservoir.

Reproduces the core of Cai et al., "Reconstructing Spiking Neural Networks Using a Single
Neuron with Autapses" (arXiv:2603.24692, TDA-SNN). An autapse is a synapse a neuron makes
onto ITSELF; a *time-delayed* autapse feeds a past spike back into the dendrites d nodes
later. Unrolling one such neuron over time turns it into a graph of temporal "nodes" wired
by its own delayed self-connections -- and that graph carries the computational capacity of
a reservoir / MLP / conv, storing and mixing far more than a memoryless LIF. One neuron,
many delayed self-loops, high per-neuron information capacity.

Base membrane dynamics are exactly the repo's LIFNeuron (v_rest=v_reset=0), which is the
paper's iterative form (eq. 1-2):

    v^t = tau * v^{t-1} * (1 - s^{t-1}) + I^t + sum_{d in D} w_a^d * s^{t-d}

The (1 - s^{t-1}) factor is a multiplicative reset: the moment the neuron fires, the membrane
is wiped to 0. So a plain LIF keeps NO leaky trace of a past spike -- once it fires it forgets,
and the ONLY channel that can carry information across a silent gap is a delayed autapse. That is
what makes the self-check un-riggable. The task is a DELAYED-MATCH (delayed-recall): a random spike
stimulus stream is followed by several SILENT nodes (no external input), and a single linear head
must recall the last stimulus bit from the neuron's state during the silence. A memoryless LIF's
silent-window state is decoupled from the stimulus (fire-and-reset erases it; no input drives it),
so it can only guess the base rate -- chance. A bank of delayed autapses (delays 1..D) acts as a
temporal delay line: it echoes the recent stimulus spikes into the silent window, where the linear
readout recovers the bit. Zeroing the autaptic weights (ablation) collapses the SAME neuron and
SAME readout back to chance -- the delay memory is load-bearing. This is the paper's central claim:
"delayed self-memory = per-neuron capacity" -- one neuron holds temporal state a memoryless unit
cannot.

Reuses cortex.lif_neuron (base LIF) and substrate._mathutils (sigmoid/dot). A drop-in denser
spiking unit for the neurogenesis / reasoning learners: more temporal capacity per neuron via
delayed self-feedback rather than more neurons. HONEST scope: this reproduces the temporal-MEMORY /
reservoir-readout regime (a linear head over the unfolded node states). It does NOT reproduce the
paper's nonlinear XOR/MLP-conv equivalences or its prototype-learning training -- a single-neuron
random reservoir with a linear readout cannot synthesize a nonlinear feature like a*b, which needs
the paper's trained (pruned) autaptic weights. Memory span is finite: recall attenuates for bits
deeper than the delay range / weight scale sustain (the paper's space-time tradeoff).

  python -m snn_data_lake.cortex.autapse_neuron        # (deterministic self-check)
"""

import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron
from snn_data_lake.substrate._mathutils import dot, sigmoid

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "_LinearReadout.train": (
        "LOCAL",
        "single-layer logistic delta rule: g = sigmoid(w.x + b) - y, then w[k] -= lr * g * x[k]. No "
        "hidden layer, so no backward pass exists to have a gradient propagated through; this module "
        "imports no torch at all. It shares the NAME _LinearReadout with the autograd readouts in "
        "hippocampus/developmental_loop_brain and reasoning/_loop_common and nothing else -- a name "
        "collision that was briefly mistaken for a third copy of them (spec FR-002h records the "
        "correction). Unlike those two this module IS reachable from a live path, which is exactly "
        "why classifying it correctly matters",
    ),
}


class AutapseNeuron:
    """A single LIF neuron with a bank of time-delayed self-connections (autapses).

    delays  : iterable of positive ints -- the autaptic delay set D (a spike feeds back after
              d nodes). weights[k] is the autaptic weight w_a for delays[k].
    weights : per-delay autaptic weights aligned to `delays`; if None, drawn from
              random.Random(seed) in [-w_scale, +w_scale]. Pass all-zeros to ablate (memoryless).

    step(current) advances one internal node: it adds the delayed autaptic current
    sum_d w_a^d * s^{t-d} to the external input, runs the base LIF, records the emitted spike,
    and returns it. Deterministic.
    """

    def __init__(self, delays, weights=None, w_scale=0.9, tau_m=5.0, v_threshold=1.0, seed=0):
        self.delays = [int(d) for d in delays]
        if any(d < 1 for d in self.delays):
            raise ValueError("autaptic delays must be >= 1 node")
        if weights is None:
            rng = random.Random(seed)
            self.weights = [rng.uniform(-w_scale, w_scale) for _ in self.delays]
        else:
            self.weights = [float(w) for w in weights]
            if len(self.weights) != len(self.delays):
                raise ValueError("weights must align 1:1 with delays")
        self.lif = LIFNeuron(tau_m=tau_m, v_threshold=v_threshold, v_reset=0.0, v_rest=0.0)
        self.reset_state()

    def reset_state(self):
        self.lif.reset_state()
        self._spikes = []  # emitted spike history s^0..s^{t-1}
        self.v = 0.0  # last membrane potential (the reservoir readout state)

    def _autaptic_current(self):
        """sum_{d in D, d <= t} w_a^d * s^{t-d} -- delayed feedback of the neuron's own spikes."""
        n = len(self._spikes)
        total = 0.0
        for w, d in zip(self.weights, self.delays):
            if w and n >= d:
                total += w * self._spikes[-d]  # s^{t-d}
        return total

    def step(self, current):
        """Advance one node with external input `current`; return the 0/1 spike emitted."""
        s = self.lif.step(current + self._autaptic_current())
        self.v = self.lif.v
        self._spikes.append(s)
        return s

    def run(self, currents):
        """Run an input stream; return the emitted spike train (list of 0/1)."""
        self.reset_state()
        return [self.step(c) for c in currents]

    def run_states(self, currents):
        """Run an input stream; return the per-node membrane trace (the unfolded reservoir state)."""
        self.reset_state()
        trace = []
        for c in currents:
            self.step(c)
            trace.append(self.v)
        return trace


class _LinearReadout:
    """Minimal logistic readout (the RC readout head), trained by gradient descent."""

    def __init__(self, n):
        self.w = [0.0] * n
        self.b = 0.0

    def train(self, xs, ys, epochs=160, lr=0.30):
        for _ in range(epochs):
            for x, y in zip(xs, ys):
                g = sigmoid(dot(self.w, x) + self.b) - y
                for k in range(len(self.w)):
                    self.w[k] -= lr * g * x[k]
                self.b -= lr * g

    def predict(self, x):
        return 1 if sigmoid(dot(self.w, x) + self.b) >= 0.5 else 0


def _standardize(train, rows):
    """Per-feature z-score using TRAIN statistics only (deterministic); apply to every row set."""
    n = len(train[0])
    mean = [sum(r[k] for r in train) / len(train) for k in range(n)]
    var = [sum((r[k] - mean[k]) ** 2 for r in train) / len(train) for k in range(n)]
    std = [(v**0.5) or 1.0 for v in var]

    def z(rowset):
        return [[(r[k] - mean[k]) / std[k] for k in range(n)] for r in rowset]

    return [z(rs) for rs in rows]


# --- the delayed-match (delayed-recall) task -------------------------------------------------
M_STIM = 12  # stimulus nodes carrying a random spike stream
K_SILENT = 8  # silent readout nodes (no external input) after the stimulus -- the gap
DELAYS = list(range(1, 7))  # autaptic delay bank D = {1..6}
I_IN = 1.6  # supra-threshold current for a 1-bit (fires + resets the membrane at once)
N_TRAIN, N_TEST = 400, 200


def _delayed_match_stream(rng):
    """One delayed-match trial: M random stimulus bits, then K silent nodes; label = last bit.

    Every 1-bit is injected supra-threshold, so it fires and resets the membrane -- erasing any
    leaky trace. The silent tail has zero external input, so the last stimulus bit can reach the
    readout window ONLY through a delayed autapse. Returns (current_stream, label).
    """
    bits = [1 if rng.random() < 0.5 else 0 for _ in range(M_STIM)]
    stream = [I_IN if b else 0.0 for b in bits] + [0.0] * K_SILENT
    return stream, bits[M_STIM - 1]  # recall the most recent stimulus after the gap


def _run_experiment(weights, seed):
    """Run the delayed-match dataset through one neuron; return (train_acc, test_acc).

    weights = the autaptic delay-line weights (a zero list -> the memoryless / ablated control).
    Same delays, same readout pipeline, same data either way -- only the autapses differ.
    """
    data_rng = random.Random(seed)
    neuron = AutapseNeuron(DELAYS, weights=weights)

    def features(stream):
        return neuron.run_states(stream)[M_STIM:]  # the silent-tail state (reservoir readout)

    xs, ys = [], []
    for _ in range(N_TRAIN + N_TEST):
        stream, label = _delayed_match_stream(data_rng)
        xs.append(features(stream))
        ys.append(label)

    xtr_raw, xte_raw = xs[:N_TRAIN], xs[N_TRAIN:]
    ytr, yte = ys[:N_TRAIN], ys[N_TRAIN:]
    xtr, xte = _standardize(xtr_raw, [xtr_raw, xte_raw])

    readout = _LinearReadout(len(xtr[0]))
    readout.train(xtr, ytr, epochs=250)
    tr_acc = sum(readout.predict(x) == y for x, y in zip(xtr, ytr)) / len(ytr)
    te_acc = sum(readout.predict(x) == y for x, y in zip(xte, yte)) / len(yte)
    return tr_acc, te_acc


def main():
    # A single autaptic delay-line (small random self-feedback weights) vs the SAME neuron ablated.
    rng = random.Random(7)
    res_w = [rng.uniform(-0.25, 0.25) for _ in DELAYS]
    zero_w = [0.0] * len(DELAYS)

    auta_tr, auta_te = _run_experiment(res_w, seed=42)
    abl_tr, abl_te = _run_experiment(zero_w, seed=42)
    base = 0.5  # balanced random bits -> chance = base rate

    print("=" * 72)
    print("TDA-LIF AUTAPSE NEURON -- one neuron, delayed self-loops = temporal memory capacity")
    print("=" * 72)
    print(f"task : delayed-match over a spike stream -- {M_STIM} random stimulus bits, then")
    print(f"       {K_SILENT} SILENT nodes; a linear head recalls the LAST stimulus bit from the")
    print("       neuron's silent-window state. Firing resets the membrane, so only a delayed")
    print("       autapse can carry the stimulus across the silence.")
    print(
        f"  autapse neuron (delays {DELAYS[0]}..{DELAYS[-1]}) : train {auta_tr:.0%}   test {auta_te:.0%}"
    )
    print(
        f"  memoryless LIF control (ablated)   : train {abl_tr:.0%}   test {abl_te:.0%}   (chance {base:.0%})"
    )
    print("=" * 72)

    # ---- self-check: deterministic asserts ----
    # 1. headline: the autapse neuron solves the temporal-memory task ...
    assert auta_te >= 0.9, f"autapse neuron must recall the delayed stimulus ({auta_te:.2f})"
    # 2. ... which the memoryless control provably CANNOT (fire+reset erases it -> chance).
    assert abl_te <= base + 0.08, (
        f"memoryless LIF control must sit at chance on delayed-match ({abl_te:.2f})"
    )
    # 3. the delay memory is load-bearing: same neuron + readout, autapses ablated -> collapse.
    assert auta_te - abl_te >= 0.3, (
        f"delayed autapses must be load-bearing ({auta_te:.2f} vs ablated {abl_te:.2f})"
    )
    # 4. mechanism: an autapse genuinely transports a past spike a memoryless LIF forgets.
    fired = AutapseNeuron([3], weights=[1.5]).run([1.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert fired[0] == 1 and fired[3] == 1, (
        f"a delay-3 autapse must re-fire the neuron 3 nodes after a lone input spike ({fired})"
    )
    forgot = AutapseNeuron([3], weights=[0.0]).run([1.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert sum(forgot) == 1, f"the memoryless control must fire once and forget ({forgot})"
    # 5. deterministic.
    assert _run_experiment(res_w, seed=42) == (auta_tr, auta_te), "same seed -> identical result"

    print(
        f"self-check OK: autapse neuron {auta_te:.0%} vs memoryless {abl_te:.0%} on delayed-match "
        f"(chance {base:.0%}); delayed self-feedback is load-bearing (ablate -> collapse); deterministic"
    )


if __name__ == "__main__":
    main()

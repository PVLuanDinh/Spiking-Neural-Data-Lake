"""
Polychronization — memory stored in spike TIMING, not weights or rates (pure Python, no tensors).

Izhikevich (2006) showed that a network with heterogeneous axonal CONDUCTION DELAYS supports Polychronous
Neuronal Groups (PNGs): sets of neurons that fire at distinct ABSOLUTE times but whose delayed spikes
CONVERGE simultaneously on a shared postsynaptic target, igniting it. Because the same neurons can converge
in many different delay-matched temporal orders, the number of PNGs a network holds far exceeds its neuron
count -- a combinatorial memory whose key is the TIMING pattern, not which neurons fire (rate-blind). This is
the network-scale form of this repo's thesis ("reason in spike timing"): the store recalls a group only when
the anchor spikes arrive in the exact relative order the delays complement. Reuses the stdlib `LIFNeuron` as a
coincidence detector (short tau_m -> spread-out inputs leak away, only coincident inputs cross threshold).
Pure standard library, deterministic. On-thesis with delay_learning / izhikevich_neuron / spike_preprocessing.

  python -m research.polychronization        # (also run by run_selfchecks.py, root on PYTHONPATH)
"""
import sys
from collections import defaultdict
from pathlib import Path

pass  # migrated: sys.path bootstrap removed (now an importable package)
from snn_data_lake.lif_neuron import LIFNeuron   # noqa: E402


class DelayNetwork:
    """Spiking net with per-synapse conduction delays; LIF neurons act as coincidence detectors."""

    def __init__(self, n, tau_m=2.0, v_threshold=1.0, refractory=3):
        self.neurons = [LIFNeuron(tau_m, v_threshold, refractory=refractory) for _ in range(n)]
        self.out = [[] for _ in range(n)]                 # out[src] = [(dst, weight, delay), ...]

    def connect(self, src, dst, weight, delay):
        self.out[src].append((dst, weight, delay))

    def run(self, forced, steps):
        """forced: {neuron: set(times)} anchor spikes injected externally. Returns {neuron: [spike times]}."""
        for nrn in self.neurons:
            nrn.reset_state()
        inbox = defaultdict(lambda: defaultdict(float))   # inbox[t][dst] += weight (delayed arrivals)
        spikes = defaultdict(list)
        for t in range(steps):
            for dst in range(len(self.neurons)):
                if t in forced.get(dst, ()):              # anchor: emit a spike regardless of current
                    self.neurons[dst].reset_state()
                    fired = 1
                else:
                    fired = self.neurons[dst].step(inbox[t].get(dst, 0.0))
                if fired:
                    spikes[dst].append(t)
                    for d2, w, dl in self.out[dst]:       # propagate along delayed axons (dl >= 1)
                        inbox[t + dl][d2] += w
        return {k: v for k, v in spikes.items()}


# ---- a network holding two PNGs over ONE shared anchor pool, keyed only by spike TIMING ----------------
A0, A1, A2, A3, T1, T2, D1, D2 = range(8)          # anchors a0..a3, convergence targets T1/T2, downstream D1/D2


def _build():
    net = DelayNetwork(8)
    # Group 1: a0,a1,a2 converge on T1 with delays 3,2,1  (key = anchors fire at offsets 0,1,2 -> all land at t+3)
    net.connect(A0, T1, 0.4, 3); net.connect(A1, T1, 0.4, 2); net.connect(A2, T1, 0.4, 1)
    # Group 2: SAME anchors converge on T2 with delays 1,2,3 (key = the REVERSED offsets 2,1,0)
    net.connect(A0, T2, 0.4, 1); net.connect(A1, T2, 0.4, 2); net.connect(A2, T2, 0.4, 3)
    # downstream hop -> each group is a time-locked SEQUENCE, not a lone neuron (needs T + a3 coincident)
    net.connect(T1, D1, 0.6, 2); net.connect(A3, D1, 0.6, 2)
    net.connect(T2, D2, 0.6, 2); net.connect(A3, D2, 0.6, 2)
    return net


def _fire(offsets, a3=3):
    """Build an anchor schedule: a0..a2 at the given offset times (+ a3 to complete the downstream hop)."""
    return {A0: {offsets[0]}, A1: {offsets[1]}, A2: {offsets[2]}, A3: {a3}}


def main():
    net = _build()
    g1 = net.run(_fire([0, 1, 2]), steps=12)          # key that delay-matches Group 1
    g2 = net.run(_fire([2, 1, 0]), steps=12)          # SAME anchors, reversed timing -> Group 2
    rate = net.run(_fire([0, 0, 0]), steps=12)        # same anchors fire TOGETHER (rate pattern, no order)

    print("=" * 60)
    print("POLYCHRONIZATION — memory in spike timing (delay-matched groups)")
    print("=" * 60)
    print(f"G1 key [0,1,2]: T1 fires {g1.get(T1)}, D1 fires {g1.get(D1)}  | T2 {g1.get(T2)}, D2 {g1.get(D2)}")
    print(f"G2 key [2,1,0]: T2 fires {g2.get(T2)}, D2 fires {g2.get(D2)}  | T1 {g2.get(T1)}, D1 {g2.get(D1)}")
    print(f"rate  [0,0,0] : T1 {rate.get(T1)}, T2 {rate.get(T2)} (no convergence -> group silent)")
    print("combinatorial: same 4 anchors, timing alone selects which of 2 groups ignites")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    # 1. delay-matched key ignites its OWN group as a time-locked sequence (T then D)
    assert g1.get(T1) and g1.get(D1) and g1[T1][0] < g1[D1][0], f"G1 key must ignite the G1 sequence: {g1}"
    assert g2.get(T2) and g2.get(D2) and g2[T2][0] < g2[D2][0], f"G2 key must ignite the G2 sequence: {g2}"
    # 2. timing-SELECTIVE / rate-blind: the same anchors fired together do NOT converge -> group silent
    assert not rate.get(T1) and not rate.get(T2), f"rate pattern (no timing) must not ignite any group: {rate}"
    # 3. distinct groups over a SHARED anchor pool: each key ignites only its own target (combinatorial capacity)
    assert not g1.get(T2) and not g2.get(T1), f"timing must select the group: G1!=G2 ({g1=}, {g2=})"
    # 4. deterministic
    assert net.run(_fire([0, 1, 2]), steps=12) == g1, "same key -> identical polychronous activation"
    print(f"self-check OK: 2 PNGs over 4 shared anchors; G1 seq T1{g1[T1]}->D1{g1[D1]}, G2 seq "
          f"T2{g2[T2]}->D2{g2[D2]}; rate pattern silent (timing-selective, rate-blind); deterministic")


if __name__ == "__main__":
    main()

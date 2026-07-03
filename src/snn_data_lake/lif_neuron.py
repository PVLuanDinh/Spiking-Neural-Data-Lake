"""
Leaky Integrate-and-Fire (LIF) neuron — the canonical zero-dep spiking primitive.

The repo's LIF dynamics were embedded in reflex.py / tempotron.py / paradigm_b_engine.py with no single
reusable class. This is that class: the membrane potential V leaks toward rest each step
(decay = exp(-1/tau_m)), integrates the input current, and fires a spike + resets when V crosses threshold,
with an optional refractory period. Pure standard library, deterministic. The building block reused by
is_snn_forward, spike_slicer, and lt_gate.

  python -m snn_data_lake.lif_neuron        # (also run by run_selfchecks.py)
"""
import math


class LIFNeuron:
    """Discrete-time leaky integrate-and-fire neuron. step(current) -> 1/0 spike."""

    def __init__(self, tau_m=10.0, v_threshold=1.0, v_reset=0.0, v_rest=0.0, refractory=0):
        self.decay = math.exp(-1.0 / tau_m)          # membrane leak per timestep
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.refractory = refractory
        self.reset_state()

    def reset_state(self):
        self.v = self.v_rest
        self._refr = 0

    def step(self, current):
        """Advance one timestep with input `current`; return 1 if the neuron spiked else 0."""
        if self._refr > 0:                           # refractory: clamped, integrates nothing
            self._refr -= 1
            self.v = self.v_reset
            return 0
        self.v = self.v_rest + (self.v - self.v_rest) * self.decay + current   # leak toward rest + integrate
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self._refr = self.refractory
            return 1
        return 0

    def run(self, currents):
        """Run a sequence of input currents; return the spike train (list of 0/1)."""
        self.reset_state()
        return [self.step(c) for c in currents]


def main():
    # 1. leak: membrane decays toward rest with no input
    n = LIFNeuron(tau_m=5.0)
    n.reset_state()
    n.v = 0.9
    vs = []
    for _ in range(5):
        n.step(0.0)
        vs.append(n.v)
    leaks = all(vs[i + 1] < vs[i] for i in range(len(vs) - 1))

    # 2. threshold: strong constant input fires + resets repeatedly
    fired = sum(LIFNeuron(tau_m=10.0).run([0.5] * 40))

    # 3. F-I curve: rising input current -> monotonically rising firing rate
    rates = [sum(LIFNeuron(tau_m=10.0).run([c] * 60)) / 60 for c in (0.05, 0.1, 0.2, 0.4)]
    monotone = all(rates[i + 1] >= rates[i] for i in range(len(rates) - 1))

    # 4. refractory: spikes are spaced by at least the refractory period
    spikes = LIFNeuron(tau_m=10.0, refractory=5).run([1.5] * 24)
    idx = [i for i, s in enumerate(spikes) if s]
    gaps = [idx[i + 1] - idx[i] for i in range(len(idx) - 1)]

    print("=" * 60)
    print("LIF NEURON — canonical leaky integrate-and-fire primitive")
    print("=" * 60)
    print(f"leak (V, no input) : {[round(v, 3) for v in vs]}")
    print(f"strong input       : {fired} spikes / 40 steps")
    print(f"F-I curve (rate)   : {[round(r, 2) for r in rates]} for I in 0.05..0.4")
    print(f"refractory=5 gaps  : {gaps} (>= refractory)")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert leaks, "membrane must leak toward rest with no input"
    assert fired > 0, "strong constant input must fire the neuron"
    assert monotone and rates[-1] > rates[0], f"firing rate must rise with input current ({rates})"
    assert not gaps or min(gaps) > 5, f"refractory must space spikes by > refractory ({gaps})"
    assert LIFNeuron().run([0.3] * 30) == LIFNeuron().run([0.3] * 30), "same input -> identical train"
    print(f"self-check OK: leak, threshold+reset ({fired} spikes), monotone F-I {rates[0]:.2f}->{rates[-1]:.2f}, "
          f"refractory spacing; deterministic")


if __name__ == "__main__":
    main()

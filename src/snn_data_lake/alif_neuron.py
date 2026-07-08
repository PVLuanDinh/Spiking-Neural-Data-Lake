"""
Adaptive-threshold Leaky Integrate-and-Fire (ALIF) neuron -- the LSNN spike-frequency-adaptation primitive.

The canonical LIFNeuron (lif_neuron.py) has a FIXED firing threshold: under a constant supra-threshold drive it
fires at a constant rate forever and keeps no trace of its own recent activity. Bellec, Salaj, Subramoney,
Legenstein & Maass (NeurIPS 2018, "Long short-term memory and Learning-to-learn in networks of spiking neurons"
-- the LSNN) add ONE slow state variable that turns the LIF into a working-memory element. An adaptation
variable a(t) jumps up by 1 on every spike and decays with a SLOW time constant tau_a (>> the membrane tau_m);
it raises the *effective* firing threshold:

    V_th_eff(t) = v_threshold + beta * a(t)

Two consequences give the LSNN its name. (1) spike-frequency adaptation: each spike lifts the threshold, so a
constant input makes the neuron fire fast then progressively slower -- its inter-spike intervals grow, where a
plain LIF's stay flat. (2) LONG SHORT-TERM MEMORY: because tau_a >> tau_m, the elevated threshold OUTLASTS the
membrane by many membrane time constants -- after the input (and the fast membrane) have decayed back to rest,
a(t) is still clearly elevated, a slow trace of recent activity the plain LIF membrane cannot hold. That slow
threshold state is the working memory recurrent LSNNs use to bridge long delays, and the neuron model on which
this repo's e-prop (research/eprop.py) learns.

Distinct from the AdEx neuron (research/adex_neuron.py): AdEx adapts a subthreshold + spike-triggered CURRENT w
that is SUBTRACTED from the membrane; ALIF adapts the THRESHOLD itself. Different mechanism, same headline
behaviour (fire-fast-then-slow). Reduces EXACTLY to lif_neuron.LIFNeuron when beta = 0. Pure standard library,
deterministic by construction (no RNG, no wall-clock).

  python -m snn_data_lake.alif_neuron        # (also run by run_selfchecks.py)
"""
import math

from snn_data_lake.lif_neuron import LIFNeuron


class ALIFNeuron:
    """Discrete-time adaptive-threshold LIF neuron (Bellec et al. 2018). step(current) -> 1/0 spike."""

    def __init__(self, tau_m=10.0, tau_a=200.0, v_threshold=1.0, beta=1.5, v_reset=0.0, v_rest=0.0, refractory=0):
        self.decay = math.exp(-1.0 / tau_m)          # membrane leak per timestep (fast)
        self.rho = math.exp(-1.0 / tau_a)            # adaptation leak per timestep (slow: tau_a >> tau_m)
        self.v_threshold = v_threshold               # baseline threshold (the a=0 threshold, = plain LIF)
        self.beta = beta                             # threshold gain: how much one unit of adaptation lifts V_th
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.refractory = refractory
        self.reset_state()

    def reset_state(self):
        self.v = self.v_rest
        self.a = 0.0                                 # adaptation variable -> the slow threshold memory
        self._refr = 0

    def threshold(self):
        """Effective firing threshold right now: v_threshold + beta * a (rises with recent spiking)."""
        return self.v_threshold + self.beta * self.a

    def step(self, current):
        """Advance one timestep with input `current`; return 1 if the neuron spiked else 0."""
        self.a *= self.rho                           # adaptation always leaks (slow, tau_a >> tau_m)
        if self._refr > 0:                           # refractory: clamped, integrates nothing
            self._refr -= 1
            self.v = self.v_reset
            return 0
        self.v = self.v_rest + (self.v - self.v_rest) * self.decay + current   # leak toward rest + integrate
        if self.v >= self.v_threshold + self.beta * self.a:                    # threshold ADAPTS with a
            self.v = self.v_reset
            self._refr = self.refractory
            self.a += 1.0                            # a spike raises the adaptation (and thus the threshold)
            return 1
        return 0

    def run(self, currents):
        """Run a sequence of input currents; return the spike train (list of 0/1)."""
        self.reset_state()
        return [self.step(c) for c in currents]


def _isis(train):
    """Inter-spike intervals (steps between consecutive spikes) of a 0/1 spike train."""
    idx = [i for i, s in enumerate(train) if s]
    return [idx[i + 1] - idx[i] for i in range(len(idx) - 1)]


def main():
    tau_m, v_th, drive = 10.0, 1.0, 0.7

    # (a) spike-frequency adaptation: identical constant supra-threshold input into an ALIF and a plain LIF.
    #     The ALIF threshold climbs with each spike -> inter-spike intervals GROW; the LIF's ISI stays flat.
    alif = ALIFNeuron(tau_m=tau_m, tau_a=300.0, v_threshold=v_th, beta=1.2)
    alif_train = alif.run([drive] * 250)
    alif_isi = _isis(alif_train)
    lif_isi = _isis(LIFNeuron(tau_m=tau_m, v_threshold=v_th).run([drive] * 250))

    # (b) LONG MEMORY IN THE THRESHOLD: drive to build up adaptation, then cut the input. The membrane (tau_m)
    #     collapses within a few steps, but the adaptation trace (tau_a >> tau_m) stays elevated far longer,
    #     then finally returns to baseline. That slow trace is the LSNN's "long short-term memory".
    mem = ALIFNeuron(tau_m=tau_m, tau_a=300.0, v_threshold=v_th, beta=1.2)
    for _ in range(200):
        mem.step(drive)
    a_drive = mem.a                                  # adaptation the moment the input stops
    for _ in range(5 * int(tau_m)):                  # 5 membrane time constants of silence
        mem.step(0.0)
    a_5tau_m, v_5tau_m = mem.a, mem.v
    for _ in range(2000):                            # many adaptation time constants later -> recovery
        mem.step(0.0)
    a_recovered = mem.a

    # (c) beta = 0 collapses the ALIF exactly onto the plain LIF -- identical spike train on the same input.
    seq = [0.4] * 100 + [0.0] * 40 + [0.6] * 60 + [0.15] * 50
    alif_b0 = ALIFNeuron(tau_m=tau_m, v_threshold=v_th, beta=0.0).run(seq)
    lif_ref = LIFNeuron(tau_m=tau_m, v_threshold=v_th).run(seq)

    # (d) determinism: no RNG, no wall-clock -> byte-identical spike train on a repeat run.
    repeat_isi = _isis(ALIFNeuron(tau_m=tau_m, tau_a=300.0, v_threshold=v_th, beta=1.2).run([drive] * 250))

    print("=" * 66)
    print("ALIF NEURON -- adaptive-threshold LIF, long-memory adaptation (Bellec 2018)")
    print("=" * 66)
    print(f"input              : constant I={drive} into ALIF (tau_m={tau_m}, tau_a=300, beta=1.2) vs plain LIF")
    print(f"ALIF ISI (grows)   : {alif_isi}")
    print(f"LIF  ISI (flat)    : {lif_isi[:len(alif_isi)]} ...")
    print(f"threshold memory   : a={a_drive:.3f} at input-stop -> {a_5tau_m:.3f} after 5*tau_m "
          f"({100 * a_5tau_m / a_drive:.0f}% held) while membrane v={v_5tau_m:.4f} (~0); a->{a_recovered:.4f} recovered")
    print(f"beta=0 == LIF      : {alif_b0 == lif_ref}")
    print("=" * 66)

    # ---- self-check (deterministic asserts, zero-dep) ----
    # (a) spike-frequency adaptation: ALIF ISIs strictly grow, plain LIF ISIs are (near-)constant
    assert len(alif_isi) >= 5, f"ALIF must fire a train under constant drive ({alif_isi})"
    assert all(y > x for x, y in zip(alif_isi, alif_isi[1:])), f"ALIF ISIs must strictly grow (adaptation) ({alif_isi})"
    assert alif_isi[-1] >= 3 * alif_isi[0], f"adaptation must clearly lengthen ISIs ({alif_isi[0]} -> {alif_isi[-1]})"
    assert max(lif_isi) - min(lif_isi) <= 1, f"plain LIF ISIs must be (near-)constant -- no adaptation ({lif_isi[:6]})"
    # (b) long memory: the adaptation trace outlasts the membrane, then recovers to baseline
    assert a_5tau_m > 0.5 * a_drive, f"threshold memory must persist many tau_m past the input ({a_5tau_m:.3f} vs {a_drive:.3f})"
    assert abs(v_5tau_m) < 0.05 * v_th, f"membrane must have collapsed within 5*tau_m ({v_5tau_m:.4f})"
    assert a_recovered < 0.02 * a_drive, f"adaptation must return to baseline eventually ({a_recovered:.4f})"
    # (c) correctness anchor: beta=0 is exactly a plain LIF on the same input
    assert alif_b0 == lif_ref, "beta=0 ALIF spike train must equal the plain LIF spike train"
    # (d) determinism: identical input -> identical spike train
    assert repeat_isi == alif_isi, "same input -> identical spike train (deterministic)"
    print(f"self-check OK: ALIF ISI {alif_isi[0]}->{alif_isi[-1]} grows vs flat LIF {lif_isi[0]}; "
          f"threshold memory {100 * a_5tau_m / a_drive:.0f}% held after 5*tau_m (v~0), recovers; beta=0==LIF; deterministic")


if __name__ == "__main__":
    main()

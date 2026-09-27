"""
Adaptive exponential integrate-and-fire (AdEx) neuron -- spike-frequency adaptation, pure Python.

The repo's spiking neurons cover leaky-integrate (LIF), quadratic-with-recovery (Izhikevich), kernel
(SRM) and resonator (resonate-and-fire) dynamics, but none captures the SHARP exponential spike upswing
plus a slow ADAPTATION current. Brette and Gerstner (2005, "Adaptive exponential integrate-and-fire
model as an effective description of neuronal activity", J. Neurophysiol. 94:3637-3642) add two things
to LIF: (1) an exponential term g_L*DeltaT*exp((V - V_T)/DeltaT) that reproduces the real Na+ spike
initiation, and (2) a second state variable w -- an adaptation current with a subthreshold coupling "a"
and a spike-triggered jump "b" -- that makes the neuron fire fast then slow down (spike-frequency
adaptation) or burst, depending on the parameter set (Naud, Marcille, Clopath, Gerstner 2008,
Biol. Cybern. 99:335-347). Two coupled equations, in physical units (pF, nS, mV, pA, ms):

    C dV/dt = -g_L (V - E_L) + g_L DeltaT exp((V - V_T)/DeltaT) - w + I
    tau_w dw/dt = a (V - E_L) - w        ; on spike: V -> V_reset, w -> w + b

Why on-thesis: adaptation is the neuromorphic way a sensor neuron reports CHANGE rather than absolute
level -- it habituates to a constant drive and re-sensitises to novelty, the spike-domain analogue of
this repo's drift_monitor / valence machinery, at the single-neuron scale and at zero energy budget.
Pure standard library, deterministic (fixed-step forward Euler, no randomness, no wall-clock). On-thesis
with lif_neuron / izhikevich_neuron / resonate_and_fire / intrinsic_plasticity.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.growth.adex_neuron        # (also run by run_selfchecks.py)


P7 NOTE (2026-08-04, on promotion into cortex/): the phrase "pure standard library" above
described obedience to the zero-dependency rule, abolished 2026-07-20. The stdlib implementation
is KEPT because it is correct on its merits here, not because a rule requires it: this is a
SCALAR single-neuron model with a few floats of state, stepped per-neuron, exactly like its now
sibling `growth/lif_neuron.py`. numpy pays off on populations, not on one neuron -- vectorizing a
scalar step would be slower and would duplicate dynamics P2 says must have one definition. What
was removed is the BOAST, not the design.
"""

import math

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "AdEx.step": (
        "NOT_A_RULE",
        "advances AdEx membrane dynamics one timestep. Caught by the walk because it "
        "augmented-assigns `w` -- which here is the ADAPTATION CURRENT, not a synapse. research.md R1 "
        "predicted this exact false positive by name",
    ),
}

# ---- exp-argument cap: the spike upswing diverges by design; cap the exponent so forward Euler cannot
#      raise OverflowError before the V >= V_peak cutoff fires on the same or next step (exp(30) ~ 1e13). ----
_MAX_EXP_ARG = 30.0


class AdEx:
    """Two-variable adaptive exponential integrate-and-fire neuron (Brette and Gerstner 2005)."""

    def __init__(
        self,
        C=200.0,
        g_L=12.0,
        E_L=-70.0,
        V_T=-50.0,
        delta_T=2.0,
        a=2.0,
        tau_w=300.0,
        b=60.0,
        V_reset=-58.0,
        V_peak=0.0,
        dt=0.1,
    ):
        self.C, self.g_L, self.E_L = C, g_L, E_L  # capacitance pF, leak nS, rest mV
        self.V_T, self.delta_T = V_T, delta_T  # exp-threshold mV, sharpness mV
        self.a, self.tau_w, self.b = a, tau_w, b  # subthreshold nS, adaptation ms, spike-jump pA
        self.V_reset, self.V_peak, self.dt = V_reset, V_peak, dt
        self.reset()

    def reset(self):
        self.V = self.E_L
        self.w = 0.0

    def step(self, I):
        """Advance one dt with input current I (pA); return 1 on a spike, else 0."""
        arg = (self.V - self.V_T) / self.delta_T
        if arg > _MAX_EXP_ARG:
            arg = _MAX_EXP_ARG
        exp_term = self.g_L * self.delta_T * math.exp(arg)
        dV = (-self.g_L * (self.V - self.E_L) + exp_term - self.w + I) / self.C
        dw = (self.a * (self.V - self.E_L) - self.w) / self.tau_w
        self.V += dV * self.dt
        self.w += dw * self.dt
        if self.V >= self.V_peak or math.isinf(self.V) or math.isnan(self.V):
            self.V = self.V_reset
            self.w += self.b  # spike-triggered adaptation increment
            return 1
        return 0

    def run(self, I, ms):
        """Constant current I (pA) for `ms` milliseconds; return the list of spike times (ms)."""
        self.reset()
        n = int(round(ms / self.dt))
        return [k * self.dt for k in range(n) if self.step(I)]


def _isis(spike_times):
    return [b - a for a, b in zip(spike_times, spike_times[1:])]


def main():
    ms = 1000.0

    # 1. ADAPTING neuron (a=2 nS, b=60 pA): fires fast then slows -> inter-spike intervals grow.
    adapting = AdEx(a=2.0, b=60.0, tau_w=300.0)
    st_adapt = adapting.run(500.0, ms)
    isi_adapt = _isis(st_adapt)

    # 2. TONIC neuron (a=0, b=0): no adaptation -> regular firing, constant inter-spike intervals.
    tonic = AdEx(a=0.0, b=0.0)
    st_tonic = tonic.run(500.0, ms)
    isi_tonic = _isis(st_tonic)

    # 3. f-I curve of the tonic neuron: more current -> more spikes (monotone).
    currents = [300.0, 500.0, 700.0]
    fI = [len(AdEx(a=0.0, b=0.0).run(I, ms)) for I in currents]

    # 4. subthreshold: current below rheobase -> no spike at all.
    st_sub = AdEx(a=0.0, b=0.0).run(100.0, ms)

    print("=" * 64)
    print("AdEx -- adaptive exponential integrate-and-fire (Brette & Gerstner 2005)")
    print("=" * 64)
    print(
        f"adapting neuron : {len(st_adapt)} spikes/s, ISI {isi_adapt[0]:.1f} -> {isi_adapt[-1]:.1f} ms (grows)"
    )
    print(
        f"tonic neuron    : {len(st_tonic)} spikes/s, ISI {min(isi_tonic):.1f}..{max(isi_tonic):.1f} ms (flat)"
    )
    print(f"f-I curve       : I {currents} pA -> {fI} spikes/s (monotone)")
    print(f"subthreshold    : I=100 pA -> {len(st_sub)} spikes (below rheobase)")
    print("=" * 64)

    # ---- self-check (deterministic asserts) ----
    assert len(st_adapt) >= 3 and len(st_tonic) >= 3, "both neurons must fire a train under 500 pA"
    # spike-frequency adaptation: ISIs of the adapting neuron increase then saturate. Allow one dt of slack:
    # at the steady-state plateau the ISI wobbles by a single fixed-step grid cell (e.g. 76.0 vs 76.1 ms).
    _tol = adapting.dt + 1e-9
    assert all(y >= x - _tol for x, y in zip(isi_adapt, isi_adapt[1:])), (
        f"adapting ISIs must not shrink ({isi_adapt})"
    )
    assert isi_adapt[-1] > 1.3 * isi_adapt[0], (
        f"adaptation must lengthen ISIs ({isi_adapt[0]:.2f} -> {isi_adapt[-1]:.2f})"
    )
    # tonic neuron is regular: ISI spread is tiny relative to the mean
    mean_tonic = sum(isi_tonic) / len(isi_tonic)
    assert (max(isi_tonic) - min(isi_tonic)) < 0.05 * mean_tonic, (
        f"tonic ISIs must be near-constant ({isi_tonic})"
    )
    # adaptation reduces the firing rate: fewer spikes than the identically-driven tonic neuron
    assert len(st_adapt) < len(st_tonic), (
        "adaptation must lower the firing rate vs the non-adapting neuron"
    )
    # f-I monotone and subthreshold silent
    assert fI[0] < fI[1] < fI[2], f"f-I curve must be strictly increasing ({fI})"
    assert len(st_sub) == 0, "current below rheobase must not spike"
    # determinism: identical params + input -> identical spike train
    assert AdEx(a=2.0, b=60.0, tau_w=300.0).run(500.0, ms) == st_adapt, (
        "same params -> identical train (deterministic)"
    )
    print(
        f"self-check OK: adaptation grew ISI {isi_adapt[0]:.1f}->{isi_adapt[-1]:.1f} ms "
        f"({len(st_adapt)} vs {len(st_tonic)} tonic spikes); f-I {fI} monotone; subthreshold silent; deterministic"
    )


if __name__ == "__main__":
    main()

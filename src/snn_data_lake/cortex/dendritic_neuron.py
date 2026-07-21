"""
Two-compartment dendritic neuron with BAC firing -- the core's first NONLINEAR DENDRITIC unit.

Every other neuron in this package is a POINT neuron: a single compartment that integrates one input
current and fires -- lif_neuron (leaky integrate-and-fire) and izhikevich_neuron in the product,
srm / adex / resonate_and_fire in research/. This module adds the first neuron with a real *dendrite*:
a reduced two-compartment pyramidal cell (soma/basal + apical tuft) that performs nonlinear dendritic
coincidence detection via BAC firing -- "Backpropagation-Activated Ca2+ spike firing" (Larkum).

Mechanism (reduced, spike/rate level -- NOT Hodgkin-Huxley, no conductances, no real ion channels):
  * SOMA    membrane V_s leaks (decay = exp(-1/tau_s)), integrates the BASAL feed-forward current,
            fires a somatic spike + resets when V_s >= v_thresh.
  * APICAL  membrane V_a leaks (tau_a, slower -> sustained), integrates the APICAL top-down/context
            current. Alone it settles BELOW the Ca2+ threshold ca_thresh -- apical input by itself is
            sub-threshold and produces no dendritic spike (apical isolation).
  * BAC     a somatic spike BACKPROPAGATES: it adds a depolarizing bump to the apical compartment.
            Only when that backprop bump lands on top of COINCIDENT apical input does V_a cross
            ca_thresh -> a dendritic Ca2+ PLATEAU that injects a strong, sustained boost current into
            the soma -> a BURST. Apical alone -> no somatic spike -> no backprop -> no plateau. Basal
            alone -> soma spikes but V_a (no apical drive) never reaches ca_thresh, so the rate
            saturates at a basal maximum. Apical AND basal coincident -> plateau -> burst whose rate
            EXCEEDS the basal-only maximum (supralinear -- the Larkum apical-amplification signature).

Net effect = COINCIDENCE DETECTION: the cell bursts only when top-down (apical) and feed-forward
(basal) input co-occur, at a rate no amount of basal input alone can reach.

Distinct from siblings: lif_neuron / izhikevich_neuron are single-compartment point neurons with no
dendritic nonlinearity. htm_sequence_memory uses binary distal "segments" that PREDICT the next
SDR -- a point cell with no compartmental membrane, no plateau potential, no supralinear coincidence
burst. This is a two-compartment membrane model whose apical+basal COINCIDENCE fires a Ca2+
plateau/burst that beats the basal-only rate. Different mechanism, different math.

Honesty: a reduced rate/spike caricature of the Larkum BAC mechanism -- synthetic input, no
conductance-based channels; it is nonlinear dendritic coincidence detection, NOT cognition/AGI.
Refs: Larkum, "A cellular mechanism for cortical associations" (Trends Neurosci 2013); Sacramento /
Greedy et al., "Nonlinear Dendritic Coincidence Detection for Supervised Learning" (arXiv:2107.05336);
apical amplification / isolation / drive regimes (arXiv:2311.06074).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.dendritic_neuron
"""

import math
import random


class DendriticNeuron:
    """Reduced two-compartment pyramidal neuron. step(basal, apical) -> 1/0 somatic spike.

    Exposes .plateau_active (a Ca2+ plateau is currently injecting boost) and .ca_spikes (count of
    dendritic Ca2+ plateaus ignited). Deterministic; no internal randomness."""

    def __init__(
        self,
        tau_s=8.0,
        tau_a=15.0,
        v_thresh=1.0,
        v_reset=0.0,
        ca_thresh=1.0,
        bp_bump=0.20,
        plateau_boost=0.8,
        plateau_dur=8,
        refractory=0,
    ):
        self.decay_s = math.exp(-1.0 / tau_s)  # soma membrane leak per timestep
        self.decay_a = math.exp(-1.0 / tau_a)  # apical membrane leak (slower -> sustained)
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.ca_thresh = ca_thresh
        self.bp_bump = bp_bump  # backprop depolarization added to apical on a soma spike
        self.plateau_boost = plateau_boost  # current injected into soma while a plateau is active
        self.plateau_dur = plateau_dur  # plateau lifetime in steps
        self.refractory = refractory
        self.reset_state()

    def reset_state(self):
        self.v_s = 0.0
        self.v_a = 0.0
        self._plateau = 0
        self._refr = 0
        self.plateau_active = False
        self.ca_spikes = 0

    def step(self, basal, apical=0.0):
        """Advance one timestep. basal = feed-forward current, apical = top-down/context current.
        Returns 1 if the soma spiked else 0."""
        # --- apical compartment: leak toward rest + integrate the top-down current ---
        self.v_a = self.v_a * self.decay_a + apical
        # --- Ca2+ plateau: sustain an active one, or ignite when V_a crosses ca_thresh ---
        boost = 0.0
        if self._plateau > 0:  # sustained plateau keeps injecting boost
            self._plateau -= 1
            boost = self.plateau_boost
        if (
            self._plateau <= 0 and self.v_a >= self.ca_thresh
        ):  # BAC: apical (+ backprop bump) coincidence
            self._plateau = self.plateau_dur
            self.ca_spikes += 1
            boost = self.plateau_boost
        self.plateau_active = self._plateau > 0
        # --- soma compartment: leak + basal feed-forward + plateau boost, then fire + reset ---
        if self._refr > 0:  # refractory: clamped, integrates nothing
            self._refr -= 1
            self.v_s = self.v_reset
            return 0
        self.v_s = self.v_s * self.decay_s + basal + boost
        if self.v_s >= self.v_thresh:
            self.v_s = self.v_reset
            self._refr = self.refractory
            self.v_a += self.bp_bump  # backpropagating AP depolarizes the apical dendrite
            return 1
        return 0

    def run(self, basal_seq, apical_seq=None):
        """Run paired basal/apical current sequences; return the somatic spike train (list of 0/1)."""
        self.reset_state()
        if apical_seq is None:
            apical_seq = [0.0] * len(basal_seq)
        return [self.step(b, a) for b, a in zip(basal_seq, apical_seq)]


def _rate(train):
    return sum(train) / len(train) if train else 0.0


def _poisson(seed, p, amp, n):
    """Deterministic Bernoulli current train: amp with prob p else 0, from a seeded RNG."""
    rng = random.Random(seed)
    return [amp if rng.random() < p else 0.0 for _ in range(n)]


def main():
    T = 300
    basal_i, apical_i = 0.25, 0.057

    # 1. BASAL DRIVE: strong basal alone -> steady somatic firing at a basal-max rate, no plateau
    nb = DendriticNeuron()
    basal_rate = _rate(nb.run([basal_i] * T, [0.0] * T))
    basal_plateaus = nb.ca_spikes

    # 2. APICAL ISOLATION: apical alone -> essentially silent; no Ca2+ spike without backprop
    na = DendriticNeuron()
    apical_rate = _rate(na.run([0.0] * T, [apical_i] * T))
    apical_plateaus = na.ca_spikes

    # 3/4. BAC COINCIDENCE (headline): apical AND basal together -> Ca2+ plateau + supralinear burst
    nc = DendriticNeuron()
    both_rate = _rate(nc.run([basal_i] * T, [apical_i] * T))
    both_plateaus = nc.ca_spikes

    # 5. DETERMINISM: same seed -> same Poisson input -> identical spike train (two fresh cells)
    b1, a1 = _poisson(11, 0.85, basal_i, T), _poisson(12, 0.85, apical_i, T)
    b2, a2 = _poisson(11, 0.85, basal_i, T), _poisson(12, 0.85, apical_i, T)
    train1 = DendriticNeuron().run(b1, a1)
    train2 = DendriticNeuron().run(b2, a2)

    gain = both_rate / basal_rate if basal_rate else float("inf")
    print("=" * 66)
    print("DENDRITIC NEURON -- two-compartment BAC firing (apical-basal coincidence)")
    print("=" * 66)
    print(f"basal drive     (basal only) : rate {basal_rate:.3f}  Ca2+ plateaus {basal_plateaus}")
    print(
        f"apical isolation (apical only): rate {apical_rate:.3f}  Ca2+ plateaus {apical_plateaus}"
    )
    print(f"BAC coincidence  (both)      : rate {both_rate:.3f}  Ca2+ plateaus {both_plateaus}")
    print(f"supralinear gain             : both / basal-max = {gain:.2f}x")
    print("=" * 66)

    # ---- self-check: additive -- a new member of the neuron-model family (lif/izhikevich),
    # changes no existing behaviour; ceiling: reduced rate/spike caricature, not conductance-based HH.
    # Deterministic asserts, ASCII-only prints ----
    assert basal_rate > 0.0, "basal drive alone must make the soma fire (basal-max rate)"
    assert basal_plateaus == 0, "basal drive alone must NOT ignite a Ca2+ plateau (no apical drive)"
    assert apical_rate <= 0.02, (
        f"apical input alone must leave the soma near-silent (got {apical_rate:.3f})"
    )
    assert apical_plateaus == 0, (
        "apical alone must stay sub-threshold: no Ca2+ spike without backprop"
    )
    assert both_plateaus > 0, "coincident apical+basal must ignite a dendritic Ca2+ plateau"
    assert both_rate > basal_rate * 1.5, (
        f"BAC must be SUPRALINEAR: both {both_rate:.3f} must clearly exceed basal-max {basal_rate:.3f}"
    )
    assert both_rate > apical_rate and basal_rate > apical_rate, (
        "coincidence detector: high only when apical AND basal co-occur, not for either alone"
    )
    assert train1 == train2, "same seed/input -> identical spike train (deterministic)"
    print(
        f"self-check OK: basal-max {basal_rate:.3f}, apical-only {apical_rate:.3f}, "
        f"BAC {both_rate:.3f} ({gain:.1f}x supralinear), plateau fired ({both_plateaus} Ca2+ spikes); "
        f"deterministic"
    )


if __name__ == "__main__":
    main()

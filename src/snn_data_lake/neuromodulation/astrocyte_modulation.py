"""
Astrocyte / tripartite-synapse modulation -- slow, LOCAL, activity-driven glial gain.

An astrocyte process wraps a GROUP of synapses and forms a THIRD element of the synapse (Araque,
Parpura, Sanzgiri & Haydon, Trends Neurosci 22:208, 1999; Volterra & Meldolesi, Nat Rev Neurosci
6:626, 2005). It slowly integrates the LOCAL synaptic activity into intracellular calcium (Ca); when
Ca crosses a threshold the astrocyte releases gliotransmitters that scale the wrapped synapses'
efficacy (a gain) on a SLOW timescale, then Ca -- and the gain -- decays slowly. Net: a local,
activity-driven, slow feedback that gates synaptic transmission.

Model here (reduced, phenomenological):
  - one slow Ca variable per astrocyte DOMAIN: ca += ca_rise*activity - ca_decay*ca, on a LONG time
    constant (tau_ca = 1/ca_decay = 50 ticks >> the LIF membrane tau_m = 10);
  - a threshold: gliotransmission engages only when ca >= ca_thresh, so a slow integrator with a
    threshold -- brief or weak activity never triggers it;
  - a potentiating gain (g >= 1): while engaged the wrapped synapses' drive is scaled by
    1 + gain*(ca - ca_thresh); the gain relaxes SLOWLY back to 1 (tau_gain) after activity stops.
Multiple domains live in one astrocyte, so locality is demonstrable: an active domain and a quiet
domain hold DIFFERENT Ca / gain at the same time.

DISTINCTNESS (why this is a new primitive, not a duplicate):
  - cortisol.py / acetylcholine.py are GLOBAL scalar neuromodulators -- ONE number for the whole
    network. The astrocyte is LOCAL (per-domain Ca/gain, driven by nearby activity) and SLOW; two
    domains carry different modulation at once, which a single global scalar cannot represent.
  - short_term_plasticity.py is a FAST per-spike presynaptic resource dynamic (depression/
    facilitation) recovering over a short tau. The astrocyte is a SEPARATE THIRD element that slowly
    integrates activity over a LONG window and feeds back a gain -- a different timescale + element.
  - homeostatic_scaling.py / intrinsic_plasticity.py are the NEURON's own long-term rate/
    excitability set-point control. The astrocyte modulates SYNAPTIC transmission via glial Ca, not
    the neuron's own homeostasis.

  python -m snn_data_lake.neuromodulation.astrocyte_modulation        # (also run by run_selfchecks.py)

A reduced slow-Ca + threshold-gliotransmission model of tripartite synapse on synthetic
activity -- phenomenological glial gain, NOT IP3 / Ca-induced-Ca-release biophysics, no diffusion or
astrocyte network, not cognition.
"""

import math
import random

from snn_data_lake.cortex.lif_neuron import LIFNeuron


class Astrocyte:
    """Slow, local, activity-driven glial gain over one or more synaptic domains."""

    def __init__(
        self,
        n_domains=2,
        ca_rise=0.02,
        ca_decay=0.02,
        ca_thresh=0.5,
        gain=2.0,
        tau_gain=30.0,
        ca_noise=0.0,
        seed=0,
    ):
        self.n_domains = n_domains
        self.ca_rise = ca_rise
        self.ca_decay = ca_decay  # tau_ca = 1/ca_decay (LONG relative to tau_m)
        self.ca_thresh = ca_thresh
        self.gain = gain
        self.gain_relax = math.exp(-1.0 / tau_gain)  # slow return of the gain toward 1
        self.ca_noise = ca_noise  # optional Ca microdomain jitter (default off)
        self._rng = random.Random(seed)
        self.reset()

    def reset(self):
        """Clear all domains: Ca -> 0, gain -> 1 (no modulation)."""
        self.calcium = [0.0] * self.n_domains
        self.gains = [1.0] * self.n_domains

    def _gliotransmission(self, ca):
        """Gain from Ca: exactly 1 below threshold, potentiating (>1) once Ca crosses it."""
        if ca < self.ca_thresh:
            return 1.0
        return 1.0 + self.gain * (ca - self.ca_thresh)

    def step(self, activity_per_domain):
        """Advance Ca + gain one slow tick given per-domain activity; return per-domain gains."""
        for i in range(self.n_domains):
            a = max(0.0, activity_per_domain[i])
            if self.ca_noise:
                a = max(0.0, a + self._rng.gauss(0.0, self.ca_noise))
            # slow leaky Ca integration -- a LONG-window integrator of local activity
            self.calcium[i] += self.ca_rise * a - self.ca_decay * self.calcium[i]
            if self.calcium[i] < 0.0:
                self.calcium[i] = 0.0
            target = self._gliotransmission(self.calcium[i])
            # the gain follows the Ca-driven target SLOWLY (glia are slow) and relaxes slowly to 1
            self.gains[i] = target + (self.gains[i] - target) * self.gain_relax
        return list(self.gains)

    def modulated_transmission(self, domain, presynaptic_drive):
        """Wrapped-synapse output: presynaptic drive scaled by the domain's current glial gain."""
        return presynaptic_drive * self.gains[domain]


def _run(astro, activities):
    """Drive a single-domain astrocyte through activities; return (ca_trace, gain_trace)."""
    ca, g = [], []
    for a in activities:
        astro.step([a])
        ca.append(astro.calcium[0])
        g.append(astro.gains[0])
    return ca, g


def main():
    t_on, t_off = 100, 60
    n_persist = 15  # persistence horizon after the drive stops

    # 1. sustained-high drive -> Ca ramps past threshold -> gliotransmission engages
    sus_ca, sus_g = _run(Astrocyte(n_domains=1), [1.0] * t_on + [0.0] * t_off)
    sus_peak_g = max(sus_g)
    cross_tick = next(i for i, c in enumerate(sus_ca) if c >= 0.5)
    persist_g = sus_g[t_on + n_persist - 1]  # gain n_persist ticks after the drive stops

    # 1b. brief-high drive -> Ca stays subthreshold -> no gliotransmission
    brief_ca, brief_g = _run(Astrocyte(n_domains=1), [1.0] * 5 + [0.0] * (t_on + t_off - 5))
    brief_peak_g, brief_ca_max = max(brief_g), max(brief_ca)

    # 1c. low sustained drive -> also subthreshold
    _, low_g = _run(Astrocyte(n_domains=1), [0.2] * t_on)
    low_gain = low_g[-1]

    # 3. locality: ONE astrocyte, domain A driven, domain B quiet -> different Ca / gain
    loc = Astrocyte(n_domains=2)
    for _ in range(t_on):
        loc.step([1.0, 0.0])
    gain_active, gain_quiet = loc.gains
    ca_active, ca_quiet = loc.calcium

    # 4. functional: the modulated gain changes a downstream LIF neuron's output
    drive = 0.11
    base_spikes = sum(LIFNeuron().run([drive] * 120))
    mod_astro = Astrocyte(n_domains=1)
    for _ in range(t_on):
        mod_astro.step([1.0])
    mod_current = mod_astro.modulated_transmission(0, drive)
    mod_spikes = sum(LIFNeuron().run([mod_current] * 120))

    # 5. determinism: two fresh instances, same seed/inputs -> identical traces
    seq = [[1.0]] * 30 + [[0.0]] * 30
    d1 = Astrocyte(n_domains=1, seed=0)
    d2 = Astrocyte(n_domains=1, seed=0)
    tr1 = [(round(d1.step(a)[0], 12), round(d1.calcium[0], 12)) for a in seq]
    tr2 = [(round(d2.step(a)[0], 12), round(d2.calcium[0], 12)) for a in seq]

    def sample(xs):
        return [round(xs[i], 3) for i in range(0, len(xs), 20)]

    print("=" * 68)
    print("ASTROCYTE / TRIPARTITE-SYNAPSE -- slow local activity-driven glial gain")
    print("=" * 68)
    print("tau_ca = 50 ticks (slow integrator) vs LIF membrane tau_m = 10 (fast)")
    print(f"sustained ca (every 20): {sample(sus_ca)}  crosses thresh 0.5 at tick {cross_tick}")
    print(f"sustained gain (ev. 20): {sample(sus_g)}  peak gain {sus_peak_g:.3f}")
    print(f"brief ca (every 20)    : {sample(brief_ca)}  max ca {brief_ca_max:.3f} (< thresh)")
    print(f"brief gain (every 20)  : {sample(brief_g)}  peak gain {brief_peak_g:.3f} (~1)")
    print(f"low-drive gain         : {low_gain:.3f} (~1, subthreshold)")
    print(f"persistence: gain +{n_persist} ticks after drive stops = {persist_g:.3f} (still on)")
    print(
        f"locality  : domain A ca {ca_active:.3f} gain {gain_active:.3f}  |  "
        f"domain B ca {ca_quiet:.3f} gain {gain_quiet:.3f}"
    )
    print(
        f"downstream LIF spikes  : baseline {base_spikes}  ->  modulated {mod_spikes}"
        f"  (gain {mod_astro.gains[0]:.2f} on drive {drive})"
    )
    print("=" * 68)

    # ---- self-checks: deterministic asserts ----
    # 1. slow activity-dependent threshold gliotransmission
    assert sus_peak_g - 1.0 > 0.2, (
        f"sustained-high must engage gliotransmission (gain {sus_peak_g:.3f} >> 1)"
    )
    assert abs(brief_peak_g - 1.0) < 0.05 and brief_ca_max < 0.5, (
        f"brief drive must stay subthreshold (ca {brief_ca_max:.3f}, gain {brief_peak_g:.3f} ~ 1)"
    )
    assert abs(low_gain - 1.0) < 0.05, f"low drive must not engage (gain {low_gain:.3f} ~ 1)"
    assert cross_tick > 20, f"Ca must ramp SLOWLY, not per-spike (crosses tick {cross_tick})"
    # 2. slow timescale / persistence (tau_ca=50 >> membrane tau=10; still modulated N ticks later)
    assert persist_g - 1.0 > 0.1, (
        f"gain must persist {n_persist} ticks after offset (tau_ca=50 vs tau_m=10): {persist_g:.3f}"
    )
    # 3. locality -- an active and a quiet domain in the SAME astrocyte end clearly different
    assert gain_active - gain_quiet > 0.3 and ca_active - ca_quiet > 0.3, (
        f"active vs quiet domain must differ (gain {gain_active:.3f} vs {gain_quiet:.3f})"
    )
    assert abs(gain_quiet - 1.0) < 1e-9, f"quiet domain must be unmodulated (gain {gain_quiet:.3f})"
    # 4. functional effect on transmission
    assert mod_spikes > base_spikes, (
        f"gliotransmission must change downstream output ({mod_spikes} vs {base_spikes} spikes)"
    )
    # 5. determinism
    assert tr1 == tr2, "same seed/inputs must give identical Ca/gain traces"
    print(
        f"self-check OK: sustained {sus_peak_g:.2f} engages vs brief {brief_peak_g:.2f} ~ 1; "
        f"persists {persist_g:.2f} +{n_persist}; A {gain_active:.2f} != B {gain_quiet:.2f}; "
        f"LIF {base_spikes}->{mod_spikes} spikes; deterministic"
    )


if __name__ == "__main__":
    main()

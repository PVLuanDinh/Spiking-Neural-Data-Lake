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
        """Wrapped-synapse output: presynaptic drive scaled by the domain's current glial
        gain. Producer-side convenience over THE one transmission mechanism below -- the
        VolumeTransmitter consumer calls the same function with the gain it RECEIVED on
        the declared channel (T027 review: the first draft re-implemented `drive * gain`
        inline, leaving this capability with zero consumers and a P2 second mechanism)."""
        return modulated_transmission(self.gains[domain], presynaptic_drive)


def modulated_transmission(gain, presynaptic_drive):
    """THE tripartite-synapse transmission mechanism: presynaptic drive scaled by the
    glial gain -- one definition, called by the Astrocyte producer-side method above and
    by the VolumeTransmitter channel consumer below (P2: exactly one path)."""
    return presynaptic_drive * gain


def declare_ports(manifest):
    """Manifest registrant (specs/022 T027, the glia wire): the astrocyte's per-domain
    gain field as a VolumeTransmitter-style DIFFUSE channel. OUT `glial_gain_out` carries
    the domain gains; IN `synaptic_gain_in` is the wrapped tissue's receiving surface (the
    VolumeTransmitter relay below), joined `sum` over its single source. The edge is a
    SELF-EDGE deliberately: the tripartite synapse's third element modulates transmission
    in the SAME tissue it wraps -- producer and consumer share the faculty the way glia
    and synapses share the neuropil. Delivery is all-or-none (the whole gain field each
    tick); a distance-graded delivery would need T020's graded association binding, which
    finalizes at T030 -- stated, not smuggled."""
    from snn_data_lake.substrate.wiring_manifest import Port

    manifest.declare_port(
        "astrocyte_modulation",
        Port(
            name="glial_gain_out",
            region="neuromodulation",
            shape=(2,),
            dtype="f4",
            unit="gain_x",
            encoding="VEC_DENSE",
            kind="OUT",
            declaring_module="astrocyte_modulation",
            delivery_policy="all-or-none",
        ),
    )
    manifest.declare_port(
        "astrocyte_modulation",
        Port(
            name="synaptic_gain_in",
            region="neuromodulation",
            shape=(2,),
            dtype="f4",
            unit="gain_x",
            encoding="VEC_DENSE",
            kind="IN",
            declaring_module="astrocyte_modulation",
            join_policy="sum",
        ),
    )
    manifest.declare_edge(
        "astrocyte_modulation", "glial_gain_out", "astrocyte_modulation", "synaptic_gain_in"
    )


#: the declared self-edge, exactly as the manifest carries it -- the charge evidence cites it
GLIA_EDGE = ("astrocyte_modulation", "glial_gain_out", "astrocyte_modulation", "synaptic_gain_in")


class VolumeTransmitter:
    """The diffuse channel's first live consumer (specs/022 T027): a wrapped-synapse relay
    that routes a fixed presynaptic drive through the RECEIVED glial gain field and reports
    the downstream LIF spike count per tick. It owns no astrocyte -- the gain arrives on
    the declared channel, which is what makes the neuter meaningful: pin the field to 1.0
    and the relay must fall back to baseline transmission."""

    def __init__(self, domain=0, drive=0.11, window=30):
        self.domain = domain
        self.drive = drive
        self.window = window  # LIF integration window per delivered gain frame

    # distinct-from: sensorimotor/closed_loop.py -- the arm's signal->command loop; and
    # distinct-from: substrate/cockpit.py -- an HTTP handler test shim. Neither scales a
    #   synaptic current by a received glial gain; phrase overlap only. Read.
    def transmit(self, gain_vec):
        """Scale the fixed presynaptic current through modulated_transmission() using this
        domain's RECEIVED gain, run the LIF window, return its spike count."""
        gain = float(gain_vec[self.domain])
        current = modulated_transmission(gain, self.drive)
        return sum(LIFNeuron().run([current] * self.window))


def _glia_consumer_factory():
    """Fresh ring + JoinConsumer over the declared `synaptic_gain_in` port, per replay arm."""
    from snn_data_lake.substrate.signal_ring import JoinConsumer, SignalRing
    from snn_data_lake.substrate.wiring_manifest import WiringManifest

    m = WiringManifest()
    declare_ports(m)
    port = m.ports[("astrocyte_modulation", "synaptic_gain_in")]
    ring = SignalRing(slots=128)
    return ring, JoinConsumer(ring, port)


def consume_gain_field(consumer):
    """The downstream computation under measurement: each delivered gain frame drives the
    VolumeTransmitter relay; returns [(tick, spikes), ...]. LOSS IS FATAL (the T026
    ring-overrun lesson: a measurement over a partial record is a different experiment)."""
    vt = VolumeTransmitter()
    out = []
    joined, lost = consumer.poll_joined()
    assert not lost, f"charge measurement lost {lost} frames -- ring overrun, never smooth this"
    for tick, vec in joined:
        out.append((tick, vt.transmit(vec)))
    return out


def _gain_frames(n_ticks=80, engage_at=10):
    """Recorded frames from the REAL astrocyte: quiet until `engage_at`, then sustained
    activity drives Ca past threshold and the gain field engages. n_ticks=80 because the
    glia are SLOW by construction (tau_ca = 50; the first 40-tick schedule left Ca at
    ~0.45 < the 0.5 threshold and the measurement honestly REFUSED -- the refusal
    criterion working). -> (frames, engage_at)."""
    astro = Astrocyte(n_domains=2)
    frames = []
    for t in range(1, n_ticks + 1):
        gains = astro.step([1.0 if t >= engage_at else 0.0, 0.0])
        frames.append((t, "astrocyte_modulation", list(gains)))
    return frames, engage_at


def _measure_glia_charge():
    """The replay-twice A/B: live gain field vs a pinned unmodulated field [1.0, 1.0].
    -> (live, pinned_out, reads, responded): `responded` = spikes in the settled tail
    exceed the pre-engagement baseline (the glial field, not the schedule, moved
    transmission)."""
    from snn_data_lake.instruments.port_charge_probe import replay_twice

    frames, engage_at = _gain_frames()
    n = frames[-1][0]
    live, pinned_out, reads = replay_twice(
        frames, _glia_consumer_factory, consume_gain_field, [1.0, 1.0]
    )
    pre = max(s for t, s in live if t < engage_at)
    post = max(s for t, s in live if t > n - 15)  # the settled tail, past the slow Ca ramp
    return live, pinned_out, reads, post > pre


def record_charge():
    """Measure the A/B and record the attribution via THE one writer. REFUSES a
    non-diverged or non-responding run -- evidence is a measurement, never a declaration."""
    from snn_data_lake.instruments.port_charge_probe import record_attribution

    live, pinned_out, reads, responded = _measure_glia_charge()
    if live == pinned_out or not responded:
        print(f"record_charge REFUSED for {GLIA_EDGE}: no divergence or no response")
        return 1
    path = record_attribution(
        GLIA_EDGE,
        {"reads": reads, "n_frames": len(live)},
        "replay_twice: the real astrocyte's engaging gain field vs a pinned unmodulated "
        "[1.0, 1.0]; downstream LIF spike counts diverged and rose after Ca crossed "
        "threshold -- the diffuse channel, not the schedule, moved transmission; "
        "self-check re-measures and requires `measured` to reproduce",
    )
    print(f"recorded {GLIA_EDGE} into {path}")
    return 0


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
    # ---- THE DIFFUSE CHANNEL IS CHARGED, measured (specs/022 T027) ---------------------
    from snn_data_lake.instruments.port_charge_probe import read_attribution

    live, pinned_out, reads, responded = _measure_glia_charge()
    assert reads == len(live), f"the neuter must be READ every frame ({reads} vs {len(live)})"
    assert live != pinned_out, (
        "pinning the gain field to unmodulated [1.0, 1.0] must change downstream "
        "transmission -- an undiverged channel is a declared, not a charged, port"
    )
    assert responded, (
        "engagement must RAISE downstream spikes over the pre-engagement baseline -- "
        "divergence without response attributes nothing"
    )
    rec = read_attribution(GLIA_EDGE)
    assert rec is not None and rec.get("diverged") is True, (
        f"committed charge evidence must carry the diverged {GLIA_EDGE} attribution -- "
        "regenerate with --record-charge and commit it"
    )
    fresh = {"reads": reads, "n_frames": len(live)}
    assert rec.get("measured", {}) == fresh, (
        f"recorded charge evidence does not reproduce: {rec.get('measured')} vs {fresh} -- "
        "regenerate with --record-charge in the same commit"
    )
    print(
        f"self-check OK: sustained {sus_peak_g:.2f} engages vs brief {brief_peak_g:.2f} ~ 1; "
        f"persists {persist_g:.2f} +{n_persist}; A {gain_active:.2f} != B {gain_quiet:.2f}; "
        f"LIF {base_spikes}->{mod_spikes} spikes; deterministic. DIFFUSE CHANNEL CHARGED "
        f"(measured): live vs pinned diverged over {reads} read frames and engagement "
        f"raised transmission; committed attribution reproduces."
    )


if __name__ == "__main__":
    import sys as _sys

    if "--record-charge" in _sys.argv[1:]:
        _sys.exit(record_charge())
    main()

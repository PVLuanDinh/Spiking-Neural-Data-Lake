"""
Thalamic reticular nucleus (TRN) -- a GENUINELY SPIKING GABAergic attentional gate on thalamic relay.

The thalamus is a RELAY, but the TRN is what makes it a GATE: a thin shell of inhibitory (GABAergic)
neurons wrapped over the relay that, under cortico-reticular / global-workspace feedback, SUPPRESSES the
non-attended relay channels while letting the attended channel(s) pass. This is Crick's "searchlight" of
attention implemented as selective gain control -- an attentional GAIN/relay-gating FUNCTION, NOT
awareness or consciousness.

This is a SPIKING CIRCUIT (leaky integrate-and-fire neurons, reusing cortex.lif_neuron.LIFNeuron), not a
rate equation. Gain control EMERGES from spiking feedback inhibition:

  1. One LIF RELAY neuron per channel receives a top-down excitatory drive (attended == full drive, the
     rest a low baseline). Left alone every driven relay would fire.
  2. An inhibitory TRN POOL of LIF neurons with a spread of thresholds reads the relays' summed spiking
     (a leaky trace of the collateral activity). Because their thresholds differ, the NUMBER of TRN
     neurons recruited scales smoothly with total relay firing -- a graded population inhibition built
     from binary spikes.
  3. The pooled TRN spikes feed back as a divisive inhibitory current onto every relay -- EXCEPT the
     attended channel, which top-down attention DIS-INHIBITS (its share of the TRN gate is lifted). So
     the attended relay fires freely while the rest are inhibited toward silence.
  4. The feedback closes the loop: more competing relay spikes -> more TRN neurons recruited -> more
     inhibition -> each channel's firing is pushed back down. Total relay throughput is normalized and a
     channel's rate is DIVIDED by the pool's activity -- a spiking realization of divisive normalization
     (Carandini & Heeger's inhibitory-pool feedback), so adding distractors lowers per-channel throughput.

Throughput is read from RELAY SPIKE COUNTS. `mask()` turns those spike rates into per-channel relay gains
(summing to a fixed budget, so the gate only REDISTRIBUTES gain) -- now sourced from spikes, not arithmetic.

Fair control (un-riggable -- same noisy input, same total gain budget in all three conditions):
  (a) gate OPEN  : uniform gain 1.0 on every channel (the plain, undifferentiated relay).
  (b) TRN gating : the attended-channel spike-derived mask above.
  (c) RANDOM mask: the SAME gain VALUES as (b) permuted onto random channels (identical gain
      distribution and identical total energy -- only the ASSIGNMENT is randomized).
  Readout signal-to-distractor ratio (SDR) = attended-channel activity / mean distractor activity.
  Attended gating must RAISE SDR over the open gate; the random mask (same energy, wrong channels)
  must NOT -- proving the benefit is SELECTIVITY, not merely applying gain/energy anywhere.

Honest framing: a reduced, spiking model of TRN attentional gain-gating on synthetic noisy channels. It
is additive/opt-in and models a relay-gating FUNCTION, not cognition/awareness/AGI. The normalization is
emergent feedback, not an exact algebraic divide; it is a small LIF circuit, not a full cortical model.
Credit: Crick 1984 (searchlight); Carandini & Heeger 2012 (normalization); McAlonan et al. 2006 (TRN attention).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.thalamus.reticular_gate
"""
import random
import sys

from snn_data_lake.cortex.lif_neuron import LIFNeuron


class ReticularGate:
    """Spiking TRN attentional gate: per-channel LIF relay neurons gated by an inhibitory TRN pool whose
    spiking feedback normalizes relay throughput (a spiking divisive normalization); attention dis-inhibits
    the attended relay so its spikes pass while distractor channels are inhibited toward silence."""

    def __init__(self, n_channels=16, beta=0.6, sigma=0.5, exponent=2.0, contrast=0.55, seed=7):
        self.n = n_channels
        self.beta = beta                       # relay->TRN feedforward gain (drives the inhibitory pool)
        self.sigma = sigma                     # relay excitatory drive scale (top-down drive per unit attention)
        self.exponent = exponent               # attention -> dis-inhibition sharpness on the attended relay
        self.contrast = contrast               # non-attended attention drive (attended == 1.0)
        self.budget = float(n_channels)        # total gain; the OPEN gate is 1.0 per channel
        # --- internal spiking-circuit constants (not part of the public signature) ---
        self.steps = 80                        # timesteps to run the relay/TRN loop
        self.g_inhib = 2.5                     # strength of TRN feedback inhibition onto the relays
        self.leak_r = 0.7                      # leak of the relay-activity trace feeding the TRN pool
        self.leak_i = 0.3                      # leak of the pooled inhibitory current onto the relays
        self.tau_relay = 10.0                  # relay LIF membrane time constant
        self.refr_relay = 2                    # relay refractory (spaces relay spikes)
        self.tau_trn = 1.2                     # TRN LIF time constant (fast: near-instant pool read-out)
        self.vt_lo = 0.5                       # lowest TRN threshold (first neurons recruited)
        self.vt_hi = 3.0                       # highest TRN threshold (recruited only under heavy activity)
        self.rng = random.Random(seed)

    def attention(self, attended):
        """Cortico-reticular / global-workspace feedback: drive 1.0 on the attended channel(s),
        a low baseline elsewhere (this module takes the attended set as input; it does not pick)."""
        att = set(attended if isinstance(attended, (list, tuple, set)) else [attended])
        return [1.0 if i in att else self.contrast for i in range(self.n)]

    def gate(self, attention, steps=None):
        """Run the spiking TRN gate over `steps` timesteps and return (rates, relay_spikes, trn_spikes).

        rates[i]       = relay channel i's spikes / step (its throughput).
        relay_spikes   = per-channel list of 0/1 relay spike trains.
        trn_spikes     = per-neuron list of 0/1 inhibitory TRN spike trains.

        Each relay LIF neuron gets top-down excitatory drive sigma*attention[i]. An inhibitory TRN pool of
        LIF neurons with spread thresholds reads a leaky trace of total relay spiking; the count recruited
        scales with pool activity (graded population inhibition from binary spikes). That pooled inhibition
        feeds back divisively onto every relay, lifted on the attended channel via dis-inhibition. No RNG:
        LIF dynamics are deterministic."""
        steps = self.steps if steps is None else steps
        relays = [LIFNeuron(tau_m=self.tau_relay, refractory=self.refr_relay) for _ in range(self.n)]
        m = self.n                                             # TRN pool size (a shell over the relays)
        trn = [LIFNeuron(tau_m=self.tau_trn, v_threshold=self.vt_lo + (self.vt_hi - self.vt_lo) * j / (m - 1))
               for j in range(m)]
        amax = max(attention) if attention else 1.0
        disinhib = [(a / amax) ** self.exponent for a in attention]   # attended ~1 (released); distractors ~0
        counts = [0] * self.n
        relay_spikes = [[] for _ in range(self.n)]
        trn_spikes = [[] for _ in range(m)]
        inhib = 0.0                                            # pooled inhibitory current onto the relays
        relay_trace = 0.0                                      # leaky trace of total relay spiking
        for _ in range(steps):
            pool = 0                                           # how many TRN neurons the pool recruits now
            for j in range(m):
                sj = trn[j].step(self.beta * relay_trace)      # TRN reads the relay collateral trace
                trn_spikes[j].append(sj)
                pool += sj
            inhib = self.leak_i * inhib + pool / m             # graded, leaky inhibitory drive in [0, ~1]
            s_sum = 0
            for i in range(self.n):
                drive = self.sigma * attention[i]              # top-down excitatory drive to the relay
                gated_inhib = self.g_inhib * inhib * (1.0 - disinhib[i])   # TRN gate, lifted on the attended
                s = relays[i].step(drive - gated_inhib)
                relay_spikes[i].append(s)
                counts[i] += s
                s_sum += s
            relay_trace = self.leak_r * relay_trace + s_sum    # feed this step's relay spikes back to the TRN
        rates = [c / steps for c in counts]
        return rates, relay_spikes, trn_spikes

    def mask(self, attention):
        """Attention drive -> per-channel relay GAINS (sum == budget). Runs the spiking TRN gate and reads
        each channel's relay SPIKE RATE as its throughput, then normalizes those rates to the fixed gain
        budget so the gate only REDISTRIBUTES gain -- now via spiking feedback inhibition. The attended
        relay (TRN-dis-inhibited) spikes; distractor relays are inhibited toward silence."""
        rates, _, _ = self.gate(attention)
        tot = sum(rates)
        if tot <= 0.0:                                         # degenerate: winner-take-all fallback
            w = attention.index(max(attention))
            m = [1.0 if i == w else 0.0 for i in range(self.n)]
            tot = 1.0
            return [self.budget * mi / tot for mi in m]
        return [self.budget * r / tot for r in rates]          # redistribute the FIXED gain budget

    def channels(self, signal, steps=48):
        """Synthetic noisy multi-channel input. Every channel (signal and distractors alike) is
        drawn from the SAME magnitude distribution -- magnitude alone gives no cue to which is the
        signal; only the attention drive knows. Returns per-channel mean rectified activity."""
        raw = []
        for _ in range(self.n):
            samples = [max(0.0, 1.0 + self.rng.uniform(-0.25, 0.25)) for _ in range(steps)]
            raw.append(sum(samples) / steps)
        # keep the channel index `signal` merely labelled as the signal; do not privilege its magnitude
        return raw

    @staticmethod
    def sdr(gains, raw, signal):
        """Signal-to-distractor ratio at the readout: gated attended activity / mean gated distractor."""
        n = len(raw)
        sig = gains[signal] * raw[signal]
        dis = [gains[i] * raw[i] for i in range(n) if i != signal]
        return sig / (sum(dis) / len(dis))

    def random_sdr(self, gains, raw, signal, trials=400):
        """Mean SDR when the SAME gain values are permuted onto random channels (equal total energy,
        random assignment). Averaged over many shuffles so no single lucky/unlucky draw is cherry-picked."""
        vals = []
        for _ in range(trials):
            perm = gains[:]
            self.rng.shuffle(perm)
            vals.append(self.sdr(perm, raw, signal))
        return sum(vals) / len(vals)


def main():
    signal = 5
    gate = ReticularGate(seed=7)
    raw = gate.channels(signal)
    att = gate.attention(signal)

    open_gains = [gate.budget / gate.n] * gate.n     # (a) uniform gain 1.0 -- the open relay
    rates, relay_spikes, trn_spikes = gate.gate(att)  # the spiking TRN run behind (b)
    trn_gains = gate.mask(att)                        # (b) attended-channel TRN gating (spike-derived)

    sdr_open = gate.sdr(open_gains, raw, signal)
    sdr_trn = gate.sdr(trn_gains, raw, signal)
    sdr_rand = gate.random_sdr(trn_gains, raw, signal)   # (c) same gains, random channels

    raw_sdr = gate.sdr([1.0] * gate.n, raw, signal)      # magnitude-only SDR of the raw input

    # preservation / attenuation, measured against the open gate at the SAME total gain budget:
    sig_open = open_gains[signal] * raw[signal]
    sig_trn = trn_gains[signal] * raw[signal]
    dist_open = sum(open_gains[i] * raw[i] for i in range(gate.n) if i != signal) / (gate.n - 1)
    dist_trn = sum(trn_gains[i] * raw[i] for i in range(gate.n) if i != signal) / (gate.n - 1)

    # relay spike rates: attended fires, distractors are inhibited toward silence
    r_att = rates[signal]
    r_dist = sum(rates[i] for i in range(gate.n) if i != signal) / (gate.n - 1)

    # spiking DIVISIVE NORMALIZATION: pick a distractor probe; adding competitors recruits more of the TRN
    # pool (more feedback inhibition) and pushes that channel's own relay firing DOWN.
    probe = 0
    att_few = [1.0 if i == signal else (gate.contrast if i == probe else 0.0) for i in range(gate.n)]
    att_many = att                                       # attended + every distractor driven
    rates_few, _, trn_few = gate.gate(att_few)
    rates_many, _, trn_many = gate.gate(att_many)
    probe_few, probe_many = rates_few[probe], rates_many[probe]
    trn_pool_few = sum(sum(t) for t in trn_few)          # total inhibitory spikes with few competitors
    trn_pool_many = sum(sum(t) for t in trn_many)        # ... with many competitors (feedback scales up)

    print("=" * 66)
    print("TRN RETICULAR GATE -- SPIKING thalamic attentional gain gate (LIF relay + TRN pool)")
    print("=" * 66)
    print(f"channels = {gate.n}, attended (signal) channel = {signal}, total gain budget = {gate.budget:.0f}")
    print(f"raw input SDR (magnitude only, no cue): {raw_sdr:.2f}  (channels are indistinguishable by size)")
    print("-" * 66)
    print(f"relay firing  : attended {r_att:.3f} spikes/step vs distractors ~{r_dist:.3f} (inhibited toward silence)")
    print(f"TRN pool      : {trn_pool_few} inhibitory spikes (few competitors) -> {trn_pool_many} (many); feedback scales")
    print("-" * 66)
    print(f"(a) gate OPEN  / uniform gain : SDR = {sdr_open:.2f}   [gain 1.00 on every channel]")
    print(f"(b) TRN attended-gate         : SDR = {sdr_trn:.2f}   [signal gain {trn_gains[signal]:.2f}, distractors ~{trn_gains[0]:.2f}]")
    print(f"(c) RANDOM mask (same gains)  : SDR = {sdr_rand:.2f}   [same gain values, random channels]")
    print("-" * 66)
    print(f"attended channel PRESERVED : gated activity {sig_open:.3f} (open) -> {sig_trn:.3f} (TRN, x{sig_trn / sig_open:.2f})")
    print(f"distractors ATTENUATED     : mean gated {dist_open:.3f} (open) -> {dist_trn:.3f} (TRN, x{dist_trn / dist_open:.2f})")
    print(f"divisive (spiking)         : probe relay {probe_few:.3f} (few) -> {probe_many:.3f} (many competitors)")
    print("=" * 66)

    # ---- self-check: deterministic asserts, zero-dep, ASCII-only ----
    # (b) GENUINELY SPIKING: discrete 0/1 spikes from LIF relay + inhibitory TRN pool; no numpy/torch.
    all_relay = [s for tr in relay_spikes for s in tr]
    all_trn = [s for tr in trn_spikes for s in tr]
    assert set(all_relay) <= {0, 1} and set(all_trn) <= {0, 1}, "relay/TRN activity must be discrete 0/1 spikes"
    assert sum(all_relay) > 0 and sum(all_trn) > 0, "both the relay neurons and the inhibitory TRN pool must spike"
    assert "numpy" not in sys.modules and "torch" not in sys.modules, "must be pure stdlib (no numpy/torch)"

    # (a) FUNCTION: attended relay passes (high spike rate), distractors inhibited toward silence.
    assert r_att > 3.0 * r_dist, f"attended relay must fire far above distractors ({r_att:.3f} vs {r_dist:.3f})"
    assert r_att > 0.1 and r_dist < 0.06, f"attended must spike, distractors near-silence ({r_att:.3f}, {r_dist:.3f})"

    # (a) DIVISIVE via SPIKING inhibition: adding distractors recruits more TRN pool spiking and pushes the
    # probe channel's own relay firing down -- its response is divided by the pool's activity.
    assert trn_pool_many > trn_pool_few, f"more competitors must recruit more TRN inhibition ({trn_pool_many} > {trn_pool_few})"
    assert probe_many < probe_few, f"adding distractors must lower per-channel throughput ({probe_many:.3f} < {probe_few:.3f})"
    assert probe_few > 0.0, "the probe channel must fire when it has few competitors (headroom to be suppressed)"

    # gate/normalize contract: attended preserved, distractors attenuated, budget only redistributed.
    assert abs(raw_sdr - 1.0) < 0.15, f"raw channels must be ~indistinguishable by magnitude (SDR {raw_sdr:.2f})"
    assert sdr_trn > 2.0 * sdr_open, f"attended gating must raise SDR over the open gate ({sdr_trn:.2f} vs {sdr_open:.2f})"
    assert sdr_trn > 2.0 * sdr_rand, f"attended gating must beat the random mask ({sdr_trn:.2f} vs {sdr_rand:.2f})"
    assert sdr_rand < 1.5 * sdr_open, f"random mask (same energy, wrong channels) must NOT improve on the open gate ({sdr_rand:.2f} vs {sdr_open:.2f})"
    assert sig_trn > sig_open, f"the attended channel's signal must be PRESERVED/boosted ({sig_trn:.3f} > {sig_open:.3f})"
    assert dist_trn < dist_open, f"distractor channels must be ATTENUATED ({dist_trn:.3f} < {dist_open:.3f})"
    assert abs(sum(trn_gains) - gate.budget) < 1e-9, "TRN mask must only redistribute the fixed gain budget"

    # (c) DETERMINISM: same seed -> identical channels, spike-derived mask, and random SDR.
    a = ReticularGate(seed=7)
    b = ReticularGate(seed=7)
    ra, rb = a.channels(signal), b.channels(signal)
    ga, gb = a.mask(a.attention(signal)), b.mask(b.attention(signal))
    sa = a.random_sdr(ga, ra, signal)
    sb = b.random_sdr(gb, rb, signal)
    assert ra == rb and ga == gb and sa == sb, "same seed -> identical channels, mask and random SDR"

    print(f"self-check OK: SPIKING gate -- attended relay {r_att:.2f} vs distractors {r_dist:.2f} spikes/step; "
          f"open SDR {sdr_open:.2f} -> attended SDR {sdr_trn:.2f} (x{sdr_trn / sdr_open:.1f}); "
          f"random (same gains) {sdr_rand:.2f} ~= open; TRN feedback divisive probe {probe_few:.2f}->{probe_many:.2f}; "
          f"signal x{sig_trn / sig_open:.1f} preserved, distractors x{dist_trn / dist_open:.2f}; deterministic")


if __name__ == "__main__":
    main()

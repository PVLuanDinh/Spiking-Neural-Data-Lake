"""
Thalamic reticular nucleus (TRN) -- the GABAergic attentional gain gate on thalamic relay.

The thalamus is a RELAY, but the TRN is what makes it a GATE: a thin shell of inhibitory
(GABAergic) neurons wrapped over the relay that, under cortico-reticular / global-workspace
feedback, SUPPRESSES the non-attended relay channels while letting the attended channel(s)
pass. This is Crick's "searchlight" of attention implemented as selective gain control -- it
models an attentional GAIN/relay-gating FUNCTION, NOT awareness or consciousness.

Mechanism (rate-level, deterministic):
  1. An attention drive vector (from cortex / the global workspace -- passed in as an arg; this
     module does NOT decide what to attend to) marks the attended channel(s).
  2. Divisive normalization (reused from cortex.divisive_normalization) turns that drive into a
     bounded salience over the channel pool -- the attended channel dominates the pool.
  3. A LATERAL-INHIBITION surround (the TRN's mutual GABAergic inhibition) divides each channel's
     salience by its pool context, m_i = sal_i / (1 + beta * (pool - sal_i)), pushing the
     non-attended channels down further while the attended channel is spared.
  4. The mask is renormalized to a FIXED total gain budget, so gating only REDISTRIBUTES gain --
     it never injects extra energy. The relay output is per-channel gain x the channel's signal.

Fair control (un-riggable -- same noisy input, same total gain budget in all three conditions):
  (a) gate OPEN  : uniform gain 1.0 on every channel (the plain, undifferentiated relay).
  (b) TRN gating : the attended-channel mask above.
  (c) RANDOM mask: the SAME gain VALUES as (b) permuted onto random channels (identical gain
      distribution and identical total energy -- only the ASSIGNMENT is randomized).
  Readout signal-to-distractor ratio (SDR) = attended-channel activity / mean distractor activity.
  Attended gating must RAISE SDR over the open gate; the random mask (same energy, wrong channels)
  must NOT -- proving the benefit is SELECTIVITY, not merely applying gain/energy anywhere.

Honest framing: a reduced, rate-level model of TRN attentional gain-gating on synthetic noisy
channels. It is additive/opt-in and models a relay-gating FUNCTION, not cognition/awareness/AGI.
Credit: Crick 1984 (searchlight); Carandini & Heeger 2012 (normalization); McAlonan et al. 2006 (TRN attention).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.thalamus.reticular_gate
"""
import random

from snn_data_lake.cortex.divisive_normalization import DivisiveNormalization


class ReticularGate:
    """TRN attentional gain gate: divisive-normalized salience + lateral inhibition -> relay gains."""

    def __init__(self, n_channels=16, beta=0.6, sigma=0.5, exponent=2.0, contrast=0.55, seed=7):
        self.n = n_channels
        self.beta = beta                       # TRN lateral-inhibition surround strength
        self.contrast = contrast               # non-attended attention drive (attended == 1.0)
        self.budget = float(n_channels)        # total gain; the OPEN gate is 1.0 per channel
        self.dn = DivisiveNormalization(sigma=sigma, n=exponent)
        self.rng = random.Random(seed)

    def attention(self, attended):
        """Cortico-reticular / global-workspace feedback: drive 1.0 on the attended channel(s),
        a low baseline elsewhere (this module takes the attended set as input; it does not pick)."""
        att = set(attended if isinstance(attended, (list, tuple, set)) else [attended])
        return [1.0 if i in att else self.contrast for i in range(self.n)]

    def mask(self, attention):
        """Attention drive -> TRN relay gains (sum == budget). Divisive-normalized salience passed
        through a lateral-inhibition surround that divides each channel by its pool context, then
        renormalized to the fixed gain budget so the gate only REDISTRIBUTES gain."""
        sal = self.dn.normalize(attention)                     # bounded salience pool; attended leads
        pool = sum(sal)
        m = [s / (1.0 + self.beta * (pool - s)) for s in sal]  # TRN lateral-inhibition surround
        tot = sum(m)
        if tot <= 0.0:                                         # degenerate: winner-take-all fallback
            w = sal.index(max(sal))
            m = [1.0 if i == w else 0.0 for i in range(self.n)]
            tot = 1.0
        return [self.budget * mi / tot for mi in m]            # redistribute the FIXED gain budget

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
    trn_gains = gate.mask(att)                        # (b) attended-channel TRN gating

    sdr_open = gate.sdr(open_gains, raw, signal)
    sdr_trn = gate.sdr(trn_gains, raw, signal)
    sdr_rand = gate.random_sdr(trn_gains, raw, signal)   # (c) same gains, random channels

    raw_sdr = gate.sdr([1.0] * gate.n, raw, signal)      # magnitude-only SDR of the raw input

    # preservation / attenuation, measured against the open gate at the SAME total gain budget:
    sig_open = open_gains[signal] * raw[signal]
    sig_trn = trn_gains[signal] * raw[signal]
    dist_open = sum(open_gains[i] * raw[i] for i in range(gate.n) if i != signal) / (gate.n - 1)
    dist_trn = sum(trn_gains[i] * raw[i] for i in range(gate.n) if i != signal) / (gate.n - 1)

    print("=" * 66)
    print("TRN RETICULAR GATE -- thalamic attentional gain gate (relay + gate)")
    print("=" * 66)
    print(f"channels = {gate.n}, attended (signal) channel = {signal}, total gain budget = {gate.budget:.0f}")
    print(f"raw input SDR (magnitude only, no cue): {raw_sdr:.2f}  (channels are indistinguishable by size)")
    print("-" * 66)
    print(f"(a) gate OPEN  / uniform gain : SDR = {sdr_open:.2f}   [gain 1.00 on every channel]")
    print(f"(b) TRN attended-gate         : SDR = {sdr_trn:.2f}   [signal gain {trn_gains[signal]:.2f}, distractors ~{trn_gains[0]:.2f}]")
    print(f"(c) RANDOM mask (same gains)  : SDR = {sdr_rand:.2f}   [same gain values, random channels]")
    print("-" * 66)
    print(f"attended channel PRESERVED : gated activity {sig_open:.3f} (open) -> {sig_trn:.3f} (TRN, x{sig_trn / sig_open:.2f})")
    print(f"distractors ATTENUATED     : mean gated {dist_open:.3f} (open) -> {dist_trn:.3f} (TRN, x{dist_trn / dist_open:.2f})")
    print("=" * 66)

    # ---- self-check: deterministic asserts, zero-dep, ASCII-only ----
    assert abs(raw_sdr - 1.0) < 0.15, f"raw channels must be ~indistinguishable by magnitude (SDR {raw_sdr:.2f})"
    assert sdr_trn > 2.0 * sdr_open, f"attended gating must raise SDR over the open gate ({sdr_trn:.2f} vs {sdr_open:.2f})"
    assert sdr_trn > 2.0 * sdr_rand, f"attended gating must beat the random mask ({sdr_trn:.2f} vs {sdr_rand:.2f})"
    assert sdr_rand < 1.5 * sdr_open, f"random mask (same energy, wrong channels) must NOT improve on the open gate ({sdr_rand:.2f} vs {sdr_open:.2f})"
    assert sig_trn > sig_open, f"the attended channel's signal must be PRESERVED/boosted ({sig_trn:.3f} > {sig_open:.3f})"
    assert dist_trn < dist_open, f"distractor channels must be ATTENUATED ({dist_trn:.3f} < {dist_open:.3f})"
    assert abs(sum(trn_gains) - gate.budget) < 1e-9, "TRN mask must only redistribute the fixed gain budget"

    a = ReticularGate(seed=7)
    b = ReticularGate(seed=7)
    ra, rb = a.channels(signal), b.channels(signal)
    ga, gb = a.mask(a.attention(signal)), b.mask(b.attention(signal))
    sa = a.random_sdr(ga, ra, signal)
    sb = b.random_sdr(gb, rb, signal)
    assert ra == rb and ga == gb and sa == sb, "same seed -> identical channels, mask and random SDR"

    print(f"self-check OK: open SDR {sdr_open:.2f} -> attended SDR {sdr_trn:.2f} (x{sdr_trn / sdr_open:.1f}); "
          f"random (same gains) {sdr_rand:.2f} ~= open (selectivity, not gain, is the win); "
          f"signal x{sig_trn / sig_open:.1f} preserved, distractors x{dist_trn / dist_open:.2f} attenuated; deterministic")


if __name__ == "__main__":
    main()

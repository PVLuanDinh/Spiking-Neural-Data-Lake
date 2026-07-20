"""
v2.63 — Burst multiplexing: ONE spike train carries TWO independent signals (Naud & Sprekeler 2018).

Rate/latency coding (spike_preprocessing) and time-to-first-spike (temporal_coding_storage) each read ONE
scalar per neuron off a spike train. Naud & Sprekeler showed a single train has room for TWO independent
signals at once -- a MULTIPLEXED neural code:
  - channel A rides the EVENT RATE   : how often the neuron fires an event (an isolated spike OR a burst onset);
  - channel B rides the BURST FRACTION: of those events, what fraction are BURSTS (>=2 tight spikes) vs singles.
Event rate and burst probability are (approximately) independent degrees of freedom of the same train, so a
downstream reader can pull BOTH signals back out with little crosstalk. Biologically: somatic drive sets the
event rate, apical/dendritic drive sets the burst probability -- two input streams, one axon.

This distills that idea to a non-riggable honesty check. TWO genuinely independent source signals A(t), B(t)
(sinusoids at DIFFERENT integer frequencies -> orthogonal over the record, phases from the seeded rng; B is
NOT derived from A) are encoded into a single spike-time train; a decoder that sees ONLY the spike times
re-detects events and bursts and reconstructs A and B. The defining test is LOW CROSSTALK: each decoded
channel must track ITS OWN source far better than the other. The decoder uses only a FIXED affine rescale by
the code's KNOWN range (never a fit to the true signals; Pearson correlation is affine-invariant anyway).
On-thesis with spike_preprocessing / temporal_coding_storage / rank_order_coding -- a spike-CODING primitive,
not a plasticity rule. Pure standard library, deterministic (one per-instance random.Random threaded through
signals AND spike generation), ASCII-only.

Reference: Naud, R. & Sprekeler, H. (2018), "Sparse bursts optimize information transmission in a multiplexed
neural code", PNAS 115(27):E6329-E6338, DOI 10.1073/pnas.1720995115.

  python research/burst_multiplexing.py        # (also run by run_selfchecks.py, root on PYTHONPATH)
"""
import math
import random

# --- code constants: the KNOWN range of the encoder. The decoder rescales by THESE only --
#     (a fixed affine map from the code's own range) -- never by any fit to the true signals.
E_LO, E_HI = 0.15, 0.75      # per-bin event probability spans this as channel A goes 0 -> 1
B_LO, B_HI = 0.10, 0.90      # per-event burst probability spans this as channel B goes 0 -> 1
BURST_SIZE = 3               # spikes packed into one burst
SLOT = 10.0                  # time units between successive bin onsets
INTRA = 1.0                  # intra-burst spike spacing (<< SLOT, so a burst reads as one clustered event)
GROUP_GAP = 3.0              # decoder: spikes closer than this belong to the SAME event (INTRA < GAP < SLOT)


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / (math.sqrt(vx * vy) or 1.0)


def _sine_signal(rng, n, cycles, amp=0.4, center=0.5):
    """A slow bounded signal in [center-amp, center+amp]. Integer `cycles` -> distinct-frequency sinusoids
    are orthogonal over the record, so two of them are genuinely independent (not just in expectation);
    the phase is drawn from the seeded rng."""
    phase = rng.uniform(0.0, 2.0 * math.pi)
    return [center + amp * math.sin(2.0 * math.pi * cycles * i / n + phase) for i in range(n)]


def encode(a, b, bins, rng):
    """Emit a spike-time train for one window given scalar signals a, b in [0,1].
    Each bin fires an EVENT with prob affine(a); an event is a BURST with prob affine(b), else a single spike.
    The reader downstream sees ONLY the returned spike times -- not a, b, or the event/burst bookkeeping."""
    p_event = E_LO + a * (E_HI - E_LO)
    p_burst = B_LO + b * (B_HI - B_LO)
    spikes = []
    for bin_i in range(bins):
        if rng.random() < p_event:
            t0 = bin_i * SLOT
            if rng.random() < p_burst:
                spikes.extend(t0 + k * INTRA for k in range(BURST_SIZE))   # burst packet
            else:
                spikes.append(t0)                                          # single spike
    return spikes


def decode(spikes):
    """From spike TIMES alone, group spikes into events (a burst if a group has >=2 spikes).
    Returns (n_events, n_bursts). No peeking at the source signals."""
    if not spikes:
        return 0, 0
    events, bursts, size = 0, 0, 1
    for prev, cur in zip(spikes, spikes[1:]):
        if cur - prev <= GROUP_GAP:
            size += 1                       # same event (burst continues)
        else:
            events += 1
            bursts += size >= 2
            size = 1
    events += 1                             # close the final group
    bursts += size >= 2
    return events, bursts


def _rescale(value, lo, hi):
    """Fixed affine map from the code's KNOWN range back to [0,1] (not fit to any true signal)."""
    return min(1.0, max(0.0, (value - lo) / (hi - lo)))


def simulate(seed, n_windows=300, bins=80):
    """Full multiplex round-trip. Returns (correlations, decoded/true traces)."""
    rng = random.Random(seed)
    a_true = _sine_signal(rng, n_windows, cycles=5)      # channel A source
    b_true = _sine_signal(rng, n_windows, cycles=7)      # channel B source -- different freq -> independent
    a_dec, b_dec = [], []
    for a, b in zip(a_true, b_true):
        n_events, n_bursts = decode(encode(a, b, bins, rng))
        rate_est = n_events / bins                       # -> channel A carrier
        frac_est = (n_bursts / n_events) if n_events else 0.0   # -> channel B carrier
        a_dec.append(_rescale(rate_est, E_LO, E_HI))
        b_dec.append(_rescale(frac_est, B_LO, B_HI))
    corr = {
        "aa": _pearson(a_dec, a_true),   # channel A vs its own signal
        "ab": _pearson(a_dec, b_true),   # channel A crosstalk (vs the OTHER signal)
        "bb": _pearson(b_dec, b_true),   # channel B vs its own signal
        "ba": _pearson(b_dec, a_true),   # channel B crosstalk (vs the OTHER signal)
        "src": _pearson(a_true, b_true),  # independence of the two SOURCE signals
    }
    return corr, (a_true, b_true, a_dec, b_dec)


def main():
    seed = 7
    c, _ = simulate(seed)
    aa, ab, bb, ba, src = c["aa"], c["ab"], c["bb"], c["ba"], c["src"]

    print("=" * 68)
    print("BURST MULTIPLEXING -- one spike train, two independent signals (Naud-Sprekeler 2018)")
    print("=" * 68)
    print(f"sources A,B: independent sinusoids (5 vs 7 cycles); corr(A_true,B_true) = {src:+.3f}")
    print("decode from spike TIMES only; fixed affine rescale by the code's known range (no signal fit)")
    print("-" * 68)
    print(f"{'channel':<9}| {'carrier':<15}| {'corr(own signal)':<17}| corr(other = crosstalk)")
    print("-" * 68)
    print(f"{'A':<9}| {'event-rate':<15}| {aa:<17.3f}| {ab:+.3f}")
    print(f"{'B':<9}| {'burst-fraction':<15}| {bb:<17.3f}| {ba:+.3f}")
    print("=" * 68)
    print(f"multiplexing: A own {aa:.3f} >> cross {abs(ab):.3f}   |   B own {bb:.3f} >> cross {abs(ba):.3f}")
    print("=" * 68)

    # ---- self-check (deterministic asserts; nothing rigged) -------------------------------
    # (a) channel A (event rate) recovered
    assert aa > 0.7, f"channel A (event rate) must track its signal (corr {aa:.3f})"
    # (b) channel B (burst fraction) recovered
    assert bb > 0.7, f"channel B (burst fraction) must track its signal (corr {bb:.3f})"
    # sources genuinely independent (orthogonal sinusoids)
    assert abs(src) < 0.2, f"the two SOURCE signals must be independent (corr {src:.3f})"
    # (c) LOW CROSSTALK -- the defining multiplexing property: each channel tracks ITS OWN
    #     signal far better than the other, so two channels share one train with little interference.
    assert abs(ab) < 0.3 and abs(ba) < 0.3, f"crosstalk must be low (A/B {ab:.3f}, B/A {ba:.3f})"
    assert aa > abs(ab) + 0.4, f"channel A must track its OWN signal >> the other ({aa:.3f} vs {abs(ab):.3f})"
    assert bb > abs(ba) + 0.4, f"channel B must track its OWN signal >> the other ({bb:.3f} vs {abs(ba):.3f})"
    # (d) determinism: same seed -> byte-identical decoded correlations
    assert simulate(seed)[0] == c, "same seed must reproduce identical decoded correlations"
    print(f"self-check OK: A(event-rate) corr {aa:.3f} own vs {abs(ab):.3f} cross; "
          f"B(burst-fraction) corr {bb:.3f} own vs {abs(ba):.3f} cross; two signals, one train, deterministic")


if __name__ == "__main__":
    main()

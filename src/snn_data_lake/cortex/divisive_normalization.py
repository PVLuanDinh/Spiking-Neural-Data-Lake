"""
Divisive normalization (Carandini-Heeger) -- the canonical neural gain-control computation.

A NEW gain-control primitive (additive, opt-in). Each unit's response is its own drive divided by the
SUMMED activity of a normalization POOL, so every unit is suppressed by the total context around it.
This is FAST, INSTANTANEOUS gain control by the pool's CURRENT drives -- a single operation that yields
contrast SATURATION, contrast-INVARIANT tuning, cross-unit SUPPRESSION and winner-sharpening, the
computation Carandini & Heeger show recurs across V1, MT, LIP, attention and olfaction:

    r_i = d_i**n / (sigma**n + sum_j d_j**n)     (d_i >= 0; n = exponent; sigma = semi-saturation)

DISTINCT from the repo's other gain/competition modules (different timescale AND different math):
  - homeostatic_scaling.py: SLOW multiplicative synaptic scaling toward a rate set-point over many
    steps (a homeostatic set-point with state and a time constant). This is FAST/instantaneous gain
    control by the pool's CURRENT drives -- no state, no set-point, no learning.
  - intrinsic_plasticity.py: adapts a SINGLE neuron's own gain/bias over time. This is a POOL
    operation dividing each unit by the shared pool sum -- context, not self-adaptation.
  - lca_sparse_coding.py lateral inhibition: SUBTRACTIVE competition (u += dt*(b - u - inhib), the
    atom-overlap term is SUBTRACTED). This DIVIDES by the pool sum -- multiplicative gain control, and
    that division is precisely what makes the tuning shape contrast-INVARIANT and the response
    SATURATE. Different operation, not just different parameters.

A rate-level pool operation -- a reduced model of the canonical divisive-normalization
computation on synthetic nonnegative drives, NOT a full cortical circuit and NOT cognition/AGI. It is
purely additive/opt-in (a new normalization/gain-control primitive; changes no existing behaviour).
Credit: Carandini & Heeger, "Normalization as a canonical neural computation", Nat Rev Neurosci 13 (2012).

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.divisive_normalization
"""
from snn_data_lake.cortex.lif_neuron import LIFNeuron


class DivisiveNormalization:
    """Divide each unit's drive^n by (sigma^n + pool sum of drive^n) -- fast pool gain control."""

    def __init__(self, sigma=0.5, n=2.0):
        self.sigma, self.n = sigma, n

    def normalize(self, drives):
        """r_i = d_i^n / (sigma^n + sum_j d_j^n) over a pool of nonnegative drives d (deterministic)."""
        p = [d ** self.n for d in drives]             # each unit's own drive raised to the exponent
        denom = self.sigma ** self.n + sum(p)         # semi-saturation + summed POOL activity
        return [pi / denom for pi in p]               # divisive: every unit shares the one denominator

    def spike_rates(self, drives, steps=120, gain=0.5):
        """Optional spiking view: the normalized response sets each LIF unit's constant drive current;
        return spikes/step per unit. Deterministic (LIFNeuron has no randomness). Shows that the
        normalized pool gain -- not the raw drive -- controls firing rate."""
        rates = []
        for r in self.normalize(drives):
            rates.append(sum(LIFNeuron(tau_m=10.0).run([gain * r] * steps)) / steps)
        return rates


def main():
    dn = DivisiveNormalization(sigma=0.5, n=2.0)
    pattern = [0.2, 0.5, 1.0, 0.6, 0.3]               # a tuning bump; unit 2 is the preferred stimulus
    win = pattern.index(max(pattern))

    low = dn.normalize([0.5 * d for d in pattern])    # low "contrast"  (overall drive x0.5)
    high = dn.normalize([4.0 * d for d in pattern])   # high "contrast" (overall drive x4 -> x8 more)
    huge = dn.normalize([40.0 * d for d in pattern])  # very high contrast -> near the asymptote

    # 1. SATURATION: input up x8 (0.5x -> 4x) but the winner response grows far sub-linearly, toward
    #    the sigma-negligible limit d_win^n / sum_j d_j^n.
    raw_ratio = 4.0 / 0.5                              # = 8.0 (absolute drive scaling low -> high)
    grow = high[win] / low[win]                        # normalized winner growth (saturating)
    asymptote = pattern[win] ** dn.n / sum(d ** dn.n for d in pattern)

    # 2. CONTRAST-INVARIANT TUNING: the shape normalized to its own peak is identical across contrast,
    #    and the winner + rank order are preserved.
    shape_low = [r / low[win] for r in low]
    shape_high = [r / high[win] for r in high]
    shape_drift = max(abs(a - b) for a, b in zip(shape_low, shape_high))
    order_low = sorted(range(len(low)), key=lambda i: low[i], reverse=True)
    order_high = sorted(range(len(high)), key=lambda i: high[i], reverse=True)

    # 3. CROSS-SUPPRESSION: adding a strong competitor to the pool grows the shared denominator and
    #    suppresses the target unit's response.
    solo = dn.normalize([1.0, 0.3, 0.2])
    with_comp = dn.normalize([1.0, 0.3, 0.2, 3.0])

    # 4. BOUNDED OUTPUT: the pool sum stays below 1 and rises toward 1 as contrast grows.
    sums = (sum(low), sum(high), sum(huge))

    # spiking view: the preferred unit has the highest LIF firing rate.
    rates = dn.spike_rates([4.0 * d for d in pattern])

    print("=" * 60)
    print("DIVISIVE NORMALIZATION (Carandini-Heeger) -- canonical pool gain control")
    print("=" * 60)
    print(f"pool drives (tuning bump): {pattern}, preferred unit = {win}")
    print(f"1 saturation : winner {low[win]:.3f} (low) -> {high[win]:.3f} (high, input x{raw_ratio:.0f})")
    print(f"               grows only x{grow:.2f}, approaching asymptote {asymptote:.3f}")
    print(f"2 invariance : winner unchanged ({win}); rank order {order_low} == {order_high}")
    print(f"               tuning-shape drift across contrast = {shape_drift:.1e} (contrast-invariant)")
    print(f"2b compress  : raw winner range x{raw_ratio:.1f} -> normalized x{grow:.2f} (compressed, order kept)")
    print(f"3 suppression: target {solo[0]:.3f} (alone) -> {with_comp[0]:.3f} (strong competitor added)")
    print(f"4 bounded    : pool sum {sums[0]:.3f} -> {sums[1]:.3f} -> {sums[2]:.3f} (< 1, rising toward 1)")
    print(f"spiking view : LIF rates {[round(r, 2) for r in rates]}, max at unit {rates.index(max(rates))}")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep, ASCII-only ----
    assert grow < 2.0 and grow < raw_ratio, f"winner must saturate, growing far sub-linearly (x{grow:.2f} vs input x{raw_ratio:.0f})"
    assert abs(high[win] - asymptote) < 0.02, f"high-contrast response must approach the asymptote ({high[win]:.3f} vs {asymptote:.3f})"
    assert huge[win] > high[win] and abs(huge[win] - asymptote) < 1e-3, f"response must keep rising toward the asymptote ({huge[win]:.4f})"
    assert order_low == order_high, f"rank order must be contrast-invariant ({order_low} != {order_high})"
    assert low.index(max(low)) == win == high.index(max(high)), "the argmax winner must be contrast-invariant"
    assert shape_drift < 1e-9, f"peak-normalized tuning shape must be identical across contrast ({shape_drift:.1e})"
    assert grow < raw_ratio, f"normalization must compress the winner's absolute range (x{grow:.2f} < x{raw_ratio:.1f})"
    assert with_comp[0] < 0.5 * solo[0], f"a strong pool competitor must suppress the target ({with_comp[0]:.3f} < {solo[0]:.3f})"
    assert all(s < 1.0 for s in sums), f"pool output must stay bounded below 1 ({tuple(round(s, 3) for s in sums)})"
    assert sums[1] > sums[0] and sums[2] > 0.99, f"pool sum must rise toward 1 as contrast grows ({tuple(round(s, 3) for s in sums)})"
    assert rates[win] == max(rates) and rates[win] > 0, f"the preferred unit must have the highest LIF rate ({rates})"
    assert DivisiveNormalization().normalize(pattern) == DivisiveNormalization().normalize(pattern), "same drives -> identical responses"
    print(f"self-check OK: saturation {low[win]:.2f}->{high[win]:.2f} (input x{raw_ratio:.0f}, asymptote {asymptote:.2f}); "
          f"winner {win} + order invariant (drift {shape_drift:.0e}); "
          f"suppression {solo[0]:.2f}->{with_comp[0]:.2f}; bounded sum {sums[1]:.2f}<1; deterministic")


if __name__ == "__main__":
    main()

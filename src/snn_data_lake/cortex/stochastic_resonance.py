"""
Stochastic resonance -- noise as a RESOURCE: an OPTIMAL noise level MAXIMIZES weak-signal detection.

A weak periodic signal s(t) = amp * sin(2*pi*f*t) whose amplitude is BELOW a detector's threshold
almost never crosses it, so it is (nearly) invisible at the output. Add zero-mean Gaussian noise of
amplitude sigma: now the noise occasionally pushes signal+noise over threshold, and -- crucially -- it
does so preferentially near the signal's peaks (where s(t) is already largest), so the output spike
train becomes correlated with the signal. Detection quality is NON-MONOTONIC in sigma: ~0 at sigma=0
(subthreshold, no crossings), rising to a PEAK at an intermediate OPTIMAL sigma, then falling at large
sigma (noise swamps the signal). That interior peak IS stochastic resonance (Benzi, Sutera & Vulpiani
1981; in neurons Wiesenfeld & Moss, Nature 373:33, 1995; review Gammaitoni, Hanggi, Jung & Marchesoni,
Rev Mod Phys 70:223, 1998).

Detector: a bare upward-threshold-crossing unit (fires when signal+noise crosses `threshold` going up).
No membrane dynamics -- the level-crossing detector is the canonical SR device and keeps the noise
optimum uncontaminated by leak/refractory time constants. (LIFNeuron would also work; not needed here.)
Metric: the output spike train's spectral SNR = power at the signal frequency bin divided by the mean
background power at neighbouring off-signal bins (averaged over a few noise realizations). Because the
signal fits a whole number of cycles in the window, its energy sits in one clean integer bin with no
leakage, so the off-bins are pure background. The SNR genuinely PEAKS at an interior sigma.

DISTINCTNESS (this is not any other repo module):
  - reflex.py does threshold detection but treats noise ONLY as something to be robust to -- there is no
    noise-as-a-resource, no optimal-noise level, no SNR-vs-noise curve. THIS module shows detection is a
    NON-MONOTONIC function of noise with an interior optimum: noise ENHANCES weak-signal detection.
  - neuronal_avalanches.py also has an "optimal operating point" (criticality), but that is a
    branching-cascade / power-law phenomenon over a whole NETWORK; stochastic resonance is an optimal
    NOISE level for weak-signal detection in a single THRESHOLD unit -- a different mechanism and a
    different observable (an SNR-vs-noise peak, not an avalanche-size power law). Not intrinsic_plasticity
    / homeostatic_scaling (gain / rate set-points, not noise), not bayesian_snn (uncertainty).

Minimal faithful core -- a reduced upward-threshold-crossing detector with additive Gaussian
noise on a synthetic subthreshold sinusoid, scored by a spectral SNR, to exhibit the stochastic-resonance
level (an SNR-vs-noise peak). NOT a full spiking network / biophysics; small fixed sizes for a
reproducible fixed-seed curve; not cognition.

  python -m snn_data_lake.cortex.stochastic_resonance        # (also run by run_selfchecks.py)
"""
import math
import random


class StochasticResonance:
    """Subthreshold sinusoid + Gaussian noise into a threshold unit; the SNR-vs-noise curve peaks."""

    def __init__(self, threshold=1.0, amp=0.7, freq=0.05, dt=1.0, n_steps=3000, seed=0, repeats=6):
        if amp >= threshold:
            raise ValueError("amp must be subthreshold (amp < threshold) for stochastic resonance")
        self.threshold = threshold
        self.amp = amp                                    # signal amplitude, kept BELOW threshold
        self.freq = freq
        self.dt = dt
        self.n_steps = n_steps
        self.seed = seed
        self.repeats = repeats                            # noise realizations averaged per sigma
        self.w = 2.0 * math.pi * freq * dt                # radians of signal phase advanced per step
        self._sig_bin = round(n_steps * freq * dt)        # signal frequency bin = whole cycles in window

    def _signal(self, k):
        return self.amp * math.sin(self.w * k)

    def _rng(self, sigma, rep):
        # deterministic noise stream keyed on (seed, repeat, sigma): no global random.seed, no builtin hash
        key = (self.seed & 0xFFFFFFFF) * 1000003 ^ (rep * 2654435761) ^ int(round(sigma * 1e6))
        return random.Random(key)

    def detect(self, sigma, rep=0):
        """Output spike train (0/1 per step): fire on UPWARD crossing of threshold by signal + noise."""
        rng = self._rng(sigma, rep)
        train = []
        prev_above = False
        for k in range(self.n_steps):
            x = self._signal(k) + rng.gauss(0.0, sigma)
            above = x >= self.threshold
            train.append(1 if (above and not prev_above) else 0)   # spike only on the upward crossing
            prev_above = above
        return train

    def _power(self, idx, b):
        """Squared magnitude of the DFT at bin b, summed over the spike-time indices idx."""
        c = s = 0.0
        for k in idx:
            a = 2.0 * math.pi * b * k / self.n_steps
            c += math.cos(a)
            s += math.sin(a)
        return c * c + s * s

    def snr(self, sigma):
        """Detection metric at noise level sigma: signal-band power / mean off-band background power."""
        sig = self._sig_bin
        off = [sig + d for d in (-6, -5, -4, -3, -2, 2, 3, 4, 5, 6)]   # clean background bins (no leakage)
        acc = []
        for rep in range(self.repeats):
            idx = [k for k, v in enumerate(self.detect(sigma, rep)) if v]
            if not idx:                                   # no spikes (subthreshold): no response at all
                acc.append(0.0)
                continue
            pbg = sum(self._power(idx, b) for b in off) / len(off)
            acc.append(self._power(idx, sig) / pbg if pbg > 0 else 0.0)
        return sum(acc) / len(acc)

    def sweep(self, sigmas):
        """The stochastic-resonance curve: the detection metric at each noise level (list of SNRs)."""
        return [self.snr(sigma) for sigma in sigmas]


def main():
    sr = StochasticResonance()          # threshold=1.0, amp=0.7 (subthreshold), freq=0.05, n_steps=3000
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.6, 2.2]
    curve = sr.sweep(sigmas)

    zero_spikes = sum(sr.detect(0.0))                     # subthreshold signal alone: no crossings
    m0 = curve[0]
    imax = max(range(len(curve)), key=lambda i: curve[i])
    peak_sigma, peak = sigmas[imax], curve[imax]
    last = curve[-1]
    interior = 0 < imax < len(curve) - 1
    ratio_low = "inf" if m0 == 0 else f"{peak / m0:.1f}x"
    ratio_high = peak / last if last > 0 else float("inf")

    print("=" * 64)
    print("STOCHASTIC RESONANCE -- optimal noise maximizes weak-signal detection")
    print("=" * 64)
    print(f"subthreshold signal: amp={sr.amp} < threshold={sr.threshold}; "
          f"spikes at sigma=0: {zero_spikes}")
    scale = peak if peak > 0 else 1.0
    for sg, m in zip(sigmas, curve):
        bar = "#" * int(round(40 * m / scale))
        mark = "  <- peak (optimal noise)" if sg == peak_sigma else ""
        print(f"  sigma={sg:<4} SNR={m:8.2f} |{bar}{mark}")
    print("-" * 64)
    print(f"peak at sigma={peak_sigma} (interior={interior}); "
          f"SNR sigma=0 {m0:.2f} -> peak {peak:.2f} -> max-sigma({sigmas[-1]}) {last:.2f}")
    print(f"margin: peak/low-end = {ratio_low} , peak/high-end = {ratio_high:.1f}x")
    print("=" * 64)

    # ---- self-check: deterministic asserts, zero-dep ----
    # 1. subthreshold signal alone is (nearly) undetected: no crossings, no signal-band response
    assert zero_spikes == 0, f"subthreshold signal must not cross threshold at sigma=0 ({zero_spikes})"
    assert m0 < 1.0, f"no signal-band response without noise ({m0:.3f})"
    # 2. stochastic-resonance curve: NON-MONOTONIC with a strictly INTERIOR peak
    assert interior, f"peak must be interior to the sweep, not an endpoint (idx {imax}/{len(curve) - 1})"
    assert peak > m0 and peak > last, f"peak {peak:.2f} must exceed both ends ({m0:.2f}, {last:.2f})"
    # 3. the peak is a REAL optimum: clear margin over the high-noise end (low end is 0)
    assert ratio_high > 3.0, f"peak must clearly beat the high-noise end ({ratio_high:.1f}x)"
    # 4. determinism: same params/seed -> identical sweep
    assert StochasticResonance().sweep(sigmas) == StochasticResonance().sweep(sigmas), "deterministic"
    print(f"self-check OK: subthreshold undetected (sigma=0 SNR {m0:.2f}); SR peak at sigma={peak_sigma} "
          f"(interior), SNR {m0:.2f}->{peak:.2f}->{last:.2f}, peak/high-end {ratio_high:.1f}x; deterministic")


if __name__ == "__main__":
    main()

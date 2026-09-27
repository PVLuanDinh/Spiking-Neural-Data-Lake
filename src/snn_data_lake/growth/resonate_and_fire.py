"""
Resonate-and-fire neuron — a frequency-selective spiking neuron (pure Python, no tensors).

The repo's neurons (LIF, Izhikevich, SRM) are INTEGRATORS: they sum input and fire on amplitude. Izhikevich's
(2001) resonate-and-fire neuron is a RESONATOR: a damped 2-D subthreshold oscillation (state z = x + iy,
dz/dt = (b + i*omega) z + input) that builds up only when the input arrives at its natural frequency omega, so
it fires selectively for that frequency and ignores others -- a spiking band-pass filter. A BANK of them tuned to
different frequencies is a neuromorphic spectral analyser: it reads the oscillation content of a spike/signal
stream (a CPG gait rhythm, a periodic telemetry channel) directly in spikes, no FFT. Pure standard library,
deterministic. On-thesis with cpg_oscillator / izhikevich_neuron / spike_preprocessing.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.growth.resonate_and_fire        # (also run by run_selfchecks.py)


P7 NOTE (2026-08-04, on promotion into cortex/): the phrase "pure standard library" above
described obedience to the zero-dependency rule, abolished 2026-07-20. The stdlib implementation
is KEPT because it is correct on its merits here, not because a rule requires it: this is a
SCALAR single-neuron model with a few floats of state, stepped per-neuron, exactly like its now
sibling `growth/lif_neuron.py`. numpy pays off on populations, not on one neuron -- vectorizing a
scalar step would be slower and would duplicate dynamics P2 says must have one definition. What
was removed is the BOAST, not the design.
"""

import math


class ResonateFire:
    """Damped 2-D oscillator that fires when driven near its resonant frequency (a spiking band-pass filter)."""

    def __init__(self, freq_hz, damping=2.0, threshold=0.3, dt=0.001):
        self.omega = 2.0 * math.pi * freq_hz
        self.b = -damping  # subthreshold decay (negative -> damped oscillation)
        self.threshold = threshold
        self.dt = dt
        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = 0.0

    def step(self, current):
        x, y = self.x, self.y
        self.x = x + (self.b * x - self.omega * y + current) * self.dt  # rotate + decay + drive
        self.y = y + (self.omega * x + self.b * y) * self.dt
        if self.y >= self.threshold:  # oscillation crossed threshold -> spike
            self.x = self.y = 0.0
            return 1
        return 0

    def run(self, signal):
        self.reset()
        return sum(self.step(c) for c in signal)


def _sine(freq_hz, steps, dt=0.001, amp=1.0):
    return [amp * math.sin(2.0 * math.pi * freq_hz * t * dt) for t in range(steps)]


def main():
    dt, steps = 0.001, 2000  # 2 seconds

    # 1. tuning curve: a 10 Hz-tuned neuron responds only near 10 Hz (band-pass)
    tuned = ResonateFire(10.0, dt=dt)
    probe = [4, 7, 10, 13, 18]
    tuning = [tuned.run(_sine(f, steps, dt)) for f in probe]
    peak = tuning[probe.index(10)]
    off = max(tuning[i] for i in range(len(probe)) if probe[i] != 10)

    # 2. spectral decomposition: a bank reads the frequency content of a two-tone signal (10 Hz + 20 Hz)
    bank_freqs = [5, 10, 15, 20]
    bank = [ResonateFire(f, dt=dt) for f in bank_freqs]
    mixed = [a + b for a, b in zip(_sine(10, steps, dt), _sine(20, steps, dt))]
    response = [n.run(mixed) for n in bank]
    # sustained resonance = repeated firing; a lone spike is just the signal-onset transient, not a detection
    detected = [bank_freqs[i] for i, r in enumerate(response) if r >= 2]

    print("=" * 60)
    print("RESONATE-AND-FIRE — frequency-selective spiking neuron")
    print("=" * 60)
    print(f"tuning (10 Hz neuron): freqs {probe} Hz -> spikes {tuning} (peak at 10, ~0 off)")
    print(f"bank {bank_freqs} Hz on a 10+20 Hz signal -> spikes {response}")
    print(f"detected frequencies : {detected} Hz (the two tones present)")
    print("=" * 60)

    # ---- self-check: deterministic asserts ----
    assert peak > 0 and off == 0, f"the neuron must resonate at its tuned frequency only ({tuning})"
    assert set(detected) == {10, 20}, (
        f"the bank must detect exactly the two tones present ({detected})"
    )
    assert response[bank_freqs.index(10)] >= 2 and response[bank_freqs.index(15)] < 2, (
        "off-tone neurons stay below sustained resonance"
    )
    assert ResonateFire(10.0, dt=dt).run(_sine(10, steps, dt)) == peak, (
        "same signal -> identical response (deterministic)"
    )
    print(
        f"self-check OK: band-pass resonance (peak {peak} at 10 Hz, {off} off); bank decomposed 10+20 Hz "
        f"signal to {detected}; deterministic"
    )


if __name__ == "__main__":
    main()

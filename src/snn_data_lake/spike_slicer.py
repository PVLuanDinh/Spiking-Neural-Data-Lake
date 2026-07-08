"""
SpikeSlicer — pure-Python event-stream segmenter (spike-density + position-aware slicing).

Slices a continuous stream into semantically coherent segments without tensors. Each frame is TTFS-encoded
(reused `spike_preprocessing.encode_latency`) to a spike count = local spike DENSITY; a LIF "boundary" neuron
(reused `lif_neuron.LIFNeuron`) integrates the density-CHANGE signal and fires a cut where the stream shifts
regime. Position-aware in two senses: the neuron's refractory period enforces a minimum segment length (no
two cuts too close), and the firing threshold is set relative to the density-change distribution (mu + k*sd)
so cuts land on the significant boundaries, not noise. Pure standard library, deterministic. The zero-dep
counterpart to the torch SpikeSlicer in research/neuromorphic_torch.

  python -m snn_data_lake.spike_slicer        # (also run by run_selfchecks.py)
"""
import random

from snn_data_lake.lif_neuron import LIFNeuron
from snn_data_lake.spike_preprocessing import encode_latency


def stream_density(frames):
    """Per-frame spike density = number of TTFS spikes (reuses encode_latency)."""
    return [len(encode_latency(f)) for f in frames]


class SpikeSlicer:
    """Fire a cut where spike density shifts; refractory enforces a minimum (position-aware) segment length."""

    def __init__(self, min_segment=4, sensitivity=1.5, tau=2.0):
        self.min_segment = min_segment
        self.sensitivity = sensitivity
        self.tau = tau

    def density_change(self, densities):
        return [0.0] + [abs(densities[t] - densities[t - 1]) for t in range(1, len(densities))]

    def slice(self, densities):
        """Return the list of cut positions (segment boundaries) for a per-frame density sequence."""
        sig = self.density_change(densities)
        mu = sum(sig) / len(sig)
        sd = (sum((x - mu) ** 2 for x in sig) / len(sig)) ** 0.5 or 1e-9
        thr = mu + self.sensitivity * sd                          # position-aware boundary threshold
        neuron = LIFNeuron(tau_m=self.tau, v_threshold=thr, refractory=self.min_segment)
        return [t for t, s in enumerate(neuron.run(sig)) if s]


def _segmented_stream(rng, seg_active, per=12, dim=40):
    """Build a stream of `per` frames per segment; each segment has a distinct number of active features
    (-> distinct spike density). Returns (frames, true_boundaries)."""
    frames, bounds, t = [], [], 0
    for k in seg_active:
        for _ in range(per):
            f = [0.0] * dim
            for i in range(k):
                f[i] = 0.5 + rng.uniform(-0.05, 0.05)             # active feature (stays above floor)
            frames.append(f)
        t += per
        bounds.append(t)
    return frames, bounds[:-1]                                    # drop the trailing end-of-stream


def main():
    rng = random.Random(1)
    frames, true_bounds = _segmented_stream(rng, seg_active=[8, 22, 13, 28])
    densities = stream_density(frames)
    cuts = SpikeSlicer(min_segment=5, sensitivity=1.3).slice(densities)

    tol = 2
    matched = sum(any(abs(c - b) <= tol for c in cuts) for b in true_bounds)
    recall = matched / len(true_bounds)

    print("=" * 60)
    print("SPIKESLICER — pure-Python spike-density stream segmenter")
    print("=" * 60)
    print(f"stream: {len(frames)} frames, {len(true_bounds) + 1} segments (active/seg [8,22,13,28])")
    print(f"true boundaries : {true_bounds}")
    print(f"detected cuts   : {cuts}")
    print(f"boundary recall : {recall:.0%} (+-{tol} steps); {len(cuts)} cuts")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert recall == 1.0, f"SpikeSlicer must recover every segment boundary ({matched}/{len(true_bounds)})"
    assert len(cuts) <= len(true_bounds) + 1, f"must not over-segment ({len(cuts)} cuts)"
    assert all(cuts[i + 1] - cuts[i] >= 5 for i in range(len(cuts) - 1)), "cuts must respect min segment length"
    assert SpikeSlicer().slice(densities) == SpikeSlicer().slice(densities), "deterministic"
    print(f"self-check OK: {recall:.0%} boundary recall, {len(cuts)} cuts respecting min-segment; "
          f"density-change slicing wired to encode_latency; deterministic")


if __name__ == "__main__":
    main()

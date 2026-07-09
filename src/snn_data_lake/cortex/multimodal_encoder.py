"""
v2.6 — multimodal spike encoder: fuse FOUR sensor modalities (visual, temperature,
audio, distance) into the repo's existing N=64 spike feature substrate.

Reuses (does NOT reimplement encoding):
  * population_encoding.PopulationEncoder.activations — spread a normalized scalar
    across a band of overlapping Gaussian receptive fields (values in [0,1]).
  * spike_preprocessing.N / encode_latency — the fused [0,1]^N vector this module
    emits feeds encode_latency UNCHANGED, so the fused code flows through the
    existing matcher / learner pipeline exactly like today's single-modality window.

Layout — the 64 features are BANDED by modality so each stays separable in the code:
    visual      -> features [ 0:16]
    temperature -> features [16:32]
    audio       -> features [32:48]
    distance    -> features [48:64]

Each modality's raw input (scalar or vector) is clamped to [0,1] then population-
encoded into its band: per-element Gaussian activations are AVERAGED, so the band
stays in [0,1] regardless of how many elements the modality carries. A modality
passed as None is a graceful dropout -> its band is all zeros; the other bands still
encode (missing-modality-safe).

The four streams are SYNTHETIC here — no real camera / thermistor / mic /
rangefinder. This is the fusion + encoding architecture a real sensor array would
feed; the banding keeps each modality separable in the 64-feature spike code. Inputs
are expected already normalized to [0,1] (a real driver scales raw physical units
first); we clamp defensively. This is a spike ENCODER, not "thinking".

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.multimodal_encoder
"""
from snn_data_lake.cortex.population_encoding import PopulationEncoder
from snn_data_lake.cortex.spike_preprocessing import N, encode_latency

# 4 equal bands tiling the N=64 substrate, one per modality (16 features each).
_BANDS = (("visual", 0, 16), ("temperature", 16, 32), ("audio", 32, 48), ("distance", 48, 64))
# one RBF population encoder per band; only .activations (Gaussian, in [0,1]) is used.
_ENCODERS = {name: PopulationEncoder(n=hi - lo, lo=0.0, hi=1.0) for name, lo, hi in _BANDS}


def _as_list(x):
    """Coerce a scalar or sequence into a list of floats; None stays None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def _encode_band(values, enc):
    """Normalized values (scalar or vector) -> band of enc.n floats in [0,1].
    Per-element Gaussian activations averaged (average of [0,1] stays in [0,1])."""
    acc = [0.0] * enc.n
    for v in values:
        a = enc.activations(min(1.0, max(0.0, v)))     # clamp to [0,1], then RBF-encode
        for j in range(enc.n):
            acc[j] += a[j]
    return [s / len(values) for s in acc]


def encode_multimodal(visual=None, temperature=None, audio=None, distance=None):
    """Fuse four modalities into one [0,1]^N feature vector, banded by modality.

    Any modality None (or empty) -> its band is zeros (missing-modality-safe). The
    result feeds spike_preprocessing.encode_latency exactly like a single-modality
    window. Deterministic (no RNG)."""
    inputs = {"visual": visual, "temperature": temperature, "audio": audio, "distance": distance}
    feats = [0.0] * N
    for name, lo, hi in _BANDS:
        vals = _as_list(inputs[name])
        if not vals:                                   # None or empty -> graceful dropout
            continue
        feats[lo:hi] = _encode_band(vals, _ENCODERS[name])
    return feats


def main():
    visual = [0.2, 0.8, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4]   # flattened intensity grid
    temperature = 0.6                                    # normalized scalar
    audio = [0.1, 0.5, 0.9, 0.3, 0.6]                    # short amplitude vector
    distance = 0.35                                      # normalized scalar

    full = encode_multimodal(visual, temperature, audio, distance)

    print("=" * 60)
    print("MULTIMODAL SPIKE ENCODER (visual|temperature|audio|distance -> N=64)")
    print("=" * 60)
    for name, lo, hi in _BANDS:
        b = full[lo:hi]
        print(f"{name:11s} [{lo:2d}:{hi:2d}]  min={min(b):.3f} max={max(b):.3f} mean={sum(b) / len(b):.3f}")
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    # 1. fixed shape: exactly N floats in [0,1]
    assert len(full) == N, f"expected {N} features, got {len(full)}"
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in full), "features must be floats in [0,1]"

    # 2. separability: a distinct value in ONE modality changes ONLY its band
    changed = encode_multimodal(visual, 0.9, audio, distance)     # only temperature differs
    for name, lo, hi in _BANDS:
        if name == "temperature":
            assert changed[lo:hi] != full[lo:hi], "a distinct temperature must change its band code"
        else:
            assert changed[lo:hi] == full[lo:hi], f"{name} band moved when only temperature changed"

    # 3. missing-modality-safe: None zeros its band, other bands unchanged, no raise
    dropped = encode_multimodal(visual, None, audio, distance)
    assert dropped[16:32] == [0.0] * 16, "dropped modality band must be all zeros"
    for name, lo, hi in _BANDS:
        if name != "temperature":
            assert dropped[lo:hi] == full[lo:hi], f"{name} band moved when temperature dropped"
    assert encode_multimodal() == [0.0] * N, "all modalities None must give an all-zero vector"

    # 4. pipeline-compatible: feeds encode_latency unchanged -> a spike list
    spikes = encode_latency(full)
    assert isinstance(spikes, list), "encode_latency must return a spike list"
    assert all(len(ev) == 2 and 0 <= ev[1] < N for ev in spikes), "spikes are (time, feature) pairs in range"

    # 5. deterministic: same inputs -> identical vector
    assert encode_multimodal(visual, temperature, audio, distance) == full, "encoder must be deterministic"

    print(f"self-check OK: N={N} banded (visual|temperature|audio|distance), per-band isolation, "
          f"missing-modality-safe, {len(spikes)} spikes via encode_latency, deterministic")


if __name__ == "__main__":
    main()

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
from snn_data_lake.cortex.spike_preprocessing import N, encode_latency, van_rossum_distance

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
        a = enc.activations(min(1.0, max(0.0, v)))  # clamp to [0,1], then RBF-encode
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
        if not vals:  # None or empty -> graceful dropout
            continue
        feats[lo:hi] = _encode_band(vals, _ENCODERS[name])
    return feats


class MultiModalEncoder:
    """GROWABLE-width multimodal encoder: the modality set AND the substrate width are runtime
    state, not the fixed N=64. Modalities tile the feature vector as contiguous bands; add_modality
    APPENDS a new band at the high end, so every existing band keeps its [lo, hi) range and its code
    BIT-IDENTICAL (function-preserving growth) while the width grows. That is what lifts the fixed
    4-modality / N=64 ceiling the audit found: a 5th (or Nth) sensor is just a new band, and the
    wider [0,1]^width vector flows through encode_latency + van_rossum_distance UNCHANGED because
    both are already width-parametric (encode_latency reads len(vec); the distances take n_features).

    The DEFAULT bands reproduce the module-level encode_multimodal 4x16=64 substrate exactly, so a
    default instance is a drop-in for the fixed encoder; growth is opt-in via add_modality. A brain
    trained/served at one width keeps that width (its frozen weights are shape-tied); growing the
    width builds a WIDER substrate that NEW learning uses -- honest scope, no silent re-tile of a
    frozen artifact."""

    _DEFAULT = (("visual", 16), ("temperature", 16), ("audio", 16), ("distance", 16))

    def __init__(self, bands=None):
        self._bands = []  # ordered [(name, lo, hi), ...]
        self._enc = {}  # name -> PopulationEncoder for that band
        self.width = 0
        for name, w in bands if bands is not None else self._DEFAULT:
            self.add_modality(name, w)

    def add_modality(self, name, width=16):
        """Register a new modality as a band APPENDED at the high end. Function-preserving: existing
        bands keep their ranges and codes; only the total width grows (by `width`). Returns the new
        width. Raises on a duplicate name or non-positive width."""
        if any(n == name for n, _, _ in self._bands):
            raise ValueError(f"modality {name!r} already registered")
        if width <= 0:
            raise ValueError("modality band width must be positive")
        lo, hi = self.width, self.width + width
        self._bands.append((name, lo, hi))
        self._enc[name] = PopulationEncoder(n=width, lo=0.0, hi=1.0)
        self.width = hi
        return self.width

    def band(self, name):
        """(lo, hi) feature range of a registered modality; raises KeyError if unknown."""
        for n, lo, hi in self._bands:
            if n == name:
                return lo, hi
        raise KeyError(name)

    def modalities(self):
        """Registered modality names, in band order."""
        return [n for n, _, _ in self._bands]

    def encode(self, **values):
        """Fuse any subset of registered modalities into one [0,1]^width vector, banded by modality.
        A modality that is unspecified / None / empty -> its band is zeros (missing-modality-safe).
        An unknown modality name raises. Deterministic (no RNG)."""
        unknown = set(values) - {n for n, _, _ in self._bands}
        if unknown:
            raise KeyError(f"unregistered modalities: {sorted(unknown)}")
        feats = [0.0] * self.width
        for name, lo, hi in self._bands:
            vals = _as_list(values.get(name))
            if not vals:  # unspecified / None / empty -> graceful dropout
                continue
            feats[lo:hi] = _encode_band(vals, self._enc[name])
        return feats


def main():
    visual = [0.2, 0.8, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4]  # flattened intensity grid
    temperature = 0.6  # normalized scalar
    audio = [0.1, 0.5, 0.9, 0.3, 0.6]  # short amplitude vector
    distance = 0.35  # normalized scalar

    full = encode_multimodal(visual, temperature, audio, distance)

    print("=" * 60)
    print("MULTIMODAL SPIKE ENCODER (visual|temperature|audio|distance -> N=64)")
    print("=" * 60)
    for name, lo, hi in _BANDS:
        b = full[lo:hi]
        print(
            f"{name:11s} [{lo:2d}:{hi:2d}]  min={min(b):.3f} max={max(b):.3f} mean={sum(b) / len(b):.3f}"
        )
    print("=" * 60)

    # ---- self-checks --------------------------------------------------------
    # 1. fixed shape: exactly N floats in [0,1]
    assert len(full) == N, f"expected {N} features, got {len(full)}"
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in full), (
        "features must be floats in [0,1]"
    )

    # 2. separability: a distinct value in ONE modality changes ONLY its band
    changed = encode_multimodal(visual, 0.9, audio, distance)  # only temperature differs
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
    assert all(len(ev) == 2 and 0 <= ev[1] < N for ev in spikes), (
        "spikes are (time, feature) pairs in range"
    )

    # 5. deterministic: same inputs -> identical vector
    assert encode_multimodal(visual, temperature, audio, distance) == full, (
        "encoder must be deterministic"
    )

    # ---- GROWABLE-WIDTH encoder: lifts the fixed 4-modality / N=64 ceiling ----------------------
    enc = MultiModalEncoder()
    # 6. the default growable encoder is a byte-for-byte drop-in for the fixed 4x16=64 encoder
    assert enc.width == N, f"default growable encoder width must match the fixed substrate {N}"
    assert (
        enc.encode(visual=visual, temperature=temperature, audio=audio, distance=distance) == full
    ), "default MultiModalEncoder must reproduce encode_multimodal byte-for-byte"

    # 7. FUNCTION-PRESERVING growth: adding a 5th sensor grows the width, existing bands unchanged
    new_w = enc.add_modality("tactile", 16)
    assert new_w == N + 16 and enc.width == N + 16, (
        "adding a modality must grow the width by its band"
    )
    grown = enc.encode(
        visual=visual,
        temperature=temperature,
        audio=audio,
        distance=distance,
        tactile=[0.7, 0.2, 0.9],
    )
    assert grown[:N] == full, (
        "growth must be FUNCTION-PRESERVING: the original 64 features are unchanged"
    )
    lo, hi = enc.band("tactile")
    assert (lo, hi) == (N, N + 16) and any(grown[lo:hi]), (
        "the 5th modality must occupy the appended band"
    )

    # 8. the WIDER substrate flows end-to-end through the EXISTING spike pipeline (encode_latency +
    #    van_rossum_distance are width-parametric): a difference confined to the 5th modality is
    #    detectable at width 80, so the grown width is real signal, not a padded no-op.
    a = enc.encode(
        visual=visual, temperature=temperature, audio=audio, distance=distance, tactile=[0.1]
    )
    b = enc.encode(
        visual=visual, temperature=temperature, audio=audio, distance=distance, tactile=[0.9]
    )
    ea, eb = encode_latency(a), encode_latency(b)
    assert all(0 <= i < enc.width for _, i in ea + eb), "grown-width spikes must stay in [0, width)"
    assert any(i >= N for _, i in ea + eb), (
        "the appended band must actually emit spikes at the grown width"
    )
    assert a[:N] == b[:N], "the two inputs differ ONLY in the tactile band"
    assert van_rossum_distance(ea, eb, n_features=enc.width) > 0.0, (
        "a difference confined to the appended 5th modality must be detectable at width 80"
    )

    # 9. missing-modality-safe + deterministic at the grown width; unknown modality rejected
    assert enc.encode() == [0.0] * enc.width, "all-absent must give an all-zero vector at any width"
    assert enc.encode(tactile=[0.9]) == enc.encode(tactile=[0.9]), (
        "grown encoder must be deterministic"
    )
    try:
        enc.encode(sonar=[0.5])
        raised = False
    except KeyError:
        raised = True
    assert raised, "an unregistered modality must raise, not silently vanish"

    print(
        f"self-check OK: N={N} banded (visual|temperature|audio|distance), per-band isolation, "
        f"missing-modality-safe, {len(spikes)} spikes via encode_latency, deterministic; "
        f"GROWABLE: default==fixed(W={N}), function-preserving +tactile -> W={enc.width} "
        f"separable through encode_latency+van_rossum"
    )


if __name__ == "__main__":
    main()

r"""
Served-frozen DEEPER spiking encoder -- DCLS learnable delays + adaptive-LIF (the Tier-3 encoder).

Honest framing (read this first). This is the pure-stdlib SERVING half of a deeper recurrent-free
spiking encoder whose weights AND per-synapse conduction delays are LEARNED in the torch tier
(research/rsnn_delays_torch.py, surrogate-gradient BPTT on a real temporal spiking dataset -- SHD
preferred, N-MNIST fallback) and then FROZEN and exported as `snn_rsnn/1` JSON. This module reads
that JSON back with a trust-boundary check and runs the forward with NO torch and NO numpy -- exactly
the serve-frozen pattern cerebellum/jepa_world_model.py uses for `snn_jepa/2`. It raises the FEATURE
ceiling on TEMPORAL data: `features(spikes)` returns the penultimate layer's per-neuron SPIKE-COUNT
vector, the drop-in replacement for a frozen conv-STDP `features()` that the SLDA-RFF readout consumes
(research/developmental_loop_slda.py). It is NOT a claim of beating all comers; it is the field-standard
delay+adaptive-LIF recipe (Hammouamri, Khalfaoui-Hassani & Masquelier, "Learning Delays in Spiking
Neural Networks using Dilated Convolutions with Learnable Spacings", ICLR 2024), served zero-dep.

Why delays. In an SNN a spike takes TIME to travel between neurons; a per-synapse conduction delay
shifts a presynaptic spike's arrival, so a downstream neuron can be tuned to fire on a specific
temporal PATTERN (coincidence of spikes that were emitted at DIFFERENT times). Static-rate / zero-delay
features throw that timing away. DCLS learns each delay as a continuous position (a Gaussian bump along
the time axis of a 1-D conv kernel) in the torch tier; at export the positions are finalized to INTEGER
taps, and here each synapse is a plain integer delay read out of a per-input-channel ring buffer.

Adaptive-LIF (ALIF). Each neuron has a per-neuron (heterogeneous, trained) membrane decay `beta` and a
per-neuron adaptation coupling `a` driving an adaptive threshold thr = v_th + a*b, where b is a
spike-triggered adaptation variable decaying at rho. Heterogeneous time constants + spike-frequency
adaptation are what let a feedforward SNN hold temporal context without recurrence.

Exact discrete dynamics (mirrored bit-for-bit by the torch reference so the bridge check matches):
  I_j[t] = sum_i  W[j][i] * x_i[t - d[j][i]]          (delayed synaptic current; x before t=0 is 0)
  u_j[t] = beta_j * u_j[t-1] + I_j[t]                 (leaky integrate, beta in (0,1))
  thr_j  = v_th_j + a_j * b_j[t-1]                     (adaptive threshold)
  s_j[t] = 1 if u_j[t] >= thr_j else 0                 (hard Heaviside; torch uses a surrogate ONLY in backward)
  u_j[t] = u_j[t] - thr_j * s_j[t]                     (reset by subtraction)
  b_j[t] = rho * b_j[t-1] + s_j[t]                     (adaptation increments on a spike)
feature_j = sum_t s_j[t]  over the LAST (penultimate-to-the-readout) spiking layer.

Persisted core format ('snn_rsnn/1'):
  { "format": "snn_rsnn/1", "n_in": N, "rho": r,
    "layers": [ { "W": [out][in], "delays": [out][in] (int taps in [0,max_delay-1]),
                  "max_delay": M, "beta": [out], "a": [out], "v_th": [out] }, ... ] }
Linear.weight is [out][in] (no transpose), layers in APPLICATION order -- identical to the torch export.

Pure stdlib (json, math), ASCII-only prints, deterministic. NO torch, NO numpy in this module.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.sensorimotor.rsnn_encoder
"""

import json

_FORMAT = "snn_rsnn/1"


def _shape_ok(M, rows, cols):
    """True iff M is a rows-by-cols list-of-lists (the same shape guard load_weights in jepa uses)."""
    return (
        isinstance(M, list)
        and len(M) == rows
        and all(isinstance(row, list) and len(row) == cols for row in M)
    )


def _vec_ok(v, n):
    return isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float)) for x in v)


class RsnnEncoder:
    """Frozen forward of a delay + adaptive-LIF spiking encoder, served pure-stdlib.

    Build directly with (n_in, layers, rho) for the self-check, or construct empty and load_weights()
    a torch-tier `snn_rsnn/1` export. features(spikes) runs the ALIF loop with integer-tap ring-buffer
    delay lines and returns the penultimate layer's spike-count feature vector."""

    def __init__(self, n_in=1, layers=None, rho=0.9):
        self.n_in = int(n_in)
        self.rho = float(rho)
        self.layers = layers if layers is not None else []
        if self.layers:
            self._validate_geometry()

    # -- geometry validation (raise ValueError on any inconsistency, like jepa load_weights) ----------
    def _validate_geometry(self):
        prev = self.n_in
        for li, L in enumerate(self.layers):
            W, delays = L.get("W"), L.get("delays")
            nout = len(W) if isinstance(W, list) else -1
            maxd = int(L.get("max_delay", 0))
            if not (nout > 0 and maxd > 0):
                raise ValueError(f"rsnn layer {li}: non-positive out-width or max_delay")
            if not _shape_ok(W, nout, prev):
                raise ValueError(f"rsnn layer {li}: W must be [out={nout}][in={prev}]")
            if not (
                isinstance(delays, list)
                and len(delays) == nout
                and all(isinstance(r, list) and len(r) == prev for r in delays)
            ):
                raise ValueError(f"rsnn layer {li}: delays must be [out={nout}][in={prev}]")
            for r in delays:
                for d in r:
                    if not (isinstance(d, int) and 0 <= d < maxd):
                        raise ValueError(
                            f"rsnn layer {li}: delay {d!r} out of [0,{maxd}) or non-integer"
                        )
            for key in ("bias", "beta", "a", "v_th"):
                if not _vec_ok(L.get(key), nout):
                    raise ValueError(f"rsnn layer {li}: '{key}' must be a length-{nout} vector")
            prev = nout

    def feat_dim(self):
        """Feature width == out-width of the LAST (penultimate-to-readout) spiking layer."""
        return len(self.layers[-1]["W"])

    # -- one ALIF layer forward with integer-tap ring-buffer delay lines -----------------------------
    @staticmethod
    def _layer_forward(X, L, rho):
        """X: list of T frames, each a length-nin list of 0/1. Returns (spike frames Y, spike counts)."""
        W, delays = L["W"], L["delays"]
        bias, beta, a, v_th, maxd = L["bias"], L["beta"], L["a"], L["v_th"], int(L["max_delay"])
        nout = len(W)
        nin = len(W[0]) if nout else 0
        T = len(X)
        # per-input-channel circular delay line of length maxd (integer taps read d steps back)
        ring = [[0.0] * maxd for _ in range(nin)]
        pos = 0
        u = [0.0] * nout
        b = [0.0] * nout
        counts = [0.0] * nout
        Y = []
        for t in range(T):
            xt = X[t]
            for i in range(nin):  # write this step's spikes into the ring
                ring[i][pos] = xt[i]
            frame = [0.0] * nout
            for j in range(nout):
                Wj, dj = W[j], delays[j]
                cur = bias[j]  # per-neuron bias current
                for i in range(nin):  # + delayed synaptic current I_j[t]
                    cur += Wj[i] * ring[i][(pos - dj[i]) % maxd]
                uj = beta[j] * u[j] + cur  # leaky integrate
                thr = v_th[j] + a[j] * b[j]  # adaptive threshold
                sp = 1.0 if uj >= thr else 0.0  # hard Heaviside
                u[j] = uj - thr * sp  # reset by subtraction
                b[j] = rho * b[j] + sp  # adaptation
                frame[j] = sp
                counts[j] += sp
            Y.append(frame)
            pos = (pos + 1) % maxd
        return Y, counts

    def forward_spikes(self, spikes):
        """Run every layer; return (last-layer spike frames, per-layer spike-count vectors)."""
        if not self.layers:
            raise ValueError("rsnn encoder has no layers (build or load_weights first)")
        X = spikes
        per_layer_counts = []
        for L in self.layers:
            X, counts = self._layer_forward(X, L, self.rho)
            per_layer_counts.append(counts)
        return X, per_layer_counts

    def features(self, spikes):
        """Penultimate SPIKE-COUNT feature vector for `spikes` (a list of T frames, each length n_in).

        This is the frozen `features()` the SLDA-RFF readout consumes -- one integer-valued count per
        penultimate-layer neuron, summed over the T time steps."""
        if len(spikes) and len(spikes[0]) != self.n_in:
            raise ValueError(f"rsnn features: frame width {len(spikes[0])} != n_in {self.n_in}")
        _, per_layer_counts = self.forward_spikes(spikes)
        return per_layer_counts[-1]

    # -- weight I/O with a trust-boundary check (mirrors jepa save/load_weights) ----------------------
    def save_weights(self, path):
        """Persist as tagged JSON. Floats round-trip exactly; delays stay ints."""
        obj = {"format": _FORMAT, "n_in": self.n_in, "rho": self.rho, "layers": self.layers}
        with open(path, "w", encoding="ascii") as f:
            json.dump(obj, f)

    def load_weights(self, path):
        """Load `snn_rsnn/1` JSON, validating the format tag and EVERY shape; ValueError on any mismatch
        (never trust an unvalidated weight file). Delays must be integer taps in [0, max_delay)."""
        with open(path, encoding="ascii") as f:
            obj = json.load(f)
        fmt = obj.get("format") if isinstance(obj, dict) else None
        if fmt != _FORMAT:
            raise ValueError(
                f"bad rsnn weight file: missing/incorrect 'format' tag (expected {_FORMAT})"
            )
        n_in = int(obj.get("n_in", -1))
        layers = obj.get("layers")
        if not (n_in > 0 and isinstance(layers, list) and layers):
            raise ValueError("bad rsnn weight file: non-positive n_in or empty layers")
        self.n_in = n_in
        self.rho = float(obj.get("rho", self.rho))
        self.layers = layers
        self._validate_geometry()  # raises ValueError on any shape/tap mismatch
        return self


# =====================================================================================================
# Self-check: hand-built weights prove (1) exact ALIF+delay determinism, (2) the coincidence-detection
# property that LEARNABLE DELAYS buy -- a neuron fires only when a delay ALIGNS two spikes that were
# emitted at different times; remove the delay and the same two spikes never coincide, so it stays
# silent. Un-riggable (outputs are binary spikes; the assertion is the delay flipping a spike on/off),
# structurally spiking, zero-dep. Also round-trips save/load and rejects a bad format tag.
# =====================================================================================================
def _coincidence_layer(delay_on):
    """One ALIF neuron, 2 inputs, v_th=1.5, no leak decay change: input 0 spikes at t=0, input 1 at
    t=2. With a delay of 2 on input 0 (delay_on) both arrive together at t=2 -> current 2.0 >= 1.5 ->
    ONE spike. With delay 0 on input 0 the two unit spikes never coincide -> current 1.0 < 1.5 ->
    NO spike. beta small so there is no cross-step accumulation to muddy the demonstration."""
    d0 = 2 if delay_on else 0
    return {
        "W": [[1.0, 1.0]],
        "delays": [[d0, 0]],
        "max_delay": 4,
        "bias": [0.0],
        "beta": [0.1],
        "a": [0.0],
        "v_th": [1.5],
    }


def main():
    print("=" * 74)
    print("RSNN ENCODER -- served-frozen DCLS delays + adaptive-LIF (zero-dep forward)")
    print("=" * 74)

    # 2 inputs, one spike each at DIFFERENT times: input 0 @ t=0, input 1 @ t=2, over T=4.
    spikes = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]

    enc_on = RsnnEncoder(n_in=2, layers=[_coincidence_layer(delay_on=True)], rho=0.9)
    enc_off = RsnnEncoder(n_in=2, layers=[_coincidence_layer(delay_on=False)], rho=0.9)
    count_on = enc_on.features(spikes)[0]
    count_off = enc_off.features(spikes)[0]

    print("input: spike on ch0 @ t=0, spike on ch1 @ t=2  (emitted 2 steps apart)")
    print(
        f"  delay=2 on ch0 -> the two spikes ALIGN at t=2 -> neuron fires : count = {count_on:.0f}"
    )
    print(
        f"  delay=0 on ch0 -> spikes never coincide      -> silent       : count = {count_off:.0f}"
    )

    # a small 2-layer determinism check: fixed weights -> a fixed, reproducible count vector.
    layers2 = [
        {
            "W": [[0.9, 0.0, 0.0], [0.0, 0.9, 0.9]],
            "delays": [[0, 1, 0], [0, 0, 2]],
            "max_delay": 3,
            "bias": [0.0, 0.0],
            "beta": [0.5, 0.7],
            "a": [0.2, 0.1],
            "v_th": [0.8, 0.8],
        },
        {
            "W": [[1.0, 1.0]],
            "delays": [[0, 1]],
            "max_delay": 3,
            "bias": [0.0],
            "beta": [0.6],
            "a": [0.3],
            "v_th": [1.0],
        },
    ]
    enc2 = RsnnEncoder(n_in=3, layers=layers2, rho=0.9)
    seq = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    feat2a = enc2.features(seq)
    feat2b = RsnnEncoder(n_in=3, layers=layers2, rho=0.9).features(
        seq
    )  # fresh instance, same result

    # save/load round-trip + bad-format rejection
    import os
    import tempfile

    p = os.path.join(tempfile.gettempdir(), "rsnn_selfcheck.json")
    enc2.save_weights(p)
    reloaded = RsnnEncoder().load_weights(p)
    feat2c = reloaded.features(seq)

    bad = os.path.join(tempfile.gettempdir(), "rsnn_bad.json")
    with open(bad, "w", encoding="ascii") as f:
        json.dump({"format": "not-rsnn", "n_in": 3, "layers": layers2}, f)
    rejected = False
    try:
        RsnnEncoder().load_weights(bad)
    except ValueError:
        rejected = True

    print(f"2-layer feature vector (penultimate counts) : {[int(v) for v in feat2a]}")
    print(
        f"  feat_dim = {enc2.feat_dim()}  reload-matches = {feat2c == feat2a}  "
        f"bad-format-rejected = {rejected}"
    )
    print("=" * 74)

    # ---- un-riggable, structurally-spiking, deterministic asserts (zero-dep) ----
    assert count_on == 1.0, f"delay-aligned coincidence must fire exactly once (got {count_on})"
    assert count_off == 0.0, f"without the delay the spikes must NOT coincide (got {count_off})"
    assert count_on > count_off, (
        "LEARNABLE DELAY must flip the neuron from silent to firing (the whole point)"
    )
    # every emitted value is a binary spike (structurally spiking, not a rate/analog activation)
    frames, _ = enc2.forward_spikes(seq)
    assert all(v in (0.0, 1.0) for fr in frames for v in fr), "outputs must be binary spikes"
    assert all(float(v).is_integer() for v in feat2a), "features must be integer spike COUNTS"
    assert feat2b == feat2a, "forward must be deterministic across fresh instances"
    assert feat2c == feat2a, "save/load round-trip must preserve the forward exactly"
    assert enc2.feat_dim() == 1, "feat_dim must be the last layer's out-width"
    assert rejected, "load_weights must reject a bad/missing format tag (trust boundary)"
    print(
        f"self-check OK: delays enable coincidence detection (1 vs 0 spike); ALIF forward deterministic "
        f"({[int(v) for v in feat2a]}); binary spikes + integer counts; snn_rsnn/1 round-trips + trust-checked."
    )


if __name__ == "__main__":
    main()

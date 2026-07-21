r"""
Zero-dep forward SERVER for a torch-trained 2-level Hierarchical H-JEPA (`snn_hjepa/1`).

Honest framing (read this first). The deep, nonlinear, 2-level H-JEPA is TRAINED in the torch
tier (research/hjepa_torch.py: deep ReLU-MLP encoders at L0/L1, an EMA/stop-grad target, VICReg
anti-collapse, action-conditioned residual/pure-MLP predictors) and EXPORTED as an `snn_hjepa/1`
JSON artifact. This module is the matching PURE-STDLIB core SERVER: it LOADS that artifact and
replays the forward with NO torch, NO numpy -- forward-serve only, no training here. It is the
H-JEPA analogue of cerebellum/jepa_world_model.py's `snn_jepa/2` served/frozen deep encoder, one
level up: two levels (fine L0 obs latent + abstract L1 window latent) instead of one.

The forward replicated here is byte-for-byte the torch `_HJepa` (research/hjepa_torch.py):
  * mlp(layers, x)     -- ordered Linear layers; ReLU BETWEEN layers, RAW (no activation) after
                          the last. h=x; h=W@h+b; relu(h) unless it is the final layer.
  * encode0(obs)       -- unit(mlp(enc0, obs))                             L0 fine latent on the sphere
  * encode1(window)    -- unit(mlp(enc1, meanpool([encode0(o) for o in window])))   L1 abstract latent
                          meanpool = elementwise mean of the per-tick L0 latents (averages out the
                          fast phase, keeps the slow XOR-carrying signal), then the enc1 head, then unit
  * predict0(z0, a)    -- unit(z0 + mlp(pred0, z0 ++ onehot_fine(a)))      RESIDUAL nonlinear fine predictor
  * predict1(z1, a)    -- unit(mlp(pred1, z1 ++ onehot_coarse(a)))         PURE-MLP coarse predictor
  * unit(v)            -- L2-normalize onto the unit sphere (shared _mathutils.unit).
The online encoders (enc0/enc1) are what inference uses; the EMA `*_target` copies exist only for
training and are validated-if-present but never used to serve.

Duck-typing. predict(state, command) aliases predict1, so an HJepaServe drops straight into
PrefrontalExecutive.plan() as its `predictor`, planning over the ABSTRACT (coarse) level -- exactly
how JepaWorldModel.predict() is used by plan() one level down.

Trust boundary. load() NEVER trusts an unvalidated weight file: it checks the `snn_hjepa/1` format
tag, the positive integer geometry, the action lists, and EVERY Linear layer's shape + chaining
(each layer's input must equal the previous layer's output, the first/last must match the declared
dims, and the predictor input widths must equal latent + |actions|), raising ValueError on any
mismatch. save()/round-trip is intentionally NOT provided -- this is a serve-only loader.

Pure stdlib (math, json + shared _mathutils dot/mean/unit), ASCII-only prints, deterministic.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cerebellum.hjepa_serve
"""

import json
import math

from snn_data_lake.substrate._mathutils import dot, mean, unit

HJEPA_FORMAT = "snn_hjepa/1"


# ---- forward primitives (match research/hjepa_torch.py exactly) ------------------------------------
def _mlp(layers, x):
    """Apply ordered [W, b] Linear layers to x. ReLU BETWEEN layers, RAW after the last (encoders and
    predictors normalize afterwards). Mirrors the torch `mlp()`/`nn.Sequential` forward: for each
    layer h = W @ h + b, then relu(h) UNLESS it is the final layer, whose output is returned raw."""
    h = list(x)
    last = len(layers) - 1
    for i, (W, b) in enumerate(layers):
        h = [dot(W[o], h) + b[o] for o in range(len(b))]
        if i < last:
            h = [v if v > 0.0 else 0.0 for v in h]  # ReLU only BETWEEN layers
    return h


# ---- weight-file validation (the trust boundary; never serve an unvalidated file) ------------------
def _pos_int(v, name):
    """A JSON value must be a positive integer (reject bool, which subclasses int)."""
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"bad {HJEPA_FORMAT} file: {name} must be a positive integer (got {v!r})")
    return v


def _validate_stack(name, layers, in_dim, out_dim):
    """Validate one ordered list of [W, b] Linear layers and its dimensional chaining, returning the
    parsed [(W, b), ...]. W is row-major [out][in]; each layer's input must equal the previous layer's
    output; the first layer's input must equal `in_dim` and the last layer's output must equal
    `out_dim`. Raises ValueError on any structural or shape mismatch."""
    if not (isinstance(layers, list) and len(layers) >= 1):
        raise ValueError(
            f"bad {HJEPA_FORMAT} file: {name} must be a non-empty list of [W, b] layers"
        )
    parsed = []
    cur_in = in_dim
    for k, layer in enumerate(layers):
        if not (isinstance(layer, list) and len(layer) == 2):
            raise ValueError(f"bad {HJEPA_FORMAT} file: {name}[{k}] must be a [W, b] pair")
        W, b = layer
        if not (isinstance(W, list) and len(W) >= 1 and isinstance(b, list)):
            raise ValueError(f"bad {HJEPA_FORMAT} file: {name}[{k}] W/b malformed")
        out_k = len(W)
        if len(b) != out_k:
            raise ValueError(
                f"bad {HJEPA_FORMAT} file: {name}[{k}] bias length {len(b)} != rows {out_k}"
            )
        if not all(isinstance(row, list) and len(row) == cur_in for row in W):
            raise ValueError(
                f"bad {HJEPA_FORMAT} file: {name}[{k}] every W row must have length {cur_in}"
            )
        parsed.append((W, b))
        cur_in = out_k
    if cur_in != out_dim:
        raise ValueError(
            f"bad {HJEPA_FORMAT} file: {name} final output {cur_in} != expected {out_dim}"
        )
    return parsed


class HJepaServe:
    """A pure-stdlib forward server for a torch-trained 2-level H-JEPA exported as `snn_hjepa/1`.

    load() reads + trust-validates the artifact. encode0()/encode1() are the L0 fine and L1 abstract
    encoders; predict0()/predict1() are the fine (residual) and coarse (pure-MLP) action-conditioned
    predictors; predict() aliases predict1() so the server duck-types into PrefrontalExecutive.plan()
    over the abstract level. K / latent0 / latent1 / fine_actions / coarse_actions expose the config.
    Construct via HJepaServe.load(path); the __init__ trusts already-validated components."""

    def __init__(
        self,
        *,
        in_dim,
        K,
        latent0,
        latent1,
        fine_actions,
        coarse_actions,
        tau,
        enc0,
        enc1,
        pred0,
        pred1,
    ):
        self._in_dim = int(in_dim)
        self._K = int(K)
        self._latent0 = int(latent0)
        self._latent1 = int(latent1)
        self._fine_actions = tuple(fine_actions)
        self._coarse_actions = tuple(coarse_actions)
        self._n_fine = len(self._fine_actions)
        self._n_coarse = len(self._coarse_actions)
        self._fidx = {a: i for i, a in enumerate(self._fine_actions)}
        self._cidx = {a: i for i, a in enumerate(self._coarse_actions)}
        self.tau = float(tau)
        self._enc0 = enc0  # online encoders / predictors (what inference uses)
        self._enc1 = enc1
        self._pred0 = pred0
        self._pred1 = pred1

    # ---- construction / trust boundary -------------------------------------------------------------
    @classmethod
    def load(cls, path):
        """Load + validate an `snn_hjepa/1` JSON artifact, returning a ready HJepaServe. Validates the
        format tag, the geometry (in_dim/K/latent0/latent1 > 0), the action lists, and EVERY Linear
        layer's shape and chaining across enc0/enc1/pred0/pred1 (plus the EMA target encoders when
        present). Raises ValueError on any mismatch -- an unvalidated weight file is never trusted."""
        with open(path, encoding="ascii") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or obj.get("format") != HJEPA_FORMAT:
            raise ValueError(
                f"bad {HJEPA_FORMAT} file: missing/incorrect 'format' tag (expected {HJEPA_FORMAT})"
            )

        in_dim = _pos_int(obj.get("in_dim"), "in_dim")
        K = _pos_int(obj.get("K"), "K")
        latent0 = _pos_int(obj.get("latent0"), "latent0")
        latent1 = _pos_int(obj.get("latent1"), "latent1")

        fine = obj.get("fine_actions")
        coarse = obj.get("coarse_actions")
        if not (
            isinstance(fine, list) and len(fine) >= 1 and all(isinstance(a, str) for a in fine)
        ):
            raise ValueError(
                f"bad {HJEPA_FORMAT} file: fine_actions must be a non-empty list of names"
            )
        if not (
            isinstance(coarse, list)
            and len(coarse) >= 1
            and all(isinstance(a, str) for a in coarse)
        ):
            raise ValueError(
                f"bad {HJEPA_FORMAT} file: coarse_actions must be a non-empty list of names"
            )
        n_fine, n_coarse = len(fine), len(coarse)

        # online encoders + predictors (served); enc*_target are training EMA copies -- validate if present
        enc0 = _validate_stack("enc0", obj.get("enc0"), in_dim, latent0)
        enc1 = _validate_stack("enc1", obj.get("enc1"), latent0, latent1)
        if obj.get("enc0_target") is not None:
            _validate_stack("enc0_target", obj.get("enc0_target"), in_dim, latent0)
        if obj.get("enc1_target") is not None:
            _validate_stack("enc1_target", obj.get("enc1_target"), latent0, latent1)
        # predictor input widths cross-check the action counts against the weight shapes
        pred0 = _validate_stack("pred0", obj.get("pred0"), latent0 + n_fine, latent0)
        pred1 = _validate_stack("pred1", obj.get("pred1"), latent1 + n_coarse, latent1)

        return cls(
            in_dim=in_dim,
            K=K,
            latent0=latent0,
            latent1=latent1,
            fine_actions=fine,
            coarse_actions=coarse,
            tau=float(obj.get("tau", 0.0)),
            enc0=enc0,
            enc1=enc1,
            pred0=pred0,
            pred1=pred1,
        )

    # ---- config (read-only) ------------------------------------------------------------------------
    @property
    def in_dim(self):
        return self._in_dim

    @property
    def K(self):
        return self._K

    @property
    def latent0(self):
        return self._latent0

    @property
    def latent1(self):
        return self._latent1

    @property
    def fine_actions(self):
        return self._fine_actions

    @property
    def coarse_actions(self):
        return self._coarse_actions

    # ---- action handling ---------------------------------------------------------------------------
    def _onehot(self, command, fine):
        """One-hot the fine/coarse action `command` (a name or an integer index) over its action list."""
        table = self._fidx if fine else self._cidx
        n = self._n_fine if fine else self._n_coarse
        if isinstance(command, str):
            if command not in table:
                raise ValueError(f"unknown {'fine' if fine else 'coarse'} action name: {command!r}")
            idx = table[command]
        else:
            idx = int(command)
            if not 0 <= idx < n:
                raise ValueError(f"action index {idx} out of range [0, {n})")
        oh = [0.0] * n
        oh[idx] = 1.0
        return oh

    # ---- encoders ----------------------------------------------------------------------------------
    def encode0(self, obs):
        """L0 fine encoder: unit(mlp(enc0, obs)). `obs` is a length-in_dim observation; returns a
        length-latent0 unit latent on the sphere."""
        if len(obs) != self._in_dim:
            raise ValueError(f"encode0: obs length {len(obs)} != in_dim {self._in_dim}")
        return unit(_mlp(self._enc0, list(obs)))

    def _as_latent(self, item):
        """Coerce one window item to an L0 latent: encode an observation (len in_dim), or pass through
        an already-computed L0 latent (len latent0) unchanged -- mirrors encode1_from_latents."""
        n = len(item)
        if n == self._in_dim:
            return self.encode0(item)
        if n == self._latent0:
            return list(item)
        raise ValueError(
            f"encode1 item length {n} is neither in_dim {self._in_dim} nor latent0 {self._latent0}"
        )

    def encode1(self, window):
        """L1 abstract encoder: unit(mlp(enc1, meanpool([encode0(o) for o in window]))). `window` is a
        list of observations (each len in_dim) OR of already-computed L0 latents (each len latent0);
        both are supported. meanpool is the elementwise mean of the per-tick L0 latents. Returns a
        length-latent1 unit latent."""
        if not window:
            raise ValueError("encode1: window must be a non-empty list")
        z0s = [self._as_latent(item) for item in window]
        pooled = [mean([z[j] for z in z0s]) for j in range(self._latent0)]
        return unit(_mlp(self._enc1, pooled))

    # ---- predictors --------------------------------------------------------------------------------
    def predict0(self, z0, command):
        """Fine RESIDUAL predictor: unit(z0 + mlp(pred0, z0 ++ onehot_fine(command))). `z0` is a
        length-latent0 latent; `command` is a fine action name or index. Returns the predicted
        next L0 latent on the sphere."""
        if len(z0) != self._latent0:
            raise ValueError(f"predict0: z0 length {len(z0)} != latent0 {self._latent0}")
        d = _mlp(self._pred0, list(z0) + self._onehot(command, fine=True))
        return unit([z0[i] + d[i] for i in range(self._latent0)])

    def predict1(self, z1, command):
        """Coarse PURE-MLP predictor: unit(mlp(pred1, z1 ++ onehot_coarse(command))). `z1` is a
        length-latent1 abstract latent; `command` is a coarse action name or index. Returns the
        predicted next L1 latent on the sphere (non-contractive, bounded by the unit projection)."""
        if len(z1) != self._latent1:
            raise ValueError(f"predict1: z1 length {len(z1)} != latent1 {self._latent1}")
        return unit(_mlp(self._pred1, list(z1) + self._onehot(command, fine=False)))

    def predict(self, state, command):
        """Alias of predict1 -- the duck-typed plan() interface, so an HJepaServe drops into
        PrefrontalExecutive.plan() as its predictor, planning over the ABSTRACT (coarse) level."""
        return self.predict1(state, command)


# =========================================================================================================
# Self-check: build a TINY snn_hjepa/1 by hand, serve it, and assert the forward matches an INDEPENDENT
# hand-rolled reference (raw arithmetic, not the served code path) for encode0/encode1/predict0/predict1.
# Also assert: load rejects bad-format and wrong-shape files; encode1 accepts obs- and latent-windows;
# predict(state, cmd) == predict1(state, cmd). Deterministic, ASCII-only.
# =========================================================================================================
def _build_tiny_blob():
    """A hand-set snn_hjepa/1 dict: in_dim 3, latent0 2, latent1 2, K 2; enc0 is a 2-layer MLP (exercises
    the ReLU BETWEEN layers), enc1/pred0/pred1 are single raw layers. EMA targets copy the online encoders."""
    enc0 = [
        [[[0.5, -0.5, 1.0], [1.0, 0.0, -1.0]], [0.1, -0.2]],  # layer 0: [2][3] -> relu
        [[[1.0, 0.5], [-0.5, 1.0]], [0.0, 0.05]],  # layer 1: [2][2] -> raw (last)
    ]
    enc1 = [[[[1.0, -1.0], [0.5, 0.5]], [0.0, 0.0]]]  # [2][2] latent0 -> latent1
    pred0 = [
        [[[0.2, -0.1, 0.3, 0.4], [0.1, 0.2, -0.2, 0.1]], [0.0, 0.0]]
    ]  # [2][4] (latent0 + 2 fine)
    pred1 = [
        [[[0.1, -0.2, 0.3, 0.0, 0.1], [0.2, 0.1, -0.1, 0.2, -0.3]], [0.0, 0.0]]
    ]  # [2][5] (latent1 + 3 coarse)
    return {
        "format": HJEPA_FORMAT,
        "in_dim": 3,
        "K": 2,
        "latent0": 2,
        "latent1": 2,
        "fine_actions": ["FWD", "BCK"],
        "coarse_actions": ["L", "R", "S"],
        "enc0": enc0,
        "enc1": enc1,
        "enc0_target": enc0,
        "enc1_target": enc1,  # EMA copies (validated, not served)
        "pred0": pred0,
        "pred1": pred1,
        "tau": 0.99,
        "predictors": {"pred0": "residual_mlp_unit", "pred1": "mlp_unit"},
        "meta": {"trained_by": "self-check", "unverified_on_hardware": True},
    }


def main():
    import os
    import tempfile

    # ---- independent reference forward (raw arithmetic; NOT the served code path) ----
    def r_relu(v):
        return [x if x > 0.0 else 0.0 for x in v]

    def r_dot(w, x):
        return sum(wi * xi for wi, xi in zip(w, x))

    def r_unit(v):
        nrm = math.sqrt(sum(c * c for c in v)) or 1.0
        return [c / nrm for c in v]

    def r_mlp(layers, x):
        h = list(x)
        last = len(layers) - 1
        for i, (W, b) in enumerate(layers):
            h = [r_dot(W[o], h) + b[o] for o in range(len(b))]
            if i < last:
                h = r_relu(h)
        return h

    blob = _build_tiny_blob()
    enc0, enc1 = blob["enc0"], blob["enc1"]
    pred0, pred1 = blob["pred0"], blob["pred1"]
    fine, coarse = blob["fine_actions"], blob["coarse_actions"]
    latent0 = blob["latent0"]

    def r_onehot(cmd, names):
        idx = names.index(cmd) if isinstance(cmd, str) else int(cmd)
        oh = [0.0] * len(names)
        oh[idx] = 1.0
        return oh

    def r_encode0(obs):
        return r_unit(r_mlp(enc0, obs))

    def r_encode1(window):
        z0s = [r_encode0(o) for o in window]
        pooled = [sum(z[j] for z in z0s) / len(z0s) for j in range(latent0)]
        return r_unit(r_mlp(enc1, pooled))

    def r_predict0(z0, cmd):
        d = r_mlp(pred0, list(z0) + r_onehot(cmd, fine))
        return r_unit([z0[i] + d[i] for i in range(latent0)])

    def r_predict1(z1, cmd):
        return r_unit(r_mlp(pred1, list(z1) + r_onehot(cmd, coarse)))

    def close(a, b, tol=1e-9):
        return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))

    # ---- write + load the artifact through the trust boundary ----
    tmp = os.path.join(tempfile.gettempdir(), "snn_hjepa1_selfcheck.json")
    with open(tmp, "w", encoding="ascii") as f:
        json.dump(blob, f)
    srv = HJepaServe.load(tmp)
    os.remove(tmp)

    # ---- (1) forward matches the independent reference for a couple of inputs ----
    obs_a = [1.0, 2.0, 0.5]
    obs_b = [-0.5, 0.25, 1.5]
    window = [obs_a, obs_b]
    z0a, z0b = srv.encode0(obs_a), srv.encode0(obs_b)
    z1 = srv.encode1(window)
    enc0_ok = close(z0a, r_encode0(obs_a)) and close(z0b, r_encode0(obs_b))
    enc1_ok = close(z1, r_encode1(window))
    p0_ok = close(srv.predict0(z0a, "BCK"), r_predict0(z0a, "BCK")) and close(
        srv.predict0(z0a, 0), r_predict0(z0a, 0)
    )  # name AND index forms
    p1_ok = close(srv.predict1(z1, "R"), r_predict1(z1, "R")) and close(
        srv.predict1(z1, 2), r_predict1(z1, 2)
    )
    unit_ok = abs(math.sqrt(sum(c * c for c in z1)) - 1.0) < 1e-9  # served latent is on the sphere

    # ---- (2) encode1 accepts BOTH a window-of-obs and a window-of-latents (same result) ----
    lat_window = [srv.encode0(obs_a), srv.encode0(obs_b)]
    both_windows_ok = close(srv.encode1(lat_window), srv.encode1(window))

    # ---- (3) predict(state, cmd) is exactly predict1(state, cmd) ----
    alias_ok = srv.predict(z1, "R") == srv.predict1(z1, "R")

    # ---- (4) config exposure ----
    config_ok = (
        srv.K == 2
        and srv.latent0 == 2
        and srv.latent1 == 2
        and srv.in_dim == 3
        and srv.fine_actions == ("FWD", "BCK")
        and srv.coarse_actions == ("L", "R", "S")
    )

    # ---- (5) the loader rejects bad-format and wrong-shape files ----
    def _expect_valueerror(mutate):
        bad = json.loads(json.dumps(blob))  # deep copy via round-trip
        mutate(bad)
        p = os.path.join(tempfile.gettempdir(), "snn_hjepa1_bad.json")
        with open(p, "w", encoding="ascii") as fh:
            json.dump(bad, fh)
        try:
            HJepaServe.load(p)
            return False
        except ValueError:
            return True
        finally:
            os.remove(p)

    def _bad_format(b):
        b["format"] = "snn_hjepa/2"

    def _bad_shape(b):
        b["enc0"][-1][0].append([0.0, 0.0])  # W2 -> 3 rows while bias stays len 2

    reject_format_ok = _expect_valueerror(_bad_format)
    reject_shape_ok = _expect_valueerror(_bad_shape)

    print("=" * 78)
    print("H-JEPA SERVE -- zero-dep forward server for a torch-trained 2-level snn_hjepa/1")
    print("=" * 78)
    print(
        f"config      : in_dim={srv.in_dim} K={srv.K} latent0={srv.latent0} latent1={srv.latent1} "
        f"fine={list(srv.fine_actions)} coarse={list(srv.coarse_actions)}"
    )
    print(f"encode0     : matches independent reference (raw mlp+relu+unit): {enc0_ok}")
    print(
        f"encode1     : matches reference (meanpool L0 -> enc1 -> unit): {enc1_ok}; on unit sphere: {unit_ok}"
    )
    print(f"predict0    : residual fine, matches reference (name+index): {p0_ok}")
    print(f"predict1    : pure-MLP coarse, matches reference (name+index): {p1_ok}")
    print(f"encode1     : obs-window == latent-window: {both_windows_ok}")
    print(f"predict     : predict(state,cmd) == predict1(state,cmd): {alias_ok}")
    print(
        f"trust       : load rejects bad-format={reject_format_ok}, wrong-shape={reject_shape_ok}"
    )
    print("-" * 78)

    assert enc0_ok, "encode0 must match the independent hand-rolled reference"
    assert enc1_ok, (
        "encode1 must match the independent hand-rolled reference (meanpool + enc1 head)"
    )
    assert unit_ok, "encode1 output must lie on the unit sphere"
    assert p0_ok, "predict0 (residual fine) must match the reference for name and index commands"
    assert p1_ok, "predict1 (pure-MLP coarse) must match the reference for name and index commands"
    assert both_windows_ok, (
        "encode1 must accept a window of obs OR a window of L0 latents identically"
    )
    assert alias_ok, "predict(state, cmd) must be exactly predict1(state, cmd)"
    assert config_ok, (
        "K / latent0 / latent1 / in_dim / fine_actions / coarse_actions must be exposed"
    )
    assert reject_format_ok, "load must raise ValueError on a bad format tag"
    assert reject_shape_ok, "load must raise ValueError on a wrong-shape layer"

    print(
        "self-check OK: snn_hjepa/1 forward served pure-stdlib and matches the torch _HJepa forward "
        "(encode0/encode1/predict0/predict1); dual-window encode1, predict==predict1, and a validating loader"
    )


if __name__ == "__main__":
    main()
